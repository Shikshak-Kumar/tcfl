
import os
import sys
import argparse
import time
import json
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_server import TrafficFLServer
from federated_learning.fl_client import TrafficFLClient
from utils.visualization import TrafficVisualizer
import flwr as fl

def create_client_script():
    client_script = '''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_client import TrafficFLClient
import flwr as fl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=str, required=True, help="Client ID")
    parser.add_argument("--sumo-config", type=str, required=True, help="Path to SUMO config")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--show-phase-console", action="store_true", help="Print TLS phase/time each step")
    args = parser.parse_args()
    
    client = TrafficFLClient(
        client_id=args.client_id,
        sumo_config_path=args.sumo_config,
        gui=args.gui
    )
    
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )
'''
    
    with open("client.py", "w") as f:
        f.write(client_script)
    
    if sys.platform != "win32":
        try:
            os.chmod("client.py", 0o755)
        except OSError:
            pass
    print("Client script created: client.py")

def run_server(num_rounds=10, min_clients=2, server_address="localhost:8080"):
    print("Starting Federated Learning Server...")
    print(f"Server Address: {server_address}")
    print(f"Rounds: {num_rounds}, Min Clients: {min_clients}")
    
    server = TrafficFLServer(
        num_rounds=num_rounds,
        min_clients=min_clients,
        min_fit_clients=min_clients,
        min_eval_clients=min_clients
    )
    
    server.run_federated_learning(server_address)

def run_single_client_training(
    gui: bool = False,
    show_phase_console: bool = False,
    results_dir: str = "results",
):
    print("Running single client training for testing...")
    
    client = TrafficFLClient(
        client_id="test_client",
        sumo_config_path="sumo_configs/intersection.sumocfg",
        gui=gui,
        show_phase_console=show_phase_console,
        show_gst_gui=gui
    )
    
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        current_params = client.get_parameters({})
        
        config = {
            "round": round_num,
            "episodes": 1,
            "learning_rate": 0.001
        }
        
        updated_params, num_samples, metrics = client.fit(current_params, config)
        
        eval_metrics = client.evaluate(updated_params, config)
        
        print(f"Training Metrics: {metrics}")
        
        print("\n" + "="*60)
        print("DETAILED EVALUATION METRICS")
        print("="*60)
        
        print(f"Overall Performance:")
        print(f"  Average Reward: {eval_metrics[2].get('average_reward', 0):.4f}")
        print(f"  Total Waiting Time: {eval_metrics[2].get('waiting_time', 0):.2f}s")
        print(f"  Average Queue Length: {eval_metrics[2].get('queue_length', 0):.2f}")
        print(f"  Max Queue Length: {eval_metrics[2].get('max_queue_length', 0)}")
        print(f"  Avg Waiting Time per Vehicle (episode): {eval_metrics[2].get('avg_waiting_time_per_vehicle', 0):.2f}s")
        
        gst = eval_metrics[2].get('green_signal_time', {})
        print(f"\nGreen Signal Time (GST):")
        try:
            avg_gst = gst.get('avg_gst', 0.0)
            num_lanes = gst.get('num_lanes', 0)
            print(f"  Average GST: {avg_gst:.2f}s over {num_lanes} lanes")
            per_edge = gst.get('per_edge', {})
            if per_edge:
                for edge_id, val in per_edge.items():
                    print(f"    Lane {edge_id}: {float(val):.2f}s")
            else:
                print("    Per-lane GST: Not available")
        except Exception:
            print("  GST: Not available")
        
        per_lane_metrics = eval_metrics[2].get('per_lane_metrics', {})
        lane_summary = eval_metrics[2].get('lane_summary', {})
        
        if per_lane_metrics:
            print(f"\n" + "="*80)
            print("TARGET INTERSECTION - EACH ROAD METRICS")
            print("="*80)
            
            road_count = 1
            for edge_id, lane_data in per_lane_metrics.items():
                print(f"\nROAD #{road_count}: {edge_id}")
                print("-" * 50)
                
                vehicles = lane_data.get('vehicle_count', 0)
                queue_length = lane_data.get('queue_length', 0)
                waiting_time = lane_data.get('waiting_time', 0)
                green_signal_time = lane_data.get('green_signal_time', 0)
                avg_speed = lane_data.get('average_speed', 0)
                
                print(f"Vehicles on Road: {vehicles}")
                print(f"Queue Length: {queue_length} vehicles")
                print(f"Waiting Time: {waiting_time:.2f} seconds")
                print(f"Green Signal Time: {green_signal_time:.2f} seconds")
                print(f"Average Speed: {avg_speed:.2f} m/s")
                
                print(f"Traffic Light Status: Will be shown in real-time console")
                
                occupancy = lane_data.get('occupancy_percent', 0)
                lane_length = lane_data.get('lane_length', 0)
                print(f"Road Occupancy: {occupancy:.1f}%")
                print(f"Road Length: {lane_length:.1f} meters")
                
                vehicle_types = lane_data.get('vehicle_types', {})
                if vehicle_types:
                    type_str = ", ".join([f"{vtype}: {count}" for vtype, count in vehicle_types.items()])
                    print(f"Vehicle Types: {type_str}")
                else:
                    print(f"Vehicle Types: No vehicles")
                
                road_count += 1
        
        if lane_summary:
            print(f"Lane Summary Statistics:")
            print(f"  Total Vehicles: {lane_summary.get('total_vehicles', 0)}")
            print(f"  Total Queue Length: {lane_summary.get('total_queue_length', 0)}")
            print(f"  Total Waiting Time: {lane_summary.get('total_waiting_time', 0):.2f}s")
            print(f"  Average GST: {lane_summary.get('average_gst', 0):.2f}s")
            print(f"  Min GST: {lane_summary.get('min_gst', 0):.2f}s")
            print(f"  Max GST: {lane_summary.get('max_gst', 0):.2f}s")
            print(f"  Max Queue Length: {lane_summary.get('max_queue_length', 0)}")
            print(f"  Max Waiting Time: {lane_summary.get('max_waiting_time', 0):.2f}s")
            print(f"  Average Speed: {lane_summary.get('average_speed', 0):.2f} m/s")
            print(f"  Active Lanes: {lane_summary.get('num_active_lanes', 0)}")
            print(f"  Congested Lanes: {lane_summary.get('num_congested_lanes', 0)}")
            print(f"  Total Occupancy: {lane_summary.get('total_occupancy', 0):.1f}%")
        
        print("="*60)
        try:
            round_results = {
                "round": int(round_num),
                "training_metrics": metrics,
                "evaluation_metrics": eval_metrics[2],
            }
            os.makedirs(results_dir, exist_ok=True)
            round_path = os.path.join(
                results_dir, f"single_client_round_{round_num+1}.json"
            )
            with open(round_path, "w") as f:
                json.dump(round_results, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to save round results: {e}")
        
        client.set_parameters(updated_params)
    
    client.save_training_history(os.path.join(results_dir, "single_client_training.json"))
    client.save_performance_metrics(
        os.path.join(results_dir, "single_client_performance.json")
    )
    
    print("\nSingle client training completed!")
    print("Results saved to results/ directory")

def weighted_avg_params(params_list, weights):
    """Compute weighted average of a list of parameter lists."""
    num_layers = len(params_list[0])
    agg = []
    for layer_idx in range(num_layers):
        layer_sum = None
        for params, w in zip(params_list, weights):
            if layer_sum is None:
                layer_sum = params[layer_idx] * w
            else:
                layer_sum += params[layer_idx] * w
        agg.append(layer_sum)
    return agg

def run_multi_client_simulation(results_dir: str = "results_federated", gui: bool = False, num_rounds: int = 15):
    print("STARTING STANDARD FEDERATED LEARNING SIMULATION")

    import os
    os.makedirs(results_dir, exist_ok=True)

    # Heterogeneous map data configurations
    client_configs = [
        {"id": "client_1", "config": "sumo_configs2/osm.netccfg", "gui": gui},
        {"id": "client_2", "config": "sumo_configs2/osm.polycfg", "gui": gui}
    ]

    clients = []
    for cfg in client_configs:
        print(f"Initializing {cfg['id']}...")
        clients.append(
            TrafficFLClient(
                client_id=cfg["id"],
                sumo_config_path=cfg["config"],
                gui=cfg["gui"]
            )
        )

    print(f"\nSimulation configured for {num_rounds} collaborative rounds.")
    global_params = None

    for round_num in range(num_rounds):
        # Local training phase
        print(f"Round {round_num + 1}: Local Training...")

        # TRAINING
        train_metrics_list = []
        params_list = []

        for client in clients:
            current_params = global_params if global_params is not None else client.get_parameters({})

            config = {
                "round": round_num,
                "episodes": 3,
                "learning_rate": 0.001
            }

            updated_params, _, train_metrics = client.fit(current_params, config)
            train_metrics_list.append(train_metrics)
            params_list.append(updated_params)

            print(f"  {client.client_id} training complete.")

        # Performance evaluation and aggregation
        print("Evaluating performance and calculating consensus...")
        eval_metrics_list = []
        congestion_scores = []

        for client, params in zip(clients, params_list):
            _, _, eval_metrics = client.evaluate(params, {"round": round_num})
            eval_metrics_list.append(eval_metrics)

            client.save_performance_metrics(
                os.path.join(
                    results_dir, f"{client.client_id}_round_{round_num}_eval.json"
                )
            )

            lane_summary = eval_metrics.get("lane_summary", {})
            lane_wait = float(lane_summary.get("total_waiting_time", 0.0))
            num_congested = float(lane_summary.get("num_congested_lanes", 0.0))
            avg_queue = float(lane_summary.get("total_queue_length", 0.0))

            total_lanes = float(lane_summary.get("total_lanes", 1.0))
            norm_wait = min(lane_wait / 300.0, 1.0)      # 300s = ref max wait
            norm_queue = min(avg_queue / 50.0, 1.0)      # 50 vehicles = ref max queue
            norm_cong = num_congested / max(total_lanes, 1.0)  # fraction of congested lanes

            congestion = 0.4 * norm_wait + 0.3 * norm_queue + 0.3 * norm_cong
            congestion_scores.append(max(congestion, 0.0))

        # Parameter aggregation
        print("Aggregating model parameters...")
        sum_cong = sum(congestion_scores)
        if sum_cong > 0:
            weights = [c / sum_cong for c in congestion_scores]
        else:
            n = len(congestion_scores)
            weights = [1.0 / n] * n

        global_params = weighted_avg_params(params_list, weights)

        for client in clients:
            client.set_parameters(global_params)

        # ROUND SUMMARY
        avg_reward = np.mean([m.get("average_reward", 0) for m in train_metrics_list])
        avg_waiting = np.mean([m.get("waiting_time", 0) for m in eval_metrics_list])
        print(f"\nRound {round_num + 1} Summary:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Waiting Time: {avg_waiting:.2f}")
        print(f"  Congestion scores: {[round(c, 2) for c in congestion_scores]}")
        print(f"  Aggregation weights: {[round(w, 3) for w in weights]}")

        # SAVE METRICS
        for client in clients:
            if not client.use_mock:
                detailed = client.env.get_performance_metrics()
                detailed_path = os.path.join(
                    results_dir, f"{client.client_id}_round_{round_num}_detailed.json"
                )
                with open(detailed_path, "w") as f:
                    json.dump(detailed, f, indent=2)

    print("FEDERATED LEARNING SIMULATION COMPLETED")


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Traffic Control")
    parser.add_argument("--mode", choices=["server", "client", "single", "multi"], 
                       default="single", help="Run mode")
    parser.add_argument("--client-id", type=str, help="Client ID (for client mode)")
    parser.add_argument("--sumo-config", type=str, default="sumo_configs/intersection.sumocfg",
                       help="Path to SUMO config")
    parser.add_argument("--server-address", type=str, default="localhost:8080",
                       help="Server address")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store training/eval results",
    )
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--show-phase-console", action="store_true", help="Print TLS phase/time each step")
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.mode == "server":
        run_server(args.num_rounds, args.min_clients, args.server_address)
    elif args.mode == "client":
        if not args.client_id:
            print("Error: --client-id is required for client mode")
            return
        
        client = TrafficFLClient(
            client_id=args.client_id,
            sumo_config_path=args.sumo_config,
            gui=args.gui,
            show_phase_console=args.show_phase_console
        )
        
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
    elif args.mode == "single":
        run_single_client_training(
            gui=args.gui,
            show_phase_console=args.show_phase_console,
            results_dir=args.results_dir,
        )
    elif args.mode == "multi":
        run_multi_client_simulation(results_dir=args.results_dir, gui=args.gui, num_rounds=args.num_rounds)
    
    print("Training completed!")

if __name__ == "__main__":
    create_client_script()
    main()
