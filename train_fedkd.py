import os
import sys
import json
import argparse
import numpy as np
import torch
from federated_learning.fedkd_server import TrafficFedKDServer
from federated_learning.fedkd_client import TrafficFedKDClient

def run_fedkd_simulation(num_rounds=15, results_dir="results_fedkd", gui=False):
    print("STARTING FEDKD SIMULATION")
    
    os.makedirs(results_dir, exist_ok=True)
    
    server = TrafficFedKDServer(num_rounds=num_rounds, min_clients=2)
    server.initialize_proxy_dataset(state_size=12)
    
    # Heterogeneous architecture configuration
    client_configs = [
        {
            "id": "client_1", 
            "config": "sumo_configs/osm_client1.sumocfg", 
            "hidden_dims": [256, 256, 128]
        },
        {
            "id": "client_2", 
            "config": "sumo_configs/osm_client2.sumocfg", 
            "hidden_dims": [64, 32]
        }
    ]
    
    clients = []
    for cfg in client_configs:
        print(f"Initializing {cfg['id']} with architecture: {cfg['hidden_dims']}")
        clients.append(
            TrafficFedKDClient(
                client_id=cfg["id"],
                sumo_config_path=cfg["config"],
                hidden_dims=cfg["hidden_dims"],
                gui=gui
            )
        )
        
    for round_num in range(num_rounds):
        # Local training and state collection
        print(f"Round {round_num + 1}: Local Training...")
        train_metrics = []
        observed_states_batch = []
        for client in clients:
            # Each client does standard RL training and automatically stores observed states
            metrics = client._train_agent(episodes=3)
            train_metrics.append(metrics)
            
            # Collect a sample of real states from this client
            states = client.get_observed_states(limit=200)
            observed_states_batch.append(states)
            
            print(f"  {client.client_id} Reward: {metrics['average_reward']:.4f}, States Collected: {len(states)}")
            
        # Server proxy update
        print("Updating proxy dataset...")
        server.update_proxy_dataset(observed_states_batch)
        
        if server.proxy_states.size == 0:
            print("  Warning: No states collected yet, skipping distillation this round.")
            continue

        # Logit exchange
        print("Synchronizing logits...")
        all_logits = []
        for client in clients:
            logits = client.get_logits(server.proxy_states)
            all_logits.append(logits)
            
        # Server aggregation
        print("Aggregating consensus knowledge...")
        consensus_logits = server.aggregate_logits(all_logits)
        
        # Distillation
        print("Executing Knowledge Distillation...")
        for client in clients:
            kd_res = client.distill(server.proxy_states, consensus_logits)
            print(f"  {client.client_id} KD Loss: {kd_res['distill_loss']:.6f}")
            
        # Evaluation
        print("Evaluation phase...")
        eval_results = []
        for client in clients:
            eval_metrics = client._evaluate_agent()
            eval_results.append(eval_metrics)
            
            # Save metrics
            save_path = os.path.join(results_dir, f"{client.client_id}_round_{round_num}_eval.json")
            with open(save_path, "w") as f:
                json.dump(eval_metrics, f, indent=2)
                
        avg_wait = np.mean([m['waiting_time'] for m in eval_results])
        print(f"Round Summary: Avg Waiting Time = {avg_wait:.2f}s")

    print("\n" + "="*60)
    print("FedKD SIMULATION COMPLETED")
    print(f"Results saved to {results_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--results-dir", type=str, default="results_fedkd")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()
    
    run_fedkd_simulation(num_rounds=args.rounds, results_dir=args.results_dir, gui=args.gui)
