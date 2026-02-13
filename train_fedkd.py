import os
import sys
import json
import argparse
import numpy as np
import torch
from federated_learning.fedkd_server import TrafficFedKDServer
from federated_learning.fedkd_client import TrafficFedKDClient

def run_fedkd_simulation(num_rounds=15, results_dir="results_fedkd", gui=False, num_clients=2):
    print("STARTING FEDKD SIMULATION")
    
    os.makedirs(results_dir, exist_ok=True)
    
    server = TrafficFedKDServer(num_rounds=num_rounds, min_clients=2)
    server.initialize_proxy_dataset(state_size=12)
    
    # Heterogeneous architecture configuration
    base_configs = [
        "sumo_configs2/osm.netccfg",
        "sumo_configs2/osm.polycfg"
    ]
    
    # Architecture templates: [Large, Small]
    arch_templates = [
        {"hidden": [256, 256, 128]},
        {"hidden": [64, 32]}
    ]
    
    client_configs = []
    for i in range(num_clients):
        cfg = base_configs[i % len(base_configs)]
        arch = arch_templates[i % len(arch_templates)]
        client_configs.append({
            "id": f"client_{i+1}",
            "config": cfg,
            "hidden_dims": arch["hidden"]
        })
    
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
        
        # Standardized Performance Table
        mode_str = "SUMO" if gui else "Mock"
        print(f"\nRound {round_num + 1} Client Performance Summary:")
        print(f"{'-'*90}")
        print(f"{'Client ID':<12} | {'Context/Arch':<16} | {'Reward':<12} | {'Avg Wait (s)':<14} | {'Throughput':<10} | {'Mode':<6}")
        print(f"{'-'*90}")
        for i, client in enumerate(clients):
            m = eval_results[i]
            t = train_metrics[i]
            # Use data from configs list
            arch_str = f"{client_configs[i]['hidden_dims']}"
            tp = m.get('total_vehicles', 0)
            print(f"{client.client_id:<12} | {arch_str:<16} | {t['average_reward']:>12.1f} | {m['waiting_time']:>14.2f} | {tp:>10} | {mode_str:<6}")
        print(f"{'-'*90}")
        print(f"Round Summary: Avg Waiting Time = {avg_wait:.2f}s")

    print("\n" + "="*60)
    print("FedKD SIMULATION COMPLETED")
    print(f"Results saved to {results_dir}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedKD-RL Training")
    parser.add_argument("--rounds", type=int, default=15, help="Number of federated rounds")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of simulated clients")
    parser.add_argument("--results-dir", type=str, default="results_fedkd", help="Results directory")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI (Enforces real SUMO simulation)")
    args = parser.parse_args()
    
    run_fedkd_simulation(
        num_rounds=args.rounds, 
        results_dir=args.results_dir, 
        gui=args.gui,
        num_clients=args.num_clients
    )
