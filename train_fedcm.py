"""
FedCM-RL Training Script

Federated Cross-Model Reinforcement Learning for heterogeneous
traffic signal control agents.

Based on:
- FedMD (Li & Wang 2019) - Model-heterogeneous distillation
- FedDF (Lin et al., NeurIPS 2020) - Ensemble teacher aggregation
- HeteroFL (ICLR 2020) - Heterogeneous architectures
"""

import argparse
import os
import json
import numpy as np
from typing import List, Dict

from federated_learning.fedcm_client import FedCMClient
from federated_learning.fedcm_server import FedCMServer


def run_fedcm_simulation(num_rounds: int = 15, results_dir: str = "results_fedcm",
                         gui: bool = False, num_clients: int = 3, proxy_size: int = 1000,
                         weighting_method: str = "performance"):
    """
    Run FedCM-RL simulation.
    
    6-Phase Training Loop:
    1. Local RL training
    2. Proxy dataset construction
    3. Client logit collection
    4. Ensemble teacher aggregation
    5. Cross-model distillation
    6. Evaluation
    
    Args:
        num_rounds: Number of federated rounds
        results_dir: Directory to save results
        gui: Whether to use SUMO GUI
        proxy_size: Proxy dataset size
        weighting_method: "uniform" or "performance"
    """
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("STARTING FedCM-RL SIMULATION")
    print("Federated Cross-Model Reinforcement Learning")
    print("=" * 70)
    
    # Client configurations (heterogeneous architectures)
    base_configs = [
        "sumo_configs2/osm.netccfg",
        "sumo_configs2/osm.polycfg"
    ]
    
    # Architecture templates: [Large, Medium, Small]
    arch_templates = [
        {"hidden": [256, 256, 128], "temp": 3.0, "lam": 0.3, "lr": 1e-3},
        {"hidden": [128, 64], "temp": 2.5, "lam": 0.5, "lr": 2e-3},
        {"hidden": [64, 32], "temp": 4.0, "lam": 0.7, "lr": 5e-3}
    ]
    
    client_configs = []
    for i in range(num_clients):
        cfg = base_configs[i % len(base_configs)]
        arch = arch_templates[i % len(arch_templates)]
        client_configs.append({
            "id": f"client_{i+1}",
            "config": cfg,
            "agent_type": "DQN",
            "hidden_dims": arch["hidden"],
            "temperature": arch["temp"],
            "lambda_distill": arch["lam"],
            "learning_rate": arch["lr"]
        })
    
    # Initialize server
    print(f"\nInitializing FedCM Server...")
    print(f"  Weighting method: {weighting_method}")
    print(f"  Proxy dataset size: {proxy_size}")
    print(f"  Optimizations: Model-specific params, adaptive LR")
    
    server = FedCMServer(
        state_dim=12,
        action_dim=4,
        proxy_dataset_size=proxy_size,
        weighting_method=weighting_method
    )
    
    # Initialize clients
    print("\nInitializing FedCM Clients (Heterogeneous Architectures)...")
    clients = []
    
    for config in client_configs:
        print(f"  - {config['id']}: {config['agent_type']} {config['hidden_dims']}")
        client = FedCMClient(
            client_id=config["id"],
            sumo_config_path=config["config"],
            agent_type=config["agent_type"],
            hidden_dims=config["hidden_dims"],
            gui=gui,
            results_dir=results_dir,
            temperature=config["temperature"],
            lambda_distill=config["lambda_distill"],
            distill_method="mse"
        )
        
        # Set model-specific learning rate (HeteroFL optimization)
        if "learning_rate" in config:
            client.agent.learning_rate = config["learning_rate"]
            for param_group in client.agent.optimizer.param_groups:
                param_group['lr'] = config["learning_rate"]
        clients.append(client)
    
    # Training loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'=' * 70}")
        
        # ============================================
        # Phase 1: Local RL Training
        # ============================================
        print(f"\n[Round {round_num}] Phase 1: Local RL Training...")
        
        training_metrics = []
        client_states = []
        
        for client in clients:
            print(f"  Training {client.client_id}...")
            metrics = client.train(round_num)
            training_metrics.append(metrics)
            
            # Collect states from replay buffer for proxy dataset
            if hasattr(client.agent, 'memory') and len(client.agent.memory) > 0:
                # Sample states from replay buffer
                buffer_states = [exp[0] for exp in list(client.agent.memory)[:500]]
                client_states.append(np.array(buffer_states))
            
            print(f"    Reward: {metrics['average_reward']:.2f}, "
                  f"Loss: {metrics.get('dqn_loss', 0):.4f}")
        
        # ============================================
        # Phase 2: Proxy Dataset Construction
        # ============================================
        print(f"\n[Round {round_num}] Phase 2: Constructing proxy dataset...")
        
        if len(client_states) > 0:
            proxy_states = server.construct_proxy_dataset(client_states, method="sample")
        else:
            # Fallback: generate random states
            print("  Warning: No client states available, using random proxy states")
            proxy_states = np.random.rand(proxy_size, 12)
            server.proxy_states = proxy_states
        
        print(f"  Proxy dataset shape: {proxy_states.shape}")
        
        # ============================================
        # Phase 3: Client Logit Collection
        # ============================================
        print(f"\n[Round {round_num}] Phase 3: Collecting client logits...")
        
        all_logits = []
        client_ids = []
        
        for client in clients:
            logits = client.get_logits(proxy_states)
            all_logits.append(logits)
            client_ids.append(client.client_id)
            print(f"  {client.client_id}: logits shape {logits.shape}")
        
        # ============================================
        # Phase 4: Ensemble Teacher Aggregation
        # ============================================
        print(f"\n[Round {round_num}] Phase 4: Aggregating ensemble teacher...")
        
        # Update server with client performance
        for i, client in enumerate(clients):
            waiting_time = training_metrics[i].get('waiting_time', 1000.0)
            server.update_client_performance(client.client_id, waiting_time)
        
        # Compute ensemble teacher
        teacher_logits = server.compute_ensemble_teacher(all_logits, client_ids)
        
        print(f"  Teacher logits shape: {teacher_logits.shape}")
        print(f"  Weighting: {weighting_method}")
        
        # ============================================
        # Phase 5: Cross-Model Distillation
        # ============================================
        print(f"\n[Round {round_num}] Phase 5: Cross-model distillation...")
        
        distill_results = []
        for i, client in enumerate(clients):
            # Model-specific distillation epochs (FedMD optimization)
            # Large models: 3 epochs, Medium: 3 epochs, Small: 2 epochs
            epochs_map = {0: 3, 1: 3, 2: 2}
            distill_epochs = epochs_map.get(i, 3)
            
            distill_metrics = client.distill(proxy_states, teacher_logits, epochs=distill_epochs)
            distill_results.append(distill_metrics)
            print(f"  {client.client_id}: Distill Loss = {distill_metrics['distill_loss']:.6f} ({distill_epochs} epochs)")
        
        # ============================================
        # Phase 6: Evaluation
        # ============================================
        print(f"\n[Round {round_num}] Phase 6: Evaluation...")
        
        eval_results = []
        
        for i, client in enumerate(clients):
            # Traffic performance evaluation
            eval_metrics = client.evaluate(round_num)
            eval_results.append(eval_metrics)
            
            # Save combined metrics
            save_path = os.path.join(results_dir, f"{client.client_id}_round_{round_num}_eval.json")
            combined_metrics = {
                **eval_metrics,
                'training': training_metrics[i],
                'distillation': distill_results[i],
                'architecture': client.get_architecture_info()
            }
            
            # Convert numpy types
            combined_metrics = convert_to_json_serializable(combined_metrics)
            
            with open(save_path, "w") as f:
                json.dump(combined_metrics, f, indent=2)
            
            print(f"  {client.client_id}: Waiting Time = {eval_metrics.get('waiting_time', 0):.2f}s")
        
        # ============================================
        # 7. Standardized Performance Table
        mode_str = "SUMO" if gui else "Mock"
        print(f"\nRound {round_num} Client Performance Summary:")
        print(f"{'-'*90}")
        print(f"{'Client ID':<12} | {'Context/Arch':<16} | {'Reward':<12} | {'Avg Wait (s)':<14} | {'Throughput':<10} | {'Mode':<6}")
        print(f"{'-'*90}")
        for i, client in enumerate(clients):
            e = eval_results[i]
            t = training_metrics[i]
            # Use data from configs list
            arch_str = f"{client_configs[i]['hidden_dims']}"
            tp = e.get('total_vehicles', 0)
            
            print(f"{client.client_id:<12} | {arch_str:<16} | {t['average_reward']:>12.1f} | {e.get('waiting_time', 0):>14.2f} | {tp:>10} | {mode_str:<6}")
        print(f"{'-'*90}")
        
        # Communication cost analysis
        comm_cost = server.get_communication_cost(len(clients))
        print(f"  Communication Cost:")
        print(f"    FedCM: {comm_cost['fedcm_kb']:.2f} KB")
        print(f"    FedAvg: {comm_cost['fedavg_kb']:.2f} KB")
        print(f"    Reduction: {comm_cost['reduction_percent']:.1f}%")
        
        # Policy divergence
        if round_num % 5 == 0:  # Compute every 5 rounds
            divergence = server.get_policy_divergence(all_logits)
            print(f"  Policy Divergence (KL): {divergence['mean_kl']:.4f}")
    
    # ============================================
    # Final Summary
    # ============================================
    print("\n" + "=" * 70)
    print("FedCM-RL SIMULATION COMPLETED")
    print(f"Results saved to {results_dir}")
    print("=" * 70)


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedCM-RL Training")
    parser.add_argument("--rounds", type=int, default=15, help="Number of federated rounds")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of simulated clients")
    parser.add_argument("--results-dir", type=str, default="results_fedcm", help="Results directory")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI (Enforces real SUMO simulation)")
    parser.add_argument("--proxy-size", type=int, default=2000, help="Proxy dataset size")
    parser.add_argument("--weighting", type=str, default="performance", 
                       choices=["uniform", "performance"], help="Aggregation weighting method")
    
    args = parser.parse_args()
    
    run_fedcm_simulation(
        num_rounds=args.rounds,
        results_dir=args.results_dir,
        gui=args.gui,
        num_clients=args.num_clients,
        proxy_size=args.proxy_size,
        weighting_method=args.weighting
    )
