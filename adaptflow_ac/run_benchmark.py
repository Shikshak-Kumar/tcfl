import os
import subprocess
import json
import sys
from utils.logger import logger

def run_experiment(rounds=5, nodes=4, steps=100):
    """
    Runs training for all three agents and then compares them.
    Note: Requires torch and other dependencies.
    """
    results_dir = "results/benchmark"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set PYTHONPATH to include current directory for local imports
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.getcwd()
    
    # 1. Train AdaptFlow
    print("\n[1/4] Training AdaptFlow...")
    subprocess.run([sys.executable, "train/train_adaptflow.py", "--rounds", str(rounds), "--nodes", str(nodes), "--steps", str(steps)], env=env_vars)
    
    # 2. Train FedLight
    print("\n[2/4] Training FedLight...")
    subprocess.run([sys.executable, "train/train_fedlight.py", "--rounds", str(rounds), "--nodes", str(nodes), "--steps", str(steps)], env=env_vars)
    
    # 3. Train FedDQNTsc
    print("\n[3/5] Training FedDQNTsc...")
    subprocess.run([sys.executable, "train/train_fed_dqn_tsc.py", "--mode", "mock", "--rounds", str(rounds), "--episodes-per-round", "2", "--steps", str(steps)], env=env_vars)
    
    # 4. Train MultiAgentAC
    print("\n[4/5] Training MultiAgentAC...")
    subprocess.run([sys.executable, "train/train_multi_agent_ac.py", "--mode", "mock", "--batch-size", "10", "--steps", str(steps)], env=env_vars)
    
    # 5. Run Comparison
    print("\n[5/5] Running Comparison...")
    subprocess.run([sys.executable, "compare.py", "--mode", "mock", "--episodes", "5"], env=env_vars)
    
    # Load and display results
    report_path = "results/comparison_report.json"
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            results = json.load(f)
        
        data = []
        for name, res in results.items():
            m = res['metrics']
            data.append({
                "Agent": name,
                "Avg Reward": sum(res['rewards']) / len(res['rewards']),
                "Avg Queue": m['avg_queue'],
                "Avg Wait (s)": m['avg_waiting'],
                "Throughput": m['throughput']
            })
        
        # Calculate % improvement over Random baseline
        random_wait = [d["Avg Wait (s)"] for d in data if d["Agent"] == "Random"]
        random_wait = random_wait[0] if random_wait else 1.0
        
        print("\n" + "="*90)
        print("MOCK DATA BENCHMARK RESULTS")
        print("="*90)
        print(f"{'Agent':<20} | {'Reward':<10} | {'Queue':<10} | {'Wait (s)':<10} | {'% Improv':<10}")
        print("-" * 90)
        for d in data:
            improv = ((random_wait - d["Avg Wait (s)"]) / random_wait * 100) if random_wait else 0
            print(f"{d['Agent']:<20} | {d['Avg Reward']:<10.2f} | {d['Avg Queue']:<10.2f} | {d['Avg Wait (s)']:<10.2f} | {improv:<10.1f}%")
        print("="*90)
        
        return data
    else:
        print("\nError: Comparison report not found. Ensure training scripts completed successfully.")
        return None

if __name__ == "__main__":
    run_experiment(rounds=2, nodes=4, steps=100)
