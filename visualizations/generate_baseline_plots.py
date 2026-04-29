import matplotlib.pyplot as plt
import numpy as np
import json
import os

def generate_plots(results_path="backend/comparison_results.json"):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found. Using dummy data.")
        results = {
          "FL-DQN": { "waiting_time": 45.2, "queue": 12.5, "throughput": 0.65 },
          "FedLight": { "waiting_time": 42.1, "queue": 11.2, "throughput": 0.68 },
          "AdaptFlow": { "waiting_time": 32.5, "queue": 8.4, "throughput": 0.82 }
        }
        history = {}
    else:
        with open(results_path, "r") as f:
            data = json.load(f)
            if "summary" in data:
                results = data["summary"]
                history = data["history"]
            else:
                results = data
                history = {}

    methods = list(results.keys())
    waiting_times = [results[m]["waiting_time"] for m in methods]
    throughputs = [results[m]["throughput"] for m in methods]
    queues = [results[m]["queue"] for m in methods]

    # Set style
    plt.style.use('ggplot')
    colors = ['#ff9999','#66b3ff','#52D017'] # Soft Red, Soft Blue, Emerald Green

    # 1. Bar chart: Waiting Time
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, waiting_times, color=colors, edgecolor='black', alpha=0.8)
    plt.title('Average Waiting Time Comparison (Lower is Better)', fontsize=15, fontweight='bold')
    plt.ylabel('Waiting Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/baseline_waiting_time.png', dpi=300)

    # 2. Bar chart: Throughput
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, throughputs, color=colors, edgecolor='black', alpha=0.8)
    plt.title('Throughput Comparison (Higher is Better)', fontsize=15, fontweight='bold')
    plt.ylabel('Throughput Ratio', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/baseline_throughput.png', dpi=300)

    # 3. Line Chart: Convergence (Reward vs Rounds)
    if history:
        plt.figure(figsize=(10, 6))
        for i, m in enumerate(methods):
            rewards = [h["reward"] for h in history[m]]
            plt.plot(range(1, len(rewards) + 1), rewards, label=m, color=colors[i], marker='o', markersize=4, linewidth=2)
        
        plt.title('Training Convergence (Reward per Episode)', fontsize=15, fontweight='bold')
        plt.xlabel('Federated Round', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('visualizations/baseline_convergence.png', dpi=300)
        print("Saved visualizations/baseline_convergence.png")

if __name__ == "__main__":
    generate_plots()
