import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def merge_results(base_dir="results_comparison"):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    all_summaries = []
    
    # 1. Collect all summary.json files
    for root, dirs, files in os.walk(base_dir):
        if "summary.json" in files:
            with open(os.path.join(root, "summary.json"), "r") as f:
                all_summaries.append(json.load(f))
    
    if not all_summaries:
        print("No results found to merge.")
        return

    print(f"Merging results from {len(all_summaries)} experiment runs...")

    # 2. Aggregate metrics
    aggregated = defaultdict(lambda: defaultdict(list))
    methods = ["FL-DQN", "FedLight", "AdaptFlow"]
    metrics = ["waiting_time", "queue", "throughput", "reward"]

    for summary in all_summaries:
        for method in methods:
            if method in summary:
                for metric in metrics:
                    aggregated[method][metric].append(summary[method][metric])

    # 3. Calculate Mean and Std Dev
    final_stats = {}
    for method in methods:
        final_stats[method] = {}
        for metric in metrics:
            data = aggregated[method][metric]
            if data:
                final_stats[method][metric] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data))
                }

    # 4. Generate LaTeX Table Snippet
    print("\n" + "="*50)
    print("LATEX TABLE SNIPPET (Copy into your Overleaf)")
    print("="*50)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("Method & Avg Wait (s) & Avg Queue & Throughput \\\\")
    print("\\hline")
    for method in methods:
        stats = final_stats[method]
        print(f"{method} & {stats['waiting_time']['mean']:.2f} $\\pm$ {stats['waiting_time']['std']:.2f} & "
              f"{stats['queue']['mean']:.2f} & {stats['throughput']['mean']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of AdaptFlow vs Baselines across multiple runs.}")
    print("\\end{table}")

    # 5. Generate Master Plot with Error Bars
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [final_stats[m]['waiting_time']['mean'] for m in methods]
    stds = [final_stats[m]['waiting_time']['std'] for m in methods]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=10, color=['#ff9999','#66b3ff','#52D017'], edgecolor='black', alpha=0.8)
    ax.set_title('Master Comparison: Average Waiting Time (Multi-Run)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Waiting Time (seconds)')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(base_dir, "master_comparison_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[DONE] Master plot saved to: {output_path}")

if __name__ == "__main__":
    merge_results()
