#!/usr/bin/env python3
"""
Visualization script for baseline comparison results.

Creates comprehensive graphs comparing all methods.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_comparison_results(results_file: str) -> Dict:
    """Load comparison results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_plots(summary: Dict, output_dir: str = "comparison_plots"):
    """Create comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    methods = list(summary.keys())
    if not methods:
        print("No methods to plot!")
        return
    
    # Extract metrics
    waiting_times = [summary[m]['mean_waiting_time'] for m in methods]
    waiting_stds = [summary[m]['std_waiting_time'] for m in methods]
    queue_lengths = [summary[m]['mean_queue_length'] for m in methods]
    queue_stds = [summary[m]['std_queue_length'] for m in methods]
    
    # Method names for display
    method_labels = {
        'fixed_time': 'Fixed-Time',
        'maxpressure': 'MaxPressure',
        'centralized_rl': 'Centralized RL',
        'independent_rl': 'Independent RL',
        'federated': 'Federated FL (Ours)'
    }
    
    display_names = [method_labels.get(m, m.replace('_', ' ').title()) for m in methods]
    
    # Color scheme
    colors = {
        'fixed_time': '#e74c3c',
        'maxpressure': '#3498db',
        'centralized_rl': '#1abc9c',
        'independent_rl': '#95a5a6',
        'federated': '#2ecc71'
    }
    
    bar_colors = [colors.get(m, '#34495e') for m in methods]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Baseline Comparison: Traffic Control Methods', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Waiting Time Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    bars1 = ax1.bar(range(len(methods)), waiting_times, yerr=waiting_stds, 
                    color=bar_colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Waiting Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Waiting Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight federated method
    if 'federated' in methods:
        fed_idx = methods.index('federated')
        bars1[fed_idx].set_edgecolor('gold')
        bars1[fed_idx].set_linewidth(3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, waiting_times, waiting_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + height*0.02,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Queue Length Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(methods)), queue_lengths, yerr=queue_stds,
                    color=bar_colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax2.set_title('Queue Length Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    if 'federated' in methods:
        fed_idx = methods.index('federated')
        bars2[fed_idx].set_edgecolor('gold')
        bars2[fed_idx].set_linewidth(3)
    
    for i, (bar, val, std) in enumerate(zip(bars2, queue_lengths, queue_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + height*0.02,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Improvement Percentage (relative to Fixed-Time)
    ax3 = fig.add_subplot(2, 3, 3)
    if 'fixed_time' in methods:
        baseline_waiting = waiting_times[methods.index('fixed_time')]
        improvements = [(baseline_waiting - wt) / baseline_waiting * 100 for wt in waiting_times]
        
        bars3 = ax3.bar(range(len(methods)), improvements, color=bar_colors, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Improvement Over Fixed-Time', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(display_names, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        if 'federated' in methods:
            fed_idx = methods.index('federated')
            bars3[fed_idx].set_edgecolor('gold')
            bars3[fed_idx].set_linewidth(3)
        
        for i, (bar, val) in enumerate(zip(bars3, improvements)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # 4. Radar/Spider Chart (if we have multiple metrics)
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    
    # Normalize metrics for radar chart
    max_waiting = max(waiting_times) if waiting_times else 1
    max_queue = max(queue_lengths) if queue_lengths else 1
    
    # Plot federated method if available
    if 'federated' in methods:
        fed_idx = methods.index('federated')
        angles = np.linspace(0, 2 * np.pi, 2, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fed_values = [
            1 - (waiting_times[fed_idx] / max_waiting),  # Lower is better
            1 - (queue_lengths[fed_idx] / max_queue),    # Lower is better
        ]
        fed_values += fed_values[:1]
        
        ax4.plot(angles, fed_values, 'o-', linewidth=2, label='Federated FL', color='#2ecc71')
        ax4.fill(angles, fed_values, alpha=0.25, color='#2ecc71')
    
    # Plot best classical method (MaxPressure if available)
    if 'maxpressure' in methods:
        mp_idx = methods.index('maxpressure')
        mp_values = [
            1 - (waiting_times[mp_idx] / max_waiting),
            1 - (queue_lengths[mp_idx] / max_queue),
        ]
        mp_values += mp_values[:1]
        ax4.plot(angles, mp_values, 'o-', linewidth=2, label='MaxPressure', color='#3498db')
        ax4.fill(angles, mp_values, alpha=0.15, color='#3498db')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(['Waiting Time\n(Lower Better)', 'Queue Length\n(Lower Better)'])
    ax4.set_ylim(0, 1)
    ax4.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.grid(True)
    
    # 5. Method Categories
    ax5 = fig.add_subplot(2, 3, 5)
    categories = {
        'Classical': ['fixed_time', 'actuated', 'maxpressure', 'sotl'],
        'RL-Based': ['centralized_rl', 'independent_rl', 'federated']
    }
    
    category_means = {}
    for category, method_list in categories.items():
        cat_methods = [m for m in method_list if m in methods]
        if cat_methods:
            cat_waiting = [waiting_times[methods.index(m)] for m in cat_methods]
            category_means[category] = np.mean(cat_waiting)
    
    if category_means:
        cat_names = list(category_means.keys())
        cat_values = list(category_means.values())
        cat_colors = ['#e74c3c', '#3498db']
        
        bars5 = ax5.bar(cat_names, cat_values, color=cat_colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Average Waiting Time (s)', fontsize=12, fontweight='bold')
        ax5.set_title('Performance by Category', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars5, cat_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create table
    table_data = []
    table_data.append(['Method', 'Waiting Time (s)', 'Queue Length'])
    table_data.append(['-' * 20, '-' * 20, '-' * 15])
    
    for method in methods:
        display_name = method_labels.get(method, method.replace('_', ' ').title())
        wt = f"{waiting_times[methods.index(method)]:.1f} ± {waiting_stds[methods.index(method)]:.1f}"
        ql = f"{queue_lengths[methods.index(method)]:.1f} ± {queue_stds[methods.index(method)]:.1f}"
        table_data.append([display_name, wt, ql])
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
    
    # Highlight federated row
    if 'federated' in methods:
        fed_row = methods.index('federated') + 2
        for j in range(3):
            cell = table[(fed_row, j)]
            cell.set_facecolor('#d5f4e6')
            cell.set_text_props(weight='bold')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Comparison plots saved to: {output_file}")
    
    # Also create individual plots
    create_individual_plots(summary, output_dir, method_labels, colors)
    
    plt.show()


def create_individual_plots(summary: Dict, output_dir: str, method_labels: Dict, colors: Dict):
    """Create individual comparison plots."""
    methods = list(summary.keys())
    waiting_times = [summary[m]['mean_waiting_time'] for m in methods]
    waiting_stds = [summary[m]['std_waiting_time'] for m in methods]
    queue_lengths = [summary[m]['mean_queue_length'] for m in methods]
    queue_stds = [summary[m]['std_queue_length'] for m in methods]
    display_names = [method_labels.get(m, m.replace('_', ' ').title()) for m in methods]
    bar_colors = [colors.get(m, '#34495e') for m in methods]
    
    # 1. Waiting Time Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(methods)), waiting_times, yerr=waiting_stds,
                  color=bar_colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    
    if 'federated' in methods:
        fed_idx = methods.index('federated')
        bars[fed_idx].set_edgecolor('gold')
        bars[fed_idx].set_linewidth(3)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Waiting Time (s)', fontsize=14, fontweight='bold')
    ax.set_title('Waiting Time Comparison Across Methods', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val, std) in enumerate(zip(bars, waiting_times, waiting_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + height*0.02,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waiting_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Queue Length Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(methods)), queue_lengths, yerr=queue_stds,
                  color=bar_colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)
    
    if 'federated' in methods:
        fed_idx = methods.index('federated')
        bars[fed_idx].set_edgecolor('gold')
        bars[fed_idx].set_linewidth(3)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Queue Length (vehicles)', fontsize=14, fontweight='bold')
    ax.set_title('Queue Length Comparison Across Methods', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val, std) in enumerate(zip(bars, queue_lengths, queue_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + height*0.02,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'queue_length_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Individual plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Baseline Comparison Results")
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to comparison results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    results = load_comparison_results(args.results_file)
    summary = results.get('summary', {})
    
    if not summary:
        print("Error: No summary data found in results file")
        sys.exit(1)
    
    print(f"Loaded comparison results from: {args.results_file}")
    print(f"Found {len(summary)} methods")
    
    create_comparison_plots(summary, args.output_dir)

