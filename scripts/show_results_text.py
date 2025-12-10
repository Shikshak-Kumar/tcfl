#!/usr/bin/env python3
"""
Text-based visualization of comparison results.
Does not require matplotlib - shows results in terminal.
"""

import os
import sys
import json
import argparse

def print_comparison_table(summary: dict):
    """Print a formatted comparison table."""
    
    method_labels = {
        'fixed_time': 'Fixed-Time',
        'maxpressure': 'MaxPressure',
        'centralized_rl': 'Centralized RL',
        'independent_rl': 'Independent RL',
        'federated': 'Federated FL (Ours)'
    }
    
    methods = list(summary.keys())
    
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON RESULTS".center(90))
    print("=" * 90)
    print(f"\n{'Method':<25} {'Avg Waiting Time (s)':<25} {'Avg Queue Length':<25} {'Runs':<10}")
    print("-" * 90)
    
    for method in methods:
        stats = summary[method]
        display_name = method_labels.get(method, method.replace('_', ' ').title())
        
        waiting_str = f"{stats['mean_waiting_time']:.2f} ± {stats['std_waiting_time']:.2f}"
        queue_str = f"{stats['mean_queue_length']:.2f} ± {stats['std_queue_length']:.2f}"
        
        # Highlight federated method
        if method == 'federated':
            print(f"{'⭐ ' + display_name:<25} {waiting_str:<25} {queue_str:<25} {stats['num_runs']:<10}")
        else:
            print(f"{display_name:<25} {waiting_str:<25} {queue_str:<25} {stats['num_runs']:<10}")
    
    print("-" * 90)
    
    # Calculate improvements
    if 'fixed_time' in summary:
        baseline_waiting = summary['fixed_time']['mean_waiting_time']
        baseline_queue = summary['fixed_time']['mean_queue_length']
        
        print("\n" + "=" * 90)
        print("IMPROVEMENT OVER FIXED-TIME BASELINE".center(90))
        print("=" * 90)
        print(f"\n{'Method':<25} {'Waiting Time Improvement':<30} {'Queue Length Improvement':<30}")
        print("-" * 90)
        
        for method in methods:
            if method == 'fixed_time':
                continue
            
            stats = summary[method]
            display_name = method_labels.get(method, method.replace('_', ' ').title())
            
            wait_improvement = ((baseline_waiting - stats['mean_waiting_time']) / baseline_waiting) * 100
            queue_improvement = ((baseline_queue - stats['mean_queue_length']) / baseline_queue) * 100
            
            wait_str = f"{wait_improvement:+.1f}%"
            queue_str = f"{queue_improvement:+.1f}%"
            
            if method == 'federated':
                print(f"{'⭐ ' + display_name:<25} {wait_str:<30} {queue_str:<30}")
            else:
                print(f"{display_name:<25} {wait_str:<30} {queue_str:<30}")
        
        print("-" * 90)
    
    # Best performer
    print("\n" + "=" * 90)
    print("KEY FINDINGS".center(90))
    print("=" * 90)
    
    best_waiting = min(methods, key=lambda m: summary[m]['mean_waiting_time'])
    best_queue = min(methods, key=lambda m: summary[m]['mean_queue_length'])
    
    print(f"\n✓ Best Waiting Time: {method_labels.get(best_waiting, best_waiting)} "
          f"({summary[best_waiting]['mean_waiting_time']:.2f}s)")
    print(f"✓ Best Queue Length: {method_labels.get(best_queue, best_queue)} "
          f"({summary[best_queue]['mean_queue_length']:.2f} vehicles)")
    
    if 'federated' in summary:
        fed_waiting = summary['federated']['mean_waiting_time']
        fed_queue = summary['federated']['mean_queue_length']
        
        if best_waiting == 'federated':
            print(f"\n🎉 Federated Learning achieves BEST waiting time performance!")
        if best_queue == 'federated':
            print(f"🎉 Federated Learning achieves BEST queue length performance!")
        
        # Compare with other RL methods
        if 'centralized_rl' in summary:
            cent_waiting = summary['centralized_rl']['mean_waiting_time']
            improvement = ((cent_waiting - fed_waiting) / cent_waiting) * 100
            print(f"\n📊 Federated vs Centralized RL: {improvement:+.1f}% improvement "
                  f"(with privacy preservation)")
        
        if 'independent_rl' in summary:
            ind_waiting = summary['independent_rl']['mean_waiting_time']
            improvement = ((ind_waiting - fed_waiting) / ind_waiting) * 100
            print(f"📊 Federated vs Independent RL: {improvement:+.1f}% improvement "
                  f"(demonstrates collaboration benefits)")
    
    print("\n" + "=" * 90)


def print_ascii_chart(summary: dict, metric: str = 'waiting_time'):
    """Print ASCII bar chart."""
    
    method_labels = {
        'fixed_time': 'Fixed-Time',
        'actuated': 'Actuated',
        'maxpressure': 'MaxPressure',
        'sotl': 'SOTL',
        'centralized_rl': 'Centralized RL',
        'independent_rl': 'Independent RL',
        'federated': 'Federated FL'
    }
    
    methods = list(summary.keys())
    
    if metric == 'waiting_time':
        values = [summary[m]['mean_waiting_time'] for m in methods]
        title = "Average Waiting Time (s)"
        max_val = max(values) if values else 50
    else:
        values = [summary[m]['mean_queue_length'] for m in methods]
        title = "Average Queue Length (vehicles)"
        max_val = max(values) if values else 10
    
    print(f"\n{'=' * 70}")
    print(f"{title}".center(70))
    print(f"{'=' * 70}\n")
    
    # Scale to 50 characters width
    scale = 50 / max_val if max_val > 0 else 1
    
    for method, value in zip(methods, values):
        display_name = method_labels.get(method, method.replace('_', ' ').title())
        bar_length = int(value * scale)
        bar = '█' * bar_length
        
        # Highlight federated
        if method == 'federated':
            print(f"⭐ {display_name:<20} │{bar} {value:.2f}")
        else:
            print(f"  {display_name:<20} │{bar} {value:.2f}")
    
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Display comparison results in text format")
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to comparison results JSON file"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    summary = results.get('summary', {})
    
    if not summary:
        print("Error: No summary data found in results file")
        sys.exit(1)
    
    # Print comparison table
    print_comparison_table(summary)
    
    # Print ASCII charts
    print_ascii_chart(summary, 'waiting_time')
    print_ascii_chart(summary, 'queue_length')
    
    print("\n" + "=" * 90)
    print("Results file: " + args.results_file)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

