#!/usr/bin/env python3
"""
Complete pipeline: Run comparison and generate visualizations.

This script runs the baseline comparison and automatically generates graphs.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_comparison(
    sumo_config: str,
    sumo_configs_multi: list,
    num_runs: int = 3,
    num_steps: int = 200,  # Reduced for faster testing
    gui: bool = False
):
    """Run the baseline comparison."""
    print("=" * 70)
    print("RUNNING BASELINE COMPARISON")
    print("=" * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), "compare_all_baselines.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--sumo-config", sumo_config,
        "--num-runs", str(num_runs),
        "--num-steps", str(num_steps),
        "--results-dir", "baseline_comparison"
    ]
    
    # Add multi-config files
    for config in sumo_configs_multi:
        cmd.extend(["--sumo-configs-multi", config])
    
    if gui:
        cmd.append("--gui")
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ Comparison completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Comparison failed with error: {e}")
        return False


def find_latest_results(results_dir: str = "baseline_comparison"):
    """Find the latest results file."""
    if not os.path.exists(results_dir):
        return None
    
    json_files = [f for f in os.listdir(results_dir) if f.startswith('comparison_results_') and f.endswith('.json')]
    if not json_files:
        return None
    
    # Sort by timestamp
    json_files.sort(reverse=True)
    return os.path.join(results_dir, json_files[0])


def run_visualization(results_file: str):
    """Run the visualization script."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), "visualize_comparison.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--results-file", results_file,
        "--output-dir", "comparison_plots"
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ Visualizations generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Visualization failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Comparison and Generate Visualizations")
    parser.add_argument(
        "--sumo-config",
        type=str,
        default="sumo_configs/intersection.sumocfg",
        help="SUMO config for classical baselines"
    )
    parser.add_argument(
        "--sumo-configs-multi",
        type=str,
        nargs='+',
        default=["sumo_configs/osm_client1.sumocfg", "sumo_configs/osm_client2.sumocfg"],
        help="SUMO configs for RL methods"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs (reduced for faster execution)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Simulation steps (reduced for faster execution)"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip comparison, only generate visualizations"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to existing results file (for visualization only)"
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    
    args = parser.parse_args()
    
    # Run comparison
    if not args.skip_comparison:
        success = run_comparison(
            args.sumo_config,
            args.sumo_configs_multi,
            args.num_runs,
            args.num_steps,
            args.gui
        )
        if not success:
            print("\n✗ Comparison failed. Cannot generate visualizations.")
            return
    
    # Find results file
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = find_latest_results()
    
    if not results_file or not os.path.exists(results_file):
        print(f"\n✗ Results file not found: {results_file}")
        print("Please run comparison first or specify --results-file")
        return
    
    print(f"\nUsing results file: {results_file}")
    
    # Generate visualizations
    success = run_visualization(results_file)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ COMPLETE!")
        print("=" * 70)
        print(f"\nResults: {results_file}")
        print(f"Plots: comparison_plots/")
        print("\nGenerated files:")
        print("  - baseline_comparison.png (comprehensive comparison)")
        print("  - waiting_time_comparison.png")
        print("  - queue_length_comparison.png")


if __name__ == "__main__":
    main()

