#!/usr/bin/env python3x
"""
Baseline Comparison Script

Evaluates Federated Learning performance against:
1. Fixed-Time Control
2. MaxPressure
3. Centralized RL
4. Independent RL
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, List
import traci
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.fixed_time import FixedTimeController
from baselines.maxpressure import MaxPressureController
from baselines.centralized_rl import CentralizedRLController
from baselines.independent_rl import IndependentRLController
from agents.traffic_environment import SUMOTrafficEnvironment
from federated_learning.fl_client import TrafficFLClient
from train_federated import run_multi_client_simulation


def run_classical_baseline(
    controller_class,
    controller_name: str,
    sumo_config: str,
    num_steps: int = 400,
    gui: bool = False,
    **controller_kwargs
) -> Dict:
    """Run a classical baseline controller."""
    print(f"Running {controller_name}...")
    
    env = SUMOTrafficEnvironment(sumo_config, gui=gui)
    env.start_simulation()
    
    try:
        tl_id = env.tl_id
        if not tl_id:
            tls = traci.trafficlight.getIDList()
            if tls:
                tl_id = tls[0]
        
        incoming_edges = env.incoming_edges
        
        # Create controller
        if 'incoming_edges' in controller_kwargs:
            controller_kwargs['incoming_edges'] = incoming_edges
        
        controller = controller_class(tl_id, **controller_kwargs)
        controller.reset()
        
        # Run simulation
        for step in range(num_steps):
            current_time = traci.simulation.getTime()
            controller.step(current_time)
            traci.simulationStep()
        
        metrics = env.get_performance_metrics()
        env.close()
        
        return {
            'method': controller_name,
            'metrics': metrics,
            'success': True
        }
    except Exception as e:
        print(f"Error in {controller_name}: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return {
            'method': controller_name,
            'error': str(e),
            'success': False
        }


def run_federated_method(
    sumo_configs: List[str],
    num_rounds: int = 15,
    results_dir: str = "temp_federated",
    gui: bool = False
) -> Dict:
    """Run your federated learning method."""
    print("Running Federated Learning Method...")
    
    try:
        # Use the existing multi-client simulation
        run_multi_client_simulation(results_dir=results_dir)
        
        # Load results from all clients
        if not os.path.exists(results_dir):
            return {'method': 'federated', 'error': 'Results directory not found', 'success': False}
        
        eval_files = [f for f in os.listdir(results_dir) if f.endswith('_eval.json')]
        if not eval_files:
            return {'method': 'federated', 'error': 'No results found', 'success': False}
        
        # Aggregate metrics from all clients and rounds
        all_waiting_times = []
        all_queue_lengths = []
        all_max_queue_lengths = []
        all_avg_waiting_per_vehicle = []
        
        for eval_file in eval_files:
            try:
                with open(os.path.join(results_dir, eval_file), 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for entry in data:
                        metrics = entry.get('metrics', {})
                        all_waiting_times.append(metrics.get('waiting_time', 0))
                        all_queue_lengths.append(metrics.get('queue_length', 0))
                        all_max_queue_lengths.append(metrics.get('max_queue_length', 0))
                        all_avg_waiting_per_vehicle.append(metrics.get('avg_waiting_time_per_vehicle', 0))
            except Exception as e:
                print(f"Warning: Could not load {eval_file}: {e}")
                continue
        
        if not all_waiting_times:
            return {'method': 'federated', 'error': 'No valid metrics found', 'success': False}
        
        # Calculate averages
        return {
            'method': 'federated',
            'metrics': {
                'total_waiting_time': float(np.mean(all_waiting_times)),
                'average_queue_length': float(np.mean(all_queue_lengths)),
                'max_queue_length': float(np.mean(all_max_queue_lengths)),
                'avg_waiting_time_per_vehicle': float(np.mean(all_avg_waiting_per_vehicle)),
            },
            'success': True
        }
    except Exception as e:
        print(f"Error in federated method: {e}")
        import traceback
        traceback.print_exc()
        return {'method': 'federated', 'error': str(e), 'success': False}


def compare_all_methods(
    sumo_config: str,
    sumo_configs_multi: List[str] = None,
    num_runs: int = 5,
    num_steps: int = 400,
    results_dir: str = "baseline_comparison",
    gui: bool = False
):
    """
    Compare all baseline methods.
    
    Args:
        sumo_config: Single SUMO config for classical baselines
        sumo_configs_multi: List of configs for RL methods (multi-intersection)
        num_runs: Number of runs per method
        num_steps: Simulation steps for classical methods
        results_dir: Directory to save results
        gui: Enable GUI
    """
    os.makedirs(results_dir, exist_ok=True)
    
    if sumo_configs_multi is None:
        sumo_configs_multi = [
            "sumo_configs/osm_client1.sumocfg",
            "sumo_configs/osm_client2.sumocfg"
        ]
    
    all_results = {
        'fixed_time': [],
        'maxpressure': [],
        'centralized_rl': [],
        'independent_rl': [],
        'federated': []
    }
    
    print(f"Running {num_runs} runs for each method...")
    print("=" * 60)
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{num_runs}")
        print(f"{'='*60}")
        
        # Baseline executions
        result_ft = run_classical_baseline(
            FixedTimeController,
            'fixed_time',
            sumo_config,
            num_steps=num_steps,
            gui=gui
        )
        all_results['fixed_time'].append(result_ft)
        
        result_mp = run_classical_baseline(
            MaxPressureController,
            'maxpressure',
            sumo_config,
            num_steps=num_steps,
            gui=gui,
            min_phase_duration=5.0
        )
        all_results['maxpressure'].append(result_mp)
        
        if run == 0:  # Only run once due to training time
            print("\nTraining Centralized RL (this may take a while)...")
            try:
                centralized = CentralizedRLController(
                    sumo_configs_multi,
                    gui=gui
                )
                train_results = centralized.train(num_rounds=10, episodes_per_round=2)
                eval_results = centralized.evaluate()
                all_results['centralized_rl'].append({
                    'method': 'centralized_rl',
                    'metrics': eval_results['metrics'],
                    'training_history': train_results['training_history'],
                    'success': True
                })
            except Exception as e:
                print(f"Error in centralized RL: {e}")
                all_results['centralized_rl'].append({
                    'method': 'centralized_rl',
                    'error': str(e),
                    'success': False
                })
        
        if run == 0:
            print("\nTraining Independent RL (this may take a while)...")
            try:
                independent = IndependentRLController(
                    sumo_configs_multi,
                    gui=gui
                )
                train_results = independent.train(num_rounds=10, episodes_per_round=2)
                eval_results = independent.evaluate()
                all_results['independent_rl'].append({
                    'method': 'independent_rl',
                    'metrics': eval_results['metrics'],
                    'training_history': train_results['training_history'],
                    'success': True
                })
            except Exception as e:
                print(f"Error in independent RL: {e}")
                all_results['independent_rl'].append({
                    'method': 'independent_rl',
                    'error': str(e),
                    'success': False
                })
        
        if run == 0:
            print("\nTraining Federated Learning (this may take a while)...")
            temp_dir = os.path.join(results_dir, "temp_federated")
            result_fl = run_federated_method(
                sumo_configs_multi,
                num_rounds=15,
                results_dir=temp_dir,
                gui=gui
            )
            all_results['federated'].append(result_fl)
    
    # Calculate statistics
    summary = {}
    for method, results in all_results.items():
        if not results:
            continue
        
        # Filter successful runs
        successful_results = [r for r in results if r.get('success', False)]
        if not successful_results:
            continue
        
        # Extract metrics
        if method in ['centralized_rl', 'independent_rl', 'federated']:
            # RL methods have different structure
            metrics_list = [r.get('metrics', {}) for r in successful_results]
        else:
            # Classical methods
            metrics_list = [r.get('metrics', {}) for r in successful_results]
        
        if not metrics_list:
            continue
        
        # Calculate statistics
        waiting_times = [m.get('total_waiting_time', 0) for m in metrics_list]
        queue_lengths = [m.get('average_queue_length', 0) for m in metrics_list]
        
        summary[method] = {
            'mean_waiting_time': float(np.mean(waiting_times)),
            'std_waiting_time': float(np.std(waiting_times)),
            'mean_queue_length': float(np.mean(queue_lengths)),
            'std_queue_length': float(np.std(queue_lengths)),
            'num_runs': len(successful_results)
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'comparison_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'sumo_config': sumo_config,
                'num_runs': num_runs,
                'num_steps': num_steps
            },
            'all_results': all_results,
            'summary': summary
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_file}\n")
    
    # Print summary table
    print(f"{'Method':<20} {'Avg Waiting Time':<20} {'Avg Queue Length':<20} {'Runs':<10}")
    print("-" * 70)
    for method, stats in summary.items():
        print(f"{method:<20} {stats['mean_waiting_time']:>10.2f} ± {stats['std_waiting_time']:<7.2f} "
              f"{stats['mean_queue_length']:>10.2f} ± {stats['std_queue_length']:<7.2f} "
              f"{stats['num_runs']:>10}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Baseline Comparison")
    parser.add_argument(
        "--sumo-config",
        type=str,
        default="sumo_configs/intersection.sumocfg",
        help="SUMO config file for classical baselines"
    )
    parser.add_argument(
        "--sumo-configs-multi",
        type=str,
        nargs='+',
        default=["sumo_configs/osm_client1.sumocfg", "sumo_configs/osm_client2.sumocfg"],
        help="SUMO config files for RL methods (multi-intersection)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per method (classical methods only)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=400,
        help="Simulation steps for classical methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="baseline_comparison",
        help="Results directory"
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    
    args = parser.parse_args()
    
    compare_all_methods(
        args.sumo_config,
        args.sumo_configs_multi,
        args.num_runs,
        args.num_steps,
        args.results_dir,
        args.gui
    )

