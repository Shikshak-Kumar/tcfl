import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
import glob
import json

class TrafficVisualizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_convergence(self, metrics_data: List[Dict], 
                                save_path: Optional[str] = None):
        """Plot training convergence metrics"""
        if not metrics_data:
            print("No metrics data to plot")
            return
        
        # Extract data
        rounds = [m['round'] for m in metrics_data if m['type'] == 'fit']
        rewards = [m['metrics'].get('avg_average_reward', 0) for m in metrics_data if m['type'] == 'fit']
        losses = [m['metrics'].get('avg_average_loss', 0) for m in metrics_data if m['type'] == 'fit']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards
        ax1.plot(rounds, rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Training Reward Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Training Loss Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _load_client_eval_history(self, client_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict]]:
        pattern = os.path.join(self.results_dir, f"{client_id}_round_*_eval.json")
        files = glob.glob(pattern)
        if not files:
            print(f"No eval files found for {client_id} in {self.results_dir}")
            return np.array([]), np.array([]), {}

        def _extract_round(path: str) -> int:
            name = os.path.basename(path).replace(".json", "")
            parts = name.split("_")
            try:
                return int(parts[3])
            except Exception:
                return -1

        latest_file = max(files, key=_extract_round)
        with open(latest_file, "r") as f:
            data = json.load(f)

        rounds: List[int] = []
        avg_wait_per_vehicle: List[float] = []
        lane_summaries_by_round: Dict[int, Dict] = {}

        for entry in data:
            r = int(entry.get("round", len(rounds)))
            metrics = entry.get("metrics", {})
            rounds.append(r)
            avg_wait = float(metrics.get("avg_waiting_time_per_vehicle", 0.0))
            avg_wait_per_vehicle.append(avg_wait)
            lane_summaries_by_round[r] = metrics.get("lane_summary", {})

        return np.array(rounds), np.array(avg_wait_per_vehicle), lane_summaries_by_round

    def plot_wait_and_congestion_per_client(self, save_path: Optional[str] = None):
        client_ids = ["client_1", "client_2"]

        c1_rounds, c1_wait, c1_lane = self._load_client_eval_history(client_ids[0])
        c2_rounds, c2_wait, c2_lane = self._load_client_eval_history(client_ids[1])

        if c1_rounds.size == 0 or c2_rounds.size == 0:
            print("Not enough eval data for both clients to plot.")
            return

        common_rounds = sorted(set(c1_rounds.tolist()) & set(c2_rounds.tolist()))
        if not common_rounds:
            print("No common rounds between clients to plot congestion/weights.")
            return

        c1_cong: List[float] = []
        c2_cong: List[float] = []

        for r in common_rounds:
            s1 = c1_lane.get(r, {})
            s2 = c2_lane.get(r, {})

            def compute_congestion(summary: Dict) -> float:
                lane_wait = float(summary.get("total_waiting_time", 0.0))
                num_congested = float(summary.get("num_congested_lanes", 0.0))
                total_queue = float(summary.get("total_queue_length", 0.0))
                return max(lane_wait + 20.0 * total_queue + 50.0 * num_congested, 0.0)

            c1_cong.append(compute_congestion(s1))
            c2_cong.append(compute_congestion(s2))

        c1_cong_arr = np.array(c1_cong)
        c2_cong_arr = np.array(c2_cong)
        cong_sum = c1_cong_arr + c2_cong_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            c1_w = np.where(cong_sum > 0, c1_cong_arr / cong_sum, 0.5)
            c2_w = np.where(cong_sum > 0, c2_cong_arr / cong_sum, 0.5)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]
        ax.plot(c1_rounds, c1_wait, "b-o", linewidth=2, markersize=5)
        ax.set_title("Client 1: Avg Waiting Time per Vehicle")
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg Waiting Time per Vehicle (s)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(c2_rounds, c2_wait, "g-o", linewidth=2, markersize=5)
        ax.set_title("Client 2: Avg Waiting Time per Vehicle")
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg Waiting Time per Vehicle (s)")
        ax.grid(True, alpha=0.3)

        ax_c1 = axes[0, 1]
        ax_c1.plot(common_rounds, c1_cong_arr, "r-o", linewidth=2, markersize=5, label="Congestion score")
        ax_c1.set_xlabel("Round")
        ax_c1.set_ylabel("Congestion score", color="r")
        ax_c1.tick_params(axis="y", labelcolor="r")
        ax_c1.grid(True, alpha=0.3)
        ax_c1.set_title("Client 1: Congestion vs Aggregation Weight")

        ax_c1_w = ax_c1.twinx()
        ax_c1_w.plot(common_rounds, c1_w, "b-s", linewidth=2, markersize=5, label="Weight")
        ax_c1_w.set_ylabel("Aggregation weight", color="b")
        ax_c1_w.tick_params(axis="y", labelcolor="b")

        ax_c2 = axes[1, 1]
        ax_c2.plot(common_rounds, c2_cong_arr, "r-o", linewidth=2, markersize=5, label="Congestion score")
        ax_c2.set_xlabel("Round")
        ax_c2.set_ylabel("Congestion score", color="r")
        ax_c2.tick_params(axis="y", labelcolor="r")
        ax_c2.grid(True, alpha=0.3)
        ax_c2.set_title("Client 2: Congestion vs Aggregation Weight")

        ax_c2_w = ax_c2.twinx()
        ax_c2_w.plot(common_rounds, c2_w, "b-s", linewidth=2, markersize=5, label="Weight")
        ax_c2_w.set_ylabel("Aggregation weight", color="b")
        ax_c2_w.tick_params(axis="y", labelcolor="b")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_performance_metrics(self, metrics_data: List[Dict], 
                               save_path: Optional[str] = None):
        """Plot performance metrics over rounds"""
        if not metrics_data:
            print("No metrics data to plot")
            return
        
        # Extract evaluation data
        eval_data = [m for m in metrics_data if m['type'] == 'evaluate']
        if not eval_data:
            print("No evaluation data to plot")
            return
        
        rounds = [m['round'] for m in eval_data]
        waiting_times = [m['metrics'].get('avg_waiting_time', 0) for m in eval_data]
        queue_lengths = [m['metrics'].get('avg_queue_length', 0) for m in eval_data]
        max_queue_lengths = [m['metrics'].get('avg_max_queue_length', 0) for m in eval_data]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot waiting times
        ax1.plot(rounds, waiting_times, 'g-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Average Waiting Time (s)')
        ax1.set_title('Traffic Waiting Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot queue lengths
        ax2.plot(rounds, queue_lengths, 'orange', marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('Average Queue Length')
        ax2.set_title('Traffic Queue Length')
        ax2.grid(True, alpha=0.3)
        
        # Plot max queue lengths
        ax3.plot(rounds, max_queue_lengths, 'purple', marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Federated Learning Round')
        ax3.set_ylabel('Max Queue Length')
        ax3.set_title('Maximum Queue Length')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_client_comparison(self, client_metrics: Dict, 
                             save_path: Optional[str] = None):
        """Plot comparison between different clients"""
        if not client_metrics:
            print("No client metrics to plot")
            return
        
        # Prepare data
        clients = list(client_metrics.keys())
        rewards = []
        waiting_times = []
        
        for client_id, metrics in client_metrics.items():
            if metrics:
                avg_reward = np.mean([m.get('average_reward', 0) for m in metrics])
                avg_waiting = np.mean([m.get('waiting_time', 0) for m in metrics])
                rewards.append(avg_reward)
                waiting_times.append(avg_waiting)
            else:
                rewards.append(0)
                waiting_times.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards by client
        bars1 = ax1.bar(clients, rewards, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Client Performance Comparison - Rewards')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot waiting times by client
        bars2 = ax2.bar(clients, waiting_times, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Average Waiting Time (s)')
        ax2.set_title('Client Performance Comparison - Waiting Times')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, waiting_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, training_data: List[Dict], 
                           save_path: Optional[str] = None):
        """Plot detailed learning curves"""
        if not training_data:
            print("No training data to plot")
            return
        
        # Extract data
        rounds = [d['round'] for d in training_data]
        episodes = [d['episodes'] for d in training_data]
        rewards = [d['metrics'].get('average_reward', 0) for d in training_data]
        losses = [d['metrics'].get('average_loss', 0) for d in training_data]
        steps = [d['metrics'].get('total_steps', 0) for d in training_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot rewards
        ax1.plot(rounds, rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Progression')
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Loss Progression')
        ax2.grid(True, alpha=0.3)
        
        # Plot episodes
        ax3.plot(rounds, episodes, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Episodes')
        ax3.set_title('Training Episodes per Round')
        ax3.grid(True, alpha=0.3)
        
        # Plot steps
        ax4.plot(rounds, steps, 'purple', marker='o', linewidth=2, markersize=6)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Total Steps')
        ax4.set_title('Training Steps per Round')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, metrics_data: List[Dict], 
                            client_metrics: Dict, 
                            save_path: Optional[str] = None):
        """Create a comprehensive summary report"""
        if not metrics_data:
            print("No data to create report")
            return
        
        # Create summary statistics
        eval_data = [m for m in metrics_data if m['type'] == 'evaluate']
        fit_data = [m for m in metrics_data if m['type'] == 'fit']
        
        report = {
            'summary': {
                'total_rounds': len(fit_data),
                'total_evaluations': len(eval_data),
                'num_clients': len(client_metrics)
            },
            'final_performance': {},
            'improvement': {}
        }
        
        if eval_data:
            final_metrics = eval_data[-1]['metrics']
            report['final_performance'] = {
                'average_reward': final_metrics.get('avg_average_reward', 0),
                'waiting_time': final_metrics.get('avg_waiting_time', 0),
                'queue_length': final_metrics.get('avg_queue_length', 0),
                'max_queue_length': final_metrics.get('avg_max_queue_length', 0)
            }
        
        if len(eval_data) > 1:
            initial_metrics = eval_data[0]['metrics']
            final_metrics = eval_data[-1]['metrics']
            
            report['improvement'] = {
                'reward_improvement': final_metrics.get('avg_average_reward', 0) - initial_metrics.get('avg_average_reward', 0),
                'waiting_time_reduction': initial_metrics.get('avg_waiting_time', 0) - final_metrics.get('avg_waiting_time', 0),
                'queue_length_reduction': initial_metrics.get('avg_queue_length', 0) - final_metrics.get('avg_queue_length', 0)
            }
        
        # Save report
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Print summary
        print("=== FEDERATED LEARNING SUMMARY REPORT ===")
        print(f"Total Rounds: {report['summary']['total_rounds']}")
        print(f"Total Evaluations: {report['summary']['total_evaluations']}")
        print(f"Number of Clients: {report['summary']['num_clients']}")
        print("\nFinal Performance:")
        for key, value in report['final_performance'].items():
            print(f"  {key}: {value:.4f}")
        print("\nImprovements:")
        for key, value in report['improvement'].items():
            print(f"  {key}: {value:.4f}")
        print("=" * 40)
        
        return report
