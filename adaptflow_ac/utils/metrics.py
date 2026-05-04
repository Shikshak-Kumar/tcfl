import numpy as np
import json
import os

class MetricsTracker:
    """
    Utility for tracking and logging traffic metrics for research papers.
    Tracks:
    - Average waiting time
    - Average queue length
    - Throughput
    - Congested lanes count
    - Episode reward
    """
    def __init__(self, node_ids):
        self.node_ids = node_ids
        self.history = {nid: {
            "waiting_time": [],
            "queue_length": [],
            "queue_total": [],
            "queue_max": [],
            "throughput": [],
            "congested_lanes": [],
            "rewards": []
        } for nid in node_ids}

    def update(self, node_id, metrics, reward):
        """
        metrics: dict containing 'avg_waiting', 'avg_queue', optional 'queue_total_halting',
                  'queue_max_lane', 'throughput', 'congested_lanes'
        reward: scalar reward value
        """
        if node_id not in self.history:
            return
            
        self.history[node_id]["waiting_time"].append(metrics.get("avg_waiting", 0))
        self.history[node_id]["queue_length"].append(metrics.get("avg_queue", 0))
        self.history[node_id]["queue_total"].append(
            metrics.get("queue_total_halting", metrics.get("avg_queue", 0))
        )
        self.history[node_id]["queue_max"].append(metrics.get("queue_max_lane", 0))
        self.history[node_id]["throughput"].append(
            metrics.get("arrival_rate", metrics.get("throughput", 0))
        )
        self.history[node_id]["congested_lanes"].append(metrics.get("congested_lanes", 0))
        self.history[node_id]["rewards"].append(reward)

    def get_summary(self, node_id=None):
        """Returns mean metrics for a node or all nodes."""
        if node_id:
            return {k: np.mean(v) if v else 0 for k, v in self.history[node_id].items()}
        
        summary = {}
        for nid in self.node_ids:
            summary[nid] = {k: np.mean(v) if v else 0 for k, v in self.history[nid].items()}
        return summary

    def reset_round(self) -> None:
        """Clear per-step history at the start of each federated round."""
        for nid in self.node_ids:
            for k in self.history[nid]:
                self.history[nid][k].clear()

    def save(self, path):
        """Saves history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def print_table(self):
        """Prints a summary table to console."""
        print("\n" + "="*80)
        print(f"{'Node ID':<15} | {'Wait (s)':<10} | {'Queue':<10} | {'Arr/s':<10} | {'Congest':<10} | {'Reward':<10}")
        print("-" * 80)
        for nid in self.node_ids:
            s = self.get_summary(nid)
            print(f"{nid:<15} | {s['waiting_time']:<10.2f} | {s['queue_length']:<10.2f} | {s['throughput']:<10.2f} | {s['congested_lanes']:<10.2f} | {s['rewards']:<10.2f}")
        print("="*80 + "\n")
