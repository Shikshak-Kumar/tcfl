import torch
import copy
from typing import List, Dict, Optional

class FedFlowServer:
    """
    Central Server for Inter-Cluster Meta-Aggregation (Level 2).
    Aggregates cluster models into a global baseline θ_global.
    """
    def __init__(self, cluster_ids: List[str]):
        self.cluster_ids = cluster_ids
        self.global_weights = None
        self.cluster_congestion = {cid: 1.0 for cid in cluster_ids}
        
    def update_cluster_metrics(self, cluster_id: str, congestion_score: float):
        """Update the congestion metric for a cluster (e.g. avg waiting time)."""
        if cluster_id in self.cluster_ids:
            self.cluster_congestion[cluster_id] = congestion_score
            
    def aggregate_inter_cluster(self, cluster_parameters: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate weights from multiple clusters.
        θ_global = Σ w_cluster θ_cluster
        Weights based on cluster congestion (higher congestion = slightly more priority).
        """
        if not cluster_parameters:
            return None
            
        # Calculate weights based on congestion
        # Note: We use softmax or normalized inverse-congestion or direct congestion
        # Here we use normalized congestion magnitude
        total_congestion = sum(self.cluster_congestion.values())
        if total_congestion < 1e-9:
            norm_weights = {cid: 1.0 / len(self.cluster_ids) for cid in self.cluster_ids}
        else:
            norm_weights = {cid: self.cluster_congestion[cid] / total_congestion for cid in self.cluster_ids}
        
        # Initialize global weights
        global_weights = copy.deepcopy(cluster_parameters[0])
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])
            
        # Weighted average
        for i, (cid, params) in enumerate(zip(self.cluster_ids, cluster_parameters)):
            weight = norm_weights[cid]
            for key in params.keys():
                global_weights[key] += weight * params[key]
                
        self.global_weights = global_weights
        return global_weights
        
    async def handle_update_topic(self, message: Dict):
        """Callback for PUB-SUB topic: handles incoming model from a cluster/node."""
        cid = message.get("id")
        weights = message.get("weights")
        congestion = message.get("congestion", 1.0)
        
        print(f"[Server] Received update from {cid} via PUB-SUB")
        self.update_cluster_metrics(cid, congestion)
        
        # In a real async server, we might wait for N nodes before aggregating
        # For the dummy server, we'll just store and can be triggered to aggregate
        return weights

    async def broadcast_global_weights(self, broker, topic: str):
        """Publish global weights to the broker."""
        if self.global_weights:
            print(f"[Server] Broadcasting global weights to {topic}")
            await broker.publish(topic, self.global_weights)

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return self.global_weights
        
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        self.global_weights = copy.deepcopy(weights)
