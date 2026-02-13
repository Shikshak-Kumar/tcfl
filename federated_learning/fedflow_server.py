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
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return self.global_weights
        
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        self.global_weights = copy.deepcopy(weights)
