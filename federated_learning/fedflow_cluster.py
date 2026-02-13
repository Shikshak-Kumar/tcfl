import torch
import numpy as np
import copy
from typing import List, Dict, Optional

class FedFlowCluster:
    """
    Intra-Cluster Aggregator for FedFlow-TSC.
    Performs weighted aggregation within a community or spatial cluster.
    """
    def __init__(self, cluster_id: str, agent_ids: List[str]):
        self.cluster_id = cluster_id
        self.agent_ids = agent_ids
        self.cluster_weights = None
        self.agent_flows = {aid: 1.0 for aid in agent_ids} # Magnitude of traffic flow
        
    def update_flow(self, agent_id: str, flow_magnitude: float):
        """Update the traffic flow magnitude for a specific agent node."""
        if agent_id in self.agent_ids:
            self.agent_flows[agent_id] = flow_magnitude
            
    def aggregate_intra_cluster(self, agent_parameters: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate weights of all agents in the cluster.
        θ_cluster = Σ w_i θ_i
        w_i proportional to traffic flow magnitude.
        """
        if not agent_parameters:
            return None
            
        # Calculate weights based on flow
        total_flow = sum(self.agent_flows.values())
        norm_weights = {aid: self.agent_flows[aid] / total_flow for aid in self.agent_ids}
        
        # Initialize cluster weights with zeros
        cluster_weights = copy.deepcopy(agent_parameters[0])
        for key in cluster_weights.keys():
            cluster_weights[key] = torch.zeros_like(cluster_weights[key])
            
        # Weighted average
        for i, (aid, params) in enumerate(zip(self.agent_ids, agent_parameters)):
            weight = norm_weights[aid]
            for key in params.keys():
                cluster_weights[key] += weight * params[key]
                
        self.cluster_weights = cluster_weights
        return cluster_weights
        
    def get_cluster_weights(self) -> Dict[str, torch.Tensor]:
        return self.cluster_weights
        
    def set_cluster_weights(self, weights: Dict[str, torch.Tensor]):
        self.cluster_weights = copy.deepcopy(weights)
        
    def get_avg_flow(self) -> float:
        """Return average flow in this cluster for Level 2 aggregation."""
        return sum(self.agent_flows.values()) / len(self.agent_flows)
