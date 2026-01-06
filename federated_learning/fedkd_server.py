import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from federated_learning.fl_server import TrafficFLServer

class TrafficFedKDServer(TrafficFLServer):
    """
    Extends standard FL server to support Federated Knowledge Distillation.
    Instead of parameter averaging (FedAvg), it averages soft predictions (logits).
    """
    
    def __init__(self, num_rounds: int = 15, min_clients: int = 2, 
                 proxy_set_size: int = 100):
        super().__init__(num_rounds=num_rounds, min_clients=min_clients)
        self.proxy_set_size = proxy_set_size
        self.proxy_states: Optional[np.ndarray] = None
        self.consensus_logits: Optional[np.ndarray] = None
        
    def initialize_proxy_dataset(self, state_size: int):
        """Initializes an empty proxy dataset pending real data from clients."""
        self.proxy_states = np.array([])
        print("Server initialized (waiting for real traffic states from clients).")

    def update_proxy_dataset(self, new_states_list: List[np.ndarray]):
        """Collect and accumulate real traffic states from clients."""
        if not new_states_list:
            return
            
        combined = []
        for s in new_states_list:
            if s.size > 0:
                combined.append(s)
        
        if not combined:
            return
            
        new_batch = np.concatenate(combined, axis=0)
        
        if self.proxy_states.size == 0:
            self.proxy_states = new_batch
        else:
            self.proxy_states = np.concatenate([self.proxy_states, new_batch], axis=0)
            
        # Keep it capped at proxy_set_size
        if len(self.proxy_states) > self.proxy_set_size:
            indices = np.random.choice(len(self.proxy_states), self.proxy_set_size, replace=False)
            self.proxy_states = self.proxy_states[indices]
            
        print(f"Server proxy dataset updated. Total real states collected: {len(self.proxy_states)}")

    def aggregate_logits(self, results: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Aggregate logits from all clients using weighted average.
        results: List of arrays, each of shape (proxy_set_size, action_size)
        """
        if not results:
            return None
            
        if weights is None:
            weights = [1.0 / len(results)] * len(results)
            
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Aggregate
        aggregated = np.zeros_like(results[0])
        for logit_set, w in zip(results, weights):
            aggregated += logit_set * w
            
        self.consensus_logits = aggregated
        return aggregated

    def save_fedkd_metadata(self):
        """Save consensus information and proxy dataset for auditing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"fedkd_metadata_{timestamp}.json")
        
        # We don't save the full arrays usually, just some stats
        metadata = {
            "num_rounds": self.num_rounds,
            "proxy_set_size": self.proxy_set_size,
            "consensus_logit_mean": float(np.mean(self.consensus_logits)) if self.consensus_logits is not None else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"FedKD metadata saved to {filepath}")
