"""
FedCM-RL Server

Federated Cross-Model Reinforcement Learning server for ensemble
teacher aggregation and proxy dataset management.

Based on:
- FedDF (Lin et al., NeurIPS 2020) - Ensemble distillation
- FedMD (Li & Wang 2019) - Model-heterogeneous federated learning
"""

import numpy as np
from typing import List, Dict, Optional


class FedCMServer:
    """
    Federated Cross-Model (FedCM) server.
    
    Aggregates client logits to form ensemble teacher.
    No weight averaging - only logit-level aggregation.
    """
    
    def __init__(self, state_dim: int = 12, action_dim: int = 4,
                 proxy_dataset_size: int = 1000,
                 weighting_method: str = "uniform"):
        """
        Initialize FedCM server.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            proxy_dataset_size: Size of proxy dataset
            weighting_method: "uniform" or "performance"
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.proxy_dataset_size = proxy_dataset_size
        self.weighting_method = weighting_method
        
        # Proxy dataset (shared states for logit collection)
        self.proxy_states = None
        
        # Aggregation history
        self.aggregation_history = []
        
        # Client performance tracking
        self.client_performance = {}
    
    def construct_proxy_dataset(self, client_states: List[np.ndarray],
                                 method: str = "sample") -> np.ndarray:
        """
        Construct proxy dataset from client state buffers.
        
        Args:
            client_states: List of state arrays from clients
            method: "sample" or "concat"
        
        Returns:
            proxy_states: Proxy dataset (N, state_dim)
        """
        if method == "sample":
            # Sample uniformly from all client states
            all_states = np.concatenate(client_states, axis=0)
            
            # Random sampling
            n_available = len(all_states)
            n_samples = min(self.proxy_dataset_size, n_available)
            
            indices = np.random.choice(n_available, n_samples, replace=False)
            self.proxy_states = all_states[indices]
        
        elif method == "concat":
            # Concatenate and truncate
            all_states = np.concatenate(client_states, axis=0)
            self.proxy_states = all_states[:self.proxy_dataset_size]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Constructed proxy dataset: {self.proxy_states.shape}")
        
        return self.proxy_states
    
    def aggregate_logits(self, client_logits: List[np.ndarray],
                        client_weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Aggregate client logits to form ensemble teacher.
        
        Implements FedDF-style ensemble aggregation.
        
        Args:
            client_logits: List of logit arrays (N, action_dim)
            client_weights: Optional weights for each client
        
        Returns:
            ensemble_logits: Aggregated teacher logits (N, action_dim)
        """
        if len(client_logits) == 0:
            raise ValueError("No client logits provided")
        
        # Stack logits
        stacked_logits = np.stack(client_logits, axis=0)  # (n_clients, N, action_dim)
        
        if client_weights is None:
            # Uniform weighting
            ensemble_logits = np.mean(stacked_logits, axis=0)
        else:
            # Weighted averaging
            client_weights = np.array(client_weights)
            client_weights = client_weights / client_weights.sum()  # Normalize
            
            # Weighted sum
            ensemble_logits = np.sum(
                stacked_logits * client_weights[:, None, None],
                axis=0
            )
        
        return ensemble_logits
    
    def compute_ensemble_teacher(self, client_logits: List[np.ndarray],
                                 client_ids: List[str]) -> np.ndarray:
        """
        Compute ensemble teacher using configured weighting method.
        
        Args:
            client_logits: List of client logit arrays
            client_ids: List of client IDs
        
        Returns:
            teacher_logits: Ensemble teacher logits
        """
        if self.weighting_method == "uniform":
            weights = None
        
        elif self.weighting_method == "performance":
            # Weight by inverse waiting time (better performance = higher weight)
            weights = []
            for client_id in client_ids:
                if client_id in self.client_performance:
                    waiting_time = self.client_performance[client_id]
                    # Lower waiting time = better = higher weight
                    weight = 1.0 / (waiting_time + 1.0)
                else:
                    weight = 1.0
                weights.append(weight)
        
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
        
        teacher_logits = self.aggregate_logits(client_logits, weights)
        
        # Record aggregation
        self.aggregation_history.append({
            "n_clients": len(client_logits),
            "weighting": self.weighting_method,
            "weights": weights
        })
        
        return teacher_logits
    
    def update_client_performance(self, client_id: str, waiting_time: float):
        """
        Update client performance metric.
        
        Args:
            client_id: Client identifier
            waiting_time: Average waiting time (lower is better)
        """
        self.client_performance[client_id] = waiting_time
    
    def get_proxy_states(self) -> np.ndarray:
        """
        Get proxy dataset.
        
        Returns:
            proxy_states: Proxy dataset
        """
        if self.proxy_states is None:
            raise ValueError("Proxy dataset not constructed yet")
        return self.proxy_states
    
    def get_communication_cost(self, n_clients: int) -> Dict:
        """
        Calculate communication cost for FedCM vs FedAvg.
        
        Args:
            n_clients: Number of clients
        
        Returns:
            cost: Communication cost comparison
        """
        # FedCM: Each client sends logits (proxy_size × action_dim × 4 bytes)
        logits_per_client = self.proxy_dataset_size * self.action_dim * 4
        fedcm_total = logits_per_client * n_clients
        
        # FedAvg: Assume average model size (e.g., 128-128-64 DQN)
        # Approximate: (12*128 + 128*128 + 128*64 + 64*4) * 4 bytes
        avg_model_params = (12*128 + 128*128 + 128*64 + 64*4)
        fedavg_per_client = avg_model_params * 4
        fedavg_total = fedavg_per_client * n_clients
        
        return {
            "fedcm_bytes": fedcm_total,
            "fedcm_kb": fedcm_total / 1024,
            "fedavg_bytes": fedavg_total,
            "fedavg_kb": fedavg_total / 1024,
            "reduction_percent": (1 - fedcm_total / fedavg_total) * 100,
            "n_clients": n_clients,
            "proxy_size": self.proxy_dataset_size
        }
    
    def get_policy_divergence(self, client_logits: List[np.ndarray]) -> Dict:
        """
        Measure policy divergence between clients.
        
        Uses KL divergence between client policies.
        
        Args:
            client_logits: List of client logit arrays
        
        Returns:
            divergence: Divergence metrics
        """
        from scipy.special import softmax
        from scipy.stats import entropy
        
        n_clients = len(client_logits)
        
        # Convert logits to probabilities
        client_probs = [softmax(logits, axis=1) for logits in client_logits]
        
        # Compute pairwise KL divergences
        kl_matrix = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    # Average KL divergence over all states
                    kl_divs = [
                        entropy(client_probs[i][k], client_probs[j][k])
                        for k in range(len(client_probs[i]))
                    ]
                    kl_matrix[i, j] = np.mean(kl_divs)
        
        return {
            "mean_kl": np.mean(kl_matrix[kl_matrix > 0]),
            "max_kl": np.max(kl_matrix),
            "kl_matrix": kl_matrix.tolist()
        }
