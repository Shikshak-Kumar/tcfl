"""
FedCM-RL Client

Federated Cross-Model Reinforcement Learning client for heterogeneous
traffic signal control agents.

Based on:
- FedMD (Li & Wang 2019) - Model-heterogeneous distillation
- FedDF (Lin et al., NeurIPS 2020) - Ensemble teacher aggregation
- HeteroFL (ICLR 2020) - Heterogeneous architectures
"""

import numpy as np
import torch
import torch.nn.functional as F
import shutil
import os
import json
from typing import Dict, List, Optional

from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
from federated_learning.fl_client import TrafficFLClient


class FedCMClient(TrafficFLClient):
    """
    Federated Cross-Model (FedCM) client for heterogeneous RL agents.
    
    Supports different agent architectures without weight averaging.
    Uses logit-level aggregation and cross-model distillation.
    """
    
    def __init__(self, client_id: str, sumo_config_path: str,
                 state_size: int = 12, action_size: int = 4,
                 gui: bool = False, show_phase_console: bool = False,
                 show_gst_gui: bool = False, 
                 agent_type: str = "DQN",
                 hidden_dims: List[int] = [128, 128, 64],
                 results_dir: str = "results_fedcm",
                 temperature: float = 2.0,
                 lambda_distill: float = 0.5,
                 distill_method: str = "mse"):
        """
        Initialize FedCM client.
        
        Args:
            client_id: Unique client identifier
            sumo_config_path: Path to SUMO configuration
            state_size: State dimension (default: 12)
            action_size: Action dimension (default: 4)
            gui: Use SUMO GUI
            show_phase_console: Show phase console
            show_gst_gui: Show GST GUI
            agent_type: Agent type ("DQN", "PPO", "AC")
            hidden_dims: Network architecture
            results_dir: Results directory
            temperature: Temperature for distillation
            lambda_distill: Distillation loss weight
            distill_method: "mse" or "kl"
        """
        
        # Store FedCM-specific params before parent init
        self.agent_type = agent_type
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        self.distill_method = distill_method
        self.results_dir = results_dir
        
        # Initialize parent class
        super().__init__(client_id, sumo_config_path, state_size, action_size,
                        gui, show_phase_console, show_gst_gui)
        
        # Override agent with custom architecture
        if agent_type == "DQN":
            self.agent = DQNAgent(state_size, action_size, hidden_dims=hidden_dims)
        elif agent_type == "PPO":
            raise NotImplementedError("PPO agent not yet implemented")
        elif agent_type == "AC":
            raise NotImplementedError("Actor-Critic agent not yet implemented")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Distillation history
        self.distill_history = []
        
        # Architecture info for logging
        self.architecture = {
            "type": agent_type,
            "hidden_dims": hidden_dims,
            "state_size": state_size,
            "action_size": action_size
        }
    
    def train(self, round_num: int) -> Dict:
        """
        Train client for one round (local RL training only).
        
        Args:
            round_num: Current round number
        
        Returns:
            metrics: Training metrics
        """
        metrics = self._train_agent(self.episodes_per_round)
        
        # Save training results
        os.makedirs(self.results_dir, exist_ok=True)
        save_path = os.path.join(self.results_dir, f"{self.client_id}_round_{round_num}_train.json")
        
        # Convert numpy types for JSON serialization
        metrics_serializable = self._convert_to_json_serializable(metrics)
        
        with open(save_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        
        return metrics
    
    def get_logits(self, states: np.ndarray) -> np.ndarray:
        """
        Get model logits (Q-values or policy logits) for given states.
        
        For DQN: Returns Q-values for all actions.
        For PPO/AC: Returns policy logits.
        
        Args:
            states: State array (N, state_size)
        
        Returns:
            logits: Model outputs (N, action_size)
        """
        if self.agent_type == "DQN":
            return self.agent.get_logits(states)
        elif self.agent_type == "PPO":
            raise NotImplementedError("PPO logit extraction not implemented")
        elif self.agent_type == "AC":
            raise NotImplementedError("AC logit extraction not implemented")
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def distill(self, states: np.ndarray, teacher_logits: np.ndarray,
                epochs: int = 5) -> Dict:
        """
        Perform cross-model knowledge distillation.
        
        Updates local model to match ensemble teacher predictions.
        
        Args:
            states: Proxy states (N, state_size)
            teacher_logits: Global ensemble teacher logits (N, action_size)
            epochs: Number of distillation epochs
        
        Returns:
            metrics: Distillation metrics
        """
        device = self.agent.device
        
        # Convert to tensors
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        teacher_tensor = torch.as_tensor(teacher_logits, dtype=torch.float32, device=device)
        
        distill_losses = []
        
        # Set model to train mode
        self.agent.policy_net.train()
        
        for epoch in range(epochs):
            # Get local logits
            local_logits = self.agent.policy_net(states_tensor)
            
            # Compute distillation loss
            if self.distill_method == "mse":
                # MSE loss (FedMD style)
                loss = F.mse_loss(local_logits, teacher_tensor)
            
            elif self.distill_method == "kl":
                # KL divergence with temperature scaling
                local_soft = F.log_softmax(local_logits / self.temperature, dim=1)
                teacher_soft = F.softmax(teacher_tensor / self.temperature, dim=1)
                loss = F.kl_div(local_soft, teacher_soft, reduction='batchmean')
                loss = loss * (self.temperature ** 2)  # Scale by T^2
            
            else:
                raise ValueError(f"Unknown distillation method: {self.distill_method}")
            
            # Backward pass
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()
            
            distill_losses.append(loss.item())
        
        avg_distill_loss = np.mean(distill_losses)
        
        metrics = {
            "distill_loss": avg_distill_loss,
            "epochs": epochs,
            "method": self.distill_method,
            "temperature": self.temperature,
            "lambda": self.lambda_distill
        }
        
        self.distill_history.append(metrics)
        
        return metrics
    
    def evaluate(self, round_num: int) -> Dict:
        """
        Evaluate client performance.
        
        Args:
            round_num: Current round number
        
        Returns:
            metrics: Evaluation metrics
        """
        return self._evaluate_agent()
    
    def get_architecture_info(self) -> Dict:
        """
        Get client architecture information.
        
        Returns:
            architecture: Architecture details
        """
        return self.architecture
    
    def get_communication_cost(self, n_proxy_states: int) -> Dict:
        """
        Calculate communication cost for FedCM.
        
        Args:
            n_proxy_states: Number of proxy states
        
        Returns:
            cost: Communication cost in bytes
        """
        # FedCM: Send logits (N_states × N_actions × 4 bytes)
        logits_size = n_proxy_states * self.architecture["action_size"] * 4
        
        # FedAvg: Send all model weights
        model_params = sum(p.numel() for p in self.agent.policy_net.parameters())
        weights_size = model_params * 4
        
        return {
            "fedcm_bytes": logits_size,
            "fedcm_kb": logits_size / 1024,
            "fedavg_bytes": weights_size,
            "fedavg_kb": weights_size / 1024,
            "reduction_percent": (1 - logits_size / weights_size) * 100
        }
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
