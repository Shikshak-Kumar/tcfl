import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from collections import deque

from federated_learning.gat_module import SpatioTemporalEncoder


# ---------------------------------------------------------------------------
# FedFlowDQN: GAT + Double DQN network
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FedFlowAgent: Standard MLP Double DQN (Industry Baseline)
# ---------------------------------------------------------------------------
class FedFlowAgent:
    """
    Standard MLP Double DQN Agent for FedFlow-TSC baseline.
    Removes GAT and Temporal features to highlight AdaptFlow's novelty.
    """
    def __init__(self, state_size: int, action_size: int,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 64,
                 device: str = "cpu", memory_size: int = 5000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device

        # Networks: Standard MLP
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.target_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state: np.ndarray, adj: Optional[np.ndarray] = None) -> int:
        """Standard epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.policy_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(s)
            return int(q_values.argmax().item())

    def remember(self, state, action, reward, next_state, done, *args):
        """Uniform experience replay storage."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states      = torch.FloatTensor(np.array([m[0] for m in batch])).to(self.device)
        actions     = torch.LongTensor(np.array([m[1] for m in batch])).to(self.device)
        rewards     = torch.FloatTensor(np.array([m[2] for m in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in batch])).to(self.device)
        dones       = torch.FloatTensor(np.array([m[4] for m in batch])).to(self.device)

        self.policy_net.train()
        q_eval = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_weights(self) -> Dict:
        return self.policy_net.state_dict()

    def set_weights(self, weights: Dict):
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)

    def save_model(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_network()
