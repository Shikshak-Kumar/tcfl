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
class FedFlowDQN(nn.Module):
    """
    Combined Spatio-Temporal GAT + DQN Network for FedFlow.
    """
    def __init__(self, state_size: int, action_size: int,
                 gat_heads: int = 4, gat_hidden: int = 32, time_steps: int = 4):
        super(FedFlowDQN, self).__init__()

        # Spatio-Temporal Encoder
        self.encoder = SpatioTemporalEncoder(
            nfeat=state_size, nhid=gat_hidden, nheads=gat_heads, time_steps=time_steps
        )

        # Q-Network Heads
        self.fc1 = nn.Linear(gat_hidden, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x_seq, adj):
        # x_seq: [batch_size, time_steps, num_nodes, state_size]
        # adj:   [batch_size, num_nodes, num_nodes]

        # 1. Spatio-Temporal Encoding
        h = self.encoder(x_seq, adj)        # [batch, num_nodes, gat_hidden]

        # 2. Focus on focal node (index 0)
        h_focal = h[:, 0, :]               # [batch, gat_hidden]

        # 3. Dense layers for Q-values
        x = torch.relu(self.fc1(h_focal))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# FedFlowAgent: Standard Spatio-Temporal Double DQN (Industry Baseline)
# ---------------------------------------------------------------------------
class FedFlowAgent:
    """
    Standard Spatio-Temporal Double DQN Agent for FedFlow-TSC.
    Uses uniform experience replay (Baseline).
    """
    def __init__(self, state_size: int, action_size: int,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 64,
                 device: str = "cpu", time_steps: int = 4,
                 memory_size: int = 5000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        self.time_steps = time_steps

        # Networks
        self.policy_net = FedFlowDQN(state_size, action_size,
                                     time_steps=time_steps).to(device)
        self.target_net = FedFlowDQN(state_size, action_size,
                                     time_steps=time_steps).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Standard Experience Replay
        self.memory = deque(maxlen=memory_size)

        # History for Spatio-Temporal sequences
        self.history = deque(maxlen=time_steps)

    # ------------------------------------------------------------------
    def _get_sequence(self, current_state_graph: np.ndarray) -> np.ndarray:
        """Returns a sequence of length T by padding if history is short."""
        curr_hist = list(self.history)
        if len(curr_hist) < self.time_steps:
            padding = [current_state_graph] * (self.time_steps - len(curr_hist))
            sequence = padding + curr_hist
        else:
            sequence = curr_hist
        return np.array(sequence)  # [time_steps, num_nodes, state_size]

    def get_action(self, state_graph: np.ndarray, adj: np.ndarray) -> int:
        """Select action using epsilon-greedy with Spatio-Temporal sequence."""
        self.history.append(state_graph)
        state_seq = self._get_sequence(state_graph)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.policy_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
            q_values = self.policy_net(s, a)
            return int(q_values.argmax().item())

    def remember(self, state_graph, adj, action, reward, next_state_graph, next_adj, done):
        """Store standard transition. Storing graphs to allow replay to build sequences if needed, 
        but usually we store what we need directly."""
        self.memory.append((state_graph, adj, action, reward, next_state_graph, next_adj, done))

    def replay(self) -> float:
        """Standard uniform experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)

        # For the baseline, we assume the trainer is responsible for managing sequences if needed,
        # but here we'll handle the 3D -> 4D expansion if raw graphs were stored.
        states      = torch.FloatTensor(np.array([m[0] for m in batch])).to(self.device)
        adjs        = torch.FloatTensor(np.array([m[1] for m in batch])).to(self.device)
        actions     = torch.LongTensor(np.array([m[2] for m in batch])).to(self.device)
        rewards     = torch.FloatTensor(np.array([m[3] for m in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[4] for m in batch])).to(self.device)
        next_adjs   = torch.FloatTensor(np.array([m[5] for m in batch])).to(self.device)
        dones       = torch.FloatTensor(np.array([m[6] for m in batch])).to(self.device)

        if states.dim() == 3:
            states      = states.unsqueeze(1)       # [batch, 1, nodes, feat]
            next_states = next_states.unsqueeze(1)

        self.policy_net.train()

        # Double DQN
        q_eval = self.policy_net(states, adjs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states, next_adjs).argmax(1).unsqueeze(1)
            q_next = self.target_net(next_states, next_adjs).gather(1, next_actions).squeeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    # ------------------------------------------------------------------
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
