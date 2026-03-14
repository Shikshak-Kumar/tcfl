import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Optional, Tuple

from federated_learning.gat_module import SpatioTemporalEncoder


# ---------------------------------------------------------------------------
# SumTree: Efficient priority-based sampling in O(log n)
# ---------------------------------------------------------------------------
class SumTree:
    """
    Binary tree where each leaf stores a transition priority.
    Internal nodes store the sum of their children.
    Enables O(log n) sampling proportional to priority.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity          # number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0                    # pointer to next write position
        self.n_entries = 0

    # ---- Internal helpers ----
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Iterative traversal to always land on a valid leaf node."""
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx          # leaf reached
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    @property
    def total(self) -> float:
        return float(self.tree[0])

    # ---- Public API ----
    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


# ---------------------------------------------------------------------------
# PER Buffer: wraps SumTree and exposes remember / sample / update API
# ---------------------------------------------------------------------------
class PERBuffer:
    """
    Prioritized Experience Replay buffer.
    Transitions with larger |TD-error| are sampled more frequently.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha          # prioritisation exponent (0=uniform, 1=full)
        self.epsilon = epsilon      # small constant to avoid zero priority
        self.max_priority = 1.0    # new transitions start at max priority

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition):
        """Add new transition with maximum current priority."""
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch_size transitions proportional to priority.
        Returns (transitions, indices, IS_weights).
        """
        indices = []
        transitions = []
        priorities = []
        total = self.tree.total
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            # Clamp s to (0, total) to avoid floating-point edge cases
            s = min(random.uniform(lo, hi), total - 1e-8)
            idx, priority, data = self.tree.get(s)
            # Skip None slots (unfilled positions in tree)
            if data is None:
                s = random.uniform(0, total - 1e-8)
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(max(priority, 1e-8))
            transitions.append(data)

        # Importance Sampling weights
        n = self.tree.n_entries
        probs = np.array(priorities, dtype=np.float64) / total
        # Avoid division by zero
        probs = np.clip(probs, 1e-8, None)
        weights = (n * probs) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities after a learning step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


# ---------------------------------------------------------------------------
# FedFlowDQN: GAT + Double DQN network
# ---------------------------------------------------------------------------
class FedFlowDQN(nn.Module):
    """
    Combined Spatio-Temporal GAT + DQN Network for FedFlow.
    """
    def __init__(self, state_size: int, neighbor_size: int, action_size: int,
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
# FedFlowAgent: Spatio-Temporal Graph-Aware Double DQN with PER
# ---------------------------------------------------------------------------
class FedFlowAgent:
    """
    Spatio-Temporal Graph-Aware Double DQN Agent for FedFlow-TSC.
    Now uses Prioritized Experience Replay (PER) for sample-efficient learning.
    """
    def __init__(self, state_size: int, action_size: int,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 64,
                 device: str = "cpu", time_steps: int = 4,
                 # PER hyperparameters
                 per_alpha: float = 0.6,
                 per_beta: float = 0.4,
                 per_beta_increment: float = 0.001,
                 per_epsilon: float = 1e-5,
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

        # PER parameters
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment

        # Networks
        self.policy_net = FedFlowDQN(state_size, state_size, action_size,
                                     time_steps=time_steps).to(device)
        self.target_net = FedFlowDQN(state_size, state_size, action_size,
                                     time_steps=time_steps).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # PER memory replaces plain deque
        self.memory = PERBuffer(capacity=memory_size,
                                alpha=per_alpha,
                                epsilon=per_epsilon)

        # History for Spatio-Temporal sequences
        from collections import deque
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

    def remember(self, state_seq, adj, action, reward, next_state_seq, next_adj, done):
        """Store transition in PER buffer with maximum priority."""
        self.memory.add((state_seq, adj, action, reward, next_state_seq, next_adj, done))

    def replay(self) -> float:
        """
        Sample from PER buffer proportional to priority,
        compute IS-weighted Double DQN loss, backpropagate,
        and update transition priorities with new TD-errors.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Anneal beta toward 1.0
        beta = min(1.0, self.per_beta + self.per_beta_increment)
        self.per_beta = beta

        # --- 1. Sample from PER buffer ---
        transitions, tree_indices, is_weights = self.memory.sample(self.batch_size, beta)

        states      = torch.FloatTensor(np.array([m[0] for m in transitions])).to(self.device)
        adjs        = torch.FloatTensor(np.array([m[1] for m in transitions])).to(self.device)
        actions     = torch.LongTensor(np.array([m[2] for m in transitions])).to(self.device)
        rewards     = torch.FloatTensor(np.array([m[3] for m in transitions])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[4] for m in transitions])).to(self.device)
        next_adjs   = torch.FloatTensor(np.array([m[5] for m in transitions])).to(self.device)
        dones       = torch.FloatTensor(np.array([m[6] for m in transitions])).to(self.device)
        weights_t   = torch.FloatTensor(is_weights).to(self.device)

        # Guard: SpatioTemporalEncoder expects 4D [batch, time, nodes, feat].
        # If states were stored as raw 2D [nodes, feat] per transition, they
        # stack to 3D [batch, nodes, feat] — add a time dimension in that case.
        if states.dim() == 3:
            states      = states.unsqueeze(1)       # [batch, 1, nodes, feat]
            next_states = next_states.unsqueeze(1)

        self.policy_net.train()

        # --- 2. Double DQN: Current Q values ---
        q_eval = self.policy_net(states, adjs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- 3. Double DQN: Target Q values ---
        with torch.no_grad():
            next_actions = self.policy_net(next_states, next_adjs).argmax(1).unsqueeze(1)
            q_next = self.target_net(next_states, next_adjs).gather(1, next_actions).squeeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # --- 4. IS-weighted Smooth L1 loss (element-wise, then weighted mean) ---
        td_errors = q_target - q_eval                       # [batch]
        element_loss = F.smooth_l1_loss(q_eval, q_target, reduction='none')
        loss = (weights_t * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- 5. Update priorities in PER buffer ---
        td_errors_np = td_errors.detach().cpu().numpy()
        self.memory.update_priorities(tree_indices, td_errors_np)

        # --- 6. Epsilon decay ---
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
