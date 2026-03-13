import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Dict, Optional, Tuple

from federated_learning.gat_module import SpatioTemporalEncoder

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
        # adj: [batch_size, num_nodes, num_nodes]
        
        # 1. Spatio-Temporal Encoding
        h = self.encoder(x_seq, adj) # [batch_size, num_nodes, gat_hidden]
        
        # 2. Focus on the focal node (index 0)
        h_focal = h[:, 0, :] # [batch_size, gat_hidden]
        
        # 3. Dense layers for Q-values
        x = torch.relu(self.fc1(h_focal))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class FedFlowAgent:
    """
    Spatio-Temporal Graph-Aware Double DQN Agent for FedFlow-TSC.
    """
    def __init__(self, state_size: int, action_size: int, 
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 64,
                 device: str = "cpu", time_steps: int = 4):
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
        self.policy_net = FedFlowDQN(state_size, state_size, action_size, time_steps=time_steps).to(device)
        self.target_net = FedFlowDQN(state_size, state_size, action_size, time_steps=time_steps).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=5000)
        
        # History for Spatio-Temporal sequences
        self.history = deque(maxlen=time_steps)

    def _get_sequence(self, current_state_graph: np.ndarray) -> np.ndarray:
        """Returns a sequence of length T by padding if history is short."""
        curr_hist = list(self.history)
        if len(curr_hist) < self.time_steps:
            # Pad with current state if history is not full
            padding = [current_state_graph] * (self.time_steps - len(curr_hist))
            sequence = padding + curr_hist
        else:
            sequence = curr_hist
        return np.array(sequence) # [time_steps, num_nodes, state_size]

    def get_action(self, state_graph: np.ndarray, adj: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy with Spatio-Temporal sequence.
        """
        # Update history
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
        # We store the sequences themselves in memory for easier sampling
        self.memory.append((state_seq, adj, action, reward, next_state_seq, next_adj, done))
        
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Batch shapes: [batch, time_steps, nodes, feat]
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        adjs = torch.FloatTensor(np.array([m[1] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[2] for m in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(self.device)
        next_adjs = torch.FloatTensor(np.array([m[5] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[6] for m in minibatch])).to(self.device)
        
        self.policy_net.train()
        
        # Double DQN: Current Q values
        q_eval = self.policy_net(states, adjs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Target Q values
        with torch.no_grad():
            # Action selected by policy_net
            next_actions = self.policy_net(next_states, next_adjs).argmax(1).unsqueeze(1)
            # Q-value from target_net
            q_next = self.target_net(next_states, next_adjs).gather(1, next_actions).squeeze(1)
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
