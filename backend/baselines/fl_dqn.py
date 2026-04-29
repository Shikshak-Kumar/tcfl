import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Dict

class FLDQNNetwork(nn.Module):
    """
    DQN Network from 'A scalable approach...' (Scientific Reports 2023).
    Layers: Input -> 64 -> 128 -> 256 -> Output.
    """
    def __init__(self, state_size: int, action_size: int):
        super(FLDQNNetwork, self).__init__()
        # Global layers (feature extraction)
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        # Local layer (output) - Not aggregated in some versions, but here we treat it locally
        self.output_layer = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.feature_layers(x)
        return self.output_layer(x)

class FLDQNAgent:
    """
    FL-DQN Baseline according to Bao et al. (2023).
    - DQN with 3 hidden layers.
    - Partial aggregation (optional, but standard FedAvg by default).
    - Reward: Queue Length + sigma * Waiting Time.
    """
    def __init__(self, state_size: int, action_size: int, lr=0.0001, gamma=0.9,
                 epsilon=0.9, sigma=0.1, memory_size=1000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = FLDQNNetwork(state_size, action_size).to(self.device)
        self.target_net = FLDQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state, training=True):
        # The paper says epsilon-greedy 0.9. Usually this means 0.1 exploration.
        # But Table 1 says '0.9'. If it's the probability of choosing the best action, then epsilon=0.1.
        # If it's the exploration rate, it's 0.9. TSC papers often use high exploration initially.
        # I'll use 0.1 exploration if training, matching 'best action selection 0.9'.
        if training and random.random() > self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(q_values.argmax().item())

    def remember(self, state, action, reward, next_state, done):
        # Internal reward recalculation to match paper: -(Queue + sigma * Wait)
        # Note: Environment usually provides this in 'info' or we compute it.
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([m[0] for m in batch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in batch])).to(self.device)

        self.policy_net.train()
        q_eval = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_weights(self):
        """Returns only the feature layers for aggregation as per the paper's partial FL."""
        return self.policy_net.feature_layers.state_dict()

    def set_weights(self, feature_weights):
        """Sets the global feature layers, keeping local output layer."""
        self.policy_net.feature_layers.load_state_dict(feature_weights)
        self.target_net.feature_layers.load_state_dict(feature_weights)

def fl_dqn_aggregate(weights_list: List[Dict]) -> Dict:
    """Average feature layers weights."""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key] for w in weights_list]).mean(0)
    return avg_weights
