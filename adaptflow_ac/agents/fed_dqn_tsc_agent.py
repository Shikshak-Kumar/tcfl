import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from models.fed_dqn_tsc_model import FedDQNTscModel

class FedDQNTscAgent:
    """
    DQN Agent for Federated TSC (Scientific Reports 2023).
    """
    def __init__(self, node_id, state_dim=224, action_dim=4, lr=0.0001, gamma=0.9, buffer_size=1000):
        self.node_id = node_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.9 # Start value from paper
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = FedDQNTscModel(state_dim, action_dim)
        self.target_model = FedDQNTscModel(state_dim, action_dim)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() # Equation (3) in paper
        self.memory = deque(maxlen=buffer_size)

    def update_target_network(self):
        """w⁻ = w"""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, evaluation=False):
        """ε-greedy selection."""
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Store transition in D."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor(np.array([b[4] for b in batch]))

        # Compute target: y = r + γ max Q(s', a'; w⁻)
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Current Q values: Q(s, a; w)
        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Loss: L(w) = E[(y - Q)^2]
        loss = self.criterion(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def get_weights(self):
        """Return global weights w_g."""
        return self.model.get_global_weights()

    def set_weights(self, global_weights):
        """Apply global weights w_G from server."""
        self.model.set_global_weights(global_weights)
        self.update_target_network()

    def start_fine_tuning(self):
        """Freeze global layers for Section 9."""
        self.model.freeze_global_layers()
        self.target_model.freeze_global_layers()
        # Reset optimizer to only train local layers
        local_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(local_params, lr=0.0001)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.update_target_network()
