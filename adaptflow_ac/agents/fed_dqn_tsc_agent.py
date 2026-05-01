import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from models.fed_dqn_tsc_model import FedDQNTscModel

class FedDQNTscAgent:
    """
    DQN Agent for Scientific Reports (2023) paper.
    Supports selective weight updates for Federated Learning.
    """
    def __init__(self, node_id, input_dim, action_dim, lr=0.0001, gamma=0.9, epsilon=0.1, capacity=1000):
        self.node_id = node_id
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon # 0.1 means 90% greedy
        
        self.model = FedDQNTscModel(input_dim, action_dim)
        self.target_model = FedDQNTscModel(input_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)

    def get_action(self, state, evaluation=False):
        # state: (N, M, K)
        if not evaluation and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0) # (1, N, M, K)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor([b[4] for b in batch])

        self.model.train()
        current_q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_global_weights(self):
        return self.model.get_global_weights()

    def set_global_weights(self, weights):
        self.model.load_global_weights(weights)
        self.target_model.load_global_weights(weights)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        
    def freeze_global(self):
        self.model.freeze_global_layers()
        # Re-init optimizer to only train local params
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
