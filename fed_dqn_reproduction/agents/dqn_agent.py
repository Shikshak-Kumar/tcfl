import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from fed_dqn_reproduction.models.dqn import ScientificDQN

class DQNAgent:
    def __init__(self, input_dim, action_dim, lr=0.0001, gamma=0.9, epsilon=0.9, buffer_size=1000):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.model = ScientificDQN(input_dim, action_dim)
        self.target_model = ScientificDQN(input_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def get_action(self, state):
        # state: (K, N, M) -> flattened
        state_f = state.flatten()
        if random.random() < self.epsilon:
            state_t = torch.FloatTensor(state_f).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_t)
            return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.action_dim - 1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])
        
        # Q(s, a)
        current_q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # max Q(s', a') from target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_global_weights(self):
        return self.model.get_global_weights()

    def set_global_weights(self, weights):
        self.model.set_global_weights(weights)

    def freeze_global(self, freeze=True):
        self.model.freeze_global_layers(freeze)
        # Re-initialize optimizer if freezing changes
        lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
