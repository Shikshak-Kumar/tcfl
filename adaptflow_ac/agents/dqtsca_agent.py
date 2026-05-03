import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from models.dqtsca_model import DQTSCAModel

class DQTSCAAgent:
    """
    DQTSCA Agent (CNN-based DQN).
    """
    def __init__(self, node_id, grid_size=20, action_dim=4, lr=0.00025, gamma=0.95, buffer_size=500000):
        self.node_id = node_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0 # Will decay as 1 - n/N
        self.grid_size = grid_size
        
        self.model = DQTSCAModel(grid_size, phase_dim=4, action_dim=action_dim)
        self.target_model = DQTSCAModel(grid_size, phase_dim=4, action_dim=action_dim)
        self.update_target_network()
        
        # Section 11: RMSprop optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Section 11: 500k buffer
        self.memory = deque(maxlen=buffer_size)
        self.train_step = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, occ, speed, phase, evaluation=False):
        """ε-greedy selection."""
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        occ_t = torch.FloatTensor(occ).unsqueeze(0).unsqueeze(0)
        speed_t = torch.FloatTensor(speed).unsqueeze(0).unsqueeze(0)
        phase_t = torch.FloatTensor(phase).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(occ_t, speed_t, phase_t)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_next, done):
        """
        s: (occ, speed, phase)
        """
        self.memory.append((s, a, r, s_next, done))

    def train(self, batch_size=16, total_steps=100000):
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors
        occ = torch.FloatTensor(np.array([b[0][0] for b in batch])).unsqueeze(1)
        speed = torch.FloatTensor(np.array([b[0][1] for b in batch])).unsqueeze(1)
        phase = torch.FloatTensor(np.array([b[0][2] for b in batch]))
        
        actions = torch.LongTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))
        
        occ_next = torch.FloatTensor(np.array([b[3][0] for b in batch])).unsqueeze(1)
        speed_next = torch.FloatTensor(np.array([b[3][1] for b in batch])).unsqueeze(1)
        phase_next = torch.FloatTensor(np.array([b[3][2] for b in batch]))
        
        dones = torch.FloatTensor(np.array([b[4] for b in batch]))

        # Target y = r + γ max Q(s', a')
        with torch.no_grad():
            next_q = self.target_model(occ_next, speed_next, phase_next)
            max_next_q = torch.max(next_q, dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Current Q
        current_q_all = self.model(occ, speed, phase)
        current_q = current_q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = self.criterion(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Exploration decay: ε = 1 - n/N
        self.train_step += 1
        self.epsilon = max(0.01, 1.0 - (self.train_step / total_steps))
            
        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.update_target_network()
