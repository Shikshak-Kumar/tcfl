import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from federated_learning.gat_module import CoLightEncoder

class CoLightAgent:
    """
    CoLight Baseline according to Wei et al. (2019).
    Uses Graph Attention Networks (GAT) to coordinate between intersections.
    """
    def __init__(self, state_size: int, action_size: int, nhid=32, nheads=4):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = CoLightEncoder(nfeat=state_size, nhid=nhid, nheads=nheads).to(self.device)
        self.q_network = nn.Linear(nhid, action_size).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()), 
            lr=1e-3
        )
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 0.1

    def get_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
            
        state_t = torch.FloatTensor(state).view(1, 1, -1).to(self.device)
        adj = torch.eye(1).unsqueeze(0).to(self.device) # Mock: single node attention
        
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(state_t, adj)
            q_values = self.q_network(h)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in batch:
            m = self.memory[i]
            states.append(m[0])
            actions.append(m[1])
            rewards.append(m[2])
            next_states.append(m[3])
            dones.append(m[4])
            
        states_t = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).view(-1, 1, 1).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)
        
        adj = torch.eye(1).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        
        self.encoder.train()
        h = self.encoder(states_t, adj)
        current_q = self.q_network(h).gather(2, actions_t).squeeze()
        
        with torch.no_grad():
            h_next = self.encoder(next_states_t, adj)
            max_next_q = self.q_network(h_next).max(2)[0].squeeze()
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        return {
            'encoder': self.encoder.state_dict(),
            'q_net': self.q_network.state_dict()
        }

    def set_weights(self, weights):
        self.encoder.load_state_dict(weights['encoder'])
        self.q_network.load_state_dict(weights['q_net'])
