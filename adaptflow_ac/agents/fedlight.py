import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from models.fedlight_ac import FedLightAC

class FedLightAgent:
    def __init__(self, node_id, state_dim, action_dim, lr=1e-4, gamma=0.99, capacity=10000):
        self.node_id = node_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.model = FedLightAC(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Consistent with AdaptFlow
        self.state_history = [] 
        
        self.memory = []
        self.capacity = capacity

    def get_action(self, state):
        # state: (N, D)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(state_t)
        
        probs = probs.squeeze(0).cpu().numpy()
        actions = [np.random.choice(self.action_dim, p=p) for p in probs]
        return actions

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        return self.train(batch_size)

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])

        self.model.train()
        probs, values = self.model(states)
        _, next_values = self.model(next_states)
        
        # Select local node values
        values = values[:, 0]
        next_values = next_values[:, 0]
        probs = probs[:, 0, :]
        
        targets = rewards + (1 - dones) * self.gamma * next_values.detach()
        advantages = (targets - values).detach()
        
        log_probs_all = torch.log(probs + 1e-10)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, targets)
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
