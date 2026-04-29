import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class FedLightACNetwork(nn.Module):
    """
    Actor-Critic Network from FedLight (DAC 2021).
    Layers: 64, 32.
    """
    def __init__(self, state_size: int, action_size: int):
        super(FedLightACNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.actor = nn.Linear(32, action_size)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

class FedLightAgent:
    """
    FedLight Baseline according to Ye et al. (2021).
    - Actor-Critic (A2C).
    - Reward: Negative Intersection Pressure.
    - Cloud-based best parameter dispatching.
    """
    def __init__(self, state_size: int, action_size: int, lr_actor=0.0001, lr_critic=0.0002, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = FedLightACNetwork(state_size, action_size).to(self.device)
        # Using a single optimizer for both actor and critic parts but we could separate them
        # The paper says lr_actor and lr_critic are different. 
        # I'll use separate params or a single optimizer with param groups.
        self.optimizer = optim.Adam([
            {'params': self.model.fc1.parameters()},
            {'params': self.model.fc2.parameters()},
            {'params': self.model.actor.parameters(), 'lr': lr_actor},
            {'params': self.model.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor) # Default lr is actor's
        
        self.rollout = [] # Trajectory memory (max 5 slots as per paper)
        self.batch_size = 5

    def get_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state)
            probs = F.softmax(logits, dim=-1)
            
        if training:
            m = torch.distributions.Categorical(probs)
            return m.sample().item()
        else:
            return probs.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.rollout.append((state, action, reward, next_state, done))

    def compute_gradients(self) -> List[torch.Tensor]:
        """Compute gradients locally every batch_size steps."""
        if len(self.rollout) < self.batch_size:
            return []

        states = torch.FloatTensor(np.array([m[0] for m in self.rollout])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in self.rollout])).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in self.rollout])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in self.rollout])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in self.rollout])).to(self.device)

        self.model.train()
        logits, values = self.model(states)
        _, next_values = self.model(next_states)
        
        values = values.squeeze()
        next_values = next_values.squeeze().detach()
        
        # TD Target: L(Q) = 0.5 * (R + gamma * Q_next - Q)^2
        targets = rewards + (1 - dones) * self.gamma * next_values
        advantages = targets - values.detach()
        
        # Actor loss: - Advantage * log_prob
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values, targets)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grads = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) 
                 for param in self.model.parameters()]
        
        self.rollout = [] # Reset trajectory buffer
        return grads

    def apply_gradients(self, avg_grads: List[torch.Tensor]):
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), avg_grads):
            param.grad = grad.to(self.device)
        self.optimizer.step()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

def fedlight_aggregate_grads(grads_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Average gradients across nodes."""
    avg_grads = []
    for i in range(len(grads_list[0])):
        layer_grads = torch.stack([g[i] for g in grads_list])
        avg_grads.append(layer_grads.mean(0))
    return avg_grads
