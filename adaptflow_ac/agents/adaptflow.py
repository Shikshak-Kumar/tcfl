import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from ..models.actor_critic import AdaptFlowAC
from ..utils.per import PERBuffer

class AdaptFlowAgent:
    def __init__(self, node_id, state_dim, action_dim, lr=1e-4, gamma=0.99, beta=0.01, capacity=10000):
        self.node_id = node_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta # Entropy coefficient
        
        self.model = AdaptFlowAC(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = PERBuffer(capacity)
        
        # Temporal stacking: deque of states
        self.state_history = deque(maxlen=4)
        
    def get_action(self, state, adj):
        """
        state: (num_nodes, input_dim)
        adj: (num_nodes, num_nodes)
        """
        self.state_history.append(state)
        if len(self.state_history) < 4:
            # Pad with current state if not enough history
            while len(self.state_history) < 4:
                self.state_history.appendleft(state)
        
        # Stack states: (4, num_nodes, input_dim) -> (num_nodes, 4, input_dim)
        stacked_state = np.array(self.state_history).transpose(1, 0, 2)
        
        state_t = torch.FloatTensor(stacked_state).unsqueeze(0) # (1, N, 4, D)
        adj_t = torch.FloatTensor(adj).unsqueeze(0) # (1, N, N)
        
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(state_t, adj_t)
            
        probs = probs.squeeze(0).cpu().numpy() # (N, action_dim)
        actions = [np.random.choice(self.action_dim, p=p) for p in probs]
        return actions, stacked_state

    def remember(self, state, adj, action, reward, next_state, next_adj, done):
        """
        Calculates initial priority and stores in memory.
        """
        # state and next_state are already stacked: (N, 4, D)
        self.model.eval()
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            nst = torch.FloatTensor(next_state).unsqueeze(0)
            at = torch.FloatTensor(adj).unsqueeze(0)
            nat = torch.FloatTensor(next_adj).unsqueeze(0)
            
            _, v = self.model(st, at)
            _, next_v = self.model(nst, nat)
            
            v = v.squeeze(0).numpy()
            next_v = next_v.squeeze(0).numpy()
            
            # TD Error: r + gamma * V(s') - V(s)
            td_error = reward + (1 - done) * self.gamma * next_v - v
            # Advantage as per user request: R - V(s). 
            # In actor-critic context, we often use TD error as advantage.
            # User specifically said priority = |Advantage| + TD_error
            # Let's use |td_error| + td_error (or similar) or just stick to their definition.
            # Advantage = reward - v (for single step)
            advantage = reward - v
            priority = np.mean(np.abs(advantage) + np.abs(td_error))
            
        self.memory.add(priority, (state, adj, action, reward, next_state, next_adj, done))

    def train(self, batch_size=32):
        if self.memory.tree.n_entries < batch_size:
            return 0
        
        batch, idxs, is_weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([b[0] for b in batch])
        adjs = torch.FloatTensor([b[1] for b in batch])
        actions = torch.LongTensor([b[2] for b in batch])
        rewards = torch.FloatTensor([b[3] for b in batch])
        next_states = torch.FloatTensor([b[4] for b in batch])
        next_adjs = torch.FloatTensor([b[5] for b in batch])
        dones = torch.FloatTensor([b[6] for b in batch])
        is_weights = torch.FloatTensor(is_weights)

        self.model.train()
        probs, values = self.model(states, adjs) # probs: (B, N, A), values: (B, N)
        _, next_values = self.model(next_states, next_adjs)
        
        # Select local node values
        values = values[:, self.node_id] # (B,)
        next_values = next_values[:, self.node_id] # (B,)
        probs = probs[:, self.node_id, :] # (B, A)
        
        # Target for Critic: R + gamma * V(s')
        targets = rewards + (1 - dones) * self.gamma * next_values.detach()
        
        # Advantage: targets - values
        advantages = (targets - values).detach() # (B,)
        
        # Actor Loss
        # L_actor = -log(pi(a|s)) * Advantage - beta * entropy
        log_probs_all = torch.log(probs + 1e-10) # (B, A)
        log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1) # (B,)
        actor_loss = -(log_probs * advantages).mean()
        
        # Entropy Regularization
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
        actor_loss -= self.beta * entropy
        
        # Critic Loss
        # L_critic = (targets - values)^2
        critic_loss = (is_weights * F.mse_loss(values, targets, reduction='none')).mean()
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Update PER priorities
        new_priorities = (torch.abs(rewards - values.detach()) + torch.abs(targets - values.detach())).cpu().numpy()
        for i in range(batch_size):
            self.memory.update(idxs[i], new_priorities[i])
            
        return total_loss.item()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
