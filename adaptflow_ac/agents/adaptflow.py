import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from models.actor_critic import AdaptFlowAC
from utils.per import PERBuffer

class AdaptFlowAgent:
    """
    AdaptFlow Agent: Actor-Critic + PER + GAT integration.
    """
    def __init__(self, node_id, state_dim, action_dim, actor_lr=5e-4, critic_lr=2.5e-4, gamma=0.99, beta=0.01, capacity=50000):
        self.node_id = node_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta # Initial entropy coefficient
        
        self.model = AdaptFlowAC(state_dim, action_dim)
        
        # Separate RMSprop optimizers for Actor and Critic
        self.actor_optimizer = optim.RMSprop(self.model.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.RMSprop(self.model.critic.parameters(), lr=critic_lr)
        
        self.memory = PERBuffer(capacity, alpha=0.6)

        # Temporal stacking: deque of states
        self.state_history = deque(maxlen=4)
        self.min_buffer_size = 500   # 1 episode worth — don't waste rounds on warm-up
        self.train_step = 0
        
    def _get_sequence(self, state):
        """
        state: (num_nodes, input_dim)
        Returns stacked state: (num_nodes, 4, input_dim)
        """
        self.state_history.append(state)
        if len(self.state_history) < 4:
            while len(self.state_history) < 4:
                self.state_history.appendleft(state)
        
        # Stack states: (4, num_nodes, input_dim) -> (num_nodes, 4, input_dim)
        stacked_state = np.array(self.state_history).transpose(1, 0, 2)
        return stacked_state

    def get_action(self, state, adj):
        """
        state: (num_nodes, 4, input_dim) if already stacked, or (num_nodes, input_dim)
        adj: (num_nodes, num_nodes)
        """
        if len(state.shape) == 2:
            stacked_state = self._get_sequence(state)
        else:
            stacked_state = state
            
        state_t = torch.FloatTensor(stacked_state).unsqueeze(0) # (1, N, 4, D)
        adj_t = torch.FloatTensor(adj).unsqueeze(0) # (1, N, N)
        
        self.model.actor.eval()
        with torch.no_grad():
            probs, _ = self.model.actor(state_t, adj_t)
            
        probs = probs.squeeze(0).cpu().numpy() # (N, action_dim)
        actions = [np.random.choice(self.action_dim, p=p) for p in probs]
        return actions

    def remember(self, state, adj, actions, all_rewards, next_state, next_adj, done):
        """
        Stores experience in PER. Implements Attention-guided spatial discount.
        all_rewards: list of rewards for all nodes in the local graph.
        """
        self.model.eval()
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            nst = torch.FloatTensor(next_state).unsqueeze(0)
            at = torch.FloatTensor(adj).unsqueeze(0)
            nat = torch.FloatTensor(next_adj).unsqueeze(0)
            
            # Get attention weights for spatial discount
            _, alpha = self.model.actor(st, at)
            # alpha: (1, N, N, H). Take average across heads.
            alpha = alpha.mean(dim=-1).squeeze(0).numpy() # (N, N)
            
            # Spatial discount for the central node (index 0)
            # r̃t,0 = rt,0 + Σj α0j * rt,j
            enriched_reward = all_rewards[0]
            for j in range(1, len(all_rewards)):
                enriched_reward += alpha[0, j] * all_rewards[j]
            
            v, _ = self.model.critic(st, at)
            next_v, _ = self.model.critic(nst, nat)
            
            v = v.squeeze(0).numpy()
            next_v = next_v.squeeze(0).numpy()
            
            # TD Error: r + gamma * V(s') - V(s)
            td_error = enriched_reward + (1 - done) * self.gamma * next_v[0] - v[0]
            priority = np.abs(td_error) + 0.01
            
        self.memory.add(priority, (state, adj, actions, enriched_reward, next_state, next_adj, done))

    def train(self, batch_size=32, beta_per=0.4):
        if self.memory.tree.n_entries < self.min_buffer_size:
            return 0
        
        batch, idxs, is_weights = self.memory.sample(batch_size, beta=beta_per)
        
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        adjs = torch.FloatTensor(np.array([b[1] for b in batch]))
        actions = torch.LongTensor(np.array([b[2] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[3] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[4] for b in batch]))
        next_adjs = torch.FloatTensor(np.array([b[5] for b in batch]))
        dones = torch.FloatTensor(np.array([b[6] for b in batch]))
        is_weights = torch.FloatTensor(is_weights)

        self.model.train()
        
        # 1. Critic Update (Priority 1: 5 gradient steps)
        with torch.no_grad():
            next_values, _ = self.model.critic(next_states, next_adjs)
            next_values = next_values[:, 0]
            targets = rewards + (1 - dones) * self.gamma * next_values
            
        for _ in range(5):
            values, _ = self.model.critic(states, adjs)
            values = values[:, 0]
            td_errors = targets - values
            critic_loss = (is_weights * (td_errors ** 2)).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 40.0)
            self.critic_optimizer.step()
        
        # 2. Actor Update
        probs, _ = self.model.actor(states, adjs)
        probs = probs[:, 0, :] # (B, A)
        _, global_values = self.model.critic(states, adjs)
        
        # Global Critic Baseline: A = R - V_global
        advantages = (rewards - global_values).detach()
        # Priority 1: Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        log_probs_all = torch.log(probs + 1e-10)
        log_probs = log_probs_all.gather(1, actions[:, 0].unsqueeze(-1)).squeeze(-1)
        
        entropy = -(probs * log_probs_all).sum(dim=-1).mean()
        actor_loss = -(log_probs * advantages).mean() - self.beta * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 40.0)
        self.actor_optimizer.step()
        
        # Logging every 100 steps
        self.train_step += 1
        if self.train_step % 100 == 0:
            print(f"Step {self.train_step} - Advantage mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
            print(f"  Critic loss: {critic_loss.item():.4f}, Actor loss: {actor_loss.item():.4f}")
            
        # Priority 3: Verify priorities are changing every 500 steps
        if self.train_step % 500 == 0:
            p_start = self.memory.tree.capacity - 1
            priorities = self.memory.tree.tree[p_start : p_start + self.memory.tree.n_entries]
            print(f"  PER Stats - Mean priority: {priorities.mean():.4f}, Max priority: {priorities.max():.4f}")
        
        # 3. Update PER priorities
        new_priorities = (torch.abs(td_errors.detach()) + 0.01).cpu().numpy()
        for i in range(batch_size):
            self.memory.update(idxs[i], new_priorities[i])
            
        return actor_loss.item() + critic_loss.item()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_encoder_weights(self) -> dict:
        """Return only shared encoder (GAT+LSTM+fc) weights — used for inter-cluster sync."""
        sd = self.model.state_dict()
        return {k: v for k, v in sd.items()
                if "network." in k}  # actor.network.* and critic.network.*

    def set_encoder_weights(self, enc_weights: dict):
        """Overwrite only encoder layers; preserve task-specific heads."""
        sd = self.model.state_dict()
        sd.update(enc_weights)
        self.model.load_state_dict(sd)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
