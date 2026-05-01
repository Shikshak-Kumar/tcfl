import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.multi_agent_ac_model import MultiAgentACActor, MultiAgentACCritic

class MultiAgentACAgent:
    """
    MA2C Agent from Chu et al. (2019).
    Includes LSTM hidden state management and neighbor fingerprints.
    """
    def __init__(self, agent_id, wave_dim, wait_dim, fp_dim, action_dim, 
                 lr_actor=5e-4, lr_critic=2.5e-4, gamma=0.99, beta=0.01):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta # Entropy weight
        
        self.actor = MultiAgentACActor(wave_dim, wait_dim, fp_dim, action_dim)
        self.critic = MultiAgentACCritic(wave_dim, wait_dim, fp_dim)
        
        self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=lr_critic)
        
        # LSTM Hidden States: (h, c)
        self.actor_hidden = None
        self.critic_hidden = None
        
        # Fingerprints: Latest policy simplex of neighbors
        self.fingerprint = np.ones(action_dim) / action_dim # Initial uniform
        
    def reset_hidden(self):
        self.actor_hidden = None
        self.critic_hidden = None

    def get_action(self, wave, wait, neighbors_fp, evaluation=False):
        """
        wave: (M,)
        wait: (M,)
        neighbors_fp: list of policy simplexes from neighbors
        """
        # Concatenate neighbor fingerprints
        if neighbors_fp:
            fp_input = np.concatenate(neighbors_fp)
        else:
            fp_input = np.zeros(0) # Should be handled by model input_dim
            
        wave_t = torch.FloatTensor(wave).unsqueeze(0)
        wait_t = torch.FloatTensor(wait).unsqueeze(0)
        fp_t = torch.FloatTensor(fp_input).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            probs, self.actor_hidden = self.actor(wave_t, wait_t, fp_t, self.actor_hidden)
            
        self.fingerprint = probs.squeeze(0).cpu().numpy()
        
        if evaluation:
            return torch.argmax(probs).item()
        else:
            return torch.multinomial(probs, 1).item()

    def update(self, batch):
        """
        batch: list of (wave, wait, fp, action, spatial_reward, next_wave, next_wait, next_fp, done)
        """
        self.actor.train()
        self.critic.train()
        
        # Unpack batch
        waves = torch.FloatTensor(np.array([b[0] for b in batch]))
        waits = torch.FloatTensor(np.array([b[1] for b in batch]))
        fps = torch.FloatTensor(np.array([b[2] for b in batch]))
        actions = torch.LongTensor([b[3] for b in batch])
        rewards = torch.FloatTensor([b[4] for b in batch])
        next_waves = torch.FloatTensor(np.array([b[5] for b in batch]))
        next_waits = torch.FloatTensor(np.array([b[6] for b in batch]))
        next_fps = torch.FloatTensor(np.array([b[7] for b in batch]))
        dones = torch.FloatTensor([b[8] for b in batch])

        # Critic Loss
        values, _ = self.critic(waves, waits, fps) # No hidden carry for batch update usually, or use full seq
        values = values.squeeze(-1)
        
        with torch.no_grad():
            next_values, _ = self.critic(next_waves, next_waits, next_fps)
            next_values = next_values.squeeze(-1)
            # R̃_t,i = R̂_t,i + γ * V(s')
            target_values = rewards + (1 - dones) * self.gamma * next_values
            
        critic_loss = F.mse_loss(values, target_values)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40)
        self.critic_opt.step()
        
        # Actor Loss
        probs, _ = self.actor(waves, waits, fps)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        
        # Ã_t,i = R̃_t,i - V(s)
        advantages = (target_values - values).detach()
        
        actor_loss = -(log_probs * advantages).mean()
        
        # Entropy Regularization
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        actor_loss -= self.beta * entropy
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_opt.step()
        
        return critic_loss.item(), actor_loss.item()
