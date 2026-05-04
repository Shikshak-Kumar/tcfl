import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.multi_agent_ac_model import MA2CNetwork

class MultiAgentACAgent:
    """
    MA2C Agent (Chu et al. 2019).
    Includes Actor-Critic loss with spatial advantage and neighbor fingerprinting.
    """
    def __init__(self, agent_id, wave_dim=8, wait_dim=8, fp_dim=16, action_dim=4, lr=3e-4, gamma=0.99, beta=0.01):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta # Entropy weight
        
        self.model = MA2CNetwork(wave_dim, wait_dim, fp_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Fingerprint: π_{t-1, neighbors}
        self.fingerprint = np.ones(action_dim) / action_dim
        self.hidden_state = None # (h, c)
        
        # Buffers for batch training
        self.trajectory = []

    def reset_hidden(self):
        self.hidden_state = None

    def get_action(self, wave, wait, neighbor_fps, evaluation=False):
        """
        wave, wait: np arrays
        neighbor_fps: list of np arrays (fingerprints)
        """
        # Convert to tensor and add batch/seq dims: (1, 1, D)
        wave_t = torch.FloatTensor(wave).unsqueeze(0).unsqueeze(0)
        wait_t = torch.FloatTensor(wait).unsqueeze(0).unsqueeze(0)
        
        # Concatenate neighbor fingerprints
        fp_combined = np.concatenate(neighbor_fps) if neighbor_fps else np.zeros(1)
        fp_t = torch.FloatTensor(fp_combined).unsqueeze(0).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            probs, value, next_hidden = self.model(wave_t, wait_t, fp_t, self.hidden_state)
            
        probs = probs.squeeze(0).squeeze(0).numpy()
        if evaluation:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_dim, p=probs)
            
        self.fingerprint = probs # Update my fingerprint for neighbors in next step
        self.hidden_state = next_hidden
        
        return action, probs, value.item()

    def store_transition(self, wave, wait, fp, action, reward, value, prob):
        self.trajectory.append((wave, wait, fp, action, reward, value, prob))

    def train(self, next_value, done):
        """
        Update using trajectories.
        next_value: V(s_{tB})
        """
        if not self.trajectory:
            return 0
            
        # 1. Compute spatially-discounted N-step returns
        returns = []
        R = next_value if not done else 0
        for transition in reversed(self.trajectory):
            R = transition[4] + self.gamma * R # transition[4] is spatial reward r̃
            returns.insert(0, R)
            
        # 2. Prepare tensors
        waves = torch.FloatTensor(np.array([t[0] for t in self.trajectory])).unsqueeze(0)
        waits = torch.FloatTensor(np.array([t[1] for t in self.trajectory])).unsqueeze(0)
        fps = torch.FloatTensor(np.array([t[2] for t in self.trajectory])).unsqueeze(0)
        actions = torch.LongTensor(np.array([t[3] for t in self.trajectory]))
        returns = torch.FloatTensor(np.array(returns))
        values = torch.FloatTensor(np.array([t[5] for t in self.trajectory]))
        
        self.model.train()
        probs, model_values, _ = self.model(waves, waits, fps) # No hidden reset during forward? 
        # Actually for policy gradient on sequence, we should re-pass hidden but for simplicity we use the batch
        
        probs = probs.squeeze(0)
        model_values = model_values.squeeze(0)
        
        # 3. Critic Loss: MSE(R̃ - V)
        critic_loss = 0.5 * F.mse_loss(model_values, returns)
        
        # 4. Actor Loss: log(π) * A + β * entropy
        advantages = (returns - model_values).detach()
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        actor_loss = -(log_probs * advantages).mean() - self.beta * entropy
        
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        self.trajectory = []
        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
