import numpy as np
import random
from collections import deque
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dims: List[int] = [128, 128, 64]):
        super().__init__()
        layers = []
        input_dim = state_size
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 1e-3, 
                 device: str | None = None, hidden_dims: List[int] = [128, 128, 64]):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.memory = deque(maxlen=50000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.batch_size = 64
        self.tau = 0.005
        self.train_step = 0
        
        self.reward_clip_min = -1.0
        self.reward_clip_max = 1.0
        
        self.policy_net = DQNNetwork(state_size, action_size, hidden_dims=hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_dims=hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-5
        )
        
    def update_target_model(self):
        """Soft update using Polyak averaging."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values, dim=1).item())
    
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.as_tensor(np.array([e[0] for e in minibatch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array([e[1] for e in minibatch]), dtype=torch.long, device=self.device)
        rewards_raw = np.array([e[2] for e in minibatch])
        next_states = torch.as_tensor(np.array([e[3] for e in minibatch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array([e[4] for e in minibatch]), dtype=torch.float32, device=self.device)

        rewards_clipped = np.clip(rewards_raw, self.reward_clip_min, self.reward_clip_max)
        rewards = torch.as_tensor(rewards_clipped, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.scheduler.step(loss)
        self.train_step += 1
        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())
    
    def get_logits(self, states: np.ndarray) -> np.ndarray:
        """Generation of soft predictions for distillation."""
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            logits = self.policy_net(s)
            return logits.cpu().numpy()
            
    def distill_step(self, states: np.ndarray, teacher_logits: np.ndarray) -> float:
        """Single distillation update via MSE minimization."""
        self.policy_net.train()
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(teacher_logits, dtype=torch.float32, device=self.device)
        
        q_values = self.policy_net(s)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_values, t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save_model(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_model()
    
    def get_weights(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.policy_net.parameters()]
    
    def set_weights(self, weights: List[np.ndarray]):
        with torch.no_grad():
            for param, w in zip(self.policy_net.parameters(), weights):
                param.copy_(torch.as_tensor(w, dtype=param.dtype, device=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_model_summary(self) -> str:
        num_params = sum(p.numel() for p in self.policy_net.parameters())
        return f"DQNNetwork(state={self.state_size}, actions={self.action_size}, params={num_params})"
