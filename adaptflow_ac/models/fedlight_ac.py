import torch
import torch.nn as nn
import torch.nn.functional as F

class FedLightAC(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(FedLightAC, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        s_out = self.shared(x)
        probs = F.softmax(self.actor(s_out), dim=-1)
        values = self.critic(s_out).squeeze(-1)
        return probs, values
