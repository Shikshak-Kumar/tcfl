import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal import TemporalEncoder
from .gat import GATEncoder

class AdaptFlowAC(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(AdaptFlowAC, self).__init__()
        
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim)
        self.gat_encoder = GATEncoder(hidden_dim, hidden_dim // 2, hidden_dim)
        
        # Shared Encoder
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic Head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, seq_len, input_dim)
        adj: (batch_size, num_nodes, num_nodes)
        """
        # 1. Temporal Encoding
        t_out = self.temporal_encoder(x) # (B, N, H)
        
        # 2. GAT Encoding
        g_out = self.gat_encoder(t_out, adj) # (B, N, H)
        
        # 3. Shared FC
        s_out = self.shared(g_out) # (B, N, H)
        
        # 4. Heads
        policy_logits = self.actor(s_out) # (B, N, action_dim)
        values = self.critic(s_out).squeeze(-1) # (B, N)
        
        probs = F.softmax(policy_logits, dim=-1)
        
        return probs, values
