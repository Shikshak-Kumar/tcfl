import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal import TemporalEncoder
from .gat import GATEncoder

class GATLSTMNetwork(nn.Module):
    """
    Base network structure: GAT layers followed by LSTM.
    Includes Phase Embedding and support for returning attention weights.
    """
    def __init__(self, input_dim, hidden_dim=64, heads=2, num_phases=4):
        super(GATLSTMNetwork, self).__init__()
        self.phase_embedding = nn.Embedding(num_phases, 16)
        
        # New input dim for GAT = 2 + 16 = 18
        self.gat = GATEncoder(18, hidden_dim, hidden_dim, heads=heads)
        self.lstm = TemporalEncoder(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        self.log_step = 0

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, seq_len, input_dim)
        adj: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, seq_len, input_dim = x.shape
        
        qw = x[:, :, :, :2]
        phase_idx = x[:, :, :, 2:].argmax(dim=-1) # (B, N, T)
        
        p_emb = self.phase_embedding(phase_idx) # (B, N, T, 16)
        x_emb = torch.cat([qw, p_emb], dim=-1) # (B, N, T, 18)
        
        # 1. Process each time step through GAT
        x_reshaped = x_emb.transpose(1, 2).reshape(batch_size * seq_len, num_nodes, 18)
        adj_repeated = adj.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape(batch_size * seq_len, num_nodes, num_nodes)
        
        g_out, alpha = self.gat(x_reshaped, adj_repeated) # (B*T, N, H), alpha: (B*T, N, N, H)
        
        # Log embedding variance to check for over-smoothing (Priority 2)
        if self.training:
            self.log_step += 1
            if self.log_step % 100 == 0:
                print(f"GAT embedding variance: {g_out.var(dim=0).mean().item():.4f}")

        # 2. Reshape back to (B, N, T, H)
        g_out = g_out.reshape(batch_size, seq_len, num_nodes, -1).transpose(1, 2)
        
        # 3. Temporal encoding: (B, N, H)
        t_out = self.lstm(g_out)
        
        # 4. FC layer
        out = F.relu(self.fc(t_out))
        
        # Average alpha across time steps for spatial discount (B, N, N, H)
        alpha_avg = alpha.reshape(batch_size, seq_len, num_nodes, num_nodes, -1).mean(dim=1)
        
        return out, alpha_avg

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = GATLSTMNetwork(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x, adj):
        feat, alpha = self.network(x, adj)
        logits = self.head(feat)
        return F.softmax(logits, dim=-1), alpha

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = GATLSTMNetwork(input_dim, hidden_dim)
        # Point 8: Graph-level global critic
        self.head = nn.Linear(hidden_dim, 1)
        self.global_head = nn.Linear(hidden_dim, 1)
        
        nn.init.orthogonal_(self.head.weight)
        nn.init.orthogonal_(self.global_head.weight)

    def forward(self, x, adj):
        feat, _ = self.network(x, adj)
        # Local values
        values = self.head(feat).squeeze(-1) # (B, N)
        
        # Global value: mean pooling across all nodes
        global_state = feat.mean(dim=1) # (B, H)
        global_value = self.global_head(global_state).squeeze(-1) # (B,)
        
        return values, global_value

class AdaptFlowAC(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(AdaptFlowAC, self).__init__()
        self.actor = Actor(input_dim, action_dim, hidden_dim)
        self.critic = Critic(input_dim, hidden_dim)

    def forward(self, x, adj):
        probs, alpha = self.actor(x, adj)
        values, global_value = self.critic(x, adj)
        return probs, values, global_value, alpha
