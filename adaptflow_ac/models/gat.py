import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    Improved GAT Layer with multi-head attention and masked attention.
    """
    def __init__(self, in_features, out_features, heads=4, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        nn.init.orthogonal_(self.W.weight)
        nn.init.orthogonal_(self.a)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, in_features)
        adj: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Linear projection: (B, N, H * F_out) -> (B, N, H, F_out)
        h = self.W(x).view(batch_size, num_nodes, self.heads, self.out_features)
        
        # Self-attention mechanism
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu((a_input * self.a).sum(dim=-1))
        
        mask = (adj == 0).unsqueeze(-1)
        e = e.masked_fill(mask, -1e12)
        
        alpha = F.softmax(e, dim=2)
        alpha_dropped = self.dropout_layer(alpha)
        
        h_prime = torch.einsum('bnnh,bnhf->bnhf', alpha_dropped, h)
        
        if self.concat:
            out = h_prime.reshape(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = h_prime.mean(dim=2)
            
        return out, alpha

class GATEncoder(nn.Module):
    """
    Simplified GAT Encoder with 1 layer and residual connection.
    """
    def __init__(self, in_features, hidden_features, out_features, heads=2, dropout=0.1):
        super(GATEncoder, self).__init__()
        # Reduced to 1 layer and 2 heads to prevent over-smoothing
        self.gat1 = GATLayer(in_features, out_features, heads, dropout, concat=False)
        self.residual_proj = nn.Linear(in_features, out_features)
        nn.init.orthogonal_(self.residual_proj.weight)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        h1, alpha = self.gat1(x, adj)
        # Residual Connection with ELU activation
        out = F.elu(h1 + self.residual_proj(x))
        return out, alpha
