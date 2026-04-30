import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=2, dropout=0.1):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes, in_features)
        # adj shape: (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, _ = x.shape
        
        h = self.W(x).view(batch_size, num_nodes, self.heads, self.out_features)
        
        # Self-attention mechanism
        # h: (B, N, H, F_out)
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1) # (B, N, N, H, F_out)
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1, 1) # (B, N, N, H, F_out)
        
        a_input = torch.cat([h_i, h_j], dim=-1) # (B, N, N, H, 2*F_out)
        
        e = self.leaky_relu((a_input * self.a).sum(dim=-1)) # (B, N, N, H)
        
        # Masking
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(-1) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)
        
        h_prime = torch.einsum('bnnh,bnhf->bnhf', attention, h) # (B, N, H, F_out)
        
        # Concatenate or average heads. Request didn't specify, usually concat for hidden, average for output.
        # Let's concatenate.
        return h_prime.reshape(batch_size, num_nodes, -1)

class GATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads=2, dropout=0.1):
        super(GATEncoder, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_features, heads, dropout)
        self.gat2 = GATLayer(hidden_features * heads, out_features, 1, dropout) # Final layer usually 1 head

    def forward(self, x, adj):
        # adj normalization
        # D^-1/2 * A * D^-1/2 is for GCN, GAT uses attention.
        # But user asked to "Normalize adjacency matrix".
        # Let's assume they mean adding self-loops.
        adj = adj + torch.eye(adj.size(1)).to(adj.device).unsqueeze(0)
        
        x = F.elu(self.gat1(x, adj))
        x = self.gat2(x, adj)
        return x
