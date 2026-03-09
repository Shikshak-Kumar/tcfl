import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Adapted for CoLight (Wei et al., 2019)
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [batch_size, num_nodes, in_features]
        Wh = torch.matmul(h, self.W) # [batch_size, num_nodes, out_features]
        num_nodes = Wh.size(1)

        # Attention mechanism
        # Wh1: [batch_size, num_nodes, 1, out_features]
        # Wh2: [batch_size, 1, num_nodes, out_features]
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        Wh2 = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # a_input: [batch_size, num_nodes, num_nodes, 2*out_features]
        a_input = torch.cat([Wh1, Wh2], dim=-1)
        
        # e: [batch_size, num_nodes, num_nodes]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        # adj: [batch_size, num_nodes, num_nodes]
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # h_prime: [batch_size, num_nodes, out_features]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class CoLightEncoder(nn.Module):
    """
    CoLight Encoder using Multi-head Graph Attention.
    """
    def __init__(self, nfeat, nhid, nheads, dropout=0.1, alpha=0.2):
        super(CoLightEncoder, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, nfeat]
        # adj: [batch_size, num_nodes, num_nodes]
        
        # If no adj provided (single node), use identity
        if adj is None:
            batch_size, num_nodes, _ = x.size()
            adj = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1).to(x.device)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x # [batch_size, num_nodes, nhid]
