import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalEncoder, self).__init__()
        # input_dim is state size per step (queue, wait, phase)
        # Sequence length is 4 (temporal stacking)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch_size, num_nodes, seq_len, input_dim)
        batch_size, num_nodes, seq_len, input_dim = x.shape
        
        # Flatten batch and nodes to process sequence through GRU
        x = x.view(batch_size * num_nodes, seq_len, input_dim)
        
        _, h_n = self.gru(x)
        # h_n shape: (1, batch_size * num_nodes, hidden_dim)
        
        # Reshape back: (batch_size, num_nodes, hidden_dim)
        out = h_n.view(batch_size, num_nodes, -1)
        return out
