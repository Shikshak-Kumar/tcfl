import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    Captures temporal patterns using LSTM.
    """
    def __init__(self, input_dim, hidden_dim):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, x):
        """
        x shape: (batch_size, num_nodes, seq_len, input_dim)
        Returns: (batch_size, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, seq_len, input_dim = x.shape
        
        # Flatten batch and nodes: (B*N, T, D)
        x = x.reshape(batch_size * num_nodes, seq_len, input_dim)
        
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (1, B*N, H)
        
        # Reshape back: (B, N, H)
        out = h_n.reshape(batch_size, num_nodes, -1)
        return out
