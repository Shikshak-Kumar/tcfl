import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """Orthogonal initialization as specified in Chu et al. (2019)."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTMCell):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MultiAgentACNetwork(nn.Module):
    """
    Base Network for MA2C Actor and Critic.
    Chu et al. (2019) specifies separate networks for Actor and Critic.
    """
    def __init__(self, wave_dim, wait_dim, fp_dim, hidden_dim=64):
        super(MultiAgentACNetwork, self).__init__()
        
        # Encoders
        self.wave_encoder = nn.Linear(wave_dim, 128)
        self.wait_encoder = nn.Linear(wait_dim, 32)
        self.fp_encoder = nn.Linear(fp_dim, 64)
        
        # Concatenated input dim: 128 + 32 + 64 = 224
        self.lstm = nn.LSTM(224, hidden_dim, batch_first=True)
        
        self.apply(init_weights)

    def forward(self, wave, wait, fp, hidden=None):
        # Encoders
        x_wave = F.relu(self.wave_encoder(wave))
        x_wait = F.relu(self.wait_encoder(wait))
        x_fp = F.relu(self.fp_encoder(fp))
        
        # Concatenate
        x = torch.cat([x_wave, x_wait, x_fp], dim=-1)
        
        # LSTM
        # x: (batch, seq_len, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add seq_len dim
            
        output, hidden = self.lstm(x, hidden)
        return output.squeeze(1), hidden

class MultiAgentACActor(MultiAgentACNetwork):
    def __init__(self, wave_dim, wait_dim, fp_dim, action_dim, hidden_dim=64):
        super(MultiAgentACActor, self).__init__(wave_dim, wait_dim, fp_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, wave, wait, fp, hidden=None):
        feat, hidden = super().forward(wave, wait, fp, hidden)
        logits = self.head(feat)
        return F.softmax(logits, dim=-1), hidden

class MultiAgentACCritic(MultiAgentACNetwork):
    def __init__(self, wave_dim, wait_dim, fp_dim, hidden_dim=64):
        super(MultiAgentACCritic, self).__init__(wave_dim, wait_dim, fp_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, wave, wait, fp, hidden=None):
        feat, hidden = super().forward(wave, wait, fp, hidden)
        value = self.head(feat)
        return value, hidden
