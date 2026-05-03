import torch
import torch.nn as nn
import torch.nn.functional as F

class MA2CNetwork(nn.Module):
    """
    MA2C Network (Chu et al. 2019).
    Combines wave, wait, and neighbor fingerprints into an LSTM core.
    """
    def __init__(self, wave_dim, wait_dim, fp_dim, action_dim, hidden_dim=64):
        super(MA2CNetwork, self).__init__()
        
        # FC layers for feature processing
        self.fc_wave = nn.Linear(wave_dim, hidden_dim // 4)
        self.fc_wait = nn.Linear(wait_dim, hidden_dim // 4)
        self.fc_fp = nn.Linear(fp_dim, hidden_dim // 2)
        
        # Combined LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Actor Head
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic Head
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, wave, wait, fp, hidden_state=None):
        """
        wave: (batch, seq, wave_dim)
        wait: (batch, seq, wait_dim)
        fp: (batch, seq, fp_dim) - neighbor fingerprints
        hidden_state: (h, c)
        """
        # Feature embeddings
        h_wave = F.relu(self.fc_wave(wave))
        h_wait = F.relu(self.fc_wait(wait))
        h_fp = F.relu(self.fc_fp(fp))
        
        # Concatenate: (batch, seq, hidden_dim)
        h_combined = torch.cat([h_wave, h_wait, h_fp], dim=-1)
        
        # LSTM processing
        lstm_out, next_hidden = self.lstm(h_combined, hidden_state)
        
        # Heads
        logits = self.actor_head(lstm_out)
        probs = F.softmax(logits, dim=-1)
        
        value = self.critic_head(lstm_out).squeeze(-1)
        
        return probs, value, next_hidden
