import torch
import torch.nn as nn
import torch.nn.functional as F

class FedDQNTscModel(nn.Module):
    """
    DQN Architecture from Scientific Reports 2023.
    Input: Flattened state matrix s[k][n][m] -> (4 * 7 * 8) = 224
    FC Layers: 64 -> 128 -> 256
    """
    def __init__(self, input_dim=224, action_dim=4):
        super(FedDQNTscModel, self).__init__()
        
        # Global Layers (w_g): Feature extraction shared across agents
        self.global_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Local Layer (w_l): Task-specific output head
        self.local_layer = nn.Linear(256, action_dim)
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch_size, 4, 7, 8) or (batch_size, 224)
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        
        feat = self.global_layers(x)
        q_values = self.local_layer(feat)
        return q_values

    def get_global_weights(self):
        """Returns weights for aggregation (w_g)."""
        return self.global_layers.state_dict()

    def set_global_weights(self, weights):
        """Broadcasts global weights (w_G) to local model."""
        self.global_layers.load_state_dict(weights)

    def get_local_weights(self):
        """Returns local weights (w_l)."""
        return self.local_layer.state_dict()
        
    def freeze_global_layers(self):
        """Used for fine-tuning stage (Section 9)."""
        for param in self.global_layers.parameters():
            param.requires_grad = False
