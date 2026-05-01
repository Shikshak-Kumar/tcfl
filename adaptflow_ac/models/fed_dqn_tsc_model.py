import torch
import torch.nn as nn
import torch.nn.functional as F

class FedDQNTscModel(nn.Module):
    """
    DQN architecture from Bao et al. (2023).
    Global layers (first 2) are shared.
    Local layers (output layer) are specific to each intersection.
    """
    def __init__(self, input_dim, action_dim):
        super(FedDQNTscModel, self).__init__()
        
        # Global Feature Layers
        self.global_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Local Feature Layers (Output Layers)
        # Note: 256 is mentioned as the 3rd layer
        self.local_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # x: (Batch, N, M, K)
        x = x.view(x.size(0), -1) # Flatten 3D state
        x = self.global_layers(x)
        x = self.local_layers(x)
        return x

    def get_global_weights(self):
        """Returns the state_dict of the global layers only."""
        return self.global_layers.state_dict()

    def load_global_weights(self, global_weights):
        """Loads weights into the global layers only."""
        self.global_layers.load_state_dict(global_weights)

    def freeze_global_layers(self):
        """Freezes global layers for fine-tuning."""
        for param in self.global_layers.parameters():
            param.requires_grad = False
            
    def unfreeze_local_layers(self):
        """Ensures local layers are trainable."""
        for param in self.local_layers.parameters():
            param.requires_grad = True
