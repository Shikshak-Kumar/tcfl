import torch
import torch.nn as nn
import torch.nn.functional as F

class ScientificDQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ScientificDQN, self).__init__()
        
        # Global Feature Layers (wg)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        
        # Local Feature Layers (wl)
        self.fc3 = nn.Linear(128, 256)
        self.output = nn.Linear(256, action_dim)
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)

    def get_global_weights(self):
        """Returns the weights of global layers."""
        return {
            'fc1.weight': self.fc1.weight.data.clone(),
            'fc1.bias': self.fc1.bias.data.clone(),
            'fc2.weight': self.fc2.weight.data.clone(),
            'fc2.bias': self.fc2.bias.data.clone()
        }

    def set_global_weights(self, weights):
        """Sets the weights of global layers."""
        self.fc1.weight.data.copy_(weights['fc1.weight'])
        self.fc1.bias.data.copy_(weights['fc1.bias'])
        self.fc2.weight.data.copy_(weights['fc2.weight'])
        self.fc2.bias.data.copy_(weights['fc2.bias'])

    def freeze_global_layers(self, freeze=True):
        """Freezes or unfreezes global layers."""
        for param in self.fc1.parameters():
            param.requires_grad = not freeze
        for param in self.fc2.parameters():
            param.requires_grad = not freeze
