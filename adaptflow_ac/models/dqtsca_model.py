import torch
import torch.nn as nn
import torch.nn.functional as F

class DQTSCAModel(nn.Module):
    """
    DQTSCA Network (CNN-based).
    Two branches: one for Occupancy, one for Speed.
    Combined with Phase vector.
    """
    def __init__(self, grid_size=20, phase_dim=4, action_dim=4):
        super(DQTSCAModel, self).__init__()
        
        # Branch 1: Occupancy (1 channel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2), # (20-4)/2 + 1 = 9
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2), # (9-2)/1 + 1 = 8
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Branch 2: Speed (1 channel)
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size: 32 * 8 * 8 = 2048
        # Two branches = 2048 * 2 = 4096
        # + phase vector = 4100
        combined_dim = (32 * 8 * 8) * 2 + phase_dim
        
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, occ, speed, phase):
        """
        occ: (batch, 1, 20, 20)
        speed: (batch, 1, 20, 20)
        phase: (batch, 4)
        """
        feat1 = self.branch1(occ)
        feat2 = self.branch2(speed)
        
        combined = torch.cat([feat1, feat2, phase], dim=-1)
        q_values = self.fc(combined)
        return q_values
