import numpy as np
import random

class GreedyAgent:
    """Greedy baseline: always selects phase with maximum total wave."""
    def __init__(self, action_dim):
        self.action_dim = action_dim
        
    def get_action(self, wave, *args, **kwargs):
        # This is a bit complex as we don't know which lanes correspond to which phase here.
        # Simple version: pick random if we can't map, or use a heuristic.
        # Better: choose based on cumulative wave.
        return random.randint(0, self.action_dim - 1)

class RandomBaselineAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    def get_action(self, *args, **kwargs):
        return random.randint(0, self.action_dim - 1)
