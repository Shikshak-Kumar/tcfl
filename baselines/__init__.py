"""
Classical Traffic Control Algorithms for Baseline Comparison

This module contains implementations of baseline traffic control methods:
1. Fixed-Time Control - Traditional timer-based approach
2. MaxPressure - Pressure-based optimization algorithm
3. Centralized RL - Non-federated reinforcement learning
4. Independent RL - Each intersection learns independently
"""

from .fixed_time import FixedTimeController
from .maxpressure import MaxPressureController
from .centralized_rl import CentralizedRLController
from .independent_rl import IndependentRLController

__all__ = [
    'FixedTimeController',
    'MaxPressureController',
    'CentralizedRLController',
    'IndependentRLController',
]

