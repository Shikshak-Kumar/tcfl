#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    print("  ✓ Importing baselines...")
    from baselines.fixed_time import FixedTimeController
    from baselines.actuated import ActuatedController
    from baselines.maxpressure import MaxPressureController
    from baselines.sotl import SOTLController
    from baselines.centralized_rl import CentralizedRLController
    from baselines.independent_rl import IndependentRLController
    print("    ✓ All baseline imports successful")
except Exception as e:
    print(f"    ✗ Baseline import failed: {e}")
    sys.exit(1)

try:
    print("  ✓ Importing project modules...")
    from agents.traffic_environment import SUMOTrafficEnvironment
    from agents.dqn_agent import DQNAgent
    from federated_learning.fl_client import TrafficFLClient
    from train_federated import run_multi_client_simulation
    print("    ✓ All project imports successful")
except Exception as e:
    print(f"    ✗ Project import failed: {e}")
    sys.exit(1)

try:
    print("  ✓ Importing visualization libraries...")
    import matplotlib.pyplot as plt
    import numpy as np
    print("    ✓ Visualization libraries available")
except Exception as e:
    print(f"    ✗ Visualization import failed: {e}")
    print("    Install with: pip install matplotlib numpy")
    sys.exit(1)

print("\n✓ All imports successful! Ready to run comparisons.")

