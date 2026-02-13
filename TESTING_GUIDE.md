# Testing Guide: Technical Verification and Execution

This document provides standardized procedures for verifying the operational status of the Smart Traffic Control system and its Federated Knowledge Distillation (FedKD) module.

## 1. System Requirements Verification

Before executing the simulations, verify the following prerequisites:

```bash
# Verify Python Environment
python3 --version  # Required: 3.8 or higher

# Verify SUMO Installation
sumo --version     # Required: 1.15.0 or higher

# Verify Core Dependencies (via Virtual Environment)
# Use Python 3.11 for maximum compatibility with research libraries.
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -c "import torch, numpy, flwr, traci; print('Environment check passed.')"
```

## 2. Simulation Execution

The system supports two environment modes for all federated reinforcement learning algorithms.

> [!IMPORTANT]
> - **Mock Mode** (Default): Run *without* the `--gui` flag. Uses high-fidelity internal traffic data for fast logic verification.
> - **SUMO Mode**: Run *with* the `--gui` flag. Requires an installed SUMO simulator to open a real-time visual simulation.

### 2.1 FedFlow-TSC (Hierarchical Graph-Aware)
**Goal**: Optimize traffic flow through spatial clustering and graph-coupled agents.

*   **Default Command**: `python3 train_fedflow.py`
*   **Default Behavior**: 5 rounds, 6 nodes, 2 clusters, Mock Mode.
*   **Parameters**:
    *   `--rounds`: Number of federated rounds (Default: 5).
    *   `--nodes`: Total number of simulated intersections (Default: 6).
    *   `--clusters`: Number of spatial clusters to group nodes into (Default: 2).
    *   `--gui`: Enable SUMO GUI (Action: store_true).

### 2.2 FedCM-RL (Cross-Model Distillation)
**Goal**: Enable collaborative training between agents with different AI architectures.

*   **Default Command**: `python3 train_fedcm.py`
*   **Default Behavior**: 15 rounds, 3 clients, "performance" weighting, Mock Mode.
*   **Parameters**:
    *   `--rounds`: Number of federated rounds (Default: 15).
    *   `--num-clients`: Number of simulated clients (Default: 3).
    *   `--proxy-size`: Size of the knowledge-sharing proxy dataset (Default: 2000).
    *   `--weighting`: Aggregation method: `uniform` or `performance` (Default: `performance`).
    *   `--results-dir`: Output directory (Default: `results_fedcm`).
    *   `--gui`: Enable SUMO GUI.

### 2.3 FedKD-RL (Knowledge Distillation)
**Goal**: State-representation sharing via Teacher-Student distillation.

*   **Default Command**: `python3 train_fedkd.py`
*   **Default Behavior**: 15 rounds, 2 clients, Mock Mode.
*   **Parameters**:
    *   `--rounds`: Number of federated rounds (Default: 15).
    *   `--num-clients`: Number of simulated clients (Default: 2).
    *   `--results-dir`: Output directory (Default: `results_fedkd`).
    *   `--gui`: Enable SUMO GUI.

### 2.4 FedAvg (Standard Baseline)
**Goal**: Standard federated averaging for traffic control.

*   **Default Command**: `python3 train_federated.py`
*   **Default Behavior**: 15 rounds, 2 clients, Mock Mode.
*   **Parameters**:
    *   `--rounds`: Number of federated rounds (Default: 15).
    *   `--clients`: Number of simulated clients (Default: 2).
    *   `--results-dir`: Output directory (Default: `results_federated`).
    *   `--gui`: Enable SUMO GUI.

---

## 3. Parameter Customization Guide

| Parameter | Recommended Use Case | Explanation |
| :--- | :--- | :--- |
| **`--rounds`** | Convergence testing | Increase (20+) for deeper training; decrease (1-5) for fast logic checks. |
| **`--num-clients`** | City-scale simulation | Use 10+ clients to simulate large-scale city networks. Configs will cycle automatically. |
| **`--gui`** | Visual Debugging | Pass this flag to watch vehicles move and traffic lights change in real-time. |
| **`--proxy-size`** | Distillation Quality | (FedCM/FedKD) Increase for better knowledge transfer between heterogeneous models. |
| **`--weighting`** | Heterogeneity tuning | (FedCM) Use `performance` to favor clients with lower wait times during aggregation. |

## 4. Results Analysis and Visualization

Utilize the following utilities to generate professional-grade performance reports:

1. **Dashboard Generation**:
   ```bash
   python3 visualize_results.py
   ```
   Generates comprehensive performance visualizations in `results/`.

2. **Statistical Summary**:
   ```bash
   python3 analyze_results.py
   ```
   Outputs a detailed statistical summary of traffic flow improvements.

## 5. Operational Considerations

- **Strict Separation**: CLI mode (no flag) *always* uses Mock data for performance. GUI mode (`--gui`) *always* opens SUMO.
- **Reporting**: All algorithms produce a detailed **Performance Table** at the end of each round.
- **Unit Testing**: You can verify individual component health using:
  ```bash
  python3 test_fedflow_components.py
  ```
