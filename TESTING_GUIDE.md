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

### Federated Knowledge Distillation (FedKD)
The FedKD module utilizes real-world traffic states observed during simulation for knowledge transfer.

**1. Execution without GUI (Headless/Optimized)**
Use this for faster training and technical verification.
```bash
python3 train_fedkd.py --rounds 20 --results-dir results_fedkd_final
```

**2. Execution with SUMO GUI (Visual Simulation)**
Use this to visually inspect traffic dynamics and signal transitions.
```bash
python3 train_fedkd.py --rounds 20 --results-dir results_fedkd_visual --gui
```

### Standard Federated Learning
To execute a standard federated simulation with or without GUI:

**Headless Mode:**
```bash
python3 train_federated.py --mode multi --num-rounds 15 --results-dir results_standard_final
```

**GUI Mode:**
```bash
python3 train_federated.py --mode multi --num-rounds 15 --results-dir results_standard_visual --gui
```

## 3. Results Analysis and Visualization

Utilize the following utilities to generate professional-grade performance reports:

1. **Dashboard Generation**:
   ```bash
   python3 visualize_results.py
   ```
   Generates comprehensive performance visualizations in `results/training_dashboard_latest.png`.

2. **Statistical Summary**:
   ```bash
   python3 analyze_results.py
   ```
   Outputs a detailed statistical summary of traffic flow improvements and reward metrics.

## 4. Operational Considerations

- **GUI Execution**: To visualize traffic dynamics, include the `--gui` flag. 
- **Headless Execution**: For performance-optimized training or execution on remote servers, omit the `--gui` flag.
- **Data Persistence**: Metrics are archived in `.json` format within the specified results directory for further post-processing.
