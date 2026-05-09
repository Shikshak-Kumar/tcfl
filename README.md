# Smart Traffic Control with AdaptFlow-TSC

This project implements an advanced Federated Reinforcement Learning (FRL) system for Traffic Signal Control (TSC), featuring the novel **AdaptFlow-AC** algorithm alongside several state-of-the-art baselines.

## 🌟 Key Features

- **Federated MARL**: Implements decentralized training with secure, efficient global aggregation.
- **Hybrid Architectures**: Combines Graph Attention Networks (GAT) with Actor-Critic and DQN frameworks.
- **Dynamic Clustering**: AdaptFlow-AC uses dynamic clustering to group intersections by traffic patterns for localized optimization.
- **High-Fidelity Simulation**: Integrated with SUMO (Simulation of Urban MObility) for realistic traffic modeling.


## 🔬 Research Algorithms

The repository includes implementations and benchmarking scripts for four primary algorithms:

1.  **AdaptFlow-AC**: Our flagship algorithm using Adaptive Dynamic Clustering + Hierarchical Graph-Aware Federated Actor-Critic.
2.  **FedDQN-TSC**: Federated Deep Q-Network baseline for comparative analysis.
3.  **Multi-Agent AC (MA2C)**: Standard Multi-Agent Actor-Critic implementation.
4.  **DQT-SCA**: Deep Q-Network with Spatial-Temporal Context Awareness baseline.

## 📊 Benchmarking & Results

We conduct extensive benchmarks on high-fidelity scenarios:
- **Dwarka Mor (Delhi)**: A complex 6-node intersection grid.

### Latest Results
Research runs generate comprehensive reports and training curves:
- **Summary Report**: [RESEARCH_COMPLETE.md](adaptflow_ac/results_research_sumo_configs2_20260503_192846/RESEARCH_COMPLETE.md)
- **Checkpoints**: [CHECKPOINTS.md](adaptflow_ac/results_research_sumo_configs2_20260503_192846/CHECKPOINTS.md)
- **Visualization**: Training curves are automatically generated under `results_research_.../plots_training/`.

## 📖 Documentation

*   **[AdaptFlow Simulation Flow](docs/simulation_flow.md)**: End-to-end breakdown of simulation logic.
*   **[AdaptFlow Novelties & Comparison](docs/adaptflow_novelties.md)**: Deep dive into the mathematical and architectural innovations.
*   **[Technical Docs](docs/)**: Additional guides on environment setup and algorithm details.

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **Node.js 16+**
- **SUMO (Simulation of Urban MObility)**:
    - *Mac*: `brew install sumo`
    - *Linux*: `sudo apt-get install sumo sumo-tools sumo-doc`

### 2. Installation

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Frontend
```bash
cd frontend
npm install
```

### 3. Running Research Benchmarks
To run a full comparative research suite:
```bash
cd adaptflow_ac
python run_all_research.py --rounds 10 --steps 2500
```

## 🛠️ Tech Stack
- **RL Framework**: PyTorch, SUMO, TraCI
- **Backend**: Python, FastAPI/Flask
- **Frontend**: React, Vite, TailwindCSS
- **Tools**: Matplotlib (Plotting), JSON (Metrics)
