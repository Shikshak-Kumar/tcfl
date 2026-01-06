# Intelligent Traffic Signal Control via Federated Knowledge Distillation (FedKD)

A professional research framework for optimizing urban traffic flow across heterogeneous intersections using **Federated Knowledge Distillation (FedKD)** and **Double Deep Q-Learning (Double DQN)**. This system enables collaborative learning on high-fidelity OpenStreetMap (OSM) data while preserving data privacy and supporting varied hardware architectures.

## 🔬 Core Innovations

- **Heterogeneous Architecture Support**: Allows intersections with different computational capacities (e.g., edge devices vs. cloud servers) to share knowledge via **behavioral alignment (Logits)** rather than structural replication (Weights).
- **Real-World Map Integration**: Natively utilizes the **`sumo_configs2`** dataset, derived from real-world city networks via OpenStreetMap.
- **Fairness-Aware Aggregation**: A customized congestion-weighting system that prioritizes learning from the most bottlenecked intersections.
- **Simulation-Free Verification**: Intelligent fallback mode for logic verification in environments without a running SUMO instance.

---

## 🚀 Quick Execution Guide

### 1. Environment Setup
```bash
# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install research dependencies
pip install -r requirements.txt
```

### 2. Running Simulations (Real Map Focus)

#### **Federated Knowledge Distillation (FedKD)**
*Our primary innovation—supports behavioral alignment and heterogeneous models.*
```bash
python3 train_fedkd.py --rounds 15 --results-dir results_fedkd
```

#### **Standard Federated Learning (FedAvg)**
*Baseline comparison using fairness-aware parameter synchronization.*
```bash
python3 train_federated.py --mode multi --num-rounds 15 --results-dir results_federated
```

---

## 📁 Project Architecture

```
.
├── agents/                  # RL optimization engines (Double DQN)
├── federated_learning/      # Collaborative orchestration modules (FedKD/FedAvg)
├── sumo_configs2/           # High-fidelity Real-World City Network (OSM)
├── RESEARCH_Theory.md       # Comprehensive mathematical & algorithmic framework
├── train_fedkd.py           # FedKD simulation entry point
├── train_federated.py       # Standard FL simulation entry point
├── scripts/                 # Comparative analysis & visualization utilities
└── README.md                # Project overview and execution guide
```

---

## 📉 Theoretical Foundation
For a deep dive into the mathematical justifications, reward function components, and distillation loss logic, please refer to:
👉 **[RESEARCH_Theory.md](file:///Users/shikshakkumar/Documents/WebProjects/btp/tt/RESEARCH_Theory.md)**

---

## 🧪 Technical Verification
The system includes highly detailed reporting for research analysis:
- **Waiting Time Metrics**: Reduction in vehicular delay across the network.
- **Throughput Analysis**: Volume of traffic cleared per signal cycle.
- **Knowledge Convergence**: Monitoring of distillation loss during collaborative rounds.

---
*Developed for professional academic presentation and technical review.*