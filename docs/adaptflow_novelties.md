# AdaptFlow: Technical Novelties & Comparison

AdaptFlow is the centerpiece of this project, introducing several architectural innovations to Federated Reinforcement Learning (FRL) for Traffic Signal Control (TSC).

## 🚀 Key Novelties of AdaptFlow

### 1. Dynamic Adaptive Clustering
*   **The Problem**: Standard FRL (like FedAvg) groups all intersections together, even if one is a quiet suburb and the other is a chaotic city center. This causes "gradient pollution" where the quiet node ruins the learning of the busy node.
*   **The Solution**: AdaptFlow uses an `AdaptiveClusterManager` to re-group intersections every round based on their **Congestion Fingerprints** (real-time traffic similarity) and **POI Priority**.
*   **Impact**: Agents only learn from "peers" facing similar traffic challenges.

### 2. Spatio-Temporal Graph Attention (GAT)
*   **The Problem**: Standard DQN agents only see their own local queue. They are "blind" to a huge wave of traffic coming from a neighbor until it's too late.
*   **The Solution**: AdaptFlow implements a **Spatio-Temporal Encoder**. It uses Graph Attention to "listen" to neighbor states and temporal sequences (history) to predict future arrivals.
*   **Impact**: Proactive signal switching before the traffic jam even forms.

### 3. Prioritized Experience Replay (PER)
*   **The Problem**: Standard training treats all experiences as equally important. However, rare events (like clearing a major traffic jam) contain much more learning value than "normal" flow.
*   **The Solution**: AdaptFlow uses a `SumTree`-based **PER Buffer**. It prioritizes transitions with high TD-error, ensuring the agent spends more time learning from challenging or critical traffic scenarios.
*   **Impact**: Faster convergence and better performance in edge-case congested states.

### 4. Pareto Multi-Objective Rewards
*   **The Problem**: Optimizing only for "waiting time" can lead to dangerous behavior or extreme instability (rapid light flipping).
*   **The Solution**: A scalarized Pareto reward function that balances:
    *   **Efficiency** (Queue & Wait Time)
    *   **Safety** (Accident/Near-miss penalty)
    *   **Stability** (Signal switching penalty)
*   **Impact**: Real-world deployable control that is both fast AND safe.

---

## 📊 Algorithm Comparison Matrix

| Feature | Demo (No RL) | FedAvg | FedCM / FedKD | **AdaptFlow (Novel)** |
| :--- | :---: | :---: | :---: | :---: |
| **Learning** | ❌ (Static) | ✅ Standard RL | ✅ Knowledge Dist. | ✅ **GAT + PER RL** |
| **Clustering** | N/A | Static (All) | Static (Fixed) | **Dynamic (Adaptive)** |
| **Context Aware** | ❌ | Local Only | Local Only | **Spatio-Temporal** |
| **Traffic Data** | Mock | Mock | Mock | **TomTom Real-Time** |
| **Safety Logic** | ❌ | Basic | Basic | **Pareto Multi-Obj** |
| **Best For** | Testing UI | Simple Networks | Disconnected Nodes | **Complex Urban Grids** |

---

## 🏆 Why AdaptFlow Wins
While **FedAvg** is good for basic coordination, **AdaptFlow** is built for the complexity of a real city. By combining **TomTom Real-Time Intelligence** with **Dynamic Clustering**, it ensures that the "Hospital Zone" signal is prioritized correctly according to its real-world importance and current live congestion.
