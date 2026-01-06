# Technical Guide: Federated Knowledge Distillation (FedKD)

This document provides a comprehensive overview of the Federated Knowledge Distillation (FedKD) implementation within the Smart Traffic Control framework.

## Overview of FedKD

Classical Federated Learning (e.g., FedAvg) necessitates identical neural network architectures across all participants to facilitate parameter averaging. The implemented FedKD module removes this constraint by shifting the focus from parameter exchange to knowledge exchange.

### Primary Advantages
1. **Model Heterogeneity**: Supports collaboration between intersections utilizing different model architectures (e.g., high-capacity models vs. computationally efficient models).
2. **Communication Efficiency**: Utilizes soft predictions (logits) for synchronization, which significantly reduces communication overhead compared to high-dimensional parameter vectors.
3. **Advanced Learning Paradigm**: Employs a cross-architecture distillation framework to disseminate optimized control strategies across the network.

## Execution Framework

A dedicated orchestrator, `train_fedkd.py`, manages the distillation process.

```bash
python3 train_fedkd.py --rounds 20 --results-dir results_fedkd_final
```

### Configuration Parameters
- **`--rounds`**: Specifies the number of collaborative rounds. A minimum of 15 rounds is recommended for convergence.
- **`--results-dir`**: Specifies the output directory for performance metrics.

## Real-World Data Distillation

This implementation utilizes authentic traffic states observed directly from the SUMO simulation environment, rather than synthetic or randomly generated data.

### Procedural Workflow
1. **Observation Phase**: Clients record real-world traffic states (vehicle density, queue length, waiting times) during local training sessions.
2. **Proxy Data Selection**: A representative sample of these observed states is synchronized with the server to form a Real Traffic Proxy Set.
3. **Logit Synthesis**: Client models generate Q-value predictions (logits) based on these real-world states.
4. **Knowledge Aggregation**: The server synthesizes a global consensus from these predictions.
5. **Knowledge Distillation**: Individual clients update their local models to align with the global consensus using Mean Squared Error (MSE) minimization.

## Architectural Components

- **`agents/dqn_agent.py`**: Enhanced to support variable hidden layer dimensions, logit generation, and a dedicated distillation loss function.
- **`federated_learning/fedkd_client.py`**: Manages the local distillation workflow and real-world state sampling.
- **`federated_learning/fedkd_server.py`**: Orchestrates the Real Traffic Proxy Set and facilitates global knowledge aggregation.
