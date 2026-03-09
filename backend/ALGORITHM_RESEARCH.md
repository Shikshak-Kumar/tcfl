# Research Documentation: Federated Traffic Signal Control Algorithms

This document outlines the theoretical foundations and academic references for the algorithms implemented in the FedFlow-TSC project.

---

## 1. FedFlow-TSC: Hierarchical Graph-Aware TSC
**Focus**: Spatial clustering and hierarchical parameter aggregation based on traffic flow density.

### Description
FedFlow-TSC organizes intersections into spatial clusters (communities). It uses a two-level aggregation scheme:
1.  **Intra-Cluster**: Weights are aggregated within a cluster, prioritized by the real-time traffic flow magnitude of each node.
2.  **Inter-Cluster**: Clusters are aggregated at the global server level to form a meta-policy.

### Research Citations
*   **Hierarchical FL Architecture**: Liu, L., Zhang, J., Song, S. H., & Letaief, K. B. (2020). *Client-Edge-Cloud Hierarchical Federated Learning*. In IEEE International Conference on Communications (ICC).
*   **Graph-Aware Traffic Modeling**: Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*. In Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI).

---

## 2. FedCM-RL: Cross-Model Federated RL
**Focus**: Model-heterogeneity and ensemble distillation.

### Description
FedCM (Cross-Model) allows the federated network to include clients with completely different neural network architectures (e.g., varying depths of DQNs). It uses a shared proxy dataset of traffic states to generate logit-level consensus.

### Research Citations
*   **Heterogeneous Distillation**: Li, T., & Wang, J. (2019). *FedMD: Heterogenous Federated Learning via Model Distillation*. arXiv preprint arXiv:1910.03581.
*   **Ensemble Fusion**: Lin, T., Kong, L., Stich, S. U., & Jaggi, M. (2020). *Ensemble Distillation for Robust Model Fusion in Federated Learning*. In Advances in Neural Information Processing Systems (NeurIPS).
*   **HeteroFL Framework**: Diao, E., Ding, J., & Tarokh, V. (2020). *HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients*. In International Conference on Learning Representations (ICLR).

---

## 3. FedKD-RL: Knowledge Distillation RL
**Focus**: Knowledge-sharing without raw parameter exchange.

### Description
FedKD-RL implements a "Behavioral Alignment" strategy. Instead of averaging weights, students (clients) learn from a "Global Teacher" (aggregated logits). This is significantly more communication-efficient than standard FL and protects model IP.

### Research Citations
*   **Distillation Foundations**: Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv preprint arXiv:1503.02531.
*   **IoT Distillation**: Itahara, S., Nishio, T., Koda, Y., Yamamoto, M., & Morikura, J. (2020). *Distillation-Based Federated Learning for Low-Bandwidth IoT Networks*. In IEEE 92nd Vehicular Technology Conference (VTC2020-Fall).

---

## 4. FedAvg: Federated Averaging (Standard Baseline)
**Focus**: Parameter-level synchronization and global model convergence.

### Description
The standard baseline for federated learning. In our implementation, we add a "Congestion-Aware" weight to the vanilla FedAvg to prioritize learning from the most bottlenecked intersections.

### Research Citations
*   **Algorithm Origin**: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. In AISTATS.
*   **Convergence Analysis**: Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z. (2019). *On the Convergence of FedAvg on Non-IID Data*. In International Conference on Learning Representations (ICLR).

---

## 5. Local Decision Core: Double DQN
**Focus**: Mitigating overestimation in high-variance traffic environments.

### Research Citations
*   **Double DQN Implementation**: Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning*. In AAAI Conference on Artificial Intelligence.
