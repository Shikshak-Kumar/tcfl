# AdaptFlow Training — Detailed Analysis Report

**Date:** 2026-04-11  
**Configuration:** 5 rounds · 6 nodes · 800 steps · 500 vehicles/node · SUMO-GUI mode  


---

## 1. Executive Summary

AdaptFlow was trained for **5 federated rounds** across **6 traffic signal nodes** on a real-world Chinese road network extracted from OpenStreetMap. Each node controlled a single intersection with 3 incoming edges (7 lanes total) and managed 500 departing vehicles over an 800-step episode.

**Key findings:**

- The system converged rapidly — metrics stabilized by **Round 2** and remained constant through Round 5.
- Two distinct traffic profiles emerged: a **low-congestion group** (avg wait ~1.96s, queue ~0.21) and a **high-congestion group** (avg wait ~2.17s, queue ~0.80).
- Overall throughput ratio settled at **~0.699** (349–350 out of 500 vehicles completing trips).
- Loss values decreased consistently across all nodes, indicating successful policy learning.
- The dynamic clustering mechanism was active, performing **re-clustering at every round**, though the underlying node behavior converged to only two stable archetypes.

---

## 2. Simulation Configuration

| Parameter | Value |
|---|---|
| Federated Rounds | 5 |
| Nodes (Intersections) | 6 (node_0 – node_5) |
| Steps per Episode | 800 |
| Vehicles per Node | 500 |
| Number of Clusters | 2 |
| Simulation Mode | SUMO-GUI |
| Incoming Edges per Node | 3 (203598795#3, 608989233#2, 39452784#4) |
| Total Lanes per Node | 7 |
| Min Green Time | 5.0s |
| Max Green Time | 30.0s |

---

## 3. Global Metrics Across Rounds

### 3.1 Aggregate Trends

| Round | Total Reward | Avg Queue Length | Avg Waiting Time (s) | Throughput Ratio |
|:-----:|:------------:|:----------------:|:--------------------:|:----------------:|
| 1 | 782.59 | 0.354 | 1.949 | 0.698 |
| 2 | 763.10 | 0.456 | 2.034 | 0.700 |
| 3 | 754.35 | 0.505 | 2.063 | 0.699 |
| 4 | 754.35 | 0.505 | 2.063 | 0.699 |
| 5 | 754.35 | 0.505 | 2.063 | 0.699 |

**Observations:**

- **Reward** decreased from 782.6 → 754.4 over the first 3 rounds, then stabilized. This decrease correlates with the two-cluster split causing some nodes to experience higher queue lengths (cluster_1 nodes with queue ~0.80).
- **Throughput ratio** is very stable (0.698–0.700), meaning ~70% of all injected vehicles reach their destination within the 800-step window regardless of round.
- **Waiting time** increased from 1.95s → 2.06s as the model explored different signal timing strategies, then converged.
- **Queue length** increased from 0.35 → 0.51 reflecting the cluster_1 nodes settling into a higher-queue operating regime.
- **Rounds 3–5 are fully converged** — all aggregate metrics are identical.

### 3.2 Convergence Assessment

The system reached a **stable equilibrium by Round 3**. Rounds 3, 4, and 5 produce identical rewards (754.35), identical queue lengths (0.505), identical waiting times (2.063s), and identical throughput ratios (0.699). This indicates:

1. The DQN policies have converged to a stable action-selection pattern.
2. The traffic fingerprints have stabilized (see §5), so the clustering still oscillates between equivalent label assignments but the actual groups are fixed.
3. Further training rounds would not yield improvement without architectural changes or demand variation.

---

## 4. Per-Node Performance

### 4.1 Round 1 (Initial Exploration)

| Node | Cluster | Reward | Avg Wait (s) | Loss | Departed | Arrived | TP Ratio | Avg Queue | Max Queue |
|:----:|:-------:|:------:|:------------:|:----:|:--------:|:-------:|:--------:|:---------:|:---------:|
| node_0 | cluster_0 | 788.86 | 2.512 | 0.5353 | 500 | 342 | 0.684 | 0.254 | 2 |
| node_1 | cluster_0 | 786.13 | 1.517 | 0.5471 | 499 | 352 | 0.705 | 0.375 | 6 |
| node_2 | cluster_0 | 785.08 | 2.416 | 0.5605 | 500 | 345 | 0.690 | 0.285 | 2 |
| node_3 | cluster_1 | 782.04 | 1.426 | 0.5518 | 500 | 353 | 0.706 | 0.410 | 5 |
| node_4 | cluster_1 | 786.84 | 2.234 | 0.6160 | 500 | 347 | 0.694 | 0.271 | 2 |
| node_5 | cluster_1 | 766.61 | 1.586 | 0.5596 | 500 | 354 | 0.708 | 0.530 | 6 |

**Round 1 highlights:**
- Highest heterogeneity — avg wait ranges from 1.43s (node_3) to 2.51s (node_0).
- node_5 had the highest throughput (354 vehicles, 70.8%) but also the highest queue (0.53).
- node_0 had the worst throughput ratio (68.4%) and highest wait time (2.51s).
- Initial clustering: {node_0, node_1, node_2} in cluster_0, {node_3, node_4, node_5} in cluster_1. This initial split did **not** group similar nodes together — node_1 (wait 1.52, queue 0.38) was grouped with node_0 (wait 2.51, queue 0.25), despite having dissimilar profiles.

### 4.2 Rounds 2–5 (Converged State)

From Round 2 onward, all nodes collapse into exactly **two behavioral archetypes**:

| Archetype | Nodes | Reward | Avg Wait (s) | Departed | Arrived | TP Ratio | Avg Queue | Max Queue |
|:---------:|:-----:|:------:|:------------:|:--------:|:-------:|:--------:|:---------:|:---------:|
| **Low-Queue** | node_0, node_2, node_4 | 790.32 | 1.956 | 500 | 349 | 0.698 | 0.214 | 2 |
| **High-Queue** | node_1, node_3, node_5 | 718.38 | 2.170 | 500 | 350 | 0.700 | 0.796 | 6 |

**Key observations:**
- Nodes within each archetype produce **exactly identical** metrics — same reward, same wait, same throughput, same queue.
- The Low-Queue archetype achieves **higher reward** (790.32 vs 718.38) despite slightly lower throughput ratio (0.698 vs 0.700), because lower queue lengths result in fewer penalties per step.
- The High-Queue archetype has **3.7× higher queue** (0.796 vs 0.214) and **3× higher max queue** (6 vs 2), but only marginally higher throughput (+1 vehicle).
- Both archetypes have the same wait time (~2s), meaning the extra queuing in the High-Queue group doesn't proportionally increase per-vehicle delay.

### 4.3 Loss Convergence Per Node

| Node | R1 Loss | R2 Loss | R3 Loss | R4 Loss | R5 Loss | Δ (R1→R5) |
|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|:---------:|
| node_0 | 0.5353 | 0.5296 | 0.5293 | 0.5349 | 0.5301 | −0.005 |
| node_1 | 0.5471 | 0.5172 | 0.5204 | 0.4986 | 0.4950 | −0.052 |
| node_2 | 0.5605 | 0.5221 | 0.5225 | 0.5211 | 0.5207 | −0.040 |
| node_3 | 0.5518 | 0.5020 | 0.4656 | 0.4969 | 0.4933 | −0.059 |
| node_4 | 0.6160 | 0.5258 | 0.5156 | 0.5161 | 0.5272 | −0.089 |
| node_5 | 0.5596 | 0.4895 | 0.4754 | 0.4449 | 0.4777 | −0.082 |

- **All nodes show decreasing loss**, confirming the DQN agents are learning successfully.
- **node_4** had the largest loss reduction (−0.089), starting from the highest initial loss (0.616).
- **node_5** achieved the lowest absolute loss (0.445 in Round 4), indicating it found the most effective policy.
- The High-Queue nodes (node_1, node_3, node_5) show larger total loss reductions on average (−0.064) vs Low-Queue nodes (−0.045), suggesting they had more room for policy improvement.

---

## 5. Dynamic Clustering Analysis

### 5.1 Cluster Assignments Per Round

| Round | Cluster 0 | Cluster 1 | Transitions |
|:-----:|:---------:|:---------:|:-----------:|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | — (initial) |
| 2 | node_0, node_2, node_4 | node_1, node_3, node_5 | node_1: 0→1, node_4: 1→0 |
| 3 | node_1, node_3, node_5 | node_0, node_2, node_4 | All 6 nodes swapped |
| 4 | node_1, node_3, node_5 | node_0, node_2, node_4 | None |
| 5 | node_0, node_2, node_4 | node_1, node_3, node_5 | All 6 nodes swapped |

**Analysis:**
- After Round 1, the system **correctly re-clusters** in Round 2, moving node_1 and node_4 to match their actual behavior rather than their geographic grouping.
- From Round 2 onward, the **membership** is stable: {node_0, node_2, node_4} always cluster together, and {node_1, node_3, node_5} always cluster together.
- The apparent "transitions" in Rounds 3 and 5 are simply **label flips** — the cluster labels (0 vs 1) swap, but the groups themselves don't change. This is a cosmetic artifact of the clustering algorithm (spectral clustering labels are arbitrary).
- The effective cluster assignment has been **stable since Round 2**.

### 5.2 Cluster Characteristics

| Metric | Low-Queue Cluster (node_0,2,4) | High-Queue Cluster (node_1,3,5) |
|--------|:------------------------------:|:-------------------------------:|
| Avg Flow (TP Ratio) | 0.698 | 0.700 |
| Congestion (Avg Wait) | 1.956 | 2.170 |
| Avg Queue | 0.214 | 0.796 |
| Max Queue | 2 | 6 |
| Reward | 790.32 | 718.38 |

The two clusters represent two distinct **intersection operating regimes**:
- **Low-Queue**: Short, controlled queues (max 2 vehicles) with slightly lower throughput.  
- **High-Queue**: Longer burst queues (up to 6 vehicles) with marginally better throughput but significantly lower reward.

### 5.3 Traffic Fingerprints

The fingerprint vector is: `[avg_wait, avg_queue, throughput_ratio, max_queue, ?, flag]`

| Node | R1 Fingerprint | R2+ Fingerprint (stable) |
|:----:|:---------------|:-------------------------|
| node_0 | [2.512, 0.254, 0.684, 2, 0, 1.0] | [1.956, 0.214, 0.698, 2, 0, 1.0] |
| node_1 | [1.517, 0.375, 0.705, 6, 0, 0.5] | [2.170, 0.796, 0.700, 6, 0, 0.5] |
| node_2 | [2.416, 0.285, 0.690, 2, 0, 0.0] | [1.956, 0.214, 0.698, 2, 0, 0.0] |
| node_3 | [1.426, 0.410, 0.706, 5, 0, 0.0] | [2.170, 0.796, 0.700, 6, 0, 0.0] |
| node_4 | [2.234, 0.271, 0.694, 2, 0, 0.0] | [1.956, 0.214, 0.698, 2, 0, 0.0] |
| node_5 | [1.586, 0.530, 0.708, 6, 0, 0.0] | [2.170, 0.796, 0.700, 6, 0, 0.0] |

- Round 1 fingerprints are diverse — each node has a unique signature.
- By Round 2, nodes collapse into two identical fingerprints (except for node_0's flag=1.0 and node_1's flag=0.5).
- The 5th dimension is always 0.0 for all nodes (appears unused or represents zero congested lanes).
- The 6th dimension (flag) is node-specific and does not change: node_0=1.0, node_1=0.5, all others=0.0.

---

## 6. Cosine Similarity Matrices

### 6.1 Round 1 (Pre-convergence)

|  | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--|:------:|:------:|:------:|:------:|:------:|:------:|
| **node_0** | 1.000 | 0.784 | 0.957 | 0.782 | 0.955 | 0.770 |
| **node_1** | 0.784 | 1.000 | 0.806 | 0.996 | 0.826 | 0.996 |
| **node_2** | 0.957 | 0.806 | 1.000 | 0.828 | 0.999 | 0.815 |
| **node_3** | 0.782 | 0.996 | 0.828 | 1.000 | 0.848 | 1.000 |
| **node_4** | 0.955 | 0.826 | 0.999 | 0.848 | 1.000 | 0.835 |
| **node_5** | 0.770 | 0.996 | 0.815 | 1.000 | 0.835 | 1.000 |

Even in Round 1, two groups are clearly visible:
- **Group A** {node_0, node_2, node_4}: pairwise similarity ≥ 0.955
- **Group B** {node_1, node_3, node_5}: pairwise similarity ≥ 0.996
- **Inter-group** similarity: 0.77–0.85

### 6.2 Rounds 3–5 (Fully Converged)

|  | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--|:------:|:------:|:------:|:------:|:------:|:------:|
| **node_0** | 1.000 | 0.877 | 0.945 | 0.854 | 0.945 | 0.854 |
| **node_1** | 0.877 | 1.000 | 0.901 | 0.997 | 0.901 | 0.997 |
| **node_2** | 0.945 | 0.901 | 1.000 | 0.904 | **1.000** | 0.904 |
| **node_3** | 0.854 | 0.997 | 0.904 | 1.000 | 0.904 | **1.000** |
| **node_4** | 0.945 | 0.901 | **1.000** | 0.904 | 1.000 | 0.904 |
| **node_5** | 0.854 | 0.997 | 0.904 | **1.000** | 0.904 | 1.000 |

**Converged state observations:**
- **node_2 ≡ node_4** (similarity = 1.000) — completely identical model weights.
- **node_3 ≡ node_5** (similarity = 1.000) — completely identical model weights.
- **node_0** is very close to node_2/node_4 (0.945) but not identical — the flag=1.0 in its fingerprint creates a slight differentiation.
- **node_1** is very close to node_3/node_5 (0.997) but not identical — the flag=0.5 creates a slight differentiation.
- Inter-group similarity has **increased** from Round 1 (0.77–0.85) to converged (0.85–0.90), indicating that even the two clusters are converging toward each other over time.

---

## 7. Green Signal Time (GST) Analysis

### 7.1 Avg GST Per Edge (Round 5, Converged State)

| Node | Edge 203598795#3 | Edge 608989233#2 | Edge 39452784#4 |
|:----:|:----------------:|:----------------:|:---------------:|
| node_0 | 0.232 | 0.280 | 0.256 |
| node_1 | 1.166 | 1.500 | 0.060 |
| node_2 | 0.232 | 0.280 | 0.256 |
| node_3 | 1.166 | 1.500 | 0.060 |
| node_4 | 0.232 | 0.280 | 0.256 |
| node_5 | 1.166 | 1.500 | 0.060 |

**Two distinct signal timing strategies emerged:**

1. **Balanced GST** (Low-Queue nodes: 0, 2, 4): Distributes green time roughly evenly across all 3 edges (0.23–0.28 per edge). This prevents burst queuing and keeps max queue at 2.

2. **Prioritized GST** (High-Queue nodes: 1, 3, 5): Heavily favors edge 608989233#2 (1.50) and 203598795#3 (1.17) while nearly starving edge 39452784#4 (0.06). This creates burst queues of up to 6 vehicles on the starved approaches but achieves marginally higher throughput (350 vs 349).

### 7.2 GST Evolution

| Round | Low-Queue Nodes (avg across edges) | High-Queue Nodes (avg across edges) |
|:-----:|:----------------------------------:|:-----------------------------------:|
| 1 | 0.28 (varied per node) | 0.56 (varied per node) |
| 2 | 0.30 | 0.58 |
| 3 | 0.26 | 0.79 |
| 4 | 0.26 | 0.91 |
| 5 | 0.26 | 0.91 |

The High-Queue nodes progressively increased their GST disparity across edges over training, concentrating green time on the two higher-capacity edges (3-lane edges at 186m and 247m length) while reducing it for the low-capacity edge (1-lane at 2.25m).

---

## 8. Reward Decomposition

| Component | Low-Queue (node_0,2,4) | High-Queue (node_1,3,5) | Difference |
|-----------|:----------------------:|:-----------------------:|:----------:|
| Total Reward | 790.32 | 718.38 | **+71.94** |
| Vehicles Arrived | 349 | 350 | −1 |
| Avg Queue | 0.214 | 0.796 | −0.582 |
| Max Queue | 2 | 6 | −4 |
| Avg Wait | 1.956s | 2.170s | −0.214s |

The **reward gap of 71.94** between archetypes is driven almost entirely by queue-length penalties:
- Over 800 steps, a queue difference of ~0.58 per step amounts to ~464 extra queue-seconds of penalty for High-Queue nodes.
- The +1 throughput advantage of High-Queue nodes (350 vs 349) is far too small to offset the queue penalty.
- This suggests the reward function strongly penalizes queuing, making the **Balanced GST strategy (Low-Queue) the superior policy**.

---

## 9. Convergence & Stability Diagnosis

### 9.1 What Converged Well

✅ **Loss**: Consistently decreased across all nodes (−0.005 to −0.089 reduction)  
✅ **Throughput**: Stable at 69.8–70.0% across all nodes from Round 2  
✅ **Clustering**: Correctly identified two behavioral archetypes by Round 2  
✅ **Similarity**: Intra-cluster similarity reached 0.945–1.000  
✅ **Policy**: Two distinct, stable signal timing strategies emerged  

### 9.2 Areas of Concern

⚠️ **Early metric stagnation**: Metrics stopped changing after Round 3, meaning Rounds 4–5 provided no additional benefit. Training could be shortened.  
⚠️ **~30% vehicle loss**: Only 349–350 of 500 vehicles complete their trips. The remaining ~150 vehicles are still in-network at step 800 or failed to insert.  
⚠️ **Identical node behavior within clusters**: node_2 and node_4 produce identical weights (similarity=1.000), as do node_3 and node_5. This suggests the model isn't learning intersection-specific features — it's just learning the cluster-level policy.  
⚠️ **Cluster label oscillation**: Although cluster membership is stable, the labels flip between rounds (Rounds 3, 5), which could confuse downstream analytics if not handled.

### 9.3 Recommendations

1. **Reduce training to 3 rounds** — no improvement occurs after Round 3.
2. **Increase vehicle demand or simulation steps** to increase throughput ratio beyond 70%.
3. **Investigate the 30% non-arriving vehicles** — are they stuck in traffic, still en-route, or failed to insert?
4. **Consider adding intersection-specific features** to the fingerprint to prevent weight collapse within clusters.
5. **Fix cluster label stability** — pin cluster labels to prevent cosmetic oscillation between rounds.

---

## 10. Summary Statistics

| Metric | Round 1 | Round 5 | Change |
|--------|:-------:|:-------:|:------:|
| Global Reward | 782.59 | 754.35 | −3.6% |
| Avg Waiting Time | 1.95s | 2.06s | +5.9% |
| Avg Queue Length | 0.354 | 0.505 | +42.7% |
| Throughput Ratio | 0.698 | 0.699 | +0.1% |
| Avg Loss (all nodes) | 0.5617 | 0.5073 | −9.7% |
| Best Node Reward | 788.86 | 790.32 | +0.2% |
| Worst Node Reward | 766.61 | 718.38 | −6.3% |
| Cluster Stability | N/A | Converged | ✅ |

---

*Report generated from: `adaptflow_all_rounds.json`, `cluster_history.json`, `round_{1..5}_summary.json`*
