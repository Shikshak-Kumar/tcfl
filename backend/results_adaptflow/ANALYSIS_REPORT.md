# AdaptFlow Training — Detailed Analysis Report

**Date:** 2026-04-12
**Command:** `python train_adaptflow.py --gui --steps 1000 --rounds 5`
**Configuration:** 5 rounds · 6 nodes · 1000 steps/episode · ~625 vehicles/node · SUMO-GUI mode
**SUMO Configs:** `osm_client1.sumocfg` (nodes 0, 2, 4) · `osm_client2.sumocfg` (nodes 1, 3, 5)

---

## 1. Executive Summary

AdaptFlow was trained for **5 federated rounds** across **6 traffic signal nodes** on a real-world Chinese road network (OpenStreetMap extract). Each node ran a **1,000-step SUMO episode** managing a signalised intersection with 3 incoming edges and 7 lanes.

**Key findings:**

- **Global throughput: 75.7%** — 2,838 of ~3,747 vehicles complete their trips per round, a strong result for a 1,000-step window.
- **Two stable behavioral archetypes** emerged by Round 2 and were held for all subsequent rounds: a **Low-Queue group** (nodes 0, 2, 4 — avg wait 1.978s, avg queue 0.263, max queue 3) and a **High-Queue group** (nodes 1, 3, 5 — avg wait 2.202s, avg queue 0.705, max queue 6).
- **Loss reduced by −12.8%** on average across all nodes (0.5452 → 0.4755), with the sharpest single-round drop between Round 1 and Round 2 (−10.9%).
- **Spectral re-clustering succeeded after just 1 round**: initial geographic grouping (nodes 0-1-2 vs 3-4-5) was corrected to behavioral grouping (nodes 0-2-4 vs 1-3-5) in Round 2.
- **Full behavioral convergence by Round 3**; Rounds 3–5 produce identical reward, queue, and throughput metrics. Loss continues declining marginally in Rounds 4–5.
- **Two distinct signal timing strategies** crystallized: a balanced GST strategy (Low-Queue) and a heavily biased priority-GST strategy (High-Queue) that near-starves edge `39452784#4` (0.04s green time).

---

## 2. Simulation Configuration

| Parameter | Value |
|---|---|
| Federated Rounds | 5 |
| Nodes (Intersections) | 6 (node_0 – node_5) |
| Steps per Episode | 1,000 |
| Vehicles Departed (avg/node) | 625 (osm_client1) / 624 (osm_client2) |
| Number of Clusters | 2 |
| Simulation Mode | SUMO-GUI |
| Incoming Edges per Node | 3 (`203598795#3`, `608989233#2`, `39452784#4`) |
| Total Lanes per Node | 7 |
| Min Green Time | 5.0 s |
| Max Green Time | 30.0 s |
| SUMO Config — node_0, 2, 4 | `osm_client1.sumocfg` |
| SUMO Config — node_1, 3, 5 | `osm_client2.sumocfg` |

---

## 3. Core Architecture: PER and GAT

AdaptFlow's DQN agent (`AdaptFlowDQN`) combines **Prioritized Experience Replay (PER)** for sample efficiency and a **Spatio-Temporal Graph Attention Network (GAT)** for state encoding — together enabling the agent to learn effective signal-control policies from complex, high-dimensional traffic observations.

### 3.1 Prioritized Experience Replay (PER)

Standard DQN samples past experiences uniformly at random. PER assigns a **priority** to each transition proportional to its TD-error, so the agent replays the transitions it has learned least from most often.

| Parameter | Value | Purpose |
|---|---|---|
| `alpha` (α) | 0.6 | Prioritization exponent — controls how strongly priority affects sampling |
| `beta` (β) | 0.4 → 1.0 | IS-correction annealing (+0.001/step) to de-bias non-uniform sampling |
| `epsilon` (ε) | 1e-5 | Prevents zero-priority transitions |
| `memory_size` | 5,000 | SumTree buffer capacity |
| `batch_size` | 64 | Transitions sampled per replay step |

**How it operates in this run:**
1. Each transition `(state_seq, adj, action, reward, next_state_seq, next_adj, done)` is stored in a **SumTree**; new transitions start at max priority.
2. Sampling is stratified across `batch_size` segments of the total priority range.
3. IS weights `w_i = (N·P(i))^(−β)` correct for sampling bias.
4. After computing Double-DQN loss, priorities are updated: `p_i = (|δ_i| + ε)^α`.

With 1,000 steps × 6 nodes × 5 rounds = 30,000 total transitions, PER ensures rare critical events (congestion spikes, phase-transitions) are replayed disproportionately, contributing to the −12.8% loss reduction achieved here.

### 3.2 Graph Attention Network (GAT) — Spatio-Temporal Encoder

```
Input: x_seq [batch, time_steps=4, num_nodes, state_size=12]
  │
  ├── For each time step t:
  │     └── CoLight Spatial GAT (Multi-head, nheads=4)
  │           ├── 4 × Linear(12→32) + LeakyReLU attention
  │           ├── Concatenate: [4×32 = 128]
  │           └── Output head: Linear(128→32)
  │           → h_t: [batch, num_nodes, 32]
  │
  ├── Stack across time → [batch, 4, num_nodes, 32]
  │
  └── Temporal Attention Layer
        ├── Linear(32→32) + tanh
        ├── Learnable query vector q ∈ R^32
        ├── Softmax over time steps
        └── Weighted sum → [batch, num_nodes, 32]

DQN Head (focal node h[:, 0, :]):
  Linear(32→128) → ReLU → Linear(128→128) → ReLU
  → Linear(128→64) → ReLU → Linear(64→4)  [Q-values for 4 actions]
```

**Spatial attention** (per node pair i,j): `α_ij = softmax(LeakyReLU(aᵀ·[Wh_i ∥ Wh_j]))` — learns to weight upstream neighbor congestion automatically.

**Temporal attention**: a learnable query `q` scores each of the last 4 timesteps, allowing the model to up-weight recent observations during rapid traffic changes and attend to older states during stable phases.

### 3.3 PER + GAT Synergy

The combination is especially powerful here: the GAT produces Q-estimates that account for spatio-temporal context; when those estimates are wrong (high TD-error), PER ensures those exact spatio-temporal contexts are replayed until corrected. This is why **85% of the total loss reduction occurred in Round 1→2** — the first real reclustering gave PER a fresh set of high-error states to correct.

---

## 4. Global Metrics Across Rounds

### 4.1 Aggregate Trends

| Round | Total Reward (6 nodes) | Avg Waiting Time (s) | Avg Queue Length | Avg TP Ratio | Avg Loss |
|:-----:|:----------------------:|:--------------------:|:----------------:|:------------:|:--------:|
| 1 | 5825.75 | 1.9754 | 0.4108 | 0.7571 | 0.5452 |
| 2 | 5770.20 | 2.1158 | 0.4417 | 0.7572 | 0.4857 |
| 3 | 5712.45 | 2.0898 | 0.4840 | 0.7574 | 0.4849 |
| 4 | 5712.45 | 2.0898 | 0.4840 | 0.7574 | 0.4799 |
| 5 | 5712.45 | 2.0898 | 0.4840 | 0.7574 | 0.4755 |

**Observations:**

- **Total reward** fell by −1.95% (5825.75 → 5712.45) over the first 3 rounds, then flatlined. The decrease reflects the High-Queue archetype settling into a higher-queue operating regime (avg queue 0.263 → 0.705) after reclustering.
- **Throughput ratio** is essentially constant (0.7571 → 0.7574, +0.04pp) — AdaptFlow does not sacrifice vehicle throughput during policy learning.
- **Avg wait time** shows a transient +0.14s spike in Round 2 (reclustering adjustment), then stabilises at 2.090s from Round 3 onward.
- **Loss** decreased every single round — even Rounds 3–5 where behavioral metrics are frozen — confirming continued Q-value refinement. The R1→R2 drop (−10.9%) accounts for **85%** of the total 5-round reduction.
- **Rounds 3–5 are fully converged** on all behavioral metrics (reward, queue, wait, TP identical).

### 4.2 Global Departures & Arrivals

> **Note:** All figures below are **summed across all 6 nodes**. Each individual node departs ~624–625 vehicles (see §5 for per-node breakdown).

| Round | Total Departed (6 nodes) | Per Node (avg) | Total Arrived (6 nodes) | Global TP Ratio | In-Network |
|:-----:|:------------------------:|:--------------:|:-----------------------:|:---------------:|:----------:|
| 1 | 3,747 | 624.5 | 2,837 | 0.7571 | 910 (24.3%) |
| 2 | 3,748 | 624.7 | 2,838 | 0.7572 | 910 (24.3%) |
| 3 | 3,747 | 624.5 | 2,838 | 0.7574 | 909 (24.3%) |
| 4 | 3,747 | 624.5 | 2,838 | 0.7574 | 909 (24.3%) |
| 5 | 3,747 | 624.5 | 2,838 | 0.7574 | 909 (24.3%) |

**Per-node breakdown (consistent across all rounds):**

| Node Group | Config | Departed | Arrived (R2–5) | TP Ratio |
|:----------:|:------:|:--------:|:--------------:|:--------:|
| node_0, node_2, node_4 | osm_client1 | **625** each | **476** each | 0.7616 |
| node_1, node_3, node_5 | osm_client2 | **624** each | **470** each | 0.7532 |

- The 1-vehicle difference between configs (625 vs 624) is due to the route file in `osm_client2.sumocfg` generating one fewer departure than `osm_client1.sumocfg` at the same step count.
- ~909 vehicles per round remain in-network at step 1,000. Extending to ~1,300 steps would likely push global TP above 85%.

### 4.3 Convergence Timeline

| Round | Event |
|:-----:|:------|
| 1 | Initial exploration — 6 unique behavioral profiles, geographic clustering (0-1-2 vs 3-4-5) |
| 2 | **Behavioral reclustering**: node_1 → High-Queue cluster, node_4 → Low-Queue cluster. Low-Queue archetype fully converged in this round. |
| 3 | **Full convergence**: node_1 completes transition to High-Queue archetype. Both groups stabilised. Label flip artifact (all-6-node "transition" recorded). |
| 4 | Steady state — zero transitions, identical behavioral metrics |
| 5 | Steady state — label flip artifact again; loss still declining |

---

## 5. Per-Node Performance — All Rounds

### 5.1 Round 1 — Initial Exploration

| Node | Cluster | Reward | Avg Wait (s) | Loss | Departed | Arrived | TP Ratio | Avg Queue | Max Q |
|:----:|:-------:|:------:|:------------:|:----:|:--------:|:-------:|:--------:|:---------:|:-----:|
| node_0 | cluster_0 | 975.35 | 1.9984 | 0.5290 | 625 | 475 | 0.7600 | 0.330 | 3 |
| node_1 | cluster_0 | 966.54 | 1.9311 | 0.5072 | 624 | 469 | 0.7516 | 0.422 | 5 |
| node_2 | cluster_0 | 967.27 | 2.0080 | **0.6271** | 625 | 474 | 0.7584 | 0.410 | 4 |
| node_3 | cluster_1 | 970.79 | 2.0288 | 0.5757 | 624 | 471 | 0.7548 | 0.448 | 6 |
| node_4 | cluster_1 | **981.01** | 2.1040 | 0.5311 | 625 | 476 | 0.7616 | 0.340 | 4 |
| node_5 | cluster_1 | 964.80 | **1.7821** | 0.5009 | 624 | 472 | 0.7564 | 0.515 | **6** |

**Round 1 highlights:**
- **Initial clustering is geographic, not behavioral**: nodes 0-1-2 (osm_client1/2 mix) vs 3-4-5 (osm_client1/2 mix). node_1 (wait 1.93, queue 0.42) was mis-grouped with node_0/2 despite its High-Queue behavior profile.
- **node_2** had the highest initial loss (0.6271) — indicating the greatest underfit — and went on to achieve the largest total loss reduction (−22.8%).
- **node_5** combined the lowest wait (1.782s) with the highest queue (0.515, max 6), reflecting an aggressive GST bias that processes large flows quickly but accumulates burst queues.
- **node_4** had the highest Round 1 reward (981.01) despite an elevated wait (2.104s), because its queue (0.340) was relatively low.
- Throughput was competitive across all nodes, with just a 1.0pp spread (75.2%–76.2%), confirming the federated initialization provided a reasonable starting policy.

### 5.2 Round 2 — After First Reclustering

| Node | Cluster | Reward | Avg Wait (s) | Loss | Departed | Arrived | TP Ratio | Avg Queue | Max Q |
|:----:|:-------:|:------:|:------------:|:----:|:--------:|:-------:|:--------:|:---------:|:-----:|
| node_0 | cluster_0 | 988.06 | 1.9776 | 0.4956 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_1 | cluster_1 | 973.83 | 2.3584 | 0.4953 | 625 | 470 | 0.7520 | 0.451 | 5 |
| node_2 | cluster_0 | 988.06 | 1.9776 | 0.4848 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_3 | cluster_1 | 916.09 | 2.2019 | 0.4782 | 624 | 470 | 0.7532 | 0.705 | 6 |
| node_4 | cluster_0 | 988.06 | 1.9776 | 0.4855 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_5 | cluster_1 | 916.09 | 2.2019 | 0.4750 | 624 | 470 | 0.7532 | 0.705 | 6 |

**Round 2 highlights:**
- **node_0, 2, 4 fully converged**: identical reward (988.06), wait (1.9776s), queue (0.263), and TP (0.7616) — all three nodes share the same architectural policy after receiving the same cluster-aggregate model.
- **node_1 is mid-transition**: wait jumped 1.931→2.358s (+22%), queue 0.422→0.451, as it begins adopting the High-Queue cluster's GST bias but hasn't fully committed yet.
- **node_3 and node_5 snap immediately** to the shared High-Queue policy: queue 0.448→0.705 (+57%) and 0.515→0.705 (+37%) in a single round.
- **2 true transitions occurred**: node_1 moved cluster_0→cluster_1; node_4 moved cluster_1→cluster_0.
- Largest single-round loss drop: node_3, **0.5757 → 0.4782** (−0.0975, −16.9%).

### 5.3 Round 3 — Full Convergence

| Node | Cluster* | Reward | Avg Wait (s) | Loss | Departed | Arrived | TP Ratio | Avg Queue | Max Q |
|:----:|:--------:|:------:|:------------:|:----:|:--------:|:-------:|:--------:|:---------:|:-----:|
| node_0 | High-label | 988.06 | 1.9776 | 0.4896 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_1 | Low-label | 916.09 | 2.2019 | 0.4734 | 624 | 470 | 0.7532 | 0.705 | 6 |
| node_2 | High-label | 988.06 | 1.9776 | 0.4834 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_3 | Low-label | 916.09 | 2.2019 | 0.4797 | 624 | 470 | 0.7532 | 0.705 | 6 |
| node_4 | High-label | 988.06 | 1.9776 | 0.4970 | 625 | 476 | 0.7616 | 0.263 | 3 |
| node_5 | Low-label | 916.09 | 2.2019 | 0.4864 | 624 | 470 | 0.7532 | 0.705 | 6 |

> *All 6 nodes recorded as "transitioning" in Round 3 — this is a **cluster label-flip artifact** (labels 0↔1 swapped by the spectral clustering algorithm). The actual groups (0-2-4 vs 1-3-5) did not change.*

**Round 3 highlights:**
- node_1 fully joined the High-Queue archetype: wait 2.2019s, queue 0.705 — identical to nodes 3 and 5.
- Zero behavioral variance within each group. Both archetypes are now locked.
- Loss continues declining despite fixed behavior — particularly for node_1 (0.4953→0.4734, −4.5%) and node_3 (0.4782→0.4797, marginal fluctuation).

### 5.4 Rounds 4 & 5 — Steady State

| Node | Cluster R4 | Cluster R5 | Reward | Avg Wait (s) | Loss R4 | Loss R5 | TP Ratio | Avg Queue | Max Q |
|:----:|:----------:|:----------:|:------:|:------------:|:-------:|:-------:|:--------:|:---------:|:-----:|
| node_0 | cluster_1 | cluster_0 | 988.06 | 1.9776 | 0.5001 | 0.4934 | 0.7616 | 0.263 | 3 |
| node_1 | cluster_0 | cluster_1 | 916.09 | 2.2019 | 0.4752 | **0.4537** | 0.7532 | 0.705 | 6 |
| node_2 | cluster_1 | cluster_0 | 988.06 | 1.9776 | **0.4519** | 0.4839 | 0.7616 | 0.263 | 3 |
| node_3 | cluster_0 | cluster_1 | 916.09 | 2.2019 | 0.4671 | 0.4582 | 0.7532 | 0.705 | 6 |
| node_4 | cluster_1 | cluster_0 | 988.06 | 1.9776 | 0.5035 | 0.4775 | 0.7616 | 0.263 | 3 |
| node_5 | cluster_0 | cluster_1 | 916.09 | 2.2019 | 0.4816 | 0.4867 | 0.7532 | 0.705 | 6 |

- Round 5 shows another full-label-flip (same cosmetic artifact as Round 3). Round 4 had **zero transitions** — the only round with label stability.
- **node_2** achieved the run's lowest absolute loss in Round 4: **0.4519**.
- **node_1** achieved the run's lowest absolute loss in Round 5: **0.4537** — High-Queue nodes showed stronger continued learning (avg Δ R4→R5: −0.0098) vs Low-Queue nodes (+0.0003, slight oscillation).

---

## 6. Loss Convergence — Per Node

| Node | R1 | R2 | R3 | R4 | R5 | Δ (R1→R5) | % Change |
|:----:|:--:|:--:|:--:|:--:|:--:|:----------:|:--------:|
| node_0 | 0.5290 | 0.4956 | 0.4896 | 0.5001 | 0.4934 | −0.0356 | −6.7% |
| node_1 | 0.5072 | 0.4953 | 0.4734 | 0.4752 | 0.4537 | −0.0535 | −10.6% |
| node_2 | **0.6271** | 0.4848 | 0.4834 | **0.4519** | 0.4839 | **−0.1432** | **−22.8%** |
| node_3 | 0.5757 | 0.4782 | 0.4797 | 0.4671 | 0.4582 | −0.1175 | −20.4% |
| node_4 | 0.5311 | 0.4855 | 0.4970 | 0.5035 | 0.4775 | −0.0537 | −10.1% |
| node_5 | 0.5009 | 0.4750 | 0.4864 | 0.4816 | 0.4867 | −0.0142 | −2.8% |
| **Mean** | **0.5452** | **0.4857** | **0.4849** | **0.4799** | **0.4755** | **−0.0696** | **−12.8%** |

**Key observations:**

- **node_2** has the largest loss reduction (−22.8%), driven by its unusually high initial loss (0.6271). The Round 1→2 single-round drop for node_2 was −0.1423 — nearly as large as the total change for most other nodes.
- **node_5** has the smallest reduction (−2.8%) — it started from a low baseline (0.5009) with a near-optimal High-Queue policy already in place.
- **node_0** shows mild oscillation (R3: 0.4896 → R4: 0.5001 → R5: 0.4934) consistent with PER β-annealing and federated aggregation weight perturbation.
- The **R1→R2 drop of −10.9%** accounts for 85% of the total 5-round reduction — confirming that the bulk of learning happens following the first behavioral reclustering.
- **All nodes show net decreasing loss**, confirming Double-DQN + PER is functional with no catastrophic forgetting.

---

## 7. Dynamic Clustering Analysis

### 7.1 Cluster Assignments Per Round

| Round | Low-Queue Group | High-Queue Group | True Transitions | Notes |
|:-----:|:---------------:|:----------------:|:----------------:|:-----:|
| 1 | node_0, 1, 2 (label 0) | node_3, 4, 5 (label 1) | — | Geographic split (initial) |
| 2 | node_0, 2, 4 (label 0) | node_1, 3, 5 (label 1) | node_1: 0→1, node_4: 1→0 | **Behavioral reclustering** |
| 3 | node_0, 2, 4 (label 1) | node_1, 3, 5 (label 0) | None | Label-flip artifact |
| 4 | node_0, 2, 4 (label 1) | node_1, 3, 5 (label 0) | None | Stable |
| 5 | node_0, 2, 4 (label 0) | node_1, 3, 5 (label 1) | None | Label-flip artifact |

- The **only true behavioral re-assignment** was in Round 2: node_1 moved to the High-Queue cluster and node_4 moved to the Low-Queue cluster — correcting the initial misclassification caused by geographic grouping.
- Rounds 3 and 5 each record all-6-node "transitions" in the JSON — these are **spectral clustering label-flip artifacts** (the algorithm re-assigns integer labels 0/1 arbitrarily each run). The groups themselves did not change.
- **Effective cluster stability was achieved at Round 2** — one round after the initial training episode.

### 7.2 Cluster Characteristics (Converged State, Rounds 2–5)

| Metric | Low-Queue Cluster (node_0, 2, 4) | High-Queue Cluster (node_1, 3, 5) | Ratio (LQ/HQ) |
|--------|:---------------------------------:|:----------------------------------:|:-------------:|
| Avg TP Ratio | 0.7616 | 0.7532 | 1.011× |
| Avg Waiting Time (s) | 1.9776 | 2.2019 | 0.898× (better) |
| Avg Queue Length | 0.263 | 0.705 | 0.373× (better) |
| Max Queue | 3 | 6 | 0.5× (better) |
| Reward (per node) | 988.06 | 916.09 | 1.079× (better) |
| Vehicles Departed | 625 | 624 | — |
| Vehicles Arrived | 476 | 470 | +6 |

The two clusters differ primarily in **queue accumulation** (2.68× higher in High-Queue), while throughput difference is negligible (+6 vehicles, +1.3%). The reward gap of **71.97 per node** is driven entirely by queue-length penalties accumulated over 1,000 steps.

### 7.3 Cluster-Level Fingerprints

Fingerprint format: `[avg_wait, avg_queue, throughput_ratio, max_queue, unused_dim, node_flag]`

| Node | Round 1 Fingerprint | Round 2–5 Fingerprint (stable) | Cluster |
|:----:|:--------------------|:-------------------------------|:-------:|
| node_0 | [1.998, 0.330, 0.760, 3.0, 0.0, **1.0**] | [1.978, 0.263, 0.762, 3.0, 0.0, **1.0**] | Low-Q |
| node_1 | [1.931, 0.422, 0.752, 5.0, 0.0, **0.5**] | [2.202, 0.705, 0.753, 6.0, 0.0, **0.5**] | High-Q |
| node_2 | [2.008, 0.410, 0.758, 4.0, 0.0, 0.0] | [1.978, 0.263, 0.762, 3.0, 0.0, 0.0] | Low-Q |
| node_3 | [2.029, 0.448, 0.755, 6.0, 0.0, 0.0] | [2.202, 0.705, 0.753, 6.0, 0.0, 0.0] | High-Q |
| node_4 | [2.104, 0.340, 0.762, 4.0, 0.0, 0.0] | [1.978, 0.263, 0.762, 3.0, 0.0, 0.0] | Low-Q |
| node_5 | [1.782, 0.515, 0.756, 6.0, 0.0, 0.0] | [2.202, 0.705, 0.753, 6.0, 0.0, 0.0] | High-Q |

- Round 1 has **6 unique fingerprints**; from Round 2 they collapse into **2 templates** (with only the node_flag dimension separating node_0 and node_1 from their cluster peers).
- The Low-Queue fingerprint **improved** from R1→R2: avg queue 0.330→0.263 (−20.3%), wait −0.02s, max_queue 4→3.
- The High-Queue fingerprint **degraded** for node_1 specifically as it adopted the cluster's aggressive GST bias: queue 0.422→0.705 (+67%), wait +14%.
- The 5th dimension is always 0.0 (appears unused or represents zero congested lanes at these intersections). The `node_flag` (6th dim) is intrinsic to each node and constant across rounds.

---

## 8. Cosine Similarity Matrices

Model weight similarity between nodes (cosine similarity of flattened weight vectors), confirming the two-archetype structure.

### 8.1 Round 1 (Pre-convergence)

|  | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--|:------:|:------:|:------:|:------:|:------:|:------:|
| **node_0** | 1.000 | 0.961 | 0.958 | 0.930 | 0.959 | 0.920 |
| **node_1** | 0.961 | 1.000 | 0.991 | 0.995 | 0.989 | 0.992 |
| **node_2** | 0.958 | 0.991 | 1.000 | 0.989 | **1.000** | 0.984 |
| **node_3** | 0.930 | 0.995 | 0.989 | 1.000 | 0.987 | **0.999** |
| **node_4** | 0.959 | 0.989 | **1.000** | 0.987 | 1.000 | 0.980 |
| **node_5** | 0.920 | 0.992 | 0.984 | **0.999** | 0.980 | 1.000 |

- **node_2 ≡ node_4** (sim = 1.000) even at Round 1: both use osm_client1 and had near-identical initial states.
- **node_3 ≈ node_5** (sim = 0.999): both osm_client2.
- node_0 is a structural outlier — lowest similarity to all other nodes (0.920–0.961), driven by its unique `node_flag=1.0`.
- Two-group structure was already latent in the weights before reclustering ran, validating spectral clustering's ability to detect behavioral similarity.

### 8.2 Round 2 (Post-reclustering)

|  | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--|:------:|:------:|:------:|:------:|:------:|:------:|
| **node_0** | 1.000 | 0.973 | 0.965 | 0.936 | 0.965 | 0.936 |
| **node_1** | 0.973 | 1.000 | 0.984 | 0.992 | 0.984 | 0.992 |
| **node_2** | 0.965 | 0.984 | **1.000** | 0.970 | **1.000** | 0.970 |
| **node_3** | 0.936 | 0.992 | 0.970 | 1.000 | 0.970 | **1.000** |
| **node_4** | 0.965 | 0.984 | **1.000** | 0.970 | 1.000 | 0.970 |
| **node_5** | 0.936 | 0.992 | 0.970 | **1.000** | 0.970 | 1.000 |

- **node_2 ≡ node_4** (1.000) and **node_3 ≡ node_5** (1.000): perfect pairs now confirmed in both clusters.
- node_0's distinctness from node_2/4 (0.965) and node_1's from node_3/5 (0.992) are caused by their non-zero `node_flag` dimensions.
- **Inter-cluster similarity** settled at 0.936–0.984 — meaningfully below intra-cluster (0.965–1.000).

### 8.3 Rounds 3–5 (Identical Converged Matrix)

|  | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--|:------:|:------:|:------:|:------:|:------:|:------:|
| **node_0** | 1.000 | 0.953 | 0.965 | 0.936 | 0.965 | 0.936 |
| **node_1** | 0.953 | 1.000 | 0.967 | 0.997 | 0.967 | 0.997 |
| **node_2** | 0.965 | 0.967 | **1.000** | 0.970 | **1.000** | 0.970 |
| **node_3** | 0.936 | 0.997 | 0.970 | 1.000 | 0.970 | **1.000** |
| **node_4** | 0.965 | 0.967 | **1.000** | 0.970 | 1.000 | 0.970 |
| **node_5** | 0.936 | 0.997 | 0.970 | **1.000** | 0.970 | 1.000 |

- Identical for Rounds 3, 4, and 5 — confirming true weight-space convergence.
- node_0 ↔ node_1 **decreased** from 0.973 (Round 2) to 0.953 (Round 3+): continued cluster-specific federated aggregation is slowly diverging the two archetypes.
- node_1 ↔ node_3/5 increased from 0.992 (Round 2) to 0.997 (Round 3+): node_1 aligned more tightly with its cluster peers over time.

### 8.4 Similarity Evolution Summary

| Pair | R1 | R2 | R3–R5 | Trend |
|:----:|:--:|:--:|:-----:|:-----:|
| node_0 ↔ node_2 (intra Low-Q) | 0.958 | 0.965 | 0.965 | ↑ then stable |
| node_0 ↔ node_4 (intra Low-Q) | 0.959 | 0.965 | 0.965 | ↑ then stable |
| **node_2 ↔ node_4** (intra Low-Q) | **1.000** | **1.000** | **1.000** | Perfect from R1 |
| node_1 ↔ node_3 (intra High-Q) | 0.995 | 0.992 | 0.997 | ↑ converging |
| **node_3 ↔ node_5** (intra High-Q) | **0.999** | **1.000** | **1.000** | Perfect from R2 |
| node_0 ↔ node_1 (inter-cluster) | 0.961 | 0.973 | 0.953 | ↓ diverging |
| node_0 ↔ node_3 (inter-cluster) | 0.930 | 0.936 | 0.936 | → stable |
| node_2 ↔ node_3 (inter-cluster) | 0.989 | 0.970 | 0.970 | ↓ diverged |

---

## 9. Green Signal Time (GST) Analysis

### 9.1 GST Per Edge — Full Round-by-Round Breakdown

| Node | Round | Edge 203598795#3 | Edge 608989233#2 | Edge 39452784#4 | Strategy Profile |
|:----:|:-----:|:----------------:|:----------------:|:---------------:|:----------------:|
| node_0 | 1 | 0.520 | 0.540 | 0.313 | Balanced |
| node_0 | 2 | 0.464 | 0.482 | 0.272 | Tightening |
| node_0 | 3–5 | **0.404** | **0.420** | **0.228** | **Converged balanced** |
| node_1 | 1 | 0.405 | 1.032 | 0.302 | Partial bias |
| node_1 | 2 | 0.432 | 1.212 | 0.166 | Growing bias |
| node_1 | 3–5 | **1.172** | **1.608** | **0.040** | **Converged priority** |
| node_2 | 1 | 0.865 | 0.722 | 0.103 | Biased, exploring |
| node_2 | 2 | 0.638 | 0.574 | 0.166 | Converging |
| node_2 | 3–5 | **0.404** | **0.420** | **0.228** | **Converged balanced** |
| node_3 | 1 | 0.619 | 1.179 | 0.012 | Strong priority |
| node_3 | 2 | 0.898 | 1.398 | 0.026 | Stabilising |
| node_3 | 3–5 | **1.172** | **1.608** | **0.040** | **Converged priority** |
| node_4 | 1 | 0.567 | 0.734 | 0.079 | Moderate bias |
| node_4 | 2 | 0.488 | 0.580 | 0.154 | Converging |
| node_4 | 3–5 | **0.404** | **0.420** | **0.228** | **Converged balanced** |
| node_5 | 1 | 0.754 | 1.321 | 0.032 | Strong priority |
| node_5 | 2 | 0.966 | 1.470 | 0.036 | Stabilising |
| node_5 | 3–5 | **1.172** | **1.608** | **0.040** | **Converged priority** |

### 9.2 Two Emergent Signal Timing Strategies

**Strategy A — Balanced GST** (Low-Queue nodes 0, 2, 4):

| Edge | Green Time (s) | Share |
|:----:|:--------------:|:-----:|
| 203598795#3 | 0.404 | 38.7% |
| 608989233#2 | 0.420 | 40.2% |
| 39452784#4 | 0.228 | **21.8%** |

Effect: controlled max queue of 3, lower per-step reward variance, TP ratio 76.2% (476/625).

**Strategy B — Priority GST** (High-Queue nodes 1, 3, 5):

| Edge | Green Time (s) | Share |
|:----:|:--------------:|:-----:|
| 203598795#3 | 1.172 | 41.0% |
| 608989233#2 | 1.608 | 56.3% |
| 39452784#4 | 0.040 | **1.4%** |

Effect: burst queues up to 6 on edge #4 (near-total starvation at 0.04s), but rapid clearance of the two main approaches. TP ratio 75.3% (470/624).

> **GST key insight:** Edge `39452784#4` (length ~2.25m, 1 lane — the shortest approach) receives a proportionate 21.8% green time under Strategy A vs only 1.4% under Strategy B. This near-total starvation under Strategy B is the direct cause of max-queue-6 bursts, while Strategy A caps queues at 3.

### 9.3 GST Evolution of Node_2 (Most Dramatic Transition)

Node_2 underwent the largest GST restructuring, matching its cluster reassignment from an intermediate mixed policy to the Low-Queue balanced strategy:

| Round | Edge 203598795#3 | Edge 608989233#2 | Edge 39452784#4 |
|:-----:|:----------------:|:----------------:|:---------------:|
| 1 | 0.865 | 0.722 | 0.103 |
| 2 | 0.638 | 0.574 | 0.166 |
| 3–5 | **0.404** | **0.420** | **0.228** |

Edge #3's allocation dropped by 53%, edge #4's grew by 121% — mirroring the convergence to the (node_0=node_2=node_4) balanced archetype.

---

## 10. Reward Decomposition

### 10.1 Per-Archetype Breakdown (Converged, Rounds 2–5)

| Component | Low-Queue (node_0, 2, 4) | High-Queue (node_1, 3, 5) | Difference |
|-----------|:------------------------:|:-------------------------:|:----------:|
| Reward (per node) | **988.06** | **916.09** | **+71.97** |
| Vehicles Arrived | 476 | 470 | +6 |
| Vehicles Departed | 625 | 624 | +1 |
| TP Ratio | 0.7616 | 0.7532 | +0.84pp |
| Avg Queue Length | 0.263 | 0.705 | −0.442 |
| Max Queue | 3 | 6 | −3 |
| Avg Waiting Time (s) | 1.9776 | 2.2019 | −0.224s |

The **reward gap of 71.97** per node is almost entirely explained by queue penalties: 0.442 extra queue units × 1,000 steps = ~442 queue-step penalty units for the High-Queue archetype. The +6 throughput advantage for Low-Queue cannot be offset by the High-Queue nodes' marginal TP advantage.

**Conclusion: Strategy A (Balanced GST) is the dominant policy under this reward function.**

### 10.2 Node-Level R1 → R5 Changes

| Node | R1 Reward | R5 Reward | Δ Reward | R1 Queue | R5 Queue | Δ Queue |
|:----:|:---------:|:---------:|:--------:|:--------:|:--------:|:-------:|
| node_0 | 975.35 | 988.06 | **+12.71** | 0.330 | 0.263 | −0.067 |
| node_1 | 966.54 | 916.09 | **−50.45** | 0.422 | 0.705 | +0.283 |
| node_2 | 967.27 | 988.06 | **+20.79** | 0.410 | 0.263 | −0.147 |
| node_3 | 970.79 | 916.09 | **−54.70** | 0.448 | 0.705 | +0.257 |
| node_4 | 981.01 | 988.06 | **+7.05** | 0.340 | 0.263 | −0.077 |
| node_5 | 964.80 | 916.09 | **−48.71** | 0.515 | 0.705 | +0.190 |

- **Low-Queue nodes (0, 2, 4)** all improved: avg gain of +13.5 reward units, avg queue decrease of −0.097.
- **High-Queue nodes (1, 3, 5)** all degraded: avg loss of −51.3 reward units, avg queue increase of +0.243.
- Net global reward change: −113.3 across 6 nodes — the price paid for behavioral specialization into two distinct archetypes.

---

## 11. Convergence & Stability Diagnosis

### 11.1 What Converged Well

✅ **Loss** — All 6 nodes reduced loss (−2.8% to −22.8%), avg **−12.8%**; no catastrophic forgetting observed

✅ **Behavioral archetypes** — Two stable archetypes correctly identified and held from Round 2

✅ **Spectral reclustering** — Corrected the initial geographic misclassification in exactly 1 round

✅ **Intra-cluster weight convergence** — node_2=node_4 and node_3=node_5 reached perfect similarity (1.000) from Round 2

✅ **Throughput stability** — 75.71%→75.74% across 5 rounds; policy learning did not sacrifice vehicle flow

✅ **GST convergence** — Both signal timing strategies fully crystallized by Round 3

✅ **Loss-without-behavior** — Rounds 3–5 show continued loss reduction even after behavioral freezing, validating ongoing Q-value refinement from PER

### 11.2 Areas of Concern

⚠️ **High-Queue reward regression** — Nodes 1, 3, 5 lose avg 51.3 reward units after reclustering. The cluster-aggregated model leads to queue growth without proportional throughput gain. The reward function appears to favor the balanced-GST strategy strongly.

⚠️ **24.3% vehicles remain in-network** — 909 vehicles per round do not complete trips within 1,000 steps. Extending to ~1,300 steps would likely clear this backlog and push TP above 85%.

⚠️ **Identical weights within same-config pairs** — node_2/node_4 and node_3/node_5 share the same SUMO config and receive identical federated models; by Round 2 they have perfectly identical weights (cosine sim = 1.000). These nodes are not learning intersection-specific features — only cluster-level policies.

⚠️ **Cluster label oscillation** — Rounds 3 and 5 record all-6-node "transitions" that are cosmetic label swaps. Downstream analytics treating these as real transitions will be misled. Fix: always assign label_0 to the lower-avg_wait cluster.

⚠️ **No improvement after Round 3** — Rounds 4–5 add only marginal continued loss reduction at full SUMO-GUI compute cost. Each extra round is ~15 min of simulation time with zero behavioral return.

⚠️ **Edge starvation in High-Queue strategy** — edge `39452784#4` receives only 0.04s green time under Strategy B. If this edge carries meaningful demand, this near-zero allocation may be causing hidden arrival failures not reflected in the current metrics.

### 11.3 Recommendations

1. **Reduce training to 3 rounds** — no behavioral improvement occurs beyond Round 3 for this network/vehicle configuration.
2. **Increase steps to 1,200–1,500** to push global TP above 85% and reduce in-network vehicles below 15%.
3. **Add edge-level features to the fingerprint** (e.g., edge lengths, lane counts, or encoded edge IDs) to differentiate nodes sharing the same SUMO config and prevent weight collapse within pairs.
4. **Implement canonical cluster labeling** — pin label_0 to the cluster with lower avg_wait per round to eliminate the cosmetic oscillation.
5. **Enforce minimum green time on all edges** — a minimum GST floor of 0.1–0.2s for edge `39452784#4` (currently starved to 0.04s under Strategy B) could improve the High-Queue archetype's reward by reducing burst queuing without sacrificing throughput.
6. **Investigate node_flag semantics** — the per-node flag (1.0 for node_0, 0.5 for node_1, 0.0 for others) persists through training and contributes to the slight weight non-equivalence of node_0 and node_1 relative to their cluster peers. If this is an artifact, removing or normalizing it could enable true cluster-wide weight equivalence.

---

## 12. Summary Statistics

| Metric | Round 1 | Round 5 | Change |
|--------|:-------:|:-------:|:------:|
| Global Reward (6 nodes) | 5825.75 | 5712.45 | −1.95% |
| Avg Waiting Time | 1.975s | 2.090s | +5.8% |
| Avg Queue Length | 0.411 | 0.484 | +17.8% |
| Throughput Ratio (global) | 0.7571 | 0.7574 | +0.04% |
| Avg Loss (all nodes) | 0.5452 | 0.4755 | **−12.8%** |
| Best Node Reward | 981.01 (node_4, R1) | 988.06 (all LQ, R5) | +0.71% |
| Worst Node Reward | 964.80 (node_5, R1) | 916.09 (all HQ, R5) | −5.1% |
| Highest Initial Loss | 0.6271 (node_2, R1) | — | — |
| Lowest Final Loss | — | 0.4537 (node_1, R5) | −14.0% |
| Total Vehicles Arrived (per round) | 2,837 | 2,838 | +1 |
| In-Network Vehicles | 910 | 909 | −1 |
| Correct Cluster Split (behavioral) | ❌ Geographic | ✅ Behavioral (R2) | ✅ |
| Intra-Cluster Max Similarity | 1.000 (node_2=node_4) | 1.000 | → |
| Inter-Cluster Min Similarity | 0.930 (node_0↔node_5) | 0.936 (node_0↔node_3/5) | ↑ |

---

*Report generated from: `round_{1..5}_summary.json`, `training_cluster_history.json`, `cluster_history.json`*
*Run date: 2026-04-12 | Command: `python train_adaptflow.py --gui --steps 1000 --rounds 5`*
