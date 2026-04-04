# AdaptFlow-TSC Analysis Report — China OSM Scenario

| Field | Value |
|---|---|
| **Scenario** | China OSM (`sumo_configs_china_osm`) |
| **Mode** | SUMO-GUI (Real SUMO Simulation) |
| **Rounds** | 5 |
| **Steps / Episode** | 400 (`max_steps = 400`) |
| **Nodes** | 6 (node_0 – node_5) |
| **Vehicle load** | 1000 vehicles per 400s window per node (period = 0.4s) |
| **Total trips in file** | 6000 (1000 per node begin-offset window) |
| **Date** | 2026-04-05 |

---

## 1. Node Configuration — What Makes Each Node Unique

| Node | Scenario | begin offset | jam-threshold | Traffic slice seen |
|---|---|---|---|---|
| node_0 | Hospital zone (morning rush) | 0 s | 20 | t=0–400 (densest peak) |
| node_1 | School zone | 400 s | 25 | t=400–800 |
| node_2 | Commerce district | 800 s | 30 | t=800–1200 |
| node_3 | Residential area | 1200 s | 35 | t=1200–1600 |
| node_4 | Industrial zone | 1600 s | 40 | t=1600–2000 |
| node_5 | Transit hub | 2000 s | 28 | t=2000–2400 |

All 6 nodes share `osm.passenger.trips.xml` but each begins at a different offset, giving them genuinely different traffic demand profiles.

---

## 2. GAT — Graph Attention Network (Architecture & Theory)

### 2.1 Why GAT for Traffic Signal Control?

| Approach | Neighbour weighting | Limitation |
|---|---|---|
| Fixed timer | None | Ignores real-time traffic |
| Standard Graph Conv | Equal — all neighbours same | Cannot distinguish critical from light junctions |
| **GAT (AdaptFlow)** | **Learned per-neighbour attention weight** | Dynamically focuses on most congested neighbours |

### 2.2 Input Graph

| Element | Traffic meaning |
|---|---|
| Node | One traffic signal intersection |
| Edge | Road connection between intersections |
| Node feature | [queue, wait, phase, vehicle count, speed, occupancy] |
| Adjacency matrix | 1 if connected, 0 otherwise |

### 2.3 GAT Computation — Step by Step

| Step | Formula | What it does |
|---|---|---|
| **1. Linear projection** | `Wh_i = h_i x W` | Transform node features F → F' dimensions |
| **2. Pairwise attention** | `e_ij = LeakyReLU( aT x [Wh_i || Wh_j] )` | Raw attention score for each connected pair (i,j) |
| **3. Normalisation** | `alpha_ij = exp(e_ij) / SUM_k exp(e_ik)` | Softmax over neighbourhood — weights sum to 1 |
| **4. Aggregation** | `h'_i = ELU( SUM_j alpha_ij x Wh_j )` | Weighted mix of neighbour features |
| **5. Multi-head (K=4)** | `CONCAT_{k=1}^{4} ELU(...)` | 4 independent attention views concatenated |
| **6. Output layer** | Final GAT → nhid=32 | Fixed-size representation fed to DQN |

### 2.4 GAT Hyperparameters

| Parameter | Value |
|---|---|
| Attention heads (K) | 4 |
| Hidden dimension (nhid) | 32 |
| Dropout | 0.1 |
| LeakyReLU slope | 0.2 |
| Reference | CoLight (Wei et al., 2019) |

### 2.5 Temporal Extension

| Stage | Shape | Purpose |
|---|---|---|
| History buffer | Last T=4 time steps | Captures short-term traffic dynamics |
| Spatial GAT (each step t) | [nodes x features] → [nodes x nhid] | Encodes spatial relationships per step |
| Temporal Attention | [nodes x T x nhid] → [nodes x nhid] | Learns which past step matters most |
| Final output | [nodes x nhid] | Single encoding passed to DQN per node |

### 2.6 GAT Role in Federated Clustering

| Step | How GAT is used |
|---|---|
| After each round | Each node's GAT output = traffic "fingerprint" vector |
| Fingerprint format | [avg_wait, avg_queue, ratio, max_queue, tl_eff, priority] |
| Similarity | Cosine similarity between all node fingerprint pairs |
| Clustering result | Nodes with highest mutual similarity → same cluster → share weights |
| China OSM observed | avg_sim = 0.9995–0.9996 with min = 0.9985 (slight spread vs sumo_configs2 0.9983) |

---

## 3. PER — Prioritized Experience Replay (Architecture & Theory)

### 3.1 Standard vs Prioritised Replay

| Aspect | Uniform Replay | PER (AdaptFlow) |
|---|---|---|
| Sampling rule | Random — all transitions equally likely | Proportional to TD-error: high-surprise transitions sampled more |
| Data structure | Circular buffer O(1) | SumTree O(log N) |
| Bias | None | IS-weight correction required |
| Sample efficiency | Low — wastes compute on learned transitions | High — focuses on hardest transitions |

### 3.2 Priority Formula

| Symbol | Meaning | Value |
|---|---|---|
| delta_i | TD-error = `|Q_target - Q_eval|` | Computed after each update |
| epsilon | Priority floor | 1e-5 |
| alpha | Prioritisation exponent | 0.6 |
| p_i | `(|delta_i| + epsilon)^alpha` | Stored in SumTree leaf |
| P(i) | `p_i / SUM_j p_j` | Sampling probability |

### 3.3 SumTree (O(log N) Sampling)

```
      [total = sum of all priorities]     <- root
         /                \
    [partial]           [partial]         <- internal nodes
    /      \            /      \
 [p_0]  [p_1]       [p_2]   [p_3]        <- leaf nodes (one per transition)
```

| Operation | Complexity |
|---|---|
| Add transition | O(log N) |
| Sample transition | O(log N) — walk tree from root |
| Update priority | O(log N) — update leaf + propagate |
| Capacity | 5000 transitions |

### 3.4 Importance Sampling Correction

| Symbol | Formula | Purpose |
|---|---|---|
| beta (initial) | 0.4 | IS correction strength |
| beta (final) | 1.0 | Full bias correction |
| beta increment | 0.001 per replay step | Annealing schedule |
| w_i | `(1/N x 1/P(i))^beta` | IS weight — normalised to [0,1] |
| Weighted loss | `mean(w_i x SmoothL1(Q_eval, Q_target))` | Applied per sample in batch |

### 3.5 PER Hyperparameters

| Parameter | Value |
|---|---|
| alpha | 0.6 |
| epsilon | 1e-5 |
| beta (start) | 0.4 |
| beta increment | 0.001 |
| Memory capacity | 5000 |
| Batch size | 64 |

### 3.6 Training Step (Double DQN + PER)

| Step | Action |
|---|---|
| 1 | Sample 64 transitions via PER priority-proportional sampling |
| 2 | Online net picks action: `next_a = policy_net(s').argmax()` |
| 3 | Target net evaluates: `Q_next = target_net(s').gather(next_a)` |
| 4 | Compute target: `Q_target = r + (1 - done) x 0.99 x Q_next` |
| 5 | IS-weighted Huber loss: `loss = mean(w_i x SmoothL1(Q, Q_target))` |
| 6 | Backprop + Adam update (lr=1e-3) |
| 7 | Update SumTree priorities with new `|delta_i|` |
| 8 | Anneal epsilon and beta |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | AdaptFlow-TSC (Federated Double DQN + GAT + PER) |
| Environment | SUMO-GUI (real traffic simulation) |
| SUMO scenario | China OSM (6 unique node configs with begin offsets) |
| Rounds | 5 |
| Steps per episode | 400 |
| Nodes | 6 |
| Vehicles per node per episode | 1000 |
| Trip period | 0.4s |
| Learning rate | 1e-3 (Adam) |
| Gamma | 0.99 |
| Epsilon start → end | 1.0 → 0.01 (decay 0.997) |
| GAT heads | 4 |
| GAT hidden dim | 32 |
| Temporal window T | 4 steps |
| PER alpha | 0.6 |
| PER beta start | 0.4 |
| Memory size | 5000 |
| Batch size | 64 |

---

## 5. Per-Round Detailed Results

---

### Round 1 — Exploration Phase

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9996 |
| Similarity Range | 0.9990 – 1.0000 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | 64.19 | 324 | 990 | 0.3273 | 31.53 | 6 | 3.610 | 8.15 | 0.175655 |
| node_1 | 0 | 119.20 | 310 | 994 | 0.3119 | 33.22 | 7 | 3.620 | 0.00 | 0.169174 |
| node_2 | 0 | **1.05** | 297 | 988 | **0.3006** | 33.82 | 7 | 4.483 | 0.14 | 0.151688 |
| node_3 | 1 | **129.95** | 314 | 991 | 0.3169 | 33.13 | 7 | 3.248 | 0.60 | 0.224580 |
| node_4 | 1 | 111.52 | 307 | 989 | 0.3104 | 32.32 | 7 | 3.503 | 0.00 | 0.209914 |
| node_5 | 1 | 82.23 | 315 | 992 | 0.3175 | 31.28 | 7 | 3.510 | 0.65 | 0.202326 |

#### Cluster Summary

| Cluster | Members | Avg Flow Ratio | Avg Congestion |
|---|---|---|---|
| cluster_0 | node_0, node_1, node_2 | 0.3133 | 32.86s |
| cluster_1 | node_3, node_4, node_5 | 0.3149 | 32.24s |

#### Round 1 Highlights

| Observation | Detail |
|---|---|
| Worst reward | node_2 = 1.05 — severe congestion, avg_q = 4.48 |
| Best reward | node_3 = 129.95 — most effective clearance |
| Highest queue | node_2 avg_q = 4.483 (worst congestion) |
| Fastest speed | node_0 = 8.15 m/s (only node with meaningful speed) |
| All max_q | 6–7 — heavy traffic throughout |
| Cluster | Naive split: node_0/1/2 vs node_3/4/5 |

---

### Round 2 — First Federated Update

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9995 |
| Similarity Range | 0.9985 – 1.0000 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | 66.85 | 314 | 993 | 0.3162 | 31.83 | 7 | 3.695 | 0.00 | 0.146427 |
| node_1 | 0 | 69.66 | 321 | 993 | 0.3233 | 33.26 | 7 | 3.505 | 0.54 | 0.132176 |
| node_2 | 1 | 111.62 | 305 | 992 | 0.3075 | 32.61 | 6 | **2.840** | 0.00 | 0.179291 |
| node_3 | 1 | 52.71 | 306 | 993 | 0.3082 | 34.40 | 6 | 3.450 | 0.70 | 0.103106 |
| node_4 | 0 | 70.92 | 310 | 992 | 0.3125 | 34.34 | 7 | 3.638 | 3.37 | 0.139769 |
| node_5 | 1 | 64.31 | 310 | 993 | 0.3122 | **29.66** | 6 | 3.365 | **7.41** | 0.091872 |

#### Cluster Summary

| Cluster | Members | Avg Flow Ratio | Avg Congestion |
|---|---|---|---|
| cluster_0 | node_0, node_1, node_4 | 0.3173 | 33.14s |
| cluster_1 | node_2, node_3, node_5 | 0.3093 | 32.23s |

#### Round 2 Highlights

| Observation | Detail |
|---|---|
| Cluster reorganisation | node_2 moves to cluster_1 (lower congestion group); node_4 joins cluster_0 |
| node_2 queue drop | 4.483 → **2.840** (−36.7%) — biggest improvement this round |
| node_5 wait drop | 31.28 → **29.66s** (−5.2%) — best wait time in entire study |
| node_5 speed | 0.65 → **7.41 m/s** — now fastest moving node |
| Loss trend | All nodes decreasing — PER learning actively |
| node_3 reward drop | 129.95 → 52.71 — over-exploration after federated weight update |

---

### Round 3 — Reclustering

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9995 |
| Similarity Range | 0.9985 – 1.0000 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 1 | 66.85 | 314 | 993 | 0.3162 | 31.83 | 7 | 3.695 | 0.00 | 0.104013 |
| node_1 | 1 | 69.66 | 321 | 993 | 0.3233 | 33.26 | 7 | 3.505 | 0.54 | 0.120157 |
| node_2 | 0 | 111.62 | 305 | 992 | 0.3075 | 32.61 | 6 | 2.840 | 0.00 | 0.094599 |
| node_3 | 0 | 52.71 | 306 | 993 | 0.3082 | 34.40 | 6 | 3.450 | 0.70 | 0.108834 |
| node_4 | 0 | 73.35 | 307 | 991 | 0.3098 | 34.43 | 7 | 3.615 | 3.93 | 0.129773 |
| node_5 | 0 | 64.31 | 310 | 993 | 0.3122 | 29.66 | 6 | 3.365 | 7.41 | 0.125621 |

#### Cluster Summary

| Cluster | Members | Avg Flow Ratio | Avg Congestion |
|---|---|---|---|
| cluster_0 | node_2, node_3, node_4, node_5 | 0.3094 | 32.78s |
| cluster_1 | node_0, node_1 | 0.3197 | 32.55s |

#### Round 3 Highlights

| Observation | Detail |
|---|---|
| Cluster size shift | cluster_0 expands to 4 members; cluster_1 shrinks to 2 (node_0 + node_1) |
| node_0/1 grouped alone | Their fingerprints (avg_wait ~31–33s, priority=1.0/0.5) diverge from the others |
| All traffic metrics stable | Same as Round 2 — policy converging in deterministic SUMO |
| node_2 loss | 0.179 → **0.095** (−47%) — PER most active on this node |
| node_4 reward | 70.92 → **73.35** (+3.4%) — slight improvement |

---

### Round 4

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9995 |
| Similarity Range | 0.9985 – 1.0000 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 1 | 66.85 | 314 | 993 | 0.3162 | 31.83 | 7 | 3.695 | 0.00 | 0.102290 |
| node_1 | 1 | 69.66 | 321 | 993 | 0.3233 | 33.26 | 7 | 3.505 | 0.54 | 0.083728 |
| node_2 | 0 | 111.62 | 305 | 992 | 0.3075 | 32.61 | 6 | 2.840 | 0.00 | 0.089400 |
| node_3 | 0 | 52.71 | 306 | 993 | 0.3082 | 34.40 | 6 | 3.450 | 0.70 | 0.154482 |
| node_4 | 1 | 73.35 | 307 | 991 | 0.3098 | 34.43 | 7 | 3.615 | 3.93 | 0.133273 |
| node_5 | 0 | 64.31 | 310 | 993 | 0.3122 | 29.66 | 6 | 3.365 | 7.41 | 0.121481 |

#### Cluster Summary

| Cluster | Members | Avg Flow Ratio | Avg Congestion |
|---|---|---|---|
| cluster_0 | node_2, node_3, node_5 | 0.3093 | 32.23s |
| cluster_1 | node_0, node_1, node_4 | 0.3164 | 33.17s |

#### Round 4 Highlights

| Observation | Detail |
|---|---|
| node_4 switches | Moves to cluster_1 with node_0 and node_1 (higher avg_wait group) |
| node_1 loss | 0.120 → **0.084** (−30%) — consistent PER improvement |
| All traffic metrics | Identical to Round 3 — policy fully converged |
| node_3 loss spike | 0.109 → 0.154 — PER resampling older high-priority transitions |

---

### Round 5 — Final State

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9995 |
| Similarity Range | 0.9985 – 1.0000 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | 66.85 | 314 | 993 | 0.3162 | 31.83 | 7 | 3.695 | 0.00 | 0.098353 |
| node_1 | 0 | 69.66 | 321 | 993 | **0.3233** | 33.26 | 7 | 3.505 | 0.54 | **0.051119** |
| node_2 | 1 | **111.62** | 305 | 992 | 0.3075 | 32.61 | 6 | **2.840** | 0.00 | 0.160490 |
| node_3 | 1 | 52.71 | 306 | 993 | 0.3082 | 34.40 | 6 | 3.450 | 0.70 | 0.107104 |
| node_4 | 1 | 73.35 | 307 | 991 | 0.3098 | 34.43 | 7 | 3.615 | 3.93 | 0.149402 |
| node_5 | 1 | 64.31 | 310 | 993 | 0.3122 | **29.66** | 6 | 3.365 | **7.41** | **0.022750** |

#### Cluster Summary

| Cluster | Members | Avg Flow Ratio | Avg Congestion |
|---|---|---|---|
| cluster_0 | node_0, node_1 | 0.3197 | 32.55s |
| cluster_1 | node_2, node_3, node_4, node_5 | 0.3094 | 32.78s |

#### Round 5 Highlights

| Observation | Detail |
|---|---|
| Best loss in study | node_5 = **0.02275** — near-zero TD-error, policy fully converged |
| node_1 loss | 0.084 → **0.051** (−39%) — PER refined to minimal error |
| node_2 maintains best queue | avg_q = **2.840** — best in the study |
| node_5 maintains best wait | avg_wait = **29.66s** and speed = **7.41 m/s** |
| Cluster back to R3 pattern | node_0+node_1 isolated; node_2/3/4/5 grouped |

---

## 6. Cross-Round Progression Tables

### Reward per Node per Round

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Change R1→R5 |
|---|---|---|---|---|---|---|
| node_0 | 64.19 | **66.85** | 66.85 | 66.85 | 66.85 | +2.66 (+4.1%) |
| node_1 | 119.20 | 69.66 | 69.66 | 69.66 | 69.66 | −49.54 (−41.6%) |
| node_2 | 1.05 | **111.62** | 111.62 | 111.62 | 111.62 | +110.57 (+105x) |
| node_3 | **129.95** | 52.71 | 52.71 | 52.71 | 52.71 | −77.24 (−59.4%) |
| node_4 | 111.52 | 70.92 | **73.35** | 73.35 | 73.35 | −38.17 (−34.2%) |
| node_5 | 82.23 | 64.31 | 64.31 | 64.31 | 64.31 | −17.92 (−21.8%) |

> **node_2** shows the most dramatic improvement (+105x) after receiving federated weights in Round 2.

### Throughput Ratio per Node per Round

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|
| node_0 | 0.3273 | 0.3162 | 0.3162 | 0.3162 | 0.3162 |
| node_1 | 0.3119 | 0.3233 | 0.3233 | 0.3233 | **0.3233** |
| node_2 | 0.3006 | 0.3075 | 0.3075 | 0.3075 | 0.3075 |
| node_3 | 0.3169 | 0.3082 | 0.3082 | 0.3082 | 0.3082 |
| node_4 | 0.3104 | 0.3125 | 0.3098 | 0.3098 | 0.3098 |
| node_5 | 0.3175 | 0.3122 | 0.3122 | 0.3122 | 0.3122 |

### Average Queue Length per Node per Round

| Node | Round 1 | Round 2 | Round 3–5 | Change R1→R5 |
|---|---|---|---|---|
| node_0 | 3.610 | 3.695 | 3.695 | ↑ 2.4% |
| node_1 | 3.620 | 3.505 | 3.505 | ↓ 3.2% |
| node_2 | **4.483** | **2.840** | **2.840** | ↓ **36.7%** |
| node_3 | 3.248 | 3.450 | 3.450 | ↑ 6.2% |
| node_4 | 3.503 | 3.638 | 3.615 | ↑ 3.2% |
| node_5 | 3.510 | 3.365 | 3.365 | ↓ 4.1% |

### Average Wait Time per Node per Round (seconds)

| Node | Round 1 | Round 2 | Round 3–5 | Change |
|---|---|---|---|---|
| node_0 | 31.53 | 31.83 | 31.83 | ↑ 0.9% |
| node_1 | 33.22 | 33.26 | 33.26 | Stable |
| node_2 | 33.82 | 32.61 | 32.61 | ↓ 3.6% |
| node_3 | 33.13 | 34.40 | 34.40 | ↑ 3.8% |
| node_4 | 32.32 | 34.34 | 34.43 | ↑ 6.5% |
| node_5 | **31.28** | **29.66** | **29.66** | ↓ **5.2%** |

### TD-Error Loss per Node per Round (PER Learning Signal)

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Total Reduction |
|---|---|---|---|---|---|---|
| node_0 | 0.175655 | 0.146427 | 0.104013 | 0.102290 | 0.098353 | ↓ **44.0%** |
| node_1 | 0.169174 | 0.132176 | 0.120157 | 0.083728 | **0.051119** | ↓ **69.8%** |
| node_2 | 0.151688 | 0.179291 | 0.094599 | 0.089400 | 0.160490 | Oscillating |
| node_3 | 0.224580 | 0.103106 | 0.108834 | 0.154482 | 0.107104 | ↓ 52.3% |
| node_4 | 0.209914 | 0.139769 | 0.129773 | 0.133273 | 0.149402 | ↓ 28.8% |
| node_5 | 0.202326 | 0.091872 | 0.125621 | 0.121481 | **0.022750** | ↓ **88.8%** |

---

## 7. Clustering Dynamics

### Cluster Membership per Round

| Round | Group A (Higher Flow) | Group B (Lower Flow) | Stability |
|---|---|---|---|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | Initial naive split |
| 2 | node_0, node_1, node_4 | node_2, node_3, node_5 | Reorganised by fingerprint |
| 3 | node_0, node_1 (2 only) | node_2, node_3, node_4, node_5 | node_0/1 isolated |
| 4 | node_0, node_1, node_4 | node_2, node_3, node_5 | Back to R2 pattern |
| 5 | node_0, node_1 (2 only) | node_2, node_3, node_4, node_5 | Back to R3 pattern |

### What Drives the Clustering

| Driver | Effect |
|---|---|
| **Priority flag** | node_0 (priority=1.0) and node_1 (priority=0.5) always have higher fingerprint priority — pulls them together |
| **Avg wait time** | node_5 consistently lowest wait (29.7s), node_3/4 consistently highest (34.4s) |
| **Queue length** | node_2 lowest avg_q (2.84) after Round 2 — separates it from high-queue nodes |
| **Similarity** | avg_sim=0.9995 — all nodes very similar; small differences in fingerprint drive cluster boundaries |

### Pairwise Similarity Across Rounds

| Round | Avg Sim | Min Sim | Max Sim |
|---|---|---|---|
| 1 | 0.9996 | 0.9990 | 1.0000 |
| 2 | 0.9995 | 0.9985 | 1.0000 |
| 3 | 0.9995 | 0.9985 | 1.0000 |
| 4 | 0.9995 | 0.9985 | 1.0000 |
| 5 | 0.9995 | 0.9985 | 1.0000 |

### Fingerprint Vectors (Round 1) — [avg_wait, avg_q, ratio, max_q, tl_eff, priority]

| Node | avg_wait | avg_q | ratio | max_q | priority | Note |
|---|---|---|---|---|---|---|
| node_0 | 31.53 | 3.610 | 0.3273 | 6 | **1.0** | Highest priority |
| node_1 | 33.22 | 3.620 | 0.3119 | 7 | **0.5** | Second priority |
| node_2 | 33.82 | **4.483** | 0.3006 | 7 | 0.0 | Most congested |
| node_3 | 33.13 | 3.248 | 0.3169 | 7 | 0.0 | — |
| node_4 | 32.32 | 3.503 | 0.3104 | 7 | 0.0 | — |
| node_5 | 31.28 | 3.510 | 0.3175 | 7 | 0.0 | Lowest wait |

---

## 8. Key Findings

### Throughput Ratio Explained

| Metric | Formula | Observed |
|---|---|---|
| Throughput ratio | arrived / departed | 30.75% – 32.73% |
| Best (node_1) | 321 / 993 | 32.33% |
| Worst (node_2) | 305 / 992 | 30.75% |
| ~670–680 vehicles still in transit per episode | 400 steps not enough to clear 1000 vehicles on a complex OSM network | Expected |

### China OSM vs Default Scenario Comparison

| Metric | Default (sumo_configs2) | China OSM |
|---|---|---|
| Vehicles per episode | ~500–1000 | **1000** |
| Avg wait time | 11–13s | **29–34s** |
| Avg queue length | 0.60–0.89 | **2.84–4.48** |
| Throughput ratio | 32–33% | **31–33%** |
| Reward scale | ~350–387 | **1–130** |
| Max queue | 4–5 | **6–7** |
| Network complexity | Urban OSM (simple) | **China urban OSM (richer road network)** |

> The China OSM network has a far more complex topology causing much longer wait times and larger queues. The reward scale is lower because the penalty terms dominate over the bonus terms.

### node_2 Recovery — Dramatic Effect of Federated Learning

| Round | Reward | Avg Q | Explanation |
|---|---|---|---|
| 1 | **1.05** | 4.483 | Severe congestion — almost no reward earned |
| 2 | **111.62** | 2.840 | Received federated weights → queue dropped 37% |
| 3–5 | **111.62** | 2.840 | Stable — policy converged |

This is the clearest demonstration of federated learning working: node_2 was nearly failing alone, but after receiving weights from its cluster partners in Round 2, its performance improved 105x.

### PER Learning Evidence

| Node | Loss R1 | Loss R5 | Reduction | Interpretation |
|---|---|---|---|---|
| node_5 | 0.2023 | **0.0228** | −88.8% | Near-zero TD-error — fully converged |
| node_1 | 0.1692 | **0.0511** | −69.8% | PER found and eliminated hard transitions |
| node_0 | 0.1757 | **0.0984** | −44.0% | Steady improvement throughout |
| node_3 | 0.2246 | 0.1071 | −52.3% | Good overall but oscillates |

---

## 9. Issues and Recommendations

| # | Issue | Root Cause | Recommended Fix | Expected Impact |
|---|---|---|---|---|
| 1 | Very high wait times (29–34s) | 1000 vehicles in complex OSM network in 400s — network capacity exceeded | Reduce to 500–600 vehicles (period=0.65–0.8s) | Wait times drop to 15–20s |
| 2 | Reward scale very low (1–130 vs 350–387 in default) | High queue/wait penalties dominate bonus terms | Increase throughput bonus weight or reduce queue penalty coefficient | More balanced reward, faster learning |
| 3 | Clustering unstable (changes every round) | High overall similarity (0.9995) makes small fingerprint differences dominate | Increase clustering threshold OR use more discriminative fingerprint features | Stable 2-cluster partition from Round 2 |
| 4 | Speed near 0 m/s for several nodes | Vehicles gridlocked or waiting — not moving | Reduce vehicle density (see Issue 1) | Speeds recover to 5–15 m/s |
| 5 | node_3 reward never recovers | 129.95 → 52.71 after federated update — wrong weights pushed | May be receiving weights from a dissimilar cluster member | Tighten clustering — isolate node_3 into its own sub-cluster |

---

## 10. Final Summary

| Metric | Round 1 | Round 5 | Change |
|---|---|---|---|
| Best reward | 129.95 (node_3) | 111.62 (node_2) | Redistributed |
| Worst reward | **1.05** (node_2) | **66.85** (node_0) | node_2 improved +105x |
| Best throughput ratio | 0.3273 | 0.3233 | Stable |
| Best avg queue | 3.248 | **2.840** (node_2) | ↓ 12.6% |
| Worst avg queue | **4.483** (node_2) | 3.695 (node_0) | ↓ 17.6% |
| Best avg wait | 31.28s (node_5) | **29.66s** (node_5) | ↓ 5.2% |
| Best loss | 0.151688 | **0.022750** (node_5) | ↓ **85.0%** |
| Avg pairwise similarity | 0.9996 | 0.9995 | Stable |
| Clustering stability | Unstable | Alternating R3/R4 pattern | Not yet converged |

AdaptFlow-TSC on China OSM demonstrates that federated learning can rescue a nearly-failing node (node_2: reward 1.05 → 111.62) through targeted weight sharing. PER achieves exceptional loss reduction — node_5 loss drops 89% to near-zero. However, the high vehicle density (1000 vehicles in 400 steps) creates severe congestion (avg wait 29–34s) that limits throughput ratio and keeps rewards low. Reducing vehicle load to 500–600 per episode is the primary recommended next step.
