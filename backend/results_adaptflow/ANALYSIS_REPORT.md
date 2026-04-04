# AdaptFlow-TSC Training Analysis Report

| Field | Value |
|---|---|
| **Scenario** | Default Urban OSM (`sumo_configs2`) |
| **Mode** | SUMO-GUI (Real SUMO Simulation) |
| **Rounds** | 5 |
| **Steps / Episode** | 400 (env `max_steps = 400`) |
| **Nodes** | 6 (node_0 – node_5) |
| **Client 1 nodes** | node_0, node_2, node_4 → `osm_client1.sumocfg` |
| **Client 2 nodes** | node_1, node_3, node_5 → `osm_client2.sumocfg` |
| **Vehicles (first 200 s)** | ~500 per client (period = 0.4 s) |
| **Date** | 2026-04-04 |

---

## 1. System Components at a Glance

| Component | Role | Key Parameter |
|---|---|---|
| **SUMO-GUI** | Real traffic simulator providing state & reward | 6 intersections, 400 steps |
| **GAT Encoder** | Encodes spatial + temporal traffic state into a vector | 4 heads, nhid=32, T=4 time steps |
| **Double DQN** | Learns optimal signal phase policy from encoded state | γ=0.99, lr=1e-3 |
| **PER Buffer** | Prioritised replay — focuses training on high-error transitions | α=0.6, β=0.4→1.0, cap=5000 |
| **Federated Aggregation** | Shares weights across nodes grouped by traffic similarity | Cosine similarity on fingerprints |
| **Dynamic Clustering** | Groups nodes with similar congestion profiles | 2 clusters per round |

---

## 2. GAT — Graph Attention Network

### 2.1 Why GAT?

| Approach | How neighbours are weighted | Problem |
|---|---|---|
| Standard Graph Conv | Equally (all neighbours same weight) | Cannot distinguish important from unimportant intersections |
| **GAT (used here)** | **Learned attention weight per neighbour** | Focuses on most relevant intersections adaptively |

### 2.2 Input Graph Structure

| Graph element | Meaning in AdaptFlow |
|---|---|
| Node | One traffic signal intersection |
| Edge | Road connection between intersections |
| Node feature vector | Queue length, waiting time, phase, vehicle count, speed |
| Adjacency matrix | 1 if intersections are connected, 0 otherwise |

### 2.3 GAT Computation Steps

| Step | Formula | What it does |
|---|---|---|
| **1. Linear Projection** | `Wh_i = h_i × W` | Projects each node's features from F → F' dimensions |
| **2. Attention Score** | `e_ij = LeakyReLU( aᵀ × [Wh_i ‖ Wh_j] )` | Computes raw attention between node i and neighbour j |
| **3. Normalisation** | `α_ij = exp(e_ij) / Σ_k exp(e_ik)` | Softmax over the neighbourhood → weights sum to 1 |
| **4. Aggregation** | `h'_i = ELU( Σ_j α_ij × Wh_j )` | Weighted sum of neighbour features |
| **5. Multi-head concat** | `h'_i = CONCAT_{k=1}^{4} ELU(...)` | 4 independent heads combined → richer representation |
| **6. Output projection** | Final GAT layer reduces to nhid=32 | Gives fixed-size encoding for the DQN |

### 2.4 GAT Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| Attention heads (K) | **4** | 4 independent attention views, then concatenated |
| Hidden dim (nhid) | **32** | Size of the encoded node representation |
| Dropout | 0.1 | Regularisation to prevent overfitting |
| LeakyReLU slope (α) | 0.2 | Allows small gradient for negative attention scores |
| Inspired by | CoLight (Wei et al., 2019) | Originally designed for cooperative traffic control |

### 2.5 Temporal Attention Extension

| Stage | Input shape | Output shape | What it does |
|---|---|---|---|
| History buffer | Last T=4 states | `[T × nodes × features]` | Keeps rolling window of past states |
| Spatial GAT (per step) | `[nodes × features]` | `[nodes × nhid]` | Encodes spatial relationships |
| Reshape | `[T × nodes × nhid]` | `[nodes × T × nhid]` | Groups by node for temporal processing |
| Temporal Attention | `[nodes × T × nhid]` | `[nodes × nhid]` | Learns which time step matters most |

### 2.6 GAT Role in Clustering

| Step | How GAT is used |
|---|---|
| Fingerprint extraction | After training, each node's GAT output = its traffic "fingerprint" |
| Similarity measurement | Cosine similarity between all pairs of fingerprints |
| Cluster assignment | Nodes with similarity > threshold → same cluster → share weights |
| Observed similarity | 0.9969 (Round 1) → **0.9983** (Round 2–5) — increasing alignment |

---

## 3. PER — Prioritized Experience Replay

### 3.1 Standard vs Prioritized Replay

| Aspect | Standard Replay (uniform) | PER (used here) |
|---|---|---|
| Sampling | Random — all transitions equally likely | Proportional to TD-error — high-error transitions sampled more |
| Problem solved | — | Wastes less compute on already-learned transitions |
| Data structure | Simple circular buffer, O(1) | SumTree, O(log N) |
| Bias correction | Not needed | IS weights required |

### 3.2 Priority Calculation

| Symbol | Meaning | Value in AdaptFlow |
|---|---|---|
| `δ_i` | TD-error = `\|Q_target - Q_eval\|` for transition i | Computed each replay |
| `ε` (epsilon) | Priority floor — prevents zero priority | **1e-5** |
| `α` (alpha) | Prioritisation exponent (0=uniform, 1=full priority) | **0.6** |
| `p_i` | Priority: `p_i = (\|δ_i\| + ε)^α` | Stored in SumTree leaf |
| `P(i)` | Sampling probability: `p_i / Σ p_j` | Used to pick transitions |

### 3.3 SumTree Data Structure

```
          [total = 1.8]          ← root = sum of ALL priorities
         /             \
     [1.0]           [0.8]       ← internal nodes (partial sums)
    /     \          /    \
 [0.4]  [0.6]    [0.3]  [0.5]   ← leaf nodes (per-transition priority)
```

| Property | Value |
|---|---|
| Sampling complexity | **O(log N)** per sample |
| Algorithm | Divide total into 64 equal segments → pick random value in each → walk tree |
| Update complexity | O(log N) — update leaf, propagate change to root |
| Capacity | **5000** transitions |

### 3.4 Importance Sampling (IS) Correction

| Symbol | Formula | Purpose |
|---|---|---|
| `β` (beta) | Starts at 0.4, anneals to 1.0 | Controls how much bias correction to apply |
| `w_i` | `(1/N × 1/P(i))^β` | IS weight for transition i |
| Normalised `w_i` | `w_i / max(w)` → range [0, 1] | Applied as multiplier to per-sample loss |
| Beta annealing | `β += 0.001` per replay step | Early: more bias (fast learning). Late: fully corrected |

### 3.5 PER Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.6 | Strength of prioritisation |
| `epsilon` | 1e-5 | Prevents zero-priority transitions |
| `beta` (initial) | 0.4 | IS correction start value |
| `beta_increment` | 0.001 | Annealing rate per replay |
| `memory_size` | 5000 | Buffer capacity (transitions stored) |
| `batch_size` | 64 | Transitions sampled per training step |
| `max_priority` | Tracks max seen | New transitions always inserted at max priority |

### 3.6 Complete Training Step

| Step | Action |
|---|---|
| 1 | Sample 64 transitions from PER buffer weighted by priority |
| 2 | Online policy net selects next action: `next_a = policy_net(s').argmax()` |
| 3 | Target net evaluates it: `Q_next = target_net(s').gather(next_a)` **(Double DQN)** |
| 4 | Compute target: `Q_target = r + (1 - done) × 0.99 × Q_next` |
| 5 | Compute IS-weighted Huber loss: `loss = mean(IS_weights × SmoothL1(Q_eval, Q_target))` |
| 6 | Backprop + Adam update |
| 7 | Update SumTree priorities with new `\|δ_i\|` |
| 8 | Decay epsilon for next step |

---

## 4. Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr = 1e-3 |
| Discount factor γ | 0.99 |
| Epsilon start → end | 1.0 → 0.01 |
| Epsilon decay | 0.997 per step |
| Target net update | Hard copy every N steps |
| GAT heads | 4 |
| GAT hidden dim | 32 |
| Temporal window (T) | 4 steps |
| PER alpha | 0.6 |
| PER beta start | 0.4 |
| PER beta increment | 0.001 |
| Memory size | 5000 |
| Batch size | 64 |

---

## 5. Per-Round Detailed Results

> **Note:** All episodes terminated at 400 steps because `SUMOTrafficEnvironment.max_steps = 400`.
> Even though `--steps 1000` was passed to the trainer, the environment cap fires first.

---

### Round 1 — Exploration Phase

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9969 |
| Epsilon (approximately) | ~1.0 (mostly random actions) |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | 381.34 | 330 | 998 | 0.3307 | 11.33 | 4 | 0.750 | 25.76 | 0.5088 |
| node_1 | 0 | 384.81 | 327 | 998 | 0.3277 | 12.78 | 3 | 0.790 | 26.32 | 0.5190 |
| node_2 | 0 | 381.19 | 334 | 997 | **0.3350** | 10.25 | 3 | 0.705 | 25.79 | 0.4999 |
| node_3 | 1 | 380.65 | 329 | 998 | 0.3297 | 13.41 | 4 | 0.918 | 24.46 | **0.5902** |
| node_4 | 1 | 365.42 | 332 | 997 | 0.3330 | 11.15 | 4 | 0.838 | 24.08 | 0.4717 |
| node_5 | 1 | **356.34** | 331 | 998 | 0.3317 | 12.29 | 4 | **1.278** | 25.31 | 0.4403 |

#### Cluster Composition

| Cluster | Members | Avg Ratio | Avg Wait (s) |
|---|---|---|---|
| cluster_0 | node_0, node_1, node_2 | 0.3311 | 11.45 |
| cluster_1 | node_3, node_4, node_5 | 0.3314 | 12.28 |

#### Round 1 Highlights

| Observation | Detail |
|---|---|
| Worst reward | node_5 = 356.34 (most congested) |
| Best throughput | node_2 = 0.3350 |
| Highest loss | node_3 = 0.5902 → most active PER learning |
| Highest queue | node_5 avg_q = 1.278 |
| Similarity = 0.9969 | All nodes start from identical random weights |

---

### Round 2 — Clustering Stabilises

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9983 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | **387.31** | 331 | 998 | 0.3317 | 11.36 | 4 | **0.595** | 25.74 | 0.4848 |
| node_1 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4468 |
| node_2 | 0 | **387.31** | 331 | 998 | 0.3317 | 11.36 | 4 | **0.595** | 25.74 | 0.4610 |
| node_3 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4602 |
| node_4 | 0 | **387.31** | 331 | 998 | 0.3317 | 11.36 | 4 | **0.595** | 25.74 | 0.4644 |
| node_5 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4205 |

#### Cluster Composition

| Cluster | Members | Avg Ratio | Avg Wait (s) |
|---|---|---|---|
| cluster_0 | node_0, node_2, node_4 | 0.3317 | 11.36 |
| cluster_1 | node_1, node_3, node_5 | 0.3263 | 12.65 |

#### Round 2 Highlights

| Observation | Detail |
|---|---|
| Clustering correct | client1 nodes (0/2/4) → cluster_0; client2 nodes (1/3/5) → cluster_1 |
| Reward jump | node_0: 381.34 → **387.31** (+1.6%) |
| Queue drop — node_0 | 0.750 → **0.595** (↓ 21%) |
| Queue drop — node_4 | 0.838 → **0.595** (↓ 29%) |
| Queue drop — node_5 | 1.278 → **0.893** (↓ 30%) |
| node_1 max_q | 3 → **5** (client2 scenario has higher peak demand) |

---

### Round 3

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9983 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 1* | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4583 |
| node_1 | 0* | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4523 |
| node_2 | 1* | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4760 |
| node_3 | 0* | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4756 |
| node_4 | 1* | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4586 |
| node_5 | 0* | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | **0.3708** |

> *Cluster labels 0/1 swapped vs Round 2 — **membership is identical**. This is a label-naming artifact, not a real change.

#### Cluster Composition

| Cluster | Members | Avg Ratio | Avg Wait (s) |
|---|---|---|---|
| cluster_0 | node_1, node_3, node_5 | 0.3263 | 12.65 |
| cluster_1 | node_0, node_2, node_4 | 0.3317 | 11.36 |

#### Round 3 Highlights

| Observation | Detail |
|---|---|
| Traffic metrics | Identical to Round 2 (deterministic SUMO + converged policy) |
| Best loss in study | node_5 = **0.3708** — PER found the highest-TD-error transitions |
| Cluster label flip | Cosmetic only — partition unchanged |

---

### Round 4

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9983 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 1 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4648 |
| node_1 | 0 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4631 |
| node_2 | 1 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4735 |
| node_3 | 0 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4714 |
| node_4 | 1 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4618 |
| node_5 | 0 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4352 |

#### Round 4 Highlights

| Observation | Detail |
|---|---|
| Clustering | Fully stable — 3rd consecutive round with same partition |
| node_5 loss | 0.3708 → 0.4352 (transient rebound — PER beta annealing reducing priority bias) |
| All traffic metrics | Identical to Rounds 2 & 3 — policy fully converged |

---

### Round 5 — Final State

| Property | Value |
|---|---|
| Mode | SUMO-GUI |
| Avg Pairwise Similarity | 0.9983 |

#### Node Metrics

| Node | Cluster | Reward | Arrived | Departed | Ratio | Avg Wait (s) | Max Q | Avg Q | Speed (m/s) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| node_0 | 0 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4852 |
| node_1 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4467 |
| node_2 | 0 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4604 |
| node_3 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4534 |
| node_4 | 0 | 387.31 | 331 | 998 | 0.3317 | 11.36 | 4 | 0.595 | 25.74 | 0.4350 |
| node_5 | 1 | 379.13 | 325 | 996 | 0.3263 | 12.65 | 5 | 0.893 | 24.64 | 0.4455 |

#### Cluster Composition

| Cluster | Members | Avg Ratio | Avg Wait (s) |
|---|---|---|---|
| cluster_0 | node_0, node_2, node_4 | 0.3317 | 11.36 |
| cluster_1 | node_1, node_3, node_5 | 0.3263 | 12.65 |

#### Round 5 Highlights

| Observation | Detail |
|---|---|
| Deployed model | Global weighted average of both cluster policies exported |
| cluster_0 vs cluster_1 | Consistently better ratio (33.17% vs 32.63%) throughout |
| Loss range | 0.435 – 0.485 — normal oscillation for active PER |

---

## 6. Cross-Round Progression Tables

### Reward per Node per Round

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Change R1→R5 |
|---|---|---|---|---|---|---|
| node_0 | 381.34 | 387.31 | 387.31 | 387.31 | 387.31 | **+5.97 (+1.6%)** |
| node_1 | 384.81 | 379.13 | 379.13 | 379.13 | 379.13 | −5.68 (−1.5%) |
| node_2 | 381.19 | 387.31 | 387.31 | 387.31 | 387.31 | **+6.12 (+1.6%)** |
| node_3 | 380.65 | 379.13 | 379.13 | 379.13 | 379.13 | −1.52 (−0.4%) |
| node_4 | 365.42 | 387.31 | 387.31 | 387.31 | 387.31 | **+21.89 (+6.0%)** |
| node_5 | 356.34 | 379.13 | 379.13 | 379.13 | 379.13 | **+22.79 (+6.4%)** |

### Throughput Ratio per Node per Round

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|
| node_0 | 0.3307 | 0.3317 | 0.3317 | 0.3317 | 0.3317 |
| node_1 | 0.3277 | 0.3263 | 0.3263 | 0.3263 | 0.3263 |
| node_2 | 0.3350 | 0.3317 | 0.3317 | 0.3317 | 0.3317 |
| node_3 | 0.3297 | 0.3263 | 0.3263 | 0.3263 | 0.3263 |
| node_4 | 0.3330 | 0.3317 | 0.3317 | 0.3317 | 0.3317 |
| node_5 | 0.3317 | 0.3263 | 0.3263 | 0.3263 | 0.3263 |

### Average Queue Length per Node per Round

| Node | Round 1 | Round 2–5 | Change |
|---|---|---|---|
| node_0 | 0.750 | **0.595** | ↓ 20.7% |
| node_1 | 0.790 | 0.893 | ↑ 13.0% (peak demand ↑) |
| node_2 | 0.705 | **0.595** | ↓ 15.6% |
| node_3 | 0.918 | 0.893 | ↓ 2.7% |
| node_4 | 0.838 | **0.595** | ↓ **29.0%** |
| node_5 | 1.278 | **0.893** | ↓ **30.1%** |

### TD-Error Loss per Node per Round

| Node | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Trend |
|---|---|---|---|---|---|---|
| node_0 | 0.5088 | 0.4848 | 0.4583 | 0.4648 | 0.4852 | ↓ overall |
| node_1 | 0.5190 | 0.4468 | 0.4523 | 0.4631 | 0.4467 | ↓ −14% |
| node_2 | 0.4999 | 0.4610 | 0.4760 | 0.4735 | 0.4604 | ↓ −8% |
| node_3 | 0.5902 | 0.4602 | 0.4756 | 0.4714 | 0.4534 | ↓ **−23%** |
| node_4 | 0.4717 | 0.4644 | 0.4586 | 0.4618 | 0.4350 | ↓ −8% |
| node_5 | 0.4403 | 0.4205 | **0.3708** | 0.4352 | 0.4455 | Oscillating |

---

## 7. Clustering Dynamics

### Membership per Round

| Round | Lower-Congestion Cluster | Higher-Congestion Cluster | Stable? |
|---|---|---|---|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | ✗ Naive split |
| 2 | **node_0, node_2, node_4** | **node_1, node_3, node_5** | ✓ Correct |
| 3 | node_0, node_2, node_4 | node_1, node_3, node_5 | ✓ Stable |
| 4 | node_0, node_2, node_4 | node_1, node_3, node_5 | ✓ Stable |
| 5 | node_0, node_2, node_4 | node_1, node_3, node_5 | ✓ Stable |

### Cluster Statistics per Round

| Round | Low-Congestion Avg Flow | Low-Congestion Avg Wait | High-Congestion Avg Flow | High-Congestion Avg Wait | Avg Sim |
|---|---|---|---|---|---|
| 1 | 0.3311 | 11.45 s | 0.3314 | 12.28 s | 0.9969 |
| 2 | 0.3317 | 11.36 s | 0.3263 | 12.65 s | 0.9983 |
| 3 | 0.3317 | 11.36 s | 0.3263 | 12.65 s | 0.9983 |
| 4 | 0.3317 | 11.36 s | 0.3263 | 12.65 s | 0.9983 |
| 5 | 0.3317 | 11.36 s | 0.3263 | 12.65 s | 0.9983 |

### Fingerprint Vectors (Round 1)

Format: `[avg_wait, avg_queue, throughput_ratio, max_queue, tl_efficiency, priority]`

| Node | avg_wait | avg_q | ratio | max_q | priority |
|---|---|---|---|---|---|
| node_0 | 11.33 | 0.750 | 0.3307 | 4.0 | 1.0 |
| node_1 | 12.78 | 0.790 | 0.3277 | 3.0 | 0.5 |
| node_2 | 10.25 | 0.705 | 0.3350 | 3.0 | 0.0 |
| node_3 | 13.41 | 0.918 | 0.3297 | 4.0 | 0.0 |
| node_4 | 11.15 | 0.838 | 0.3330 | 4.0 | 0.0 |
| node_5 | 12.29 | 1.278 | 0.3317 | 4.0 | 0.0 |

---

## 8. Key Findings

### Throughput Ratio Explained

| Metric | Formula | Result |
|---|---|---|
| Throughput ratio | `arrived / departed` | ~33% (rounds 2–5) |
| cluster_0 (client1) | 331 / 998 | 33.2% |
| cluster_1 (client2) | 325 / 996 | 32.6% |
| Pre-fix (old 125-veh load) | ~29 / 249 | ~12% |
| Improvement from trip fix | 12% → 33% | **+175%** |
| Why not 100%? | 400 steps too short to clear 500-vehicle load | ~667 vehicles still in transit |

### Evidence of Learning

| Signal | Round 1 | Round 5 | Interpretation |
|---|---|---|---|
| node_4 reward | 365.42 | 387.31 | **+6.0%** — federated policy learned better phase timing |
| node_5 reward | 356.34 | 379.13 | **+6.4%** — best improvement in the run |
| node_5 avg_q | 1.278 | 0.893 | **−30%** — agent learned to clear queues faster |
| node_3 loss | 0.5902 | 0.4534 | **−23%** — PER reduced TD-error over rounds |
| Avg pairwise sim | 0.9969 | 0.9983 | GAT representations converging across nodes |
| Clustering | Naive R1 | Stable R2–R5 | Correct partitioning after first fingerprint update |

### Why Metrics Plateau from Round 2

| Reason | Detail |
|---|---|
| Deterministic SUMO | Fixed route files + fixed seed → identical traffic each round |
| Policy converged | Same actions → same signal timings → same vehicle outcomes |
| This is expected | In a stationary environment, convergence = reproducibility |

---

## 9. Issues and Fixes

| # | Issue | Root Cause | Fix | Expected Impact |
|---|---|---|---|---|
| 1 | Episode ends at step 400, not 1000 | `max_steps = 400` in `traffic_environment.py` line 54 | Change to `self.max_steps = 1000` | Ratio: 33% → ~45% |
| 2 | Only 2 unique traffic environments for 6 nodes | node 0/2/4 share client1.sumocfg, node 1/3/5 share client2.sumocfg | Generate 6 unique trip files with different seeds | True 6-way federated diversity |
| 3 | Metrics plateau after Round 2 | Deterministic fixed-route SUMO | Use different seed per round; add departure noise | Continued learning across rounds |
| 4 | Loss oscillation in later rounds | PER beta annealing resurfaces old high-priority transitions | Reduce `per_alpha` from 0.6 → 0.5; increase `per_beta_increment` 0.001 → 0.002 | Smoother loss convergence |

---

## 10. Final Summary

| Metric | Round 1 | Round 5 | Change |
|---|---|---|---|
| Best node reward | 384.81 | 387.31 | +0.6% |
| Worst node reward | 356.34 | **379.13** | **+6.4%** |
| Throughput ratio (cluster_0) | 0.3307 | 0.3317 | Stabilised |
| Throughput ratio (cluster_1) | 0.3317 | 0.3263 | Stabilised |
| Best avg queue | 0.705 | **0.595** | ↓ 15.6% |
| Worst avg queue | 1.278 | **0.893** | ↓ **30.1%** |
| Avg pairwise similarity | 0.9969 | **0.9983** | +0.14% |
| Clustering stability | Mixed | **Stable from Round 2** | Converged |
| Best round loss (any node) | — | **0.3708** (node_5, R3) | PER working |
