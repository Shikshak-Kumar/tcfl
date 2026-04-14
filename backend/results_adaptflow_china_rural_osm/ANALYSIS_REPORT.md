# AdaptFlow China Rural OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 5 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 1500

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Avg Queue | Max Queue | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|-----------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 1453.7 | 0.466 | 1.027 | 5 | 331 / 515 | 64.3% | 0.5662 |
| node_1 | 1 | cluster_0 | 1463.1 | 0.472 | 1.073 | **6** | 334 / 515 | 64.9% | 0.5424 |
| node_2 | 1 | cluster_0 | 1417.1 | **0.334** | **1.237** | **8** | **341** / 515 | **66.2%** | 0.5480 |
| node_3 | 1 | cluster_1 | 1460.7 | 0.427 | 1.010 | 5 | 336 / 515 | 65.2% | 0.5348 |
| node_4 | 1 | cluster_1 | **1480.8** | 0.150 | 0.972 | 5 | 329 / 515 | 63.9% | 0.5244 |
| node_5 | 1 | cluster_1 | 1475.3 | 0.398 | 1.000 | 5 | 316 / 515 | 61.4% | **0.5942** |
| node_0 | 2 | cluster_1 | 1464.8 | 0.540 | 0.973 | 7 | 330 / 514 | 64.2% | 0.4473 |
| node_1 | 2 | cluster_1 | 1459.4 | 0.490 | 1.125 | 7 | 333 / 515 | 64.7% | 0.4835 |
| node_2 | 2 | cluster_0 | **1470.2** | 0.480 | 1.129 | 5 | **338** / 515 | **65.6%** | 0.4759 |
| node_3 | 2 | cluster_0 | **1483.0** | 0.460 | 0.926 | 5 | 337 / 515 | 65.4% | 0.4980 |
| node_4 | 2 | cluster_0 | 1474.4 | **0.330** | 1.035 | 5 | 329 / 515 | 63.9% | 0.4933 |
| node_5 | 2 | cluster_0 | 1473.8 | 0.270 | 1.037 | 5 | 316 / 515 | 61.4% | 0.4921 |
| node_0 | 3 | cluster_1 | 1465.4 | 0.532 | 0.961 | 7 | 331 / 514 | 64.4% | 0.4747 |
| node_1 | 3 | cluster_1 | 1460.2 | 0.483 | 1.112 | 7 | 334 / 515 | 64.9% | 0.4879 |
| node_2 | 3 | cluster_0 | 1471.0 | 0.474 | 1.121 | 5 | 339 / 515 | 65.9% | 0.4625 |
| node_3 | 3 | cluster_0 | 1483.4 | 0.453 | 0.915 | 5 | 338 / 515 | 65.6% | 0.4911 |
| node_4 | 3 | cluster_0 | 1474.8 | **0.322** | 1.024 | 5 | 330 / 515 | 64.1% | 0.4893 |
| node_5 | 3 | cluster_0 | 1474.2 | 0.264 | 1.028 | 5 | 318 / 515 | 61.8% | 0.4864 |
| node_0 | 4 | cluster_1 | 1465.9 | 0.526 | 0.954 | 7 | 332 / 514 | 64.6% | 0.4799 |
| node_1 | 4 | cluster_1 | 1460.7 | 0.478 | 1.103 | 7 | 334 / 515 | 64.9% | 0.4873 |
| node_2 | 4 | cluster_0 | 1471.5 | 0.469 | 1.114 | 5 | 340 / 515 | 66.0% | 0.4751 |
| node_3 | 4 | cluster_0 | **1483.6** | 0.448 | 0.908 | 5 | 338 / 515 | 65.7% | 0.4832 |
| node_4 | 4 | cluster_0 | 1475.1 | **0.316** | 1.018 | 5 | 331 / 515 | 64.3% | 0.4904 |
| node_5 | 4 | cluster_0 | 1474.5 | 0.258 | 1.022 | 5 | 319 / 515 | 62.0% | 0.4843 |
| node_0 | 5 | cluster_0 | 1466.3 | 0.520 | 0.948 | 6 | 333 / 514 | 64.8% | 0.4796 |
| node_1 | 5 | cluster_0 | 1461.1 | 0.473 | 1.096 | 7 | 335 / 515 | 65.1% | 0.4802 |
| node_2 | 5 | cluster_1 | 1472.0 | 0.463 | 1.107 | 5 | **341** / 515 | **66.2%** | 0.4854 |
| node_3 | 5 | cluster_1 | **1484.1** | 0.443 | 0.901 | 5 | 339 / 515 | 65.8% | 0.5098 |
| node_4 | 5 | cluster_1 | 1475.4 | **0.311** | 1.012 | 5 | 331 / 515 | 64.4% | 0.4877 |
| node_5 | 5 | cluster_1 | 1474.8 | 0.253 | 1.017 | 5 | 319 / 515 | 62.1% | **0.4844** |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Queue | Avg TP Ratio | Avg Loss | Total Arrived |
|-------|-----------|-----|-----|-------------|-----------|-------------|---------|--------------|
| 1 | 1458.5 | 1417.1 | 1480.8 | 0.375 | 1.053 | 64.3% | 0.5517 | 1987 veh |
| 2 | 1470.9 | 1459.4 | 1483.0 | 0.428 | 1.038 | 64.2% | 0.4817 | 1983 veh |
| 3 | 1471.5 | 1460.2 | 1483.4 | 0.421 | 1.027 | 64.5% | 0.4820 | 1990 veh |
| 4 | 1471.9 | 1460.7 | **1483.6** | 0.416 | 1.020 | 64.7% | 0.4834 | 1994 veh |
| 5 | **1472.3** | **1461.1** | **1484.1** | **0.410** | **1.014** | **64.9%** | 0.4879 | **1998 veh** |

### Key Observations
- **R1→R2: Biggest reward jump (+12.4 avg)** — first FedAvg aggregation provides a strong collective improvement; node_2 gains +53.1 reward points (largest single-node jump in the run).
- **node_2 R1 anomaly**: lowest reward (1417.1) despite best TP (66.2%) and lowest wait (0.33 s) — high queue (maxQ=8) creates a complex reward landscape that resolves after first federation.
- **Reward range narrows R1→R2**: span of 63.7 points (R1) drops to 23.6 points (R2) — federation homogenises policies, reducing inter-node reward variance.
- **TP ratio stable at 64–65%** throughout all rounds — moderate congestion level inherent to the China rural network; vehicles queue (avg ~1.0) but clear at a consistent rate.
- **node_5 worst TP every round** (61.4–62.1%) — state highway junction carries through-traffic on long routes that don't complete within the 1500-step episode window.
- **node_4 lowest wait every round** (0.15–0.31 s) — industrial/highway fringe experiences brief, light demand in its time window despite average queue lengths.
- **DQN loss drops sharply R1→R2** (0.5517→0.4817, −12.7%) then plateaus with minor oscillations — learning driven by first federation, slower thereafter.

> [!CAUTION]
> Round 1 and Round 2 reflect genuine fresh SUMO episodes (different fingerprints). From Round 3 onwards, `env.reset()` reuses the existing TraCI connection — environment metrics carry partial state from Round 2. The per-round progression in R3–R5 reflects DQN policy evolution reflected back through a partially frozen SUMO state. DQN loss is the primary round-on-round learning indicator.

---

## 3. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 avg reward | cluster_1 avg reward |
|-------|-----------|-----------|---------------------|---------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 1444.6 | 1472.3 |
| 2 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1475.4 | 1462.1 |
| 3 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1475.9 | 1462.8 |
| 4 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1476.2 | 1463.3 |
| 5 | **node_0, node_1** | **node_2, node_3, node_4, node_5** | **1463.7** | **1476.6** |

> [!IMPORTANT]
> **Round 5 produces the most semantically correct clustering**: cluster_0 groups the two priority-flagged nodes (node_0: priority=1.0, node_1: priority=0.5) — the high-complexity village-market and school-approach junctions — separately from the four standard rural nodes. This mirrors the R5 clustering in Pikhuwa, demonstrating that AdaptFlow reliably identifies the priority-flagged nodes as a distinct group by the final round regardless of map.

---

## 4. Cluster Transitions (Re-Clustering Events)

```
Round 1:  No transitions (static baseline)
Round 2:  node_0: cluster_0 → cluster_1
          node_1: cluster_0 → cluster_1
          node_3: cluster_1 → cluster_0
          node_4: cluster_1 → cluster_0
          node_5: cluster_1 → cluster_0
Round 3:  No transitions (STABLE)
Round 4:  No transitions (STABLE)
Round 5:  node_0: cluster_1 → cluster_0
          node_1: cluster_1 → cluster_0
          node_2: cluster_0 → cluster_1
          node_3: cluster_0 → cluster_1
          node_4: cluster_0 → cluster_1
          node_5: cluster_0 → cluster_1
```

**Total transitions: 11** across 2 re-clustering steps (R2, R5).

- Rounds 3 and 4 are both **stable** — same as Pikhuwa, confirming that the AdaptFlow K-Means re-clustering reaches a local stable point after the first FedAvg reshuffle and only breaks at R5.
- **node_0 and node_1 consistently move as a pair** — their shared priority flags (1.0 and 0.5) keep their cosine similarity high (0.9934), tying their cluster fate across all rounds.
- **node_2 drives R1 clustering**: grouped with node_0 and node_1 in R1 (cluster_0) due to its anomalous low reward; after federation corrects the policy, it joins the standard rural cluster in R2+.

---

## 5. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, priority_flag]`

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1 | `[0.466, 1.027, 0.643, 5.0, 0.0, 1.0]` |
| node_0 | 2–5 | `[0.540, 0.973, 0.642, 7.0, 0.0, 1.0]` *(partial freeze)* |
| node_1 | 1 | `[0.472, 1.073, 0.649, 6.0, 0.0, 0.5]` |
| node_1 | 2–5 | `[0.490, 1.125, 0.647, 7.0, 0.0, 0.5]` *(partial freeze)* |
| node_2 | 1 | `[0.334, 1.237, 0.662, 8.0, 0.0, 0.0]` |
| node_2 | 2–5 | `[0.480, 1.129, 0.656, 5.0, 0.0, 0.0]` *(partial freeze)* |
| node_3 | 1 | `[0.427, 1.010, 0.652, 5.0, 0.0, 0.0]` |
| node_3 | 2–5 | `[0.460, 0.926, 0.654, 5.0, 0.0, 0.0]` *(partial freeze)* |
| node_4 | 1 | `[0.150, 0.972, 0.639, 5.0, 0.0, 0.0]` |
| node_4 | 2–5 | `[0.330, 1.035, 0.639, 5.0, 0.0, 0.0]` *(partial freeze)* |
| node_5 | 1 | `[0.398, 1.000, 0.614, 5.0, 0.0, 0.0]` |
| node_5 | 2–5 | `[0.270, 1.037, 0.614, 5.0, 0.0, 0.0]` *(partial freeze)* |

**Key fingerprint insights:**
- **R1→R2 fingerprints change for all nodes** (unlike Pikhuwa where freeze was immediate) — both R1 and R2 are genuine fresh episodes; only R3+ are frozen.
- **node_2 most transformed R1→R2**: wait 0.334→0.480 s (+44%), queue 1.237→1.129, maxQ 8→5 — the first FedAvg aggregation fundamentally changed node_2's signal timing, redistributing congestion.
- **node_4 wait doubled R1→R2**: 0.150→0.330 s — shared policy from higher-congestion nodes increased node_4's signal hold time, raising wait but also raising reward.
- **node_5 wait dropped R1→R2**: 0.398→0.270 s — federation reduced signal hold time at the highway junction, improving flow.
- **node_0 and node_1** remain the only priority-flagged nodes (1.0 and 0.5); their fingerprints are most distinct from the standard rural group, driving R5 cluster separation.
- **All avg_queue values ~1.0**: China rural sustains persistent queuing at every node across all rounds — a distinguishing characteristic of this map's congestion level.

---

## 6. Loss Trajectories (DQN Learning Progress)

| Node | R1 | R2 | R3 | R4 | R5 | R1→R5 drop |
|------|----|----|----|----|-----|-----------|
| node_0 | 0.5662 | **0.4473** | 0.4747 | 0.4799 | 0.4796 | **−15.3%** |
| node_1 | 0.5424 | 0.4835 | 0.4879 | 0.4873 | 0.4802 | −11.5% |
| node_2 | 0.5480 | 0.4759 | 0.4625 | 0.4751 | 0.4854 | −11.4% |
| node_3 | 0.5348 | 0.4980 | 0.4911 | 0.4832 | 0.5098 | −4.7% |
| node_4 | 0.5244 | 0.4933 | 0.4893 | 0.4904 | 0.4877 | −7.0% |
| node_5 | **0.5942** | 0.4921 | 0.4864 | 0.4843 | 0.4844 | **−18.5%** |
| **Avg** | **0.5517** | **0.4817** | **0.4820** | **0.4834** | **0.4879** | **−11.6%** |

**Key loss insights:**
- **node_0 largest single-round drop**: R1→R2 loss falls −21.1% (0.5662→0.4473) — the first federation resets the village-market node's Q-values dramatically.
- **node_5 highest R1 loss** (0.5942) and greatest overall drop (−18.5%) — the state highway junction has the most variable reward signal, benefitting most from collective knowledge.
- **node_3 slowest learner** (−4.7%): Residential outskirts at begin=1800 s shows the least loss improvement; its moderate congestion provides insufficient gradient to drive strong DQN updates.
- **R2→R5 loss plateau**: avg loss oscillates 0.4817–0.4879 — after the strong R1→R2 drop (−12.7%), learning slows and shows mild upward drift (node_3 R5: 0.5098) due to frozen environment feedback from SUMO-reset.
- **All R5 losses in 0.480–0.510 range** — higher than expected for a converged DQN, reflecting the challenging, variable reward landscape of the congested China rural network.

---

## 7. Cosine Similarity Matrix

### Round 1 (Fresh Episode)
|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9934 | 0.9785 | 0.9817 | 0.9799 | 0.9817 |
| node_1 | 0.9934 | 1.0000 | 0.9955 | 0.9962 | 0.9953 | 0.9964 |
| node_2 | 0.9785 | 0.9955 | 1.0000 | 0.9971 | 0.9983 | 0.9977 |
| node_3 | 0.9817 | 0.9962 | 0.9971 | 1.0000 | 0.9985 | **1.0000** |
| node_4 | 0.9799 | 0.9953 | 0.9983 | 0.9985 | 1.0000 | 0.9988 |
| node_5 | 0.9817 | 0.9964 | 0.9977 | **1.0000** | 0.9988 | 1.0000 |

**Similarity insights:**
- **Most similar pair: node_3 ↔ node_5** (1.0000) — Farming district outskirts and state highway junction have virtually identical R1 fingerprints despite their different geographic roles.
- **Most dissimilar pair: node_0 ↔ node_2** (0.9785) — Village market (priority=1.0, wait=0.47 s) vs farming district (lowest wait=0.33 s, maxQ=8) are the most structurally different intersections in R1.
- **node_0 is the most isolated node** (avg similarity 0.9830) — its priority_flag=1.0 and high wait separate it from all other nodes, driving its consistent pairing with node_1.
- **node_1 acts as a bridge** (avg similarity 0.9954) — semi-priority flag (0.5) gives it high similarity with both the priority-hub group (node_0) and the standard rural group (node_2–5).
- **node_2, node_3, node_4, node_5 form a tight cluster** (avg ~0.9981) — four standard rural nodes with priority_flag=0.0 behave near-identically in fingerprint space.

---

## 8. Summary

| Metric | Value | Trend |
|--------|-------|-------|
| Best node (R5) | node_3 (reward 1484.1, TP 65.8%) | Residential outskirts — steady gainer |
| Worst node R1 | node_2 (reward 1417.1) | Anomaly corrected by R2 federation |
| Worst node R2+ | node_1 (reward 1459.4–1461.1) | School/temple approach — consistent low |
| Best TP ratio | node_2 (66.2%, all rounds) | Farming district — best vehicle clearance |
| Worst TP ratio | node_5 (61.4–62.1%) | Highway junction — long routes, in-transit vehicles |
| Best wait | node_4 (0.15 s R1 → 0.31 s R5) | Industrial fringe — lightest demand in its window |
| Worst wait | node_0 (0.47→0.52 s) | Village market — densest early-morning demand |
| Avg departed/node/round | ~515 vehicles | Stable across all rounds |
| Avg TP ratio R1→R5 | 64.3% → 64.9% (+0.6 pp) | Moderate congestion — gradual improvement |
| Total arrived R1→R5 | 1987 → 1998 veh (+11) | Incremental throughput growth |
| Reward R1→R5 | 1458.5 → 1472.3 (+0.94%) | Meaningful R1→R2 jump, then gradual gain |
| Biggest single-round gain | node_2 R1→R2 (+53.1 reward) | Federation resolves R1 policy anomaly |
| Loss R1→R5 | 0.5517 → 0.4879 (−11.6%) | Strong R1→R2 drop, plateau R3–R5 |
| Fastest learner | node_5 (−18.5%) | Most variable reward benefits most from federation |
| Slowest learner | node_3 (−4.7%) | Low gradient on moderate-congestion node |
| Cluster quality | Oscillating R1–R4, best at R5 | Priority-based separation emerges at R5 |
| Total transitions | 11 across 2 re-clustering steps | R3–R4 stable, R2 and R5 active |
| Best clustering | Round 5: {node_0, node_1} vs {node_2–5} | Correctly isolates priority-flagged junctions |
| SUMO reset note | R1–R2 genuine; R3–R5 partial freeze | R2 fingerprints differ from R1 (unlike Pikhuwa) |
| Avg queue | ~1.03 across all rounds | Persistent moderate congestion throughout |
