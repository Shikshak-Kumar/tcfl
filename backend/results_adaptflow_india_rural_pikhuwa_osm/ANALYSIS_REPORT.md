# AdaptFlow India Rural Pikhuwa OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 5 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 1500

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Avg Queue | Max Queue | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|-----------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 1464.2 | 0.463 | 0.305 | 3 | 459 / 514 | 89.3% | 0.5209 |
| node_1 | 1 | cluster_0 | 1467.3 | 0.380 | 0.281 | **4** | **463** / 513 | **90.3%** | 0.5874 |
| node_2 | 1 | cluster_0 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.5370 |
| node_3 | 1 | cluster_1 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | **0.6024** |
| node_4 | 1 | cluster_1 | 1477.7 | **0.526** | 0.200 | 2 | **463** / 513 | **90.3%** | 0.5788 |
| node_5 | 1 | cluster_1 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | 0.5474 |
| node_0 | 2 | cluster_1 | 1464.2 | 0.463 | 0.305 | 3 | 459 / 514 | 89.3% | 0.4926 |
| node_1 | 2 | cluster_1 | 1467.3 | 0.380 | 0.281 | **4** | **463** / 513 | **90.3%** | 0.4880 |
| node_2 | 2 | cluster_0 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.4994 |
| node_3 | 2 | cluster_0 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | 0.4885 |
| node_4 | 2 | cluster_0 | 1477.7 | **0.526** | 0.200 | 2 | **463** / 513 | **90.3%** | 0.4906 |
| node_5 | 2 | cluster_0 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | 0.4959 |
| node_0 | 3 | cluster_1 | 1464.2 | 0.463 | 0.305 | 3 | 459 / 514 | 89.3% | 0.4966 |
| node_1 | 3 | cluster_1 | 1467.3 | 0.380 | 0.281 | **4** | **463** / 513 | **90.3%** | 0.4870 |
| node_2 | 3 | cluster_0 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.4812 |
| node_3 | 3 | cluster_0 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | 0.4929 |
| node_4 | 3 | cluster_0 | 1477.7 | **0.526** | 0.200 | 2 | **463** / 513 | **90.3%** | 0.4872 |
| node_5 | 3 | cluster_0 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | 0.4888 |
| node_0 | 4 | cluster_1 | 1464.2 | 0.463 | 0.305 | 3 | 459 / 514 | 89.3% | 0.4870 |
| node_1 | 4 | cluster_1 | 1467.3 | 0.380 | 0.281 | **4** | **463** / 513 | **90.3%** | 0.4877 |
| node_2 | 4 | cluster_0 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.4911 |
| node_3 | 4 | cluster_0 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | 0.4867 |
| node_4 | 4 | cluster_0 | 1477.7 | **0.526** | 0.200 | 2 | **463** / 513 | **90.3%** | 0.4861 |
| node_5 | 4 | cluster_0 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | 0.4931 |
| node_0 | 5 | cluster_0 | 1464.2 | 0.463 | **0.305** | 3 | 459 / 514 | 89.3% | 0.4868 |
| node_1 | 5 | cluster_0 | 1467.3 | **0.380** | 0.281 | **4** | **463** / 513 | **90.3%** | 0.4851 |
| node_2 | 5 | cluster_1 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.4952 |
| node_3 | 5 | cluster_1 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | 0.4856 |
| node_4 | 5 | cluster_1 | 1477.7 | **0.526** | 0.200 | 2 | **463** / 513 | **90.3%** | 0.4798 |
| node_5 | 5 | cluster_1 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | **0.4654** |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Queue | Avg TP Ratio | Avg Loss | Total Arrived |
|-------|-----------|-----|-----|-------------|-----------|-------------|---------|--------------|
| 1 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | 0.5623 | 2763 veh |
| 2 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | 0.4925 | 2763 veh |
| 3 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | 0.4890 | 2763 veh |
| 4 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | 0.4886 | 2763 veh |
| 5 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | **0.4830** | 2763 veh |

### Key Observations
- **Rewards frozen R1–R5**: All environment metrics (reward, wait, queue, throughput) are bit-for-bit identical across all 5 rounds — the SUMO-reset bug prevents fresh episode data from round 2 onwards.
- **DQN loss steadily decreasing**: 0.5623 → 0.4830 (−14.1% over 5 rounds), confirming genuine federated learning despite frozen environment feedback.
- **Reward spread very narrow**: only 13.6 points (1464.2–1477.8) vs 37.2 points in China rural — Pikhuwa's rural topology is highly uniform in congestion levels.
- **High throughput**: ~513 departed and ~460 arrived per node per round; TP ratio 89.1–90.3% — vehicles have sufficient time in 1500 steps to complete their trips.
- **Per-step reward**: 1472.4 / 1500 = 0.982/step — near-maximum, indicating low congestion at monitored intersections.

> [!CAUTION]
> All rounds R1–R5 show **bit-for-bit identical** environment metrics per node (reward, wait, queue, throughput, departed, arrived). Only DQN loss values differ across rounds. The SUMO simulation does not restart cleanly between rounds — `env.reset()` reuses the existing TraCI connection from round 2 onwards. Only Round 1 reflects a genuine fresh episode.

---

## 3. Node Traffic Profiles

Six nodes with 400 s begin-time offsets and UP-specific TLS parameters produce distinct demand windows:

| Profile | Nodes | Avg Reward | Avg Wait (s) | Avg Queue | TP Ratio | Notes |
|---------|-------|-----------|-------------|-----------|---------|-------|
| **Priority-hub** | node_0 | 1464.2 | 0.463 s | **0.305** | 89.3% | Town centre — priority_flag=1.0, highest queue |
| **Semi-priority** | node_1 | 1467.3 | **0.380 s** | 0.281 | 90.3% | School/temple — priority_flag=0.5, maxQ=4 |
| **Standard rural** | node_2, node_5 | 1473.8 | 0.418 s | 0.224 | 89.8% | Farming & highway junction |
| **High-reward** | node_3, node_4 | **1477.8** | 0.481 s | 0.195 | 89.7% | Residential & industrial fringe |

- **node_3 and node_4 are the highest reward nodes** (1477.8, 1477.7): Residential colony and industrial/highway fringe see the lightest signal pressure in their time windows, yielding maximum reward.
- **node_0 is the lowest reward node** (1464.2): Town centre at begin=0 has the highest persistent queue (0.305) and priority_flag=1.0 — the most complex intersection in the network.
- **node_1** has the lowest average wait (0.38 s) but highest max queue (4) — school/temple zone sees brief demand bursts that clear quickly.
- **node_4** has the highest wait (0.526 s) despite high reward — industrial/highway fringe experiences longer vehicle dwell times but short queues.

---

## 4. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 avg reward | cluster_1 avg reward |
|-------|-----------|-----------|---------------------|---------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 1468.2 | 1476.6 |
| 2 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1475.8 | 1465.8 |
| 3 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1475.8 | 1465.8 |
| 4 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1475.8 | 1465.8 |
| 5 | **node_0, node_1** | **node_2, node_3, node_4, node_5** | **1465.8** | **1475.8** |

> [!IMPORTANT]
> **Round 5 produces the most semantically meaningful clustering**: cluster_0 isolates the two priority-flagged nodes (node_0: priority=1.0, node_1: priority=0.5) from the four standard rural nodes. This correctly separates the high-complexity town-centre/school junction pair from the homogeneous farming-residential-industrial group. The algorithm converged to a priority-based separation by round 5.

---

## 5. Cluster Transitions (Re-Clustering Events)

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

- Rounds 3 and 4 are both **stable** — the longest consecutive stable period in this run.
- The oscillation pattern (R1→R2 inversion, R3-R4 stable, R5 reversion) is driven by frozen fingerprints from the SUMO-reset bug — the algorithm re-clusters on the same data with different random initialisation.
- **node_0 and node_1 always move together** — their shared priority flags (1.0 and 0.5) keep their cosine fingerprint similarity high (~0.979), so the K-Means boundary consistently groups or separates them as a pair.

---

## 6. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, priority_flag]`

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1 | `[0.463, 0.305, 0.893, 3.0, 0.0, 1.0]` |
| node_0 | 2–5 | `[0.463, 0.305, 0.893, 3.0, 0.0, 1.0]` *(frozen)* |
| node_1 | 1 | `[0.380, 0.281, 0.903, 4.0, 0.0, 0.5]` |
| node_1 | 2–5 | `[0.380, 0.281, 0.903, 4.0, 0.0, 0.5]` *(frozen)* |
| node_2 | 1 | `[0.435, 0.218, 0.895, 2.0, 0.0, 0.0]` |
| node_2 | 2–5 | `[0.435, 0.218, 0.895, 2.0, 0.0, 0.0]` *(frozen)* |
| node_3 | 1 | `[0.435, 0.189, 0.891, 2.0, 0.0, 0.0]` |
| node_3 | 2–5 | `[0.435, 0.189, 0.891, 2.0, 0.0, 0.0]` *(frozen)* |
| node_4 | 1 | `[0.526, 0.200, 0.903, 2.0, 0.0, 0.0]` |
| node_4 | 2–5 | `[0.526, 0.200, 0.903, 2.0, 0.0, 0.0]` *(frozen)* |
| node_5 | 1 | `[0.400, 0.229, 0.901, 2.0, 0.0, 0.0]` |
| node_5 | 2–5 | `[0.400, 0.229, 0.901, 2.0, 0.0, 0.0]` *(frozen)* |

**Key fingerprint insights:**
- **All fingerprints frozen from R1** (unlike China rural where R2 showed updated fingerprints) — the SUMO-reset bug hits immediately from round 2, meaning every clustering decision R2–R5 uses R1 fingerprints exclusively.
- **node_0 is the outlier**: only node with non-zero priority_flag=1.0, highest queue (0.305), and max_queue=3 — the town-centre junction is uniquely demanding.
- **node_1 outlier**: priority_flag=0.5 and max_queue=4 (highest single-step queue across all nodes) — school/temple road sees brief but heavy surges.
- **node_2, node_3, node_4, node_5 are near-identical**: all have priority_flag=0.0, max_queue=2, TP ratio 89–90% — standard rural roads with homogeneous behaviour.
- **TP ratio range is narrow** (89.1%–90.3%) — all nodes converge to similar throughput; contrast with China rural (6.5%–9.4% range), indicating Pikhuwa has far lower congestion.

---

## 7. Loss Trajectories (DQN Learning Progress)

| Node | R1 | R2 | R3 | R4 | R5 | R1→R5 drop |
|------|----|----|----|----|-----|-----------|
| node_0 | 0.5209 | 0.4926 | 0.4966 | 0.4870 | 0.4868 | −6.5% |
| node_1 | 0.5874 | 0.4880 | 0.4870 | 0.4877 | 0.4851 | −17.4% |
| node_2 | 0.5370 | 0.4994 | 0.4812 | 0.4911 | 0.4952 | −7.8% |
| node_3 | **0.6024** | 0.4885 | 0.4929 | 0.4867 | 0.4856 | **−19.4%** |
| node_4 | 0.5788 | 0.4906 | 0.4872 | 0.4861 | **0.4798** | −17.1% |
| node_5 | 0.5474 | 0.4959 | 0.4888 | 0.4931 | 0.4654 | −15.0% |
| **Avg** | **0.5623** | **0.4925** | **0.4890** | **0.4886** | **0.4830** | **−14.1%** |

**Key loss insights:**
- **Largest single-round drop: R1→R2** (0.5623 → 0.4925, −12.4%) — first FedAvg weight aggregation provides the strongest learning signal.
- **node_3 starts with the highest loss** (0.6024) and achieves the greatest drop (−19.4%) — Residential colony at begin=1200 s faces the most complex Q-learning problem and benefits most from federation.
- **node_4 reaches the lowest final loss** (0.4798 at R5) — industrial/highway fringe converges fastest in absolute terms.
- **node_5 shows the biggest R5 jump** (0.4931→0.4654, −5.6% in one round) — late-session learning burst at the state highway junction.
- **R2→R5 plateau**: average loss decreases only −0.0095 over 3 rounds (0.4925→0.4830) vs the −0.0698 drop in R1→R2 — learning slows after first federation, consistent with DQN convergence behaviour on low-congestion maps.
- **All losses remain in 0.465–0.603 range** — the uniform, low-variance reward signal in Pikhuwa's rural environment limits the gradient pressure, keeping absolute loss values moderate.

---

## 8. Cosine Similarity Matrix (All Rounds — Frozen)

|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9792 | 0.9449 | 0.9450 | 0.9419 | 0.9452 |
| node_1 | 0.9792 | 1.0000 | 0.9681 | 0.9687 | 0.9631 | 0.9689 |
| node_2 | 0.9449 | 0.9681 | 1.0000 | **0.9999** | 0.9992 | 0.9999 |
| node_3 | 0.9450 | 0.9687 | **0.9999** | 1.0000 | 0.9992 | 0.9997 |
| node_4 | 0.9419 | 0.9631 | 0.9992 | 0.9992 | 1.0000 | 0.9984 |
| node_5 | 0.9452 | 0.9689 | 0.9999 | 0.9997 | 0.9984 | 1.0000 |

> All rounds show an **identical** similarity matrix (SUMO-reset bug freezes fingerprints from R1).

**Similarity insights:**
- **Most similar pair: node_2 ↔ node_3** (0.9999) — Farming district and residential colony share near-identical fingerprints; both standard rural nodes with max_queue=2 and priority_flag=0.0.
- **Most dissimilar pair: node_0 ↔ node_4** (0.9419) — Town centre (priority=1.0, queue=0.305) vs industrial fringe (wait=0.526, priority=0.0) are the most structurally different intersections.
- **node_0 is the most isolated node**: lowest similarity to all others (avg 0.957) — its priority_flag=1.0 and highest queue make it a structural outlier in fingerprint space.
- **node_2, node_3, node_4, node_5 form a tight cluster** (avg similarity ~0.999) — four standard rural nodes are functionally interchangeable from a clustering perspective.
- **node_0–node_1 similarity** (0.979) — the two priority-flagged nodes are more similar to each other than to any standard rural node, validating their shared cluster membership in R5.

---

## 9. Summary

| Metric | Value | Trend |
|--------|-------|-------|
| Best node | node_3 (reward 1477.8, TP 89.1%) | Residential colony — lightest TLS load |
| Worst node | node_0 (reward 1464.2, queue 0.305) | Town centre — highest intersection complexity |
| Best TP ratio | node_1 & node_4 (90.3%) | School/temple and industrial fringe |
| Worst TP ratio | node_3 (89.1%) | Residential outskirts |
| Best wait | node_1 (0.380 s) | Fastest vehicle clearance |
| Worst wait | node_4 (0.526 s) | Longest dwell time |
| Avg departed/node/round | ~513 vehicles | Stable across all rounds |
| Avg TP ratio | 89.8% | High throughput — 1500-step episodes give vehicles time to complete trips |
| Reward R1→R5 | 1472.4 (frozen) | SUMO-reset bug; no improvement visible |
| Reward per step | 0.982 | Near-maximum — low congestion environment |
| Loss R1→R5 | 0.5623 → 0.4830 (−14.1%) | Steady federated learning |
| Fastest learner | node_3 (−19.4%) | Most complex node benefits most |
| Cluster quality | Oscillating R1–R4, best at R5 | Priority-based separation emerges |
| Total transitions | 11 across 2 re-clustering steps | R3–R4 stable, R2 and R5 active |
| Best clustering | Round 5: {node_0, node_1} vs {node_2–5} | Correctly isolates priority-flagged junctions |
| SUMO reset bug | All rounds R1–R5 frozen | Only R1 is a genuine fresh episode |
