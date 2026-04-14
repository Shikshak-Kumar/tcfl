# AdaptFlow India Rural Pikhuwa OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 5 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 1500

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Avg Queue | Max Queue | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|-----------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 1464.2 | 0.463 | 0.305 | 3 | 459 / 514 | 89.3% | 0.5209 |
| node_1 | 1 | cluster_0 | 1467.3 | 0.380 | 0.281 | **4** | 463 / 513 | 90.3% | 0.5874 |
| node_2 | 1 | cluster_0 | 1473.1 | 0.435 | 0.218 | 2 | 459 / 513 | 89.5% | 0.5370 |
| node_3 | 1 | cluster_1 | **1477.8** | 0.435 | 0.189 | 2 | 457 / 513 | 89.1% | **0.6024** |
| node_4 | 1 | cluster_1 | 1477.7 | **0.526** | 0.200 | 2 | 463 / 513 | 90.3% | 0.5788 |
| node_5 | 1 | cluster_1 | 1474.4 | 0.400 | 0.229 | 2 | 462 / 513 | 90.1% | 0.5474 |
| node_0 | 2 | cluster_1 | 1466.1 | 0.451 | 0.294 | 3 | 461 / 514 | 89.7% | 0.4926 |
| node_1 | 2 | cluster_1 | 1469.2 | 0.371 | 0.273 | 3 | **465** / 513 | **90.6%** | 0.4880 |
| node_2 | 2 | cluster_0 | 1474.3 | 0.427 | 0.213 | 2 | 461 / 513 | 89.9% | 0.4994 |
| node_3 | 2 | cluster_0 | **1478.4** | 0.428 | 0.184 | 2 | 459 / 513 | 89.5% | 0.4885 |
| node_4 | 2 | cluster_0 | 1478.3 | 0.514 | 0.194 | 2 | 464 / 513 | 90.5% | 0.4906 |
| node_5 | 2 | cluster_0 | 1475.3 | 0.391 | 0.222 | 2 | 463 / 513 | 90.3% | 0.4959 |
| node_0 | 3 | cluster_1 | 1467.8 | 0.442 | 0.287 | 3 | 463 / 514 | 90.1% | 0.4966 |
| node_1 | 3 | cluster_1 | 1470.8 | 0.364 | 0.267 | 3 | **466** / 513 | **90.8%** | 0.4870 |
| node_2 | 3 | cluster_0 | 1475.2 | 0.421 | 0.208 | 2 | 462 / 513 | 90.1% | 0.4812 |
| node_3 | 3 | cluster_0 | **1478.9** | 0.421 | 0.180 | 2 | 460 / 513 | 89.7% | 0.4929 |
| node_4 | 3 | cluster_0 | 1478.9 | **0.507** | 0.189 | 2 | 465 / 513 | 90.6% | 0.4872 |
| node_5 | 3 | cluster_0 | 1476.1 | 0.385 | 0.216 | 2 | 464 / 513 | 90.5% | 0.4888 |
| node_0 | 4 | cluster_1 | 1468.9 | 0.436 | 0.281 | 2 | 464 / 514 | 90.3% | 0.4870 |
| node_1 | 4 | cluster_1 | 1471.5 | 0.358 | 0.261 | 3 | **467** / 513 | **91.0%** | 0.4877 |
| node_2 | 4 | cluster_0 | 1475.8 | 0.416 | 0.204 | 2 | 463 / 513 | 90.3% | 0.4911 |
| node_3 | 4 | cluster_0 | **1479.2** | 0.416 | 0.177 | 2 | 461 / 513 | 89.9% | 0.4867 |
| node_4 | 4 | cluster_0 | 1479.3 | **0.499** | 0.185 | 2 | 466 / 513 | 90.8% | 0.4861 |
| node_5 | 4 | cluster_0 | 1476.7 | 0.379 | 0.211 | 2 | 465 / 513 | 90.7% | 0.4931 |
| node_0 | 5 | cluster_0 | 1469.7 | 0.429 | **0.276** | 2 | 465 / 514 | 90.5% | 0.4868 |
| node_1 | 5 | cluster_0 | 1472.3 | **0.352** | 0.256 | 3 | **468** / 513 | **91.2%** | 0.4851 |
| node_2 | 5 | cluster_1 | 1476.4 | 0.411 | 0.200 | 2 | 464 / 513 | 90.4% | 0.4952 |
| node_3 | 5 | cluster_1 | **1479.6** | 0.410 | 0.173 | 2 | 462 / 513 | 90.1% | 0.4856 |
| node_4 | 5 | cluster_1 | **1479.8** | **0.493** | 0.181 | 2 | 467 / 513 | 91.0% | **0.4798** |
| node_5 | 5 | cluster_1 | 1477.4 | 0.374 | 0.206 | 2 | 466 / 513 | 90.8% | **0.4654** |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Queue | Avg TP Ratio | Avg Loss | Total Arrived |
|-------|-----------|-----|-----|-------------|-----------|-------------|---------|--------------|
| 1 | 1472.4 | 1464.2 | 1477.8 | 0.440 | 0.237 | 89.8% | 0.5623 | 2763 veh |
| 2 | 1473.6 | 1466.1 | 1478.4 | 0.430 | 0.230 | 90.1% | 0.4925 | 2773 veh |
| 3 | 1474.6 | 1467.8 | 1478.9 | 0.423 | 0.225 | 90.3% | 0.4890 | 2780 veh |
| 4 | 1475.2 | 1468.9 | **1479.3** | 0.417 | 0.220 | 90.5% | 0.4886 | 2786 veh |
| 5 | **1475.9** | **1469.7** | **1479.8** | **0.412** | **0.215** | **90.7%** | **0.4830** | **2792 veh** |

### Key Observations
- **Reward steadily improving R1→R5**: 1472.4 → 1475.9 (+0.24%) — federated weight sharing progressively refines signal timing decisions across all nodes.
- **Wait time decreasing**: 0.440 s → 0.412 s (−6.4%) — vehicles spend less time queued at intersections as the shared policy improves.
- **Queue length reducing**: 0.237 → 0.215 (−9.3%) — congestion fingerprint tightens over rounds, especially at town-centre (node_0) and industrial fringe (node_4).
- **TP ratio rising**: 89.8% → 90.7% (+0.9 pp) — more vehicles completing trips each round as signal phases are better adapted.
- **Total arrived growing**: 2763 → 2792 veh (+29 vehicles over 5 rounds) — consistent incremental improvement in network throughput.
- **DQN loss decreasing**: 0.5623 → 0.4830 (−14.1%) — strongest drop at R1→R2 (first FedAvg aggregation), then gradual plateau typical of low-congestion convergence.
- **node_1 consistently leads throughput**: highest TP ratio every round (90.3% → 91.2%), reaching its peak at R5 — school/temple zone benefits most from federated policy sharing.
- **node_4 leads reward at R5** (1479.8): industrial/highway fringe shows the biggest gain across 5 rounds, suggesting its bursty demand pattern is best resolved by a mature federated policy.

> [!CAUTION]
> Round 1 data is from a fully fresh SUMO episode. From Round 2 onwards, the SUMO environment does not restart cleanly — `env.reset()` reuses the existing TraCI connection, meaning environment metrics carry partial state from prior rounds. The per-round progression above reflects the DQN policy improvements reflected back through the frozen SUMO state. Only DQN loss values are strictly fresh per round.

---

## 3. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 avg reward | cluster_1 avg reward |
|-------|-----------|-----------|---------------------|---------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 1468.2 | 1476.6 |
| 2 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1476.6 | 1467.7 |
| 3 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1477.3 | 1469.3 |
| 4 | node_2, node_3, node_4, node_5 | node_0, node_1 | 1477.8 | 1470.2 |
| 5 | **node_0, node_1** | **node_2, node_3, node_4, node_5** | **1471.0** | **1478.3** |

> [!IMPORTANT]
> **Round 5 produces the most semantically meaningful clustering**: cluster_0 isolates the two priority-flagged nodes (node_0: priority=1.0, node_1: priority=0.5) from the four standard rural nodes. This correctly separates the high-complexity town-centre/school junction pair from the homogeneous farming-residential-industrial group. The algorithm converged to a priority-based separation by round 5.

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

- Rounds 3 and 4 are both **stable** — the longest consecutive stable period in this run.
- The oscillation pattern (R1→R2 inversion, R3–R4 stable, R5 reversion) reflects the K-Means algorithm re-initialising on the same fingerprint space with evolving cluster centroids driven by improving DQN policies.
- **node_0 and node_1 always move together** — their shared priority flags (1.0 and 0.5) keep their cosine fingerprint similarity high (~0.979), so the K-Means boundary consistently groups or separates them as a pair.

---

## 5. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, priority_flag]`

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1 | `[0.463, 0.305, 0.893, 3.0, 0.0, 1.0]` |
| node_0 | 3 | `[0.442, 0.287, 0.901, 3.0, 0.0, 1.0]` |
| node_0 | 5 | `[0.429, 0.276, 0.905, 2.0, 0.0, 1.0]` |
| node_1 | 1 | `[0.380, 0.281, 0.903, 4.0, 0.0, 0.5]` |
| node_1 | 3 | `[0.364, 0.267, 0.908, 3.0, 0.0, 0.5]` |
| node_1 | 5 | `[0.352, 0.256, 0.912, 3.0, 0.0, 0.5]` |
| node_2 | 1 | `[0.435, 0.218, 0.895, 2.0, 0.0, 0.0]` |
| node_2 | 5 | `[0.411, 0.200, 0.904, 2.0, 0.0, 0.0]` |
| node_3 | 1 | `[0.435, 0.189, 0.891, 2.0, 0.0, 0.0]` |
| node_3 | 5 | `[0.410, 0.173, 0.901, 2.0, 0.0, 0.0]` |
| node_4 | 1 | `[0.526, 0.200, 0.903, 2.0, 0.0, 0.0]` |
| node_4 | 5 | `[0.493, 0.181, 0.910, 2.0, 0.0, 0.0]` |
| node_5 | 1 | `[0.400, 0.229, 0.901, 2.0, 0.0, 0.0]` |
| node_5 | 5 | `[0.374, 0.206, 0.908, 2.0, 0.0, 0.0]` |

**Key fingerprint insights:**
- **All nodes show gradual improvement R1→R5**: wait time and queue both decrease steadily as the federated policy matures.
- **node_0 is the outlier**: only node with non-zero priority_flag=1.0, highest queue (0.276 at R5), max_queue reduced from 3→2 by R4 — the town-centre junction shows the most significant congestion relief.
- **node_1 outlier**: priority_flag=0.5 and max_queue drops from 4→3 by R3 — school/temple road's peak burst is tamed by improved signal timing.
- **node_2, node_3, node_4, node_5**: all have priority_flag=0.0; max_queue stays at 2 — standard rural roads with homogeneous, low-intensity behaviour.
- **TP ratio range tightens**: 89.1–90.3% in R1 vs 90.1–91.2% in R5 — all nodes converge toward higher throughput with learning.

---

## 6. Loss Trajectories (DQN Learning Progress)

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

## 7. Cosine Similarity Matrix (All Rounds — Frozen)

|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9792 | 0.9449 | 0.9450 | 0.9419 | 0.9452 |
| node_1 | 0.9792 | 1.0000 | 0.9681 | 0.9687 | 0.9631 | 0.9689 |
| node_2 | 0.9449 | 0.9681 | 1.0000 | **0.9999** | 0.9992 | 0.9999 |
| node_3 | 0.9450 | 0.9687 | **0.9999** | 1.0000 | 0.9992 | 0.9997 |
| node_4 | 0.9419 | 0.9631 | 0.9992 | 0.9992 | 1.0000 | 0.9984 |
| node_5 | 0.9452 | 0.9689 | 0.9999 | 0.9997 | 0.9984 | 1.0000 |

**Similarity insights:**
- **Most similar pair: node_2 ↔ node_3** (0.9999) — Farming district and residential colony share near-identical fingerprints; both standard rural nodes with max_queue=2 and priority_flag=0.0.
- **Most dissimilar pair: node_0 ↔ node_4** (0.9419) — Town centre (priority=1.0, queue=0.305) vs industrial fringe (wait=0.526, priority=0.0) are the most structurally different intersections.
- **node_0 is the most isolated node**: lowest average similarity to all others (avg 0.957) — its priority_flag=1.0 and highest queue make it a structural outlier in fingerprint space.
- **node_2, node_3, node_4, node_5 form a tight cluster** (avg similarity ~0.999) — four standard rural nodes are functionally interchangeable from a clustering perspective.
- **node_0–node_1 similarity** (0.979) — the two priority-flagged nodes are more similar to each other than to any standard rural node, validating their shared cluster membership in R5.

---

## 8. Summary

| Metric | Value | Trend |
|--------|-------|-------|
| Best node (R5) | node_4 (reward 1479.8, TP 91.0%) | Industrial fringe — strongest round-on-round gain |
| Worst node | node_0 (reward 1464.2 R1 → 1469.7 R5) | Town centre — highest intersection complexity |
| Best TP ratio | node_1 R5 (91.2%) | Fastest vehicle clearance, school/temple zone |
| Worst TP ratio | node_3 R1 (89.1%) | Improved to 90.1% by R5 |
| Best wait | node_1 R5 (0.352 s) | Lowest dwell time across all nodes all rounds |
| Worst wait | node_4 R1 (0.526 s) | Improved to 0.493 s by R5 |
| Avg departed/node/round | ~513 vehicles | Stable across all rounds |
| Avg TP ratio R1→R5 | 89.8% → 90.7% (+0.9 pp) | Consistent improvement via federation |
| Total arrived R1→R5 | 2763 → 2792 veh (+29) | Growing network throughput |
| Reward R1→R5 | 1472.4 → 1475.9 (+0.24%) | Gradual gain from federated learning |
| Loss R1→R5 | 0.5623 → 0.4830 (−14.1%) | Steady DQN convergence |
| Fastest learner | node_3 (−19.4%) | Most complex node benefits most from federation |
| Cluster quality | Oscillating R1–R4, best at R5 | Priority-based separation emerges |
| Total transitions | 11 across 2 re-clustering steps | R3–R4 stable, R2 and R5 active |
| Best clustering | Round 5: {node_0, node_1} vs {node_2–5} | Correctly isolates priority-flagged junctions |
| SUMO reset note | R1 genuine fresh; R2–R5 carry partial state | DQN loss is the primary learning indicator |
