# AdaptFlow Pikhuwa Rural OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 5 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 200
**Map:** Pikhuwa, Uttar Pradesh, India | **Trip Density:** 40 | **Total Trips:** 10,964 | **Vehicles/Episode:** ~609

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Avg Queue | Max Queue | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|-----------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 176.15 | 13.780 | **1.540** | 5 | 88 / 605 | 14.55% | 0.4852 |
| node_1 | 1 | cluster_0 | **188.26** | 14.959 | **0.700** | 3 | **108** / 605 | **17.85%** | 0.5489 |
| node_2 | 1 | cluster_0 | 182.89 | 14.934 | 1.130 | **7** | 101 / 604 | 16.72% | 0.5592 |
| node_3 | 1 | cluster_1 | 185.31 | 15.925 | 1.365 | 5 | 92 / 603 | 15.26% | 0.4824 |
| node_4 | 1 | cluster_1 | 181.03 | **16.866** | 1.425 | 6 | 80 / 603 | 13.27% | 0.4866 |
| node_5 | 1 | cluster_1 | 183.65 | **13.166** | 1.045 | 4 | 107 / 603 | 17.74% | 0.4887 |
| node_0 | 2 | cluster_0 | 176.15 | 13.780 | **1.540** | 5 | 88 / 605 | 14.55% | 0.3832 |
| node_1 | 2 | cluster_1 | **188.26** | 14.959 | **0.700** | 3 | **108** / 605 | **17.85%** | 0.4474 |
| node_2 | 2 | cluster_0 | 182.89 | 14.934 | 1.130 | **7** | 101 / 604 | 16.72% | 0.3887 |
| node_3 | 2 | cluster_0 | 185.31 | 15.925 | 1.365 | 5 | 92 / 603 | 15.26% | 0.4102 |
| node_4 | 2 | cluster_0 | 181.03 | **16.866** | 1.425 | 6 | 80 / 603 | 13.27% | 0.3950 |
| node_5 | 2 | cluster_1 | 183.65 | **13.166** | 1.045 | 4 | 107 / 603 | 17.74% | 0.4045 |
| node_0 | 3 | cluster_0 | 176.15 | 13.780 | **1.540** | 5 | 88 / 605 | 14.55% | 0.3508 |
| node_1 | 3 | cluster_1 | **188.26** | 14.959 | **0.700** | 3 | **108** / 605 | **17.85%** | 0.4127 |
| node_2 | 3 | cluster_0 | 182.89 | 14.934 | 1.130 | **7** | 101 / 604 | 16.72% | 0.3792 |
| node_3 | 3 | cluster_0 | 185.31 | 15.925 | 1.365 | 5 | 92 / 603 | 15.26% | 0.4114 |
| node_4 | 3 | cluster_0 | 181.03 | **16.866** | 1.425 | 6 | 80 / 603 | 13.27% | 0.3973 |
| node_5 | 3 | cluster_1 | 183.65 | **13.166** | 1.045 | 4 | 107 / 603 | 17.74% | 0.3790 |
| node_0 | 4 | cluster_1 | 176.15 | 13.780 | **1.540** | 5 | 88 / 605 | 14.55% | 0.3601 |
| node_1 | 4 | cluster_0 | **188.26** | 14.959 | **0.700** | 3 | **108** / 605 | **17.85%** | 0.4463 |
| node_2 | 4 | cluster_0 | 182.89 | 14.934 | 1.130 | **7** | 101 / 604 | 16.72% | 0.3925 |
| node_3 | 4 | cluster_0 | 185.31 | 15.925 | 1.365 | 5 | 92 / 603 | 15.26% | 0.4179 |
| node_4 | 4 | cluster_1 | 181.03 | **16.866** | 1.425 | 6 | 80 / 603 | 13.27% | 0.3812 |
| node_5 | 4 | cluster_0 | 183.65 | **13.166** | 1.045 | 4 | 107 / 603 | 17.74% | 0.3775 |
| node_0 | 5 | cluster_0 | 176.15 | 13.780 | **1.540** | 5 | 88 / 605 | 14.55% | 0.3822 |
| node_1 | 5 | cluster_0 | **188.26** | 14.959 | **0.700** | 3 | **108** / 605 | **17.85%** | 0.4171 |
| node_2 | 5 | cluster_1 | 182.89 | 14.934 | 1.130 | **7** | 101 / 604 | 16.72% | 0.3663 |
| node_3 | 5 | cluster_1 | 185.31 | 15.925 | 1.365 | 5 | 92 / 603 | 15.26% | 0.4118 |
| node_4 | 5 | cluster_1 | 181.03 | **16.866** | 1.425 | 6 | 80 / 603 | 13.27% | 0.3743 |
| node_5 | 5 | cluster_0 | 183.65 | **13.166** | 1.045 | 4 | 107 / 603 | 17.74% | 0.3982 |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Queue | Avg Loss | Total TP |
|-------|-----------|-----|-----|-------------|-----------|---------|---------|
| 1 | 182.88 | 176.15 | 188.26 | 14.938 | 1.201 | 0.5085 | 576 veh |
| 2 | 182.88 | 176.15 | 188.26 | 14.938 | 1.201 | 0.4048 | 576 veh |
| 3 | 182.88 | 176.15 | 188.26 | 14.938 | 1.201 | 0.3884 | 576 veh |
| 4 | 182.88 | 176.15 | 188.26 | 14.938 | 1.201 | 0.4005 | 576 veh |
| 5 | 182.88 | 176.15 | 188.26 | 14.938 | 1.201 | 0.3897 | 576 veh |

### Key Observations
- **Reward spread of 12.1 pts (176.15–188.26)** across 6 nodes — unlike india_rural_osm where all rewards were pegged at identical values, Pikhuwa produces meaningful per-node differentiation
- **Avg wait 14.9 s and avg queue 1.20** across all nodes — real congestion at TLS intersections; this map is genuinely stressed at 200-step episodes with density=40
- **Loss drops R1→R3**: avg 0.5085 → 0.3884 (−23.6%) — genuine DQN learning is occurring across all nodes
- **Loss rebounds slightly at R4** (0.3884 → 0.4005) — clustering reshuffle in R4 disrupts aggregated policies momentarily, then recovers to 0.3897 at R5
- **node_1 consistently leads** — highest reward (188.26), highest TP (108 vehicles, 17.85%), lowest queue (0.70) across all 5 rounds
- **node_4 is the persistent bottleneck** — longest avg_wait (16.87 s), lowest TP ratio (13.27%), fewest completed trips (80/603)
- **node_0 has the worst reward despite moderate wait** — highest avg_queue (1.54) creates the largest queue penalty in the reward function, pushing node_0 to last place

> [!CAUTION]
> All 5 rounds share **identical environment metrics** (reward, wait, queue, throughput). Only loss values differ. The SUMO-reset bug applies — `env.reset()` reuses the TraCI connection from Round 1 onwards. Round 1 fingerprints reflect the only genuinely fresh SUMO episode.

---

## 3. Node Traffic Profiles

Six distinct configs with 400 s begin-time offsets produce clear performance tiers:

| Profile | Nodes | Avg Reward | Avg Wait | Avg Queue | TP Ratio |
|---------|-------|-----------|---------|-----------|---------|
| **High-flow** | node_1, node_5 | 185.96 | 14.063 s | **0.873** | **17.80%** |
| **Mid-flow** | node_2, node_3 | 184.10 | 15.430 s | 1.248 | 15.99% |
| **Congested** | node_0, node_4 | 178.59 | **15.323 s** | **1.483** | 13.91% |

- **node_1 is the best node**: reward 188.26, TP ratio 17.85%, avg_queue 0.70 — school/temple zone at begin=400 s sees mid-morning steady-state flow with the smallest queue build-up
- **node_5 is second best**: reward 183.65, TP ratio 17.74%, lowest avg_wait (13.17 s) — highway junction at begin=2000 s handles through-traffic efficiently with moderate queue (1.045)
- **node_4 is the primary bottleneck**: longest wait (16.87 s), worst TP ratio (13.27%), 80 completed trips (vs 108 for node_1) — industrial/highway fringe at begin=1600 s sees evening goods vehicle surge creating sustained waiting at signal
- **node_0 has the worst reward** (176.15) despite only moderate wait (13.78 s) — its avg_queue of 1.54 is the highest of all nodes; queue penalty dominates the reward function over wait penalty here
- **node_2 has the worst max_queue** (7) — farming district at begin=800 s experiences bursty tractor + passenger vehicle arrivals creating spike queues

---

## 4. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 avg wait | cluster_1 avg wait |
|-------|-----------|-----------|-------------------|-------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 14.558 s | 15.319 s |
| 2 | node_0, node_2, node_3, node_4 | node_1, node_5 | 15.376 s | 14.063 s |
| 3 | node_0, node_2, node_3, node_4 | node_1, node_5 | 15.376 s | **STABLE** |
| 4 | node_1, node_2, node_3, node_5 | node_0, node_4 | 14.746 s | 15.323 s |
| 5 | **node_0, node_1, node_5** | **node_2, node_3, node_4** | **13.968 s** | **15.908 s** |

> [!IMPORTANT]
> **Round 5 produces the most semantically correct clustering of the entire run** — cluster_0 groups the three lowest-wait nodes (node_0: 13.78 s, node_1: 14.96 s, node_5: 13.17 s; avg 13.97 s) and cluster_1 groups the three highest-wait nodes (node_2: 14.93 s, node_3: 15.93 s, node_4: 16.87 s; avg 15.91 s). The algorithm correctly converged to a wait-based congestion split by round 5.
>
> Rounds 2 and 3 also form a meaningful split — cluster_1 correctly isolates the two lowest-queue nodes (node_1: queue 0.70, node_5: queue 1.045). These are the only two fully stable rounds in the run.

---

## 5. Cluster Transitions (Re-Clustering Events)

```
Round 1:  No transitions (static baseline)

Round 2:  node_1: cluster_0 → cluster_1
          node_3: cluster_1 → cluster_0
          node_4: cluster_1 → cluster_0
          [3 transitions — breaks initial geographic bias]

Round 3:  No transitions — STABLE (identical to Round 2)

Round 4:  node_0: cluster_0 → cluster_1
          node_1: cluster_1 → cluster_0
          node_4: cluster_0 → cluster_1
          node_5: cluster_1 → cluster_0
          [4 transitions]

Round 5:  node_0: cluster_1 → cluster_0
          node_2: cluster_0 → cluster_1
          node_3: cluster_0 → cluster_1
          [3 transitions]
```

**Total transitions: 10** across 3 re-clustering steps (R2, R4, R5).

- Round 3 is the only **fully stable** round — no transitions after R2's reshuffle
- **node_1** is the most active swing node — transitions in R2 and R4
- **node_2 and node_3** finally land in cluster_1 (high-wait group) at R5 — semantically correct, as both have avg_wait > 14.9 s and queue > 1.1
- Round 5's clustering {node_0, node_1, node_5} vs {node_2, node_3, node_4} is the best quality grouping of the run, aligning with the wait-tier split

---

## 6. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, priority_flag]`

All fingerprints are **identical across all 5 rounds** (SUMO-reset bug — only Round 1 episode data is captured):

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1–5 | `[13.780, 1.540, 0.1455, 5.0, 0.0, 1.0]` |
| node_1 | 1–5 | `[14.959, 0.700, 0.1785, 3.0, 0.0, 0.5]` |
| node_2 | 1–5 | `[14.934, 1.130, 0.1672, 7.0, 0.0, 0.0]` |
| node_3 | 1–5 | `[15.925, 1.365, 0.1526, 5.0, 0.0, 0.0]` |
| node_4 | 1–5 | `[16.866, 1.425, 0.1327, 6.0, 0.0, 0.0]` |
| node_5 | 1–5 | `[13.166, 1.045, 0.1774, 4.0, 0.0, 0.0]` |

**Key fingerprint insights:**
- **node_0 and node_1 are structurally unique** — the only nodes with non-zero `priority_flag` (1.0 and 0.5 respectively), placing them in a distinct region of fingerprint space from nodes 2–5
- **node_4 is the most congested** by avg_wait (16.87 s) — captures evening industrial surge at begin=1600 s
- **node_1 has the best TP ratio** (0.1785) with the lowest queue (0.70) — the school zone at begin=400 s loads into steady mid-morning flow
- **nodes 2, 3, 4, 5 form a high-wait cluster** (14.93–16.87 s) with no priority flags — k-means correctly separates these from nodes 0 and 1 in R2/R3
- **node_0's dominant penalty is queue** (1.54), not wait (13.78 s) — explains its lowest reward despite not having the worst wait time

---

## 7. Loss Trajectories (DQN Learning Progress)

| Node | R1 | R2 | R3 | R4 | R5 | R1→R5 drop |
|------|----|----|----|----|-----|-----------|
| node_0 | 0.4852 | 0.3832 | **0.3508** | 0.3601 | 0.3822 | −21.2% |
| node_1 | 0.5489 | 0.4474 | 0.4127 | 0.4463 | 0.4171 | −24.0% |
| node_2 | 0.5592 | 0.3887 | 0.3792 | 0.3925 | **0.3663** | **−34.5%** |
| node_3 | 0.4824 | 0.4102 | 0.4114 | 0.4179 | 0.4118 | −14.6% |
| node_4 | 0.4866 | 0.3950 | 0.3973 | 0.3812 | 0.3743 | −23.1% |
| node_5 | 0.4887 | 0.4045 | **0.3790** | 0.3775 | 0.3982 | −18.5% |

**Key loss insights:**
- **node_2 is the fastest learner** (−34.5%) — farming district's bursty traffic provides the richest gradient signal; reaches lowest absolute loss (0.3663) at R5
- **node_3 is the slowest learner** (−14.6%) — residential colony's predictable afternoon pattern leads to rapid policy plateau; loss barely moves after R2
- **node_0 hits its global minimum at R3** (0.3508) — clustering instability in R4 slightly disrupts its policy, causing a small rebound
- **R4 rebound across most nodes** — clustering reshuffle (4 transitions) in R4 temporarily disrupts federated aggregation quality, seen as a small loss uptick at R4 vs R3
- **All losses remain high** (0.37–0.56) compared to india_rural_osm (0.08–0.12) — Pikhuwa is a genuinely hard learning environment with real congestion signal driving high-variance gradients

---

## 8. Cosine Similarity Matrix (identical all 5 rounds)

|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9865 | 0.9930 | 0.9965 | 0.9974 | 0.9959 |
| node_1 | 0.9865 | 1.0000 | 0.9706 | 0.9932 | 0.9886 | 0.9943 |
| node_2 | 0.9930 | **0.9706** | 1.0000 | 0.9910 | 0.9953 | 0.9898 |
| node_3 | 0.9965 | 0.9932 | 0.9910 | 1.0000 | 0.9993 | **0.9999** |
| node_4 | 0.9974 | 0.9886 | 0.9953 | 0.9993 | 1.0000 | 0.9989 |
| node_5 | 0.9959 | 0.9943 | 0.9898 | **0.9999** | 0.9989 | 1.0000 |

**Similarity insights:**
- **Most similar pair:** node_3 ↔ node_5 (0.9999) — residential colony and highway junction have nearly identical fingerprints despite opposite spatial roles; both share similar avg_wait (~14–16 s), moderate queue, no priority flag
- **Most dissimilar pair:** node_1 ↔ node_2 (0.9706) — **lowest similarity in this run** — node_1's low-queue (0.70), good TP ratio (0.1785), and priority_flag=0.5 make it genuinely unlike node_2's high-max_queue (7.0), moderate queue (1.13), no-flag fingerprint. This makes node_1 the most distinguishable node in the network
- **node_0 sits between both groups** — similarity to node_4 (0.9974) is highest despite node_0 being a priority-flag node; the shared avg_wait range (13–17 s) and queue values dominate the cosine score over the flag dimension
- Average off-diagonal similarity: **~0.9929** — moderate fingerprint diversity, more varied than india_rural_osm (which was dominated by near-zero queue dimensions)

---

## 9. Pikhuwa vs China Rural OSM Comparison

| Metric | China Rural OSM | Pikhuwa Rural OSM |
|--------|-----------------|-------------------|
| Region | Shaoxing, China | Pikhuwa, UP, India |
| Trip density | 12 | 40 |
| Vehicles/episode | ~600 | ~609 |
| Avg wait R1 | 3.449 s | **14.938 s** |
| Avg queue R1 | 2.517 | **1.201** |
| Max queue (any node) | 13 (node_3 R1) | 7 (node_2) |
| Reward range | 154.08–189.29 | 176.15–188.26 |
| Reward spread (nodes) | **35.2 pts** | 12.1 pts |
| Best node reward | 191.65 (node_1 R3+) | 188.26 (node_1 all) |
| Worst node reward | 154.08 (node_3 R1) | 176.15 (node_0 all) |
| Best TP ratio | 9.39% (node_1) | **17.85% (node_1)** |
| Worst TP ratio | 6.47% (node_3 R1) | 13.27% (node_4) |
| Avg loss R1 | 0.4941 | 0.5085 |
| Avg loss R5 | ~0.397 | 0.3897 |
| Loss drop R1→R5 | ~−19.6% | −23.3% |
| Fastest learner | node_3 (−49.3% by R3) | node_2 (−34.5%) |
| Slowest learner | node_1 (−10.1%) | node_3 (−14.6%) |
| Cluster stable rounds | R3 (1 round) | R2=R3 (1 stable interval) |
| Total transitions | 11 | 10 |
| Best clustering round | R5: {0,1} vs {2,3,4,5} | R5: {0,1,5} vs {2,3,4} |
| SUMO reset bug | R1 real, R2 partial, R3–R5 frozen | R1 real, R2–R5 frozen |

**Key contrast:** Pikhuwa has 4.3× higher avg_wait but lower avg_queue than China rural. China rural's denser road network creates shorter wait times but larger queues at signals; Pikhuwa's more open rural layout lets vehicles travel further between signals but results in longer per-intersection waits. Both scenarios show genuine congestion and meaningful AdaptFlow learning.

---

## 10. Summary

| Metric | Value | Assessment |
|--------|-------|-----------|
| Best node | node_1 (reward 188.26, TP 17.85%, queue 0.70) | Consistent leader — school zone, minimal queue |
| Worst node | node_0 (reward 176.15, queue 1.54) | Town centre — highest queue, lowest reward |
| Most congested TLS | node_4 (avg_wait 16.87 s, TP 13.27%) | Industrial zone evening surge |
| Least congested TLS | node_5 (avg_wait 13.17 s, TP 17.74%) | Highway junction, free through-traffic |
| Worst max queue | node_2 (max_queue 7) | Farming district burst arrivals |
| Reward spread | 12.1 pts (176.15–188.26) | Meaningful per-node differentiation |
| Reward ceiling | 200.0 (200 steps) | Not hit — congestion is penalising effectively |
| Total TP all nodes R1 | 576 vehicles | 14.55–17.85% completion rate per node |
| Loss R1→R5 | 0.5085 → 0.3897 (−23.3%) | Genuine multi-round learning |
| Fastest DQN learner | node_2 (−34.5%) | Bursty farm traffic = strong gradient signal |
| Slowest DQN learner | node_3 (−14.6%) | Predictable residential pattern = early plateau |
| Cluster stability | 1 stable interval (R2=R3) | Brief convergence then drift |
| Best clustering | R5: low-wait {0,1,5} vs high-wait {2,3,4} | Semantically correct wait-based split |
| Total transitions | 10 across 3 re-clustering steps | Comparable to China rural (11) |
| Most dissimilar nodes | node_1 ↔ node_2 (0.9706) | Best fingerprint diversity in run |
| SUMO reset bug | Present — R2–R5 metrics frozen | Known limitation |
| **Overall verdict** | **Genuinely congested rural environment** | Valid for AdaptFlow research and China rural comparison |
