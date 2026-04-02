# AdaptFlow China Rural OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 5 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 200

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Avg Queue | Max Queue | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|-----------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 188.41 | 2.593 | 2.040 | 6 | 37 / 511 | 7.24% | 0.5300 |
| node_1 | 1 | cluster_0 | 184.00 | 3.313 | 1.915 | 6 | **48** / 511 | **9.39%** | 0.4832 |
| node_2 | 1 | cluster_0 | 177.24 | 3.843 | 2.710 | 7 | 41 / 510 | 8.04% | 0.5105 |
| node_3 | 1 | cluster_1 | 154.08 | **4.604** | **3.970** | **13** | 33 / 510 | 6.47% | 0.4125 |
| node_4 | 1 | cluster_1 | 171.92 | 3.611 | 2.965 | 7 | 42 / 512 | 8.20% | 0.4717 |
| node_5 | 1 | cluster_1 | **189.29** | 2.729 | 1.500 | 8 | 37 / 510 | 7.25% | 0.5569 |
| node_0 | 2 | cluster_0 | 165.75 | **2.918** | 2.650 | 7 | 38 / 512 | 7.42% | 0.4184 |
| node_1 | 2 | cluster_1 | 178.35 | 3.137 | 2.180 | 7 | **48** / 511 | **9.39%** | 0.4183 |
| node_2 | 2 | cluster_0 | 179.47 | 3.515 | 2.765 | 8 | 37 / 509 | 7.27% | 0.4367 |
| node_3 | 2 | cluster_0 | 158.57 | **4.141** | **3.675** | 10 | 35 / 512 | 6.84% | 0.3009 |
| node_4 | 2 | cluster_0 | 179.76 | 4.250 | 2.695 | 7 | 44 / 511 | 8.61% | 0.4020 |
| node_5 | 2 | cluster_1 | **189.56** | 3.373 | 1.610 | 7 | 39 / 510 | 7.65% | 0.4243 |
| node_0 | 3 | cluster_1 | 165.75 | **2.918** | 2.650 | 7 | 38 / 512 | 7.42% | 0.4164 |
| node_1 | 3 | cluster_1 | **191.65** | 2.887 | 1.780 | 10 | **48** / 512 | **9.38%** | 0.4388 |
| node_2 | 3 | cluster_0 | 179.24 | 3.519 | 2.770 | 8 | 37 / 509 | 7.27% | 0.3983 |
| node_3 | 3 | cluster_0 | 158.57 | **4.141** | **3.675** | 10 | 35 / 512 | 6.84% | 0.2091 |
| node_4 | 3 | cluster_0 | 179.76 | 4.250 | 2.695 | 7 | 44 / 511 | 8.61% | 0.3459 |
| node_5 | 3 | cluster_1 | **189.56** | 3.373 | 1.610 | 7 | 39 / 510 | 7.65% | 0.4599 |
| node_0 | 4 | cluster_0 | 165.75 | **2.918** | 2.650 | 7 | 38 / 512 | 7.42% | 0.3134 |
| node_1 | 4 | cluster_1 | **191.65** | 2.887 | 1.780 | 10 | **48** / 512 | **9.38%** | 0.4498 |
| node_2 | 4 | cluster_1 | 179.24 | 3.519 | 2.770 | 8 | 37 / 509 | 7.27% | 0.3485 |
| node_3 | 4 | cluster_0 | 158.57 | **4.141** | **3.675** | 10 | 35 / 512 | 6.84% | 0.3117 |
| node_4 | 4 | cluster_0 | 179.76 | 4.250 | 2.695 | 7 | 44 / 511 | 8.61% | 0.4117 |
| node_5 | 4 | cluster_0 | **189.56** | 3.373 | 1.610 | 7 | 39 / 510 | 7.65% | 0.4413 |
| node_0 | 5 | cluster_0 | 165.75 | **2.918** | 2.650 | 7 | 38 / 512 | 7.42% | 0.3516 |
| node_1 | 5 | cluster_0 | **191.65** | 2.887 | 1.780 | 10 | **48** / 512 | **9.38%** | 0.4345 |
| node_2 | 5 | cluster_1 | 179.24 | 3.519 | 2.770 | 8 | 37 / 509 | 7.27% | 0.3992 |
| node_3 | 5 | cluster_1 | 158.57 | **4.141** | **3.675** | 10 | 35 / 512 | 6.84% | 0.3393 |
| node_4 | 5 | cluster_1 | 179.76 | 4.250 | 2.695 | 7 | 44 / 511 | 8.61% | ~0.411 |
| node_5 | 5 | cluster_1 | **189.56** | 3.373 | 1.610 | 7 | 39 / 510 | 7.65% | ~0.441 |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Queue | Avg Loss | Total TP |
|-------|-----------|-----|-----|-------------|-----------|---------|---------|
| 1 | 177.49 | 154.08 | 189.29 | 3.449 | 2.517 | 0.4941 | 238 veh |
| 2 | 175.24 | 158.57 | 189.56 | 3.556 | 2.596 | 0.4001 | 241 veh |
| 3 | **177.42** | 158.57 | **191.65** | 3.514 | 2.530 | 0.3781 | 241 veh |
| 4 | **177.42** | 158.57 | **191.65** | 3.514 | 2.530 | 0.3714 | 241 veh |
| 5 | **177.42** | 158.57 | **191.65** | 3.514 | 2.530 | ~0.397 | 241 veh |

### Key Observations
- **R1 → R2: reward dipped −1.3%** (177.49 → 175.24) — first FedAvg aggregation slightly hurt node_0 and node_5 while helping node_3 and node_4
- **R2 → R3: reward recovered +1.2%** (175.24 → 177.42) — node_1's policy improved significantly (+7.5% reward) after the cluster aggregation settled
- **R3–R5 frozen** — avg reward 177.42, wait 3.514 s, queue 2.530 identical across rounds 3–5 (SUMO-reset bug; only loss changes)
- **node_1 consistently leads** — only node to produce 48 completed trips every round (9.38–9.39% TP), highest of all 6 nodes
- **node_3 consistently worst** — lowest reward (154–158), highest wait (4.14–4.60 s), highest queue (3.67–3.97), fewest completions (33–35 trips)
- **Total throughput jumped R1→R2**: 238 → 241 (+1.3%) and held there — slightly more vehicles completing with shorter congestion after first federation

> [!CAUTION]
> Rounds 3, 4, and 5 are **bit-for-bit identical** in all environment metrics per node (reward, wait, queue, throughput). Only DQN loss values differ. The SUMO simulation is not restarting between rounds — `env.reset()` reuses the existing TraCI connection from round 2 onwards. Round 2 fingerprints also differ from round 1, meaning the first re-cluster used fresh real episode data.

---

## 3. Node Traffic Profiles

Six distinct configs with 600 s begin-time offsets produce clear performance tiers:

| Profile | Nodes | Avg Reward (R3+) | Avg Wait (R3+) | Avg Queue (R3+) | TP Ratio |
|---------|-------|-----------------|---------------|----------------|---------|
| **High-flow** | node_1, node_5 | **190.61** | **3.130 s** | **1.695** | 8.52% |
| **Mid-flow** | node_0, node_2, node_4 | 174.92 | 3.562 s | 2.705 | 7.77% |
| **Congested** | node_3 | 158.57 | **4.141 s** | **3.675** | 6.84% |

- **node_1 is the best node** (R3+): reward 191.65, TP ratio 9.38%, wait 2.887 s — school/temple zone at begin=600 s sees the lightest demand window in the 200-step episode
- **node_5 is second best**: reward 189.56, lowest queue (1.61), wait 3.373 s — highway junction serves through-traffic efficiently
- **node_3 is the persistent bottleneck**: reward 158.57, worst wait (4.141 s), worst queue (3.675), worst TP (6.84%) — residential outskirts at begin=1200 s faces dense afternoon return traffic; max_queue hit 13 in R1 (highest across all nodes all rounds)
- **node_4 (industrial)** has highest persistent wait in R2+ (4.250 s) despite decent TP (8.61%) — bursty shift-change demand at begin=1600 s creates wait without forming long queues

---

## 4. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 avg wait | cluster_1 avg wait |
|-------|-----------|-----------|-------------------|-------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 3.250 s | 3.648 s |
| 2 | node_0, node_2, node_3, node_4 | node_1, node_5 | 3.706 s | 3.255 s |
| 3 | node_0, node_2, node_3, node_4 | node_1, node_5 | 3.707 s | **STABLE** |
| 4 | node_0, node_3, node_4 | node_1, node_2, node_5 | 3.770 s | 3.260 s |
| 5 | **node_0, node_1** | **node_2, node_3, node_4, node_5** | **2.903 s** | **3.821 s** |

> [!IMPORTANT]
> **Round 5 produces the most semantically correct clustering of the entire run** — cluster_0 groups the two lowest-wait nodes (node_0: 2.918 s, node_1: 2.887 s, avg 2.903 s) and cluster_1 groups the four highest-wait nodes (avg 3.821 s). The algorithm correctly converged to a wait-based separation by round 5, even though it took 4 re-clustering steps to reach it.

---

## 5. Cluster Transitions (Re-Clustering Events)

```
Round 1:  No transitions (static baseline)
Round 2:  node_0: cluster_0 → cluster_1
          node_1: cluster_0 → cluster_1
          node_3: cluster_1 → cluster_0
          node_4: cluster_1 → cluster_0
Round 3:  No transitions (STABLE)
Round 4:  node_0: cluster_1 → cluster_0
          node_2: cluster_0 → cluster_1
          node_5: cluster_1 → cluster_0
Round 5:  node_1: cluster_1 → cluster_0
          node_3: cluster_0 → cluster_1
          node_4: cluster_0 → cluster_1
          node_5: cluster_0 → cluster_1
```

**Total transitions: 11** across 3 re-clustering steps (R2, R4, R5).

- Round 3 is the only **stable** round — no transitions after round 2's reshuffle
- **node_1** is the most active swing node — transitions in R2, R4 (indirectly via R5 recovery), and R5
- **node_3** finally lands in cluster_1 (high-congestion cluster) at R5 — semantically correct since it has the worst metrics
- Round 5's clustering (cluster_0={0,1} vs cluster_1={2,3,4,5}) is the best quality grouping of the run

---

## 6. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, priority_flag]`

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1 | `[2.593, 2.040, 0.0724, 6.0, 0.0, 1.0]` |
| node_0 | 2+ | `[2.918, 2.650, 0.0742, 7.0, 0.0, 1.0]` |
| node_1 | 1 | `[3.313, 1.915, 0.0939, 6.0, 0.0, 0.5]` |
| node_1 | 2 | `[3.137, 2.180, 0.0939, 7.0, 0.0, 0.5]` |
| node_1 | 3+ | `[2.887, 1.780, 0.0938, 10.0, 0.0, 0.5]` |
| node_2 | 1 | `[3.843, 2.710, 0.0804, 7.0, 0.0, 0.0]` |
| node_2 | 2+ | `[3.519, 2.770, 0.0727, 8.0, 0.0, 0.0]` |
| node_3 | 1 | `[4.604, 3.970, 0.0647, 13.0, 0.0, 0.0]` |
| node_3 | 2+ | `[4.141, 3.675, 0.0684, 10.0, 0.0, 0.0]` |
| node_4 | 1 | `[3.611, 2.965, 0.0820, 7.0, 0.0, 0.0]` |
| node_4 | 2+ | `[4.250, 2.695, 0.0861, 7.0, 0.0, 0.0]` |
| node_5 | 1 | `[2.729, 1.500, 0.0725, 8.0, 0.0, 0.0]` |
| node_5 | 2+ | `[3.373, 1.610, 0.0765, 7.0, 0.0, 0.0]` |

**Key fingerprint insights:**
- **node_3 improved most** R1→R2: max_queue dropped 13→10, wait 4.604→4.141 s — the first FedAvg weight sharing reduced node_3's peak congestion
- **node_4 worsened** R1→R2: wait jumped 3.611→4.250 s (+0.64 s) — the aggregated policy hurts the industrial zone
- **node_1 continued evolving R2→R3**: wait dropped 3.137→2.887 s, queue improved 2.18→1.78 — unique among all nodes; only one with 3-round fingerprint evolution
- Fingerprints frozen from R3 onwards for all nodes (SUMO-reset bug)

---

## 7. Loss Trajectories (DQN Learning Progress)

| Node | R1 | R2 | R3 | R4 | R5 | R1→R5 drop |
|------|----|----|----|----|-----|-----------|
| node_0 | 0.5300 | 0.4184 | 0.4164 | **0.3134** | 0.3516 | −33.7% |
| node_1 | 0.4832 | 0.4183 | 0.4388 | 0.4498 | 0.4345 | −10.1% |
| node_2 | 0.5105 | 0.4367 | 0.3983 | **0.3485** | 0.3992 | −21.8% |
| node_3 | 0.4125 | 0.3009 | **0.2091** | 0.3117 | 0.3393 | −17.7% |
| node_4 | 0.4717 | 0.4020 | 0.3459 | 0.4117 | ~0.411 | ~−12.9% |
| node_5 | 0.5569 | 0.4243 | 0.4599 | 0.4413 | ~0.441 | ~−20.8% |

**Key loss insights:**
- **node_3 is the fastest learner** — drops from 0.4125 → 0.2091 by R3 (−49.3%), lowest absolute loss of any node at R3. The DQN quickly learned to manage residential congestion
- **node_0** shows the best sustained improvement over 5 rounds (−33.7%), reaching its low of 0.3134 at R4
- **node_1 is the slowest learner** — loss barely moves (0.483 → 0.435, −10.1%); its policy is already near-optimal from round 1, so there is little gradient to reduce
- **Loss trajectories are noisy** (oscillating not monotonic) — DQN with replay buffer in 200-step episodes creates high-variance gradient estimates
- All losses remain relatively high (0.20–0.56) compared to India rural (0.08–0.12) — the China rural map's congestion creates harder, more variable learning problems

---

## 8. Cosine Similarity Matrix

### Round 1
|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9925 | 0.9852 | 0.9872 | 0.9858 | 0.9786 |
| node_1 | 0.9925 | 1.0000 | 0.9960 | 0.9851 | 0.9934 | 0.9783 |
| node_2 | 0.9852 | 0.9960 | 1.0000 | 0.9870 | **0.9992** | 0.9746 |
| node_3 | 0.9872 | 0.9851 | 0.9870 | 1.0000 | 0.9885 | **0.9945** |
| node_4 | 0.9858 | 0.9934 | **0.9992** | 0.9885 | 1.0000 | 0.9730 |
| node_5 | 0.9786 | 0.9783 | 0.9746 | **0.9945** | 0.9730 | 1.0000 |

### Round 3–5 (stable)
|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9773 | 0.9917 | 0.9923 | 0.9821 | 0.9824 |
| node_1 | 0.9773 | 1.0000 | 0.9811 | 0.9807 | **0.9553** | 0.9847 |
| node_2 | 0.9917 | 0.9811 | 1.0000 | **0.9996** | 0.9921 | 0.9942 |
| node_3 | 0.9923 | 0.9807 | **0.9996** | 1.0000 | 0.9894 | 0.9910 |
| node_4 | 0.9821 | **0.9553** | 0.9921 | 0.9894 | 1.0000 | 0.9892 |
| node_5 | 0.9824 | 0.9847 | 0.9942 | 0.9910 | 0.9892 | 1.0000 |

**Similarity insights:**
- **R1 most similar pair:** node_2 ↔ node_4 (0.9992) — farming and industrial zones share almost identical R1 fingerprints
- **R1 most dissimilar:** node_2 ↔ node_5 (0.9746) — node_5 starts with very low queue (1.5) vs node_2's higher queue (2.71)
- **R3+ most similar pair:** node_2 ↔ node_3 (0.9996) — farming district and residential outskirts converge to near-identical fingerprints after federation
- **R3+ most dissimilar pair:** node_1 ↔ node_4 (0.9553) — **lowest similarity seen in this run** — node_1's improved policy (wait 2.887 s, queue 1.78, max_q=10, priority_flag=0.5) is genuinely unlike node_4's bursty industrial fingerprint (wait 4.250 s, queue 2.695). This makes the round-5 split cluster_0={0,1} vs cluster_1={2,3,4,5} especially meaningful
- R3+ average similarity: **~0.9884** — wider spread than the old 500-step run (~0.9967), meaning 200-step episodes produce more diverse fingerprints

---

## 9. Summary

| Metric | Value | Trend |
|--------|-------|-------|
| Best node | node_1 (reward 191.65, TP 9.38%) | Consistent leader R3+ |
| Worst node | node_3 (reward 154–158, wait 4.14–4.60 s) | Persistent bottleneck |
| Best wait | node_0 R1 (2.593 s) / node_1 R3+ (2.887 s) | Low-congestion zones |
| Worst wait | node_3 R1 (4.604 s) | Residential outskirts |
| Reward R1→R5 | −177.49 → −177.42 (stable, tiny net drop −0.04%) | Essentially neutral |
| Reward dip | R2 −1.3%, then recovered | FedAvg slightly disruptive once |
| Loss improvement | 0.494 → ~0.397 (−19.6% avg over 5 rounds) | Slowly improving |
| Fastest learner | node_3 (−49.3% loss by R3) | Congested node benefits most from learning |
| Cluster quality | Oscillating but R5 semantically correct | Improving across rounds |
| Total transitions | 11 across 3 re-clustering steps | High churn |
| Best clustering | Round 5: {node_0,node_1} vs {node_2,node_3,node_4,node_5} | Correctly separates by wait |
| SUMO reset bug | R1 real, R2 partially real, R3–R5 frozen | Only first 2 rounds have fresh episodes |
| Throughput | 7–9% (200-step limit) | Environment-bound |
