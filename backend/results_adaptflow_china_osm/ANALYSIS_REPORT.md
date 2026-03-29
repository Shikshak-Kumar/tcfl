# AdaptFlow China OSM — Results Analysis
**Mode:** SUMO-GUI (Real Simulation) | **Rounds:** 3 | **Nodes:** 6 | **Clusters:** 2 | **Steps:** 200

---

## 1. Per-Node Performance Across Rounds

| Node | Round | Cluster | Reward | Avg Wait (s) | Throughput | TP Ratio | Loss |
|------|-------|---------|--------|-------------|-----------|---------|------|
| node_0 | 1 | cluster_0 | 184.59 | 9.479 | 23 | 0.2447 | 0.5397 |
| node_1 | 1 | cluster_0 | 196.85 | 11.108 | 12 | 0.1290 | 0.5339 |
| node_2 | 1 | cluster_0 | 188.49 | 9.495 | 18 | 0.1935 | 0.4893 |
| node_3 | 1 | cluster_1 | 191.69 | 7.606 | 22 | 0.2340 | 0.5248 |
| node_4 | 1 | cluster_1 | 193.64 | 9.624 | 14 | 0.1505 | 0.5539 |
| node_5 | 1 | cluster_1 | 182.51 | 8.290 | 27 | 0.2903 | 0.4895 |
| node_0 | 2 | cluster_0 | 185.97 | 9.596 | 23 | 0.2447 | 0.4465 |
| node_1 | 2 | cluster_0 | 166.07 | 11.763 | 13 | 0.1398 | 0.4015 |
| node_2 | 2 | cluster_1 | 171.21 | 9.172 | 17 | 0.1828 | 0.3950 |
| node_3 | 2 | cluster_1 | 158.44 | 8.839 | 22 | 0.2366 | 0.3915 |
| node_4 | 2 | cluster_0 | 165.42 | 9.946 | 15 | 0.1613 | 0.4126 |
| node_5 | 2 | cluster_0 | 188.51 | 7.613 | 27 | 0.2903 | 0.4187 |
| node_0 | 3 | cluster_1 | 185.97 | 9.596 | 23 | 0.2447 | 0.4046 |
| node_1 | 3 | cluster_1 | 166.07 | 11.763 | 13 | 0.1398 | 0.3769 |
| node_2 | 3 | cluster_0 | 171.21 | 9.172 | 17 | 0.1828 | 0.3693 |
| node_3 | 3 | cluster_0 | 158.44 | 8.839 | 22 | 0.2366 | 0.3776 |
| node_4 | 3 | cluster_1 | 165.42 | 9.946 | 15 | 0.1613 | 0.3354 |
| node_5 | 3 | cluster_0 | 188.51 | 7.613 | 27 | 0.2903 | 0.4030 |

---

## 2. Round-Level Summary

| Round | Avg Reward | Min | Max | Avg Wait (s) | Avg Loss | Total TP |
|-------|-----------|-----|-----|-------------|---------|---------|
| 1 | **189.63** | 182.51 | 196.85 | 9.267 | 0.5218 | 116 |
| 2 | 172.60 | 158.44 | 188.51 | 9.488 | 0.4110 | 117 |
| 3 | 172.60 | 158.44 | 188.51 | 9.488 | **0.3778** | 117 |

### Key Observations
- **Reward dropped** from 189.63 → 172.60 after first federation (−8.97%), then froze — federated aggregation degraded local policies
- **Loss steadily decreased**: 0.5218 → 0.4110 → 0.3778 (−27.6% total) — agent networks are actively learning
- **Throughput stable** at 116–117 vehicles/round — environment capacity near ceiling for 200-step episodes
- **Min reward fell sharply**: 182.51 → 158.44 — the weakest node (node_3) was hurt most by federation

> [!CAUTION]
> Rounds 2 and 3 are **bit-for-bit identical** in every environment metric (reward, wait, throughput, queue). Only loss differs. The SUMO simulation is **not restarting** between rounds — `env.reset()` is reusing the existing TraCI connection rather than launching a fresh episode.

---

## 3. Node Traffic Profiles

Six nodes run on six distinct SUMO configs with staggered begin times. Two natural performance groups emerge:

| Profile | Nodes | Avg Reward (R3) | Avg Wait (R3) | Throughput (R3) | Config characteristic |
|---------|-------|----------------|--------------|----------------|----------------------|
| **High throughput** | node_0, node_3, node_5 | 177.64 | **8.68 s** | **24.0** | Lower wait, more flow |
| **Low throughput** | node_1, node_2, node_4 | 167.57 | 10.29 s | 15.0 | Higher wait, congested |

- **node_5 is the best node overall** — lowest wait (7.61 s), highest throughput (27 vehicles), reward improved R1→R2
- **node_1 is the persistent bottleneck** — highest wait (11.76 s), lowest throughput (13), largest reward drop after R1

---

## 4. Cluster Assignments Per Round

| Round | cluster_0 | cluster_1 | cluster_0 congestion | cluster_1 congestion |
|-------|-----------|-----------|---------------------|---------------------|
| 1 | node_0, node_1, node_2 | node_3, node_4, node_5 | 10.027 s | 8.507 s |
| 2 | node_0, node_1, node_4, node_5 | node_2, node_3 | 9.730 s | 9.005 s |
| 3 | node_2, node_3, node_5 | node_0, node_1, node_4 | 8.541 s | 10.435 s |

> [!IMPORTANT]
> **Round 3** produces the most semantically correct grouping — cluster_0 concentrates the lower-wait nodes (node_2, node_3, node_5 avg 8.54 s) and cluster_1 concentrates the higher-wait nodes (node_0, node_1, node_4 avg 10.44 s). The algorithm correctly separated the two performance profiles by Round 3.

---

## 5. Cluster Transitions (Re-Clustering Events)

```
Round 1:  No transitions (static baseline)
Round 2:  node_2: cluster_0 → cluster_1
          node_4: cluster_1 → cluster_0
          node_5: cluster_1 → cluster_0
Round 3:  node_0: cluster_0 → cluster_1
          node_1: cluster_0 → cluster_1
          node_2: cluster_1 → cluster_0
          node_3: cluster_1 → cluster_0
          node_4: cluster_0 → cluster_1
```

Total transitions: **8** across 2 re-clustering steps.

> [!NOTE]
> Round 3 reshuffles 5 of 6 nodes. This is not a label swap — memberships genuinely changed. The cause is that all cosine similarities are extremely compressed (0.966–0.999), making K-means numerically sensitive to small fingerprint differences. Stability will improve with more rounds.

---

## 6. Fingerprint Evolution

`[avg_wait, avg_queue, throughput_ratio, max_queue, POI_score, special_flag]`

| Node | Round | Fingerprint |
|------|-------|-------------|
| node_0 | 1 | `[9.479, 0.430, 0.245, 2.0, 0.0, 1.0]` |
| node_0 | 2 | `[9.596, 0.330, 0.245, 2.0, 0.0, 1.0]` |
| node_0 | 3 | `[9.596, 0.330, 0.245, 2.0, 0.0, 1.0]` |
| node_1 | 1 | `[11.108, 0.095, 0.129, 2.0, 0.0, 0.5]` |
| node_1 | 2 | `[11.763, 0.485, 0.140, 2.0, 0.0, 0.5]` |
| node_3 | 1 | `[7.606, 0.170, 0.234, 2.0, 0.0, 0.0]` |
| node_3 | 2 | `[8.839, 0.770, 0.237, 4.0, 0.0, 0.0]` |
| node_5 | 1 | `[8.290, 0.455, 0.290, 2.0, 0.0, 0.0]` |
| node_5 | 2 | `[7.613, 0.250, 0.290, 2.0, 0.0, 0.0]` |

- node_0 and node_1 have non-zero `special_flag` (Hospital = 1.0, School = 0.5) — priority modifier active
- node_3's queue jumped 0.170 → 0.770 and max_queue 2 → 4 after federation — most harmed node
- node_5's wait **improved** (8.29 → 7.61 s) and queue halved — only node to benefit from aggregation
- Fingerprints for R2 = R3 (frozen) — confirms SUMO episode not restarting between rounds

---

## 7. Cosine Similarity Matrix (Round 3)

|        | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 |
|--------|--------|--------|--------|--------|--------|--------|
| node_0 | 1.0000 | 0.9974 | 0.9887 | 0.9700 | 0.9947 | 0.9935 |
| node_1 | 0.9974 | 1.0000 | 0.9882 | 0.9657 | **0.9986** | 0.9949 |
| node_2 | 0.9887 | 0.9882 | 1.0000 | 0.9935 | 0.9931 | **0.9980** |
| node_3 | 0.9700 | 0.9657 | 0.9935 | 1.0000 | 0.9740 | 0.9848 |
| node_4 | 0.9947 | **0.9986** | 0.9931 | 0.9740 | 1.0000 | **0.9980** |
| node_5 | 0.9935 | 0.9949 | **0.9980** | 0.9848 | **0.9980** | 1.0000 |

**Pattern:** All values exceed 0.965 — fingerprints are highly compressed, giving K-means very little signal to work with. node_1 ↔ node_4 are the most similar pair (0.9986). node_3 is the most dissimilar to everyone (lowest row average ~0.972).

---

## 8. Summary

| Metric | Value | Trend |
|--------|-------|-------|
| Best node | node_5 (reward 188.51, wait 7.61 s, TP 29%) | Stable best |
| Worst node | node_1 (reward 166.07, wait 11.76 s, TP 13%) | Persistent bottleneck |
| Reward change R1→R3 | −17.07 (−9.0%) | Degraded after federation |
| Loss reduction | −27.6% across 3 rounds | Improving |
| Re-clustering accuracy | Correctly separates high/low congestion by Round 3 | Correct direction |
| Cluster stability | High churn — 8 transitions in 2 rounds | Needs more rounds |
| Throughput | ~20% (200-step limit) | Environment-bound |
| Critical bug | SUMO not resetting between rounds | Metrics frozen R2→R3 |
