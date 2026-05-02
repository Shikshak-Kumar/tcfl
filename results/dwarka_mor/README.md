# Results — Dwarka Mor, Delhi (Urban OSM Map)

Training and evaluation results for three traffic signal control algorithms
on the **Dwarka Mor, Delhi** SUMO map (`backend/sumo_configs2/`).

---

## Map & Simulation Setup

| Property | Value |
|---|---|
| Map | Dwarka Mor, Delhi (real OpenStreetMap export) |
| Simulation engine | SUMO (headless, no GUI) |
| TLS controlled | 6 intersections (node_0 – node_5) |
| Steps per episode | 500 (FedDQN) / 720 (MA2C) |
| Mode | `SUMO-headless` |

---

## Algorithms Compared

| Folder | Algorithm | Reference |
|---|---|---|
| `adaptflow/` | **AdaptFlow-TSC** (our method) | This paper |
| `fed_dqn_tsc/` | FedDQN-TSC (Baseline 1) | Ye et al., Scientific Reports 2023 |
| `multi_agent_ac/` | MA2C (Baseline 2) | Chu et al., IEEE TITS 2019 |
| `plots/` | Comparison charts | — |

---

## Key Metric Definitions

| Metric | Definition | Better |
|---|---|---|
| `avg_waiting_time` | Avg seconds a vehicle waits at a red light per step per lane | Lower |
| `avg_queue_length` | Avg halted vehicles per lane per step | Lower |
| `throughput_ratio` | `arrived / (arrived + remaining_queue)` — fraction of vehicles that cleared | Higher (→ 1.0) |
| `avg_reward` | Per-step reward signal (negative = congestion cost) | Higher (→ 0) |
| `loss` | Neural network training loss | Lower / decreasing |

---

## Training Configuration (same for all algorithms — fair comparison)

| Parameter | FedDQN-TSC | MA2C | AdaptFlow-TSC |
|---|---|---|---|
| FL Rounds | 10 | — | 10 |
| Episodes per round | 20 | — | 10 |
| Total steps | ~100,000 | 30,000 | ~60,000 |
| Nodes | 6 | 6 | 6 |

---

## Quick Summary of Results

| Algorithm | Avg Wait (s) | Avg Queue | TP Ratio | Converged? |
|---|---|---|---|---|
| **AdaptFlow-TSC** | 0.59–3.17s/node | 0.03–0.11 | ~0.342 | Yes (clustering adapted) |
| **FedDQN-TSC** | 7.4s (R1) → 11.7s (R10) | 0.63 → 1.00 | 0.994 (high) | Partially |
| **MA2C** | 0–2.1s (fluctuating) | 0–0.55 | 0.64–1.0 | Fluctuating |

> See individual folder READMEs for detailed per-round and per-node breakdowns.
