# Train vs Simulate Flow

This document explains what happens internally when you **train** an algorithm or **simulate** it via the frontend, for all supported algorithms: **FedAvg**, **FedCM**, **FedKD**, **FedFlow**, and **AdaptFlow**.

---

## 🏋️ TRAIN (`python train_<algo>.py`)

```
START
  │
  ▼
Trainer created
  │  - Creates N agents with fresh (random) weights
  │  - Sets up traffic environments:
  │      Mock (default) | TomTom (real-time) | SUMO / SUMO-GUI (headless/GUI)
  │
  ▼
For each Federated Round (1 to NUM_ROUNDS):
  │
  ├─ Step 1: Local Training
  │     Each agent runs in its own environment for NUM_STEPS
  │     → collects (state, action, reward, next_state) experience
  │     → trains local DQN via replay buffer
  │
  ├─ Step 2: [AdaptFlow / FedFlow only] Clustering
  │     FedFlow   → static clusters assigned at init, reused every round
  │     AdaptFlow → dynamic re-clustering based on congestion fingerprints
  │                  (Round 1: static; Round 2+: re-cluster by similarity)
  │
  ├─ Step 3: Federated Aggregation
  │     FedAvg / FedCM / FedKD → flat weighted average of all agent weights
  │     FedFlow / AdaptFlow     → hierarchical:
  │                                  intra-cluster average → inter-cluster average
  │     → ALL agents now share identical global weights
  │
  └─ Step 4: Save Checkpoints
        results_<algo>/node_X_round_N_model.pt    ← per-node checkpoint
        results_<algo>/round_N_summary.json        ← round metrics
  │
  ▼
After ALL Rounds Complete:
  │
  ├─ results_<algo>/<algo>_global_mock.pt          ← final global model (Mock/TomTom)
  ├─ results_<algo>/<algo>_global_sumo.pt          ← final global model (SUMO)
  ├─ results_<algo>/adaptflow_all_rounds.json      ← (AdaptFlow) combined history
  ├─ results_<algo>/cluster_history.json           ← (AdaptFlow/FedFlow) cluster log
  └─ saved_models/<algo>/model.pt                  ← optimized TorchScript for deployment ✅

END
```

---

## 🎮 SIMULATE (Frontend → "Simulate" Button)

```
User clicks Simulate in UI
  │
  ▼
Frontend sends config over WebSocket
  │  { algorithm, intersections, city, use_tomtom, sumo_scenario, target_pois }
  │
  ▼
server.py → run_<algo>_simulation(websocket, config)
  │
  ▼
Trainer / Clients created
  │  - Fresh agents with random weights
  │  - Environments built from frontend intersection pins
  │
  ▼
get_algo_model_path() — resolve which saved model to load
  │
  │  Priority order (first match wins):
  │  1. saved_models/<algo>_china_osm/model.pt   (if china_osm scenario)
  │  2. saved_models/<algo>_china/model.pt        (if china scenario)
  │  3. saved_models/<algo>/model.pt              ← production model ✅
  │  4. results_<algo>/<algo>_global_sumo.pt      ← SUMO training checkpoint
  │  5. results_<algo>/<algo>_global_mock.pt      ← Mock training checkpoint
  │  6. results_<algo>/*.pt (most recently modified)
  │  7. No file found → agents run with random weights ⚠️
  │
  ▼
Load weights into ALL agents
  │
  ▼
Run 200 Inference Steps (NO training, NO weight updates)
  │  - agent.act(state) / agent.get_action(state)  ← picks actions only
  │  - env.step(action)                             ← steps the environment
  │  - Stream step result to frontend via WebSocket
  │      { step, intersections: { reward, queue, wait, vehicles, ... }, global: { avg_reward, total_queue } }
  │
  ▼
Save session data + generate plots (viz_manager)

END
```

---

## Algorithm-Specific Details

| Algorithm | Train Script | Results Dir | Agent Type | Aggregation |
|-----------|-------------|-------------|------------|-------------|
| **FedAvg** | `train_federated.py` | `results_federated/` | `TrafficFLClient` (DQN) | Flat FedAvg |
| **FedCM** | `train_fedcm.py` | `results_fedcm_sumo/` | `DQNAgent` | Flat weighted avg |
| **FedKD** | `train_fedkd.py` | `results_fedkd_sumo/` | `DQNAgent` | Knowledge distillation |
| **FedFlow** | `train_fedflow.py` | `results_fedflow/` | `FedFlowAgent` | Hierarchical (static clusters) |
| **AdaptFlow** | `train_adaptflow.py` | `results_adaptflow/` | `AdaptFlowAgent` | Hierarchical (dynamic clusters) |

---

## Key Difference: Train vs Simulate

| | Train | Simulate |
|---|---|---|
| **Weights** | Start random → **learned each round** | Loaded from file → **frozen** |
| **Saves model?** | ✅ Yes — checkpoints + global model | ❌ No |
| **Federated rounds?** | ✅ Yes | ❌ No |
| **Environments** | Configured by training script | Built from frontend intersection pins |
| **Output** | `.pt` model files + JSON metrics | Live WebSocket stream to frontend UI |
| **Duration** | Minutes to hours (GPU recommended) | ~20 seconds (200 steps × 0.1s) |

---

## Where Models Live

```
backend/
├── results_adaptflow/           ← raw training outputs (checkpoints per round)
│   ├── adaptflow_global_mock.pt ← final model after training (fallback)
│   └── node_X_round_N_model.pt  ← per-node per-round snapshots
│
├── saved_models/                ← optimized production models (priority 1 for simulation)
│   ├── adaptflow/model.pt
│   ├── fedflow/model.pt
│   ├── fedcm/model.pt
│   ├── fedkd/model.pt
│   └── federated/model.pt
│
└── save_all_models.py           ← manually re-export all algos to saved_models/
```

> **Tip:** If simulation behaves poorly (random actions), it means no trained model was found.
> Run the corresponding `train_<algo>.py` first, or run `python save_all_models.py` to export existing weights.
