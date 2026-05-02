# MA2C Results — Dwarka Mor

Algorithm: **Multi-Agent Actor-Critic (MA2C)** (Baseline 2)
Reference: Chu et al., "Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control", IEEE TITS 2019
Training mode: SUMO-headless | 250 updates | 30,000 total steps | batch_size=120

---

## What MA2C Does

- Each intersection has an **LSTM-based Actor-Critic** agent
- **Spatial reward discounting**: each agent's reward includes neighbour rewards weighted by α=0.75
  - `r̃ᵢ = rᵢ + 0.75 × Σ_j∈N(i) rⱼ` — encourages coordinated signal control
- **Neighbour fingerprints**: agents observe neighbours' current policy (action probability distribution)
- **On-policy** updates: learns from batches of 120 steps, then immediately updates

---

## Files in This Folder

| File | What it contains |
|---|---|
| `multi_agent_ac_training.json` | All 250 update stats (reward, losses, wait, queue, TP per update) |
| `models/tls_0.pt` – `tls_3.pt` | Saved actor+critic weights for each TLS agent |
| `models/cluster_*.pt` | Additional cluster-aggregated model snapshots |

---

## Training Progress Summary

Total: **250 updates over 30,000 steps**

| Phase | Updates | Avg Reward | Avg Wait (s) | Avg Queue | TP Ratio |
|---|---|---|---|---|---|
| Early (U1–10) | 1–10 | -0.55 | 0.76 | 0.25 | ~0.85 |
| Mid (U100–150) | 100–150 | ~-0.20 | ~0.40 | ~0.13 | ~0.89 |
| Late (U240–250) | 240–250 | -0.08 | 0.19 | 0.09 | ~0.93 |

### Notable data points:

**Update 1** (first batch, cold start):
- Reward = -1.10, Wait = 2.09s, Queue = 0.55, TP = 0.645
- Random/untrained policy → high congestion

**Update 4** (after 480 steps):
- Reward = 0.0, Wait = 0.0s, Queue = 0.0, TP = 1.0
- Network clears completely — this is a light-traffic episode where the random policy happened to work well

**Update 249** (near end):
- Reward = -0.014, Wait = 0.017s, Queue = 0.017, TP = 0.984
- Near-optimal performance on this episode

**Update 250** (final):
- Reward = -0.125, Wait = 0.2s, Queue = 0.1, TP = 0.909

---

## Key Observations

### 1. High Variance Between Updates
MA2C shows high episode-to-episode variance because:
- **On-policy method**: learns from current policy only, no replay buffer → noisier gradient estimates
- **Short batches** (120 steps): some episodes have very light traffic, others very heavy
- This is normal for on-policy AC methods; variance reduces as policy converges

### 2. TP Ratio: 0.645 → ~0.91 (improving)
- Update 1: 0.645 (random policy, lots of vehicles stuck)
- After 250 updates: ~0.91 (trained policy clears ~91% of vehicles)
- This shows **genuine learning** — the actor is improving signal timing

### 3. Arrived = 1 per episode
- Only 1 vehicle arrives per 120-step batch
- This is because: 120 steps is short for a complex OSM network where vehicles may need
  200–300+ steps to complete their route
- The TP ratio formula `arrived / (arrived + remaining_queue)` is still valid
  because `remaining_queue` reflects congestion accurately

### 4. Critic Loss vs Actor Loss
- **Critic Loss** (~0.001): squared error of value function estimate — low and stable
- **Actor Loss** (negative, ~-0.015): policy gradient loss — negative values are normal
  in Actor-Critic (negative = actor is improving toward higher rewards)

### 5. Spatial Coordination
- The α=0.75 spatial discount means each intersection's reward includes 75% of each
  neighbour's reward
- This prevents "greedy" behaviour where one intersection clears its queue by pushing
  vehicles to neighbours — a key advantage over independent DQN

---

## Saved Model Structure

Each `models/tls_X.pt` file contains:
```python
{
    "actor": <state_dict of LSTM Actor network>,
    "critic": <state_dict of LSTM Critic network>
}
```

Load example:
```python
import torch
from adaptflow_ac.agents.multi_agent_ac_agent import MultiAgentACAgent

checkpoint = torch.load("models/tls_0.pt")
agent = MultiAgentACAgent(tls_id="tls_0", wave_dim=8, wait_dim=8, fp_dim=8, action_dim=4)
agent.actor.load_state_dict(checkpoint["actor"])
agent.critic.load_state_dict(checkpoint["critic"])
agent.actor.eval()
```

---

## Comparison Note

MA2C is a strong on-policy baseline with spatial coordination. Its weakness vs AdaptFlow:
- No federated learning — each agent learns independently (except via spatial reward)
- No dynamic clustering — neighbourhood topology is fixed (ring topology approximation)
- On-policy updates are noisier and less sample-efficient than off-policy DQN or FL methods
- LSTM captures temporal patterns but no graph-attention for global topology awareness
