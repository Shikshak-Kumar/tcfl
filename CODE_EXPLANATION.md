# Code Explanation: How Your Federated Learning Traffic Control System Works

## 🎯 Overview

Your system uses **Federated Learning** to train multiple traffic intersections collaboratively. Each intersection learns locally, then shares only model weights (not raw data) with others, creating a smarter traffic control system while preserving privacy.

---

## 📊 System Architecture

```
┌─────────────────┐
│  SUMO Traffic   │  ← Traffic Simulation
│   Environment   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DQN Agent     │  ← Deep Q-Network (AI Brain)
│  (Neural Net)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FL Client       │  ← Individual Intersection
│ (Local Training)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fairness-Aware  │  ← Aggregation with Weights
│  Aggregation    │
└─────────────────┘
```

---

## 🔄 Complete Flow: Step-by-Step

### **PHASE 1: Initialization**

#### 1.1 Create Clients (Intersections)
```python
# train_federated.py:227-248
clients = []
client_configs = [
    {"id": "client_1", "config": "sumo_configs/osm_client1.sumocfg"},
    {"id": "client_2", "config": "sumo_configs/osm_client2.sumocfg"}
]
```
**What happens:**
- Creates 2 client objects (representing 2 intersections)
- Each client gets its own SUMO traffic simulation
- Each client has its own DQN agent (neural network)

#### 1.2 Initialize DQN Agent
```python
# fl_client.py:26
self.agent = DQNAgent(state_size=12, action_size=4)
```
**What happens:**
- Creates neural network: 12 inputs → 128 → 128 → 64 → 4 outputs
- 12 inputs = traffic state (vehicle count, queue, waiting time per edge)
- 4 outputs = Q-values for 4 traffic light phases
- Initializes experience replay memory (50,000 capacity)
- Sets exploration rate (epsilon = 1.0, starts random)

#### 1.3 Initialize Traffic Environment
```python
# fl_client.py:28
self.env = SUMOTrafficEnvironment(sumo_config_path, gui=False)
```
**What happens:**
- Connects to SUMO simulation
- Discovers traffic light ID and incoming edges
- Sets up state/action/reward system
- Configures phase durations and safety constraints

---

### **PHASE 2: Federated Learning Rounds** (Repeats 15 times)

#### 2.1 Start Round
```python
# train_federated.py:253
for round_num in range(num_rounds):  # 15 rounds
    global_params = None  # First round: no global model yet
```

**What happens:**
- Begins new federated learning round
- First round: each client starts with random weights
- Later rounds: clients start with aggregated global model

---

### **PHASE 3: Local Training** (Each Client Trains Independently)

#### 3.1 Receive Global Model
```python
# train_federated.py:261
current_params = global_params if global_params is not None else client.get_parameters({})
client.set_parameters(current_params)
```
**What happens:**
- Client receives global model weights from previous round
- Updates its local neural network with these weights
- **Privacy preserved:** Only weights shared, no traffic data

#### 3.2 Train on Local Data
```python
# fl_client.py:82-126 (_train_agent method)
for episode in range(episodes):  # 3 episodes per round
    state = env.reset()  # Start new traffic simulation
    for step in range(max_steps_per_episode):  # Up to 1000 steps
        action = agent.act(state, training=True)  # Choose action
        next_state, reward, done = env.step(action)  # Execute action
        agent.remember(state, action, reward, next_state, done)  # Store experience
        if memory full:
            loss = agent.replay()  # Learn from experiences
```

**What happens (detailed):**

**Step 3.2.1: Get State**
```python
# traffic_environment.py:154-172
state = self.get_state()
# Returns: [vehicle_count_edge1, queue_edge1, waiting_edge1,
#           vehicle_count_edge2, queue_edge2, waiting_edge2, ...]
# Normalized to 0-1 range (12 values total)
```
- Observes current traffic conditions
- Measures: vehicles, queues, waiting times on each incoming road
- Normalizes values for neural network

**Step 3.2.2: Choose Action**
```python
# dqn_agent.py:68-75
action = agent.act(state, training=True)
# Returns: 0, 1, 2, or 3 (traffic light phase)
```
- **Exploration:** Random action (epsilon% chance) - explores new strategies
- **Exploitation:** Best action from neural network - uses learned knowledge
- Epsilon decays over time (starts random, becomes smarter)

**Step 3.2.3: Execute Action**
```python
# traffic_environment.py:344-426
next_state, reward, done, info = env.step(action)
```
- Sets traffic light phase (0=Green NS, 1=Yellow, 2=Green EW, 3=Yellow)
- Advances SUMO simulation by 1 second
- Collects new state and metrics

**Step 3.2.4: Calculate Reward**
```python
# traffic_environment.py:490-523
reward = self._calculate_reward()
# Formula:
# reward = 0.6*(wait_component) + 0.6*(queue_component) + 
#          0.4*(speed_component) + 0.02*(throughput_component)
# Clipped to [-1, +1]
```
- **Reward components:**
  - Waiting time: Lower is better (60% weight)
  - Queue length: Lower is better (60% weight)
  - Speed: Higher is better (40% weight)
  - Throughput: More vehicles is better (2% weight)
- Clipped to [-1, +1] to prevent Q-value explosion

**Step 3.2.5: Store Experience**
```python
# dqn_agent.py:64-66
agent.remember(state, action, reward, next_state, done)
# Stores: (state, action, reward, next_state, done) in memory
```
- Saves experience tuple in replay buffer
- Used later for learning (experience replay)

**Step 3.2.6: Learn from Experiences**
```python
# dqn_agent.py:77-121
if len(memory) > batch_size:  # 64 experiences
    loss = agent.replay()
```
- Samples 64 random experiences from memory
- Uses **Double DQN** algorithm:
  - Policy network selects best action
  - Target network evaluates Q-value
- Updates neural network weights using gradient descent
- Clips gradients to prevent instability
- Soft updates target network (Polyak averaging)

**Step 3.2.7: Update Exploration**
```python
# dqn_agent.py:118-119
if epsilon > epsilon_min:
    epsilon *= epsilon_decay  # 0.997
```
- Reduces exploration rate each step
- Starts at 100% random, decays to 5% random
- Balances exploration vs exploitation

#### 3.3 Return Updated Weights
```python
# fl_client.py:61-65
return (
    self.get_parameters(config),  # Updated neural network weights
    episodes * max_steps,         # Number of training samples
    training_metrics               # Performance metrics
)
```
**What happens:**
- Client finishes local training
- Returns updated model weights
- **Key:** Only weights shared, never raw traffic data

---

### **PHASE 4: Evaluation** (Test Performance)

#### 4.1 Evaluate Each Client
```python
# train_federated.py:283-291
for client, params in zip(clients, params_list):
    eval_metrics = client.evaluate(params, {"round": round_num})
```
**What happens:**
- Tests trained model on new traffic simulation
- Agent uses learned policy (no exploration)
- Measures: waiting time, queue length, throughput

#### 4.2 Calculate Congestion Scores
```python
# train_federated.py:293-304
lane_wait = eval_metrics.get("total_waiting_time", 0.0)
num_congested = eval_metrics.get("num_congested_lanes", 0.0)
avg_queue = eval_metrics.get("total_queue_length", 0.0)

norm_wait = min(lane_wait / 300.0, 1.0)      # Normalize waiting
norm_queue = min(avg_queue / 50.0, 1.0)      # Normalize queue
norm_cong = num_congested / total_lanes      # Normalize congestion

congestion = 0.4 * norm_wait + 0.3 * norm_queue + 0.3 * norm_cong
```
**What happens:**
- Calculates congestion score for each intersection
- **Formula:** Weighted combination of waiting time (40%), queue length (30%), congested lanes (30%)
- Higher score = more congested intersection
- Used for fairness-aware aggregation

---

### **PHASE 5: Fairness-Aware Aggregation** (Your Key Innovation!)

#### 5.1 Calculate Aggregation Weights
```python
# train_federated.py:306-312
sum_cong = sum(congestion_scores)
if sum_cong > 0:
    weights = [c / sum_cong for c in congestion_scores]
else:
    weights = [1.0 / n] * n  # Equal weights if no congestion
```
**What happens:**
- **Fairness-aware weighting:** More congested intersections get higher weights
- **Why:** Helps congested intersections learn faster
- **Example:** If client_1 has congestion=0.7 and client_2 has congestion=0.3:
  - client_1 weight = 0.7 / (0.7+0.3) = 0.7 (70%)
  - client_2 weight = 0.3 / (0.7+0.3) = 0.3 (30%)

#### 5.2 Aggregate Model Weights
```python
# train_federated.py:213-225
def weighted_avg_params(params_list, weights):
    for layer_idx in range(num_layers):
        layer_sum = None
        for params, w in zip(params_list, weights):
            layer_sum += params[layer_idx] * w
        agg.append(layer_sum)
```
**What happens:**
- Combines neural network weights from all clients
- **Weighted average:** Each client's weights multiplied by its congestion weight
- Creates new global model
- **Privacy preserved:** Only weights aggregated, no data shared

#### 5.3 Distribute Global Model
```python
# train_federated.py:316-317
for client in clients:
    client.set_parameters(global_params)
```
**What happens:**
- Sends aggregated global model to all clients
- Clients update their local models
- Next round starts with improved global knowledge

---

### **PHASE 6: Save Results**

#### 6.1 Save Training Metrics
```python
# train_federated.py:273-277
client.save_training_history(
    f"{client.client_id}_round_{round_num}_train.json"
)
```
**What happens:**
- Saves: average reward, loss, steps per round
- Used for analysis and visualization

#### 6.2 Save Evaluation Metrics
```python
# train_federated.py:287-291
client.save_performance_metrics(
    f"{client.client_id}_round_{round_num}_eval.json"
)
```
**What happens:**
- Saves: waiting time, queue length, per-lane metrics
- Used for comparison with baselines

#### 6.3 Save Detailed Metrics
```python
# train_federated.py:329-335
detailed = client.env.get_performance_metrics()
# Saves: per-lane metrics, green signal times, vehicle types, etc.
```
**What happens:**
- Saves comprehensive performance data
- Includes: per-road metrics, lane summaries, GST calculations

---

## 🧠 Key Components Explained

### **1. DQN Agent (Deep Q-Network)**

**Purpose:** AI brain that learns optimal traffic light control

**How it works:**
- **State → Q-values:** Neural network predicts Q-value for each action
- **Q-value:** Expected future reward for taking that action
- **Action selection:** Chooses action with highest Q-value
- **Learning:** Updates Q-values based on actual rewards received

**Key features:**
- **Experience Replay:** Learns from past experiences (not just current)
- **Target Network:** Stable learning (separate network for Q-value estimation)
- **Double DQN:** Reduces overestimation bias
- **Epsilon-Greedy:** Balances exploration vs exploitation

### **2. Traffic Environment**

**Purpose:** Interface between AI and SUMO traffic simulation

**State representation (12 values):**
- For each of 4 incoming roads:
  - Vehicle count (normalized 0-1)
  - Queue length (normalized 0-1)
  - Waiting time (normalized 0-1)

**Actions (4 phases):**
- Phase 0: Green for North-South roads
- Phase 1: Yellow transition
- Phase 2: Green for East-West roads
- Phase 3: Yellow transition

**Reward function:**
- Encourages: Low waiting time, low queues, high speed, high throughput
- Discourages: Long waits, long queues, slow traffic

### **3. Federated Learning Client**

**Purpose:** Represents one intersection learning locally

**Key methods:**
- `fit()`: Train locally on local traffic data
- `evaluate()`: Test model performance
- `get_parameters()`: Extract model weights
- `set_parameters()`: Update model weights

**Privacy:** Never shares raw traffic data, only model weights

### **4. Fairness-Aware Aggregation**

**Purpose:** Your innovation - weights aggregation based on congestion

**How it works:**
1. Calculate congestion score for each client
2. Normalize congestion scores
3. Use congestion as aggregation weights
4. More congested intersections contribute more to global model

**Benefits:**
- Helps congested intersections learn faster
- Improves fairness across intersections
- Better overall system performance

---

## 🔄 Complete Round Flow Diagram

```
ROUND START
    │
    ├─→ Client 1: Receive global model
    │   ├─→ Train locally (3 episodes)
    │   ├─→ Update neural network
    │   └─→ Return updated weights
    │
    ├─→ Client 2: Receive global model
    │   ├─→ Train locally (3 episodes)
    │   ├─→ Update neural network
    │   └─→ Return updated weights
    │
    ├─→ Evaluate both clients
    │   ├─→ Measure performance
    │   └─→ Calculate congestion scores
    │
    ├─→ Fairness-Aware Aggregation
    │   ├─→ Calculate weights (based on congestion)
    │   ├─→ Weighted average of model weights
    │   └─→ Create new global model
    │
    ├─→ Distribute global model to all clients
    │
    └─→ Save results (training, evaluation, detailed metrics)
    
NEXT ROUND (repeat 15 times)
```

---

## 📈 Learning Process

### **Round 1:**
- Agents start random (epsilon = 1.0)
- Random actions, poor performance
- Begin learning basic patterns

### **Rounds 2-5:**
- Epsilon decays (0.997 each step)
- Agents explore less, exploit more
- Learn which phases work better
- Performance improves gradually

### **Rounds 6-10:**
- Agents mostly exploit learned knowledge
- Fine-tune strategies
- Global model improves through aggregation
- Significant performance gains

### **Rounds 11-15:**
- Agents converge to optimal policy
- Stable performance
- Small improvements
- Ready for evaluation

---

## 🎯 Key Innovations in Your Code

### **1. Fairness-Aware Aggregation**
```python
# Lines 293-312 in train_federated.py
congestion = 0.4 * norm_wait + 0.3 * norm_queue + 0.3 * norm_cong
weights = [c / sum_cong for c in congestion_scores]
```
- **Innovation:** Weight aggregation based on congestion
- **Benefit:** Helps congested intersections learn faster
- **Result:** Better fairness and overall performance

### **2. Privacy-Preserving Learning**
- **Innovation:** Only model weights shared, never raw traffic data
- **Benefit:** Privacy preserved while enabling collaboration
- **Implementation:** `get_parameters()` returns weights, not data

### **3. Collaborative Learning**
- **Innovation:** Multiple intersections share knowledge
- **Benefit:** Each intersection benefits from others' learning
- **Result:** Better than independent learning

---

## 🔍 Important Code Sections

### **Reward Calculation** (traffic_environment.py:490-523)
- Combines multiple factors (waiting, queue, speed, throughput)
- Normalized and clipped to [-1, +1]
- Encourages optimal traffic flow

### **State Representation** (traffic_environment.py:154-172)
- 12-dimensional state vector
- Normalized to 0-1 range
- Captures traffic conditions on all incoming roads

### **Neural Network Architecture** (dqn_agent.py:9-23)
- Input: 12 (state size)
- Hidden: 128 → 128 → 64
- Output: 4 (action size)
- ~50,000 parameters total

### **Experience Replay** (dqn_agent.py:77-121)
- Stores 50,000 experiences
- Samples random batches for learning
- Breaks correlation between consecutive experiences
- Stabilizes learning

---

## 📊 Data Flow

```
SUMO Simulation
    ↓ (traffic data)
Environment.get_state()
    ↓ (12-dim state vector)
DQN Agent.act()
    ↓ (action: 0-3)
Environment.step()
    ↓ (reward, next_state)
DQN Agent.remember()
    ↓ (experience stored)
DQN Agent.replay()
    ↓ (neural network updated)
FL Client.get_parameters()
    ↓ (model weights)
Fairness-Aware Aggregation
    ↓ (global model)
FL Client.set_parameters()
    ↓ (updated model)
Next Round
```

---

## 🎓 Summary

**Your system:**
1. ✅ Creates multiple traffic intersections (clients)
2. ✅ Each learns locally using DQN (Deep Q-Network)
3. ✅ Trains on local traffic data (privacy preserved)
4. ✅ Calculates congestion scores for fairness
5. ✅ Aggregates models with congestion-based weights
6. ✅ Distributes improved global model
7. ✅ Repeats for 15 rounds
8. ✅ Saves all metrics for analysis

**Key advantages:**
- 🔒 **Privacy:** No raw data sharing
- 🤝 **Collaboration:** Knowledge sharing between intersections
- ⚖️ **Fairness:** Congestion-aware aggregation
- 📈 **Performance:** Better than baselines

This is a complete federated learning system for traffic control! 🚀




