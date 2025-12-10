# Comparison Methods

## Overview

This comparison evaluates **5 traffic control methods**:
- 4 baseline methods (for comparison)
- 1 federated learning method (your method)

---

## Methods Compared

### 1. Fixed-Time Control
- **Type:** Traditional baseline
- **Method:** Timer-based, fixed phase durations
- **Why included:** Represents traditional traffic control systems
- **Expected performance:** Baseline (worst performance)

### 2. MaxPressure
- **Type:** State-of-the-art classical method
- **Method:** Pressure-based optimization (maximizes queue difference)
- **Why included:** Best performing classical algorithm
- **Expected performance:** Good (better than Fixed-Time, worse than RL methods)
- **Reference:** Varaiya, P. (2013). "Max pressure control of a network of signalized intersections"

### 3. Centralized RL
- **Type:** Non-federated reinforcement learning
- **Method:** Single RL agent trained on pooled data from all intersections
- **Why included:** Shows privacy benefits of federated learning
- **Expected performance:** Good (similar to federated, but requires data sharing)
- **Key difference:** Shares raw traffic data (privacy concern)

### 4. Independent RL
- **Type:** Non-collaborative reinforcement learning
- **Method:** Each intersection trains its own RL agent independently
- **Why included:** Shows collaboration benefits of federated learning
- **Expected performance:** Good (worse than federated due to no collaboration)
- **Key difference:** No knowledge sharing between intersections

### 5. Federated FL (Your Method) ⭐
- **Type:** Federated learning with fairness-aware aggregation
- **Method:** Collaborative learning with privacy preservation
- **Why included:** Your proposed method
- **Expected performance:** Best (combines privacy + collaboration + fairness)
- **Key advantages:**
  - Privacy-preserving (no raw data sharing)
  - Collaborative (knowledge sharing)
  - Fairness-aware (congestion-based aggregation)

---

## Why These 4 Baselines?

### Fixed-Time
- **Purpose:** Traditional baseline
- **Shows:** Improvement over conventional systems

### MaxPressure
- **Purpose:** Best classical method
- **Shows:** RL methods outperform even state-of-the-art classical approaches

### Centralized RL
- **Purpose:** Performance comparison
- **Shows:** Federated learning achieves similar/better performance while preserving privacy

### Independent RL
- **Purpose:** Collaboration comparison
- **Shows:** Federated learning benefits from knowledge sharing

---

## Expected Results

Based on typical traffic control performance:

| Method | Waiting Time | Queue Length | Key Finding |
|--------|-------------|--------------|-------------|
| Fixed-Time | Highest | Highest | Traditional baseline |
| MaxPressure | Medium-High | Medium-High | Best classical method |
| Centralized RL | Low | Low | Good performance, but privacy concern |
| Independent RL | Low-Medium | Low-Medium | Good, but no collaboration |
| **Federated FL** | **Lowest** | **Lowest** | **Best overall** |

---

## For Your Research Paper

### Comparison Table Format

| Method | Avg Waiting Time (s) | Avg Queue Length | Improvement Over Fixed-Time |
|--------|---------------------|------------------|----------------------------|
| Fixed-Time | X ± Y | A ± B | Baseline |
| MaxPressure | X ± Y | A ± B | +X% |
| Centralized RL | X ± Y | A ± B | +X% |
| Independent RL | X ± Y | A ± B | +X% |
| **Federated FL (Ours)** | **X ± Y** | **A ± B** | **+X%** |

### Key Points to Highlight

1. **Better than Fixed-Time:** Shows improvement over traditional systems
2. **Better than MaxPressure:** Shows RL methods outperform classical optimization
3. **Privacy-preserving:** Similar performance to Centralized RL without data sharing
4. **Collaborative:** Better than Independent RL due to knowledge sharing
5. **Fairness-aware:** Your method includes fairness considerations

---

## Running the Comparison

```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400
```

This will compare all 5 methods and generate results.

