# Baseline Traffic Control Algorithms

This directory contains implementations of classical traffic control algorithms for comparison with your federated learning approach.

## Implemented Baselines

### 1. **Fixed-Time Control** (`fixed_time.py`)
- **Era**: Pre-1980s (traditional)
- **Method**: Cycles through phases with fixed durations regardless of traffic
- **Reference**: Traditional traffic engineering practice
- **Use Case**: Baseline representing traditional timer-based systems

### 2. **MaxPressure** (`maxpressure.py`)
- **Era**: 2013-present (state-of-the-art classical)
- **Method**: Pressure-based optimization (maximizes queue difference)
- **Reference**: Varaiya, P. (2013). "Max pressure control of a network of signalized intersections"
- **Use Case**: High-performance classical algorithm, important baseline

### 3. **Centralized RL** (`centralized_rl.py`)
- **Era**: 2010s-present (RL era)
- **Method**: Single RL agent trained on pooled data from all intersections
- **Reference**: Standard centralized RL approach
- **Use Case**: Demonstrates benefits of federated learning (privacy, scalability)

### 4. **Independent RL** (`independent_rl.py`)
- **Era**: 2010s-present (RL era)
- **Method**: Each intersection trains its own RL agent independently
- **Reference**: Standard independent RL approach
- **Use Case**: Demonstrates benefits of collaboration in federated learning

## Usage

### Quick Comparison

Run all baselines against your federated learning method:

```bash
python scripts/compare_all_baselines.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 5 \
    --results-dir baseline_comparison
```

### Individual Baseline Testing

```python
from baselines.fixed_time import FixedTimeController
from agents.traffic_environment import SUMOTrafficEnvironment
import traci

# Setup environment
env = SUMOTrafficEnvironment("sumo_configs/intersection.sumocfg", gui=False)
env.start_simulation()

# Create controller
controller = FixedTimeController(env.tl_id)
controller.reset()

# Run simulation
for step in range(400):
    current_time = traci.simulation.getTime()
    controller.step(current_time)
    traci.simulationStep()

# Get metrics
metrics = env.get_performance_metrics()
env.close()
```

## Expected Results

Your federated learning method should outperform:

1. **Fixed-Time**: Should show significant improvement (20-40% reduction in waiting time)
2. **MaxPressure**: May be competitive, but FL should show better fairness and performance
3. **Centralized RL**: Similar performance, but FL has privacy/scalability benefits
4. **Independent RL**: Should show clear benefit of collaboration (10-30% improvement)

## Key Metrics for Comparison

- **Waiting Time**: Total and average per vehicle
- **Queue Length**: Average and maximum
- **Throughput**: Vehicles per hour
- **Fairness**: Distribution of waiting times across intersections
- **Convergence**: Time/rounds to reach stable performance

## Research Paper Comparison Table

For your paper, create a table like this:

| Method | Avg Waiting Time (s) | Avg Queue Length | Throughput (veh/h) | Fairness Index |
|--------|---------------------|------------------|-------------------|----------------|
| Fixed-Time | X ± Y | A ± B | C | D |
| MaxPressure | X ± Y | A ± B | C | D |
| Centralized RL | X ± Y | A ± B | C | D |
| Independent RL | X ± Y | A ± B | C | D |
| **Federated FL (Ours)** | **X ± Y** | **A ± B** | **C** | **D** |

## Notes

- **Statistical Significance**: Run multiple seeds (5-10 runs) and report mean ± std
- **Fairness**: Your fairness-aware aggregation should show better fairness metrics
- **Privacy**: Emphasize that FL preserves privacy (no data sharing) vs centralized RL
- **Scalability**: Show that FL scales better than centralized RL with more intersections

## References

1. Varaiya, P. (2013). Max pressure control of a network of signalized intersections. *Transportation Research Part C*, 36, 177-195.

2. Gershenson, C. (2005). Self-organizing traffic lights. *Complex Systems*, 16(1), 29-53.

3. Traffic Engineering Handbook (various editions). Institute of Transportation Engineers.

