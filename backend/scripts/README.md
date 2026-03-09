# Baseline Comparison Scripts

This directory contains scripts for comparing your federated learning method against classical traffic control algorithms.

## Quick Start

### Option 1: Run Everything (Recommended)

Run comparison and generate visualizations in one command:

```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --num-runs 3 \
    --num-steps 200
```

### Option 2: Run Separately

1. **Run comparison:**
```bash
python3 scripts/compare_all_baselines.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 5 \
    --num-steps 400 \
    --results-dir baseline_comparison
```

2. **Generate visualizations:**
```bash
python3 scripts/visualize_comparison.py \
    --results-file baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json \
    --output-dir comparison_plots
```

## Scripts

### `compare_all_baselines.py`
Main comparison script that runs all baseline methods:
- Fixed-Time Control
- Actuated Control
- MaxPressure
- SOTL
- Centralized RL
- Independent RL
- Federated Learning (your method)

**Output:** JSON file with comparison results

### `visualize_comparison.py`
Creates comprehensive visualization plots:
- Waiting time comparison
- Queue length comparison
- Improvement percentages
- Radar charts
- Summary tables

**Output:** PNG files in `comparison_plots/` directory

### `run_and_visualize.py`
Convenience script that runs comparison and visualization together.

## Parameters

### Comparison Parameters
- `--sumo-config`: Single intersection config for classical baselines
- `--sumo-configs-multi`: Multiple configs for RL methods
- `--num-runs`: Number of runs per method (default: 5)
- `--num-steps`: Simulation steps (default: 400)
- `--results-dir`: Output directory (default: baseline_comparison)
- `--gui`: Enable SUMO GUI (slower but visual)

### Visualization Parameters
- `--results-file`: Path to comparison results JSON
- `--output-dir`: Output directory for plots (default: comparison_plots)

## Output Files

### Comparison Results
- `baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json`
  - Contains all raw results and summary statistics

### Visualizations
- `comparison_plots/baseline_comparison.png` - Comprehensive 6-panel comparison
- `comparison_plots/waiting_time_comparison.png` - Waiting time bar chart
- `comparison_plots/queue_length_comparison.png` - Queue length bar chart

## Notes

- **RL methods** (Centralized RL, Independent RL, Federated) take longer to train
  - They only run once (not multiple runs) due to training time
- **Classical methods** run quickly and can be run multiple times for statistics
- **Reduced steps**: Use `--num-steps 200` for faster testing
- **GUI**: Use `--gui` flag to see SUMO simulation (slower)

## Troubleshooting

### "SUMO not found"
Make sure SUMO is installed and in PATH:
```bash
sumo --version
```

### "No results found"
- Check that SUMO config files exist
- Verify paths are correct
- Check for errors in console output

### "Import errors"
Make sure you're running from project root:
```bash
cd /path/to/project
python3 scripts/compare_all_baselines.py ...
```

## Example Output

After running, you'll see:
```
============================================================
COMPARISON SUMMARY
============================================================

Method               Avg Waiting Time    Avg Queue Length    Runs
----------------------------------------------------------------------
fixed_time                   45.23 ± 2.15        8.45 ± 0.32          5
actuated                     38.12 ± 1.89        7.23 ± 0.28          5
maxpressure                   32.45 ± 1.67        6.12 ± 0.25          5
sotl                         35.67 ± 1.92        6.89 ± 0.30          5
centralized_rl               28.34 ± 1.45        5.45 ± 0.22          1
independent_rl                30.12 ± 1.56        5.78 ± 0.24          1
federated                     25.67 ± 1.23        4.89 ± 0.19          1
```

## For Research Paper

Use the generated plots and statistics in your paper:
1. Include comparison table (from summary)
2. Use bar charts for visual comparison
3. Highlight federated method (gold border in plots)
4. Report statistical significance (mean ± std)

