#!/bin/bash
# Copy-Paste Commands - Run These in Order
# Copy and paste each section into terminal

# ============================================================================
# SECTION 1: PREREQUISITES CHECK
# ============================================================================

echo "=== Checking Prerequisites ==="
python3 --version
sumo --version
which sumo

# ============================================================================
# SECTION 2: INSTALLATION
# ============================================================================

echo "=== Installing Dependencies ==="
cd /path/to/tt  # CHANGE THIS to your project path
pip3 install -r requirements.txt --break-system-packages

# ============================================================================
# SECTION 3: VERIFICATION
# ============================================================================

echo "=== Verifying Installation ==="
python3 scripts/test_imports.py
ls sumo_configs/
ls scripts/
ls baselines/

# ============================================================================
# SECTION 4: QUICK TEST RUN
# ============================================================================

echo "=== Running Quick Test ==="
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 3 \
    --num-steps 200 \
    --results-dir baseline_comparison

# ============================================================================
# SECTION 5: VIEW RESULTS
# ============================================================================

echo "=== Viewing Results ==="
LATEST_RESULTS=$(ls -t baseline_comparison/comparison_results_*.json | head -1)
echo "Latest results file: $LATEST_RESULTS"
python3 scripts/show_results_text.py --results-file "$LATEST_RESULTS"

# ============================================================================
# SECTION 6: FULL COMPARISON (Run after test succeeds)
# ============================================================================

echo "=== Running Full Comparison ==="
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400 \
    --results-dir baseline_comparison

# ============================================================================
# SECTION 7: GENERATE PLOTS (If needed separately)
# ============================================================================

echo "=== Generating Plots ==="
LATEST_RESULTS=$(ls -t baseline_comparison/comparison_results_*.json | head -1)
python3 scripts/visualize_comparison.py \
    --results-file "$LATEST_RESULTS" \
    --output-dir comparison_plots

# ============================================================================
# TROUBLESHOOTING COMMANDS (Use if needed)
# ============================================================================

# If SUMO not found, add to PATH:
# export SUMO_HOME=/path/to/sumo
# export PATH=$PATH:$SUMO_HOME/bin

# If Python packages fail:
# pip3 install numpy matplotlib torch flwr --break-system-packages

# If traci module not found:
# export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools
# pip3 install sumolib traci --break-system-packages

