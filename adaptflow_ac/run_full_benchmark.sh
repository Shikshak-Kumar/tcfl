#!/bin/bash

# Exit on error
set -e

echo "=========================================================="
echo "Starting Full AdaptFlow Training and Comparison Benchmark"
echo "=========================================================="

# Define scenarios and their result directories
SCENARIOS=("china_osm" "china_rural_osm" "pikhuwa_osm" "china")
RESULTS_DIRS=(
    "backend/results_adaptflow_china_osm"
    "backend/results_adaptflow_china_rural_osm"
    "backend/results_adaptflow_india_rural_pikhuwa_osm"
    "backend/results_adaptflow_china"
)

# Step 1: Training on Mock Data
echo -e "\n[Step 1] Training all scenarios on Mock Data..."
for i in "${!SCENARIOS[@]}"; do
    SCENARIO=${SCENARIOS[$i]}
    RES_DIR=${RESULTS_DIRS[$i]}
    
    echo -e "\n--- Training Scenario: $SCENARIO ---"
    python3 train/train_adaptflow.py \
        --sumo-scenario "$SCENARIO" \
        --rounds 5 \
        --steps 500 \
        --nodes 4 \
        --results-dir "$RES_DIR"
done

# Step 2: Comparison in Mock Mode
echo -e "\n[Step 2] Running Comparisons in Mock Mode..."
for i in "${!SCENARIOS[@]}"; do
    SCENARIO=${SCENARIOS[$i]}
    RES_DIR=${RESULTS_DIRS[$i]}
    MODEL_PATH="$RES_DIR/adaptflow_r5.pt"
    
    if [ -f "$MODEL_PATH" ]; then
        echo -e "\n--- Comparing weights from: $SCENARIO ---"
        python3 compare.py \
            --mode mock \
            --af-model "$MODEL_PATH"
        
        # Move specific report to avoid overwriting
        mkdir -p "results/benchmarks/$SCENARIO"
        mv results/comparison_report.json "results/benchmarks/$SCENARIO/"
        cp -r results/comparison_plots "results/benchmarks/$SCENARIO/"
    else
        echo "Warning: Model $MODEL_PATH not found, skipping comparison."
    fi
done

echo -e "\n=========================================================="
echo "Benchmark Complete! Results are in results/benchmarks/"
echo "=========================================================="
