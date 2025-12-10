#!/bin/bash
# Quick Start Script for Baseline Comparison
# Run this script to set up and run the comparison

set -e  # Exit on error

echo "=========================================="
echo "Baseline Comparison - Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${YELLOW}[1/6]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} Found: $PYTHON_VERSION"
echo ""

# Step 2: Check SUMO
echo -e "${YELLOW}[2/6]${NC} Checking SUMO installation..."
if ! command -v sumo &> /dev/null; then
    echo -e "${RED}✗ SUMO not found in PATH.${NC}"
    echo "Please install SUMO from: https://sumo.dlr.de/docs/Downloads.php"
    echo "Or add SUMO to PATH: export PATH=\$PATH:\$SUMO_HOME/bin"
    exit 1
fi
SUMO_VERSION=$(sumo --version 2>&1 | head -1)
echo -e "${GREEN}✓${NC} Found: $SUMO_VERSION"
echo ""

# Step 3: Check project files
echo -e "${YELLOW}[3/6]${NC} Checking project files..."
if [ ! -f "scripts/compare_all_baselines.py" ]; then
    echo -e "${RED}✗ Project files not found.${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi
if [ ! -f "sumo_configs/intersection.sumocfg" ]; then
    echo -e "${RED}✗ SUMO config files not found.${NC}"
    echo "Please ensure sumo_configs/ directory exists with config files."
    exit 1
fi
echo -e "${GREEN}✓${NC} Project files found"
echo ""

# Step 4: Install dependencies
echo -e "${YELLOW}[4/6]${NC} Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --break-system-packages --quiet
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} requirements.txt not found, installing core packages..."
    python3 -m pip install numpy matplotlib torch flwr --break-system-packages --quiet
    echo -e "${GREEN}✓${NC} Core packages installed"
fi
echo ""

# Step 5: Test imports
echo -e "${YELLOW}[5/6]${NC} Testing imports..."
if python3 scripts/test_imports.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} All imports successful"
else
    echo -e "${YELLOW}⚠${NC} Some imports failed, but continuing..."
fi
echo ""

# Step 6: Run comparison
echo -e "${YELLOW}[6/6]${NC} Running baseline comparison..."
echo ""
echo "Choose run mode:"
echo "1) Quick test (3 runs, 200 steps) - ~10-30 minutes"
echo "2) Full comparison (10 runs, 400 steps) - ~1-3 hours"
read -p "Enter choice [1 or 2]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        python3 scripts/run_and_visualize.py \
            --sumo-config sumo_configs/intersection.sumocfg \
            --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
            --num-runs 3 \
            --num-steps 200 \
            --results-dir baseline_comparison
        ;;
    2)
        echo ""
        echo "Running full comparison (this will take a while)..."
        python3 scripts/run_and_visualize.py \
            --sumo-config sumo_configs/intersection.sumocfg \
            --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
            --num-runs 10 \
            --num-steps 400 \
            --results-dir baseline_comparison
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Comparison Complete!${NC}"
echo "=========================================="
echo ""
echo "Results saved to: baseline_comparison/"
echo "Plots saved to: comparison_plots/"
echo ""
echo "View results:"
echo "  python3 scripts/show_results_text.py --results-file baseline_comparison/comparison_results_*.json"
echo ""

