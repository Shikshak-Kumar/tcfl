# Complete Command Reference
## All Commands You Need to Run - Explained

This document lists every command you need to run, in order, with explanations.

---

## 📋 Table of Contents
1. [Prerequisites Check](#prerequisites-check)
2. [Installation Commands](#installation-commands)
3. [Verification Commands](#verification-commands)
4. [Running Comparisons](#running-comparisons)
5. [Viewing Results](#viewing-results)
6. [Troubleshooting Commands](#troubleshooting-commands)

---

## 1. Prerequisites Check

### Check Python Version
```bash
python3 --version
```
**What it does:** Verifies Python 3.8+ is installed  
**Expected output:** `Python 3.8.x` or higher  
**If fails:** Install Python from python.org

### Check SUMO Installation
```bash
sumo --version
```
**What it does:** Verifies SUMO is installed and accessible  
**Expected output:** `SUMO version X.X.X`  
**If fails:** Install SUMO from https://sumo.dlr.de/docs/Downloads.php

### Check if SUMO is in PATH
```bash
which sumo
```
**What it does:** Shows where SUMO is installed  
**Expected output:** `/path/to/sumo/bin/sumo`  
**If fails:** Add SUMO to PATH (see troubleshooting)

---

## 2. Installation Commands

### Navigate to Project Directory
```bash
cd /path/to/tt
```
**What it does:** Changes to project root directory  
**Replace:** `/path/to/tt` with your actual project path

### Install All Python Dependencies
```bash
pip3 install -r requirements.txt --break-system-packages
```
**What it does:** Installs all required Python packages (numpy, matplotlib, torch, flwr, etc.)  
**Expected output:** Shows packages being downloaded and installed  
**Time:** 2-5 minutes  
**If fails:** Try without `--break-system-packages` or use virtual environment

### Install Individual Packages (if requirements.txt fails)
```bash
pip3 install numpy matplotlib torch flwr sumolib traci --break-system-packages
```
**What it does:** Installs core packages individually  
**Use when:** requirements.txt installation fails

### Install with Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**What it does:** Creates isolated Python environment, then installs packages  
**Why use:** Avoids system package conflicts  
**To deactivate:** `deactivate`

---

## 3. Verification Commands

### Test All Imports
```bash
python3 scripts/test_imports.py
```
**What it does:** Verifies all Python modules can be imported  
**Expected output:** `✓ All imports successful!`  
**If fails:** Install missing packages shown in error

### Check Project Files Exist
```bash
ls scripts/
ls baselines/
ls sumo_configs/
```
**What it does:** Lists files in each directory  
**Expected output:** Should see Python scripts and SUMO config files  
**If fails:** Files are missing, check project structure

### Verify SUMO Config Files
```bash
ls sumo_configs/intersection.sumocfg
ls sumo_configs/osm_client1.sumocfg
ls sumo_configs/osm_client2.sumocfg
```
**What it does:** Checks that required SUMO config files exist  
**Expected output:** File paths listed  
**If fails:** Config files missing, check project

### Check Python Package Installation
```bash
pip3 list | grep -E "numpy|matplotlib|torch|flwr"
```
**What it does:** Shows if key packages are installed  
**Expected output:** Lists installed packages with versions  
**If fails:** Packages not installed, run installation commands

---

## 4. Running Comparisons

### Quick Test Run (Recommended First)
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 3 \
    --num-steps 200 \
    --results-dir baseline_comparison
```
**What it does:** Runs all 7 baseline methods with reduced parameters  
**Time:** 10-30 minutes  
**Output:** Creates results JSON and plots  
**Why:** Tests that everything works before full run

### Full Comparison (For Research Paper)
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400 \
    --results-dir baseline_comparison
```
**What it does:** Runs complete comparison with full parameters  
**Time:** 1-3 hours  
**Output:** Complete results with statistical significance  
**Use for:** Final paper results

### Run Comparison Only (Without Visualization)
```bash
python3 scripts/compare_all_baselines.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400 \
    --results-dir baseline_comparison
```
**What it does:** Runs comparison but doesn't generate plots  
**Time:** 1-3 hours  
**Output:** Only JSON results file  
**Use when:** You want to generate plots separately

### Run with GUI (Visual SUMO Simulation)
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --num-runs 3 \
    --num-steps 200 \
    --gui
```
**What it does:** Opens SUMO GUI to watch simulation  
**Time:** Longer (GUI slows down)  
**Use when:** You want to see traffic simulation visually

---

## 5. Viewing Results

### View Text Results in Terminal
```bash
python3 scripts/show_results_text.py \
    --results-file baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json
```
**What it does:** Displays formatted comparison table and ASCII charts  
**Expected output:** Summary table, improvement percentages, ASCII bar charts  
**Replace:** `YYYYMMDD_HHMMSS` with actual timestamp from filename

### Find Latest Results File
```bash
ls -t baseline_comparison/comparison_results_*.json | head -1
```
**What it does:** Lists results files sorted by time, shows most recent  
**Expected output:** Path to latest results file  
**Use with:** Copy output to use in other commands

### Generate Visualization Plots Only
```bash
python3 scripts/visualize_comparison.py \
    --results-file baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json \
    --output-dir comparison_plots
```
**What it does:** Creates PNG plot files from existing results  
**Time:** 10-30 seconds  
**Output:** 3 PNG files in comparison_plots/  
**Use when:** You have results but need to regenerate plots

### View Results JSON File
```bash
cat baseline_comparison/comparison_results_*.json | python3 -m json.tool | less
```
**What it does:** Pretty-prints JSON file for reading  
**Expected output:** Formatted JSON data  
**Exit:** Press `q` to quit

### List Generated Plot Files
```bash
ls -lh comparison_plots/*.png
```
**What it does:** Shows all generated plot files with sizes  
**Expected output:** List of PNG files (baseline_comparison.png, waiting_time_comparison.png, queue_length_comparison.png)

---

## 6. Troubleshooting Commands

### Add SUMO to PATH (Linux/Mac)
```bash
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin
```
**What it does:** Adds SUMO to current session PATH  
**Replace:** `/path/to/sumo` with actual SUMO installation path  
**Permanent:** Add to `~/.bashrc` or `~/.zshrc`

### Add SUMO to PATH Permanently
```bash
echo 'export SUMO_HOME=/path/to/sumo' >> ~/.bashrc
echo 'export PATH=$PATH:$SUMO_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```
**What it does:** Adds SUMO to PATH permanently  
**Replace:** `/path/to/sumo` with actual path  
**Verify:** Run `sumo --version` after

### Install SUMO Python Tools
```bash
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools
pip3 install sumolib traci --break-system-packages
```
**What it does:** Makes SUMO Python tools available  
**Use when:** Getting "No module named 'traci'" error

### Check SUMO Installation Location
```bash
find /usr -name "sumo" 2>/dev/null
find /opt -name "sumo" 2>/dev/null
find ~ -name "sumo" 2>/dev/null
```
**What it does:** Searches for SUMO installation  
**Use when:** Don't know where SUMO is installed

### Verify Python Can Import SUMO
```bash
python3 -c "import traci; print('SUMO Python tools OK')"
```
**What it does:** Tests if SUMO Python interface works  
**Expected output:** `SUMO Python tools OK`  
**If fails:** SUMO Python tools not installed or not in PYTHONPATH

### Check File Permissions
```bash
chmod +x scripts/*.py
```
**What it does:** Makes Python scripts executable  
**Use when:** Getting "Permission denied" errors

### Reinstall Specific Package
```bash
pip3 install --upgrade --force-reinstall <package-name> --break-system-packages
```
**What it does:** Reinstalls a specific package  
**Replace:** `<package-name>` with actual package (e.g., matplotlib, numpy)  
**Use when:** Package seems corrupted

### Check Python Version Compatibility
```bash
python3 -c "import sys; print(sys.version_info >= (3, 8))"
```
**What it does:** Checks if Python version is 3.8+  
**Expected output:** `True`  
**If False:** Need to upgrade Python

---

## 7. Complete Workflow Commands

### Complete Setup and Run (Copy-Paste Ready)

```bash
# Step 1: Navigate to project
cd /path/to/tt

# Step 2: Check prerequisites
python3 --version
sumo --version

# Step 3: Install dependencies
pip3 install -r requirements.txt --break-system-packages

# Step 4: Verify installation
python3 scripts/test_imports.py

# Step 5: Quick test run
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 3 \
    --num-steps 200

# Step 6: View results
LATEST_RESULTS=$(ls -t baseline_comparison/comparison_results_*.json | head -1)
python3 scripts/show_results_text.py --results-file "$LATEST_RESULTS"

# Step 7: Full comparison (if test worked)
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400
```

---

## 8. One-Line Commands (Quick Reference)

### Setup
```bash
pip3 install -r requirements.txt --break-system-packages && python3 scripts/test_imports.py
```

### Quick Test
```bash
python3 scripts/run_and_visualize.py --sumo-config sumo_configs/intersection.sumocfg --num-runs 3 --num-steps 200
```

### View Latest Results
```bash
python3 scripts/show_results_text.py --results-file $(ls -t baseline_comparison/comparison_results_*.json | head -1)
```

### Generate Plots from Latest Results
```bash
python3 scripts/visualize_comparison.py --results-file $(ls -t baseline_comparison/comparison_results_*.json | head -1) --output-dir comparison_plots
```

---

## 9. Command Parameters Explained

### Comparison Script Parameters

| Parameter | What It Does | Example Values |
|-----------|--------------|----------------|
| `--sumo-config` | Single intersection config for classical baselines | `sumo_configs/intersection.sumocfg` |
| `--sumo-configs-multi` | Multiple configs for RL methods | `sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg` |
| `--num-runs` | Number of independent runs per method | `3` (test), `10` (paper) |
| `--num-steps` | Simulation steps per run | `200` (test), `400` (paper) |
| `--results-dir` | Where to save results | `baseline_comparison` |
| `--gui` | Enable SUMO GUI (slower) | (flag, no value) |

### Visualization Script Parameters

| Parameter | What It Does | Example Values |
|-----------|--------------|----------------|
| `--results-file` | Path to results JSON file | `baseline_comparison/comparison_results_20241210_123456.json` |
| `--output-dir` | Where to save plots | `comparison_plots` |

---

## 10. Expected File Structure After Running

```
tt/
├── baseline_comparison/
│   └── comparison_results_YYYYMMDD_HHMMSS.json  ← Results file
├── comparison_plots/
│   ├── baseline_comparison.png                   ← Main comparison plot
│   ├── waiting_time_comparison.png               ← Waiting time chart
│   └── queue_length_comparison.png              ← Queue length chart
├── scripts/
│   ├── compare_all_baselines.py                  ← Main script
│   ├── visualize_comparison.py                  ← Plot generator
│   └── show_results_text.py                     ← Text viewer
└── ...
```

---

## 11. Quick Command Cheat Sheet

```bash
# SETUP
pip3 install -r requirements.txt --break-system-packages
python3 scripts/test_imports.py

# QUICK TEST
python3 scripts/run_and_visualize.py --sumo-config sumo_configs/intersection.sumocfg --num-runs 3 --num-steps 200

# FULL RUN
python3 scripts/run_and_visualize.py --sumo-config sumo_configs/intersection.sumocfg --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg --num-runs 10 --num-steps 400

# VIEW RESULTS
python3 scripts/show_results_text.py --results-file baseline_comparison/comparison_results_*.json

# GENERATE PLOTS
python3 scripts/visualize_comparison.py --results-file baseline_comparison/comparison_results_*.json --output-dir comparison_plots
```

---

## ✅ Verification Checklist

After running commands, verify:

- [ ] Results JSON file exists: `ls baseline_comparison/comparison_results_*.json`
- [ ] Plot files exist: `ls comparison_plots/*.png` (should see 3 files)
- [ ] Results show all 7 methods: Check JSON or text output
- [ ] Federated FL shows best performance: Check summary statistics

---

**All commands explained!** Copy and paste these commands in order to set up and run the comparison system.

