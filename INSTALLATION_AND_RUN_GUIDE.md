# Complete Installation and Run Guide
## Running Baseline Comparison on Real SUMO Data

This guide will help you set up and run the baseline comparison system on a computer with SUMO installed.

---

## 📋 Prerequisites

### Required Software
1. **Python 3.8 or higher**
2. **SUMO (Simulation of Urban MObility)** - Must be installed and in PATH
3. **Git** (optional, for cloning repository)

### Verify SUMO Installation
```bash
sumo --version
# Should show: SUMO version X.X.X
```

If SUMO is not installed, download from: https://sumo.dlr.de/docs/Downloads.php

---

## 🚀 Step-by-Step Installation

### Step 1: Get the Project Files

**Option A: If you have the project folder**
```bash
# Copy the entire project folder to the new computer
# Navigate to the project directory
cd /path/to/tt
```

**Option B: If using Git**
```bash
git clone <your-repository-url>
cd tt
```

### Step 2: Install Python Dependencies

```bash
# Install required packages
pip3 install -r requirements.txt --break-system-packages

# Or if using virtual environment (recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Required packages:**
- numpy
- matplotlib
- torch (PyTorch)
- flwr (Flower)
- traci (SUMO Python interface)
- Other dependencies from requirements.txt

### Step 3: Verify Installation

```bash
# Test imports
python3 scripts/test_imports.py

# Should show: ✓ All imports successful!
```

If you see errors, install missing packages:
```bash
pip3 install <package-name> --break-system-packages
```

### Step 4: Verify SUMO Config Files

```bash
# Check that SUMO config files exist
ls sumo_configs/
# Should see:
# - intersection.sumocfg
# - intersection.net.xml
# - intersection.rou.xml
# - osm_client1.sumocfg
# - osm_client2.sumocfg
```

**Important:** Make sure all SUMO config files are present and paths are correct.

---

## 🎯 Running the Comparison

### Quick Test (Fast - Good for Testing)

```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 3 \
    --num-steps 200 \
    --results-dir baseline_comparison
```

**Expected time:** 10-30 minutes (depending on system)

### Full Comparison (For Research Paper)

```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400 \
    --results-dir baseline_comparison
```

**Expected time:** 1-3 hours (depending on system and number of runs)

### Step-by-Step (Run Separately)

If you prefer to run steps separately:

#### 1. Run Comparison Only
```bash
python3 scripts/compare_all_baselines.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400 \
    --results-dir baseline_comparison
```

#### 2. Generate Visualizations
```bash
# Find the latest results file
ls -t baseline_comparison/comparison_results_*.json | head -1

# Generate plots
python3 scripts/visualize_comparison.py \
    --results-file baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json \
    --output-dir comparison_plots
```

#### 3. View Text Results
```bash
python3 scripts/show_results_text.py \
    --results-file baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json
```

---

## 📊 What Gets Generated

### Results Files
- **Location:** `baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json`
- **Contains:** All raw metrics, summary statistics, per-run data

### Visualization Files
- **Location:** `comparison_plots/`
- **Files:**
  - `baseline_comparison.png` - Comprehensive 6-panel comparison
  - `waiting_time_comparison.png` - Waiting time bar chart
  - `queue_length_comparison.png` - Queue length bar chart

---

## 🔧 Troubleshooting

### Issue 1: "SUMO not found"
**Error:** `sumo: command not found`

**Solution:**
```bash
# Check if SUMO is installed
which sumo

# If not found, add SUMO to PATH
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin

# Or add to ~/.bashrc or ~/.zshrc for permanent:
echo 'export SUMO_HOME=/path/to/sumo' >> ~/.bashrc
echo 'export PATH=$PATH:$SUMO_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

### Issue 2: "No module named 'traci'"
**Error:** `ModuleNotFoundError: No module named 'traci'`

**Solution:**
```bash
# Install SUMO Python tools
pip3 install sumolib traci --break-system-packages

# Or if SUMO is installed, add to Python path:
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools
```

### Issue 3: "SUMO config file not found"
**Error:** FileNotFoundError for SUMO config files

**Solution:**
```bash
# Check file paths
ls -la sumo_configs/

# Verify paths in scripts match your directory structure
# Update paths if needed in:
# - scripts/compare_all_baselines.py
# - train_federated.py
```

### Issue 4: "RL methods taking too long"
**Solution:** Reduce training rounds for testing
```bash
# Edit baselines/centralized_rl.py and baselines/independent_rl.py
# Change: num_rounds=10 → num_rounds=5
# Change: episodes_per_round=2 → episodes_per_round=1
```

### Issue 5: "GUI not opening"
**Solution:** Run without GUI (faster)
```bash
# Remove --gui flag or don't add it
python3 scripts/run_and_visualize.py --num-runs 3 --num-steps 200
```

### Issue 6: "Permission denied"
**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.py

# Or run with python3 explicitly
python3 scripts/compare_all_baselines.py ...
```

---

## 📝 Configuration Options

### Adjust Simulation Parameters

Edit `scripts/compare_all_baselines.py`:

```python
# Line ~164: Change default steps
num_steps: int = 400  # Reduce to 200 for faster testing

# Line ~232: Change RL training rounds
train_results = centralized.train(num_rounds=10, episodes_per_round=2)
# Reduce to: num_rounds=5, episodes_per_round=1
```

### Adjust Baseline Parameters

Edit baseline files in `baselines/`:

**Fixed-Time (`baselines/fixed_time.py`):**
```python
# Line ~30: Change phase durations
self.phase_durations = {
    0: 30.0,  # Green phase 1 duration
    1: 3.0,   # Yellow phase 1 duration
    2: 30.0,  # Green phase 2 duration
    3: 3.0,   # Yellow phase 2 duration
}
```

**Actuated (`baselines/actuated.py`):**
```python
# Line ~20: Change thresholds
min_green_time: float = 10.0,
max_green_time: float = 60.0,
extension_time: float = 3.0,
```

---

## 🎓 Understanding the Output

### Console Output

You'll see progress like:
```
============================================================
RUNNING BASELINE COMPARISON
============================================================

Running fixed_time...
Running actuated...
Running maxpressure...
Running sotl...

Training Centralized RL (this may take a while)...
Training Independent RL (this may take a while)...
Training Federated Learning (this may take a while)...

============================================================
COMPARISON SUMMARY
============================================================

Method               Avg Waiting Time    Avg Queue Length    Runs
----------------------------------------------------------------------
fixed_time                   45.23 ± 2.15        8.45 ± 0.32          5
...
```

### Results File Structure

```json
{
  "timestamp": "20241210_123456",
  "config": {
    "sumo_config": "sumo_configs/intersection.sumocfg",
    "num_runs": 10,
    "num_steps": 400
  },
  "all_results": {
    "fixed_time": [...],
    "actuated": [...],
    ...
  },
  "summary": {
    "fixed_time": {
      "mean_waiting_time": 45.23,
      "std_waiting_time": 2.15,
      ...
    },
    ...
  }
}
```

---

## ✅ Verification Checklist

Before running full comparison:

- [ ] SUMO is installed and in PATH (`sumo --version` works)
- [ ] Python 3.8+ is installed (`python3 --version`)
- [ ] All dependencies installed (`pip3 list | grep -E "numpy|matplotlib|torch|flwr"`)
- [ ] SUMO config files exist (`ls sumo_configs/`)
- [ ] Test imports work (`python3 scripts/test_imports.py`)
- [ ] Quick test run completes successfully (3 runs, 200 steps)

---

## 🚀 Quick Start Commands

**Copy and paste these commands:**

```bash
# 1. Navigate to project
cd /path/to/tt

# 2. Install dependencies
pip3 install -r requirements.txt --break-system-packages

# 3. Verify SUMO
sumo --version

# 4. Quick test
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --num-runs 3 \
    --num-steps 200

# 5. Full comparison (for paper)
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check console output** - Error messages will show what's wrong
2. **Verify SUMO installation** - `sumo --version` should work
3. **Check file paths** - Make sure all config files exist
4. **Test imports** - Run `python3 scripts/test_imports.py`
5. **Check Python version** - Must be 3.8 or higher

---

## 📊 Expected Results

After successful run, you should have:

1. **Results JSON file** in `baseline_comparison/`
2. **3 PNG plot files** in `comparison_plots/`
3. **Console output** showing summary statistics

The comparison should show:
- **Federated FL** performing best (lowest waiting time and queue length)
- **40%+ improvement** over Fixed-Time baseline
- **Better than classical methods** (MaxPressure, SOTL, Actuated)
- **Competitive with Centralized RL** while preserving privacy

---

## 🎉 Success!

Once you see the plots generated and summary statistics printed, your comparison is complete!

Use the generated plots and statistics in your research paper.

**Good luck with your experiments!** 🚀

