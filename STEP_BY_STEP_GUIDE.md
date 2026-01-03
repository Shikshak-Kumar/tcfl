# Step-by-Step Guide: Running Baseline Comparison
## Simple Instructions - Follow These Steps

---

## 🎯 What You'll Do

You'll run a comparison of 5 traffic control methods (4 baselines + your federated method) and generate graphs showing which one performs best.

**Time needed:** 
- Quick test: 10-30 minutes
- Full comparison: 1-3 hours

---

## ✅ STEP 1: Check Your Computer Has Everything

### 1.1 Check Python
Open terminal and type:
```bash
python3 --version
```
**What you should see:** `Python 3.8.x` or higher  
**If you see an error:** Install Python from python.org

### 1.2 Check SUMO
Type:
```bash
sumo --version
```
**What you should see:** `SUMO version X.X.X`  
**If you see "command not found":** 
- Download SUMO from: https://sumo.dlr.de/docs/Downloads.php
- Install it
- Add to PATH (see troubleshooting below)

**✅ If both commands work, go to STEP 2**

---

## ✅ STEP 2: Copy Project to Your Computer

### 2.1 Copy the Project Folder
- Copy the entire `tt` folder to your computer
- Remember where you put it (e.g., `/Users/yourname/Documents/tt` or `C:\Users\yourname\Documents\tt`)

### 2.2 Open Terminal in Project Folder
```bash
cd /path/to/tt
```
**Replace** `/path/to/tt` with your actual folder path

**Example:**
```bash
cd /Users/john/Documents/tt
```

**✅ If you're in the project folder, go to STEP 3**

---

## ✅ STEP 3: Install Python Packages

### 3.1 Install All Dependencies
Type this command:
```bash
pip3 install -r requirements.txt --break-system-packages
```

**What it does:** Installs numpy, matplotlib, torch, flwr, and other required packages  
**Time:** 2-5 minutes  
**What you'll see:** Packages downloading and installing

### 3.2 If That Fails, Try This:
```bash
pip3 install numpy matplotlib torch flwr --break-system-packages
```

**✅ When installation finishes, go to STEP 4**

---

## ✅ STEP 4: Verify Everything Works

### 4.1 Test Imports
Type:
```bash
python3 scripts/test_imports.py
```

**What you should see:** `✓ All imports successful!`  
**If you see errors:** Install the missing packages shown in the error

### 4.2 Check Files Exist
Type:
```bash
ls scripts/
ls baselines/
ls sumo_configs/
```

**What you should see:** Lists of Python files and SUMO config files  
**If files are missing:** Check that you copied the entire project folder

**✅ If everything checks out, go to STEP 5**

---

## ✅ STEP 5: Run Quick Test (Recommended First)

### 5.1 Run Quick Test
Copy and paste this entire command:
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 3 \
    --num-steps 200 \
    --results-dir baseline_comparison
```

**What it does:** Runs all 5 methods (4 baselines + your federated method) with quick settings  
**Time:** 10-30 minutes  
**What you'll see:** Progress messages as each method runs

### 5.2 Wait for Completion
You'll see messages like:
```
Running fixed_time...
Running maxpressure...
Training Centralized RL...
Training Independent RL...
Training Federated Learning...
============================================================
COMPARISON SUMMARY
============================================================
```

**✅ When it finishes, go to STEP 6**

---

## ✅ STEP 6: View Your Results

### 6.1 View Text Results
Type:
```bash
python3 scripts/show_results_text.py \
    --results-file baseline_comparison/comparison_results_*.json
```

**What you'll see:** 
- Comparison table showing all methods
- Improvement percentages
- ASCII bar charts
- Summary statistics

### 6.2 Check Plot Files Were Created
Type:
```bash
ls comparison_plots/*.png
```

**What you should see:** 3 PNG files:
- `baseline_comparison.png` (main comparison)
- `waiting_time_comparison.png`
- `queue_length_comparison.png`

**✅ If you see results and plots, go to STEP 7**

---

## ✅ STEP 7: Run Full Comparison (For Your Paper)

### 7.1 Run Full Comparison
Copy and paste this entire command:
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
**Why:** More runs = better statistics for your paper

### 7.2 Wait for Completion
This will take longer. You can leave it running.

**✅ When finished, go to STEP 8**

---

## ✅ STEP 8: Get Final Results

### 8.1 View Final Results
Type:
```bash
python3 scripts/show_results_text.py \
    --results-file baseline_comparison/comparison_results_*.json
```

### 8.2 Open Plot Files
- Navigate to `comparison_plots/` folder
- Open `baseline_comparison.png` in any image viewer
- This is your main comparison figure for your paper!

**✅ You're done! Use these results in your paper.**

---

## 🔧 Troubleshooting

### Problem: "SUMO not found"
**Solution:**
```bash
# Find where SUMO is installed
find /usr -name "sumo" 2>/dev/null
find /opt -name "sumo" 2>/dev/null

# Add to PATH (replace /path/to/sumo with actual path)
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin

# Test
sumo --version
```

### Problem: "No module named 'traci'"
**Solution:**
```bash
# Add SUMO tools to Python path
export PYTHONPATH=$PYTHONPATH:$SUMO_HOME/tools

# Install SUMO Python tools
pip3 install sumolib traci --break-system-packages
```

### Problem: "Permission denied"
**Solution:**
```bash
chmod +x scripts/*.py
```

### Problem: "File not found"
**Solution:**
- Make sure you're in the project root directory: `cd /path/to/tt`
- Check files exist: `ls sumo_configs/`
- Verify paths in commands match your folder structure

---

## What You'll Get

After completing all steps:

1. **Results JSON file:** `baseline_comparison/comparison_results_YYYYMMDD_HHMMSS.json`
   - Contains all raw data and statistics

2. **3 Plot Files:** `comparison_plots/*.png`
   - Ready to include in your research paper

3. **Summary Statistics:**
   - Comparison table showing all 7 methods
   - Improvement percentages
   - Statistical significance (mean ± std)

---

## Quick Summary

**Do these 8 steps:**
1. ✅ Check Python and SUMO installed
2. ✅ Copy project folder to computer
3. ✅ Install Python packages (`pip3 install -r requirements.txt`)
4. ✅ Verify everything works (`python3 scripts/test_imports.py`)
5. ✅ Run quick test (10-30 min)
6. ✅ View test results
7. ✅ Run full comparison (1-3 hours)
8. ✅ Get final results and plots

**That's it!** Your comparison is complete and ready for your paper.

---

## Tips

- **Start with quick test** (STEP 5) to make sure everything works
- **Run full comparison** (STEP 7) when you're ready for paper results
- **Save the results file** - you'll need it for your paper
- **The plots are publication-ready** - use them directly in your paper

---

## Need Help?

If you get stuck:
1. Check the error message - it usually tells you what's wrong
2. See `ALL_COMMANDS.md` for detailed command explanations
3. See `INSTALLATION_AND_RUN_GUIDE.md` for more troubleshooting



