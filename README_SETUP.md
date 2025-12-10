# 🚀 Quick Setup Guide for New Computer

## One-Command Setup (Easiest)

If you have SUMO installed, just run:

```bash
bash QUICK_START.sh
```

This script will:
1. ✅ Check Python installation
2. ✅ Check SUMO installation  
3. ✅ Verify project files
4. ✅ Install dependencies
5. ✅ Test imports
6. ✅ Run comparison

---

## Manual Setup (Step-by-Step)

### 1. Prerequisites Check

```bash
# Check Python (need 3.8+)
python3 --version

# Check SUMO (must be installed)
sumo --version
```

### 2. Install Dependencies

```bash
# From project root directory
pip3 install -r requirements.txt --break-system-packages
```

### 3. Verify Setup

```bash
# Test that everything works
python3 scripts/test_imports.py
```

### 4. Run Comparison

**Quick test (10-30 min):**
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --num-runs 3 \
    --num-steps 200
```

**Full comparison (1-3 hours):**
```bash
python3 scripts/run_and_visualize.py \
    --sumo-config sumo_configs/intersection.sumocfg \
    --sumo-configs-multi sumo_configs/osm_client1.sumocfg sumo_configs/osm_client2.sumocfg \
    --num-runs 10 \
    --num-steps 400
```

---

## 📁 What You Need

### Required Files
- `scripts/compare_all_baselines.py` - Main comparison script
- `scripts/visualize_comparison.py` - Plot generator
- `scripts/run_and_visualize.py` - Complete pipeline
- `baselines/` - All baseline implementations
- `sumo_configs/` - SUMO configuration files
- `requirements.txt` - Python dependencies

### Required Software
- Python 3.8+
- SUMO (installed and in PATH)
- pip3 (Python package manager)

---

## 🔧 Common Issues

### SUMO Not Found
```bash
# Add SUMO to PATH
export SUMO_HOME=/path/to/sumo
export PATH=$PATH:$SUMO_HOME/bin

# Verify
sumo --version
```

### Python Packages Not Installing
```bash
# Use --break-system-packages flag
pip3 install <package> --break-system-packages

# Or use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Config Files Not Found
```bash
# Check files exist
ls sumo_configs/

# Verify paths in scripts match your setup
```

---

## 📊 Expected Output

After successful run:

1. **Results JSON:** `baseline_comparison/comparison_results_*.json`
2. **Plots:** `comparison_plots/*.png` (3 files)
3. **Console:** Summary statistics table

---

## 📖 Full Documentation

See `INSTALLATION_AND_RUN_GUIDE.md` for complete detailed instructions.

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] SUMO installed and in PATH
- [ ] Dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Test imports work (`python3 scripts/test_imports.py`)
- [ ] SUMO config files present
- [ ] Quick test completes successfully

Once all checked, you're ready to run the full comparison! 🎉

