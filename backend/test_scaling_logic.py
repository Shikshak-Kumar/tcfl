import subprocess
import re

def test_auto_scaling(script_path, nodes):
    print(f"Testing {script_path} with {nodes} nodes...")
    try:
        # Run with --rounds 0 (if supported) or just check the printed [AUTO] line
        # We'll use a trick: run wait 1 second and then kill it, or use a flag if it exists
        # Actually, let's just check the help or run with 1 round
        cmd = ["python3", script_path, "--nodes", str(nodes), "--rounds", "1"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give it a few seconds to print the [AUTO] line then kill it
        output = ""
        for _ in range(20):
            line = process.stdout.readline()
            if line:
                output += line
                if "[AUTO]" in line:
                    break
            else:
                break
        process.terminate()
        
        match = re.search(r"Selected (\d+) dummy servers", output)
        if match:
            clusters = int(match.group(1))
            expected = max(1, nodes // 4)
            print(f"  Nodes: {nodes}, Auto Clusters: {clusters}, Expected: {expected}")
            return clusters == expected
        else:
            print("  [AUTO] line not found in output!")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    scripts = ["train_fedflow.py", "train_adaptflow.py"]
    node_counts = [2, 6, 12]
    
    all_passed = True
    for script in scripts:
        for n in node_counts:
            if not test_auto_scaling(script, n):
                all_passed = False
    
    if all_passed:
        print("\n✅ ALL AUTO-SCALING TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
