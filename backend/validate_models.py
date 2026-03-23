import os
import torch
import json
import numpy as np

def validate_all():
    print("Starting Model Validation for Production Readiness...")
    base_dir = "saved_models"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return
    
    algos = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for algo in algos:
        print(f"\nValidating {algo}...")
        algo_path = os.path.join(base_dir, algo)
        metadata_path = os.path.join(algo_path, "metadata.json")
        model_path = os.path.join(algo_path, "model.pt")
        
        # 1. Check metadata
        if not os.path.exists(metadata_path):
            print(f"  [FAILED] Missing metadata.json")
            continue
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        state_size = metadata.get("state_size", 12)
        action_size = metadata.get("action_size", 4)
        time_steps = metadata.get("time_steps", 1)
        
        # 2. Load model (TorchScript)
        if not os.path.exists(model_path):
            print(f"  [WARNING] model.pt missing (weights-only export?)")
            continue
            
        try:
            model = torch.jit.load(model_path)
            model.eval()
            print(f"  Loaded TorchScript model successfully.")
            
            # 3. Test Inference
            with torch.no_grad():
                if algo == "adaptflow":
                    # AdaptFlowDQN expects (x_seq, adj)
                    # dummy x_seq: [batch, time_steps, num_nodes, state_size]
                    # dummy adj: [batch, num_nodes, num_nodes]
                    dummy_x = torch.randn(1, time_steps, 2, state_size) # 2 nodes dummy
                    dummy_adj = torch.ones(1, 2, 2)
                    output = model(dummy_x, dummy_adj)
                else:
                    # Generic MLP
                    dummy_input = torch.randn(1, state_size)
                    output = model(dummy_input)
                
                print(f"  Inference successful. Output shape: {output.shape}")
                if output.shape[-1] != action_size:
                    print(f"  [FAILED] Output shape mismatch: expected {action_size}, got {output.shape[-1]}")
                else:
                    print(f"  [PASSED] Consistency check complete.")
                    
        except Exception as e:
            print(f"  [FAILED] Error during loading/inference: {e}")

if __name__ == "__main__":
    validate_all()
