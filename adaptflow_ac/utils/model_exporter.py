import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ModelExporter:
    """
    Utility to export PyTorch models in a deployment-ready format.
    Saves weights (state_dict), optimized TorchScript, and metadata.
    """
    
    @staticmethod
    def export(model: nn.Module, 
               metadata: Dict[str, Any], 
               base_path: str,
               model_name: str = "model"):
        """
        Export a model and its metadata.
        
        Args:
            model: The PyTorch nn.Module to export.
            metadata: Dictionary containing architecture, version, and preprocessing info.
            base_path: Directory where artifacts will be saved.
            model_name: Base name for the model files.
        """
        os.makedirs(base_path, exist_ok=True)
        
        # 1. Save Raw Weights (state_dict)
        weights_path = os.path.join(base_path, f"{model_name}_weights.pt")
        torch.save(model.state_dict(), weights_path)
        
        # 2. Save Optimized TorchScript
        # We use scripting because AdaptFlow has some conditional logic (GAT) 
        # but for simple MLPs tracing is also fine. Scripting is generally more robust for nn.Modules.
        model.eval()
        try:
            # Try scripting first
            scripted_model = torch.jit.script(model)
            scripted_path = os.path.join(base_path, f"{model_name}.pt")
            scripted_model.save(scripted_path)
            metadata["export_method"] = "torch_jit_script"
        except Exception as e:
            print(f"Warning: Scripting failed, falling back to tracing: {e}")
            # Fallback to tracing (requires a dummy input)
            state_size = metadata.get("state_size", 12)
            time_steps = metadata.get("time_steps", 1)
            
            if time_steps > 1:
                # [batch, time_steps, num_nodes, state_size] - for AdaptFlow
                # However, AdaptFlowDQN.forward expects (x_seq, adj)
                # This makes tracing complex. We'll stick to scripting or raw state_dict for GAT.
                print("Tracing skipped for complex GAT model. Only weights and metadata saved.")
                metadata["export_status"] = "weights_only"
            else:
                dummy_input = torch.randn(1, state_size)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_path = os.path.join(base_path, f"{model_name}.pt")
                traced_model.save(traced_path)
                metadata["export_method"] = "torch_jit_trace"

        # 3. Save Metadata
        metadata_path = os.path.join(base_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Successfully exported model to {base_path}")

def get_deployment_metadata(algo_name: str, agent: Any) -> Dict[str, Any]:
    """Extract metadata from an agent for deployment."""
    metadata = {
        "algorithm": algo_name,
        "version": "1.0.0",
        "state_size": getattr(agent, "state_size", 12),
        "action_size": getattr(agent, "action_size", 4),
        "device": "cpu", # Production usually runs on CPU
        "preprocessing": {
            "normalization": "none",
            "clipping": "none"
        }
    }
    
    if hasattr(agent, "time_steps"):
        metadata["time_steps"] = agent.time_steps
        
    if hasattr(agent, "hidden_dims"):
        metadata["hidden_dims"] = agent.hidden_dims
        
    return metadata
