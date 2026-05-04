import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModelExporter:
    """Export PyTorch models for deployment (weights + optional TorchScript)."""

    @staticmethod
    def export(
        model: nn.Module,
        metadata: Dict[str, Any],
        base_path: str,
        model_name: str = "model",
    ):
        os.makedirs(base_path, exist_ok=True)
        weights_path = os.path.join(base_path, f"{model_name}_weights.pt")
        torch.save(model.state_dict(), weights_path)

        model.eval()
        try:
            scripted_model = torch.jit.script(model)
            scripted_path = os.path.join(base_path, f"{model_name}.pt")
            scripted_model.save(scripted_path)
            metadata["export_method"] = "torch_jit_script"
        except Exception as e:
            print(f"Warning: Scripting failed, weights only: {e}")
            metadata["export_status"] = "weights_only"

        metadata_path = os.path.join(base_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Successfully exported model to {base_path}")


def get_deployment_metadata(algo_name: str, agent: Any) -> Dict[str, Any]:
    metadata = {
        "algorithm": algo_name,
        "version": "1.0.0",
        "state_size": getattr(agent, "state_size", 12),
        "action_size": getattr(agent, "action_size", 4),
        "device": "cpu",
        "preprocessing": {"normalization": "none", "clipping": "none"},
    }
    if hasattr(agent, "time_steps"):
        metadata["time_steps"] = agent.time_steps
    if hasattr(agent, "hidden_dims"):
        metadata["hidden_dims"] = agent.hidden_dims
    return metadata
