import os
import torch
import numpy as np
from utils.model_exporter import ModelExporter, get_deployment_metadata
from agents.adaptflow_agent import AdaptFlowAgent
from agents.fedflow_agent import FedFlowAgent
from agents.dqn_agent import DQNAgent

def save_all():
    print("Starting Global Model Export for Deployment...")
    
    # Base directory for all saved models
    base_save_dir = "saved_models"
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Mapping of algorithm to their results directory and weights filename
    algos = {
        "adaptflow": {
            "results_dir": "results_adaptflow",
            "weights": "adaptflow_global_mock.pt",
            "agent_class": AdaptFlowAgent,
            "params": {"state_size": 12, "action_size": 4}
        },
        "fedflow": {
            "results_dir": "results_fedflow",
            "weights": "fedflow_global_mock.pt",
            "agent_class": FedFlowAgent,
            "params": {"state_size": 12, "action_size": 4}
        },
        "fedcm": {
            "results_dir": "results_fedcm", # Check the actual filename
            "weights": "fedcm_global_mock.pt",
            "agent_class": DQNAgent,
            "params": {"state_size": 12, "action_size": 4, "hidden_dims": [128, 128, 64]}
        },
        "fedkd": {
            "results_dir": "results_fedkd_sumo",
            "weights": "fedkd_global_mock.pt",
            "agent_class": DQNAgent,
            "params": {"state_size": 12, "action_size": 4, "hidden_dims": [128, 128, 64]}
        },
        "federated": {
            "results_dir": "simulation_results", # Standard federated often goes here
            "weights": "global_model.pt",
            "agent_class": DQNAgent,
            "params": {"state_size": 12, "action_size": 4, "hidden_dims": [128, 128, 64]}
        }
    }
    
    for name, info in algos.items():
        print(f"\nProcessing {name}...")
        
        # 1. Instantiate Agent
        agent = info["agent_class"](**info["params"])
        
        # 2. Load Weights
        weights_path = os.path.join(info["results_dir"], info["weights"])
        if os.path.exists(weights_path):
            try:
                agent.load_model(weights_path)
                print(f"  Loaded weights from {weights_path}")
            except Exception as e:
                print(f"  Warning: Could not load weights for {name}: {e}")
        else:
            print(f"  Warning: No weights found at {weights_path}. Exporting initial/random model.")
            
        # 3. Export
        metadata = get_deployment_metadata(name, agent)
        metadata["source_weights"] = weights_path if os.path.exists(weights_path) else "random"
        
        algo_save_path = os.path.join(base_save_dir, name)
        ModelExporter.export(agent.policy_net, metadata, algo_save_path)

    print("\nAll models exported successfully to /saved_models/")

if __name__ == "__main__":
    save_all()
