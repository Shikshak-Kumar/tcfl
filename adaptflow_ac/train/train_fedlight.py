"""
FedLight Training Script.
Standard Federated Reinforcement Learning for Traffic Signal Control.

Usage:
  python train_fedlight.py --rounds 10 --nodes 6
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

from env.mock_env import MockEnv as MockTrafficEnvironment
from agents.fedlight import FedLightAgent
from utils.logger import logger
from utils.sumo_scenario import (
    distinct_results_dir,
    effective_sumo_headless,
    effective_sumo_scenario,
    effective_training_gui,
    get_sumo_config_paths,
    scenario_label_for_log,
)

class FedLightTrainer:
    def __init__(
        self,
        num_nodes: int = 6,
        gui: bool = False,
        results_dir: str = "results_fedlight",
        sumo_scenario: Optional[str] = None,
        sumo_headless: bool = False,
        steps: int = 200,
    ):
        self.num_nodes = num_nodes
        self.sumo_headless = sumo_headless
        self.gui = effective_training_gui(sumo_scenario, False, gui, sumo_headless)
        self.results_dir = results_dir
        self.sumo_scenario = sumo_scenario
        self.steps = steps
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_round_results = []

        # 1. Environments
        self.sumo_configs = get_sumo_config_paths(effective_sumo_scenario(sumo_scenario))
        self.envs: Dict[str, object] = {}
        self._setup_environments()

        # 2. Local Agents (detect state_dim from environment)
        sample_env = self.envs["node_0"]
        sample_state = sample_env.reset()
        if isinstance(sample_state, dict):
            # If dict, extract first node's features
            first_tid = list(sample_state.keys())[0]
            state_dim = len(sample_state[first_tid]) if not isinstance(sample_state[first_tid], dict) else len(sample_state[first_tid].get("wave", [])) * 2
        else:
            state_dim = sample_state.shape[-1] if len(sample_state.shape) > 1 else len(sample_state)
            
        self.agents: Dict[str, FedLightAgent] = {}
        for i in range(num_nodes):
            self.agents[f"node_{i}"] = FedLightAgent(node_id=i, state_dim=state_dim, action_dim=4)

    def _setup_environments(self):
        if self.gui or self.sumo_headless:
            from env.sumo_env import SumoEnv as SUMOTrafficEnvironment
            for i in range(self.num_nodes):
                config = self.sumo_configs[i % len(self.sumo_configs)]
                self.envs[f"node_{i}"] = SUMOTrafficEnvironment(config, gui=self.gui, max_steps=self.steps)
        else:
            for i in range(self.num_nodes):
                self.envs[f"node_{i}"] = MockTrafficEnvironment(num_intersections=1) # Per node
                
    def run_round(self, round_idx: int):
        logger.header(f"FEDLIGHT ROUND {round_idx}")
        node_metrics = {}
        node_losses = {}

        # 1. Local Training
        for nid, agent in self.agents.items():
            env = self.envs[nid]
            state = env.reset()
            total_reward = 0

            for _ in range(self.steps):
                # FedLight usually takes single state
                if hasattr(env, 'get_states'):
                    s = env.get_states()
                else:
                    s = state
                
                # FedLightAgent.get_action expects (N, D)
                action_list = agent.get_action(s)
                action = action_list[0] if isinstance(action_list, list) else action_list
                
                next_state, reward, done, info = env.step([action] if not isinstance(action, list) else action)
                
                if hasattr(env, 'get_states'):
                    ns = env.get_states()
                else:
                    ns = next_state
                
                agent.remember(s, action, reward, ns, done)
                state = next_state
                # Handle rewards (could be list or dict from MockEnv)
                if isinstance(reward, dict):
                    r_val = np.mean(list(reward.values()))
                else:
                    r_val = np.mean(reward)
                total_reward += r_val
                if done: break

            loss = agent.train()
            metrics = env.get_metrics()
            metrics["total_reward"] = total_reward
            node_metrics[nid] = metrics
            node_losses[nid] = loss
            
            print(f"    {nid}: Reward={total_reward:.1f}, AvgWait={metrics.get('avg_waiting_time_per_vehicle', 0):.2f}s, Loss={loss:.4f}")

        # 2. Federated Aggregation (FedAvg)
        print(f"\n  [Aggregation] Standard FedAvg across {self.num_nodes} nodes")
        all_weights = [agent.get_weights() for agent in self.agents.values()]
        
        # Simple averaging
        global_weights = {}
        for key in all_weights[0].keys():
            global_weights[key] = torch.stack([w[key] for w in all_weights]).mean(dim=0)
            
        for agent in self.agents.values():
            agent.set_weights(global_weights)

        # 3. Logging
        round_results = {"round": round_idx, "nodes": {}}
        table_rows = []
        for nid in sorted(self.agents.keys()):
            m = node_metrics[nid]
            table_rows.append([nid, f"{m.get('total_reward',0):.1f}", f"{m.get('avg_waiting_time_per_vehicle',0):.2f}s", f"{node_losses.get(nid,0):.4f}"])
            round_results["nodes"][nid] = m
            
        logger.table(["Node", "Reward", "Wait Time", "Loss"], table_rows)
        self.all_round_results.append(round_results)

    def train(self, num_rounds: int = 10):
        for r in range(1, num_rounds + 1):
            self.run_round(r)
        
        global_model_path = os.path.join(self.results_dir, "fedlight_global.pt")
        self.agents["node_0"].save_model(global_model_path)
        logger.success(f"FedLight Training Complete. Global model: {global_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--sumo-headless", action="store_true")
    parser.add_argument("--sumo-scenario", type=str, default=None)
    args = parser.parse_args()

    results_dir = distinct_results_dir("results_fedlight", "results_fedlight", args.sumo_scenario)
    trainer = FedLightTrainer(num_nodes=args.nodes, gui=args.gui, results_dir=results_dir, sumo_scenario=args.sumo_scenario, sumo_headless=args.sumo_headless, steps=args.steps)
    trainer.train(num_rounds=args.rounds)
