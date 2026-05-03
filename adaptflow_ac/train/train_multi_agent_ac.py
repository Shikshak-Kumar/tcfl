import os
import sys
import torch
import numpy as np
import traci
from env.multi_agent_ac_env import MultiAgentACEnv
from agents.multi_agent_ac_agent import MultiAgentACAgent

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

class MA2CTrainer:
    def __init__(self, sumocfg_path, num_agents=4, batch_size=120, episodes=10):
        self.sumocfg_path = sumocfg_path
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.episodes = episodes
        
        # Initialize env
        self.env = MultiAgentACEnv(sumocfg_path)
        self.env.initialize()
        
        # Initialize agents (fingerprint dim = 4 neighbors * 4 actions = 16)
        self.agents = [MultiAgentACAgent(i, fp_dim=16) for i in range(num_agents)]
        self.tl_ids = self.env.tls_ids[:num_agents]

    def _get_neighbor_fingerprints(self, agent_idx):
        """Get π_{t-1} of neighbors."""
        fps = []
        # Heuristic neighbors for chain/grid
        neighbors = []
        if agent_idx > 0: neighbors.append(agent_idx - 1)
        if agent_idx < self.num_agents - 1: neighbors.append(agent_idx + 1)
        
        for n in neighbors:
            fps.append(self.agents[n].fingerprint)
            
        # Pad to 4 neighbors * 4 actions = 16
        while len(fps) < 4:
            fps.append(np.zeros(4))
        return fps

    def train(self):
        print("Starting MA2C Training (Chu et al. 2019 Reproduction)")
        
        for ep in range(self.episodes):
            traci.start(["sumo", "-c", self.sumocfg_path, "--no-step-log", "true", "--no-warnings", "true"])
            
            for agent in self.agents:
                agent.reset_hidden()
                
            states = [self.env.get_local_state(tid) for tid in self.tl_ids]
            episode_reward = 0
            
            for step in range(500):
                # 1. Collect neighbor fingerprints and choose actions
                actions = []
                probs_list = []
                values_list = []
                fp_list = [] # Store fingerprints used for this step's update
                
                for i in range(self.num_agents):
                    fps = self._get_neighbor_fingerprints(i)
                    fp_list.append(np.concatenate(fps))
                    act, prob, val = self.agents[i].get_action(states[i]["wave"], states[i]["wait"], fps)
                    actions.append(act)
                    probs_list.append(prob)
                    values_list.append(val)
                
                # 2. Execute actions
                for i, tid in enumerate(self.tl_ids):
                    traci.trafficlight.setPhase(tid, actions[i])
                traci.simulationStep()
                
                # 3. Observe next states and raw rewards
                next_states = [self.env.get_local_state(tid) for tid in self.tl_ids]
                raw_rewards = [self.env.get_raw_reward(tid) for tid in self.tl_ids]
                
                # 4. Compute spatial rewards r̃
                spatial_rewards = self.env.get_spatial_rewards(raw_rewards)
                episode_reward += np.mean(spatial_rewards)
                
                # 5. Store transitions
                for i in range(self.num_agents):
                    self.agents[i].store_transition(
                        states[i]["wave"], states[i]["wait"], fp_list[i],
                        actions[i], spatial_rewards[i], values_list[i], probs_list[i]
                    )
                
                # 6. Update if batch full
                if (step + 1) % self.batch_size == 0:
                    for i in range(self.num_agents):
                        # Get V(s') for bootstrap
                        fps_next = self._get_neighbor_fingerprints(i)
                        _, _, v_next = self.agents[i].get_action(next_states[i]["wave"], next_states[i]["wait"], fps_next)
                        self.agents[i].train(v_next, done=False)
                
                states = next_states
                
            traci.close()
            print(f"  Episode {ep+1}: Avg Spatial Reward = {episode_reward/500:.2f}")

        # Save models
        os.makedirs("results/ma2c", exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save_model(f"results/ma2c/agent_{i}.pt")
        print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    args = parser.parse_args()
    
    trainer = MA2CTrainer(args.sumocfg, num_agents=args.nodes)
    trainer.train()
