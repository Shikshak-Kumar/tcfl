import os
import sys
import torch
import numpy as np
import traci
from env.fed_dqn_tsc_env import FedDQNTscEnv
from agents.fed_dqn_tsc_agent import FedDQNTscAgent

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

class FedDQNTscTrainer:
    def __init__(self, sumocfg_path, num_nodes=4, rounds=10, episodes_per_round=5):
        self.sumocfg_path = sumocfg_path
        self.num_nodes = num_nodes
        self.rounds = rounds
        self.episodes_per_round = episodes_per_round
        
        # Initialize env wrapper
        self.env_wrapper = FedDQNTscEnv(sumocfg_path, sigma=0.1)
        
        # Initialize agents
        self.agents = [FedDQNTscAgent(i) for i in range(num_nodes)]
        self.tl_ids = [] # To be populated during SUMO start

    def aggregate_fed_avg(self):
        """w_G = (1/N) Σ w_{g,i}"""
        print("\n[Federated Learning] Performing FedAvg aggregation...")
        global_weights_list = [agent.get_weights() for agent in self.agents]
        
        # Average weights
        aggregated_weights = {}
        for key in global_weights_list[0].keys():
            aggregated_weights[key] = torch.stack([w[key] for w in global_weights_list]).mean(dim=0)
            
        # Broadcast w_G
        for agent in self.agents:
            agent.set_weights(aggregated_weights)

    def train(self):
        print("Starting Federated DQN Training (Scientific Reports 2023 Reproduction)")
        
        # Start SUMO once to get TLS IDs
        traci.start(["sumo", "-c", self.sumocfg_path, "--no-step-log", "true", "--no-warnings", "true"])
        self.tl_ids = traci.trafficlight.getIDList()
        if len(self.tl_ids) < self.num_nodes:
            print(f"Warning: Only {len(self.tl_ids)} intersections found, expected {self.num_nodes}")
            self.num_nodes = len(self.tl_ids)
        traci.close()

        total_episodes = 0
        
        for r in range(self.rounds):
            print(f"\n--- Round {r+1}/{self.rounds} ---")
            
            for ep in range(self.episodes_per_round):
                total_episodes += 1
                traci.start(["sumo", "-c", self.sumocfg_path, "--no-step-log", "true", "--no-warnings", "true"])
                
                episode_rewards = [0] * self.num_nodes
                states = [self.env_wrapper.get_state(self.tl_ids[i]) for i in range(self.num_nodes)]
                
                for step in range(500): # max steps per episode
                    actions = []
                    for i in range(self.num_nodes):
                        actions.append(self.agents[i].get_action(states[i]))
                    
                    # Execute actions
                    for i, tl_id in enumerate(self.tl_ids):
                        traci.trafficlight.setPhase(tl_id, actions[i])
                    
                    traci.simulationStep()
                    
                    next_states = []
                    for i in range(self.num_nodes):
                        next_state = self.env_wrapper.get_state(self.tl_ids[i])
                        reward = self.env_wrapper.get_reward(self.tl_ids[i])
                        done = step == 499
                        
                        self.agents[i].remember(states[i], actions[i], reward, next_state, done)
                        self.agents[i].train(batch_size=32)
                        
                        episode_rewards[i] += reward
                        next_states.append(next_state)
                    
                    states = next_states
                    
                traci.close()
                print(f"  Episode {total_episodes}: Avg Reward = {np.mean(episode_rewards):.2f}")
                
                # Target network update every 30 episodes (Section 11)
                if total_episodes % 30 == 0:
                    print("  [Target Update] Syncing target networks...")
                    for agent in self.agents:
                        agent.update_target_network()

            # Federated Aggregation Round (Section 8)
            self.aggregate_fed_avg()

        # Section 9: Fine-Tuning
        print("\n--- Phase 2: Fine-Tuning (Local Layers Only) ---")
        for agent in self.agents:
            agent.start_fine_tuning()
            
        for ep in range(5): # Fine-tuning episodes
            traci.start(["sumo", "-c", self.sumocfg_path, "--no-step-log", "true", "--no-warnings", "true"])
            states = [self.env_wrapper.get_state(self.tl_ids[i]) for i in range(self.num_nodes)]
            for step in range(500):
                actions = [self.agents[i].get_action(states[i]) for i in range(self.num_nodes)]
                for i, tl_id in enumerate(self.tl_ids):
                    traci.trafficlight.setPhase(tl_id, actions[i])
                traci.simulationStep()
                for i in range(self.num_nodes):
                    next_state = self.env_wrapper.get_state(self.tl_ids[i])
                    reward = self.env_wrapper.get_reward(self.tl_ids[i])
                    self.agents[i].remember(states[i], actions[i], reward, next_state, step==499)
                    self.agents[i].train()
                states = [self.env_wrapper.get_state(self.tl_ids[i]) for i in range(self.num_nodes)]
            traci.close()
            print(f"  Fine-tune Episode {ep+1} complete.")

        # Save final models
        os.makedirs("results/fed_dqn", exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save_model(f"results/fed_dqn/agent_{i}.pt")
        print("\nTraining complete. Models saved to results/fed_dqn/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    
    trainer = FedDQNTscTrainer(args.sumocfg, num_nodes=args.nodes, rounds=args.rounds)
    trainer.train()
