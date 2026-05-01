import os
import torch
import numpy as np
import argparse
from env.fed_dqn_tsc_env import FedDQNTscEnv
from agents.fed_dqn_tsc_agent import FedDQNTscAgent

from env.mock_env import MockEnv

def train_federated(args):
    # 1. Environment Setup
    if args.mode == 'mock':
        env = MockEnv(num_intersections=4, state_mode='3d')
    else:
        env = FedDQNTscEnv(args.sumocfg, gui=args.gui, max_steps=args.steps)
    num_nodes = env.num_intersections
    
    # State space: N_feat * Max_Lanes * Max_Forks
    input_dim = env.n_features * env.max_lanes * env.max_forks
    
    # 2. Agents Setup
    agents = []
    for i, tls_id in enumerate(env.tls_ids):
        action_dim = env.tls_info[tls_id]["num_phases"]
        agents.append(FedDQNTscAgent(i, input_dim, action_dim))
        
    print(f"Starting Federated Training with {num_nodes} nodes...")
    
    for round_idx in range(args.rounds):
        print(f"\n--- FL Round {round_idx+1}/{args.rounds} ---")
        
        # Local Training for 20 episodes (as per paper)
        for ep in range(args.episodes_per_round):
            states = env.reset()
            total_rewards = np.zeros(num_nodes)
            done = False
            
            while not done:
                actions = []
                for i in range(num_nodes):
                    actions.append(agents[i].get_action(states[i]))
                
                next_states, rewards, done, _ = env.step(actions)
                
                for i in range(num_nodes):
                    agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)
                    agents[i].train(batch_size=32)
                
                states = next_states
                total_rewards += rewards
                
            print(f"  Episode {ep+1}: Avg Reward: {np.mean(total_rewards):.2f}")
            
            # Target Network Update (every 30 episodes)
            global_ep_idx = round_idx * args.episodes_per_round + ep
            if (global_ep_idx + 1) % 30 == 0:
                print(f"    Updating Target Networks...")
                for a in agents:
                    a.update_target_network()
                    
        # Federated Aggregation (every 20 episodes / every round)
        print(f"  Aggregating Global Layers...")
        global_weights_list = [a.get_global_weights() for a in agents]
        
        # FedAvg on global layers
        avg_weights = {}
        for key in global_weights_list[0].keys():
            avg_weights[key] = torch.stack([w[key] for w in global_weights_list]).mean(dim=0)
            
        # Broadcast back
        for a in agents:
            a.set_global_weights(avg_weights)
            
    # Save the global model weights
    os.makedirs('results', exist_ok=True)
    torch.save(avg_weights, 'results/fed_dqn_tsc_global.pt')
    print("\nGlobal weights saved to results/fed_dqn_tsc_global.pt")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='sumo', choices=['mock', 'sumo'])
    parser.add_argument("--sumocfg", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-round", type=int, default=20) # aggregation every 20 ep
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    train_federated(args)
