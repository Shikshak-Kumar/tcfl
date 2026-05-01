import os
import torch
import numpy as np
import argparse
from env.fed_dqn_tsc_env import FedDQNTscEnv
from agents.fed_dqn_tsc_agent import FedDQNTscAgent

def fine_tune(args):
    # 1. Environment Setup (New Intersection)
    env = FedDQNTscEnv(args.sumocfg, gui=args.gui, max_steps=args.steps)
    num_nodes = env.num_intersections
    input_dim = env.n_features * env.max_lanes * env.max_forks
    
    # 2. Agents Setup
    agents = []
    for i, tls_id in enumerate(env.tls_ids):
        action_dim = env.tls_info[tls_id]["num_phases"]
        agent = FedDQNTscAgent(i, input_dim, action_dim)
        
        # Load Global Weights
        if os.path.exists(args.global_model):
            print(f"Loading global weights from {args.global_model} for node {i}...")
            global_weights = torch.load(args.global_model)
            agent.set_global_weights(global_weights)
        
        # Freeze Global Layers
        agent.freeze_global()
        agents.append(agent)
        
    print(f"Starting Fine-tuning on {num_nodes} new intersections...")
    
    for ep in range(args.episodes):
        states = env.reset()
        total_rewards = np.zeros(num_nodes)
        done = False
        
        while not done:
            actions = []
            for i in range(num_nodes):
                actions.append(agents[i].get_action(states[i]))
            
            next_states, rewards, done, _ = env.step(actions)
            
            for i in range(num_nodes):
                # Only training local layers here (optimizer filtered in agent.freeze_global)
                agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)
                agents[i].train(batch_size=32)
            
            states = next_states
            total_rewards += rewards
            
        print(f"  Episode {ep+1}: Avg Reward: {np.mean(total_rewards):.2f}")
        
    # Save fine-tuned models
    os.makedirs('results/fine_tuned', exist_ok=True)
    for i, a in enumerate(agents):
        a.save_model(f'results/fine_tuned/node_{i}.pt')
        
    print("\nFine-tuning complete. Models saved to results/fine_tuned/")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, required=True)
    parser.add_argument("--global-model", type=str, default='results/fed_dqn_tsc_global.pt')
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    fine_tune(args)
