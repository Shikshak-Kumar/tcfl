import argparse
import numpy as np
import torch
import os
from collections import deque
from env.mock_env import MockEnv
from env.sumo_env import SumoEnv
from agents.adaptflow import AdaptFlowAgent
from agents.fedlight import FedLightAgent
from utils.federated import federated_aggregate, get_congestion_weights
from utils.plotting import plot_results

def train_adaptflow(env, num_episodes=50, batch_size=32, aggregation_freq=5):
    state_dim = 3 # queue, wait, phase
    action_dim = 4 # phases
    num_nodes = env.num_intersections
    
    # Each node has its own agent
    agents = [AdaptFlowAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        states = env.reset() / 100.0 # Normalize
        total_reward = 0
        done = False
        
        # Track history per node
        node_histories = [deque(maxlen=4) for _ in range(num_nodes)]
        
        while not done:
            # Each agent picks action based on its local view and adjacency
            all_actions = []
            all_stacked = []
            for i in range(num_nodes):
                node_histories[i].append(states[i])
                while len(node_histories[i]) < 4:
                    node_histories[i].appendleft(states[i])
                
                # Each agent's model will receive the full graph state.
                act, stacked = agents[i].get_action(states, env.adj)
                all_actions.append(act[i]) # Only take action for node i
                all_stacked.append(stacked) # (N, 4, D)
            
            next_states, rewards, done, metrics = env.step(all_actions)
            next_states = next_states / 100.0
            
            # Each agent remembers its own experience
            for i in range(num_nodes):
                next_stacked = np.array(agents[i].state_history).transpose(1, 0, 2)
                agents[i].remember(all_stacked[i], env.adj, all_actions[i], rewards[i], next_stacked, env.adj, done)
                agents[i].train(batch_size)
            
            states = next_states
            total_reward += np.mean(rewards)
            
        episode_rewards.append(total_reward)
        print(f"AdaptFlow Episode {ep+1}/{num_episodes}, Reward: {total_reward:.2f}")
        
        # Federated Aggregation
        if (ep + 1) % aggregation_freq == 0:
            print(f"  Performing Federated Aggregation at episode {ep+1}...")
            local_weights = [a.get_weights() for a in agents]
            node_metrics = [{'avg_queue': states[i][0]} for i in range(num_nodes)]
            importance = get_congestion_weights(node_metrics)
            
            global_weights = federated_aggregate(local_weights, weights=importance)
            
            # Broadcast back
            for a in agents:
                a.set_weights(global_weights)
    
    # Save the final global model
    os.makedirs('results', exist_ok=True)
    agents[0].save_model('results/af_model.pt')
    print("AdaptFlow model saved to results/af_model.pt")
                
    return episode_rewards, metrics

def train_fedlight(env, num_episodes=50, batch_size=32, aggregation_freq=5):
    state_dim = 3
    action_dim = 4
    num_nodes = env.num_intersections
    
    # Each node has its own agent
    agents = [FedLightAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        states = env.reset() / 100.0
        total_reward = 0
        done = False
        
        while not done:
            all_actions = []
            for i in range(num_nodes):
                act = agents[i].get_action(states)
                all_actions.append(act[i])
                
            next_states, rewards, done, metrics = env.step(all_actions)
            next_states = next_states / 100.0
            
            for i in range(num_nodes):
                agents[i].remember(states, all_actions[i], rewards[i], next_states, done)
                agents[i].train(batch_size)
            
            states = next_states
            total_reward += np.mean(rewards)
            
        episode_rewards.append(total_reward)
        print(f"FedLight Episode {ep+1}/{num_episodes}, Reward: {total_reward:.2f}")
        
        if (ep + 1) % aggregation_freq == 0:
            print(f"  Performing Federated Aggregation (FedAvg) at episode {ep+1}...")
            local_weights = [a.get_weights() for a in agents]
            global_weights = federated_aggregate(local_weights, weights=None)
            for a in agents:
                a.set_weights(global_weights)
    
    # Save the final global model
    os.makedirs('results', exist_ok=True)
    agents[0].save_model('results/fl_model.pt')
    print("FedLight model saved to results/fl_model.pt")
                
    return episode_rewards, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'sumo'])
    parser.add_argument('--sumocfg', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    
    if args.mode == 'mock':
        env = MockEnv(num_intersections=4)
    else:
        if args.sumocfg is None:
            # Try to find a default sumocfg in the workspace
            args.sumocfg = 'backend/sumo_configs/china_osm.sumocfg' # Placeholder
        env = SumoEnv(args.sumocfg)
        
    print(f"Starting Training in {args.mode.upper()} mode...")
    
    af_rewards, af_metrics = [], {}
    fl_rewards, fl_metrics = [], {}
    
    try:
        # 1. Train AdaptFlow-AC
        af_rewards, af_metrics = train_adaptflow(env, num_episodes=args.episodes)
        
        # 2. Train FedLight Baseline
        fl_rewards, fl_metrics = train_fedlight(env, num_episodes=args.episodes)
    except Exception as e:
        print(f"Training interrupted: {e}")
    
    # Plotting
    results = {}
    if af_rewards:
        results['AdaptFlow-AC'] = {'rewards': af_rewards, 'metrics': af_metrics}
    if fl_rewards:
        results['FedLight'] = {'rewards': fl_rewards, 'metrics': fl_metrics}
    
    if results:
        plot_results(results)
        print("Results plotted in results/ directory.")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
