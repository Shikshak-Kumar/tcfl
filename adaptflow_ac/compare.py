import argparse
import numpy as np
import torch
import os
import json
from adaptflow_ac.env.mock_env import MockEnv
from adaptflow_ac.env.sumo_env import SumoEnv
from adaptflow_ac.agents.adaptflow import AdaptFlowAgent
from adaptflow_ac.agents.fedlight import FedLightAgent
from adaptflow_ac.utils.plotting import plot_results

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    def get_action(self, states, *args):
        # states: (N, D)
        return [np.random.randint(self.action_dim) for _ in range(len(states))], None

def evaluate_agent(agent_type, env, episodes=5, model_path=None):
    state_dim = 3
    action_dim = 4
    num_nodes = env.num_intersections
    
    # Initialize agents
    if agent_type == 'adaptflow':
        agents = [AdaptFlowAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    elif agent_type == 'fedlight':
        agents = [FedLightAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    elif agent_type == 'random':
        agents = [RandomAgent(action_dim) for _ in range(num_nodes)]
    else:
        raise ValueError("Unknown agent type")
        
    # Load model if provided
    if model_path and os.path.exists(model_path) and agent_type != 'random':
        print(f"Loading {agent_type} model from {model_path}...")
        for a in agents:
            a.load_model(model_path)
            
    all_rewards = []
    final_metrics = {
        'avg_queue': [],
        'avg_waiting': [],
        'throughput': []
    }
    
    for ep in range(episodes):
        states = env.reset() / 100.0
        total_reward = 0
        done = False
        
        while not done:
            all_actions = []
            for i in range(num_nodes):
                if agent_type == 'adaptflow':
                    act, _ = agents[i].get_action(states, env.adj)
                    all_actions.append(act[i])
                elif agent_type == 'fedlight':
                    act = agents[i].get_action(states)
                    all_actions.append(act[i])
                else:
                    act, _ = agents[i].get_action(states)
                    all_actions.append(act[i])
                    
            next_states, rewards, done, metrics = env.step(all_actions)
            states = next_states / 100.0
            total_reward += np.mean(rewards)
            
        all_rewards.append(total_reward)
        final_metrics['avg_queue'].append(metrics['avg_queue'])
        final_metrics['avg_waiting'].append(metrics['avg_waiting'])
        final_metrics['throughput'].append(metrics['throughput'])
        
    # Average across episodes
    summary = {
        'rewards': all_rewards,
        'metrics': {k: np.mean(v) for k, v in final_metrics.items()}
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'sumo'])
    parser.add_argument('--sumocfg', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--af-model', type=str, default='results/af_model.pt')
    parser.add_argument('--fl-model', type=str, default='results/fl_model.pt')
    args = parser.parse_args()
    
    if args.mode == 'mock':
        env = MockEnv(num_intersections=4)
    else:
        env = SumoEnv(args.sumocfg or 'backend/sumo_configs/intersection.sumocfg')
        
    print(f"Starting Comparison in {args.mode.upper()} mode...")
    
    results = {}
    
    # 1. Evaluate AdaptFlow-AC
    print("\nEvaluating AdaptFlow-AC...")
    results['AdaptFlow-AC'] = evaluate_agent('adaptflow', env, args.episodes, args.af_model)
    
    # 2. Evaluate FedLight
    print("\nEvaluating FedLight...")
    results['FedLight'] = evaluate_agent('fedlight', env, args.episodes, args.fl_model)
    
    # 3. Evaluate Random Baseline
    print("\nEvaluating Random Baseline...")
    results['Random'] = evaluate_agent('random', env, args.episodes)
    
    # Print Table
    print("\n" + "="*50)
    print(f"{'Agent':<15} | {'Reward':<10} | {'Queue':<10} | {'Wait (s)':<10}")
    print("-"*50)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<15} | {np.mean(res['rewards']):<10.2f} | {m['avg_queue']:<10.2f} | {m['avg_waiting']:<10.2f}")
    print("="*50)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_report.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    plot_results(results, save_dir='results/comparison_plots')
    print("\nComparison report saved to results/comparison_report.json")
    print("Comparison plots saved to results/comparison_plots/")

if __name__ == "__main__":
    main()
