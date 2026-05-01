import argparse
import numpy as np
import torch
import os
import json
from env.mock_env import MockEnv
from env.sumo_env import SumoEnv
from env.fed_dqn_tsc_env import FedDQNTscEnv
from env.multi_agent_ac_env import MultiAgentACEnv
from agents.adaptflow import AdaptFlowAgent
from agents.fedlight import FedLightAgent
from agents.fed_dqn_tsc_agent import FedDQNTscAgent
from agents.multi_agent_ac_agent import MultiAgentACAgent
from utils.plotting import plot_results

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    def get_action(self, states, *args):
        # states: (N, D)
        return [np.random.randint(self.action_dim) for _ in range(len(states))], None

def evaluate_agent(agent_type, env, episodes=5, model_path=None):
    state_dim = 6 # Updated to match AdaptFlow 2.0 (2 node features + 4 one-hot phase)
    action_dim = 4
    num_nodes = env.num_intersections
    
    # Initialize agents
    if agent_type == 'adaptflow':
        agents = [AdaptFlowAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    elif agent_type == 'fedlight':
        agents = [FedLightAgent(i, state_dim, action_dim) for i in range(num_nodes)]
    elif agent_type == 'fed_dqn_tsc':
        input_dim = 6 * 8 * 4 # features * lanes * forks (defaults in FedDQNTscEnv)
        agents = [FedDQNTscAgent(i, input_dim, action_dim) for i in range(num_nodes)]
    elif agent_type == 'multi_agent_ac':
        # Assuming defaults from MultiAgentACEnv
        wave_dim = 4 # Example
        wait_dim = 4
        fp_dim = action_dim * 2 # 2 neighbors example
        agents = [MultiAgentACAgent(tls_id, wave_dim, wait_dim, fp_dim, action_dim) for tls_id in range(num_nodes)]
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
        states = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            all_actions = []
            for i in range(num_nodes):
                if agent_type == 'adaptflow':
                    # Normalize mock states to match AdaptFlow training distribution
                    q = states[:, 0]
                    w = states[:, 1]
                    p = states[:, 2].astype(int)
                    
                    norm_q = np.clip(q / 5.0, 0.0, 2.0)
                    norm_w = np.clip(w / 100.0, 0.0, 2.0)
                    
                    # Construct 6-dim state: [norm_q, norm_w, phase_onehot]
                    s_af = []
                    for j in range(num_nodes):
                        p_oh = np.zeros(4)
                        if p[j] < 4: p_oh[p[j]] = 1.0
                        s_af.append([norm_q[j], norm_w[j]] + p_oh.tolist())
                    s_af = np.array(s_af)
                    
                    acts = agents[i].get_action(s_af, env.adj)
                    all_actions.append(acts[i])
                elif agent_type == 'fedlight':
                    s_fl = states[:, :6]
                    act = agents[i].get_action(s_fl)
                    all_actions.append(act[i])
                elif agent_type == 'fed_dqn_tsc':
                    # FedDQNTsc agent expects (6, 8, 4) or similar 3D state
                    act = agents[i].get_action(states[i], evaluation=True)
                    all_actions.append(act)
                elif agent_type == 'multi_agent_ac':
                    # MultiAgentAC expects wave, wait, and neighbor fingerprints
                    tls_id = env.tls_ids[i]
                    wave = states[tls_id]["wave"] if isinstance(states, dict) else states[i, :4]
                    wait = states[tls_id]["wait"] if isinstance(states, dict) else states[i, 4:8]
                    
                    neighbors = env.neighbors[tls_id]
                    neighbor_indices = [env.tls_ids.index(nid) for nid in neighbors]
                    fps = [agents[idx].fingerprint for idx in neighbor_indices]
                    
                    act = agents[i].get_action(wave, wait, fps, evaluation=True)
                    all_actions.append(act)
                elif agent_type == 'random':
                    s_rnd = states[:, :3]
                    act, _ = agents[i].get_action(s_rnd)
                    all_actions.append(act[i])
                else:
                    all_actions.append(0)
                    
            next_states, rewards, done, metrics = env.step(all_actions)
            states = next_states
            
            # Handle rewards (could be list or dict)
            if isinstance(rewards, dict):
                r_val = np.mean(list(rewards.values()))
            else:
                r_val = np.mean(rewards)
            total_reward += r_val
            
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
    parser.add_argument('--fed-dqn-model', type=str, default='results/fed_dqn_tsc_global.pt')
    parser.add_argument('--mac-model-dir', type=str, default='results/multi_agent_ac')
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
    
    # 3. Evaluate FedDQNTsc
    print("\nEvaluating FedDQNTsc (Scientific Reports 2023)...")
    if args.mode == 'sumo':
        sci_env = FedDQNTscEnv(args.sumocfg or 'backend/sumo_configs/intersection.sumocfg')
        results['FedDQNTsc'] = evaluate_agent('fed_dqn_tsc', sci_env, args.episodes, args.fed_dqn_model)
        sci_env.close()
    else:
        # Mock mode - use 3D state mode for FedDQNTsc
        fed_dqn_env = MockEnv(num_intersections=4, state_mode='3d')
        results['FedDQNTsc'] = evaluate_agent('fed_dqn_tsc', fed_dqn_env, args.episodes, args.fed_dqn_model)
        
    # 4. Evaluate MultiAgentAC
    print("\nEvaluating MultiAgentAC (Chu et al. 2019)...")
    if args.mode == 'sumo':
        ma2c_env = MultiAgentACEnv(args.sumocfg or 'backend/sumo_configs/intersection.sumocfg')
        results['MultiAgentAC'] = evaluate_agent('multi_agent_ac', ma2c_env, args.episodes, args.mac_model_dir)
        ma2c_env.close()
    else:
        # Mock mode - use dict state mode for MultiAgentAC
        mac_env = MockEnv(num_intersections=4, state_mode='dict')
        results['MultiAgentAC'] = evaluate_agent('multi_agent_ac', mac_env, args.episodes, args.mac_model_dir)

    # 5. Evaluate Random Baseline
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
