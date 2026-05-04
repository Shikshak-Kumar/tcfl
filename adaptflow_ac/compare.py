import argparse
import glob
import numpy as np
import torch
import os
import json
from env.mock_env import MockEnv
from env.sumo_env import SumoEnv
from env.fed_dqn_tsc_env import FedDQNTscEnv
from env.multi_agent_ac_env import MultiAgentACEnv
from env.dqtsca_env import DQTSCAEnv
from agents.adaptflow import AdaptFlowAgent
from agents.fed_dqn_tsc_agent import FedDQNTscAgent
from agents.multi_agent_ac_agent import MultiAgentACAgent
from agents.dqtsca_agent import DQTSCAAgent
from utils.plotting import plot_results

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    def get_action(self, *args, **kwargs):
        return np.random.randint(self.action_dim)

def evaluate_agent(agent_type, env, episodes=5, model_path=None):
    num_nodes = getattr(env, 'num_intersections', getattr(env, 'num_agents', 1))
    action_dim = 4
    
    if agent_type == 'adaptflow':
        agents = [AdaptFlowAgent(i, state_dim=4, action_dim=action_dim) for i in range(num_nodes)]
    elif agent_type == 'fed_dqn_tsc':
        agents = [FedDQNTscAgent(i, state_dim=224, action_dim=action_dim) for i in range(num_nodes)]
    elif agent_type == 'multi_agent_ac':
        agents = [MultiAgentACAgent(i, wave_dim=8, wait_dim=8, fp_dim=16, action_dim=action_dim) for i in range(num_nodes)]
    elif agent_type == 'dqtsca':
        agents = [DQTSCAAgent(i, grid_size=20, action_dim=action_dim) for i in range(num_nodes)]
    elif agent_type == 'random':
        agents = [RandomAgent(action_dim) for _ in range(num_nodes)]
    else:
        raise ValueError("Unknown agent type")
        
    if model_path and os.path.exists(model_path) and agent_type != 'random':
        print(f"Loading {agent_type} model from {model_path}...")
        for a in agents:
            try: a.load_model(model_path)
            except: pass
            
    all_rewards = []
    final_metrics = {'avg_queue': [], 'avg_waiting': [], 'throughput': []}
    
    for ep in range(episodes):
        states = env.reset() if hasattr(env, 'reset') else None
        total_reward = 0
        done = False
        step_count = 0
        if agent_type == 'multi_agent_ac':
            for a in agents: a.reset_hidden()
        
        while not done and step_count < 500:
            all_actions = []
            for i in range(num_nodes):
                if agent_type == 'adaptflow':
                    # Simplified for comparison
                    acts = agents[i].get_action(states, getattr(env, 'adj', np.eye(num_nodes)))
                    all_actions = acts
                    break
                elif agent_type == 'fed_dqn_tsc':
                    s = env.get_state(env.tl_ids[i]) if hasattr(env, 'get_state') else np.zeros((4, 7, 8))
                    all_actions.append(agents[i].get_action(s, evaluation=True))
                elif agent_type == 'multi_agent_ac':
                    s_dict = env.get_local_state(env.tls_ids[i]) if hasattr(env, 'get_local_state') else {"wave":np.zeros(8),"wait":np.zeros(8)}
                    act, _, _ = agents[i].get_action(s_dict["wave"], s_dict["wait"], [np.ones(4)/4]*4, evaluation=True)
                    all_actions.append(act)
                elif agent_type == 'dqtsca':
                    occ, speed = env.get_dtse(env.tl_ids[i]) if hasattr(env, 'get_dtse') else (np.zeros((20,20)), np.zeros((20,20)))
                    phase = env.get_phase_vector(env.tl_ids[i]) if hasattr(env, 'get_phase_vector') else np.zeros(4)
                    all_actions.append(agents[i].get_action(occ, speed, phase, evaluation=True))
                elif agent_type == 'random':
                    all_actions.append(agents[i].get_action())
            
            # Pad or trim all_actions
            if len(all_actions) < num_nodes: all_actions.extend([0]*(num_nodes - len(all_actions)))
            all_actions = all_actions[:num_nodes]

            if hasattr(env, 'step'):
                next_states, rewards, done, metrics = env.step(all_actions)
                states = next_states
            else: break
            
            r_val = np.mean(list(rewards.values())) if isinstance(rewards, dict) else np.mean(rewards)
            total_reward += r_val
            step_count += 1
            
        all_rewards.append(total_reward)
        final_metrics['avg_queue'].append(metrics.get('avg_queue', 0))
        final_metrics['avg_waiting'].append(metrics.get('avg_waiting', 0))
        final_metrics['throughput'].append(metrics.get('throughput', 0))
        
    return {'rewards': all_rewards, 'metrics': {k: np.mean(v) for k, v in final_metrics.items()}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 'sumo'])
    parser.add_argument('--sumocfg', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument(
        '--research-dir',
        type=str,
        default=None,
        help='If set, load checkpoints from <dir>/adaptflow|fed_dqn_tsc|multi_agent_ac|dqtsca',
    )
    args = parser.parse_args()
    cfg = args.sumocfg or 'backend/sumo_configs/intersection.sumocfg'
    results = {}

    if args.research_dir:
        rd = os.path.abspath(args.research_dir)
        g = sorted(glob.glob(os.path.join(rd, 'adaptflow', 'adaptflow_global_*.pt')))
        af_pt = g[-1] if g else None
        models = {
            'AdaptFlow-AC': ('adaptflow', af_pt),
            'FedDQNTsc': ('fed_dqn_tsc', os.path.join(rd, 'fed_dqn_tsc', 'agent_0.pt')),
            'MultiAgentAC': ('multi_agent_ac', os.path.join(rd, 'multi_agent_ac', 'agent_0.pt')),
            'DQTSCA': ('dqtsca', os.path.join(rd, 'dqtsca', 'model.pt')),
            'Random': ('random', None),
        }
    else:
        models = {
            'AdaptFlow-AC': ('adaptflow', 'results/af_model.pt'),
            'FedDQNTsc': ('fed_dqn_tsc', 'results/fed_dqn/agent_0.pt'),
            'MultiAgentAC': ('multi_agent_ac', 'results/ma2c/agent_0.pt'),
            'DQTSCA': ('dqtsca', 'results/dqtsca/model.pt'),
            'Random': ('random', None),
        }
    
    import traci
    for name, (atype, mpath) in models.items():
        print(f"\nEvaluating {name}...")
        try: traci.start(["sumo", "-c", cfg])
        except: pass
        
        if atype == 'fed_dqn_tsc':
            env = FedDQNTscEnv(cfg)
            env.tl_ids = list(traci.trafficlight.getIDList())
        elif atype == 'multi_agent_ac':
            env = MultiAgentACEnv(cfg)
            env.tls_ids = list(traci.trafficlight.getIDList())
        elif atype == 'dqtsca':
            env = DQTSCAEnv(cfg)
            env.tl_ids = list(traci.trafficlight.getIDList())
        else:
            env = SumoEnv(cfg, gui=False, max_steps=500)
        
        results[name] = evaluate_agent(atype, env, args.episodes, mpath)
        traci.close()
    
    print("\n" + "="*50 + f"\n{'Agent':<15} | {'Reward':<10} | {'Queue':<10} | {'Wait (s)':<10}\n" + "-"*50)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<15} | {np.mean(res['rewards']):<10.2f} | {m['avg_queue']:<10.2f} | {m['avg_waiting']:<10.2f}")
    print("="*50)
    
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_report.json', 'w') as f: json.dump(results, f, indent=4)
    plot_results(results, save_dir='results/comparison_plots')

if __name__ == "__main__":
    main()
