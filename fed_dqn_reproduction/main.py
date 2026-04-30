import numpy as np
import torch
import copy
from fed_dqn_reproduction.env.scientific_env import ScientificMockEnv
from fed_dqn_reproduction.agents.dqn_agent import DQNAgent

def aggregate_global_weights(agent_weights_list):
    """ FedAvg on global layers only. """
    global_w = {}
    keys = agent_weights_list[0].keys()
    for key in keys:
        global_w[key] = torch.stack([w[key] for w in agent_weights_list]).mean(0)
    return global_w

def run_experiment(mode, num_episodes=100, fl_interval=20, target_update_interval=30):
    """
    Modes:
    1. IDQN: Independent DQN (no FL)
    2. IDQN_tuned: IDQN + fine-tuning
    3. Fed_trained: FL (partial) without fine-tuning
    4. Proposed: FL (partial) + local fine-tuning
    """
    print(f"\n--- Running Experiment: {mode} ---")
    num_nodes = 4
    env = ScientificMockEnv(num_intersections=num_nodes)
    
    input_dim = 4 * 7 * 3 # forks * features * lanes
    action_dim = 4
    
    agents = [DQNAgent(input_dim, action_dim) for _ in range(num_nodes)]
    
    # Phase 1: Training (with or without FL)
    for ep in range(num_episodes):
        states = env.reset()
        total_rewards = np.zeros(num_nodes)
        done = False
        
        while not done:
            actions = [agents[i].get_action(states[i]) for i in range(num_nodes)]
            next_states, rewards, done, metrics = env.step(actions)
            
            for i in range(num_nodes):
                agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)
                agents[i].train()
                total_rewards[i] += rewards[i]
                
            states = next_states
            
        # Target network update
        if (ep + 1) % target_update_interval == 0:
            for a in agents:
                a.update_target_network()
                
        # FL Aggregation
        if mode in ['Fed_trained', 'Proposed'] and (ep + 1) % fl_interval == 0:
            print(f"  Episode {ep+1}: Global Aggregation...")
            local_wg = [a.get_global_weights() for a in agents]
            wG = aggregate_global_weights(local_wg)
            for a in agents:
                a.set_global_weights(wG)
                
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}, Avg Reward: {np.mean(total_rewards):.2f}, Halted: {metrics['halted_vehicles']:.2f}")

    # Phase 2: Fine-Tuning (if applicable)
    if mode in ['IDQN_tuned', 'Proposed']:
        print(f"  Phase 2: Local Fine-Tuning (30 episodes)...")
        # Proposed freezes wg
        if mode == 'Proposed':
            for a in agents:
                a.freeze_global(True)
        
        for ep in range(30):
            states = env.reset()
            total_rewards = np.zeros(num_nodes)
            done = False
            while not done:
                actions = [agents[i].get_action(states[i]) for i in range(num_nodes)]
                next_states, rewards, done, metrics = env.step(actions)
                for i in range(num_nodes):
                    agents[i].remember(states[i], actions[i], rewards[i], next_states[i], done)
                    agents[i].train()
                    total_rewards[i] += rewards[i]
                states = next_states
                
    return metrics

if __name__ == "__main__":
    modes = ['IDQN', 'IDQN_tuned', 'Fed_trained', 'Proposed']
    results = {}
    for mode in modes:
        results[mode] = run_experiment(mode, num_episodes=60) # Reduced for demo
        
    print("\n" + "="*50)
    print(f"{'Method':<15} | {'Halted':<10} | {'Wait Time':<10}")
    print("-"*50)
    for mode, metrics in results.items():
        print(f"{mode:<15} | {metrics['halted_vehicles']:<10.2f} | {metrics['waiting_time']:<10.2f}")
    print("="*50)
