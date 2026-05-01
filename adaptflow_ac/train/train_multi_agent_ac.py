import os
import torch
import numpy as np
import argparse
from env.multi_agent_ac_env import MultiAgentACEnv
from agents.multi_agent_ac_agent import MultiAgentACAgent

def get_spatial_reward(tls_id, rewards_dict, neighbors_map, alpha=0.75):
    """
    Method 2: Spatial discount factor alpha for learning stability.
    r̃t,i = Σd Σj|d(i,j)=d α^d * rt,j where α = 0.75
    For simplicity, we'll use 1-hop neighborhood: r_i + alpha * sum(r_neighbors)
    """
    total_r = rewards_dict[tls_id]
    for neighbor_id in neighbors_map[tls_id]:
        total_r += alpha * rewards_dict[neighbor_id]
    return total_r

from env.mock_env import MockEnv

def train_multi_agent_ac(args):
    # 1. Environment Setup
    if args.mode == 'mock':
        env = MockEnv(num_intersections=4, state_mode='dict')
    else:
        env = MultiAgentACEnv(args.sumocfg, gui=args.gui, max_steps=args.steps)
    
    # 2. Agents Setup
    agents = {}
    for tls_id in env.tls_ids:
        # wave: N_lanes, wait: N_lanes, fp: ActionDim * N_neighbors
        wave_dim = len(env.tls_info[tls_id]["lanes"])
        wait_dim = wave_dim
        # Sum of action_dims of neighbors
        fp_dim = sum([env.tls_info[nid]["action_dim"] for nid in env.neighbors[tls_id]])
        action_dim = env.tls_info[tls_id]["action_dim"]
        
        agents[tls_id] = MultiAgentACAgent(tls_id, wave_dim, wait_dim, fp_dim, action_dim)
        
    print(f"Starting Multi-Agent AC Training with {env.num_agents} agents...")
    
    total_steps = 0
    max_total_steps = 1000000 # 1M total steps
    
    while total_steps < max_total_steps:
        # Minibatch collection
        minibatch = {tls_id: [] for tls_id in env.tls_ids}
        
        obs = env.reset()
        for a in agents.values(): a.reset_hidden()
        
        for step_idx in range(args.batch_size):
            # Collect actions
            actions_list = []
            current_fp = {tls_id: agents[tls_id].fingerprint for tls_id in env.tls_ids}
            
            actions = []
            for tls_id in env.tls_ids:
                # Neighbors fingerprints for observability
                neighbors_fp = [current_fp[nid] for nid in env.neighbors[tls_id]]
                act = agents[tls_id].get_action(obs[tls_id]["wave"], obs[tls_id]["wait"], neighbors_fp)
                actions.append(act)
                
            next_obs, rewards, done, _ = env.step(actions)
            
            # Spatial discounting for rewards
            spatial_rewards = {}
            for tls_id in env.tls_ids:
                spatial_rewards[tls_id] = get_spatial_reward(tls_id, rewards, env.neighbors, alpha=args.alpha)
            
            # Store transitions
            for i, tls_id in enumerate(env.tls_ids):
                # fingerprints are used in next step, so we store the current ones
                fp = np.concatenate([current_fp[nid] for nid in env.neighbors[tls_id]]) if env.neighbors[tls_id] else np.zeros(0)
                next_fp = np.concatenate([agents[nid].fingerprint for nid in env.neighbors[tls_id]]) if env.neighbors[tls_id] else np.zeros(0)
                
                minibatch[tls_id].append((
                    obs[tls_id]["wave"], obs[tls_id]["wait"], fp, 
                    actions[i], spatial_rewards[tls_id], 
                    next_obs[tls_id]["wave"], next_obs[tls_id]["wait"], next_fp, 
                    done
                ))
            
            obs = next_obs
            total_steps += 1
            if done: break
            
        # Update agents simultaneously
        c_losses = []
        a_losses = []
        for tls_id in env.tls_ids:
            if minibatch[tls_id]:
                c_loss, a_loss = agents[tls_id].update(minibatch[tls_id])
                c_losses.append(c_loss)
                a_losses.append(a_loss)
                
        print(f"Step {total_steps}: Avg Critic Loss: {np.mean(c_losses):.4f}, Avg Actor Loss: {np.mean(a_losses):.4f}")
        
    # Save models
    os.makedirs('results/multi_agent_ac', exist_ok=True)
    for tls_id, agent in agents.items():
        agent.save_model(f'results/multi_agent_ac/{tls_id}.pt')
        
    print("\nTraining complete. Models saved to results/multi_agent_ac/")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='sumo', choices=['mock', 'sumo'])
    parser.add_argument("--sumocfg", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--steps", type=int, default=720) # Horizon
    parser.add_argument("--alpha", type=float, default=0.75) # Spatial discount
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    train_multi_agent_ac(args)
