import torch
import numpy as np
import matplotlib.pyplot as plt
from env.mock_env import MockEnv
from agents.adaptflow import AdaptFlowAgent

def run_sanity_checks():
    print("Running AdaptFlow Sanity Checks...")
    
    # 1. Single Intersection Sanity Check
    print("\n[Check 1] Single Intersection Sanity Check")
    # Initialize MockEnv with num_intersections=1
    env = MockEnv(num_intersections=1)
    # The agent expects a state_dim that includes [queue, wait, phase_onehot]
    # For 4 phases, state_dim is 2 + 4 = 6
    agent = AdaptFlowAgent(node_id=0, state_dim=6, action_dim=4)
    
    # Simple training loop for 200 steps
    state = env.reset()
    rewards = []
    for t in range(200):
        # Mock adjacency for single node
        adj = np.eye(1)
        # Ensure state has 6 features
        state_graph = state[:, :6] # (1, 6)
        state_seq = agent._get_sequence(state_graph)
        
        action = agent.get_action(state_graph, adj)
        next_state, reward_dict, done, _ = env.step(action)
        
        # MockEnv returns {tls_0: reward}
        local_reward = reward_dict["tls_0"]
        all_rewards = [local_reward]

        next_state_graph = next_state[:, :6]
        next_state_seq = agent._get_sequence(next_state_graph)
        
        agent.remember(state_seq, adj, action, all_rewards, next_state_seq, adj, done)
        
        if agent.memory.tree.n_entries > 32:
            agent.train(batch_size=32)
            
        state = next_state
        rewards.append(local_reward)
        if done: break
    
    print(f"  Final reward: {np.sum(rewards):.2f}")
    if np.sum(rewards) > -500: # Adjust threshold based on environment
        print("  PASS: Model is learning.")
    else:
        print("  WARNING: Low reward, check reward scaling.")

    # 2. GAT Output Check
    print("\n[Check 2] GAT Output Check")
    # Sample attention weights
    batch_size, num_nodes, seq_len, feat_dim = 2, 3, 4, 6
    x = torch.randn(batch_size, num_nodes, seq_len, feat_dim)
    adj = torch.ones(batch_size, num_nodes, num_nodes)
    
    agent.model.train()
    # Model now returns probs, values, global_value, alpha
    out, _, _, _ = agent.model(x, adj)
    print(f"  GAT output shape: {out.shape}")
    if not torch.allclose(out[0, 0], out[0, 1]):
        print("  PASS: GAT output is node-specific (not over-smoothed yet).")
    else:
        print("  WARNING: GAT output is identical across nodes. GAT might be over-smoothing.")

    # 3. PER Check
    print("\n[Check 3] PER Check")
    priorities = [agent.memory.tree.tree[i + agent.memory.tree.capacity - 1] for i in range(agent.memory.tree.n_entries)]
    mean_priority = np.mean(priorities)
    print(f"  Mean priority: {mean_priority:.4f}")
    if mean_priority > 0.01:
        print("  PASS: Priorities are being updated.")
    else:
        print("  WARNING: Priorities are near zero.")

    # 4. Advantage Check
    print("\n[Check 4] Advantage Check")
    with torch.no_grad():
        v, gv = agent.model.critic(x, adj)
        print(f"  Value range: {v.min().item():.4f} to {v.max().item():.4f}")
        print(f"  Global Value: {gv.mean().item():.4f}")
    if v.std().item() > 0:
        print("  PASS: Critic is predicting varied values.")
    else:
        print("  WARNING: Critic output is constant.")

if __name__ == "__main__":
    run_sanity_checks()
