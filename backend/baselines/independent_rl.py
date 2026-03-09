"""
Independent Reinforcement Learning Baseline

Each intersection learns independently without sharing knowledge.
Used to demonstrate the benefits of federated learning (collaboration).

Note: Uses same DQN agent but each intersection trains separately.
"""

import numpy as np
from typing import Dict, List
from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment


class IndependentRLController:
    """
    Independent RL controller (no collaboration baseline).
    
    Each intersection trains its own agent independently.
    """
    
    def __init__(
        self,
        sumo_config_paths: List[str],
        state_size: int = 12,
        action_size: int = 4,
        gui: bool = False
    ):
        """
        Initialize independent RL controllers.
        
        Args:
            sumo_config_paths: List of SUMO config paths (one per intersection)
            state_size: State space size
            action_size: Action space size
            gui: Enable GUI
        """
        self.sumo_config_paths = sumo_config_paths
        self.state_size = state_size
        self.action_size = action_size
        self.gui = gui
        
        # One agent per intersection (no sharing)
        self.agents = [
            DQNAgent(state_size, action_size)
            for _ in sumo_config_paths
        ]
        
        # Environments for each intersection
        self.environments = [
            SUMOTrafficEnvironment(config, gui=gui)
            for config in sumo_config_paths
        ]
        
        self.episodes_per_round = 10
        self.max_steps_per_episode = 1000
    
    def train(self, num_rounds: int = 15, episodes_per_round: int = 3) -> Dict:
        """
        Train independent RL agents (each learns separately).
        
        Args:
            num_rounds: Number of training rounds
            episodes_per_round: Episodes per round per intersection
            
        Returns:
            Training metrics
        """
        training_history = []
        
        for round_num in range(num_rounds):
            print(f"\n--- Independent RL Round {round_num + 1} ---")
            
            round_rewards = []
            round_losses = []
            
            # Train each intersection independently
            for agent_idx, (agent, config_path) in enumerate(zip(self.agents, self.sumo_config_paths)):
                self.environments[agent_idx].close()
                self.environments[agent_idx] = SUMOTrafficEnvironment(
                    config_path,
                    gui=self.gui
                )
                env = self.environments[agent_idx]
                
                intersection_rewards = []
                intersection_losses = []
                
                for episode in range(episodes_per_round):
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    
                    for step in range(self.max_steps_per_episode):
                        action = agent.act(state, training=True)
                        next_state, reward, done, info = env.step(action)
                        
                        agent.remember(state, action, reward, next_state, done)
                        
                        if len(agent.memory) > agent.batch_size:
                            loss = agent.replay()
                            if loss is not None:
                                intersection_losses.append(loss)
                        
                        state = next_state
                        episode_reward += reward
                        episode_steps += 1
                        
                        if done:
                            break
                    
                    intersection_rewards.append(episode_reward)
                
                round_rewards.extend(intersection_rewards)
                round_losses.extend(intersection_losses)
            
            avg_reward = np.mean(round_rewards) if round_rewards else 0
            
            training_history.append({
                'round': round_num,
                'average_reward': avg_reward,
                'average_loss': np.mean(round_losses) if round_losses else 0.0,
                'episodes': len(self.agents) * episodes_per_round
            })
            
            print(f"Round {round_num + 1}: Avg Reward = {avg_reward:.4f}")
        
        return {
            'training_history': training_history,
            'final_metrics': training_history[-1] if training_history else {}
        }
    
    def evaluate(self) -> Dict:
        """
        Evaluate independent RL agents.
        
        Returns:
            Evaluation metrics
        """
        all_metrics = []
        
        for agent, env, config_path in zip(self.agents, self.environments, self.sumo_config_paths):
            env.close()
            env = SUMOTrafficEnvironment(config_path, gui=self.gui)
            state = env.reset()
            
            total_reward = 0
            total_steps = 0
            
            for step in range(self.max_steps_per_episode):
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                total_reward += reward
                total_steps += 1
                
                if done:
                    break
            
            performance = env.get_performance_metrics()
            all_metrics.append({
                'total_reward': total_reward,
                'average_reward': total_reward / max(total_steps, 1),
                'waiting_time': performance['total_waiting_time'],
                'queue_length': performance['average_queue_length'],
                'max_queue_length': performance['max_queue_length'],
            })
        
        # Aggregate metrics across all intersections
        return {
            'method': 'independent_rl',
            'metrics': {
                'avg_waiting_time': np.mean([m['waiting_time'] for m in all_metrics]),
                'avg_queue_length': np.mean([m['queue_length'] for m in all_metrics]),
                'avg_reward': np.mean([m['average_reward'] for m in all_metrics]),
            },
            'per_intersection': all_metrics
        }

