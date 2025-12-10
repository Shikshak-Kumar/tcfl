"""
Centralized Reinforcement Learning Baseline

This is a non-federated version where all data is pooled together.
Used to demonstrate the benefits of federated learning (privacy, scalability).

Note: This uses the same DQN agent but trains on combined data from all intersections.
"""

import numpy as np
from typing import Dict, List, Tuple
from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
import os
import json


class CentralizedRLController:
    """
    Centralized RL controller (non-federated baseline).
    
    Trains a single agent on combined data from all intersections.
    """
    
    def __init__(
        self,
        sumo_config_paths: List[str],
        state_size: int = 12,
        action_size: int = 4,
        gui: bool = False
    ):
        """
        Initialize centralized RL controller.
        
        Args:
            sumo_config_paths: List of SUMO config paths (all intersections)
            state_size: State space size
            action_size: Action space size
            gui: Enable GUI
        """
        self.sumo_config_paths = sumo_config_paths
        self.state_size = state_size
        self.action_size = action_size
        self.gui = gui
        
        # Single agent trained on all data
        self.agent = DQNAgent(state_size, action_size)
        
        # Environments for each intersection
        self.environments = [
            SUMOTrafficEnvironment(config, gui=gui)
            for config in sumo_config_paths
        ]
        
        self.episodes_per_round = 10
        self.max_steps_per_episode = 1000
    
    def train(self, num_rounds: int = 15, episodes_per_round: int = 3) -> Dict:
        """
        Train centralized RL agent.
        
        Args:
            num_rounds: Number of training rounds
            episodes_per_round: Episodes per round per intersection
            
        Returns:
            Training metrics
        """
        training_history = []
        
        for round_num in range(num_rounds):
            print(f"\n--- Centralized RL Round {round_num + 1} ---")
            
            total_reward = 0
            total_steps = 0
            losses = []
            
            # Train on all intersections (pooled data)
            for env_idx, env in enumerate(self.environments):
                env.close()
                self.environments[env_idx] = SUMOTrafficEnvironment(
                    self.sumo_config_paths[env_idx],
                    gui=self.gui
                )
                env = self.environments[env_idx]
                
                for episode in range(episodes_per_round):
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    
                    for step in range(self.max_steps_per_episode):
                        action = self.agent.act(state, training=True)
                        next_state, reward, done, info = env.step(action)
                        
                        self.agent.remember(state, action, reward, next_state, done)
                        
                        if len(self.agent.memory) > self.agent.batch_size:
                            loss = self.agent.replay()
                            if loss is not None:
                                losses.append(loss)
                        
                        state = next_state
                        episode_reward += reward
                        episode_steps += 1
                        
                        if done:
                            break
                    
                    total_reward += episode_reward
                    total_steps += episode_steps
            
            num_episodes = len(self.environments) * episodes_per_round
            avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
            
            training_history.append({
                'round': round_num,
                'average_reward': avg_reward,
                'total_steps': total_steps,
                'average_loss': np.mean(losses) if losses else 0.0,
                'episodes': num_episodes
            })
            
            print(f"Round {round_num + 1}: Avg Reward = {avg_reward:.4f}")
        
        return {
            'training_history': training_history,
            'final_metrics': training_history[-1] if training_history else {}
        }
    
    def evaluate(self) -> Dict:
        """
        Evaluate centralized RL agent.
        
        Returns:
            Evaluation metrics
        """
        all_metrics = []
        
        for env in self.environments:
            env.close()
            env = SUMOTrafficEnvironment(env.sumo_config_path, gui=self.gui)
            state = env.reset()
            
            total_reward = 0
            total_steps = 0
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.act(state, training=False)
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
            'method': 'centralized_rl',
            'metrics': {
                'avg_waiting_time': np.mean([m['waiting_time'] for m in all_metrics]),
                'avg_queue_length': np.mean([m['queue_length'] for m in all_metrics]),
                'avg_reward': np.mean([m['average_reward'] for m in all_metrics]),
            },
            'per_intersection': all_metrics
        }
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.agent.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model."""
        self.agent.load_model(filepath)

