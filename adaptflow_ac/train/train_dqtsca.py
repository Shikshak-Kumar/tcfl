import os
import sys
import torch
import numpy as np
import traci
from env.dqtsca_env import DQTSCAEnv
from agents.dqtsca_agent import DQTSCAAgent

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

class DQTSCATrainer:
    def __init__(self, sumocfg_path, episodes=10, total_train_steps=100000):
        self.sumocfg_path = sumocfg_path
        self.episodes = episodes
        self.total_train_steps = total_train_steps
        
        self.env = DQTSCAEnv(sumocfg_path)
        self.agent = DQTSCAAgent("agent_0")
        
        # Internal state for phase tracking
        self.current_action_idx = 0
        self.yellow_time = 3
        self.all_red_time = 2

    def _execute_phase_transition(self, tl_id, new_action_idx):
        """
        Automatic transition: Yellow -> All Red
        """
        if new_action_idx == self.current_action_idx:
            # Stay in green
            traci.trafficlight.setPhase(tl_id, self.env.phase_map[new_action_idx])
            for _ in range(5): traci.simulationStep() # 5s step
            return
        
        # 1. Yellow Phase
        # Assuming yellow phases are at current_action_idx * 2 + 1
        yellow_phase = self.env.phase_map[self.current_action_idx] + 1
        traci.trafficlight.setPhase(tl_id, yellow_phase)
        for _ in range(self.yellow_time): traci.simulationStep()
        
        # 2. All Red (Heuristic: just set a non-existent phase or stay in yellow if all-red not defined)
        # For simplicity, we step 2 more times in a "safe" state.
        for _ in range(self.all_red_time): traci.simulationStep()
        
        # 3. New Green Phase
        traci.trafficlight.setPhase(tl_id, self.env.phase_map[new_action_idx])
        for _ in range(5): traci.simulationStep()
        
        self.current_action_idx = new_action_idx

    def train(self):
        print("Starting DQTSCA Training (Single Agent CNN Reproduction)")
        
        global_step = 0
        for ep in range(self.episodes):
            traci.start(["sumo", "-c", self.sumocfg_path, "--no-step-log", "true", "--no-warnings", "true"])
            tl_id = traci.trafficlight.getIDList()[0]
            
            self.env.prev_delay = self.env.get_cumulative_delay(tl_id)
            occ, speed = self.env.get_dtse(tl_id)
            phase_vec = self.env.get_phase_vector(tl_id)
            state = (occ, speed, phase_vec)
            
            total_reward = 0
            
            for step in range(200): # steps per episode
                # 1. Choose action
                action_idx = self.agent.get_action(occ, speed, phase_vec)
                
                # 2. Execute with transition
                self._execute_phase_transition(tl_id, action_idx)
                
                # 3. Observe
                next_occ, next_speed = self.env.get_dtse(tl_id)
                next_phase_vec = self.env.get_phase_vector(tl_id)
                reward = self.env.get_reward(tl_id)
                done = step == 199
                
                next_state = (next_occ, next_speed, next_phase_vec)
                
                # 4. Store and train
                self.agent.remember(state, action_idx, reward, next_state, done)
                loss = self.agent.train(batch_size=16, total_steps=self.total_train_steps)
                
                state = next_state
                occ, speed, phase_vec = next_occ, next_speed, next_phase_vec
                total_reward += reward
                global_step += 1
                
                if global_step % 100 == 0:
                    self.agent.update_target_network()
            
            traci.close()
            print(f"  Episode {ep+1}: Reward = {total_reward:.2f}, Epsilon = {self.agent.epsilon:.4f}")

        # Save model
        os.makedirs("results/dqtsca", exist_ok=True)
        self.agent.save_model("results/dqtsca/model.pt")
        print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, required=True)
    args = parser.parse_args()
    
    trainer = DQTSCATrainer(args.sumocfg)
    trainer.train()
