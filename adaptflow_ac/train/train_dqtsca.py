import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import traci

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.dqtsca_agent import DQTSCAAgent
from env.dqtsca_env import DQTSCAEnv
from utils.traci_network_metrics import (
    EpisodeThroughputTally,
    episode_arrival_rate,
    network_approach_lane_means,
)
from utils.traci_tls_phases import tls_phase_count


class DQTSCATrainer:
    def __init__(
        self,
        sumocfg_path: str,
        episodes: int = 10,
        steps_per_episode: int = 500,
        total_train_steps: int = 100000,
        results_dir: str = "results/dqtsca",
        seed: int = 42,
    ):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.total_train_steps = total_train_steps
        self.results_dir = results_dir
        np.random.seed(seed)

        self.env = DQTSCAEnv(sumocfg_path)
        self.agent = DQTSCAAgent("agent_0")
        self.current_action_idx = 0
        self.yellow_time = 3
        self.all_red_time = 2
        self._episode_traci_steps = 0

    def _step_sim(self, tally: EpisodeThroughputTally, n: int = 1) -> None:
        for _ in range(n):
            traci.simulationStep()
            tally.step()
        self._episode_traci_steps += int(n)

    def _execute_phase_transition(
        self, tl_id: str, new_action_idx: int, tally: EpisodeThroughputTally
    ) -> None:
        if new_action_idx == self.current_action_idx:
            traci.trafficlight.setPhase(tl_id, self.env.phase_map[new_action_idx])
            self._step_sim(tally, 5)
            return
        n_phase = tls_phase_count(tl_id)
        yellow_phase = min(
            self.env.phase_map[self.current_action_idx] + 1, max(0, n_phase - 1)
        )
        traci.trafficlight.setPhase(tl_id, yellow_phase)
        self._step_sim(tally, self.yellow_time)
        self._step_sim(tally, self.all_red_time)
        traci.trafficlight.setPhase(tl_id, self.env.phase_map[new_action_idx])
        self._step_sim(tally, 5)
        self.current_action_idx = new_action_idx

    def train(self) -> None:
        print("DQTSCA — single-intersection CNN-DQN")
        os.makedirs(self.results_dir, exist_ok=True)
        curve: List[Dict[str, Any]] = []
        global_step = 0

        for ep in range(self.episodes):
            traci.start(
                [
                    "sumo",
                    "-c",
                    self.sumocfg_path,
                    "--no-step-log",
                    "true",
                    "--no-warnings",
                    "true",
                ]
            )
            self.env.phase_map = {}
            tl_id = traci.trafficlight.getIDList()[0]
            self.current_action_idx = 0
            self.env.prev_delay = self.env.get_cumulative_delay(tl_id)
            tally = EpisodeThroughputTally()
            self._episode_traci_steps = 0
            occ, speed = self.env.get_dtse(tl_id)
            phase_vec = self.env.get_phase_vector(tl_id)
            state = (occ, speed, phase_vec)
            total_reward = 0.0
            losses: List[float] = []

            for step in range(self.steps_per_episode):
                action_idx = self.agent.get_action(occ, speed, phase_vec)
                self._execute_phase_transition(tl_id, action_idx, tally)

                next_occ, next_speed = self.env.get_dtse(tl_id)
                next_phase_vec = self.env.get_phase_vector(tl_id)
                reward = self.env.get_reward(tl_id)
                done = step == self.steps_per_episode - 1
                next_state = (next_occ, next_speed, next_phase_vec)

                self.agent.remember(state, action_idx, reward, next_state, done)
                loss = self.agent.train(
                    batch_size=16, total_steps=self.total_train_steps
                )
                if loss:
                    losses.append(loss)

                state = next_state
                occ, speed, phase_vec = next_occ, next_speed, next_phase_vec
                total_reward += reward
                global_step += 1
                if global_step % 100 == 0:
                    self.agent.update_target_network()

            aq, qtot, qmax, aw = network_approach_lane_means([tl_id])
            curve.append(
                {
                    "episode": ep + 1,
                    "total_reward": float(total_reward),
                    "avg_loss": float(np.mean(losses)) if losses else 0.0,
                    "avg_queue": aq,
                    "avg_queue_total": qtot,
                    "avg_queue_max": qmax,
                    "avg_wait": aw,
                    "tp_ratio": tally.ratio(),
                    "arrival_rate": episode_arrival_rate(tally, self._episode_traci_steps),
                    "epsilon": float(self.agent.epsilon),
                }
            )
            traci.close()
            print(
                f"  Episode {ep + 1}: R={total_reward:.2f}  loss={curve[-1]['avg_loss']:.6f}  "
                f"Q={aq:.4f}  W={aw:.3f}  TP={curve[-1]['tp_ratio']:.4f}  "
                f"Arr/s={curve[-1]['arrival_rate']:.4f}"
            )

        self.agent.save_model(os.path.join(self.results_dir, "model.pt"))
        with open(os.path.join(self.results_dir, "dqtsca_training.json"), "w") as f:
            json.dump(curve, f, indent=2)
        print(f"Training complete → {self.results_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", type=str, required=True)
    p.add_argument("--rounds", type=int, default=10, help="Episodes (align with other methods)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--total-train-steps", type=int, default=100000)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.results_dir or os.path.join(_ROOT, "results", "dqtsca")
    trainer = DQTSCATrainer(
        args.sumocfg,
        episodes=args.rounds,
        steps_per_episode=args.steps,
        total_train_steps=args.total_train_steps,
        results_dir=out,
        seed=args.seed,
    )
    trainer.train()
