import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import traci

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.multi_agent_ac_agent import MultiAgentACAgent
from env.multi_agent_ac_env import MultiAgentACEnv
from utils.traci_network_metrics import (
    EpisodeThroughputTally,
    episode_arrival_rate,
    network_approach_lane_means,
)
from utils.traci_tls_phases import set_phase_from_discrete_action


class MA2CTrainer:
    def __init__(
        self,
        sumocfg_path: str,
        num_agents: int = 6,
        batch_size: int = 100,
        episodes: int = 10,
        steps_per_episode: int = 500,
        results_dir: str = "results/multi_agent_ac",
        seed: int = 42,
    ):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.num_agents = num_agents
        self.batch_size = max(1, batch_size)
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.results_dir = results_dir
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = MultiAgentACEnv(self.sumocfg_path)
        self.env.initialize()
        self.agents = [MultiAgentACAgent(i, fp_dim=16) for i in range(num_agents)]
        self.tl_ids = []

    def _get_neighbor_fingerprints(self, agent_idx: int):
        fps = []
        neighbors = []
        if agent_idx > 0:
            neighbors.append(agent_idx - 1)
        if agent_idx < self.num_agents - 1:
            neighbors.append(agent_idx + 1)
        for n in neighbors:
            fps.append(self.agents[n].fingerprint)
        while len(fps) < 4:
            fps.append(np.zeros(4))
        return fps

    def train(self) -> None:
        print("MA2C — Multi-agent A2C (Chu et al. 2019)")
        os.makedirs(self.results_dir, exist_ok=True)

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
        all_tls = list(traci.trafficlight.getIDList())
        traci.close()
        self.tl_ids = all_tls[: self.num_agents]
        if len(self.tl_ids) < self.num_agents:
            self.num_agents = len(self.tl_ids)
            self.agents = self.agents[: self.num_agents]

        training_log: List[Dict[str, Any]] = []
        by_round: List[Dict[str, Any]] = []
        update_id = 0

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
            tally = EpisodeThroughputTally()
            for agent in self.agents:
                agent.reset_hidden()

            states = [self.env.get_local_state(tid) for tid in self.tl_ids]
            episode_reward = 0.0
            ep_losses: List[float] = []

            for step in range(self.steps_per_episode):
                actions = []
                probs_list = []
                values_list = []
                fp_list = []

                for i in range(self.num_agents):
                    fps = self._get_neighbor_fingerprints(i)
                    fp_concat = np.concatenate(fps)
                    fp_list.append(fp_concat)
                    act, prob, val = self.agents[i].get_action(
                        states[i]["wave"], states[i]["wait"], fps
                    )
                    actions.append(act)
                    probs_list.append(prob)
                    values_list.append(val)

                for i, tid in enumerate(self.tl_ids):
                    set_phase_from_discrete_action(tid, actions[i], n_actions=4)
                traci.simulationStep()
                tally.step()

                next_states = [self.env.get_local_state(tid) for tid in self.tl_ids]
                raw_rewards = [self.env.get_raw_reward(tid) for tid in self.tl_ids]
                spatial_rewards = self.env.get_spatial_rewards(raw_rewards)
                episode_reward += float(np.mean(spatial_rewards))

                for i in range(self.num_agents):
                    self.agents[i].store_transition(
                        states[i]["wave"],
                        states[i]["wait"],
                        fp_list[i],
                        actions[i],
                        spatial_rewards[i],
                        values_list[i],
                        probs_list[i],
                    )

                if (step + 1) % self.batch_size == 0:
                    update_id += 1
                    batch_losses = []
                    for i in range(self.num_agents):
                        fps_next = self._get_neighbor_fingerprints(i)
                        _, _, v_next = self.agents[i].get_action(
                            next_states[i]["wave"],
                            next_states[i]["wait"],
                            fps_next,
                        )
                        loss = self.agents[i].train(v_next, done=False)
                        batch_losses.append(loss)
                    ep_losses.extend(batch_losses)
                    aq, qtot, qmax, aw = network_approach_lane_means(self.tl_ids)
                    training_log.append(
                        {
                            "update": update_id,
                            "episode": ep + 1,
                            "total_steps": (ep * self.steps_per_episode) + step + 1,
                            "avg_loss": float(np.mean(batch_losses)),
                            "avg_reward": float(np.mean(spatial_rewards)),
                            "avg_wait": aw,
                            "avg_queue": aq,
                            "avg_queue_total": qtot,
                            "avg_queue_max": qmax,
                            "tp_ratio": tally.ratio(),
                            "arrival_rate": episode_arrival_rate(
                                tally, (ep * self.steps_per_episode) + step + 1
                            ),
                        }
                    )

                states = next_states

            aq, qtot, qmax, aw = network_approach_lane_means(self.tl_ids)
            by_round.append(
                {
                    "round": ep + 1,
                    "avg_reward": episode_reward / max(1, self.steps_per_episode),
                    "avg_wait": aw,
                    "avg_queue": aq,
                    "avg_queue_total": qtot,
                    "avg_queue_max": qmax,
                    "tp_ratio": tally.ratio(),
                    "arrival_rate": episode_arrival_rate(tally, self.steps_per_episode),
                    "avg_loss": float(np.mean(ep_losses))
                    if ep_losses
                    else 0.0,
                }
            )

            traci.close()
            arr = episode_arrival_rate(tally, self.steps_per_episode)
            print(
                f"  Episode {ep + 1}/{self.episodes}: "
                f"spatial_reward/step={episode_reward / self.steps_per_episode:.4f}  "
                f"TP={tally.ratio():.4f}  Arr/s={arr:.4f}"
            )

        for i, agent in enumerate(self.agents):
            agent.save_model(os.path.join(self.results_dir, f"agent_{i}.pt"))

        with open(
            os.path.join(self.results_dir, "multi_agent_ac_training.json"), "w"
        ) as f:
            json.dump(training_log, f, indent=2)
        with open(
            os.path.join(self.results_dir, "multi_agent_ac_by_round.json"), "w"
        ) as f:
            json.dump(by_round, f, indent=2)

        print(f"Training complete → {self.results_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", type=str, required=True)
    p.add_argument("--nodes", type=int, default=6)
    p.add_argument("--rounds", type=int, default=10, help="Episodes (= FL rounds for comparison)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.results_dir or os.path.join(_ROOT, "results", "multi_agent_ac")
    trainer = MA2CTrainer(
        args.sumocfg,
        num_agents=args.nodes,
        batch_size=args.batch_size,
        episodes=args.rounds,
        steps_per_episode=args.steps,
        results_dir=out,
        seed=args.seed,
    )
    trainer.train()
