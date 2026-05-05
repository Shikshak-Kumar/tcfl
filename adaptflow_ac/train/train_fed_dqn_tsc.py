import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import traci

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.fed_dqn_tsc_agent import FedDQNTscAgent
from env.fed_dqn_tsc_env import FedDQNTscEnv
from utils.traci_network_metrics import (
    EpisodeThroughputTally,
    episode_arrival_rate,
    network_approach_lane_means,
)
from utils.traci_tls_phases import set_phase_from_discrete_action


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _episode_metrics(
    tl_ids: List[str], tally: EpisodeThroughputTally, n_steps: int
) -> Dict[str, float]:
    aq, q_tot, q_max, aw = network_approach_lane_means(tl_ids)
    return {
        "avg_queue": float(aq),
        "avg_queue_total": float(q_tot),
        "avg_queue_max": float(q_max),
        "avg_wait": float(aw),
        "avg_tp_ratio": float(tally.ratio()),
        "avg_arrival_rate": float(episode_arrival_rate(tally, n_steps)),
    }


class FedDQNTscTrainer:
    def __init__(
        self,
        sumocfg_path: str,
        num_nodes: int = 6,
        rounds: int = 10,
        episodes_per_round: int = 1,
        steps_per_episode: int = 500,
        results_dir: str = "results/fed_dqn_tsc",
        skip_finetune: bool = True,
        seed: int = 42,
    ):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.num_nodes = num_nodes
        self.rounds = rounds
        self.episodes_per_round = episodes_per_round
        self.steps_per_episode = steps_per_episode
        self.results_dir = results_dir
        self.skip_finetune = skip_finetune
        _set_seed(seed)

        self.env_wrapper = FedDQNTscEnv(self.sumocfg_path, sigma=0.1)
        self.agents = [FedDQNTscAgent(i) for i in range(num_nodes)]
        self.tl_ids: List[str] = []

    def aggregate_fed_avg(self) -> None:
        print("\n[Federated Learning] Performing FedAvg aggregation...")
        global_weights_list = [agent.get_weights() for agent in self.agents]
        aggregated_weights = {}
        for key in global_weights_list[0].keys():
            aggregated_weights[key] = torch.stack(
                [w[key] for w in global_weights_list]
            ).mean(dim=0)
        for agent in self.agents:
            agent.set_weights(aggregated_weights)

    def train(self) -> None:
        print("FedDQN-TSC — SUMO (Scientific Reports 2023 style)")
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
        self.tl_ids = list(traci.trafficlight.getIDList())
        traci.close()
        if len(self.tl_ids) < self.num_nodes:
            print(
                f"Warning: {len(self.tl_ids)} TLS found; using num_nodes={len(self.tl_ids)}"
            )
            self.num_nodes = len(self.tl_ids)
        self.agents = self.agents[: self.num_nodes]

        all_rounds: List[Dict[str, Any]] = []
        total_episodes = 0

        for r in range(self.rounds):
            print(f"\n--- Round {r + 1}/{self.rounds} ---")
            round_losses: List[float] = []
            round_rewards_ep: List[float] = []
            round_wait_ep: List[float] = []
            round_queue_ep: List[float] = []
            round_queue_total_ep: List[float] = []
            round_queue_max_ep: List[float] = []
            round_tp_ep: List[float] = []
            round_arr_ep: List[float] = []

            for _ep in range(self.episodes_per_round):
                total_episodes += 1
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
                episode_rewards = [0.0] * self.num_nodes
                losses_this_ep: List[float] = []
                states = [
                    self.env_wrapper.get_state(self.tl_ids[i])
                    for i in range(self.num_nodes)
                ]

                for step in range(self.steps_per_episode):
                    actions = [self.agents[i].get_action(states[i]) for i in range(self.num_nodes)]
                    for i, tl_id in enumerate(self.tl_ids[: self.num_nodes]):
                        set_phase_from_discrete_action(tl_id, actions[i], n_actions=4)
                    traci.simulationStep()
                    tally.step()

                    next_states = []
                    for i in range(self.num_nodes):
                        next_state = self.env_wrapper.get_state(self.tl_ids[i])
                        reward = self.env_wrapper.get_reward(self.tl_ids[i])
                        done = step == self.steps_per_episode - 1
                        self.agents[i].remember(states[i], actions[i], reward, next_state, done)
                        loss = self.agents[i].train(batch_size=32)
                        if loss:
                            losses_this_ep.append(loss)
                        episode_rewards[i] += reward
                        next_states.append(next_state)
                    states = next_states

                m = _episode_metrics(
                    self.tl_ids[: self.num_nodes], tally, self.steps_per_episode
                )
                avg_r = float(np.mean(episode_rewards))
                avg_loss = float(np.mean(losses_this_ep)) if losses_this_ep else 0.0
                round_rewards_ep.append(avg_r)
                round_wait_ep.append(m["avg_wait"])
                round_queue_ep.append(m["avg_queue"])
                round_queue_total_ep.append(m["avg_queue_total"])
                round_queue_max_ep.append(m["avg_queue_max"])
                round_tp_ep.append(m["avg_tp_ratio"])
                round_arr_ep.append(m["avg_arrival_rate"])
                round_losses.append(avg_loss)

                traci.close()
                print(
                    f"  Episode {total_episodes}: AvgReward={avg_r:.4f}  "
                    f"Loss={avg_loss:.6f}  Qμ={m['avg_queue']:.4f}  "
                    f"QΣ={m['avg_queue_total']:.2f}  Qmax={m['avg_queue_max']:.2f}  "
                    f"Wait={m['avg_wait']:.3f}  TP={m['avg_tp_ratio']:.4f}  "
                    f"Arr/s={m['avg_arrival_rate']:.4f}"
                )

                if total_episodes % 30 == 0:
                    print("  [Target Update] Syncing target networks...")
                    for agent in self.agents:
                        agent.update_target_network()

            round_record = {
                "round": r + 1,
                "avg_reward": float(np.mean(round_rewards_ep)),
                "avg_loss": float(np.mean(round_losses)),
                "avg_wait": float(np.mean(round_wait_ep)),
                "avg_queue": float(np.mean(round_queue_ep)),
                "avg_queue_total": float(np.mean(round_queue_total_ep)),
                "avg_queue_max": float(np.mean(round_queue_max_ep)),
                "avg_tp_ratio": float(np.mean(round_tp_ep)),
                "avg_arrival_rate": float(np.mean(round_arr_ep)),
                "rewards_per_episode": round_rewards_ep,
                "wait_per_episode": round_wait_ep,
                "tp_per_episode": round_tp_ep,
                "arrival_rate_per_episode": round_arr_ep,
                "episodes_per_round": self.episodes_per_round,
                "steps_per_episode": self.steps_per_episode,
                "mode": "SUMO-headless",
            }
            all_rounds.append(round_record)
            with open(
                os.path.join(self.results_dir, f"round_{r + 1}_summary.json"),
                "w",
            ) as f:
                json.dump(round_record, f, indent=2)

            self.aggregate_fed_avg()

        if not self.skip_finetune:
            print("\n--- Phase 2: Fine-Tuning (Local Layers Only) ---")
            for agent in self.agents:
                agent.start_fine_tuning()
            for ep in range(5):
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
                states = [
                    self.env_wrapper.get_state(self.tl_ids[i])
                    for i in range(self.num_nodes)
                ]
                for step in range(self.steps_per_episode):
                    actions = [
                        self.agents[i].get_action(states[i]) for i in range(self.num_nodes)
                    ]
                    for i, tl_id in enumerate(self.tl_ids[: self.num_nodes]):
                        set_phase_from_discrete_action(tl_id, actions[i], n_actions=4)
                    traci.simulationStep()
                    for i in range(self.num_nodes):
                        next_state = self.env_wrapper.get_state(self.tl_ids[i])
                        reward = self.env_wrapper.get_reward(self.tl_ids[i])
                        self.agents[i].remember(
                            states[i],
                            actions[i],
                            reward,
                            next_state,
                            step == self.steps_per_episode - 1,
                        )
                        self.agents[i].train()
                    states = [
                        self.env_wrapper.get_state(self.tl_ids[i])
                        for i in range(self.num_nodes)
                    ]
                traci.close()
                print(f"  Fine-tune Episode {ep + 1} complete.")

        for i, agent in enumerate(self.agents):
            agent.save_model(os.path.join(self.results_dir, f"agent_{i}.pt"))

        with open(
            os.path.join(self.results_dir, "fed_dqn_tsc_all_rounds.json"), "w"
        ) as f:
            json.dump(all_rounds, f, indent=2)

        print(f"\nTraining complete. Models and JSON → {self.results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=str, required=True)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes-per-round", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: results/fed_dqn_tsc next to cwd)",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Run extra fine-tuning phase (off by default for fair comparison).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = args.results_dir or os.path.join(_ROOT, "results", "fed_dqn_tsc")
    trainer = FedDQNTscTrainer(
        args.sumocfg,
        num_nodes=args.nodes,
        rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        steps_per_episode=args.steps,
        results_dir=out,
        skip_finetune=not args.fine_tune,
        seed=args.seed,
    )
    trainer.train()

