"""
FedDQN-TSC Training Script.

Federated DQN Traffic Signal Control (Ye et al., Scientific Reports 2023).
Agents share global feature-extractor layers via FedAvg every 20 episodes;
local output layers remain agent-specific (personalised federated learning).

Usage:
  python train/train_fed_dqn_tsc.py --mode mock --rounds 5
  python train/train_fed_dqn_tsc.py --mode sumo --sumo-scenario dwarka_mor --real-sumo
  python train/train_fed_dqn_tsc.py --mode sumo --sumocfg /abs/path/to.sumocfg --real-sumo
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# ── path bootstrap (run from adaptflow_ac/ or its parent) ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # adaptflow_ac/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from env.fed_dqn_tsc_env import FedDQNTscEnv
from env.mock_env import MockEnv
from agents.fed_dqn_tsc_agent import FedDQNTscAgent


def _build_env(args):
    """Build the environment from CLI arguments."""
    if args.mode == "mock":
        env = MockEnv(num_intersections=4, state_mode="3d", max_steps=args.steps)
        return env

    # SUMO mode: resolve config path
    sumocfg = args.sumocfg
    if sumocfg is None:
        from utils.sumo_scenario import get_sumo_config_paths, effective_sumo_headless
        scenario = getattr(args, "sumo_scenario", None)
        paths = get_sumo_config_paths(scenario)
        sumocfg = paths[0] if paths else None

    if sumocfg is None:
        raise ValueError("SUMO mode requires --sumocfg or --sumo-scenario.")

    gui = getattr(args, "gui", False)
    env = FedDQNTscEnv(sumocfg, gui=gui, max_steps=args.steps)
    return env


def _convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(i) for i in obj]
    return obj


def _build_env_for_round(args, round_idx: int):
    """
    Rotate through all 6 node configs across rounds so FedDQN sees the same
    diversity of traffic patterns (time windows, congestion levels) as AdaptFlow.
    Round i → config[i % n_configs].  Mock mode is unaffected.
    """
    if args.mode == "mock":
        return _build_env(args)

    from utils.sumo_scenario import get_sumo_config_paths
    paths = get_sumo_config_paths(getattr(args, "sumo_scenario", None))
    if not paths:
        return _build_env(args)

    # Cycle through node configs: round 0→node0, round 1→node1, …, round 6→node0, …
    config_path = paths[round_idx % len(paths)]
    gui = getattr(args, "gui", False)
    from env.fed_dqn_tsc_env import FedDQNTscEnv
    print(f"  [Config] Round {round_idx + 1} using: {os.path.basename(config_path)}")
    return FedDQNTscEnv(config_path, gui=gui, max_steps=args.steps)


def train_federated(args):
    os.makedirs(args.results_dir, exist_ok=True)

    # Build initial env for agent shape discovery
    env = _build_env(args)
    num_nodes = env.num_intersections

    input_dim = env.n_features * env.max_lanes * env.max_forks

    agents: List[FedDQNTscAgent] = []
    for i, tls_id in enumerate(env.tls_ids):
        action_dim = env.tls_info[tls_id]["num_phases"]
        agents.append(FedDQNTscAgent(i, input_dim, action_dim))

    print(f"\n{'='*65}")
    print(f"  FedDQN-TSC Training  —  {num_nodes} intersections")
    print(f"  Rounds: {args.rounds}  |  Episodes/round: {args.episodes_per_round}")
    print(f"  Steps/episode: {args.steps}  |  Mode: {args.mode.upper()}")
    print(f"  Config rotation: all 6 node configs cycled across rounds (same diversity as AdaptFlow)")
    print(f"  Results: {args.results_dir}/")
    print(f"{'='*65}\n")

    all_round_stats: List[Dict] = []
    avg_weights = None

    for round_idx in range(args.rounds):
        # Rotate SUMO config each round so FedDQN sees all 6 traffic patterns
        if args.mode == "sumo" and round_idx > 0:
            try:
                env.close()
            except Exception:
                pass
            env = _build_env_for_round(args, round_idx)

        print(f"\n--- FL Round {round_idx + 1}/{args.rounds} ---")
        round_rewards: List[float] = []
        round_losses: List[float] = []
        round_wait: List[float] = []
        round_queue: List[float] = []
        round_tp: List[float] = []

        for ep in range(args.episodes_per_round):
            states = env.reset()
            total_rewards = np.zeros(num_nodes)
            done = False
            ep_loss = 0.0
            ep_steps = 0

            while not done:
                actions = [agents[i].get_action(states[i]) for i in range(num_nodes)]
                next_states, rewards, done, _ = env.step(actions)

                for i in range(num_nodes):
                    agents[i].remember(
                        states[i], actions[i], float(rewards[i]),
                        next_states[i], done
                    )
                    loss = agents[i].train(batch_size=32)
                    ep_loss += loss

                states = next_states
                total_rewards += rewards
                ep_steps += 1

            # Collect per-episode metrics from env
            ep_stats = env.episode_stats() if hasattr(env, "episode_stats") else {}
            mean_ep_reward = float(np.mean(total_rewards) / max(1, ep_steps))
            ep_loss_val = ep_loss / max(1, ep_steps * num_nodes)
            ep_wait = ep_stats.get("avg_wait", 0.0)
            ep_queue = ep_stats.get("avg_queue", 0.0)
            ep_tp = ep_stats.get("tp_ratio", 0.0)
            ep_arrived = ep_stats.get("arrived", 0)

            round_rewards.append(mean_ep_reward)
            round_losses.append(ep_loss_val)
            round_wait.append(ep_wait)
            round_queue.append(ep_queue)
            round_tp.append(ep_tp)

            eps_val = agents[0].epsilon if agents else 0.0
            print(f"  Ep {ep + 1:3d}/{args.episodes_per_round}: "
                  f"Reward={mean_ep_reward:+.4f}  Loss={ep_loss_val:.6f}  "
                  f"Wait={ep_wait:.2f}s  Queue={ep_queue:.2f}  "
                  f"TP={ep_tp:.3f}  Arrived={ep_arrived}  eps={eps_val:.3f}")

            # Target network sync every 30 global episodes
            global_ep = round_idx * args.episodes_per_round + ep
            if (global_ep + 1) % 30 == 0:
                for a in agents:
                    a.update_target_network()

        # ── FedAvg aggregation (global layers only) ──────────────────────────
        print(f"  Aggregating global layers (FedAvg)...")
        global_weights_list = [a.get_global_weights() for a in agents]
        avg_weights = {}
        for key in global_weights_list[0].keys():
            avg_weights[key] = torch.stack(
                [w[key].float() for w in global_weights_list]
            ).mean(dim=0)
        for a in agents:
            a.set_global_weights(avg_weights)

        # ── save round summary ────────────────────────────────────────────────
        round_stats = {
            "round": round_idx + 1,
            "avg_reward": float(np.mean(round_rewards)),
            "avg_loss": float(np.mean(round_losses)),
            "avg_wait": float(np.mean(round_wait)),
            "avg_queue": float(np.mean(round_queue)),
            "avg_tp_ratio": float(np.mean(round_tp)),
            "rewards_per_episode": round_rewards,
            "wait_per_episode": round_wait,
            "tp_per_episode": round_tp,
        }
        all_round_stats.append(round_stats)
        print(f"\n  Round {round_idx + 1} Summary:")
        print(f"    AvgReward  = {round_stats['avg_reward']:+.4f}")
        print(f"    AvgLoss    = {round_stats['avg_loss']:.6f}")
        print(f"    AvgWait    = {round_stats['avg_wait']:.2f} s/lane")
        print(f"    AvgQueue   = {round_stats['avg_queue']:.2f} veh/lane")
        print(f"    TP Ratio   = {round_stats['avg_tp_ratio']:.3f}")

        round_file = os.path.join(args.results_dir, f"round_{round_idx + 1}_summary.json")
        with open(round_file, "w") as f:
            json.dump(_convert(round_stats), f, indent=2)

    env.close()

    # ── save global weights ───────────────────────────────────────────────────
    if avg_weights is not None:
        model_path = os.path.join(args.results_dir, "fed_dqn_tsc_global.pt")
        torch.save(avg_weights, model_path)
        print(f"\n  Global weights saved to {model_path}")

    # ── save all-rounds JSON ──────────────────────────────────────────────────
    final_path = os.path.join(args.results_dir, "fed_dqn_tsc_all_rounds.json")
    with open(final_path, "w") as f:
        json.dump(_convert(all_round_stats), f, indent=2)

    print(f"\n{'='*65}")
    print(f"  FedDQN-TSC TRAINING COMPLETE")
    print(f"  Results saved to {args.results_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedDQN-TSC Training")
    parser.add_argument("--mode", type=str, default="sumo",
                        choices=["mock", "sumo"],
                        help="'mock' for synthetic traffic, 'sumo' for real SUMO")
    parser.add_argument("--sumocfg", type=str, default=None,
                        help="Absolute path to .sumocfg (overrides --sumo-scenario)")
    parser.add_argument("--sumo-scenario", "--sumo_scenario",
                        type=str, default="dwarka_mor", dest="sumo_scenario",
                        help="Named scenario: dwarka_mor | china_osm | default …")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of federated rounds")
    parser.add_argument("--episodes-per-round", "--episodes_per_round",
                        type=int, default=20, dest="episodes_per_round",
                        help="Local training episodes before each FedAvg (default 20)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Simulation steps per episode")
    parser.add_argument("--gui", action="store_true",
                        help="Open sumo-gui (GUI mode)")
    parser.add_argument("--real-sumo", "--sumo-headless",
                        action="store_true", dest="real_sumo",
                        help="Run headless SUMO (no GUI)")
    parser.add_argument("--results-dir", "--results_dir",
                        type=str, default=None, dest="results_dir",
                        help="Output directory for results (default: auto-named by scenario)")

    args = parser.parse_args()

    # Resolve results directory — use scenario slug so each map gets its own folder
    if args.results_dir is None:
        from utils.sumo_scenario import normalize_scenario
        scenario_slug = normalize_scenario(getattr(args, "sumo_scenario", None) or "dwarka_mor")
        args.results_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "results", scenario_slug, "fed_dqn_tsc"
        ))

    # For SUMO mode without explicit sumocfg, resolve via sumo_scenario
    if args.mode == "sumo" and args.sumocfg is None:
        from utils.sumo_scenario import get_sumo_config_paths, effective_sumo_headless
        paths = get_sumo_config_paths(args.sumo_scenario)
        if paths:
            # Use node 0 config (FedDQN runs a single multi-intersection sim)
            args.sumocfg = paths[0]
            print(f"[FedDQN-TSC] Using SUMO config: {args.sumocfg}")

    train_federated(args)
