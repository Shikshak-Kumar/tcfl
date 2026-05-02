"""
Multi-Agent Actor-Critic (MA2C) Training Script.

Based on Chu et al. "Multi-Agent Deep Reinforcement Learning for Large-Scale
Traffic Signal Control", IEEE TITS 2019.

Key features:
  - LSTM-based actor and critic per agent
  - Neighbour fingerprints for non-stationarity mitigation
  - Spatial reward discounting: r̃ᵢ = rᵢ + α·Σ_j r_j  (α = 0.75)
  - Entropy regularisation for exploration

Usage:
  python train/train_multi_agent_ac.py --mode mock --steps 500
  python train/train_multi_agent_ac.py --mode sumo --sumo-scenario dwarka_mor --real-sumo
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

# ── path bootstrap ─────────────────────────────────────────────────────────────
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

from env.multi_agent_ac_env import MultiAgentACEnv
from env.mock_env import MockEnv
from agents.multi_agent_ac_agent import MultiAgentACAgent


def _build_env(args):
    """Build the environment from CLI arguments."""
    if args.mode == "mock":
        return MockEnv(num_intersections=4, state_mode="dict", max_steps=args.steps)

    sumocfg = args.sumocfg
    if sumocfg is None:
        from utils.sumo_scenario import get_sumo_config_paths
        paths = get_sumo_config_paths(args.sumo_scenario)
        sumocfg = paths[0] if paths else None

    if sumocfg is None:
        raise ValueError("SUMO mode requires --sumocfg or --sumo-scenario.")

    gui = getattr(args, "gui", False)
    return MultiAgentACEnv(sumocfg, gui=gui, max_steps=args.steps)


def get_spatial_reward(
    tls_id: str,
    rewards_dict: Dict[str, float],
    neighbors_map: Dict[str, List[str]],
    alpha: float = 0.75,
) -> float:
    """Spatial discount: r̃ᵢ = rᵢ + α·Σ_j∈N(i) rⱼ  (1-hop, Chu et al. 2019)."""
    total = rewards_dict.get(tls_id, 0.0)
    for nb_id in neighbors_map.get(tls_id, []):
        total += alpha * rewards_dict.get(nb_id, 0.0)
    return total


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


def train_multi_agent_ac(args):
    os.makedirs(args.results_dir, exist_ok=True)

    env = _build_env(args)

    # Build agents
    agents: Dict[str, MultiAgentACAgent] = {}
    for tls_id in env.tls_ids:
        info = env.tls_info[tls_id]
        wave_dim = len(info["lanes"])
        wait_dim = wave_dim
        action_dim = info["action_dim"]
        # fingerprint dim = sum of action_dims of all neighbours
        fp_dim = sum(
            env.tls_info[nb]["action_dim"]
            for nb in env.neighbors.get(tls_id, [])
        )
        agents[tls_id] = MultiAgentACAgent(
            tls_id, wave_dim, wait_dim, fp_dim, action_dim
        )

    print(f"\n{'='*65}")
    print(f"  MA2C Training  —  {env.num_agents} agents")
    print(f"  Total steps target: {args.max_total_steps:,}")
    print(f"  Batch size: {args.batch_size}  |  Spatial alpha: {args.alpha}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Results: {args.results_dir}/")
    print(f"{'='*65}\n")

    total_steps = 0
    update_count = 0
    all_update_stats: List[Dict] = []

    while total_steps < args.max_total_steps:
        minibatch: Dict[str, List] = {tid: [] for tid in env.tls_ids}

        obs = env.reset()
        for a in agents.values():
            a.reset_hidden()

        episode_rewards = {tid: 0.0 for tid in env.tls_ids}
        batch_steps = 0

        for _ in range(args.batch_size):
            # Current fingerprints (policy simplexes)
            current_fp = {tid: agents[tid].fingerprint for tid in env.tls_ids}

            # Collect actions
            actions: List[int] = []
            for tls_id in env.tls_ids:
                nb_fps = [current_fp[nb] for nb in env.neighbors.get(tls_id, [])]
                act = agents[tls_id].get_action(
                    obs[tls_id]["wave"], obs[tls_id]["wait"], nb_fps
                )
                actions.append(act)

            next_obs, rewards_raw, done, _ = env.step(actions)

            # Rewards: dict or array → dict
            if isinstance(rewards_raw, np.ndarray):
                rewards_dict = {
                    tid: float(rewards_raw[i]) for i, tid in enumerate(env.tls_ids)
                }
            else:
                rewards_dict = {k: float(v) for k, v in rewards_raw.items()}

            # Spatial reward discounting
            spatial_rewards = {
                tid: get_spatial_reward(tid, rewards_dict, env.neighbors, args.alpha)
                for tid in env.tls_ids
            }

            # Store transitions
            for i, tls_id in enumerate(env.tls_ids):
                nb_list = env.neighbors.get(tls_id, [])
                fp = (
                    np.concatenate([current_fp[nb] for nb in nb_list])
                    if nb_list else np.zeros(0, dtype=np.float32)
                )
                next_fp = (
                    np.concatenate([agents[nb].fingerprint for nb in nb_list])
                    if nb_list else np.zeros(0, dtype=np.float32)
                )
                minibatch[tls_id].append((
                    obs[tls_id]["wave"],
                    obs[tls_id]["wait"],
                    fp,
                    actions[i],
                    spatial_rewards[tls_id],
                    next_obs[tls_id]["wave"],
                    next_obs[tls_id]["wait"],
                    next_fp,
                    done,
                ))
                episode_rewards[tls_id] += spatial_rewards[tls_id]

            obs = next_obs
            total_steps += 1
            batch_steps += 1
            if done:
                break

        # Collect episode stats before next reset
        ep_stats = env.episode_stats() if hasattr(env, "episode_stats") else {}
        ep_wait  = ep_stats.get("avg_wait", 0.0)
        ep_queue = ep_stats.get("avg_queue", 0.0)
        ep_tp    = ep_stats.get("tp_ratio", 0.0)
        ep_arr   = ep_stats.get("arrived", 0)

        # Simultaneous agent updates
        c_losses: List[float] = []
        a_losses: List[float] = []
        for tls_id in env.tls_ids:
            if minibatch[tls_id]:
                c_loss, a_loss = agents[tls_id].update(minibatch[tls_id])
                c_losses.append(c_loss)
                a_losses.append(a_loss)

        update_count += 1
        avg_c = float(np.mean(c_losses)) if c_losses else 0.0
        avg_a = float(np.mean(a_losses)) if a_losses else 0.0
        avg_r = float(np.mean(list(episode_rewards.values())))

        stat = {
            "update": update_count,
            "total_steps": total_steps,
            "avg_critic_loss": avg_c,
            "avg_actor_loss": avg_a,
            "avg_reward": avg_r,
            "avg_wait": ep_wait,
            "avg_queue": ep_queue,
            "tp_ratio": ep_tp,
            "arrived": ep_arr,
        }
        all_update_stats.append(stat)

        if update_count % 10 == 0 or total_steps >= args.max_total_steps:
            print(f"  Update {update_count:4d} | Steps {total_steps:7,} | "
                  f"Reward={avg_r:+.4f}  CriticL={avg_c:.6f}  ActorL={avg_a:.6f}  "
                  f"Wait={ep_wait:.2f}s  Queue={ep_queue:.2f}  "
                  f"TP={ep_tp:.3f}  Arrived={ep_arr}")

    env.close()

    # ── save models ───────────────────────────────────────────────────────────
    model_dir = os.path.join(args.results_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    for tls_id, agent in agents.items():
        safe_id = tls_id.replace("/", "_").replace(":", "_")
        agent_path = os.path.join(model_dir, f"{safe_id}.pt")
        import torch
        torch.save(
            {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
            agent_path,
        )

    # ── save training stats ───────────────────────────────────────────────────
    stats_path = os.path.join(args.results_dir, "multi_agent_ac_training.json")
    with open(stats_path, "w") as f:
        json.dump(_convert(all_update_stats), f, indent=2)

    print(f"\n{'='*65}")
    print(f"  MA2C TRAINING COMPLETE  |  {update_count} updates, {total_steps:,} steps")
    print(f"  Models saved to {model_dir}/")
    print(f"  Stats saved to {stats_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MA2C Multi-Agent AC Training")
    parser.add_argument("--mode", type=str, default="sumo",
                        choices=["mock", "sumo"],
                        help="'mock' for synthetic traffic, 'sumo' for real SUMO")
    parser.add_argument("--sumocfg", type=str, default=None,
                        help="Absolute path to .sumocfg (overrides --sumo-scenario)")
    parser.add_argument("--sumo-scenario", "--sumo_scenario",
                        type=str, default="dwarka_mor", dest="sumo_scenario",
                        help="Named scenario: dwarka_mor | china_osm | default …")
    parser.add_argument("--batch-size", "--batch_size",
                        type=int, default=120, dest="batch_size",
                        help="On-policy minibatch collection size (default 120)")
    parser.add_argument("--steps", type=int, default=720,
                        help="Episode horizon length in SUMO steps")
    parser.add_argument("--max-total-steps", "--max_total_steps",
                        type=int, default=50000, dest="max_total_steps",
                        help="Total training steps (default 50000 for research run)")
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Spatial discount factor α (default 0.75)")
    parser.add_argument("--gui", action="store_true",
                        help="Open sumo-gui (GUI mode)")
    parser.add_argument("--real-sumo", "--sumo-headless",
                        action="store_true", dest="real_sumo",
                        help="Run headless SUMO (no GUI)")
    parser.add_argument("--results-dir", "--results_dir",
                        type=str, default=None, dest="results_dir",
                        help="Output directory (default: auto-named by scenario)")

    args = parser.parse_args()

    # Resolve results directory
    if args.results_dir is None:
        args.results_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "results", "dwarka_mor", "multi_agent_ac"
        ))

    # Resolve SUMO config for SUMO mode
    if args.mode == "sumo" and args.sumocfg is None:
        from utils.sumo_scenario import get_sumo_config_paths
        paths = get_sumo_config_paths(args.sumo_scenario)
        if paths:
            args.sumocfg = paths[0]
            print(f"[MA2C] Using SUMO config: {args.sumocfg}")

    train_multi_agent_ac(args)
