"""
Research Paper Comparison — Dwarka Mor (Delhi) Urban Map.

Reads training results from results/dwarka_mor/ and generates:
  1. Convergence curves (reward per round/update)
  2. Waiting-time comparison bar chart
  3. Throughput comparison bar chart
  4. Queue-length comparison bar chart
  5. Combined radar chart

Usage:
  python compare_dwarka_mor.py
  python compare_dwarka_mor.py --results-dir results/dwarka_mor
  python compare_dwarka_mor.py --out-dir results/dwarka_mor/plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Algorithm labels and colours (paper-quality) ─────────────────────────────
ALGOS = {
    "adaptflow":      {"label": "AdaptFlow-TSC (Ours)", "color": "#E63946",  "ls": "-",  "marker": "o"},
    "fed_dqn_tsc":    {"label": "FedDQN-TSC",            "color": "#457B9D",  "ls": "--", "marker": "s"},
    "multi_agent_ac": {"label": "MA2C",                  "color": "#2A9D8F",  "ls": ":",  "marker": "^"},
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Optional[Dict | List]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_adaptflow(results_dir: str) -> Dict:
    """Load AdaptFlow results from results_dir/adaptflow/."""
    base = os.path.join(results_dir, "adaptflow")
    data = _load_json(os.path.join(base, "adaptflow_all_rounds.json"))
    if data is None:
        return {}

    rewards, wait_times, throughputs, queues = [], [], [], []
    for r in data:
        nodes = r.get("nodes", {})
        round_rewards = [v.get("total_reward", 0) for v in nodes.values()]
        wait_vals = [v.get("avg_waiting_time", 0) for v in nodes.values()]
        tp_vals = [
            v.get("metrics", {}).get("throughput_ratio", 0) for v in nodes.values()
        ]
        q_vals = [
            v.get("metrics", {}).get("average_queue_length", 0)
            for v in nodes.values()
        ]
        rewards.append(float(np.mean(round_rewards)))
        wait_times.append(float(np.mean(wait_vals)))
        throughputs.append(float(np.mean(tp_vals)))
        queues.append(float(np.mean(q_vals)))

    return {
        "rewards": rewards,
        "avg_waiting_time": wait_times,
        "throughput_ratio": throughputs,
        "average_queue_length": queues,
    }


def load_fed_dqn_tsc(results_dir: str) -> Dict:
    """Load FedDQN-TSC results from results_dir/fed_dqn_tsc/."""
    base = os.path.join(results_dir, "fed_dqn_tsc")
    data = _load_json(os.path.join(base, "fed_dqn_tsc_all_rounds.json"))
    if data is None:
        return {}

    rewards, losses, wait_times, queues, throughputs = [], [], [], [], []
    for r in data:
        rewards.append(float(r.get("avg_reward", 0)))
        losses.append(float(r.get("avg_loss", 0)))
        wait_times.append(float(r.get("avg_wait", 0)))
        queues.append(float(r.get("avg_queue", 0)))
        throughputs.append(float(r.get("avg_tp_ratio", 0)))

    return {
        "rewards": rewards,
        "losses": losses,
        "avg_waiting_time": wait_times,
        "average_queue_length": queues,
        "throughput_ratio": throughputs,
    }


def load_multi_agent_ac(results_dir: str) -> Dict:
    """Load MA2C results from results_dir/multi_agent_ac/."""
    base = os.path.join(results_dir, "multi_agent_ac")
    data = _load_json(os.path.join(base, "multi_agent_ac_training.json"))
    if data is None:
        return {}

    rewards, c_losses, a_losses = [], [], []
    wait_times, queues, throughputs = [], [], []
    for u in data:
        rewards.append(float(u.get("avg_reward", 0)))
        c_losses.append(float(u.get("avg_critic_loss", 0)))
        a_losses.append(float(u.get("avg_actor_loss", 0)))
        wait_times.append(float(u.get("avg_wait", 0)))
        queues.append(float(u.get("avg_queue", 0)))
        throughputs.append(float(u.get("tp_ratio", 0)))

    # Downsample to ~same resolution as round-based methods
    step = max(1, len(rewards) // 20)

    return {
        "rewards": rewards[::step],
        "critic_losses": c_losses[::step],
        "avg_waiting_time": wait_times,
        "average_queue_length": queues,
        "throughput_ratio": throughputs,
    }


# ── Paper-quality plot helpers ─────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_convergence(data: Dict[str, Dict], out_path: str):
    """Reward convergence curves for all algorithms."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, cfg in ALGOS.items():
        d = data.get(key, {})
        rewards = d.get("rewards", [])
        if not rewards:
            continue
        x = np.arange(1, len(rewards) + 1)
        ax.plot(x, rewards, label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], marker=cfg["marker"],
                linewidth=2.0, markersize=5, markevery=max(1, len(x)//10))

    ax.set_xlabel("Training Round / Update Block")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Convergence — Dwarka Mor (Delhi) Urban Map")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_bar_metric(values: Dict[str, float], ylabel: str,
                    title: str, out_path: str, lower_is_better: bool = True):
    """Single metric bar chart with annotations."""
    _style()
    keys = [k for k in ALGOS if k in values]
    if not keys:
        return
    vals = [values[k] for k in keys]
    colors = [ALGOS[k]["color"] for k in keys]
    labels = [ALGOS[k]["label"] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.55)

    # Annotate bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # Highlight best
    best_idx = np.argmin(vals) if lower_is_better else np.argmax(vals)
    bars[best_idx].set_edgecolor("#333333")
    bars[best_idx].set_linewidth(2.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_radar(summary: Dict[str, Dict[str, float]], out_path: str):
    """Radar/spider chart for multi-metric comparison."""
    _style()
    metrics = ["Reward (norm)", "Throughput", "Low Wait (norm)", "Low Queue (norm)"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)

    for key, cfg in ALGOS.items():
        vals_raw = summary.get(key, {})
        if not vals_raw:
            continue
        vals = [
            vals_raw.get("reward_norm", 0),
            vals_raw.get("throughput_ratio", 0),
            vals_raw.get("wait_inv_norm", 0),
            vals_raw.get("queue_inv_norm", 0),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, label=cfg["label"], color=cfg["color"],
                linewidth=2.0, linestyle=cfg["ls"])
        ax.fill(angles, vals, alpha=0.10, color=cfg["color"])

    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    ax.set_title("Multi-Metric Comparison\n(Dwarka Mor Delhi Urban)", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(results_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Loading results from: {results_dir}")

    # Load results
    af_data  = load_adaptflow(results_dir)
    fd_data  = load_fed_dqn_tsc(results_dir)
    ma_data  = load_multi_agent_ac(results_dir)

    if not any([af_data, fd_data, ma_data]):
        print("  No result files found — run train_dwarka_mor.ps1 first.")
        return

    algo_data = {
        "adaptflow": af_data,
        "fed_dqn_tsc": fd_data,
        "multi_agent_ac": ma_data,
    }

    print(f"\n  Generating plots in: {out_dir}")

    # 1. Convergence
    plot_convergence(algo_data, os.path.join(out_dir, "convergence.png"))

    # 2. Final-round performance metrics from AdaptFlow (has full metrics)
    # For baselines, we use proxy estimates from reward
    wait_values: Dict[str, float] = {}
    tp_values: Dict[str, float] = {}
    queue_values: Dict[str, float] = {}

    # Use actual measured metrics from training logs (last 3 rounds/updates)
    if af_data.get("avg_waiting_time"):
        wait_values["adaptflow"]  = float(np.mean(af_data["avg_waiting_time"][-3:]))
        tp_values["adaptflow"]    = float(np.mean(af_data["throughput_ratio"][-3:]))
        queue_values["adaptflow"] = float(np.mean(af_data["average_queue_length"][-3:]))

    if fd_data.get("avg_waiting_time"):
        wait_values["fed_dqn_tsc"]  = float(np.mean(fd_data["avg_waiting_time"][-3:]))
        tp_values["fed_dqn_tsc"]    = float(np.mean(fd_data["throughput_ratio"][-3:]))
        queue_values["fed_dqn_tsc"] = float(np.mean(fd_data["average_queue_length"][-3:]))

    if ma_data.get("avg_waiting_time"):
        # Use last 10% of updates for stable final-performance estimate
        tail = max(1, len(ma_data["avg_waiting_time"]) // 10)
        wait_values["multi_agent_ac"]  = float(np.mean(ma_data["avg_waiting_time"][-tail:]))
        tp_values["multi_agent_ac"]    = float(np.mean(ma_data["throughput_ratio"][-tail:]))
        queue_values["multi_agent_ac"] = float(np.mean(ma_data["average_queue_length"][-tail:]))

    if wait_values:
        plot_bar_metric(wait_values,
                        "Avg Waiting Time per Vehicle (s)",
                        "Waiting Time — Dwarka Mor Urban",
                        os.path.join(out_dir, "waiting_time.png"),
                        lower_is_better=True)

    if tp_values:
        plot_bar_metric(tp_values,
                        "Throughput Ratio",
                        "Vehicle Throughput — Dwarka Mor Urban",
                        os.path.join(out_dir, "throughput.png"),
                        lower_is_better=False)

    if queue_values:
        plot_bar_metric(queue_values,
                        "Average Queue Length (vehicles)",
                        "Queue Length — Dwarka Mor Urban",
                        os.path.join(out_dir, "queue_length.png"),
                        lower_is_better=True)

    # 3. Radar chart
    # Normalise metrics to [0,1] for radar
    all_waits  = list(wait_values.values())
    all_tps    = list(tp_values.values())
    all_queues = list(queue_values.values())

    all_rewards = {
        "adaptflow":      float(np.mean(af_data["rewards"][-3:])) if af_data.get("rewards") else 0,
        "fed_dqn_tsc":    float(np.mean(fd_data["rewards"][-3:])) if fd_data.get("rewards") else 0,
        "multi_agent_ac": float(np.mean(ma_data["rewards"][-3:])) if ma_data.get("rewards") else 0,
    }

    r_min, r_max   = min(all_rewards.values()), max(all_rewards.values()) + 1e-9
    w_min, w_max   = min(all_waits) if all_waits else 0, max(all_waits) + 1e-9 if all_waits else 1
    q_min, q_max   = min(all_queues) if all_queues else 0, max(all_queues) + 1e-9 if all_queues else 1
    t_min, t_max   = min(all_tps) if all_tps else 0, max(all_tps) + 1e-9 if all_tps else 1

    radar_summary: Dict[str, Dict[str, float]] = {}
    for key in ALGOS:
        radar_summary[key] = {
            "reward_norm":    (all_rewards.get(key, 0) - r_min) / (r_max - r_min),
            "throughput_ratio": (tp_values.get(key, 0) - t_min) / (t_max - t_min),
            "wait_inv_norm":  1.0 - (wait_values.get(key, 0) - w_min) / (w_max - w_min),
            "queue_inv_norm": 1.0 - (queue_values.get(key, 0) - q_min) / (q_max - q_min),
        }

    plot_radar(radar_summary, os.path.join(out_dir, "radar.png"))

    # 4. Save summary JSON
    summary = {
        "map": "Dwarka Mor (Delhi) Urban OSM",
        "metrics": {
            "avg_waiting_time_s": wait_values,
            "throughput_ratio":   tp_values,
            "avg_queue_length":   queue_values,
            "final_avg_reward":   {k: round(v, 4) for k, v in all_rewards.items()},
        },
        "winner": {
            "waiting_time": min(wait_values, key=wait_values.get) if wait_values else "N/A",
            "throughput":   max(tp_values, key=tp_values.get) if tp_values else "N/A",
            "queue_length": min(queue_values, key=queue_values.get) if queue_values else "N/A",
        },
    }
    summary_path = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n  {'='*55}")
    print(f"  COMPARISON COMPLETE — plots saved to {out_dir}")
    print(f"  {'='*55}")
    if summary["winner"]["waiting_time"]:
        w = summary["winner"]["waiting_time"]
        print(f"  Best waiting time  : {ALGOS[w]['label'] if w in ALGOS else w}")
    if summary["winner"]["throughput"]:
        t = summary["winner"]["throughput"]
        print(f"  Best throughput    : {ALGOS[t]['label'] if t in ALGOS else t}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwarka Mor comparison plots")
    parser.add_argument("--results-dir", default=None,
                        help="Base results dir (default: results/dwarka_mor/ relative to this script)")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for plots (default: results/dwarka_mor/plots/)")
    args = parser.parse_args()

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(_this_dir, "results", "dwarka_mor")
    out_dir = args.out_dir or os.path.join(results_dir, "plots")

    main(results_dir, out_dir)
