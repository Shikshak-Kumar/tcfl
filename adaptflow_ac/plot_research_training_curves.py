"""
Line plots for research comparison: reward, loss, queue, waiting time, throughput.

Throughput panel uses **arrival_rate** (vehicles reaching destination per simulation second) when
present in logs; it varies with policy. Legacy **throughput_ratio** (arrived/departed) is often ~0.5
and is kept in JSON for reference only.

Reads training_manifest.json + per-algorithm JSON produced by run_research_sumo_configs2.py.

Usage:
  python plot_research_training_curves.py --results-dir results_research_sumo_configs2_*
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ALGOS = [
    ("adaptflow", "AdaptFlow-TSC", "#E63946", "-", "o"),
    ("fed_dqn_tsc", "FedDQN-TSC", "#457B9D", "--", "s"),
    ("multi_agent_ac", "MA2C", "#2A9D8F", ":", "^"),
    ("dqtsca", "DQTSCA", "#9B59B6", "-.", "D"),
]


def _load(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def series_adaptflow(adir: str) -> Dict[str, List[float]]:
    path = os.path.join(adir, "adaptflow_all_rounds.json")
    data = _load(path)
    if not data:
        return {}
    rewards, queues, waits, tps, losses = [], [], [], [], []
    for r in data:
        nodes = r.get("nodes", {})
        if not nodes:
            continue
        rewards.append(float(np.mean([v.get("total_reward", 0) for v in nodes.values()])))
        queues.append(
            float(
                np.mean(
                    [
                        float(
                            v.get("metrics", {}).get(
                                "queue_total_halting",
                                v.get("metrics", {}).get("average_queue_length", 0),
                            )
                        )
                        for v in nodes.values()
                    ]
                )
            )
        )
        waits.append(float(np.mean([v.get("avg_waiting_time", 0) for v in nodes.values()])))
        tps.append(
            float(
                np.mean(
                    [
                        float(
                            v.get("metrics", {}).get(
                                "arrival_rate",
                                v.get("metrics", {}).get("throughput_ratio", 0),
                            )
                        )
                        for v in nodes.values()
                    ]
                )
            )
        )
        losses.append(float(np.mean([v.get("loss", 0) for v in nodes.values()])))
    return {
        "x": list(range(1, len(rewards) + 1)),
        "reward": rewards,
        "queue": queues,
        "wait": waits,
        "tp": tps,
        "loss": losses,
    }


def series_fed_dqn(fdir: str) -> Dict[str, List[float]]:
    data = _load(os.path.join(fdir, "fed_dqn_tsc_all_rounds.json"))
    if not data:
        return {}
    xs, rewards, queues, waits, tps, losses = [], [], [], [], [], []
    for row in data:
        xs.append(int(row.get("round", len(xs) + 1)))
        rewards.append(float(row.get("avg_reward", 0)))
        queues.append(
            float(row.get("avg_queue_total", row.get("avg_queue", 0)))
        )
        waits.append(float(row.get("avg_wait", 0)))
        tps.append(float(row.get("avg_arrival_rate", row.get("avg_tp_ratio", 0))))
        losses.append(float(row.get("avg_loss", 0)))
    return {"x": xs, "reward": rewards, "queue": queues, "wait": waits, "tp": tps, "loss": losses}


def series_ma2c(mdir: str) -> Dict[str, List[float]]:
    path = os.path.join(mdir, "multi_agent_ac_by_round.json")
    data = _load(path)
    if not data:
        return {}
    xs, rewards, queues, waits, tps, losses = [], [], [], [], [], []
    for row in data:
        xs.append(int(row.get("round", len(xs) + 1)))
        rewards.append(float(row.get("avg_reward", 0)))
        queues.append(
            float(row.get("avg_queue_total", row.get("avg_queue", 0)))
        )
        waits.append(float(row.get("avg_wait", 0)))
        tps.append(float(row.get("arrival_rate", row.get("tp_ratio", 0))))
        losses.append(float(row.get("avg_loss", 0)))
    return {"x": xs, "reward": rewards, "queue": queues, "wait": waits, "tp": tps, "loss": losses}


def series_dqtsca(ddir: str) -> Dict[str, List[float]]:
    data = _load(os.path.join(ddir, "dqtsca_training.json"))
    if not data:
        return {}
    xs, rewards, queues, waits, tps, losses = [], [], [], [], [], []
    for row in data:
        xs.append(int(row.get("episode", len(xs) + 1)))
        rewards.append(float(row.get("total_reward", 0)))
        queues.append(
            float(row.get("avg_queue_total", row.get("avg_queue", 0)))
        )
        waits.append(float(row.get("avg_wait", 0)))
        tps.append(float(row.get("arrival_rate", row.get("tp_ratio", 0))))
        losses.append(float(row.get("avg_loss", 0)))
    return {"x": xs, "reward": rewards, "queue": queues, "wait": waits, "tp": tps, "loss": losses}


def _plot_panel(
    ax: plt.Axes,
    series_map: Dict[str, Dict[str, List[float]]],
    key: str,
    ylabel: str,
    title: str,
) -> None:
    for aid, label, color, ls, mk in _ALGOS:
        s = series_map.get(aid)
        if not s or not s.get(key):
            continue
        x, y = s["x"], s[key]
        ax.plot(x, y, label=label, color=color, linestyle=ls, marker=mk, markevery=max(1, len(x) // 8))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Round / episode index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, framealpha=0.85)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    base = os.path.abspath(args.results_dir)
    out = os.path.abspath(args.out_dir or os.path.join(base, "plots_training"))
    os.makedirs(out, exist_ok=True)

    sub = {
        "adaptflow": os.path.join(base, "adaptflow"),
        "fed_dqn_tsc": os.path.join(base, "fed_dqn_tsc"),
        "multi_agent_ac": os.path.join(base, "multi_agent_ac"),
        "dqtsca": os.path.join(base, "dqtsca"),
    }
    series_map = {
        "adaptflow": series_adaptflow(sub["adaptflow"]),
        "fed_dqn_tsc": series_fed_dqn(sub["fed_dqn_tsc"]),
        "multi_agent_ac": series_ma2c(sub["multi_agent_ac"]),
        "dqtsca": series_dqtsca(sub["dqtsca"]),
    }

    manifest_path = os.path.join(base, "training_manifest.json")
    title_suffix = ""
    if os.path.isfile(manifest_path):
        man = _load(manifest_path)
        if isinstance(man, dict):
            hp = man.get("hyperparameters", {})
            title_suffix = (
                f" (rounds={hp.get('rounds')}, steps={hp.get('steps_per_episode')}, "
                f"nodes={hp.get('nodes')}, seed={hp.get('seed')})"
            )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    _plot_panel(axes[0], series_map, "reward", "Reward", "Episode / round reward" + title_suffix)
    _plot_panel(axes[1], series_map, "loss", "Loss", "Training loss")
    _plot_panel(
        axes[2],
        series_map,
        "queue",
        "Queue (mean QΣ per TLS, veh)",
        "Queue length",
    )
    _plot_panel(axes[3], series_map, "wait", "Avg wait (s)", "Waiting time")
    _plot_panel(
        axes[4],
        series_map,
        "tp",
        "Arrival rate (veh / sim s)",
        "Throughput (arrival rate)",
    )
    axes[5].axis("off")
    fig.tight_layout()
    combined = os.path.join(out, "training_curves_combined.png")
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined}")

    singles = [
        ("reward", "Reward"),
        ("loss", "Loss"),
        ("queue", "Queue (mean QΣ per TLS)"),
        ("wait", "Waiting time"),
        ("tp", "Arrival rate (veh / sim s)"),
    ]
    for key, name in singles:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _plot_panel(ax, series_map, key, name, f"{name} — comparison{title_suffix}")
        fig.tight_layout()
        fp = os.path.join(out, f"line_{key}.png")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fp}")


if __name__ == "__main__":
    main()
