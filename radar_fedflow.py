"""Radar chart for FedFlow results across rounds."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Helvetica"]

with open("results_fedflow/fedflow_all_rounds.json") as f:
    data = json.load(f)

# ── Aggregate per-round averages ──────────────────────────────────────
rounds_summary = []
for rd in data:
    nodes = rd["nodes"]
    metrics_list = [n["metrics"] for n in nodes.values()]

    avg_reward = np.mean([n["total_reward"] for n in nodes.values()])
    avg_wait = np.mean([m.get("avg_waiting_time_per_vehicle", 0) for m in metrics_list])
    avg_tp_ratio = np.mean([m.get("throughput_ratio", 0) for m in metrics_list])
    avg_queue = np.mean([m.get("average_queue_length", 0) for m in metrics_list])
    avg_loss = np.mean([n["loss"] for n in nodes.values()])
    # Invert congested lanes so higher = better
    avg_congested = np.mean(
        [m.get("lane_summary", {}).get("num_congested_lanes", 0) for m in metrics_list]
    )

    rounds_summary.append(
        {
            "round": rd["round"],
            "reward": avg_reward,
            "wait": avg_wait,
            "tp_ratio": avg_tp_ratio,
            "queue": avg_queue,
            "loss": avg_loss,
            "congested": avg_congested,
        }
    )

# ── Define radar dimensions ───────────────────────────────────────────
# We normalise each metric to [0, 1] where 1 = best.
categories = [
    "Reward\n(higher=better)",
    "Low Wait Time\n(lower=better)",
    "Throughput\nRatio",
    "Low Queue\n(lower=better)",
    "Learning\n(low loss)",
    "No Congestion\n(0 lanes)",
]
N = len(categories)


def normalise(rounds_summary):
    """Return list of normalised value-arrays, one per round."""
    result = []
    for rs in rounds_summary:
        vals = [
            rs["reward"] / 200.0,  # max possible ~200
            1.0 - min(rs["wait"] / 10.0, 1.0),  # invert: lower wait = better
            min(rs["tp_ratio"] / 0.2, 1.0),  # scale tp_ratio
            1.0 - min(rs["queue"] / 5.0, 1.0),  # invert: lower queue = better
            1.0 - min(rs["loss"] / 1.0, 1.0),  # invert: lower loss = better
            1.0 - min(rs["congested"] / 5.0, 1.0),  # invert: 0 congested = best
        ]
        result.append(vals)
    return result


norm_data = normalise(rounds_summary)

# ── Radar plot ────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

colors = ["#58a6ff", "#3fb950", "#f0883e"]
fills = ["#58a6ff22", "#3fb95022", "#f0883e22"]

for i, (vals, rs) in enumerate(zip(norm_data, rounds_summary)):
    values = vals + vals[:1]
    ax.plot(
        angles,
        values,
        "o-",
        linewidth=2.2,
        color=colors[i],
        label=f"Round {rs['round']}",
        markersize=7,
        zorder=3,
    )
    ax.fill(angles, values, alpha=0.12, color=colors[i])

# Gridlines & labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, color="#c9d1d9", fontweight="600")
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#8b949e")
ax.set_ylim(0, 1.05)

# Style grid
ax.spines["polar"].set_color("#30363d")
ax.xaxis.grid(True, color="#30363d", linewidth=0.6)
ax.yaxis.grid(True, color="#30363d", linewidth=0.6)
ax.tick_params(axis="x", pad=18)

# Title
ax.set_title(
    "FedFlow-TSC Performance Radar\n(SUMO | 6 Nodes | 2 Clusters | 3 Rounds)",
    fontsize=16,
    fontweight="bold",
    color="#f0f6fc",
    pad=30,
)

# Legend
legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.28, 1.12),
    fontsize=11,
    frameon=True,
    fancybox=True,
    edgecolor="#30363d",
    facecolor="#161b22",
    labelcolor="#c9d1d9",
)

# ── Annotation box with raw values ───────────────────────────────────
info_lines = []
for rs in rounds_summary:
    info_lines.append(
        f"R{rs['round']}: Reward={rs['reward']:.1f}  "
        f"Wait={rs['wait']:.2f}s  "
        f"TP={rs['tp_ratio']:.4f}  "
        f"Loss={rs['loss']:.4f}"
    )
info_text = "\n".join(info_lines)

fig.text(
    0.50,
    0.02,
    info_text,
    ha="center",
    va="bottom",
    fontsize=9.5,
    color="#8b949e",
    family="monospace",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d", alpha=0.9
    ),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(
    "results_fedflow/fedflow_radar.png",
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print("Saved → results_fedflow/fedflow_radar.png")
plt.show()
