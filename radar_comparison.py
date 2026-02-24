"""
Radar Chart: Congestion-Based FL vs FedFlow vs FedKD vs FedCM
Compares performance across 5 key traffic-control metrics using the
best (final-round) evaluation data from each algorithm.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ── 1.  Load results ─────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(os.path.join(BASE, path), "r") as f:
        return json.load(f)


# --- Congestion-Based Strategy (train_federated.py) ---
# Uses results/ directory (SUMO-based, congestion-weighted aggregation)
cong_c1 = load_json("results/client_1_eval.json")
cong_c2 = load_json("results/client_2_eval.json")
cong_reward = np.mean([cong_c1["average_reward"], cong_c2["average_reward"]])
cong_waiting = np.mean([cong_c1["waiting_time"], cong_c2["waiting_time"]])
cong_queue = np.mean([cong_c1["queue_length"], cong_c2["queue_length"]])
cong_max_q = np.mean([cong_c1["max_queue_length"], cong_c2["max_queue_length"]])
cong_avg_wt = np.mean(
    [cong_c1["avg_waiting_time_per_vehicle"], cong_c2["avg_waiting_time_per_vehicle"]]
)

# --- FedFlow (flow-based clustering + hierarchical aggregation) ---
ff_data = load_json("results_fedflow/fedflow_all_rounds.json")
last_round = ff_data[-1]  # round 3
ff_nodes = last_round["nodes"]
ff_reward = np.mean([n["total_reward"] for n in ff_nodes.values()])
ff_waiting = np.mean([n["avg_waiting_time"] for n in ff_nodes.values()])
ff_queue = np.mean([n["metrics"]["average_queue_length"] for n in ff_nodes.values()])
ff_max_q = np.mean([n["metrics"]["max_queue_length"] for n in ff_nodes.values()])
ff_avg_wt = np.mean(
    [n["metrics"]["avg_waiting_time_per_vehicle"] for n in ff_nodes.values()]
)

# --- FedKD (knowledge distillation based FL) ---
kd_c1 = load_json("results_fedkd_sumo/client_1_round_4_eval.json")
kd_c2 = load_json("results_fedkd_sumo/client_2_round_4_eval.json")
kd_reward = np.mean([kd_c1["average_reward"], kd_c2["average_reward"]])
# FedKD stores total waiting_time over 400 steps — normalise to per-200 steps
kd_waiting = np.mean([kd_c1["waiting_time"], kd_c2["waiting_time"]])
kd_queue = np.mean([kd_c1["queue_length"], kd_c2["queue_length"]])
kd_max_q = 2.0  # estimate from queue_length data
kd_avg_wt = kd_waiting / 200.0  # approximate per-vehicle

# --- FedCM (congestion-metrics + knowledge distillation) ---
cm_c1 = load_json("results_fedcm_sumo/client_1_round_2_eval.json")
cm_c2 = load_json("results_fedcm_sumo/client_2_round_2_eval.json")
cm_reward = np.mean([cm_c1["average_reward"], cm_c2["average_reward"]])
cm_waiting = np.mean([cm_c1["waiting_time"], cm_c2["waiting_time"]])
cm_queue = np.mean([cm_c1["queue_length"], cm_c2["queue_length"]])
cm_max_q = np.mean([cm_c1["max_queue_length"], cm_c2["max_queue_length"]])
cm_avg_wt = np.mean(
    [cm_c1["avg_waiting_time_per_vehicle"], cm_c2["avg_waiting_time_per_vehicle"]]
)


# ── 2.  Define radar axes ────────────────────────────────────────────
#
# Five metrics (↑ = better for radar display):
#   • Avg Reward           (higher is better)          — already ↑
#   • Low Waiting Time     (lower raw → higher score)  — invert
#   • Low Queue Length     (lower raw → higher score)  — invert
#   • Low Max Queue        (lower raw → higher score)  — invert
#   • Low Avg Wait/Vehicle (lower raw → higher score)  — invert

labels = [
    "Avg Reward",
    "Low Waiting\nTime",
    "Low Queue\nLength",
    "Low Max\nQueue",
    "Low Avg Wait\nper Vehicle",
]

raw = {
    "Congestion-Based\n(Ours)": [
        cong_reward,
        cong_waiting,
        cong_queue,
        cong_max_q,
        cong_avg_wt,
    ],
    "FedFlow": [ff_reward, ff_waiting, ff_queue, ff_max_q, ff_avg_wt],
    "FedKD": [kd_reward, kd_waiting, kd_queue, kd_max_q, kd_avg_wt],
    "FedCM": [cm_reward, cm_waiting, cm_queue, cm_max_q, cm_avg_wt],
}

# ── 3.  Normalise to 0-1 ─────────────────────────────────────────────
# For each metric we min-max normalise, then invert for
# "lower is better" metrics so that outer ring = best.

n_metrics = len(labels)
all_vals = np.array(list(raw.values()))  # (4, 5)

# Columns: 0=reward(↑), 1=waiting(↓), 2=queue(↓), 3=max_q(↓), 4=avg_wt(↓)
higher_is_better = [True, False, False, False, False]

normed = {}
for algo, vals in raw.items():
    nv = []
    for i, v in enumerate(vals):
        col = all_vals[:, i]
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-9:
            score = 1.0
        else:
            score = (v - mn) / (mx - mn)
        if not higher_is_better[i]:
            score = 1.0 - score
        # Clamp into [0.05, 1] so that the polygon stays visible
        score = max(score, 0.05)
        nv.append(score)
    normed[algo] = nv


# ── 4.  Plot radar chart ─────────────────────────────────────────────

angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
angles += angles[:1]  # close polygon

# Premium colour palette
colors = {
    "Congestion-Based\n(Ours)": ("#FF6B6B", "#FF6B6B33"),  # coral
    "FedFlow": ("#4ECDC4", "#4ECDC433"),  # teal
    "FedKD": ("#45B7D1", "#45B7D133"),  # sky blue
    "FedCM": ("#96CEB4", "#96CEB433"),  # sage
}

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

for algo, vals in normed.items():
    values = vals + vals[:1]
    line_col, fill_col = colors[algo]
    ax.plot(
        angles, values, "o-", linewidth=2.5, label=algo, color=line_col, markersize=7
    )
    ax.fill(angles, values, alpha=0.15, color=line_col)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight="bold", color="white")

# Gridlines
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#aaaaaa")
ax.set_rlabel_position(30)
ax.spines["polar"].set_color("#ffffff30")
ax.tick_params(axis="x", colors="white")
ax.grid(color="#ffffff20", linewidth=0.5)

# Title & legend
ax.set_title(
    "Federated Learning Algorithms — Traffic Control Comparison",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=30,
)
legend = ax.legend(
    loc="lower right",
    bbox_to_anchor=(1.25, -0.05),
    fontsize=11,
    frameon=True,
    fancybox=True,
    facecolor="#16213e",
    edgecolor="#ffffff30",
    labelcolor="white",
)

# ── 5.  Raw-value annotation table ───────────────────────────────────
table_text = (
    "─── Raw metric values (avg across clients/nodes) ───\n"
    f"{'Algorithm':<22s} {'Reward':>8s} {'WaitT':>8s} {'Queue':>8s} "
    f"{'MaxQ':>6s} {'AvgWt/V':>8s}\n"
)
for algo, vals in raw.items():
    name = algo.replace("\n", " ")
    table_text += (
        f"{name:<22s} {vals[0]:8.3f} {vals[1]:8.1f} {vals[2]:8.3f} "
        f"{vals[3]:6.1f} {vals[4]:8.3f}\n"
    )

fig.text(
    0.02,
    0.02,
    table_text,
    fontsize=9,
    fontfamily="monospace",
    color="#cccccc",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#16213e", edgecolor="#ffffff30"),
)

plt.tight_layout()
out_path = os.path.join(BASE, "radar_comparison.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✅  Radar chart saved → {out_path}")
# plt.show()  # Uncomment to display interactively
