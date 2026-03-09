import os
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Default base directory (old behaviour)
RESULTS_DIR = "results"

# Optional: multiple intersection scenarios with separate results folders
# Edit these names according to your experiments:
#   - "3_lane": folder where you stored 3‑lane intersection results
#   - "4_lane": folder for 4‑lane intersection (default)
#   - "5_lane": folder for 5‑lane intersection
SCENARIOS: Dict[str, str] = {
    "3_lane": "results_3lane",
    "4_lane": "results_4lane",
    "5_lane": "results_5lane",
}

# If each experiment has two clients with ids "client_1" and "client_2"
DEFAULT_CLIENT_IDS: List[str] = ["client_1", "client_2"]


def _load_client_detailed_history(
    client_id: str, results_dir: str = RESULTS_DIR
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Load per-round detailed metrics for a given client from *_detailed.json files.

    Returns:
        rounds: list of round indices
        lane_wait_times: lane-level total waiting time (from lane_summary.total_waiting_time)
        intersection_wait_times: intersection-level total waiting time (top-level total_waiting_time)
        avg_queue_lengths: average queue length over the episode
    """
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return [], [], [], []

    rounds: List[int] = []
    lane_wait_times: List[float] = []
    intersection_wait_times: List[float] = []
    avg_queue_lengths: List[float] = []

    prefix = f"{client_id}_round_"
    suffix = "_detailed.json"

    for filename in os.listdir(results_dir):
        if not (filename.startswith(prefix) and filename.endswith(suffix)):
            continue

        # Extract round index from e.g. client_1_round_4_detailed.json
        try:
            middle = filename[len(prefix) : -len(suffix)]
            round_idx = int(middle)
        except Exception:
            continue

        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r") as f:
                data: Dict = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        lane_summary: Dict = data.get("lane_summary", {})

        lane_wait = float(lane_summary.get("total_waiting_time", 0.0))
        intersection_wait = float(data.get("total_waiting_time", 0.0))
        avg_queue = float(data.get("average_queue_length", 0.0))

        rounds.append(round_idx)
        lane_wait_times.append(lane_wait)
        intersection_wait_times.append(intersection_wait)
        avg_queue_lengths.append(avg_queue)

    # Sort by round index so plots are in correct order
    if rounds:
        sorted_idx = sorted(range(len(rounds)), key=lambda i: rounds[i])
        rounds = [rounds[i] for i in sorted_idx]
        lane_wait_times = [lane_wait_times[i] for i in sorted_idx]
        intersection_wait_times = [intersection_wait_times[i] for i in sorted_idx]
        avg_queue_lengths = [avg_queue_lengths[i] for i in sorted_idx]

    return rounds, lane_wait_times, intersection_wait_times, avg_queue_lengths


def _load_client_lane_queues(
    client_id: str, results_dir: str = RESULTS_DIR
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    Load per-round queue length for each lane of the intersection for a given client.

    Returns:
        rounds: list of round indices
        lane_to_queues: dict mapping lane_id -> list of queue lengths aligned with rounds
    """
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return [], {}

    prefix = f"{client_id}_round_"
    suffix = "_detailed.json"

    rounds: List[int] = []
    per_round_lane_queues: List[Dict[str, float]] = []
    all_lanes: set[str] = set()

    for filename in os.listdir(results_dir):
        if not (filename.startswith(prefix) and filename.endswith(suffix)):
            continue

        try:
            middle = filename[len(prefix) : -len(suffix)]
            round_idx = int(middle)
        except Exception:
            continue

        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r") as f:
                data: Dict = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        per_lane: Dict = data.get("per_lane_metrics", {})
        lane_queues_for_round: Dict[str, float] = {}

        for lane_id, lane_data in per_lane.items():
            q = float(lane_data.get("queue_length", 0.0))
            lane_queues_for_round[lane_id] = q
            all_lanes.add(lane_id)

        rounds.append(round_idx)
        per_round_lane_queues.append(lane_queues_for_round)

    if not rounds:
        return [], {}

    # Sort by round index
    sorted_idx = sorted(range(len(rounds)), key=lambda i: rounds[i])
    rounds = [rounds[i] for i in sorted_idx]
    per_round_lane_queues = [per_round_lane_queues[i] for i in sorted_idx]

    lane_to_queues: Dict[str, List[float]] = {}
    for lane_id in sorted(all_lanes):
        series: List[float] = []
        for lane_queues in per_round_lane_queues:
            series.append(float(lane_queues.get(lane_id, 0.0)))
        lane_to_queues[lane_id] = series

    return rounds, lane_to_queues


def plot_client_waiting_and_queue(client_id: str, results_dir: str = RESULTS_DIR) -> None:
    """
    Plot three graphs for a given client:
      1) Waiting time for lanes (lane_summary.total_waiting_time)
      2) Waiting time for intersection (total_waiting_time)
      3) Average queue length (average_queue_length)
    """
    (
        rounds,
        lane_wait_times,
        intersection_wait_times,
        avg_queue_lengths,
    ) = _load_client_detailed_history(client_id, results_dir)

    if not rounds:
        print(f"No detailed *_detailed.json files found for {client_id} in {results_dir}")
        return

    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Client {client_id} - Waiting Time and Queue Length per Round", fontsize=14)

    # 1) Waiting time for lane (from lane_summary)
    ax = axes[0]
    ax.plot(rounds, lane_wait_times, marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Round")
    ax.set_ylabel("Lane waiting time (s)")
    ax.set_title("Lane waiting time")
    ax.grid(True, alpha=0.3)

    # 2) Waiting time for intersection (top-level)
    ax = axes[1]
    ax.plot(rounds, intersection_wait_times, marker="o", linestyle="-", color="tab:green")
    ax.set_xlabel("Round")
    ax.set_ylabel("Intersection waiting time (s)")
    ax.set_title("Intersection waiting time")
    ax.grid(True, alpha=0.3)

    # 3) Average queue length
    ax = axes[2]
    ax.plot(rounds, avg_queue_lengths, marker="o", linestyle="-", color="tab:orange")
    ax.set_xlabel("Round")
    ax.set_ylabel("Average queue length")
    ax.set_title("Average queue length")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_client_lane_queues(client_id: str, results_dir: str = RESULTS_DIR) -> None:
    """
    Plot queue length over rounds for each lane of the intersection for a given client.
    Each lane appears as a separate line in the same figure.
    """
    rounds, lane_to_queues = _load_client_lane_queues(client_id, results_dir)

    if not rounds or not lane_to_queues:
        print(f"No per-lane queue data found for {client_id} in {results_dir}")
        return

    plt.style.use("seaborn-v0_8")

    plt.figure(figsize=(10, 6))
    for lane_id, queues in lane_to_queues.items():
        plt.plot(rounds, queues, marker="o", linestyle="-", label=lane_id)

    plt.xlabel("Round")
    plt.ylabel("Queue length (vehicles)")
    plt.title(f"Client {client_id} - Queue length per lane over rounds")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Lane ID", fontsize=8)
    plt.tight_layout()
    plt.show()


def _discover_clients(results_dir: str = RESULTS_DIR) -> List[str]:
    """Discover client IDs from *_detailed.json filenames in the results directory."""
    if not os.path.isdir(results_dir):
        return []

    client_ids = set()
    for filename in os.listdir(results_dir):
        if not (filename.endswith("_detailed.json") and filename.startswith("client_")):
            continue
        # filename pattern: client_1_round_4_detailed.json -> client_1
        parts = filename.split("_")
        if len(parts) >= 3:
            client_ids.add(f"{parts[0]}_{parts[1]}")
    return sorted(client_ids)


def plot_clients_across_scenarios(
    scenarios: Dict[str, str], client_ids: List[str]
) -> None:
    """
    High-level helper:
    For each client (client_1, client_2, ...), first plot all graphs for 3‑lane,
    then 4‑lane, then 5‑lane (or whatever scenarios you configure).
    """
    for cid in client_ids:
        print(f"\n==============================")
        print(f"CLIENT: {cid}")
        print(f"==============================")
        for scen_name, results_dir in scenarios.items():
            print(f"\n--- Scenario: {scen_name} (results from '{results_dir}') ---")
            if not os.path.isdir(results_dir):
                print(f"  [SKIP] directory not found: {results_dir}")
                continue

            # Check if this client has any *_detailed.json files in this scenario
            pattern_prefix = f"{cid}_round_"
            has_files = any(
                fn.startswith(pattern_prefix) and fn.endswith("_detailed.json")
                for fn in os.listdir(results_dir)
            )
            if not has_files:
                print(f"  [SKIP] no detailed files for {cid} in {results_dir}")
                continue

            print(f"  Plotting waiting/queue metrics for {cid} in {scen_name}...")
            plot_client_waiting_and_queue(cid, results_dir)
            print(f"  Plotting lane-wise queues for {cid} in {scen_name}...")
            plot_client_lane_queues(cid, results_dir)


if __name__ == "__main__":
    # Master mode: plot for multiple intersection types (3-lane, 4-lane, 5-lane)
    # with two clients in each scenario.
    plot_clients_across_scenarios(SCENARIOS, DEFAULT_CLIENT_IDS)


