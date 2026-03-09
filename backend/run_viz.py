from utils.visualization import TrafficVisualizer
from plot_client_lane_waiting import (
    SCENARIOS,
    DEFAULT_CLIENT_IDS,
    plot_clients_across_scenarios,
)


if __name__ == "__main__":
    # 1) Old-style graphs (per-client waiting + congestion/weight) for each scenario
    for scen_name, results_dir in SCENARIOS.items():
        print(f"\n========== Scenario: {scen_name} (dir={results_dir}) ==========")
        viz = TrafficVisualizer(results_dir)
        viz.plot_wait_and_congestion_per_client()

    # 2) New detailed graphs (waiting time & lane-wise queues) across all scenarios
    print(
        "\n========== Per-client waiting / per-lane queues across scenarios =========="
    )
    plot_clients_across_scenarios(SCENARIOS, DEFAULT_CLIENT_IDS)