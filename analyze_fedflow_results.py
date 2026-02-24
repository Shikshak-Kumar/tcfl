"""Quick analysis of FedFlow results."""

import json

with open("results_fedflow/fedflow_all_rounds.json") as f:
    data = json.load(f)

print(f"Total rounds: {len(data)}\n")

for rd in data:
    r = rd["round"]
    mode = rd["mode"]
    print(f"{'=' * 70}")
    print(f"  ROUND {r} ({mode})")
    print(f"{'=' * 70}")

    for nid, node in rd["nodes"].items():
        m = node["metrics"]
        cid = node["cluster_id"]
        reward = node["total_reward"]
        avg_wait = m.get("avg_waiting_time_per_vehicle", 0)
        total_wait = m.get("total_waiting_time", 0)
        throughput = m.get("throughput", 0)
        departed = m.get("total_departed", 0)
        tp_ratio = m.get("throughput_ratio", 0)
        avg_queue = m.get("average_queue_length", 0)
        loss = node["loss"]
        ls = m.get("lane_summary", {})
        congested = ls.get("num_congested_lanes", 0)

        print(f"\n  {nid} ({cid}):")
        print(f"    Reward:         {reward:.2f}")
        print(f"    Avg Wait/Veh:   {avg_wait:.2f}s")
        print(f"    Total Wait:     {total_wait:.1f}s")
        print(
            f"    Throughput:     {throughput} arrived / {departed} departed (ratio: {tp_ratio:.4f})"
        )
        print(f"    Avg Queue:      {avg_queue:.2f}")
        print(f"    Loss:           {loss:.4f}")
        print(f"    Congested:      {congested} lanes")

    ci = rd.get("cluster_info", {})
    if ci:
        print(f"\n  Cluster Summary:")
        for cid, info in ci.items():
            print(
                f"    {cid}: congestion={info.get('congestion', 0):.4f}, avg_flow={info.get('avg_flow', 0):.4f}"
            )
    print()

# Cross-round comparison table
print(f"\n{'=' * 90}")
print(f"  CROSS-ROUND COMPARISON")
print(f"{'=' * 90}")
print(
    f"{'Node':<10} | {'Round':<6} | {'Reward':<10} | {'AvgWait':<10} | {'Throughput':<12} | {'TP Ratio':<10} | {'Loss':<8}"
)
print(f"{'-' * 90}")
for rd in data:
    r = rd["round"]
    for nid, node in rd["nodes"].items():
        m = node["metrics"]
        print(
            f"{nid:<10} | R{r:<5} | {node['total_reward']:>10.2f} | {m.get('avg_waiting_time_per_vehicle', 0):>8.2f}s | {m.get('throughput', 0):>5}/{m.get('total_departed', 0):>5} dep | {m.get('throughput_ratio', 0):>10.4f} | {node['loss']:>8.4f}"
        )
    print(f"{'-' * 90}")
