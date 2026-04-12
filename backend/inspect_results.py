import json

base = r'c:\Users\madhu\btp\tcfl\backend\results_adaptflow'

rounds = {}
for r in range(1, 6):
    with open(f'{base}/round_{r}_summary.json') as f:
        rounds[r] = json.load(f)

NODES = ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5']

# Extract per-node loss across rounds
print("=== PER NODE LOSS ACROSS ROUNDS ===")
print(f"{'Node':<10} {'R1':>8} {'R2':>8} {'R3':>8} {'R4':>8} {'R5':>8} {'delta':>8}")
for n in NODES:
    losses = [rounds[r]['nodes'][n]['loss'] for r in range(1,6)]
    delta = losses[-1] - losses[0]
    print(f"{n:<10} {losses[0]:>8.4f} {losses[1]:>8.4f} {losses[2]:>8.4f} {losses[3]:>8.4f} {losses[4]:>8.4f} {delta:>+8.4f}")

print()
print("=== PER NODE REWARD ACROSS ROUNDS ===")
print(f"{'Node':<10} {'R1':>10} {'R2':>10} {'R3':>10} {'R4':>10} {'R5':>10}")
for n in NODES:
    rewards = [rounds[r]['nodes'][n]['total_reward'] for r in range(1,6)]
    print(f"{n:<10} {rewards[0]:>10.4f} {rewards[1]:>10.4f} {rewards[2]:>10.4f} {rewards[3]:>10.4f} {rewards[4]:>10.4f}")

print()
print("=== PER NODE WAIT TIME ACROSS ROUNDS ===")
print(f"{'Node':<10} {'R1':>8} {'R2':>8} {'R3':>8} {'R4':>8} {'R5':>8}")
for n in NODES:
    waits = [rounds[r]['nodes'][n]['avg_waiting_time'] for r in range(1,6)]
    print(f"{n:<10} {waits[0]:>8.4f} {waits[1]:>8.4f} {waits[2]:>8.4f} {waits[3]:>8.4f} {waits[4]:>8.4f}")

print()
print("=== PER NODE QUEUE LENGTH ACROSS ROUNDS ===")
print(f"{'Node':<10} {'R1':>8} {'R2':>8} {'R3':>8} {'R4':>8} {'R5':>8} {'MaxQ_R5':>8}")
for n in NODES:
    queues = [rounds[r]['nodes'][n]['metrics']['average_queue_length'] for r in range(1,6)]
    maxq = rounds[5]['nodes'][n]['metrics']['max_queue_length']
    print(f"{n:<10} {queues[0]:>8.4f} {queues[1]:>8.4f} {queues[2]:>8.4f} {queues[3]:>8.4f} {queues[4]:>8.4f} {maxq:>8}")

print()
print("=== PER NODE THROUGHPUT RATIO ACROSS ROUNDS ===")
print(f"{'Node':<10} {'R1':>8} {'R2':>8} {'R3':>8} {'R4':>8} {'R5':>8}")
for n in NODES:
    tps = [rounds[r]['nodes'][n]['metrics']['throughput_ratio'] for r in range(1,6)]
    print(f"{n:<10} {tps[0]:>8.4f} {tps[1]:>8.4f} {tps[2]:>8.4f} {tps[3]:>8.4f} {tps[4]:>8.4f}")

print()
print("=== GST PER EDGE ALL ROUNDS ===")
for n in NODES:
    print(f"\n{n}:")
    for r in range(1,6):
        nd = rounds[r]['nodes'][n]
        gst = nd['metrics'].get('green_signal_time_avg_per_edge', {})
        cl = nd['cluster_id']
        print(f"  R{r} ({cl}): {gst}")

print()
print("=== DEPARTURES/ARRIVALS ===")
for r in range(1,6):
    total_dep = sum(rounds[r]['nodes'][n]['metrics']['total_departed'] for n in NODES)
    total_arr = sum(rounds[r]['nodes'][n]['metrics']['throughput'] for n in NODES)
    print(f"Round {r}: total_departed={total_dep}, total_arrived={total_arr}, global_tp={total_arr/total_dep:.4f}")

print()
print("=== STEPS PER NODE ===")
for n in NODES:
    steps = [rounds[r]['nodes'][n]['metrics']['steps'] for r in range(1,6)]
    print(f"{n}: {steps}")

print()
print("=== CLUSTER GROUPS PER ROUND ===")
for r in range(1,6):
    data = rounds[r]
    groups = data['clustering']['groups']
    ci = data['cluster_info']
    print(f"Round {r}: {groups}")
    for cluster, info in ci.items():
        print(f"  {cluster}: flow={info['avg_flow']:.4f} congestion={info['congestion']:.4f} members={info['members']}")
