import json, os

results_dir = r'c:\Users\madhu\btp\tcfl\backend\results_adaptflow'
nodes = ['node_0','node_1','node_2','node_3','node_4','node_5']
rounds = [1,2,3]

all_data = {}

for r in rounds:
    all_data[r] = {}
    for nid in nodes:
        path = os.path.join(results_dir, f'{nid}_round_{r}_eval.json')
        with open(path) as f:
            d = json.load(f)
        m = d.get('metrics', {})
        all_data[r][nid] = {
            'cluster': d['cluster_id'],
            'reward': d['total_reward'],
            'avg_wait': d['avg_waiting_time'],
            'total_wait': m.get('total_waiting_time', 0),
            'throughput': m.get('throughput', 0),
            'tp_ratio': m.get('throughput_ratio', 0),
            'avg_queue': m.get('average_queue_length', 0),
            'max_queue': m.get('max_queue_length', 0),
            'steps': m.get('steps', 200),
            'total_departed': m.get('total_departed', 0),
            'loss': d['loss'],
        }

for r in rounds:
    print(f"\n{'='*90}")
    print(f"ROUND {r}")
    print(f"{'='*90}")
    print(f"{'Node':<8} {'Cluster':<12} {'Reward':>10} {'AvgWait(s)':>12} {'TotalWait(s)':>14} {'Throughput':>12} {'TP%':>8} {'AvgQueue':>10} {'MaxQ':>6} {'Loss':>8}")
    print(f"{'-'*90}")
    for nid in nodes:
        d = all_data[r][nid]
        print(f"{nid:<8} {d['cluster']:<12} {d['reward']:>10.2f} {d['avg_wait']:>12.4f} {d['total_wait']:>14.1f} {d['throughput']:>12} {d['tp_ratio']*100:>8.2f} {d['avg_queue']:>10.3f} {d['max_queue']:>6} {d['loss']:>8.4f}")

print(f"\n{'='*90}")
print("WAITING TIME PROGRESSION (Avg Wait per Vehicle in seconds)")
print(f"{'='*90}")
print(f"{'Node':<8} {'Priority':<10} {'Round 1':>10} {'Round 2':>10} {'Round 3':>10} {'Trend':>10}")
print(f"{'-'*50}")
priority_map = {'node_0': 'Tier-2', 'node_1': 'Tier-1', 'node_2': 'Tier-3', 'node_3': 'Tier-1', 'node_4': 'Tier-2', 'node_5': 'Tier-1'}
for nid in nodes:
    r1 = all_data[1][nid]['avg_wait']
    r2 = all_data[2][nid]['avg_wait']
    r3 = all_data[3][nid]['avg_wait']
    delta = r3 - r1
    trend = f"{'↓' if delta < 0 else '↑' if delta > 0 else '='} {abs(delta):.4f}s"
    print(f"{nid:<8} {priority_map[nid]:<10} {r1:>10.4f} {r2:>10.4f} {r3:>10.4f} {trend:>10}")

print(f"\n{'='*90}")
print("CLUSTER ASSIGNMENTS ACROSS ROUNDS")
print(f"{'='*90}")
print(f"{'Node':<8} {'Round 1':>10} {'Round 2':>10} {'Round 3':>10}")
print(f"{'-'*40}")
for nid in nodes:
    c1 = all_data[1][nid]['cluster']
    c2 = all_data[2][nid]['cluster']
    c3 = all_data[3][nid]['cluster']
    print(f"{nid:<8} {c1:>10} {c2:>10} {c3:>10}")
