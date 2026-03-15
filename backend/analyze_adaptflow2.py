import json
import os
import glob

results_dir = 'results_adaptflow'

# Gather all per-node eval files and organize by round
node_data = {}  # round -> list of (node_id, data)

for fn in sorted(glob.glob(os.path.join(results_dir, 'node_*_round_*_eval.json'))):
    basename = os.path.basename(fn)
    parts = basename.replace('_eval.json', '').split('_')
    # parts: ['node', N, 'round', R]
    node_num = int(parts[1])
    round_num = int(parts[3])
    with open(fn) as f:
        data = json.load(f)
    if round_num not in node_data:
        node_data[round_num] = []
    node_data[round_num].append((f'node_{node_num}', data))

print('=== Per-Round Summary (all nodes across all eval files) ===')
for rnd in sorted(node_data.keys()):
    nodes = node_data[rnd]
    rewards = [d['total_reward'] for _, d in nodes]
    waits = [d['avg_waiting_time'] for _, d in nodes]
    losses = [d['loss'] for _, d in nodes if d.get('loss') is not None]
    throughputs = [d['metrics'].get('throughput', d['metrics'].get('total_vehicles', 0)) for _, d in nodes]
    
    # determine mode from reward sign / structure
    has_sumo_throughput = 'throughput' in nodes[0][1]['metrics']
    mode = 'SUMO' if has_sumo_throughput else 'Mock'
    
    print(f'\nRound {rnd} ({mode}, {len(nodes)} nodes):')
    print(f'  Avg Reward:     {sum(rewards)/len(rewards):.2f}  (min={min(rewards):.2f}, max={max(rewards):.2f})')
    print(f'  Avg Wait (s):   {sum(waits)/len(waits):.3f}  (min={min(waits):.3f}, max={max(waits):.3f})')
    print(f'  Total Thruput:  {sum(throughputs)}')
    if losses:
        print(f'  Avg Loss:       {sum(losses)/len(losses):.4f}')

print()
print('=== SUMO Nodes Detail (node_0 to node_5) ===')
sumo_nodes = ['node_0','node_1','node_2','node_3','node_4','node_5']
print(f'{"Node":<10} ' + '  '.join([f'R{r}_reward  R{r}_wait' for r in range(1,4)]))
for nid in sumo_nodes:
    row = f'{nid:<10} '
    for r in range(1, 4):
        fn = os.path.join(results_dir, f'{nid}_round_{r}_eval.json')
        with open(fn) as f:
            d = json.load(f)
        row += f'{d["total_reward"]:>10.2f}  {d["avg_waiting_time"]:>6.3f}s  '
    print(row)

print()
print('=== Mock Nodes Summary (node_6 to node_25, all rounds) ===')
mock_nodes = [f'node_{i}' for i in range(6, 26)]
all_mock_rewards_by_round = {}
for nid in mock_nodes:
    for r in range(1, 6):
        fn = os.path.join(results_dir, f'{nid}_round_{r}_eval.json')
        if os.path.exists(fn):
            with open(fn) as f:
                d = json.load(f)
            if r not in all_mock_rewards_by_round:
                all_mock_rewards_by_round[r] = []
            all_mock_rewards_by_round[r].append(d['total_reward'])

for r in sorted(all_mock_rewards_by_round.keys()):
    rews = all_mock_rewards_by_round[r]
    print(f'  Round {r}: avg_reward={sum(rews)/len(rews):.2f}, min={min(rews):.2f}, max={max(rews):.2f}, n={len(rews)}')
