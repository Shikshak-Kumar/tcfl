import json

with open('results_adaptflow/adaptflow_all_rounds.json', 'r') as f:
    all_rounds = json.load(f)

print(f'Number of rounds: {len(all_rounds)}')
print()

for rnd in all_rounds:
    round_num = rnd['round']
    mode = rnd.get('mode', 'unknown')
    cluster_info = rnd.get('cluster_info', {})
    nodes = rnd.get('nodes', {})
    clustering = rnd.get('clustering', {})
    assignments = clustering.get('assignments', {})

    print(f'=== Round {round_num} | Mode: {mode} ===')
    print(f'  Cluster assignments: {dict(sorted(assignments.items()))}')
    for cid, cdata in cluster_info.items():
        members = cdata.get('members', [])
        congestion = cdata.get('congestion', 0)
        avg_flow = cdata.get('avg_flow', 0)
        print(f'  {cid}: members={members}, congestion={congestion:.3f}, avg_flow={avg_flow:.4f}')

    rewards = []
    waiting_times = []
    throughputs = []
    losses = []
    for nid, ndata in nodes.items():
        r = ndata.get('total_reward', 0)
        w = ndata.get('avg_waiting_time', 0)
        t = ndata.get('metrics', {}).get('throughput', 0)
        l = ndata.get('loss', None)
        rewards.append(r)
        waiting_times.append(w)
        throughputs.append(t)
        if l is not None:
            losses.append(l)
        print(f'  {nid}: reward={r:.2f}, avg_wait={w:.3f}s, throughput={t}, loss={l}')

    avg_loss_str = f'{sum(losses)/len(losses):.4f}' if losses else 'N/A'
    print(f'  >> Global avg: reward={sum(rewards)/len(rewards):.2f}, wait={sum(waiting_times)/len(waiting_times):.3f}s, throughput_total={sum(throughputs)}, avg_loss={avg_loss_str}')
    print()

# Cluster history
print('=== Cluster History ===')
with open('results_adaptflow/cluster_history.json', 'r') as f:
    ch = json.load(f)

print(f'Num rounds tracked: {ch["num_rounds"]}')
for i, hist in enumerate(ch['cluster_history']):
    print(f'  Round {i+1}: {hist}')

print()
print('=== Cluster Transitions ===')
for t in ch['transitions']:
    if t['transitions']:
        print(f'  Round {t["round"]}: {t["transitions"]}')
    else:
        print(f'  Round {t["round"]}: No transitions')
