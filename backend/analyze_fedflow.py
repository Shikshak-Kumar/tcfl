import json, os, sys

# enforce UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = "results_fedflow"

with open(os.path.join(RESULTS_DIR, "fedflow_all_rounds.json")) as f:
    all_rounds = json.load(f)

SEP = "=" * 68
sep = "-" * 68

print(SEP)
print("FEDFLOW RESULTS ANALYSIS")
print(SEP)
print(f"Mode   : {all_rounds[0]['mode']}")
print(f"Rounds : {len(all_rounds)}")
n_nodes = len(all_rounds[0]['nodes'])
node_ids = list(all_rounds[0]['nodes'].keys())
print(f"Nodes  : {n_nodes}  ({', '.join(node_ids)})")

# ── Round-level averages ──────────────────────────────────────────────
print(f"\n{sep}\nROUND-LEVEL AVERAGES (across all nodes)\n{sep}")
print(f"{'Rnd':<5} {'Reward':>10} {'Wait(s)':>10} {'Loss':>10} {'Throughput':>12} {'Thru%':>7} {'AvgQueue':>10} {'Departed':>10}")
round_data = []
for r in all_rounds:
    rnd   = r['round']
    nodes = list(r['nodes'].values())
    d = dict(
        round     = rnd,
        reward    = sum(n['total_reward'] for n in nodes)           / n_nodes,
        wait      = sum(n['avg_waiting_time'] for n in nodes)       / n_nodes,
        loss      = sum(n['loss'] for n in nodes)                   / n_nodes,
        tp        = sum(n['metrics']['throughput'] for n in nodes)  / n_nodes,
        tr        = sum(n['metrics']['throughput_ratio'] for n in nodes) / n_nodes,
        qlen      = sum(n['metrics']['average_queue_length'] for n in nodes) / n_nodes,
        departed  = sum(n['metrics']['total_departed'] for n in nodes) / n_nodes,
    )
    round_data.append(d)
    print(f"  {rnd:<3} {d['reward']:>10.2f} {d['wait']:>10.2f} {d['loss']:>10.4f} "
          f"{d['tp']:>12.1f} {d['tr']*100:>6.2f}% {d['qlen']:>10.3f} {d['departed']:>10.1f}")

# ── Trend ─────────────────────────────────────────────────────────────
if len(round_data) > 1:
    print(f"\n{sep}\nTREND  (Round 1 -> Round {round_data[-1]['round']})\n{sep}")
    first, last = round_data[0], round_data[-1]
    metrics = [
        ('reward',   'Total Reward',      True),
        ('wait',     'Avg Wait (s)',       False),
        ('loss',     'Loss',              False),
        ('tp',       'Throughput (veh)',   True),
        ('tr',       'Throughput Ratio',   True),
        ('qlen',     'Avg Queue Length',   False),
        ('departed', 'Total Departed',     True),
    ]
    for key, label, higher_better in metrics:
        v1, v2  = first[key], last[key]
        delta   = v2 - v1
        pct     = (delta / v1 * 100) if v1 != 0 else float('inf')
        arrow   = "UP" if delta > 0 else "DOWN"
        good    = (delta > 0) if higher_better else (delta < 0)
        tag     = "[GOOD]" if good else "[WARN]"
        print(f"  {tag} {label:<22}: {v1:.3f} -> {v2:.3f}  ({arrow} {abs(pct):.1f}%)")

# ── Per-node across rounds ────────────────────────────────────────────
print(f"\n{sep}\nPER-NODE BREAKDOWN ACROSS ROUNDS\n{sep}")
for nid in node_ids:
    print(f"\n  Node: {nid}")
    print(f"  {'Rnd':<5} {'Reward':>10} {'Wait(s)':>10} {'Loss':>10} {'ThrPut':>8} {'Cluster'}")
    for r in all_rounds:
        n = r['nodes'][nid]
        print(f"    {r['round']:<3} {n['total_reward']:>10.2f} {n['avg_waiting_time']:>10.2f} "
              f"{n['loss']:>10.4f} {n['metrics']['throughput']:>8}   {n['cluster_id']}")

# ── Round summary JSONs ───────────────────────────────────────────────
print(f"\n{sep}\nROUND SUMMARY JSONs\n{sep}")
for rnum in range(1, len(all_rounds)+1):
    path = os.path.join(RESULTS_DIR, f"round_{rnum}_summary.json")
    if not os.path.exists(path): continue
    with open(path) as f:
        rs = json.load(f)
    print(f"\n  round_{rnum}_summary.json  — top-level keys: {list(rs.keys())}")
    for k, v in rs.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"      {k}: {v}")
        elif isinstance(v, dict):
            print(f"      {k}: dict  ({len(v)} entries)")
        elif isinstance(v, list):
            print(f"      {k}: list  ({len(v)} items)")

# ── Node eval files ───────────────────────────────────────────────────
print(f"\n{sep}\nNODE EVAL FILES\n{sep}")
print(f"  {'File':<36} {'Reward':>10} {'Wait(s)':>10} {'Loss':>10} {'Throughput':>12}")
for nid_num in range(n_nodes):
    for rnum in range(1, len(all_rounds)+1):
        path = os.path.join(RESULTS_DIR, f"node_{nid_num}_round_{rnum}_eval.json")
        if not os.path.exists(path): continue
        with open(path) as f:
            ev = json.load(f)
        reward = ev.get('total_reward', ev.get('reward', 'N/A'))
        wait   = ev.get('avg_waiting_time', ev.get('average_waiting_time', 'N/A'))
        loss   = ev.get('loss', 'N/A')
        tp     = ev.get('metrics', {}).get('throughput', ev.get('throughput', 'N/A'))
        fname  = f"node_{nid_num}_round_{rnum}_eval.json"
        rw = f"{reward:.2f}" if isinstance(reward, float) else str(reward)
        rw2 = f"{wait:.2f}"  if isinstance(wait,   float) else str(wait)
        rl = f"{loss:.4f}"   if isinstance(loss,   float) else str(loss)
        rt = str(tp)
        print(f"  {fname:<36} {rw:>10} {rw2:>10} {rl:>10} {rt:>12}")

print("\nDone.")
