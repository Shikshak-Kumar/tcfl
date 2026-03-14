import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = "results_fedflow"

for rnum in [1,2,3]:
    path = os.path.join(RESULTS_DIR, f"round_{rnum}_summary.json")
    with open(path) as f:
        rs = json.load(f)
    print(f"\nRound {rnum} cluster_congestion: {rs['cluster_congestion']}")
    for cid, cv in rs['clusters'].items():
        print(f"  {cid}: avg_flow={cv['avg_flow']:.4f}  agents={cv['agent_ids']}")
