import json
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/Users/shikshakkumar/Downloads/development/tcfl/adaptflow_ac/results_research_sumo_configs2_20260503_192846"

def get_af_data():
    path = os.path.join(base_dir, "adaptflow", "adaptflow_all_rounds.json")
    with open(path) as f:
        data = json.load(f)
    rounds = [r['round'] for r in data]
    waits = [np.mean([n['avg_waiting_time'] for n in r['nodes'].values()]) for r in data]
    queues = [np.mean([n['metrics']['average_queue_length'] for n in r['nodes'].values()]) for r in data]
    return rounds, waits, queues

def get_fed_dqn_data():
    path = os.path.join(base_dir, "fed_dqn_tsc", "fed_dqn_tsc_all_rounds.json")
    with open(path) as f:
        data = json.load(f)
    rounds = [r['round'] for r in data]
    waits = [r['avg_wait'] for r in data]
    queues = [r['avg_queue'] if 'avg_queue' in r else r.get('avg_queue_total', 0) for r in data]
    return rounds, waits, queues

def get_ma2c_data():
    path = os.path.join(base_dir, "multi_agent_ac", "multi_agent_ac_training.json")
    with open(path) as f:
        data = json.load(f)
    # Group by episode
    episodes = {}
    for entry in data:
        ep = entry['episode']
        if ep not in episodes:
            episodes[ep] = {'wait': [], 'queue': []}
        episodes[ep]['wait'].append(entry['avg_wait'])
        episodes[ep]['queue'].append(entry['avg_queue'])
    
    rounds = sorted(episodes.keys())
    waits = [np.mean(episodes[r]['wait']) for r in rounds]
    queues = [np.mean(episodes[r]['queue']) for r in rounds]
    return rounds, waits, queues

def get_dqtsca_data():
    path = os.path.join(base_dir, "dqtsca", "dqtsca_training.json")
    with open(path) as f:
        data = json.load(f)
    rounds = [r['episode'] for r in data]
    waits = [r['avg_wait'] for r in data]
    queues = [r['avg_queue'] for r in data]
    return rounds, waits, queues

# Extract
af_r, af_w, af_q = get_af_data()
fd_r, fd_w, fd_q = get_fed_dqn_data()
ma_r, ma_w, ma_q = get_ma2c_data()
dq_r, dq_w, dq_q = get_dqtsca_data()

# Plot Waiting Time
plt.figure(figsize=(10, 6))
plt.plot(af_r, af_w, label='AdaptFlow-AC', marker='o')
plt.plot(fd_r, fd_w, label='FedDQN-TSC', marker='s')
plt.plot(ma_r, ma_w, label='Multi-Agent AC', marker='^')
plt.plot(dq_r, dq_w, label='DQT-SCA', marker='x')
plt.title('Average Waiting Time per Round')
plt.xlabel('Round / Episode')
plt.ylabel('Waiting Time (s)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(base_dir, 'scratch', 'waiting_time_plot.png'))

# Plot Queue Length
plt.figure(figsize=(10, 6))
plt.plot(af_r, af_q, label='AdaptFlow-AC', marker='o')
plt.plot(fd_r, fd_q, label='FedDQN-TSC', marker='s')
plt.plot(ma_r, ma_q, label='Multi-Agent AC', marker='^')
plt.plot(dq_r, dq_q, label='DQT-SCA', marker='x')
plt.title('Average Queue Length per Round')
plt.xlabel('Round / Episode')
plt.ylabel('Queue Length (vehicles)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(base_dir, 'scratch', 'queue_length_plot.png'))

# Pie Chart for Throughput Distribution (Final Round of AdaptFlow)
path = os.path.join(base_dir, "adaptflow", "adaptflow_all_rounds.json")
with open(path) as f:
    data = json.load(f)
final_round = data[-1]
node_names = list(final_round['nodes'].keys())
throughputs = [n['metrics']['throughput'] for n in final_round['nodes'].values()]

plt.figure(figsize=(8, 8))
plt.pie(throughputs, labels=node_names, autopct='%1.1f%%', startangle=140)
plt.title('Throughput Distribution across Nodes (AdaptFlow Final Round)')
plt.savefig(os.path.join(base_dir, 'scratch', 'throughput_pie.png'))

print("Plots generated successfully in scratch directory.")
