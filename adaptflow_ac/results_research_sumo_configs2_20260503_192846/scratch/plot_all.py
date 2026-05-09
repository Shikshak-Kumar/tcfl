import json
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/Users/shikshakkumar/Downloads/development/tcfl/adaptflow_ac/results_research_sumo_configs2_20260503_192846"
plt.style.use('dark_background')

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

colors = ['#00FFCC', '#FF00FF', '#FFFF00', '#FF3300'] # Vibrant Cyberpunk colors
labels = ['AdaptFlow-AC', 'FedDQN-TSC', 'Multi-Agent AC', 'DQT-SCA']
datasets_w = [(af_r, af_w), (fd_r, fd_w), (ma_r, ma_w), (dq_r, dq_w)]
datasets_q = [(af_r, af_q), (fd_r, fd_q), (ma_r, ma_q), (dq_r, dq_q)]

def plot_vibrant(datasets, title, ylabel, filename):
    plt.figure(figsize=(12, 7))
    for i, (r, val) in enumerate(datasets):
        color = colors[i]
        # Glowing effect
        for j in range(1, 10):
            plt.plot(r, val, color=color, linewidth=j, alpha=0.05)
        plt.plot(r, val, color=color, linewidth=2, label=labels[i])
        plt.scatter(r, val, color=color, s=30, alpha=0.8)
    
    plt.title(title, fontsize=16, color='white', pad=20)
    plt.xlabel('Round / Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(color='#333333', linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.path.join(base_dir, 'scratch', filename), dpi=300, bbox_inches='tight')

plot_vibrant(datasets_w, 'Average Waiting Time per Round', 'Waiting Time (s)', 'waiting_time_vibrant.png')
plot_vibrant(datasets_q, 'Average Queue Length per Round', 'Queue Length (vehicles)', 'queue_length_vibrant.png')

print("Vibrant plots generated successfully.")
