import matplotlib.pyplot as plt
import os

def plot_results(results, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training Convergence (Episode Reward)
    plt.figure(figsize=(10, 5))
    for model_name, data in results.items():
        plt.plot(data['rewards'], label=model_name)
    plt.title('Training Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.close()
    
    # 2. Bar Chart: Average Waiting Time
    plt.figure(figsize=(8, 5))
    model_names = list(results.keys())
    avg_waits = [results[m]['metrics']['avg_waiting'] for m in model_names]
    plt.bar(model_names, avg_waits, color=['blue', 'green'])
    plt.title('Average Waiting Time')
    plt.ylabel('Seconds')
    plt.savefig(os.path.join(save_dir, 'waiting_time.png'))
    plt.close()
    
    # 3. Bar Chart: Throughput
    plt.figure(figsize=(8, 5))
    throughputs = [results[m]['metrics']['throughput'] for m in model_names]
    plt.bar(model_names, throughputs, color=['blue', 'green'])
    plt.title('Total Throughput')
    plt.ylabel('Vehicles')
    plt.savefig(os.path.join(save_dir, 'throughput.png'))
    plt.close()
