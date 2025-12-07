#!/usr/bin/env python3
"""
Training Results Visualization
Generates graphs to analyze federated learning performance
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_all_results(results_dir="results"):
    """Load all training and evaluation results"""
    clients_data = {}
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        # Parse filename: client_1_round_0_train.json
        parts = filename.replace('.json', '').split('_')
        if len(parts) < 4:
            continue
            
        client_id = f"{parts[0]}_{parts[1]}"
        round_num = int(parts[2].replace('round', ''))
        file_type = parts[3]  # train, eval, or detailed
        
        if client_id not in clients_data:
            clients_data[client_id] = {'train': {}, 'eval': {}, 'detailed': {}}
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if file_type == 'train' and isinstance(data, list) and len(data) > 0:
            # Get the last entry (current round's metrics)
            clients_data[client_id]['train'][round_num] = data[-1]['metrics']
        elif file_type == 'eval' and isinstance(data, list) and len(data) > 0:
            clients_data[client_id]['eval'][round_num] = data[-1]['metrics']
        elif file_type == 'detailed':
            clients_data[client_id]['detailed'][round_num] = data
    
    return clients_data

def create_training_dashboard(clients_data, output_dir="results"):
    """Create comprehensive training dashboard"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('🚦 Federated Learning Traffic Control - Training Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    colors = {'client_1': '#2ecc71', 'client_2': '#3498db'}
    
    # 1. Average Reward Over Rounds
    ax1 = fig.add_subplot(3, 3, 1)
    for client_id, data in clients_data.items():
        rounds = sorted(data['train'].keys())
        rewards = [data['train'][r]['average_reward'] for r in rounds]
        ax1.plot(rounds, rewards, 'o-', label=client_id, color=colors.get(client_id, 'gray'), 
                linewidth=2, markersize=8)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('📈 Reward Progress', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Loss Over Rounds
    ax2 = fig.add_subplot(3, 3, 2)
    for client_id, data in clients_data.items():
        rounds = sorted(data['train'].keys())
        losses = [data['train'][r]['average_loss'] for r in rounds]
        ax2.plot(rounds, losses, 's-', label=client_id, color=colors.get(client_id, 'gray'),
                linewidth=2, markersize=8)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('📉 Training Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Waiting Time Over Rounds
    ax3 = fig.add_subplot(3, 3, 3)
    for client_id, data in clients_data.items():
        rounds = sorted(data['eval'].keys())
        waiting = [data['eval'][r].get('waiting_time', 0) for r in rounds]
        ax3.plot(rounds, waiting, '^-', label=client_id, color=colors.get(client_id, 'gray'),
                linewidth=2, markersize=8)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Waiting Time (s)', fontsize=12)
    ax3.set_title('⏱️ Total Waiting Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Queue Length Over Rounds
    ax4 = fig.add_subplot(3, 3, 4)
    for client_id, data in clients_data.items():
        rounds = sorted(data['eval'].keys())
        queues = [data['eval'][r].get('queue_length', 0) for r in rounds]
        ax4.plot(rounds, queues, 'd-', label=client_id, color=colors.get(client_id, 'gray'),
                linewidth=2, markersize=8)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Avg Queue Length', fontsize=12)
    ax4.set_title('🚗 Average Queue Length', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward Improvement Bar Chart
    ax5 = fig.add_subplot(3, 3, 5)
    client_ids = list(clients_data.keys())
    first_rewards = []
    last_rewards = []
    for client_id in client_ids:
        rounds = sorted(clients_data[client_id]['train'].keys())
        if rounds:
            first_rewards.append(clients_data[client_id]['train'][rounds[0]]['average_reward'])
            last_rewards.append(clients_data[client_id]['train'][rounds[-1]]['average_reward'])
    
    x = np.arange(len(client_ids))
    width = 0.35
    ax5.bar(x - width/2, first_rewards, width, label='First Round', color='#e74c3c', alpha=0.8)
    ax5.bar(x + width/2, last_rewards, width, label='Last Round', color='#27ae60', alpha=0.8)
    ax5.set_xlabel('Client', fontsize=12)
    ax5.set_ylabel('Reward', fontsize=12)
    ax5.set_title('🎯 Reward: First vs Last Round', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(client_ids)
    ax5.legend()
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Waiting Time Reduction
    ax6 = fig.add_subplot(3, 3, 6)
    first_waiting = []
    last_waiting = []
    for client_id in client_ids:
        rounds = sorted(clients_data[client_id]['eval'].keys())
        if rounds:
            first_waiting.append(clients_data[client_id]['eval'][rounds[0]].get('waiting_time', 0))
            last_waiting.append(clients_data[client_id]['eval'][rounds[-1]].get('waiting_time', 0))
    
    ax6.bar(x - width/2, first_waiting, width, label='First Round', color='#e74c3c', alpha=0.8)
    ax6.bar(x + width/2, last_waiting, width, label='Last Round', color='#27ae60', alpha=0.8)
    ax6.set_xlabel('Client', fontsize=12)
    ax6.set_ylabel('Waiting Time (s)', fontsize=12)
    ax6.set_title('⬇️ Waiting Time Reduction', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(client_ids)
    ax6.legend()
    
    # 7. Per-Lane Metrics Heatmap (Last Round)
    ax7 = fig.add_subplot(3, 3, 7)
    # Get last round detailed data for client_1
    if 'client_1' in clients_data and clients_data['client_1']['detailed']:
        last_round = max(clients_data['client_1']['detailed'].keys())
        detailed = clients_data['client_1']['detailed'][last_round]
        lane_metrics = detailed.get('per_lane_metrics', {})
        
        if lane_metrics:
            lanes = list(lane_metrics.keys())
            metrics = ['vehicle_count', 'queue_length', 'waiting_time']
            data_matrix = []
            for lane in lanes:
                row = [lane_metrics[lane].get(m, 0) for m in metrics]
                data_matrix.append(row)
            
            im = ax7.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')
            ax7.set_xticks(range(len(metrics)))
            ax7.set_xticklabels(['Vehicles', 'Queue', 'Wait Time'], fontsize=10)
            ax7.set_yticks(range(len(lanes)))
            ax7.set_yticklabels([l.replace('_0', '') for l in lanes], fontsize=10)
            ax7.set_title(f'🛣️ Lane Metrics (Round {last_round})', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax7)
    
    # 8. Green Signal Time Distribution
    ax8 = fig.add_subplot(3, 3, 8)
    if 'client_1' in clients_data and clients_data['client_1']['eval']:
        last_round = max(clients_data['client_1']['eval'].keys())
        gst = clients_data['client_1']['eval'][last_round].get('green_signal_time', {})
        per_edge = gst.get('per_edge', {})
        
        if per_edge:
            edges = list(per_edge.keys())
            times = [per_edge[e] for e in edges]
            colors_pie = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            ax8.pie(times, labels=[e.replace('A1', '') for e in edges], autopct='%1.1f%%',
                   colors=colors_pie[:len(edges)], startangle=90)
            ax8.set_title('🚦 Green Signal Time Distribution', fontsize=14, fontweight='bold')
    
    # 9. Summary Stats
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary stats
    summary_text = "📊 TRAINING SUMMARY\n" + "="*40 + "\n\n"
    
    for client_id in client_ids:
        rounds = sorted(clients_data[client_id]['train'].keys())
        if rounds:
            first_r = clients_data[client_id]['train'][rounds[0]]['average_reward']
            last_r = clients_data[client_id]['train'][rounds[-1]]['average_reward']
            improvement = ((last_r - first_r) / abs(first_r)) * 100 if first_r != 0 else 0
            
            first_w = clients_data[client_id]['eval'][rounds[0]].get('waiting_time', 0)
            last_w = clients_data[client_id]['eval'][rounds[-1]].get('waiting_time', 0)
            wait_reduction = ((first_w - last_w) / first_w) * 100 if first_w != 0 else 0
            
            summary_text += f"🔹 {client_id.upper()}\n"
            summary_text += f"   Rounds: {len(rounds)}\n"
            summary_text += f"   Reward: {first_r:.1f} → {last_r:.1f}\n"
            summary_text += f"   Improvement: {improvement:+.1f}%\n"
            summary_text += f"   Wait Time Reduction: {wait_reduction:.1f}%\n\n"
    
    # Check if reward is positive
    all_positive = all(
        clients_data[c]['train'][max(clients_data[c]['train'].keys())]['average_reward'] > 0
        for c in client_ids if clients_data[c]['train']
    )
    
    if all_positive:
        summary_text += "✅ STATUS: TRAINING SUCCESSFUL!\n"
        summary_text += "   Rewards are POSITIVE"
    else:
        summary_text += "⚠️ STATUS: TRAINING IN PROGRESS\n"
        summary_text += "   Rewards improving but not yet positive"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'training_dashboard_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Dashboard saved to: {output_path}")
    
    # Also save as latest
    latest_path = os.path.join(output_dir, 'training_dashboard_latest.png')
    plt.savefig(latest_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Latest dashboard: {latest_path}")
    
    plt.show()
    
    return output_path

def print_training_summary(clients_data):
    """Print text summary of training"""
    print("\n" + "="*60)
    print("🚦 FEDERATED LEARNING TRAINING SUMMARY")
    print("="*60)
    
    for client_id, data in clients_data.items():
        print(f"\n📍 {client_id.upper()}")
        print("-"*40)
        
        rounds = sorted(data['train'].keys())
        if not rounds:
            print("   No training data found")
            continue
            
        print(f"   Total Rounds: {len(rounds)}")
        
        # Reward trend
        rewards = [data['train'][r]['average_reward'] for r in rounds]
        print(f"\n   📈 REWARD:")
        print(f"      First: {rewards[0]:.2f}")
        print(f"      Last:  {rewards[-1]:.2f}")
        print(f"      Best:  {max(rewards):.2f} (Round {rounds[rewards.index(max(rewards))]})")
        print(f"      Trend: {'↑ Improving' if rewards[-1] > rewards[0] else '↓ Declining'}")
        
        # Loss trend
        losses = [data['train'][r]['average_loss'] for r in rounds]
        print(f"\n   📉 LOSS:")
        print(f"      First: {losses[0]:.4f}")
        print(f"      Last:  {losses[-1]:.4f}")
        print(f"      Trend: {'↓ Decreasing (Good!)' if losses[-1] < losses[0] else '↑ Increasing'}")
        
        # Waiting time
        if data['eval']:
            waiting = [data['eval'][r].get('waiting_time', 0) for r in rounds if r in data['eval']]
            if waiting:
                print(f"\n   ⏱️ WAITING TIME:")
                print(f"      First: {waiting[0]:.0f}s")
                print(f"      Last:  {waiting[-1]:.0f}s")
                reduction = ((waiting[0] - waiting[-1]) / waiting[0]) * 100 if waiting[0] > 0 else 0
                print(f"      Reduction: {reduction:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Loading results...")
    clients_data = load_all_results("results")
    
    if not clients_data:
        print("❌ No results found in 'results' directory")
        exit(1)
    
    print(f"Found data for {len(clients_data)} clients")
    
    # Print text summary
    print_training_summary(clients_data)
    
    # Create visual dashboard
    print("\nGenerating dashboard...")
    create_training_dashboard(clients_data)

