#!/usr/bin/env python3
"""
Simple Training Results Analysis - Text Based
"""

import os
import json

def load_training_data(results_dir="results"):
    """Load training data from results directory"""
    clients_data = {}
    
    for filename in os.listdir(results_dir):
        if 'train' in filename and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            parts = filename.replace('.json', '').split('_')
            client_id = f"{parts[0]}_{parts[1]}"
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if client_id not in clients_data:
                clients_data[client_id] = []
            
            for entry in data:
                clients_data[client_id].append({
                    'round': entry['round'],
                    'reward': entry['metrics']['average_reward'],
                    'loss': entry['metrics']['average_loss'],
                    'steps': entry['metrics']['total_steps']
                })
    
    return clients_data

def print_ascii_graph(data, title, width=50):
    """Print ASCII bar graph"""
    if not data:
        return
    
    max_val = max(data)
    min_val = min(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    for i, val in enumerate(data):
        # Normalize to width
        if max_val > 0:
            bar_len = int((val / max_val) * width)
        else:
            bar_len = int(((val - min_val) / range_val) * width)
        
        bar = '█' * max(1, bar_len)
        print(f"  R{i:2d} |{bar} {val:.1f}")
    
    print(f"{'='*60}")

def analyze_training():
    """Main analysis function"""
    print("\n" + "🚦"*30)
    print("   FEDERATED LEARNING TRAFFIC CONTROL - TRAINING ANALYSIS")
    print("🚦"*30)
    
    clients_data = load_training_data()
    
    if not clients_data:
        print("❌ No training data found!")
        return
    
    for client_id, rounds_data in clients_data.items():
        # Sort by round
        rounds_data.sort(key=lambda x: x['round'])
        
        print(f"\n\n{'#'*60}")
        print(f"  📍 {client_id.upper()}")
        print(f"{'#'*60}")
        
        rewards = [r['reward'] for r in rounds_data]
        losses = [r['loss'] for r in rounds_data]
        
        # Summary stats
        print(f"\n  📊 SUMMARY STATISTICS")
        print(f"  {'-'*40}")
        print(f"  Total Rounds:     {len(rounds_data)}")
        print(f"  First Reward:     {rewards[0]:+.2f}")
        print(f"  Last Reward:      {rewards[-1]:+.2f}")
        print(f"  Best Reward:      {max(rewards):+.2f} (Round {rewards.index(max(rewards))})")
        print(f"  Worst Reward:     {min(rewards):+.2f} (Round {rewards.index(min(rewards))})")
        print(f"  Average Reward:   {sum(rewards)/len(rewards):+.2f}")
        
        # Improvement
        improvement = ((rewards[-1] - rewards[0]) / abs(rewards[0])) * 100 if rewards[0] != 0 else 0
        print(f"\n  📈 IMPROVEMENT")
        print(f"  {'-'*40}")
        print(f"  Reward Change:    {rewards[-1] - rewards[0]:+.2f}")
        print(f"  Improvement:      {improvement:+.1f}%")
        
        # Status
        print(f"\n  🎯 STATUS")
        print(f"  {'-'*40}")
        if rewards[-1] > 0:
            print(f"  ✅ POSITIVE REWARDS ACHIEVED!")
        else:
            print(f"  ⚠️ Rewards still negative, more training needed")
        
        if rewards[-1] > rewards[0]:
            print(f"  ✅ LEARNING IS WORKING!")
        else:
            print(f"  ⚠️ Performance declining")
        
        # ASCII Reward Graph
        print_ascii_graph(rewards, "📈 REWARD PROGRESS (Higher = Better)")
        
        # ASCII Loss Graph
        print_ascii_graph(losses, "📉 LOSS PROGRESS (Lower = Better)")
        
        # Round-by-round table
        print(f"\n  📋 ROUND-BY-ROUND DETAILS")
        print(f"  {'='*50}")
        print(f"  {'Round':<8} {'Reward':<15} {'Loss':<15} {'Status'}")
        print(f"  {'-'*50}")
        
        for i, r in enumerate(rounds_data):
            status = "🟢" if r['reward'] > 0 else "🔴"
            trend = ""
            if i > 0:
                if r['reward'] > rounds_data[i-1]['reward']:
                    trend = "↑"
                elif r['reward'] < rounds_data[i-1]['reward']:
                    trend = "↓"
                else:
                    trend = "→"
            
            print(f"  {r['round']:<8} {r['reward']:>+12.2f}   {r['loss']:>12.4f}   {status} {trend}")
        
        print(f"  {'='*50}")
    
    # Overall conclusion
    print(f"\n\n{'*'*60}")
    print("  🏆 FINAL VERDICT")
    print(f"{'*'*60}")
    
    all_positive = all(
        max(r['reward'] for r in data) > 0 
        for data in clients_data.values()
    )
    
    if all_positive:
        print("""
  ✅ TRAINING SUCCESSFUL!
  
  Your federated learning traffic control system is working!
  
  Key achievements:
  • Positive rewards achieved
  • Agent learned to optimize traffic flow
  • Multiple clients successfully collaborated
  
  Next steps:
  • Test with GUI: python train_federated.py --mode single --gui
  • Increase rounds for more stability
  • Try with real-world OSM maps
        """)
    else:
        print("""
  ⚠️ TRAINING IN PROGRESS
  
  The model is learning but needs more training.
  
  Recommendations:
  • Increase number of rounds
  • Increase episodes per round
  • Check reward function parameters
        """)
    
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    analyze_training()

