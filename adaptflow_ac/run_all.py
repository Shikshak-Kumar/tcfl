import subprocess
import argparse
import os

def run_command(command):
    print(f"\n>>> Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Error executing command: {command}")

def main():
    parser = argparse.ArgumentParser(description="AdaptFlow Research Suite: Unified Runner")
    parser.add_argument("--sumocfg", type=str, required=True, help="Path to .sumocfg file")
    parser.add_argument("--action", type=str, choices=['train', 'compare', 'all'], default='all', 
                        help="Action to perform: train all, compare all, or both (all)")
    parser.add_argument("--nodes", type=int, default=4, help="Number of intersections")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds/Episodes for training")
    
    args = parser.parse_args()
    
    sumocfg = args.sumocfg
    nodes = args.nodes
    rounds = args.rounds
    
    if args.action in ['train', 'all']:
        print("\n" + "="*60)
        print("PHASE 1: TRAINING ALL 4 ALGORITHMS")
        print("="*60)
        
        # 1. AdaptFlow-AC
        run_command(f"python train/train_adaptflow.py --sumocfg {sumocfg} --rounds {rounds} --nodes {nodes}")
        
        # 2. FedDQNTsc (2023)
        run_command(f"python train/train_fed_dqn_tsc.py --sumocfg {sumocfg} --rounds {rounds} --nodes {nodes}")
        
        # 3. MultiAgentAC (2019)
        run_command(f"python train/train_multi_agent_ac.py --sumocfg {sumocfg} --nodes {nodes}")
        
        # 4. DQTSCA (Single Agent CNN)
        run_command(f"python train/train_dqtsca.py --sumocfg {sumocfg}")

    if args.action in ['compare', 'all']:
        print("\n" + "="*60)
        print("PHASE 2: RUNNING COMPREHENSIVE COMPARISON")
        print("="*60)
        
        run_command(f"python compare.py --mode sumo --sumocfg {sumocfg} --episodes 5")

if __name__ == "__main__":
    main()
