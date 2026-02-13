import os
import torch
import numpy as np
import random
from typing import List, Dict, Tuple

from agents.mock_traffic_environment import MockTrafficEnvironment
from agents.fedflow_agent import FedFlowAgent
from federated_learning.fedflow_cluster import FedFlowCluster
from federated_learning.fedflow_server import FedFlowServer

class FedFlowTrainer:
    """
    Orchestrator for Hierarchical Graph-Aware Federated RL.
    """
    def __init__(self, num_nodes: int = 6, num_clusters: int = 2, gui: bool = False):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.gui = gui
        
        # 1. Graph Setup
        self.adj = self._create_mock_graph()
        
        # 2. Level 0: Local Agents
        self.agents = {}
        for i in range(num_nodes):
            aid = f"node_{i}"
            self.agents[aid] = FedFlowAgent(state_size=12, action_size=4)
            
        # 3. Level 1: Clusters
        self.clusters = []
        nodes_per_cluster = num_nodes // num_clusters
        for c in range(num_clusters):
            c_nodes = [f"node_{i}" for i in range(c*nodes_per_cluster, (c+1)*nodes_per_cluster)]
            self.clusters.append(FedFlowCluster(f"cluster_{c}", c_nodes))
            
        # 4. Level 2: Server
        self.server = FedFlowServer([c.cluster_id for c in self.clusters])
        
        # 5. Environments (Mock or SUMO)
        self.envs = {}
        if not self.gui:
            from agents.mock_traffic_environment import MockTrafficEnvironment
            print("FedFlow-TSC: CLI Mode (Strictly Mock)")
            for i in range(num_nodes):
                self.envs[f"node_{i}"] = MockTrafficEnvironment(f"config_{i}")
        else:
            from agents.traffic_environment import SUMOTrafficEnvironment
            print("FedFlow-TSC: GUI Mode (Strictly SUMO)")
            for i in range(num_nodes):
                config = f"sumo_configs2/osm.netccfg"
                self.envs[f"node_{i}"] = SUMOTrafficEnvironment(config, gui=self.gui)

        # 6. Wire up neighbors for Coupled Flow (Mock only)
        if not self.gui:
            for i in range(num_nodes):
                nid = f"node_{i}"
                for j in range(num_nodes):
                    if i != j and self.adj[i, j] > 0:
                        self.envs[nid].add_neighbor(self.envs[f"node_{j}"])
        
    def _create_mock_graph(self) -> np.ndarray:
        """Create a ring-like adjacency matrix for the nodes."""
        adj = np.eye(self.num_nodes)
        for i in range(self.num_nodes):
            adj[i, (i+1) % self.num_nodes] = 1 # Connect to next
            adj[(i+1) % self.num_nodes, i] = 1 # Bi-directional
        return adj

    def _get_node_graph_state(self, node_id: str, local_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the graph representation for a node.
        Includes local state and neighbor states.
        """
        node_idx = int(node_id.split('_')[1])
        neighbors = np.where(self.adj[node_idx] > 0)[0]
        
        # Focal node is at index 0
        node_states = [local_state]
        for nb in neighbors:
            if nb != node_idx:
                # In real life, we fetch neighbor state. Here we mock it based on local + noise
                node_states.append(local_state + np.random.normal(0, 0.1, local_state.shape))
        
        # Pad or truncate to fixed size for batching if needed
        # For simplicity, we assume fixed neighbors in this mock step
        # Let's say max 3 nodes total (focal + 2 neighbors)
        state_graph = np.stack(node_states[:3]) 
        adj_node = np.eye(len(state_graph))
        for i in range(len(state_graph)):
            for j in range(len(state_graph)):
                adj_node[i,j] = 1 # Fully connected neighborhood subgroup
                
        return state_graph, adj_node

    def run_round(self, round_idx: int):
        print(f"\n--- FedFlow Bound {round_idx} ---")
        
        # 1. Local Training (Nodes)
        node_metrics = {}
        for nid, agent in self.agents.items():
            env = self.envs[nid]
            state = env.reset()
            total_reward = 0
            
            # PressLight Reward Calculation (Mocked via environment pressure)
            # Pressure = incoming_queue (MockEnv track this) - outgoing (mocked)
            
            for _ in range(200): # Steps per episode
                state_graph, adj_node = self._get_node_graph_state(nid, state)
                action = agent.get_action(state_graph, adj_node)
                next_state, reward, done, info = env.step(action)
                
                # PressLight refinement: Use actual pressure from info
                # MockEnv reward is already queue-based, which correlates with pressure
                
                next_state_graph, next_adj_node = self._get_node_graph_state(nid, next_state)
                agent.remember(state_graph, adj_node, action, reward, 
                               next_state_graph, next_adj_node, done)
                
                state = next_state
                total_reward += reward
                if done: break
                
            loss = agent.replay()
            metrics = env.get_performance_metrics()
            metrics['total_reward'] = total_reward # Inject for table
            node_metrics[nid] = metrics
            
            # Ensure wait time unit is correct
            wait_time = metrics.get('avg_waiting_time_per_vehicle', 0.0)
            if self.gui: # SUMO performance metrics structure is complex
                wait_time = metrics.get('total_waiting_time', 0.0) / max(1, metrics.get('total_vehicles', 1))

            print(f"  {nid}: Reward={total_reward:.1f}, AvgWait={wait_time:.2f}s")
            
        # 2. Intra-Cluster Aggregation (Level 1)
        cluster_params = []
        for cluster in self.clusters:
            # Update flow magnitude per node (using throughput from metrics)
            for nid in cluster.agent_ids:
                throughput = node_metrics[nid]['total_vehicles']
                cluster.update_flow(nid, float(throughput))
            
            # Aggregate local weights
            agent_params = [self.agents[nid].get_weights() for nid in cluster.agent_ids]
            c_weights = cluster.aggregate_intra_cluster(agent_params)
            cluster_params.append(c_weights)
            
            # Update local agents with cluster weights (broadcast down)
            for nid in cluster.agent_ids:
                self.agents[nid].set_weights(c_weights)
                
            print(f"  {cluster.cluster_id} Aggregated (Avg Flow: {cluster.get_avg_flow():.1f})")
            
        # 3. Inter-Cluster Aggregation (Level 2)
        for i, cluster in enumerate(self.clusters):
            # Congestion = Avg Waiting Time
            wait_times = []
            for nid in cluster.agent_ids:
                wt = node_metrics[nid].get('avg_waiting_time_per_vehicle', 0.0)
                if self.gui:
                    wt = node_metrics[nid].get('total_waiting_time', 0.0) / max(1, node_metrics[nid].get('total_vehicles', 1))
                wait_times.append(wt)
                
            congestion = np.mean(wait_times)
            self.server.update_cluster_metrics(cluster.cluster_id, float(congestion))
            
        global_weights = self.server.aggregate_inter_cluster(cluster_params)
        
        # 4. Global Broadcast (Back to clusters and then nodes)
        for nid in self.agents:
            self.agents[nid].set_weights(global_weights)
            
        print(f"  Global Meta-Aggregation Complete.")
        
        # 5. Standardized Performance Table
        mode_str = "SUMO" if self.gui else "Mock"
        print(f"\nRound {round_idx} Client Performance Summary:")
        print(f"{'-'*90}")
        print(f"{'Client ID':<12} | {'Context/Arch':<16} | {'Reward':<12} | {'Avg Wait (s)':<14} | {'Throughput':<10} | {'Mode':<6}")
        print(f"{'-'*90}")
        for nid in sorted(self.agents.keys()):
            m = node_metrics[nid]
            wt = m.get('avg_waiting_time_per_vehicle', 0.0)
            if self.gui:
                wt = m.get('total_waiting_time', 0.0) / max(1, m.get('total_vehicles', 1))
            tp = m.get('total_vehicles', 0)
            
            # Find which cluster this node belongs to
            cid = "Unknown"
            for c in self.clusters:
                if nid in c.agent_ids:
                    cid = c.cluster_id
                    break
                    
            print(f"{nid:<12} | {cid:<16} | {m.get('total_reward', 0):>12.1f} | {wt:>14.2f} | {tp:>10} | {mode_str:<6}")
        print(f"{'-'*90}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FedFlow-TSC Training")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--nodes", type=int, default=6, help="Number of nodes")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI (Enforces real SUMO simulation)")
    args = parser.parse_args()

    trainer = FedFlowTrainer(num_nodes=args.nodes, num_clusters=args.clusters, gui=args.gui)
    for r in range(1, args.rounds + 1):
        trainer.run_round(r)
    print("\nFedFlow-TSC Training Completed.")
