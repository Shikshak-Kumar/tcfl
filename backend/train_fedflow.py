import os
import json
import torch
import numpy as np
import random
from utils.model_exporter import ModelExporter, get_deployment_metadata
from typing import List, Dict, Tuple, Optional

from agents.mock_traffic_environment import MockTrafficEnvironment
from agents.fedflow_agent import FedFlowAgent
from federated_learning.fedflow_cluster import FedFlowCluster
from federated_learning.fedflow_server import FedFlowServer
from utils.logger import logger
from utils.sumo_scenario import (
    deployment_model_subdir,
    distinct_results_dir,
    effective_sumo_scenario,
    get_sumo_config_paths,
)


class FedFlowTrainer:
    """
    Orchestrator for Hierarchical Graph-Aware Federated RL.
    """

    def __init__(
        self,
        num_nodes: int = 6,
        num_clusters: int = 2,
        gui: bool = False,
        results_dir: str = "results_fedflow",
        use_tomtom: bool = False,
        target_pois: Optional[List[str]] = None,
        sumo_scenario: Optional[str] = None,
    ):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.gui = gui
        self.results_dir = results_dir
        self.use_tomtom = use_tomtom
        self.target_pois = target_pois
        self.sumo_scenario = sumo_scenario
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_round_results = []

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
        
        logger.section("Static Cluster Formation")
        table_headers = ["Cluster ID", "Members"]
        table_rows = []
        
        for c in range(num_clusters):
            c_nodes = [
                f"node_{i}"
                for i in range(c * nodes_per_cluster, (c + 1) * nodes_per_cluster)
            ]
            self.clusters.append(FedFlowCluster(f"cluster_{c}", c_nodes))
            table_rows.append([f"cluster_{c}", ", ".join(c_nodes)])
        
        logger.table(table_headers, table_rows)

        # 4. Level 2: Server
        self.server = FedFlowServer([c.cluster_id for c in self.clusters])

        # 5. Environments (Mock or SUMO)
        # Use real SUMO configs, cycling through available ones
        self.sumo_configs = get_sumo_config_paths(effective_sumo_scenario(sumo_scenario))
        self.envs = {}
        if not self.gui:
            if self.use_tomtom:
                from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
                from utils.tomtom_api import CITY_COORDINATES
                
                print("FedFlow-TSC: CLI Mode (TomTom Real-Time Traffic)")
                cities = list(CITY_COORDINATES.keys())
                from utils.tomtom_api import get_api_key
                api_key = get_api_key()
                for i in range(num_nodes):
                    config = self.sumo_configs[i % len(self.sumo_configs)]
                    city = cities[i % len(cities)]
                    lat, lon = CITY_COORDINATES[city]
                    self.envs[f"node_{i}"] = TomTomTrafficEnvironment(
                        sumo_config_path=config,
                        tomtom_api_key=api_key,
                        lat=lat,
                        lon=lon,
                        tl_id=f"{city}_{i}",
                        target_pois=self.target_pois
                    )
                    print(f"  node_{i} -> {config} ({city})")
            else:
                from agents.mock_traffic_environment import MockTrafficEnvironment

                print("FedFlow-TSC: CLI Mode (Mock with real config mapping)")
                for i in range(num_nodes):
                    config = self.sumo_configs[i % len(self.sumo_configs)]
                    self.envs[f"node_{i}"] = MockTrafficEnvironment(config)
                    print(f"  node_{i} -> {config}")
        else:
            from agents.traffic_environment import SUMOTrafficEnvironment

            print("FedFlow-TSC: GUI Mode (SUMO Simulation)")
            for i in range(num_nodes):
                config = self.sumo_configs[i % len(self.sumo_configs)]
                self.envs[f"node_{i}"] = SUMOTrafficEnvironment(config, gui=self.gui)
                print(f"  node_{i} -> {config}")

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
            adj[i, (i + 1) % self.num_nodes] = 1  # Connect to next
            adj[(i + 1) % self.num_nodes, i] = 1  # Bi-directional
        return adj

    def _get_node_graph_state(
        self, node_id: str, local_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the graph representation for a node.
        Includes local state and neighbor states.
        """
        node_idx = int(node_id.split("_")[1])
        neighbors = np.where(self.adj[node_idx] > 0)[0]

        # Focal node is at index 0
        node_states = [local_state]
        for nb in neighbors:
            if nb != node_idx:
                # In real life, we fetch neighbor state. Here we mock it based on local + noise
                node_states.append(
                    local_state + np.random.normal(0, 0.1, local_state.shape)
                )

        # Focal node + all neighbors
        state_graph = np.stack(node_states)
        adj_node = np.eye(len(state_graph))
        for i in range(len(state_graph)):
            for j in range(len(state_graph)):
                adj_node[i, j] = 1  # Fully connected neighborhood subgroup

        return state_graph, adj_node

    def run_round(self, round_idx: int):
        logger.header(f"FedFlow Round {round_idx}")

        # 1. Local Training (Nodes)
        node_metrics = {}
        node_losses = {}
        for nid, agent in self.agents.items():
            env = self.envs[nid]
            state = env.reset()
            total_reward = 0

            for _ in range(200):  # Steps per episode
                # FedFlow baseline uses local state (non-graph)
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                state = next_state
                total_reward += reward
                if done:
                    break

            loss = agent.replay()  # Replay memory
            metrics = env.get_performance_metrics()
            metrics["total_reward"] = total_reward
            node_metrics[nid] = metrics
            node_losses[nid] = loss

            wait_time = metrics.get("avg_waiting_time_per_vehicle", 0.0)
            if self.gui:
                wait_time = metrics.get("total_waiting_time", 0.0) / max(
                    1, metrics.get("total_vehicles", 1)
                )

            print(
                f"  {nid}: Reward={total_reward:.1f}, AvgWait={wait_time:.2f}s, Loss={loss:.4f}"
            )

            # Close SUMO connection after each node to avoid "already active" error
            if self.gui:
                env.stop_simulation()

        # 2. Intra-Cluster Aggregation (Level 1)
        cluster_params = []
        cluster_info = {}
        for cluster in self.clusters:
            for nid in cluster.agent_ids:
                throughput = node_metrics[nid].get("throughput_ratio", 0.0)
                cluster.update_flow(nid, float(throughput))

            agent_params = [self.agents[nid].get_weights() for nid in cluster.agent_ids]
            c_weights = cluster.aggregate_intra_cluster(agent_params)
            cluster_params.append(c_weights)

            for nid in cluster.agent_ids:
                self.agents[nid].set_weights(c_weights)

            avg_flow = cluster.get_avg_flow()
            cluster_info[cluster.cluster_id] = {
                "avg_flow": avg_flow,
                "agent_ids": cluster.agent_ids,
                "agent_flows": {k: v for k, v in cluster.agent_flows.items()},
            }
            print(f"  {cluster.cluster_id} Aggregated (Avg Flow: {avg_flow:.1f})")

        # 3. Inter-Cluster Aggregation (Level 2)
        cluster_congestion = {}
        for i, cluster in enumerate(self.clusters):
            wait_times = []
            for nid in cluster.agent_ids:
                wt = node_metrics[nid].get("avg_waiting_time_per_vehicle", 0.0)
                if self.gui:
                    wt = node_metrics[nid].get("avg_waiting_time_per_vehicle", 0.0)
                wait_times.append(wt)

            congestion = float(np.mean(wait_times))
            self.server.update_cluster_metrics(cluster.cluster_id, congestion)
            cluster_congestion[cluster.cluster_id] = congestion

        global_weights = self.server.aggregate_inter_cluster(cluster_params)

        # 4. Global Broadcast
        for nid in self.agents:
            self.agents[nid].set_weights(global_weights)

        print(f"  Global Meta-Aggregation Complete.")

        # 5. Standardized Performance Table
        mode_str = "SUMO" if self.gui else "Mock"
        logger.section(f"Round {round_idx} Client Performance Summary")
        
        table_headers = ["Client ID", "Cluster", "Reward", "Wait", "Queue", "TP Ratio"]
        table_rows = []

        round_results = {
            "round": round_idx,
            "mode": mode_str,
            "nodes": {},
            "clusters": cluster_info,
            "cluster_congestion": cluster_congestion,
        }

        for nid in sorted(self.agents.keys()):
            m = node_metrics[nid]
            wt = f"{m.get('avg_waiting_time_per_vehicle', 0):.2f}s"
            tp = f"{m.get('throughput_ratio', 0):.2f}"
            rw = f"{m.get('total_reward', 0):.1f}"
            aq = f"{m.get('average_queue_length', 0):.1f}"

            cid = "Unknown"
            for c in self.clusters:
                if nid in c.agent_ids:
                    cid = c.cluster_id
                    break

            table_rows.append([nid, cid, rw, wt, aq, tp])

            # Save per-node results
            node_result = {
                "node_id": nid,
                "cluster_id": cid,
                "round": round_idx,
                "total_reward": m.get("total_reward", 0),
                "avg_waiting_time": wt,
                "total_vehicles": tp,
                "loss": node_losses.get(nid, 0.0),
                "mode": mode_str,
                "metrics": m,
            }
            round_results["nodes"][nid] = node_result

            # Save per-node JSON file
            node_file = os.path.join(
                self.results_dir, f"{nid}_round_{round_idx}_eval.json"
            )
            with open(node_file, "w") as f:
                json.dump(convert_to_json_serializable(node_result), f, indent=2)
                
            # Save the local model for this round
            model_path = os.path.join(self.results_dir, f"{nid}_round_{round_idx}_model.pt")
            self.agents[nid].save_model(model_path)

        # Also save the GLOBAL aggregated model for this round
        global_model_path = os.path.join(self.results_dir, f"global_round_{round_idx}_model.pt")
        # Just use node_0 since all nodes have global weights now
        self.agents["node_0"].save_model(global_model_path)

        print(f"{'-' * 90}")

        # Save round summary JSON
        round_file = os.path.join(self.results_dir, f"round_{round_idx}_summary.json")
        with open(round_file, "w") as f:
            json.dump(convert_to_json_serializable(round_results), f, indent=2)

        self.all_round_results.append(round_results)
        print(f"  Results saved to {self.results_dir}/")

    def train(self, num_rounds: int = 3):
        """Run num_rounds of hierarchical federated training."""
        for r in range(1, num_rounds + 1):
            self.run_round(r)

        # Save final global model weights
        mode_label = "sumo" if self.gui else "mock"
        global_model_path = os.path.join(
            self.results_dir, f"fedflow_global_{mode_label}.pt"
        )
        # All agents share global weights after broadcast in run_round
        self.agents["node_0"].save_model(global_model_path)
        print(f"\n[Final] Global model saved to {global_model_path}")

        # 4. EXPORT TO Production (saved_models/)
        try:
            print(f"\n[Production] Exporting optimized model for deployment...")
            agent = self.agents["node_0"]
            metadata = get_deployment_metadata("fedflow", agent)
            metadata["mode"] = mode_label
            metadata["source_weights"] = global_model_path
            metadata["sumo_scenario"] = effective_sumo_scenario(self.sumo_scenario)

            save_path = os.path.join(
                "saved_models",
                deployment_model_subdir("fedflow", self.sumo_scenario),
            )
            ModelExporter.export(agent.policy_net, metadata, save_path)
        except Exception as e:
            print(f"[Production] Warning: Export failed: {e}")


def convert_to_json_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    else:
        return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedFlow-TSC Training")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--nodes", type=int, default=6, help="Number of nodes")
    parser.add_argument("--clusters", type=int, default=0, help="Number of clusters (0 for auto)")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_fedflow",
        help="Results directory (default + --sumo-scenario china → results_fedflow_china)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO GUI (Enforces real SUMO simulation)",
    )
    parser.add_argument(
        "--use-tomtom", action="store_true", help="Use real-time TomTom traffic data"
    )
    parser.add_argument(
        "--target-pois",
        type=str,
        default=None,
        help="Comma-separated list of target POI categories",
    )
    parser.add_argument(
        "--sumo-scenario",
        type=str,
        default=None,
        choices=["default", "china"],
        help="SUMO map preset: default=osm clients, china=sumo_configs_china (omit flag to use SUMO_SCENARIO env)",
    )

    args = parser.parse_args()

    results_dir = distinct_results_dir(
        "results_fedflow", args.results_dir, args.sumo_scenario
    )
    if results_dir != args.results_dir:
        print(
            f"[FedFlow] China scenario: writing results to {results_dir}/ (distinct from default OSM runs)"
        )

    # Automatic cluster selection for performance optimization
    num_clusters = args.clusters
    if num_clusters <= 0:
        num_clusters = max(1, (args.nodes + 3) // 4)
        print(f"\n[AUTO] Optimizing performance: Selected {num_clusters} aggregation servers for {args.nodes} nodes.")

    # Parse target_pois if provided
    target_pois_list = None
    if args.target_pois:
        target_pois_list = [p.strip() for p in args.target_pois.split(",")]

    trainer = FedFlowTrainer(
        num_nodes=args.nodes,
        num_clusters=num_clusters,
        gui=args.gui,
        results_dir=results_dir,
        use_tomtom=args.use_tomtom,
        target_pois=target_pois_list,
        sumo_scenario=args.sumo_scenario,
    )
    trainer.train(num_rounds=args.rounds)

    # Save final combined results
    final_file = os.path.join(results_dir, "fedflow_all_rounds.json")
    with open(final_file, "w") as f:
        json.dump(convert_to_json_serializable(trainer.all_round_results), f, indent=2)

    print(f"\nFedFlow-TSC Training Completed. All results saved to {results_dir}/")
