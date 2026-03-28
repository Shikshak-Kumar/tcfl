"""
AdaptFlow-TSC Training Script.

Adaptive Dynamic Clustering + Hierarchical Graph-Aware Federated RL.
Extends FedFlow with per-round congestion-based re-clustering.

Novel: Intersections are dynamically re-grouped every round based on
real-time congestion fingerprint similarity, so dissimilar intersections
no longer pollute each other's model updates.

Usage:
  python train_adaptflow.py --rounds 3 --nodes 6 --clusters 2
  python train_adaptflow.py --rounds 5 --nodes 6 --clusters 3 --use-tomtom
"""

import os
import json
import torch
import numpy as np
import random
from utils.model_exporter import ModelExporter, get_deployment_metadata
from typing import List, Dict, Tuple, Optional

from agents.mock_traffic_environment import MockTrafficEnvironment
from agents.adaptflow_agent import AdaptFlowAgent
from federated_learning.fedflow_cluster import FedFlowCluster
from federated_learning.fedflow_server import FedFlowServer
from federated_learning.adaptive_clustering import (
    AdaptiveClusterManager,
    extract_fingerprint,
    cosine_similarity_matrix,
)
from utils.logger import logger
from utils.sumo_scenario import (
    distinct_results_dir,
    effective_sumo_scenario,
    get_sumo_config_paths,
    deployment_model_subdir,
)


class AdaptFlowTrainer:
    """
    Orchestrator for AdaptFlow: Adaptive Dynamic Clustering
    + Hierarchical Graph-Aware Federated RL.

    Key difference from FedFlowTrainer:
      - Round 1: static clustering (no prior metrics)
      - Round 2+: re-cluster based on congestion fingerprints
    """

    def __init__(
        self,
        num_nodes: int = 6,
        num_clusters: int = 2,
        gui: bool = False,
        results_dir: str = "results_adaptflow",
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

        # 1. Graph Setup (ring topology)
        self.adj = self._create_mock_graph()

        # 2. Local Agents
        self.agents: Dict[str, AdaptFlowAgent] = {}
        for i in range(num_nodes):
            self.agents[f"node_{i}"] = AdaptFlowAgent(state_size=12, action_size=4)

        # 3. Adaptive Cluster Manager (THE NOVELTY)
        self.cluster_manager = AdaptiveClusterManager(
            num_clusters=num_clusters, seed=42
        )

        # 4. Server (Level 2 — inter-cluster aggregation)
        # Cluster IDs will be dynamically created each round
        self.server = None

        # 5. Environments
        self.sumo_configs = get_sumo_config_paths(effective_sumo_scenario(sumo_scenario))
        self.envs: Dict[str, object] = {}
        self._setup_environments()

        # 6. Wire neighbors (Mock only)
        self.priority_tiers: Dict[str, int] = {}
        if not self.gui:
            for i in range(num_nodes):
                nid = f"node_{i}"
                # Assign mock priority tiers (Tier 1 for node_0, Tier 2 for node_1, etc.)
                # In real use, this comes from the environment config
                if i == 0:
                    self.priority_tiers[nid] = 1  # Hospital
                elif i == 1:
                    self.priority_tiers[nid] = 2  # School
                else:
                    self.priority_tiers[nid] = 3  # Normal

                for j in range(num_nodes):
                    if i != j and self.adj[i, j] > 0:
                        self.envs[nid].add_neighbor(self.envs[f"node_{j}"])

    def _setup_environments(self):
        """Initialize traffic environments for each node."""
        if not self.gui:
            if self.use_tomtom:
                from agents.tomtom_traffic_environment import TomTomTrafficEnvironment
                from utils.tomtom_api import CITY_COORDINATES

                print("AdaptFlow-TSC: CLI Mode (TomTom Real-Time Traffic)")
                cities = list(CITY_COORDINATES.keys())
                from utils.tomtom_api import get_api_key

                api_key = get_api_key()
                for i in range(self.num_nodes):
                    config = self.sumo_configs[i % len(self.sumo_configs)]
                    city = cities[i % len(cities)]
                    lat, lon = CITY_COORDINATES[city]
                    self.envs[f"node_{i}"] = TomTomTrafficEnvironment(
                        sumo_config_path=config,
                        tomtom_api_key=api_key,
                        lat=lat,
                        lon=lon,
                        tl_id=f"{city}_{i}",
                        target_pois=self.target_pois,
                    )
                    print(f"  node_{i} -> {config} ({city})")
            else:
                print("AdaptFlow-TSC: CLI Mode (Mock with real config mapping)")
                for i in range(self.num_nodes):
                    config = self.sumo_configs[i % len(self.sumo_configs)]
                    self.envs[f"node_{i}"] = MockTrafficEnvironment(config)
                    print(f"  node_{i} -> {config}")
        else:
            from agents.traffic_environment import SUMOTrafficEnvironment

            print("AdaptFlow-TSC: GUI Mode (SUMO Simulation)")
            for i in range(self.num_nodes):
                config = self.sumo_configs[i % len(self.sumo_configs)]
                self.envs[f"node_{i}"] = SUMOTrafficEnvironment(config, gui=self.gui)
                print(f"  node_{i} -> {config}")

    def _create_mock_graph(self) -> np.ndarray:
        """Create a ring-like adjacency matrix."""
        adj = np.eye(self.num_nodes)
        for i in range(self.num_nodes):
            adj[i, (i + 1) % self.num_nodes] = 1
            adj[(i + 1) % self.num_nodes, i] = 1
        return adj

    def _get_node_graph_state(
        self, node_id: str, local_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct graph representation: local + neighbor states."""
        node_idx = int(node_id.split("_")[1])
        neighbors = np.where(self.adj[node_idx] > 0)[0]

        node_states = [local_state]
        for nb in neighbors:
            if nb != node_idx:
                node_states.append(
                    local_state + np.random.normal(0, 0.1, local_state.shape)
                )

        state_graph = np.stack(node_states)
        adj_node = np.ones((len(state_graph), len(state_graph)))
        return state_graph, adj_node
    def run_round(self, round_idx: int):
        """Execute one round of AdaptFlow training."""
        logger.header(f"ADAPTFLOW ROUND {round_idx}")

        # ── Step 1: Local Training ───────────────────────────────────
        print(f"\n  [Step 1] Local Training...")
        node_metrics = {}
        node_losses = {}

        for nid, agent in self.agents.items():
            env = self.envs[nid]
            state = env.reset()
            total_reward = 0

            for _ in range(200):
                state_graph, adj_node = self._get_node_graph_state(nid, state)
                state_seq = agent._get_sequence(state_graph)
                action = agent.get_action(state_graph, adj_node)
                next_state, reward, done, info = env.step(action)

                next_state_graph, next_adj_node = self._get_node_graph_state(nid, next_state)
                next_state_seq = agent._get_sequence(next_state_graph)

                agent.remember(state_seq, adj_node, action, reward, next_state_seq, next_adj_node, done)

                state = next_state
                total_reward += reward
                if done:
                    break

            loss = agent.replay()
            metrics = env.get_performance_metrics()
            metrics["total_reward"] = total_reward
            node_metrics[nid] = metrics
            node_losses[nid] = loss

            p_rews = info.get("pareto_rewards", {})
            wait_time = metrics.get("avg_waiting_time_per_vehicle", 0.0)
            print(
                f"    {nid}: Reward={total_reward:.1f} (Q:{p_rews.get('queue', 0):.2f}, W:{p_rews.get('wait', 0):.2f}), "
                f"AvgWait={wait_time:.2f}s, Loss={loss:.4f}"
            )

            if self.gui:
                env.stop_simulation()

        # ── Step 2: Dynamic Re-Clustering ───────────────────────────
        if round_idx == 1:
            logger.section("Step 2: Initial Static Clustering")
            nodes_per_cluster = self.num_nodes // self.num_clusters
            assignments = {f"node_{i}": i // nodes_per_cluster for i in range(self.num_nodes)}
            # Initial clustering logic calls manager
            self.cluster_manager.recluster(node_metrics, round_idx, self.priority_tiers)
            self.cluster_manager.cluster_history[-1] = assignments
        else:
            logger.section("Step 2: Dynamic Re-Clustering (Congestion + POI Priority)")
            assignments = self.cluster_manager.recluster(node_metrics, round_idx, self.priority_tiers)

            transitions = self.cluster_manager.get_latest_transitions()
            if transitions["transitions"]:
                for nid, change in transitions["transitions"].items():
                    logger.warning(f"Cluster Transition: {nid} moved from cluster_{change['from']} to cluster_{change['to']}", prefix="REFRESH")
            else:
                logger.success("Cluster membership remains stable.", prefix="STABLE")

        # Cluster metadata for metrics
        cluster_groups = self.cluster_manager.get_cluster_groups(assignments)
        
        # Display Cluster Performance Summary
        table_headers = ["Cluster ID", "Nodes", "Avg Reward", "Avg Wait (s)", "Avg Queue"]
        table_rows = []
        for cid, members in sorted(cluster_groups.items()):
            avg_rew = np.mean([node_metrics[nid].get("total_reward", 0) for nid in members])
            avg_wait = np.mean([node_metrics[nid].get("avg_waiting_time_per_vehicle", 0) for nid in members])
            avg_queue = np.mean([node_metrics[nid].get("average_queue_length", 0) for nid in members])
            table_rows.append([f"cluster_{cid}", len(members), f"{avg_rew:.1f}", f"{avg_wait:.2f}s", f"{avg_queue:.2f}"])
        
        logger.table(table_headers, table_rows)
        
        # Clustering Insights
        fingerprints = self.cluster_manager.fingerprint_history[-1]
        node_ids_sim, sim_matrix = cosine_similarity_matrix(fingerprints)
        avg_sim = np.mean(sim_matrix)
        mean_fp_mag = np.mean([np.linalg.norm(fp) for fp in fingerprints.values()])
        
        print(f"\n    [Clustering Analysis] Round {round_idx}")
        print(f"    - Average Pairwise Similarity Index: {avg_sim:.4f}")
        print(f"    - Mean Fingerprint Magnitude: {mean_fp_mag:.4f}")

        # Similarity Matrix Table
        sim_headers = ["Similarity"] + [f"n_{nid.split('_')[1]}" for nid in node_ids_sim]
        sim_rows = []
        for i, nid in enumerate(node_ids_sim):
            row = [nid] + [f"{sim_matrix[i][j]:.2f}" for j in range(len(node_ids_sim))]
            sim_rows.append(row)
        logger.table(sim_headers, sim_rows)

        # ── Step 3: Build Clusters & Aggregate ──────────────────────
        print(f"\n  [Step 3] Hierarchical Aggregation")
        clusters = []
        cluster_ids = sorted(cluster_groups.keys())
        for cid in cluster_ids:
            members = cluster_groups[cid]
            cluster = FedFlowCluster(f"cluster_{cid}", members)
            clusters.append(cluster)

        # Intra-cluster aggregation
        cluster_params = []
        cluster_info = {}
        for cluster in clusters:
            for nid in cluster.agent_ids:
                throughput = node_metrics[nid].get("throughput_ratio", 0.0)
                cluster.update_flow(nid, max(float(throughput), 0.01))

            agent_params = [self.agents[nid].get_weights() for nid in cluster.agent_ids]
            c_weights = cluster.aggregate_intra_cluster(agent_params)
            cluster_params.append(c_weights)

            for nid in cluster.agent_ids:
                self.agents[nid].set_weights(c_weights)

            avg_flow = cluster.get_avg_flow()
            cluster_info[cluster.cluster_id] = {
                "avg_flow": avg_flow,
                "members": cluster.agent_ids,
                "congestion": float(np.mean([node_metrics[nid].get("avg_waiting_time_per_vehicle", 0) for nid in cluster.agent_ids])),
            }
            print(f"    {cluster.cluster_id}: {len(cluster.agent_ids)} nodes, avg_flow={avg_flow:.3f}, congestion={cluster_info[cluster.cluster_id]['congestion']:.2f}s")

        # Inter-cluster aggregation
        self.server = FedFlowServer([c.cluster_id for c in clusters])
        for cluster in clusters:
            cong = cluster_info[cluster.cluster_id]["congestion"]
            self.server.update_cluster_metrics(cluster.cluster_id, cong)

        global_weights = self.server.aggregate_inter_cluster(cluster_params)
        for nid in self.agents:
            self.agents[nid].set_weights(global_weights)

        # ── Step 4: Save & Final Terminal Table ─────────────────────
        mode_str = "SUMO" if self.gui else "Mock"
        round_results = {
            "round": round_idx,
            "mode": mode_str,
            "clustering": {
                "assignments": assignments,
                "groups": {str(k): v for k, v in cluster_groups.items()},
                "fingerprints": {nid: fp.tolist() for nid, fp in fingerprints.items()},
                "similarity_matrix": sim_matrix.tolist(),
                "similarity_node_order": node_ids_sim,
            },
            "cluster_info": cluster_info,
            "nodes": {},
        }

        logger.section(f"Round {round_idx} End-to-End Performance Summary")
        table_headers = ["Node", "Cluster", "Reward", "Wait Time", "Queue", "TP Ratio", "Loss"]
        table_rows = []

        for nid in sorted(self.agents.keys()):
            m = node_metrics[nid]
            wt = m.get("avg_waiting_time_per_vehicle", 0.0)
            tp = m.get("throughput_ratio", 0)
            cid = assignments.get(nid, -1)
            aq = m.get("average_queue_length", 0.0)

            table_rows.append([nid, f"cluster_{cid}", f"{m.get('total_reward', 0):.1f}", f"{wt:.2f}s", f"{aq:.1f}", f"{tp:.2f}", f"{node_losses.get(nid, 0.0):.4f}"])
        
            round_results["nodes"][nid] = {
                "node_id": nid, "cluster_id": f"cluster_{cid}", "total_reward": m.get("total_reward", 0),
                "avg_waiting_time": wt, "loss": node_losses.get(nid, 0.0), "metrics": m,
            }

            # Save JSONs
            node_file = os.path.join(self.results_dir, f"{nid}_round_{round_idx}_eval.json")
            with open(node_file, "w") as f:
                json.dump(convert_to_json_serializable(round_results["nodes"][nid]), f, indent=2)
            
            # Save models
            model_path = os.path.join(self.results_dir, f"{nid}_round_{round_idx}_model.pt")
            self.agents[nid].save_model(model_path)

        logger.table(table_headers, table_rows)
        
        # Save round summary
        round_file = os.path.join(self.results_dir, f"round_{round_idx}_summary.json")
        with open(round_file, "w") as f:
            json.dump(convert_to_json_serializable(round_results), f, indent=2)

        self.all_round_results.append(round_results)
        logger.success(f"Round {round_idx} data persisted to {self.results_dir}/")

    def train(self, num_rounds: int = 3):
        """Run full AdaptFlow training."""
        print("\n" + "=" * 70)
        print("  ADAPTFLOW-TSC: Adaptive Dynamic Clustering")
        print("  Federated Traffic Signal Control")
        print("=" * 70)
        print(f"  Nodes: {self.num_nodes}, Clusters: {self.num_clusters}")
        print(f"  Rounds: {num_rounds}")
        print(f"  Results: {self.results_dir}/")
        print("=" * 70)

        for r in range(1, num_rounds + 1):
            self.run_round(r)

        # Save cluster history
        history = self.cluster_manager.get_history_summary()
        history_file = os.path.join(self.results_dir, "cluster_history.json")
        with open(history_file, "w") as f:
            json.dump(convert_to_json_serializable(history), f, indent=2)

        # Save all rounds combined
        final_file = os.path.join(self.results_dir, "adaptflow_all_rounds.json")
        with open(final_file, "w") as f:
            json.dump(convert_to_json_serializable(self.all_round_results), f, indent=2)

        # Save final globally-aggregated model weights.
        # After the last round all agents hold identical weights (global broadcast in step 3d),
        # so saving node_0 is equivalent to saving any node.
        mode_label = "sumo" if self.gui else "mock"
        global_model_path = os.path.join(
            self.results_dir, f"adaptflow_global_{mode_label}.pt"
        )
        self.agents["node_0"].save_model(global_model_path)
        print(
            f"  [{mode_label.upper()} Model] Global weights saved to {global_model_path}"
        )

        # 4. EXPORT TO Production (saved_models/)
        try:
            print(f"\n  [Production] Exporting optimized model for deployment...")
            agent = self.agents["node_0"]
            metadata = get_deployment_metadata("adaptflow", agent)
            metadata["mode"] = mode_label
            metadata["source_weights"] = global_model_path
            metadata["sumo_scenario"] = effective_sumo_scenario(self.sumo_scenario)

            save_path = os.path.join(
                "saved_models",
                deployment_model_subdir("adaptflow", self.sumo_scenario),
            )
            ModelExporter.export(agent.policy_net, metadata, save_path)
        except Exception as e:
            print(f"  [Production] Warning: Export failed: {e}")

        print(f"\n{'=' * 70}")
        print(f"  ADAPTFLOW TRAINING COMPLETE")
        print(f"  All results saved to {self.results_dir}/")
        
        # Save training-specific clustering history for UI Partitioning
        try:
            history = self.cluster_manager.get_history_summary()
            history_file = os.path.join(self.results_dir, "training_cluster_history.json")
            with open(history_file, "w") as f:
                json.dump(convert_to_json_serializable(history), f, indent=2)
            print(f"  [UI] Training analytics history saved to {history_file}")
        except Exception as e:
            print(f"  [UI] Warning: Failed to save training analytics: {e}")
            
        print(f"{'=' * 70}")



def convert_to_json_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    else:
        return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AdaptFlow-TSC Training")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--nodes", type=int, default=6, help="Number of nodes")
    parser.add_argument(
        "--clusters", type=int, default=2, help="Number of clusters (0 for auto)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_adaptflow",
        help="Results directory (default + --sumo-scenario china → results_adaptflow_china)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO GUI (real SUMO simulation)",
    )
    parser.add_argument(
        "--use-tomtom",
        action="store_true",
        help="Use real-time TomTom traffic data",
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
        help="SUMO map preset (omit flag to use SUMO_SCENARIO env)",
    )

    args = parser.parse_args()

    results_dir = distinct_results_dir(
        "results_adaptflow", args.results_dir, args.sumo_scenario
    )
    if results_dir != args.results_dir:
        print(
            f"[AdaptFlow] China scenario: writing results to {results_dir}/ (distinct from default OSM runs)"
        )

    # Automatic cluster selection for performance optimization
    num_clusters = args.clusters
    if num_clusters <= 0:
        num_clusters = max(1, (args.nodes + 3) // 4)
        print(f"\n[AUTO] Optimizing performance: Selected {num_clusters} aggregation servers for {args.nodes} nodes.")

    target_pois_list = None
    if args.target_pois:
        target_pois_list = [p.strip() for p in args.target_pois.split(",")]

    trainer = AdaptFlowTrainer(
        num_nodes=args.nodes,
        num_clusters=num_clusters,
        gui=args.gui,
        results_dir=results_dir,
        use_tomtom=args.use_tomtom,
        target_pois=target_pois_list,
        sumo_scenario=args.sumo_scenario,
    )
    trainer.train(num_rounds=args.rounds)
