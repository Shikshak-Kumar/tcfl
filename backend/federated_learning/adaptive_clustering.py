"""
AdaptFlow: Adaptive Dynamic Clustering for Federated Traffic Signal Control.

This module implements congestion fingerprinting and K-means based dynamic
cluster formation. Instead of static spatial clustering, nodes are re-grouped
every round based on real-time congestion similarity.

Novel contribution: No existing FL-TSC paper dynamically re-clusters
intersections based on real-time congestion fingerprints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ── Congestion & Priority Fingerprint ────────────────────────────────

# The 6 dimensions of a combined fingerprint
FINGERPRINT_KEYS = [
    "avg_waiting_time_per_vehicle",
    "average_queue_length",
    "throughput_ratio",
    "max_queue_length",
    "num_congested_lanes",
    "priority_score", # New POI-based dimension
]


def extract_fingerprint(metrics: Dict, priority_tier: int = 3) -> np.ndarray:
    """
    Extract a 6-dimensional fingerprint from node performance and priority.

    Dimensions:
      [avg_wait, avg_queue, throughput, max_queue, congested_lanes, priority_score]
      
    Priority Score: Tier 1 (Hospitals) = 1.0, Tier 2 (Schools) = 0.5, Tier 3 = 0.0
    """
    lane_summary = metrics.get("lane_summary", {})
    
    # Map tier to numeric importance (reversed because Tier 1 is highest priority)
    priority_map = {1: 1.0, 2: 0.5, 3: 0.0}
    priority_score = priority_map.get(priority_tier, 0.0)

    vals = [
        float(metrics.get("avg_waiting_time_per_vehicle", 0.0)),
        float(metrics.get("average_queue_length", 0.0)),
        float(metrics.get("throughput_ratio", 0.0)),
        float(metrics.get("max_queue_length", 0.0)),
        float(lane_summary.get("num_congested_lanes", 0.0)),
        priority_score,
    ]
    return np.array(vals, dtype=np.float64)


def normalize_fingerprints(
    fingerprints: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Min-max normalize fingerprints. Priority dimension is kept as-is if already [0,1].
    """
    if not fingerprints:
        return {}

    node_ids = list(fingerprints.keys())
    matrix = np.stack([fingerprints[nid] for nid in node_ids])  # (N, 6)

    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins

    # Avoid division by zero
    ranges[ranges < 1e-9] = 1.0

    normed_matrix = (matrix - mins) / ranges
    
    # Ensure PRIORITY dimension (index 5) is effectively weighted if needed
    # We can multiply the priority column by a 'Novelty Factor' to ensure it 
    # strongly influences cluster formation.
    # normed_matrix[:, 5] *= 2.0 

    return {nid: normed_matrix[i] for i, nid in enumerate(node_ids)}


# ── K-Means Clustering ──────────────────────────────────────────────


def kmeans_cluster(
    fingerprints: Dict[str, np.ndarray],
    num_clusters: int,
    max_iter: int = 50,
    seed: Optional[int] = None,
    max_capacity: Optional[int] = 4, # Strict limit requested by user
) -> Dict[str, int]:
    """
    K-means clustering on congestion + priority fingerprints with optional capacity balancing.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    node_ids = list(fingerprints.keys())
    data = np.stack([fingerprints[nid] for nid in node_ids])  # (N, D)
    n_samples, n_features = data.shape

    num_clusters = min(num_clusters, n_samples)

    # Initialize centroids via K-means++ style
    centroids = _kmeans_pp_init(data, num_clusters, rng)

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(
            data[:, None, :] - centroids[None, :, :], axis=2
        )  # (N, K)
        new_labels = np.argmin(dists, axis=1)

        # Check convergence
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for k in range(num_clusters):
            members = data[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)

    assignments = {nid: int(labels[i]) for i, nid in enumerate(node_ids)}

    # --- Capacity Balancing (Post-Processing) ---
    if max_capacity is not None:
        assignments = _balance_cluster_capacities(data, node_ids, centroids, assignments, max_capacity)

    return assignments


def _balance_cluster_capacities(
    data: np.ndarray,
    node_ids: List[str],
    centroids: np.ndarray,
    assignments: Dict[str, int],
    max_capacity: int
) -> Dict[str, int]:
    """
    Strictly enforce a maximum number of nodes per cluster.
    Moves 'excess' nodes from overflowing clusters to the nearest cluster with space.
    """
    import copy
    current_assignments = copy.deepcopy(assignments)
    
    # 1. Map cluster to node indices
    cluster_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for i, nid in enumerate(node_ids):
        cluster_to_nodes[current_assignments[nid]].append(i)
        
    num_clusters = centroids.shape[0]
    
    while any(len(nodes) > max_capacity for nodes in cluster_to_nodes.values()):
        # Find the most overflowing cluster
        cid_over = max(cluster_to_nodes.keys(), key=lambda k: len(cluster_to_nodes[k]))
        if len(cluster_to_nodes[cid_over]) <= max_capacity:
            break
            
        # For each node in the overflowing cluster, calculate distance to all other clusters
        nodes_in_over = cluster_to_nodes[cid_over]
        
        # We want to move the node that has the smallest distance to SOME OTHER cluster that has space
        best_move = None # (node_idx, from_cid, to_cid, dist)
        
        for idx in nodes_in_over:
            node_vec = data[idx]
            # Calculate distance to all centroids
            dists = np.linalg.norm(centroids - node_vec, axis=1)
            
            # Look for clusters with space
            for cid_target in range(num_clusters):
                if cid_target == cid_over: continue
                if len(cluster_to_nodes[cid_target]) < max_capacity:
                    d = dists[cid_target]
                    if best_move is None or d < best_move[3]:
                        best_move = (idx, cid_over, cid_target, d)
        
        if best_move:
            idx, f_cid, t_cid, _ = best_move
            # Move the node
            cluster_to_nodes[f_cid].remove(idx)
            cluster_to_nodes[t_cid].append(idx)
            current_assignments[node_ids[idx]] = t_cid
        else:
            # No clusters have space! (Should not happen if num_clusters is calculated correctly)
            break
            
    return current_assignments


def _kmeans_pp_init(data: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """K-means++ initialization for better centroid seeding."""
    n_samples = data.shape[0]
    centroids = [data[rng.randint(n_samples)]]

    for _ in range(1, k):
        dists = np.min(
            [np.linalg.norm(data - c, axis=1) ** 2 for c in centroids], axis=0
        )
        probs = dists / dists.sum()
        idx = rng.choice(n_samples, p=probs)
        centroids.append(data[idx])

    return np.stack(centroids)


def cosine_similarity_matrix(
    fingerprints: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """
    Compute pairwise cosine similarity between all node fingerprints.
    Useful for analysis and visualization.

    Returns:
        (node_ids, similarity_matrix) where similarity_matrix is (N, N)
    """
    node_ids = list(fingerprints.keys())
    matrix = np.stack([fingerprints[nid] for nid in node_ids])  # (N, D)

    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normed = matrix / norms

    sim = normed @ normed.T  # (N, N)
    return node_ids, sim


# ── Cluster Manager ─────────────────────────────────────────────────


class AdaptiveClusterManager:
    """
    Manages dynamic cluster formation across training rounds,
    now incorporating POI priority for "Elite Clustering".
    """

    def __init__(self, num_clusters: int = 2, seed: int = 42):
        self.num_clusters = num_clusters
        self.seed = seed

        # History
        self.cluster_history: List[Dict[str, int]] = []
        self.fingerprint_history: List[Dict[str, np.ndarray]] = []
        self.transition_log: List[Dict] = []

    def recluster(
        self, 
        node_metrics: Dict[str, Dict], 
        round_idx: int,
        priority_tiers: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Perform dynamic re-clustering based on current node metrics AND POI priority.

        Args:
            node_metrics: {node_id: metrics_dict}
            round_idx: Current training round
            priority_tiers: Optional {node_id: tier_int} (1=Hospital, 2=School, 3=Normal)

        Returns:
            {node_id: cluster_index}
        """
        if priority_tiers is None:
            priority_tiers = {nid: 3 for nid in node_metrics}

        # 1. Extract fingerprints (including priority score)
        fingerprints = {
            nid: extract_fingerprint(m, priority_tiers.get(nid, 3)) 
            for nid, m in node_metrics.items()
        }
        self.fingerprint_history.append(fingerprints)

        # 2. Normalize
        normed = normalize_fingerprints(fingerprints)

        # 3. Cluster (Traffic Similarity + Priority Similarity)
        # Higher-priority nodes will naturally pull together if their priority dimensions match
        assignments = kmeans_cluster(
            normed, self.num_clusters, seed=self.seed + round_idx
        )

        # 4. Log transitions
        if self.cluster_history:
            prev = self.cluster_history[-1]
            transitions = {}
            for nid in assignments:
                old_c = prev.get(nid, -1)
                new_c = assignments[nid]
                if old_c != new_c:
                    transitions[nid] = {"from": old_c, "to": new_c}
            self.transition_log.append({"round": round_idx, "transitions": transitions})
        else:
            self.transition_log.append({"round": round_idx, "transitions": {}})

        self.cluster_history.append(assignments)
        return assignments

    def get_cluster_groups(self, assignments: Dict[str, int]) -> Dict[int, List[str]]:
        """Convert {node: cluster} to {cluster: [nodes]}."""
        groups = defaultdict(list)
        for nid, cid in assignments.items():
            groups[cid].append(nid)
        return dict(groups)

    def get_latest_transitions(self) -> Dict:
        """Return the most recent cluster transition log."""
        if self.transition_log:
            return self.transition_log[-1]
        return {"round": -1, "transitions": {}}

    def get_fingerprint_summary(self) -> Dict[str, List[float]]:
        """Return latest fingerprints as JSON-serializable dict."""
        if not self.fingerprint_history:
            return {}
        latest = self.fingerprint_history[-1]
        return {nid: fp.tolist() for nid, fp in latest.items()}

    def get_history_summary(self) -> Dict:
        """Full summary for saving to JSON."""
        history = {
            "num_rounds": len(self.cluster_history),
            "cluster_history": self.cluster_history,
            "transitions": self.transition_log,
            "fingerprints": [
                {nid: fp.tolist() for nid, fp in fps.items()}
                for fps in self.fingerprint_history
            ],
            "similarity_matrices": []
        }
        
        # Add similarity matrices for each round
        for fps in self.fingerprint_history:
            _, sim = cosine_similarity_matrix(fps)
            history["similarity_matrices"].append(sim.tolist())
            
        return history
