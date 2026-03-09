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


# ── Congestion Fingerprint ───────────────────────────────────────────

# The 5 dimensions of a congestion fingerprint
FINGERPRINT_KEYS = [
    "avg_waiting_time_per_vehicle",
    "average_queue_length",
    "throughput_ratio",
    "max_queue_length",
    "num_congested_lanes",
]


def extract_fingerprint(metrics: Dict) -> np.ndarray:
    """
    Extract a 5-dimensional congestion fingerprint from node performance metrics.

    The fingerprint captures the traffic state at a node:
      [avg_wait/veh, avg_queue_len, throughput_ratio, max_queue, congested_lanes]

    Args:
        metrics: Dict from env.get_performance_metrics()

    Returns:
        np.ndarray of shape (5,)
    """
    lane_summary = metrics.get("lane_summary", {})

    vals = [
        float(metrics.get("avg_waiting_time_per_vehicle", 0.0)),
        float(metrics.get("average_queue_length", 0.0)),
        float(metrics.get("throughput_ratio", 0.0)),
        float(metrics.get("max_queue_length", 0.0)),
        float(lane_summary.get("num_congested_lanes", 0.0)),
    ]
    return np.array(vals, dtype=np.float64)


def normalize_fingerprints(
    fingerprints: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Min-max normalize fingerprints across all nodes to [0, 1] per dimension.
    Prevents scale dominance (e.g., waiting time in seconds vs queue in count).
    """
    if not fingerprints:
        return {}

    node_ids = list(fingerprints.keys())
    matrix = np.stack([fingerprints[nid] for nid in node_ids])  # (N, 5)

    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins

    # Avoid division by zero for constant features
    ranges[ranges < 1e-9] = 1.0

    normed_matrix = (matrix - mins) / ranges

    return {nid: normed_matrix[i] for i, nid in enumerate(node_ids)}


# ── K-Means Clustering ──────────────────────────────────────────────


def kmeans_cluster(
    fingerprints: Dict[str, np.ndarray],
    num_clusters: int,
    max_iter: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    K-means clustering on congestion fingerprints.

    Args:
        fingerprints: Normalized fingerprints {node_id: np.ndarray(5,)}
        num_clusters: Number of clusters (K)
        max_iter: Maximum iterations
        seed: Random seed for reproducibility

    Returns:
        Dict mapping node_id -> cluster_index
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

    return {nid: int(labels[i]) for i, nid in enumerate(node_ids)}


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
    Manages dynamic cluster formation across training rounds.

    Each round:
      1. Receives per-node metrics
      2. Extracts congestion fingerprints
      3. Normalizes and clusters via K-means
      4. Returns new cluster assignments

    Tracks full history for analysis.
    """

    def __init__(self, num_clusters: int = 2, seed: int = 42):
        self.num_clusters = num_clusters
        self.seed = seed

        # History
        self.cluster_history: List[Dict[str, int]] = []
        self.fingerprint_history: List[Dict[str, np.ndarray]] = []
        self.transition_log: List[Dict] = []

    def recluster(
        self, node_metrics: Dict[str, Dict], round_idx: int
    ) -> Dict[str, int]:
        """
        Perform dynamic re-clustering based on current node metrics.

        Args:
            node_metrics: {node_id: metrics_dict} from env.get_performance_metrics()
            round_idx: Current training round

        Returns:
            {node_id: cluster_index} — new cluster assignments
        """
        # 1. Extract fingerprints
        fingerprints = {nid: extract_fingerprint(m) for nid, m in node_metrics.items()}
        self.fingerprint_history.append(fingerprints)

        # 2. Normalize
        normed = normalize_fingerprints(fingerprints)

        # 3. Cluster
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
        return {
            "num_rounds": len(self.cluster_history),
            "cluster_history": self.cluster_history,
            "transitions": self.transition_log,
            "fingerprints": [
                {nid: fp.tolist() for nid, fp in fps.items()}
                for fps in self.fingerprint_history
            ],
        }
