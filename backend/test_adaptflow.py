"""
Unit tests for AdaptFlow adaptive clustering module.
"""

import numpy as np
from federated_learning.adaptive_clustering import (
    extract_fingerprint,
    normalize_fingerprints,
    kmeans_cluster,
    cosine_similarity_matrix,
    AdaptiveClusterManager,
)


def test_fingerprint_extraction():
    """Verify correct 5D fingerprint from mock metrics."""
    print("Testing fingerprint extraction...")

    metrics = {
        "avg_waiting_time_per_vehicle": 3.5,
        "average_queue_length": 2.1,
        "throughput_ratio": 0.45,
        "max_queue_length": 5,
        "lane_summary": {
            "num_congested_lanes": 2,
        },
    }

    fp = extract_fingerprint(metrics)
    assert fp.shape == (5,), f"Expected shape (5,), got {fp.shape}"
    assert fp[0] == 3.5, f"Expected avg_wait=3.5, got {fp[0]}"
    assert fp[1] == 2.1, f"Expected avg_queue=2.1, got {fp[1]}"
    assert fp[2] == 0.45, f"Expected tp_ratio=0.45, got {fp[2]}"
    assert fp[3] == 5.0, f"Expected max_queue=5.0, got {fp[3]}"
    assert fp[4] == 2.0, f"Expected congested=2.0, got {fp[4]}"

    print("  Fingerprint extraction: OK")


def test_fingerprint_missing_keys():
    """Fingerprints should handle missing keys gracefully (defaults to 0)."""
    print("Testing fingerprint with missing keys...")

    fp = extract_fingerprint({})
    assert fp.shape == (5,), f"Expected shape (5,), got {fp.shape}"
    assert np.all(fp == 0.0), f"Expected all zeros, got {fp}"

    print("  Missing keys handling: OK")


def test_normalization():
    """Verify min-max normalization across nodes."""
    print("Testing normalization...")

    fingerprints = {
        "a": np.array([0.0, 10.0, 0.5, 2.0, 0.0]),
        "b": np.array([5.0, 20.0, 1.0, 4.0, 3.0]),
        "c": np.array([10.0, 30.0, 0.0, 6.0, 6.0]),
    }

    normed = normalize_fingerprints(fingerprints)

    # 'a' should be 0 on dims where it's min, 1 where max
    assert abs(normed["a"][0] - 0.0) < 1e-6, "a dim0 should be 0"
    assert abs(normed["c"][0] - 1.0) < 1e-6, "c dim0 should be 1"
    assert abs(normed["b"][0] - 0.5) < 1e-6, "b dim0 should be 0.5"

    print("  Normalization: OK")


def test_kmeans_separates_dissimilar():
    """Two groups of nodes with very different congestion should split."""
    print("Testing K-means separates dissimilar nodes...")

    # Group 1: High congestion
    # Group 2: Low congestion
    fingerprints = {
        "node_0": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # high
        "node_1": np.array([0.9, 0.95, 0.92, 0.88, 0.9]),  # high
        "node_2": np.array([0.85, 0.9, 0.95, 0.9, 0.85]),  # high
        "node_3": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # low
        "node_4": np.array([0.1, 0.05, 0.08, 0.12, 0.1]),  # low
        "node_5": np.array([0.15, 0.1, 0.05, 0.1, 0.15]),  # low
    }

    assignments = kmeans_cluster(fingerprints, num_clusters=2, seed=42)

    # High nodes should be in same cluster, low nodes in same cluster
    high_cluster = assignments["node_0"]
    low_cluster = assignments["node_3"]
    assert high_cluster != low_cluster, (
        "High and low groups should be in different clusters"
    )
    assert assignments["node_1"] == high_cluster, "node_1 should be with node_0"
    assert assignments["node_2"] == high_cluster, "node_2 should be with node_0"
    assert assignments["node_4"] == low_cluster, "node_4 should be with node_3"
    assert assignments["node_5"] == low_cluster, "node_5 should be with node_3"

    print("  K-means separation: OK")


def test_kmeans_three_clusters():
    """Test with 3 distinct groups."""
    print("Testing K-means with 3 clusters...")

    fingerprints = {
        "a": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
        "c": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "d": np.array([0.55, 0.55, 0.55, 0.55, 0.55]),
        "e": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "f": np.array([0.95, 0.95, 0.95, 0.95, 0.95]),
    }

    assignments = kmeans_cluster(fingerprints, num_clusters=3, seed=42)

    # a,b in one; c,d in another; e,f in third
    assert assignments["a"] == assignments["b"], "a and b should cluster together"
    assert assignments["c"] == assignments["d"], "c and d should cluster together"
    assert assignments["e"] == assignments["f"], "e and f should cluster together"
    assert assignments["a"] != assignments["c"], "low != mid"
    assert assignments["c"] != assignments["e"], "mid != high"

    print("  3-cluster separation: OK")


def test_cosine_similarity():
    """Verify cosine similarity matrix computation."""
    print("Testing cosine similarity...")

    fingerprints = {
        "a": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "b": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # identical to a
        "c": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),  # orthogonal to a
    }

    node_ids, sim = cosine_similarity_matrix(fingerprints)

    idx_a = node_ids.index("a")
    idx_b = node_ids.index("b")
    idx_c = node_ids.index("c")

    assert abs(sim[idx_a, idx_b] - 1.0) < 1e-6, "a and b should be identical (sim=1)"
    assert abs(sim[idx_a, idx_c] - 0.0) < 1e-6, "a and c should be orthogonal (sim=0)"

    print("  Cosine similarity: OK")


def test_cluster_manager_history():
    """Verify cluster manager tracks history across rounds."""
    print("Testing cluster manager history tracking...")

    mgr = AdaptiveClusterManager(num_clusters=2, seed=42)

    # Round 1 metrics: all similar
    metrics_r1 = {
        "node_0": {
            "avg_waiting_time_per_vehicle": 1.0,
            "average_queue_length": 1.0,
            "throughput_ratio": 0.5,
            "max_queue_length": 2,
            "lane_summary": {"num_congested_lanes": 0},
        },
        "node_1": {
            "avg_waiting_time_per_vehicle": 1.1,
            "average_queue_length": 1.1,
            "throughput_ratio": 0.5,
            "max_queue_length": 2,
            "lane_summary": {"num_congested_lanes": 0},
        },
        "node_2": {
            "avg_waiting_time_per_vehicle": 1.2,
            "average_queue_length": 0.9,
            "throughput_ratio": 0.5,
            "max_queue_length": 2,
            "lane_summary": {"num_congested_lanes": 0},
        },
        "node_3": {
            "avg_waiting_time_per_vehicle": 5.0,
            "average_queue_length": 4.0,
            "throughput_ratio": 0.1,
            "max_queue_length": 8,
            "lane_summary": {"num_congested_lanes": 3},
        },
    }

    assignments_r1 = mgr.recluster(metrics_r1, round_idx=1)
    assert len(assignments_r1) == 4
    assert len(mgr.cluster_history) == 1
    assert len(mgr.fingerprint_history) == 1

    # Round 2 metrics: node_2 becomes congested, should move
    metrics_r2 = {
        "node_0": metrics_r1["node_0"],
        "node_1": metrics_r1["node_1"],
        "node_2": {
            "avg_waiting_time_per_vehicle": 6.0,
            "average_queue_length": 5.0,
            "throughput_ratio": 0.08,
            "max_queue_length": 10,
            "lane_summary": {"num_congested_lanes": 4},
        },
        "node_3": metrics_r1["node_3"],
    }

    assignments_r2 = mgr.recluster(metrics_r2, round_idx=2)
    assert len(mgr.cluster_history) == 2
    assert len(mgr.transition_log) == 2

    # node_2 should now be with node_3 (both congested)
    assert assignments_r2["node_2"] == assignments_r2["node_3"], (
        "node_2 (now congested) should cluster with node_3"
    )
    assert assignments_r2["node_0"] == assignments_r2["node_1"], (
        "node_0 and node_1 (both low) should cluster together"
    )

    print(f"  History tracking: OK (2 rounds recorded)")
    print(f"  Transition log: {mgr.transition_log}")


if __name__ == "__main__":
    try:
        test_fingerprint_extraction()
        test_fingerprint_missing_keys()
        test_normalization()
        test_kmeans_separates_dissimilar()
        test_kmeans_three_clusters()
        test_cosine_similarity()
        test_cluster_manager_history()
        print("\n✅ ALL ADAPTFLOW TESTS PASSED.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
