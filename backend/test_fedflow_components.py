import torch
import numpy as np
from federated_learning.gat_module import CoLightEncoder
from agents.fedflow_agent import FedFlowAgent
from federated_learning.fedflow_cluster import FedFlowCluster

def test_gat_shapes():
    print("Testing GAT Module Shapes...")
    batch_size = 4
    num_nodes = 3
    feat_dim = 12
    encoder = CoLightEncoder(nfeat=feat_dim, nhid=32, nheads=4)
    
    x = torch.randn(batch_size, num_nodes, feat_dim)
    adj = torch.ones(batch_size, num_nodes, num_nodes)
    output = encoder(x, adj)
    
    assert output.shape == (batch_size, num_nodes, 32), f"Expected (4,3,32), got {output.shape}"
    print("  GAT Shapes: OK")

def test_agent_replay():
    print("Testing Agent Replay Logic...")
    agent = FedFlowAgent(state_size=12, action_size=4)
    
    # Fill memory
    for _ in range(70):
        s = np.random.rand(3, 12)
        a_mat = np.eye(3)
        agent.remember(s, a_mat, 1, 1.0, s, a_mat, False)
        
    loss = agent.replay()
    assert loss > 0, "Loss should be positive after replay"
    print(f"  Agent Replay: OK (Initial Loss: {loss:.4f})")

def test_cluster_aggregation():
    print("Testing Cluster Aggregation Logic...")
    agent_ids = ["node_0", "node_1"]
    cluster = FedFlowCluster("cluster_test", agent_ids)
    
    # Mock parameters
    p1 = {"fc1.weight": torch.ones(2, 2)}
    p2 = {"fc1.weight": torch.ones(2, 2) * 2}
    
    # Scenario: node_0 has 3x flow of node_1
    cluster.update_flow("node_0", 3.0)
    cluster.update_flow("node_1", 1.0)
    
    agg = cluster.aggregate_intra_cluster([p1, p2])
    # Expected: (3*1 + 1*2) / 4 = 1.25
    expected = (3.0 * 1.0 + 1.0 * 2.0) / 4.0
    val = agg["fc1.weight"][0,0].item()
    
    assert abs(val - expected) < 1e-5, f"Expected {expected}, got {val}"
    print(f"  Cluster Aggregation: OK (Weighted Mean: {val:.4f})")

if __name__ == "__main__":
    try:
        test_gat_shapes()
        test_agent_replay()
        test_cluster_aggregation()
        print("\nALL FEDFLOW COMPONENTS VERIFIED SUCCESSFULLY.")
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
