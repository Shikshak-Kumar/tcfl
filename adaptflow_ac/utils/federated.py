import torch
import copy

def federated_aggregate(local_params_list, weights=None):
    """
    local_params_list: list of state_dict
    weights: list of floats (importance for each node)
    """
    if weights is None:
        # Uniform averaging (FedAvg)
        weights = [1.0 / len(local_params_list)] * len(local_params_list)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
    global_dict = copy.deepcopy(local_params_list[0])
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, local_dict in enumerate(local_params_list):
            global_dict[key] += local_dict[key] * weights[i]
            
    return global_dict

def get_congestion_weights(node_metrics):
    """
    node_metrics: list of dicts with 'avg_queue'
    """
    queues = [m['avg_queue'] for m in node_metrics]
    total_queue = sum(queues)
    if total_queue == 0:
        return [1.0 / len(queues)] * len(queues)
    return [q / total_queue for q in queues]
