
import torch 
import copy


def federated_average(weights):
    """
    Returns the average of the weights.
    """
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    return weights_avg

