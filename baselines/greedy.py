import torch
import numpy as np
from statistics import mean, stdev as std
from utils_execution import compute_tour_cost


def greedy_rollout(data, W):
    num_nodes = data.x.shape[0]
    tour = np.zeros((num_nodes,))

    current_node = data.start_route.argmax().item()
    visited_nodes = data.start_route.numpy()

    W_p = torch.softmax(-W, 1)

    for i in range(tour.shape[0]):
        tour[i] = (W_p[current_node] * (1-visited_nodes)).argmax().item()
        current_node = int(tour[i])
        visited_nodes[current_node] = 1.

    assert visited_nodes.all()

    return [tour, np.roll(tour, -1)]


def greedy_baseline(data, return_ratio=True):
    num_nodes = data[0].x.shape[0]
    tours = np.array([
        greedy_rollout(data[i].clone(),
                       data[i].clone().edge_attr.reshape(num_nodes, num_nodes))
        for i in range(data.num_samples)
    ])
    tours = torch.Tensor(tours).long()
    tour_lengths = [compute_tour_cost(y, x.edge_attr).item() for x, y in zip(data, tours)]
    ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]

    return (mean(ratios), std(ratios)) if return_ratio else (mean(tour_lengths), std(tour_lengths))
