import torch
import numpy as np
from statistics import mean, stdev as std
from utils_execution import compute_tour_cost


def random_baseline(data, num_repetitions=100, return_ratio=True):
    means = []
    stds = []
    for _ in range(num_repetitions):
        tours = [np.random.permutation(data.num_nodes) for _ in range(data.num_samples)]
        tours = torch.Tensor(np.array([[np.roll(t, -1), t] for t in tours])).long()

        tour_lengths = [compute_tour_cost(y, x.edge_attr).item() for x, y in zip(data, tours)]
        ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]

        means.append(mean(ratios) if return_ratio else mean(tour_lengths))
        stds.append(std(ratios) if return_ratio else std(tour_lengths))

    return means, stds
