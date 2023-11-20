import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from statistics import mean, stdev as std
from utils_execution import compute_tour_cost


def christofides_algorithm(data):

    data.weight = data.edge_attr
    G = to_networkx(data, edge_attrs=['weight'], to_undirected=True)
    tour = np.array(nx.approximation.christofides(G)[:-1])

    return [tour, np.roll(tour, -1)]


def christofides_baseline(data, return_ratio=True):
    tours = np.array([
        christofides_algorithm(data[i])
        for i in range(data.num_samples)
    ])
    tours = torch.Tensor(tours).long()

    tour_lengths = [compute_tour_cost(y, x.edge_attr).item() for x, y in zip(data, tours)]
    ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]

    return (mean(ratios), std(ratios)) if return_ratio else (mean(tour_lengths), std(tour_lengths))

