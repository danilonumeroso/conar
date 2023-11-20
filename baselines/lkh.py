import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
from statistics import mean, stdev as std
from utils_execution import compute_tour_cost
import elkai


def LKH(data):

    data.weight = data.edge_attr
    G = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[0]*1000
    problem = elkai.DistanceMatrix(
        G.int().tolist()
    )
    tour = np.array(problem.solve_tsp()[:-1])

    return [tour, np.roll(tour, -1)]


def LKH_baseline(data, return_ratio=True):
    tours = np.array([
        LKH(data[i])
        for i in range(data.num_samples)
    ])
    tours = torch.Tensor(tours).long()

    tour_lengths = [compute_tour_cost(y, x.edge_attr).item() for x, y in zip(data, tours)]
    ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]

    return (mean(ratios), std(ratios)) if return_ratio else (mean(tour_lengths), std(tour_lengths))

