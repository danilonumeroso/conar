import os
import os.path as osp
import math
import numpy as np
import json
import glob

import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset, download, extract_gz, download_url, extract_tar
import networkx as nx
from clrs import Stage, Location, Type
from clrs._src.samplers import FloydWarshallSampler
from clrs._src.algorithms import floyd_warshall
import gurobipy as gp
from gurobipy import GRB

def dominating_set(r, matrix, k):
    n = matrix.shape[0]
    prunedMatrix = (matrix <= r).astype(int)

    m = gp.Model("mip1")

    m.params.BestObjStop = k

    y = [m.addVar(vtype=GRB.BINARY, name="%s" % str(i)) for i in range(n)]

    m.setObjective(sum(y), GRB.MINIMIZE)
    m.addConstrs((sum(np.multiply(prunedMatrix.T[i], y).tolist()) >= 1) for i in range(n))
    m.optimize()
    runtime = m.Runtime

    num_centers = float('inf') if m.Status in [GRB.Status.INFEASIBLE, GRB.Status.UNBOUNDED] else m.objVal
    sd = None
    if num_centers != float('inf'):
        sd = json.loads(m.getJSONSolution())

    return num_centers, sd

def solve(G, k):
    # global total_runtime, k, runtime, num_centers
    dists = dict(nx.all_pairs_dijkstra_path_length(G))
    dists_mat = np.array([[dists[i][j] for i in range(G.number_of_nodes())] for j in range(G.number_of_nodes())])
    total_runtime = 0
    ordered_sizes = sorted(list(set([item for row in dists.values() for item in row.values()])))

    upper = len(ordered_sizes) - 1
    lower = 0
    best_solution_size = float("inf")
    while lower + 1 < upper:
        mid = (lower + upper) // 2
        num_centers, sd = dominating_set(ordered_sizes[mid], dists_mat, k)
        if num_centers <= k:
            upper = mid
            idxs = [int(var['VarName']) for var in sd['Vars']]
        else:
            lower = mid
    # print(matrix)
    # print(dists_mat)
    # print(dists_mat[idxs], idxs)
    assert ordered_sizes[upper] >= np.min(dists_mat[idxs], axis=0).max()
    # print("solution size:", ordered_sizes[upper])
    return idxs, ordered_sizes[upper]

def sample_VKC(num_nodes, rng, k=5):
    sampler = FloydWarshallSampler(floyd_warshall, {}, 1, num_nodes)
    while True:
        matrix = sampler._sample_data(num_nodes, (0.5,))[0]
        G = nx.from_numpy_array(matrix)
        if nx.is_connected(G):
            break
    return matrix, solve(G, k)


def get_vertex_k_center_spec(use_hints=False, use_coordinates=False):
    spec = {
        'edge_attr': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        # 'farthest':  (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    }

    return spec

def create_Data_point(matrix, locations, farthest, maxlen=None):
    num_nodes = matrix.shape[0]
    edges = [(u,v) for u,v in nx.from_numpy_array(matrix).edges]+[(v,u) for u,v in nx.from_numpy_array(matrix).edges]
    edge_index = torch.tensor(edges).T
    edge_attr = torch.tensor(matrix)[edge_index[0], edge_index[1]]
    selected = torch.zeros(num_nodes, dtype=torch.float)
    selected[locations] = 1

    data = Data(
        edge_index=edge_index.clone(),
        edge_attr=edge_attr.float(),
        selected=selected,
        lengths=torch.tensor(num_nodes).expand(1),
        farthest=farthest,
    )
    return data

class VKCLarge(Dataset):
    def __init__(self,
                 root,
                 num_nodes,
                 num_samples,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_hints=False,
                 use_coordinates=False,
                 k=5,
                 seed=0,
                 **unused_kwargs):
        self.split = split
        self.seed = seed
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.k = k
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.spec = get_vertex_k_center_spec(use_hints=use_hints, use_coordinates=use_coordinates)

    @property
    def processed_dir(self):
        return osp.join(self.root, f'num_nodes_{self.num_nodes}', f'k_{self.k}', 'processed', self.split)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.num_samples)]

    def process(self):
        num_samples = self.num_samples
        num_nodes = self.num_nodes
        self._rng = np.random.RandomState(self.seed)

        for i, processed_path in enumerate(self.processed_paths):
            matrix, (locations, farthest) = sample_VKC(self.num_nodes, self._rng, k=self.k)
            data = create_Data_point(matrix, locations, farthest)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, processed_path)

        for f in glob.glob("*.res"):
            os.remove(f)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print ("Wrong number of arguments")
    #     print ("exact input_file_path n k instance_format")
    #     sys.exit()
    # input_file  = sys.argv[1]
    # n = int(sys.argv[2])
    # k = int(sys.argv[3])
    # instance_format = sys.argv[4]
    # matrix = createGraph(input_file, instance_format)
    k = 5
    # nx.seed
    matrix = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.5, seed=5), nonedge=float('inf'))
    np.random.seed(5)
    ws = np.random.rand(*matrix.shape)
    matrix = matrix*(np.triu(ws)+np.triu(ws).T)
    solve(matrix, k)
