import os
import os.path as osp
import glob
import numpy as np
import torch
import re

from concorde.tsp import TSPSolver
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset, download, extract_gz, download_url, extract_tar
from torch_geometric.utils import dense_to_sparse
from utils_execution import edge_one_hot_encode_pointers, compute_tour_cost
from clrs import Stage, Location, Type

LOW_, HIGH_ = 0, 1000
def sample_tsp(num_samples, num_nodes, num_coords=2, coord_distrib="integers", seed=0):

    rng = np.random.default_rng(seed)
    sampler = getattr(rng, coord_distrib)

    x = sampler(low=LOW_, high=HIGH_, size=[num_samples, num_nodes, num_coords])

    sols = [
        TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D").solve(verbose=False)
        for coords in x
    ]
    y = np.array([sol.tour for sol in sols])
    costs = np.array([sol.optimal_value for sol in sols])
    tour_starts = rng.choice(num_nodes, size=(num_samples, ))

    return x / HIGH_, y, costs, tour_starts


def get_predecessors(tour):
    assert tour[0] == 0
    pred = torch.zeros_like(tour)
    for i in range(0, len(tour)):
        pred[tour[i]] = tour[i-1]
    # pred[tour[0]] = pred[tour[-1]]
    return pred


def read_tsp_file_contents(filename: str):
    with open(filename) as f:
        content = [line.strip() for line in f.read().splitlines()]
    return content


def get_num_nodes(content) -> int:
    for line in content:
        if line.startswith("DIMENSION"):
            parts = line.split(":")
            return int(parts[1])
    assert False


def get_coords(content):
    i = content.index("NODE_COORD_SECTION") + 1
    num_nodes = get_num_nodes(content)
    coords = []
    for index in range(i, i + num_nodes):
        parts = content[index].strip()
        city_coords_parts = re.findall(r"[+-]?\d+(?:\.\d+)?", parts)
        coords.append([float(city_coords_parts[1]), float(city_coords_parts[2])])
    return np.array(coords)


def is_euc_data_type(content):
    for line in content:
        if 'EUC_2D' in line or 'GEO' in line:
            return True
    return False


def test_get_predecessors():
    out = get_predecessors(torch.tensor([0, 3, 1, 2]))
    assert (out == torch.tensor([2, 3, 1, 0])).all(), out


test_get_predecessors()

def create_Data_point(coords, tour, start_tour, num_nodes, maxlen=None):
    sr = torch.zeros((num_nodes, ))
    sr[start_tour] = 1
    edge_index = dense_to_sparse(torch.ones(size=(num_nodes, num_nodes)))[0]
    pred = get_predecessors(torch.tensor(tour).long())
    idx_start_tour = np.where(tour == start_tour)[0][0]
    step_by_step_tour = np.roll(tour, -idx_start_tour)
    assert step_by_step_tour[0] == start_tour
    one_hot_tour = torch.tensor(np.eye(num_nodes)[step_by_step_tour]).T # Timestep x Node
    mask_in_tour = one_hot_tour.cumsum(0).T # Node x Timestep
    if maxlen is not None:
        one_hot_tour = torch.nn.functional.pad(one_hot_tour, (0, maxlen-one_hot_tour.shape[-1]))
        mask_in_tour = torch.nn.functional.pad(mask_in_tour, (0, maxlen-mask_in_tour.shape[-1]))
    data = Data(x=torch.tensor(coords),
                xc=torch.tensor(coords[:, 0]).float(),
                yc=torch.tensor(coords[:, 1]).float(),
                start_route=sr.clone(),
                predecessor_index=pred,
                predecessor_edge_1h=edge_one_hot_encode_pointers(
                    pred, edge_index),
                one_hot_tour_temporal=one_hot_tour,
                mask_in_tour_temporal=mask_in_tour,
                lengths=torch.tensor(num_nodes).expand(1),
                edge_index=edge_index.clone())
    src, dst = data.edge_index

    # adding: edge fts (euclidean distance)
    x_s, x_d = data.x[src], data.x[dst]
    data.edge_attr = (x_s - x_d).pow(2).sum(-1).sqrt().float()
    data.edge_attr = data.edge_attr.masked_fill(src == dst, data.edge_attr.max() + 1)

    data.optimal_value = compute_tour_cost(
        tour=torch.stack([torch.arange(data.predecessor_index.shape[0]),
                          data.predecessor_index]),
        weights=data.edge_attr)
    return data

def get_TSP_spec(use_hints=False, use_coordinates=False):
    spec = {
        'xc': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'yc': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'edge_attr': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'start_route': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'mask_in_tour_temporal': (Stage.HINT, Location.NODE, Type.MASK),
        'one_hot_tour_temporal': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'predecessor_index': (Stage.OUTPUT, Location.NODE, Type.POINTER)
    }
    if not use_hints:
        spec = {k: v for k, v in spec.items() if v[0] != Stage.HINT}

    if not use_coordinates:
        del spec['xc'], spec['yc']
    return spec

class TSPLarge(Dataset):
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
                 seed=0,
                 **unused_kwargs):
        self.split = split
        self.seed = seed
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.spec = get_TSP_spec(use_hints=use_hints, use_coordinates=use_coordinates)

    @property
    def processed_dir(self):
        return osp.join(self.root, f'num_nodes_{self.num_nodes}', 'processed', self.split)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.num_samples)]

    def process(self):
        num_samples = self.num_samples
        num_nodes = self.num_nodes
        self._rng = np.random.RandomState(self.seed)

        for i, processed_path in enumerate(self.processed_paths):
            x, y, costs, starts = sample_tsp(1, num_nodes, seed=self._rng.randint(2**31-1))
            x_i, y_i, c_i = x[0], y[0], costs[0]
            s_i = self._rng.choice(num_nodes)
            data = create_Data_point(x_i, y_i, s_i, num_nodes)
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

class TSPLIB(InMemoryDataset):
    @property
    def processed_file_names(self):
        return ['processed.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.split, 'processed')

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    def raw_file_names(self):
        return ['a280.tsp', 'ali535.tsp', 'att48.tsp', 'att532.tsp', 'bayg29.tsp',
                'bays30.tsp', 'berlin52.tsp', 'bier127.tsp', 'brazil58.tsp',
                'brd14051.tsp', 'brg180.tsp', 'burma14.tsp', 'ch130.tsp',
                'ch150.tsp', 'd1291.tsp', 'd15112.tsp', 'd1655.tsp', 'd18512.tsp',
                'd198.tsp', 'd2103.tsp', 'd493.tsp', 'd657.tsp', 'dantzig42.tsp',
                'dsj1000.tsp', 'eil101.tsp', 'eil51.tsp', 'eil76.tsp', 'fl1400.tsp',
                'fl1577.tsp', 'fl3795.tsp', 'fl417.tsp', 'fnl4461.tsp', 'fri26.tsp',
                'gil262.tsp', 'gr120.tsp', 'gr137.tsp', 'gr17.tsp', 'gr202.tsp',
                'gr21.tsp', 'gr229.tsp', 'gr24.tsp', 'gr431.tsp', 'gr48.tsp',
                'gr666.tsp', 'gr96.tsp', 'hk48.tsp', 'kroA100.tsp', 'kroA150.tsp',
                'kroA200.tsp', 'kroB100.tsp', 'kroB150.tsp', 'kroB200.tsp', 'kroC100.tsp',
                'kroD100.tsp', 'kroE100.tsp', 'lin105.tsp', 'lin318.tsp', 'linhp318.tsp',
                'nrw1379.tsp', 'p654.tsp', 'pa561.tsp', 'pcb1173.tsp', 'pcb3038.tsp',
                'pcb442.tsp', 'pla33810.tsp', 'pla7397.tsp', 'pla85900.tsp', 'pr1002.tsp'
                , 'pr107.tsp', 'pr124.tsp', 'pr136.tsp', 'pr144.tsp', 'pr152.tsp', 'pr226.tsp',
                'pr2392.tsp', 'pr264.tsp', 'pr299.tsp', 'pr439.tsp', 'pr76.tsp', 'rat195.tsp',
                'rat575.tsp', 'rat783.tsp', 'rat99.tsp', 'rd100.tsp', 'rd400.tsp', 'rl11849.tsp',
                'rl1304.tsp', 'rl1323.tsp', 'rl1889.tsp', 'rl5915.tsp', 'rl5934.tsp', 'si1032.tsp',
                'si175.tsp', 'si535.tsp', 'st70.tsp', 'swiss42.tsp', 'ts225.tsp', 'tsp225.tsp',
                'u1060.tsp', 'u1432.tsp', 'u159.tsp', 'u1817.tsp', 'u2152.tsp', 'u2319.tsp', 'u574.tsp',
                'u724.tsp', 'ulysses16.tsp', 'ulysses22.tsp', 'usa13509.tsp', 'vm1084.tsp', 'vm1748.tsp']

    def download(self):
        folder = self.raw_dir

        path = download_url('http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz',
                            folder, log=False)
        extract_tar(path, folder, log=False)
        for filename in os.listdir(folder):
            if filename.endswith('tsp.gz'):
                extract_gz(osp.join(folder, filename), folder, log=False)

        for filename in os.listdir(folder):
            if filename.endswith('.gz'):
                os.remove(osp.join(folder, filename))

    def __init__(self,
                 root,
                 num_nodes=300,
                 split='test_all',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 seed=0,
                 **unused_kwargs):

        self.max_num_nodes = num_nodes
        self.split = split
        self.seed = seed
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.spec = get_TSP_spec(use_hints=False)


    def process(self):
        _rng = np.random.RandomState(self.seed)
        data_dict = {}
        for filename in os.listdir(self.raw_dir):
            content = read_tsp_file_contents(osp.join(self.raw_dir, filename))
            if is_euc_data_type(content):
                data_dict[content[0].split(':')[1].strip()] = content

        x = [get_coords(data_dict[key]) for key in data_dict]
        x = list(filter(lambda coords: coords.shape[0] < self.max_num_nodes, x))

        sols = [
            TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D").solve(verbose=False)
            for coords in x
        ]
        y = [np.array(sol.tour) for sol in sols]

        # adding: nodes, labels, edges
        min_coords = [a.min() for a in x]
        max_coords = [a.max() for a in x]
        data_list = []
        for x_i, y_i, min_x, max_x, name in zip(x, y, min_coords, max_coords, data_dict.keys()):
            num_nodes = x_i.shape[0]

            s_i = _rng.choice(num_nodes)
            data = create_Data_point((x_i - min_x)/(max_x - min_x),
                                     y_i, s_i, num_nodes,
                                     maxlen=self.max_num_nodes)
            data_list.append(data)

        data, slices = self.collate(data_list)
        # name: (stage, loc, datatype)
        torch.save((data, slices), self.processed_paths[0])
        for f in glob.glob("*.res"):
            os.remove(f)



class TSP(InMemoryDataset):
    @property
    def processed_file_names(self):
        return ['processed.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, f'num_nodes_{self.num_nodes}', 'processed', self.split)

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
                 seed=0,
                 **unused_kwargs):
        self.split = split
        self.seed = seed
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.spec = get_TSP_spec(use_hints=use_hints, use_coordinates=use_coordinates)

    def process(self):
        num_samples = self.num_samples
        num_nodes = self.num_nodes

        x, y, costs, starts = sample_tsp(num_samples, num_nodes, seed=self.seed)

        # adding: nodes, labels, edges
        edge_index = dense_to_sparse(torch.ones(size=(num_nodes, num_nodes)))[0]
        data_list = []
        for x_i, y_i, c_i, s_i in zip(x, y, costs, starts):
            data = create_Data_point(x_i, y_i, s_i, num_nodes)
            data_list.append(data)

        data, slices = self.collate(data_list)
        # name: (stage, loc, datatype)
        torch.save((data, slices), self.processed_paths[0])
        for f in glob.glob("*.res"):
            os.remove(f)


if __name__ == '__main__':
    scp = TSP('./data/tmp', 10, 10, 'train')
    ldr = DataLoader(scp, batch_size=2)
    items = list(iter(ldr))
    breakpoint()
