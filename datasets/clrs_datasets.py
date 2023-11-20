import os.path as osp
from tqdm import tqdm
import clrs
from clrs._src.algorithms import *#mst_prim, bellman_ford, graham_scan, bfs, topological_sort, activity_selector, task_scheduling, matrix_chain_order, lcs_length, floyd_warshall, bridges
from clrs._src.samplers import Sampler, _batch_hints
from clrs._src.specs import SPECS, Stage, Location, Type

import time
import copy
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
from utils_execution import check_edge_index_sorted, edge_one_hot_encode_pointers
from datasets.geometric_sampler import build_geometric_sampler

NON_GRAPH_ALGORITHMS = ['activity_selector', 'task_scheduling', 'matrix_chain_order', 'lcs_length']

_algorithms = {
    'mst_prim': mst_prim,
    'bellman_ford': bellman_ford,
    'graham_scan': graham_scan,
    'bfs': bfs,
    'topological_sort': topological_sort,
    'activity_selector': activity_selector,
    'task_scheduling': task_scheduling,
    'matrix_chain_order': matrix_chain_order,
    'lcs_length': lcs_length,
    'floyd_warshall': floyd_warshall,
    'bridges': bridges,
    'minimum': minimum,
    'insertion_sort': insertion_sort,
    'dijkstra': dijkstra,
    'mst_kruskal': mst_kruskal,
}
def _iterate_sampler(sampler, batch_size):
  while True:
      yield sampler.next(batch_size)

def _load_inputs(data, feedback, spec):
    spec = copy.deepcopy(spec)
    attrs, adj, num_nodes = None, None, None
    for inp in feedback.features.inputs:
        if inp.name == 'pos':
            num_nodes = inp.data.shape[1]

        if inp.name == 'A':
            attrs = torch.tensor(inp.data[0], dtype=torch.float32)
            continue
        if inp.name == 'adj':
            adj = torch.tensor(inp.data[0])
            continue
        new_name = inp.name
        if spec[inp.name][1:] == (Location.NODE, Type.POINTER):
            new_name = f'{new_name}_index'
            spec[new_name] = spec.pop(inp.name)
        setattr(data, new_name, torch.tensor(inp.data[0], dtype=torch.float32))

    if adj is None:
        adj = torch.ones((num_nodes, num_nodes))
    data.edge_index = adj.nonzero().T
    if attrs is not None:
        attrs = torch.tensor(attrs, dtype=torch.float32)
        data.A = attrs[data.edge_index[0], data.edge_index[1]]
    data.num_nodes = adj.shape[0]
    data.lengths = torch.tensor(feedback.features.lengths[0]).expand(1)

    return data, spec

def _load_hints_and_outputs(data, feedback, spec):
    spec = copy.deepcopy(spec)
    def _prep_probe(unpr_probe, name):
        probe = torch.tensor(unpr_probe, dtype=torch.float32)

        if spec[name][1] == Location.NODE and spec[name][2] not in (Type.POINTER, Type.SHOULD_BE_PERMUTATION) and 'stack_prev' not in name:
            # print(name, spec[name], probe.shape)
            if spec[name][0] == Stage.HINT:
                probe = probe.transpose(0, 1)
        elif spec[name][-2] == Location.EDGE:
            if spec[name][0] == Stage.HINT:
                if spec[name][2] not in (Type.POINTER, Type.SHOULD_BE_PERMUTATION):
                    probe = probe[:, data.edge_index[0], data.edge_index[1]].transpose(0, 1)
                else:
                    probe = probe[:, data.edge_index[0], data.edge_index[1]].long()
            if spec[name][0] == Stage.OUTPUT:
                probe = probe[data.edge_index[0], data.edge_index[1]]
        elif spec[name][-2] == Location.GRAPH:
            probe = probe.unsqueeze(0)
            assert probe.dtype == torch.float32
        elif spec[name][1] == Location.NODE and spec[name][2] in (Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION): # i.e it's a predecessor-like
            probe = probe.long()

        return probe

    for hint in feedback.features.hints:
        probe = _prep_probe(hint.data[:, 0], hint.name)
        new_name = hint.name
        # if hint.name == 'color_af':
        #     breakpoint(pecs.SPECS['floyd_warshall'])
        if spec[hint.name][2] in (Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION):
            new_name = f'{new_name}_index'
        spec[f'{new_name}_temporal'] = spec.pop(hint.name)
        setattr(data, f'{new_name}_temporal', probe)

    for out in feedback.outputs:
        new_name = out.name
        if spec[out.name][2] in (Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION):
            new_name = f'{new_name}_index'
        spec[f'{new_name}'] = spec.pop(out.name)
        probe = _prep_probe(out.data[0], new_name)
        setattr(data, new_name, probe)

    return data, spec

def _pad_hints(hints, length):
    for i, hint in enumerate(hints):
        padder_arr = []
        assert length >= hint.data.shape[0], "please pad with a non-smaller length"
        old_dim = hint.data.shape[0]
        padder_arr.append((0, length-hint.data.shape[0]))
        padder_arr.extend([(0, 0) for _ in range(len(hint.data.shape)-1)])
        hints[i].data = np.pad(hint.data, padder_arr, mode='constant')
        assert hints[i].data[old_dim:].sum() == 0

    return hints

class CLRS(torch_geometric.data.InMemoryDataset):
    @property
    def processed_file_names(self):
        return ['processed.pt']

    @property
    def processed_dir(self):
        if self.sampler_type == 'geometric' or self.algorithm in NON_GRAPH_ALGORITHMS:
            return osp.join(self.root, self.algorithm, f'num_nodes_{self.num_nodes}', f'randomise_pos_{self.randomise_pos}', f'sampler_type_{self.sampler_type}', 'processed', self.split)
        return osp.join(self.root, self.algorithm, f'num_nodes_{self.num_nodes}', f'randomise_pos_{self.randomise_pos}', f'sampler_type_{self.sampler_type}', f'p_{self.p}', 'processed', self.split)

    def __init__(self,
                 root,
                 num_nodes,
                 num_samples,
                 algorithm='mst_prim',
                 split='train',
                 sampler_type='geometric',
                 randomise_pos=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 p=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                 seed=0):
        self.split = split
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.seed = seed
        self.sampler_type = sampler_type
        self.algorithm = algorithm
        self._rng = np.random.RandomState(self.seed)
        self.randomise_pos = randomise_pos
        self.samplers = dict()
        if self.algorithm in ['articulation_points', 'bridges', 'mst_kruskal']:
            p = tuple(prob/2. for prob in p)
        self.p = p
        super().__init__(root=root,
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.data, self.slices, self.spec, self.attribute_list = torch.load(self.processed_paths[0])

    def get_sampler(self, size):
        if self.split == 'train':
            size = self._rng.randint(size // 2, size+1)
        if size not in self.samplers:
            spec = SPECS[self.algorithm]
            algo = _algorithms[self.algorithm]
            if self.sampler_type == 'geometric' and self.algorithm == 'mst_prim':
                self.samplers[size] = build_geometric_sampler(-1, algo, spec, length=size, seed=self.seed)
            else:
                self.samplers[size] = clrs.build_sampler(self.algorithm, -1, length=size, p=self.p, seed=self.seed, min_length=self.num_nodes)

            smplr, spec = self.samplers[size]
            smplr.iterator = _iterate_sampler(smplr, 1)
            if self.randomise_pos:
                smplr.iterator = clrs.process_random_pos(smplr.iterator, self._rng)
            self.samplers[size] = (smplr, spec)



        return self.samplers[size]

    def process(self):
        feedbacks = []
        maxlen = 0
        for i in range(self.num_samples):
            generator, spec = self.get_sampler(self.num_nodes)

            fdb = next(generator.iterator)
            while fdb.features.lengths <= 1:
                fdb = next(generator.iterator)

            feedbacks.append(fdb)
            maxlen = max(maxlen,  int(feedbacks[-1].features.hints[0].data.shape[0]))

        print("Max len is", maxlen)

        data_objs = []
        for i in tqdm(range(self.num_samples)):
            generator, spec = self.get_sampler(self.num_nodes)

            feedback = feedbacks[i]
            feedback.features._replace(hints=_pad_hints(feedback.features.hints, maxlen))
            data = Data()
            data, new_spec = _load_inputs(data, feedback, spec)
            data, new_spec = _load_hints_and_outputs(data, feedback, new_spec)
            assert data.lengths > 1

            check_edge_index_sorted(data.edge_index)

            data_objs.append(data)
        data, slices = self.collate(data_objs)
        attribute_list = list(new_spec.keys())
        torch.save((data, slices, new_spec, attribute_list), self.processed_paths[0])


if __name__ == '__main__':
    scp = CLRS('./data/clrs/', 'train')
    ldr = DataLoader(scp, batch_size=2)
    items = list(iter(ldr))
    breakpoint()
