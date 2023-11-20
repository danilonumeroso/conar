import torch
from pytorch_lightning.utilities.seed import seed_everything
from hyperparameters import get_hyperparameters
from datasets.constants import _DATASET_CLASSES, _DATASET_SPECS
from datasets._configs import CONFIGS
from plotting import plot_histogram_boxplot

def construct_by_num_nodes_and_splits(dataset_names, splits, do_plot=False):

    for dn in dataset_names:
        def _construct(split, nn, st, p):
            ns = CONFIGS[dn][split]['num_samples']
            dataclass = _DATASET_SPECS[dn]['dataclass']
            rd = _DATASET_SPECS[dn]['rootdir']
            offset = nn*3
            if split == 'val':
                offset += 1
            if split == 'test':
                offset += 2
            print(f'constructing algorithm {dn}/{dataclass}/{split}/{rd}')
            f = dataclass(rd,
                          algorithm=dn,
                          split=split,
                          num_nodes=nn,
                          num_samples=ns,
                          seed=(get_hyperparameters()['seed'] + offset),
                          p=p,
                          sampler_type=st)
            return f

        for split in splits:
            nns = CONFIGS[dn][split]['num_nodes']
            if isinstance(nns, int):
                nns = [nns]
            for nn in nns:
                if 'tsp' not in dn:
                    for st in ['default', 'geometric']:
                        for p in [(1.0,), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]:
                            f = _construct(split, nn, st, p)
                else:
                    f = _construct(split, nn, None, None)
            if do_plot:
                plot_histogram_boxplot(f.data.edge_index.cpu().numpy(), name=str(f))

def print_by_num_nodes_and_splits(dataset_names, splits, do_plot=False):

    for dn in dataset_names:
        def _construct(split, nn):
            ns = CONFIGS[dn][split]['num_samples']
            dataclass = _DATASET_SPECS[dn]['dataclass']
            rd = _DATASET_SPECS[dn]['rootdir']
            offset = nn*3
            if split == 'val':
                offset += 1
            if split == 'test':
                offset += 2
            f = dataclass(rd, algorithm=dn, split=split, num_nodes=nn, num_samples=ns, seed=(get_hyperparameters()['seed']+offset))
            print('class', dataclass, 'nn', nn, 'sum', f[0].edge_index.sum())
            return f

        for split in splits:
            nns = CONFIGS[dn][split]['num_nodes']
            if isinstance(nns, int):
                nns = [nns]
            for nn in nns:
                f = _construct(split, nn)
            # f = _construct(split)
            if do_plot:
                plot_histogram_boxplot(f.data.edge_index.cpu().numpy(), name=str(f))

if __name__ == "__main__":
    seed = get_hyperparameters()['seed']
    seed_everything(seed)
    print(f"SEEDED with {seed}")

    construct_by_num_nodes_and_splits(['mst_prim', 'bellman_ford', 'bfs', 'topological_sort', 'activity_selector', 'task_scheduling', 'lcs_length', 'tsp', 'tsp_large', 'graham_scan', 'VKC', 'floyd_warshall', 'minimum', 'insertion_sort'], ['train', 'val', 'test'])
    construct_by_num_nodes_and_splits(['tsp_large'], ['test_'+sfx for sfx in ['20', '60', '80', '100', '200', '1000']])
    construct_by_num_nodes_and_splits(['VKC'], ['test_'+sfx for sfx in ['20', '60', '80', '100', '200']])
    construct_by_num_nodes_and_splits(['tsplib'], ['test_all'])
    print_by_num_nodes_and_splits(['mst_prim', 'tsp', 'tsp_large'], ['train', 'val', 'test'])
