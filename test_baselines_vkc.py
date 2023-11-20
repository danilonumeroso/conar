"""
Script to test several algorithmic baselines for VKC.

Usage:
   test_baselines_vkc.py [options]

Options:
    -h --help              Show this screen.

    --algorithm A          VKC algorithm to run. [default: CDS]

    --test_set (test|test_60|test_100|test_200|test_1000)   Which test set to use.
"""
import schema
from docopt import docopt
import numpy as np
from datasets.constants import _DATASET_ROOTS, _DATASET_CLASSES
from datasets._configs import CONFIGS
from baselines.VKC_baselines import CDSBinarySearch, farthest_first

if __name__ == '__main__':
    schema = schema.Schema({
        '--help': bool,
        '--algorithm': schema.Use(str),
        '--test_set': schema.Use(str)
    })

    args = docopt(__doc__)
    args = schema.validate(args)
    ts_set = args['--test_set']
    data = _DATASET_CLASSES['VKC'](
        root=_DATASET_ROOTS['VKC'],
        split=ts_set,
        num_nodes=CONFIGS['VKC'][ts_set]['num_nodes'],
        num_samples=CONFIGS['VKC'][ts_set]['num_samples']
    )

    alg = args['--algorithm']
    RNG = np.random.RandomState(47)
    if alg == 'CDS':
        mean, std = CDSBinarySearch(data, RNG, rep=1)
        print(f'avg. farthest {round(mean, 4)} ± {round(std, 4)}')
    if alg == 'farthest_first':
        mean, std = farthest_first(data, RNG)
        print(f'avg. farthest {round(mean, 4)} ± {round(std, 4)}')

# # Example adjacency list representing the original graph G
# original_graph = [
#     [(1, 2), (2, 3), (3, 5)],
#     [(0, 2), (2, 4), (3, 1)],
#     [(0, 3), (1, 4), (3, 2)],
#     [(0, 5), (1, 1), (2, 2)]
# ]

# k = 2  # Number of centers to select
# r = 3  # Covering radius
# selected_centers, cost = VertexKCenterBinarySearch(original_graph, k, sorted([val for i in range(len(original_graph)) for nb, val in original_graph[i]]), RNG)
# print("Selected centers:", selected_centers, cost)
