"""
Script to test several algorithmic baselines for TSP.

Usage:
   test_baselines_tsp.py (--algorithm A) [options]

Options:
    -h --help              Show this screen.

    --algorithm A          TSP algorithm to run.

    --test_set (test|test_100|test_200|test_1000)   Which test set to use.
"""
import schema
from docopt import docopt
from datasets.constants import _DATASET_ROOTS, _DATASET_CLASSES
from datasets._configs import CONFIGS
from baselines import random_baseline, greedy_baseline, christofides_baseline, optimal_baseline, beam_search_baseline
from statistics import stdev as std, mean

if __name__ == '__main__':
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    schema = schema.Schema({
        '--help': bool,
        '--algorithm': schema.Use(str),
        '--test_set': schema.Use(str)
    })

    args = docopt(__doc__)
    args = schema.validate(args)

    ts_set = args['--test_set']
    data = _DATASET_CLASSES['tsp_large'](
        root=_DATASET_ROOTS['tsp_large'],
        split=ts_set,
        num_nodes=CONFIGS['tsp_large'][ts_set]['num_nodes'],
        num_samples=CONFIGS['tsp_large'][ts_set]['num_samples']
    )

    alg = args['--algorithm']

    if alg == 'random':
        means, stds = random_baseline(data, return_ratio=True)
        print(f'avg. tour len {round(mean(means), 4)} ± {round(std(means), 4)}')
        print(f'avg. std tour len {round(mean(stds), 4)} ± {round(std(stds), 4)}')
    elif alg == 'greedy':
        mean, std_dev = greedy_baseline(data, return_ratio=True)
        print(f'avg. tour len {round(mean, 4)} ± {round(std_dev, 4)}')
    elif alg == 'beam_search':
        import time
        st = time.time()
        mean, std_dev = beam_search_baseline(data, return_ratio=True)
        print(f'avg. tour len {round(mean, 4)} ± {round(std_dev, 4)} which took {time.time()-st}')
    elif alg == 'christofides':
        mean, std_dev = christofides_baseline(data, return_ratio=True)
        print(f'avg. tour len {round(mean, 4)} ± {round(std_dev, 4)}')
    elif alg == 'optimal':
        mean, std_dev = optimal_baseline(data)
        print(f'avg. tour len {round(mean, 4)} ± {round(std_dev, 4)}')
    else:
        assert False, f"{alg} baseline not implemented."
