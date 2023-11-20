import zython as zn

from torch_geometric.utils import to_dense_adj

from datasets._configs import CONFIGS
from models.tsp_reasoner import LitTSPReasoner
from datasets.tsp_datasets import TSPLarge



from tqdm import tqdm
import itertools

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import pulp


def zinc_CP():
    class TSP(zn.Model):
        def __init__(self, distances):
            self.distances = zn.Array(distances)
            self.path = zn.Array(zn.var(range(len(distances))),
                                 shape=len(distances))
            self.cost = (self._cost(distances))
            self.constraints = [zn.circuit(self.path)]

        def _cost(self, distances):
            return (zn.sum(range(1, len(distances)),
                           lambda i: self.distances[self.path[i - 1],
                                                    self.path[i]]) +
                    self.distances[self.path[len(distances) - 1],
                                   self.path[0]])

    dataset = TSPLarge(
        './data/tsp_large',
        40,
        CONFIGS['tsp_large']['test']['num_samples'],
        split='test',
        algorithm='doesntmatter',
        use_hints=False,
        use_coordinates=False)
    breakpoint()

    distances = [[0, 6, 4, 5, 8],
                 [6, 0, 4, 7, 6],
                 [4, 4, 0, 3, 4],
                 [5, 7, 3, 0, 5],
                 [8, 6, 4, 5, 0]]
    model = TSP([[d*1000 for d in dist] for dist in distances])
    result = model.solve_minimize(model.cost)
    print(result)

def pulp_CP():
    n_customer = 9
    n_point = n_customer + 1

    df = pd.DataFrame({
        'x': np.random.randint(0, 100, n_point),
        'y': np.random.randint(0, 100, n_point),
    })

    df.iloc[0]['x'] = 0
    df.iloc[0]['y'] = 0

    print(df)
    dataset = TSPLarge(
        './data/tsp_large',
        40,
        CONFIGS['tsp_large']['test']['num_samples'],
        split='test',
        algorithm='doesntmatter',
        use_hints=False,
        use_coordinates=False)
    def solve(instance, tl):
        distances = to_dense_adj(instance.edge_index, edge_attr=instance.edge_attr)[0].numpy()
        n_point = instance.num_nodes
        # distances = pd.DataFrame(distance_matrix(df[['x', 'y']].values, df[['x', 'y']].values), index=df.index, columns=df.index).values

        problem = pulp.LpProblem('tsp_mip', pulp.LpMinimize)

        # set valiables
        x = pulp.LpVariable.dicts('x', ((i, j) for i in range(n_point) for j in range(n_point)), lowBound=0, upBound=1, cat='Binary')
        # we need to keep track of the order in the tour to eliminate the possibility of subtours
        u = pulp.LpVariable.dicts('u', (i for i in range(n_point)), lowBound=1, upBound=n_point, cat='Integer')

        # set objective function
        problem += pulp.lpSum(distances[i][j] * x[i, j] for i in range(n_point) for j in range(n_point))

        # set constrains
        for i in range(n_point):
            problem += x[i, i] == 0

        for i in range(n_point):
            problem += pulp.lpSum(x[i, j] for j in range(n_point)) == 1
            problem += pulp.lpSum(x[j, i] for j in range(n_point)) == 1

        # eliminate subtour
        for i in range(n_point):
            for j in range(n_point):
                if i != j and (i != 0 and j != 0):
                    problem += u[i] - u[j] <= n_point * (1 - x[i, j]) - 1
                    
        solver = pulp.GUROBI(time_limit=tl)
        solver.buildSolverModel(problem)
        solver.callSolver(problem)
        status = solver.findSolutionValues(problem)
        return problem.solverModel.ObjVal
    
    # 0.090 0.130 0.179 0.242 0.800 65.557
    tls = {
        # 40: [0.1, 0.3, 0.5],
        # 60: [0.15, 0.45, 0.75],
        # 80: [0.2, 0.6, 1],
        # 100: [0.30, 0.9, 1.5],
        # 100: [3, 5, 10],
        # 200: [1, 2, 3],
        # 200: [30, 60],
        # 1000: [9999999],
        40: [3, 5, 10],
        60: [3, 5, 10],
        80: [3, 5, 10],
        100: [3, 5, 10],
    }
    rel_errs = {
        40: [],
        60: [],
        80: [],
        100: [],
        200: [],
        1000: [],
    }
    for size, times in tls.items():
        suffix = ''
        if size != 40:
            suffix = f"_{size}"
        dataset = TSPLarge(
            './data/tsp_large',
            size,
            CONFIGS['tsp_large']['test'+suffix]['num_samples'],
            split='test'+suffix,
            algorithm='doesntmatter',
            use_hints=False,
            use_coordinates=False)

        for i, tl in enumerate(times):
            rl = []
            for el in tqdm(dataset):
                sol = solve(el, tl)
                rl.append(max(sol/el.optimal_value - 1, 0)) # due to floating point arithmetic GUROBI sometimes reports costs better than (with ~1e-8 absolute difference) than Concorde
            rl = np.array(rl)
            rel_errs[size].append((rl.mean(), rl.std()))
            # print(rel_errs)
    from pprint import pprint
    pprint(rel_errs)

    # breakpoint()
    # solve problem
    # status = problem.solve()

    # output status, value of objective function
    # print(status, pulp.LpStatus[status], pulp.value(problem.objective))

def concorde_CP():
    # 0.090 0.130 0.179 0.242 0.800 65.557
    tls = {
        # 20: [0.1, 0.3, 0.5],
        40: [0.1, 0.3, 0.5],
        60: [0.15, 0.45, 0.75],
        80: [0.2, 0.6, 1],
        100: [0.30, 0.9, 1.5],
        # 100: [3, 5, 10],
        200: [1, 2, 3],
        # 1000: [9999999],
    }
    rel_errs = {
        40: [],
        60: [],
        80: [],
        100: [],
        200: [],
        1000: [],
    }
    for size, times in tls.items():
        suffix = ''
        if size != 40:
            suffix = f"_{size}"
        dataset = TSPLarge(
            './data/tsp_large',
            size,
            CONFIGS['tsp_large']['test'+suffix]['num_samples'],
            split='test'+suffix,
            algorithm='doesntmatter',
            use_hints=False,
            use_coordinates=False)

        for i, tl in enumerate(times):
            dataset_bound = TSPLarge(
                './data/tsp_large',
                size,
                CONFIGS['tsp_large']['test'+suffix]['num_samples'],
                split='test'+suffix,
                algorithm='doesntmatter',
                use_hints=False,
                use_coordinates=False,
                time_bound=tl)
            rl = []
            for el, sol in tqdm(zip(dataset, dataset_bound)):
                # sol = solve(el, tl)
                rl.append(sol.optimal_value/el.optimal_value - 1)
            rl = np.array(rl)
            rel_errs[size].append((rl.mean(), rl.std()))
            print(rel_errs)
    from pprint import pprint
    pprint(rel_errs)

    # breakpoint()
    # solve problem
    # status = problem.solve()

    # output status, value of objective function
    # print(status, pulp.LpStatus[status], pulp.value(problem.objective))

pulp_CP()
# zinc_CP()
# concorde_CP()
