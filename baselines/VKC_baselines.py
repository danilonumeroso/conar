import numpy as np
from torch_geometric.utils import to_networkx
from networkx import all_pairs_dijkstra_path_length
from statistics import mean, stdev as std

def BottleneckGraph(graph, r):
    bottleneck_graph = []
    
    for u, neighbors in enumerate(graph):
        bottleneck_neighbors = [(v, cost) for v, cost in neighbors if cost <= r]
        bottleneck_graph.append(bottleneck_neighbors)
    
    return bottleneck_graph

def GetDegree(vertex, pruned_graph):
    degree = 0
    
    for neighbors in pruned_graph[vertex]:
        degree += 1
    
    return degree

def distance(graph, node, selected_centers):
    mindist = 1e9
    for nb, dist in graph[node]:
        if nb in selected_centers:
            mindist = min(mindist, dist)
    return mindist

def CDS_Heuristic(graph, k, r, RNG):
    n = len(graph)
    pruned_graph = BottleneckGraph(graph, r)
    selected_centers = [RNG.randint(0, len(graph))]
    D = set(selected_centers)
    score = [GetDegree(v, pruned_graph) for v in range(n)]
    
    for _ in range(k-1):
        f = np.argmax([distance(graph, v, selected_centers) for v in range(len(pruned_graph))])
        vf = max([nb for nb, _ in pruned_graph[f] if nb not in selected_centers] + [f], key=lambda v: score[v])
        S = set([Nf for Nf, _ in pruned_graph[vf]] + [vf]) - D

        for v in S:
            for u, _ in pruned_graph[v]:
                score[u] -= 1
        
        D |= S
        selected_centers.append(vf)

    cost = np.max([distance(graph, v, selected_centers) for v in range(len(pruned_graph))])
    
    return selected_centers, cost

def CDS_Heuristic_rep(graph, k, r, RNG, rep):
    selected_centers = []
    cost = 1e9
    for _ in range(rep):
        slc, cst = CDS_Heuristic(graph, k, r, RNG)
        if cst < cost:
            selected_centers = slc
            cost = cst
    return selected_centers, cost

def VKC_BS(graph, k, ordered_weights, RNG, rep=1):
    high = len(ordered_weights)
    low = 0
    best_sol = None
    best_cost = 1e9
    
    while high - low > 1:
        mid = (high + low) // 2
        weight_mid = ordered_weights[mid]
        
        centers, cost = CDS_Heuristic_rep(graph, k, weight_mid, RNG, rep)
        if cost < best_cost:
            best_sol = centers
            best_cost = cost
        
        if cost <= weight_mid:
            high = mid
        else:
            low = mid
    
    return best_sol, best_cost

def convert(instance):
    G = to_networkx(instance, edge_attrs=['edge_attr'])
    apsp = {k: v for k, v in all_pairs_dijkstra_path_length(G, weight='edge_attr')}
    graph = [[] for _ in range(instance.num_nodes)]
    wlist = []
    for u in range(instance.num_nodes):
        for v in range(instance.num_nodes):
            graph[u].append((v, apsp[u][v]))
            wlist.append(apsp[u][v])

    return graph, wlist

def FF(graph, k, RNG):
    selected_centers = [RNG.randint(0, len(graph))]
    for _ in range(k-1):
        f = np.argmax([distance(graph, v, selected_centers) for v in range(len(graph))])
        selected_centers.append(f)

    cost = np.max([distance(graph, v, selected_centers) for v in range(len(graph))])
    
    return selected_centers, cost

def farthest_first(data, RNG):
    ratios = []
    for instance in data:
        ci, wlist = convert(instance)
        best_sol, best_cost = FF(ci, data.k, RNG)
        ratios.append((best_cost + 1e-6) / instance.farthest - 1)
        assert best_cost <= 2*instance.farthest
        assert best_cost + 1e-6 >= instance.farthest, breakpoint()
    # ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]
    return (mean(ratios), std(ratios))

def CDSBinarySearch(data, RNG, rep=1):
    ratios = []
    for instance in data:
        ci, wlist = convert(instance)
        best_sol, best_cost = VKC_BS(ci, data.k, sorted(wlist), RNG, rep=rep)
        ratios.append((best_cost + 1e-6) / instance.farthest - 1)
        assert best_cost <= 3*instance.farthest
        assert best_cost + 1e-6 >= instance.farthest, breakpoint()
    # ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]
    return (mean(ratios), std(ratios))
