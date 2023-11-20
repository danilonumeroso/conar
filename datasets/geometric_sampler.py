import torch
from clrs._src.samplers import Sampler
from clrs._src.specs import SPECS
class GeometricGraphSampler(Sampler):
    """Bellman-Ford sampler."""

    def _sample_data(
        self,
        length,
        dims=2,
    ):
        num_nodes = length
        edge_index = torch.ones(num_nodes, num_nodes).nonzero().T
        nodes = torch.tensor(self._rng.rand(num_nodes, 2))
        graph = torch.cdist(nodes, nodes)
        graph = 0.05+(graph - graph.min())/(graph.max() - graph.min())*0.9 # scales in [0.05, 0.95]
        eye = (torch.arange(graph.shape[0]), torch.arange(graph.shape[0])) # it's a square matrix
        graph[eye] = 0
        source_node = self._rng.choice(num_nodes)
        return [graph, source_node]

def build_geometric_sampler(
    num_samples,
    algorithm,
    spec,
    *args,
    seed=None,
    **kwargs,
):
    sampler = GeometricGraphSampler(
        algorithm, spec, num_samples, seed=seed, *args, **kwargs)
    return sampler, spec
