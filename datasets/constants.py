import os.path as osp
from collections import defaultdict
from datasets.clrs_datasets import CLRS
from datasets.tsp_datasets import TSP, TSPLarge, TSPLIB
from datasets.vertex_k_center import VKCLarge
from clrs._src.specs import SPECS
from clrs import Type, Location, Stage

_DATASET_CLASSES = defaultdict(lambda: CLRS) | {
    'tsp': TSP,
    'tsp_large': TSPLarge,
    'tsplib': TSPLIB,
    'VKC': VKCLarge,
}

_DATASET_ROOTS = defaultdict(lambda: osp.abspath('./data/clrs/')) | {
    'tsp': osp.abspath('./data/tsp'),
    'tsplib': osp.abspath('./data/tsplib'),
    'tsp_large': osp.abspath('./data/tsp_large'),
    'VKC': osp.abspath('./data/VKC'),
}

_MST_SPEC = {
    'edge_attr': ('input', 'edge', 'scalar'),
    'start_marked': ('input', 'node', 'mask_one'),
    'predecessor_index_temporal': ('hint', 'node', 'pointer'),
    'key_temporal': ('hint', 'node', 'scalar'),
    'marked_temporal': ('hint', 'node', 'mask'),
    'in_queue_temporal': ('hint', 'node', 'mask'),
    'chosen_nodes_temporal': ('hint', 'node', 'mask_one'),
    'output_index': ('output', 'node', 'pointer')
}

_GRAHAM_SPEC = {
    'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'xcoord': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'ycoord': (Stage.INPUT, Location.NODE, Type.SCALAR),
    'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
    'best_temporal': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'atans_temporal': (Stage.HINT, Location.NODE, Type.SCALAR),
    'in_hull_h_temporal': (Stage.HINT, Location.NODE, Type.MASK),
    'stack_prev_index_temporal': (Stage.HINT, Location.NODE, Type.POINTER),
    'last_stack_temporal': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'i_temporal': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    'phase_temporal': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL)
}

_BF_SPEC = {
    'edge_attr': ('input', 'edge', 'scalar'),
    'start_marked': ('input', 'node', 'mask_one'),
    'predecessor_index_temporal': ('hint', 'node', 'pointer'),
    'key_temporal': ('hint', 'node', 'scalar'),
    'marked_temporal': ('hint', 'node', 'mask'),
    'in_queue_temporal': ('hint', 'node', 'mask'),
    'chosen_nodes_temporal': ('hint', 'node', 'mask_one'),
    'output_index': ('output', 'node', 'pointer')
}

_DATASET_SPECS = defaultdict(lambda: dict({
    'dataclass': CLRS,
    'rootdir': _DATASET_ROOTS['default'],
    # 'dataset_spec': ,
})) | {
    'mst_prim': {
        'dataclass': CLRS,
        'rootdir': _DATASET_ROOTS['mst_prim'],
        'data_spec': _MST_SPEC,
    },
    'bellman_ford': {
        'dataclass': CLRS,
        'rootdir': _DATASET_ROOTS['mst_prim'],
        'data_spec': _BF_SPEC,
    },
    'tsp': {
        'dataclass': TSP,
        'rootdir': _DATASET_ROOTS['tsp'],
        # 'data_spec': _MST_SPEC,
    },
    'VKC': {
        'dataclass': VKCLarge,
        'rootdir': _DATASET_ROOTS['VKC'],
        # 'data_spec': _MST_SPEC,
    },
    'graham_scan': {
        'dataclass': CLRS,
        'rootdir': _DATASET_ROOTS['graham_scan'],
        'data_spec': _GRAHAM_SPEC,
    },
    'tsp_large': {
        'dataclass': TSPLarge,
        'rootdir': _DATASET_ROOTS['tsp_large'],
        # 'data_spec': _MST_SPEC,
    },
    'tsplib': {
        'dataclass': TSPLIB,
        'rootdir': _DATASET_ROOTS['tsplib'],
        # 'data_spec': _MST_SPEC,
    },
}
