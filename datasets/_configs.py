from collections import defaultdict

_DEFAULT_CONFIG = {
    "train": {
        "num_samples": 10000,
        "num_nodes": 16
    },
    "val": {
        "num_samples": 100,
        "num_nodes": 16
    },
    "test": {
        "num_samples": 100,
        "num_nodes": 64
    }
}
CONFIGS = defaultdict(lambda: _DEFAULT_CONFIG)
CONFIGS['tsp'] = {
    "train": {
        "num_samples": 10000,
        "num_nodes": 20
    },
    "val": {
        "num_samples": 100,
        "num_nodes": 20
    },
    "test": {
        "num_samples": 100,
        "num_nodes": 40
    },
    "test_100": {
        "num_samples": 32,
        "num_nodes": 100
    },
    "test_200": {
        "num_samples": 32,
        "num_nodes": 200
    },
    "test_1000": {
        "num_samples": 4,
        "num_nodes": 1000
    }
}

CONFIGS['tsp_large'] = {
    "train": {
        "num_samples": 100000,
        "num_nodes": [10, 13, 16, 19, 20]
    },
    "val": {
        "num_samples": 1000,
        "num_nodes": 20
    },
    "test": {
        "num_samples": 1000,
        "num_nodes": 40,
    },
    "test_20": {
        "num_samples": 32,
        "num_nodes": 20,
    },
    "test_60": {
        "num_samples": 32,
        "num_nodes": 60,
    },
    "test_80": {
        "num_samples": 32,
        "num_nodes": 80,
    },
    "test_100": {
        "num_samples": 32,
        "num_nodes": 100,
    },
    "test_200": {
        "num_samples": 32,
        "num_nodes": 200,
    },
    "test_1000": {
        "num_samples": 4,
        "num_nodes": 1000,
    },
}

CONFIGS['VKC'] = {
    "train": {
        "num_samples": 100000,
        "num_nodes": [10, 13, 16, 19, 20]
    },
    "val": {
        "num_samples": 1000,
        "num_nodes": 20
    },
    "test": {
        "num_samples": 1000,
        "num_nodes": 40,
    },
    "test_20":  {
        "num_samples": 32,
        "num_nodes": 20,
    },
    "test_60": {
        "num_samples": 32,
        "num_nodes": 60,
    },
    "test_80": {
        "num_samples": 32,
        "num_nodes": 80,
    },
    "test_100": {
        "num_samples": 32,
        "num_nodes": 100,
    },
    "test_200": {
        "num_samples": 32,
        "num_nodes": 200,
    },
}
CONFIGS['tsplib'] = {
    "test_all": {
        "num_samples": 1000,
        "num_nodes": 1000,
    }
}
