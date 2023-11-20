"""
Script to test the TSP model and a combination of deterministic algorithms

Usage:
   test_reasoner.py (--load-model-from LFM) [options]

Options:
    -h --help              Show this screen.

    --load-model-from LFM  Path to the model to be loaded

    --seed S               Random seed to set. [default: 47]
"""
import os
from docopt import docopt
import schema

import pytorch_lightning as pl

from models.algorithm_processor import LitAlgorithmProcessor
from hyperparameters import get_hyperparameters
from datasets.constants import _DATASET_ROOTS

if __name__ == '__main__':
    serialised_models_dir = os.path.abspath('./serialised_models/')
    hidden_dim = get_hyperparameters()['dim_latent']
    schema = schema.Schema({
        '--help': bool,
        '--load-model-from': schema.Or(None, os.path.exists),
        '--seed': schema.Use(int),
    })
    args = docopt(__doc__)
    args = schema.validate(args)

    lit_processor = LitAlgorithmProcessor.load_from_checkpoint(
        args['--load-model-from'],
        dataset_root=_DATASET_ROOTS['mst_prim'],
    )

    trainer = pl.Trainer(
        accelerator='cuda',
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
    )
    trainer.test(
        model=lit_processor,
    )
