"""
Script to test the TSP/VKC.

Usage:
    test_co.py (--load-model-from LFM) [options]

Options:
    -h --help              Show this screen.

    --load-model-from LFM  Path to the model to be loaded

    --test-size TS         Number of nodes in the test set graphs
                           [default: 40]
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
        '--test-size': schema.Use(int),
    })
    args = docopt(__doc__)
    args = schema.validate(args)

    lit_processor = LitAlgorithmProcessor.load_from_checkpoint(
        args['--load-model-from'],
    )
    trainer = pl.Trainer(
        accelerator='cuda',
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
    )
    suffix = ''
    if args['--test-size'] != 40:
        suffix = f"_{args['--test-size']}"
    trainer.test(
        model=lit_processor,
        dataloaders=lit_processor.test_dataloader(suffix=suffix)
    )
