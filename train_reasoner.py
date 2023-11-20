"""
Script to train the reasoner model.

Usage:
    train_reasoner.py [options]

Options:
    -h --help               Show this screen.

    --patience P            Patience value. If present, the training will utilise
                            early stopping based on validation loss.

    --max-epochs ME         The maximum epochs to train for. If patience value is not
                            provided it will always train for ME epochs. [default: 1000]

    --model-name MN         Name of the model when saving. Defaults to current time
                            and date if not provided.

    --processors PS         Which processors to use. String of comma separated values.
                            [default: MPNN]

    --RPHWM                 Whether to Reduce Processor set Hiddens With MLP?

    --gradient-clip-val G   Constant for gradient clipping. 0 means no clipping.
                            [default: 1]

    --xavier-on-scalars     Use Xavier initialisation for linears that encode scalars.

    --biased-gate           Bias the gating mechanism towards less updating

    --update-edges-hidden   Whether to also keep a track of hidden edge state.

    --use-LSTM              Add an LSTMCell just after the processor step
                            (in case of several processors, each has its own LSTM)

    --use-ln                Use Layer Norm in the processor.

    --algorithms ALGOS      List of algorithms to train on. Repeatable. [default: mst_prim]

    --sampler-type (default|geometric)    What sampler was used for graph generation. [default: default]

    --seed S                Random seed to set. [default: 47]
"""
import os
from datetime import datetime
from collections import defaultdict

from docopt import docopt
import schema
import torch
import pytorch_lightning as pl


from models.gnns import _PROCESSSOR_DICT
from models.algorithm_reasoner import LitAlgorithmReasoner
from models.algorithm_processor import LitAlgorithmProcessor
from hyperparameters import get_hyperparameters
from utils_execution import ReasonerZeroerCallback, get_callbacks, maybe_remove


if __name__ == '__main__':
    hidden_dim = get_hyperparameters()['dim_latent']
    serialised_models_dir = os.path.abspath('./serialised_models/')
    schema = schema.Schema({
        '--help': bool,
        '--xavier-on-scalars': bool,
        '--biased-gate': bool,
        '--update-edges-hidden': bool,
        '--use-LSTM': bool,
        '--use-ln': bool,
        '--patience': schema.Or(None, schema.Use(int)),
        '--max-epochs': schema.Or(None, schema.Use(int)),
        '--model-name': schema.Or(None, schema.Use(str)),
        '--processors': schema.And(schema.Use(lambda x: x.split(',')), lambda lst: all(x in _PROCESSSOR_DICT for x in lst)),
        '--RPHWM': bool,
        '--gradient-clip-val': schema.Use(int),
        '--algorithms': schema.Use(lambda x: x.split(',')),
        '--sampler-type': str,
        '--seed': schema.Use(int),
    })
    args = docopt(__doc__)
    args = schema.validate(args)
    name = args['--model-name'] if args['--model-name'] is not None else datetime.now().strftime('%b-%d-%Y-%H-%M')
    pl.utilities.seed.seed_everything(args['--seed'])

    lit_processor = LitAlgorithmProcessor(
        hidden_dim,
        args['--algorithms'],
        dict((algo, {'sampler_type': args['--sampler-type']}) for algo in args['--algorithms']),
        dict((algo, LitAlgorithmReasoner) for algo in args['--algorithms']),
        False, #args['--ensure-permutation'] is False for non-TSP
        reduce_proc_hid_w_MLP=args['--RPHWM'],
        update_edges_hidden=args['--update-edges-hidden'],
        use_TF=False,
        use_gate=True,
        use_LSTM=args['--use-LSTM'],
        use_ln=args['--use-ln'],
        freeze_proc=False, # We don't have a transfer task
        processors=args['--processors'],
        xavier_on_scalars=args['--xavier-on-scalars'],
        biased_gate=args['--biased-gate'],
    )
    

    all_cbs = get_callbacks(name, serialised_models_dir, args['--patience'])
    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=args['--max-epochs'],
        callbacks=all_cbs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        gradient_clip_val=args['--gradient-clip-val'],
        logger=pl.loggers.WandbLogger(project='conar', entity='d-n-d', log_model=True, group=None),
    )
    maybe_remove(f'./serialised_models/best_{name}.ckpt')
    maybe_remove(f'./serialised_models/{name}-epoch_*.ckpt')
    trainer.test(
        model=lit_processor,
    )
    trainer.fit(
        model=lit_processor,
    )
    trainer.test(
        ckpt_path='best',
    )
