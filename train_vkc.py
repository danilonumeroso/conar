"""
Script to train the VKC model and a combination of deterministic algorithms

Usage:
    train_vkc.py [options]

Options:
    -h --help              Show this screen.

    --xavier-on-scalars    Use Xavier initialisation for linears that encode scalars.

    --patience P           Patience value. If present, the training will utilise
                           early stopping based on validation loss.

    --max-epochs ME        The maximum epochs to train for. If patience value is not
                           provided it will always train for ME epochs. [default: 1000]

    --model-name MN        Name of the model when saving. Defaults to current time
                           and date if not provided.

    --gradient-clip-val G  Constant for gradient clipping. 0 means no clipping.
                           [default: 1]

    --processors PS        Which processors to use. String of comma separated values.
                           [default: MPNN]

    --RPHWM                Whether to Reduce Processor set Hiddens With MLP?

    --biased-gate          Bias the gating mechanism towards less updating

    --update-edges-hidden   Whether to also keep a track of hidden edge state.

    --use-LSTM             Add an LSTMCell just after the processor step
                           (in case of several processors, each has its own LSTM)

    --load-proc-from LPF   Path to load processor from in the 0th index of
                           processor set.

    --freeze-proc          Whether to freeze the processor at index 0 when transferring

    --double-process       Whether to process a batch two times,
                           the first time running _only_ the algorithmic processor.

    --algorithms ALGOS     List of algorithms to train on. Repeatable. [default: mst_prim]

    --seed S               Random seed to set. [default: 47]

    --test-with-val-ep EP   How often to test dataset with validation.
                            [default: 20]

    --test-train-ep EP      How often to test on train data. [default: 20]

"""
import os
from datetime import datetime

from docopt import docopt
import schema
import torch
import wandb
import pytorch_lightning as pl

from models.algorithm_processor import LitAlgorithmProcessor
from models.algorithm_reasoner import LitAlgorithmReasoner
from models.vkc_reasoner import LitVKCReasoner
from models.gnns import _PROCESSSOR_DICT
from hyperparameters import get_hyperparameters
from utils_execution import get_callbacks, maybe_remove
from datasets.constants import _DATASET_CLASSES, _DATASET_ROOTS

if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    hidden_dim = get_hyperparameters()['dim_latent']
    serialised_models_dir = os.path.abspath('./serialised_models/')
    schema = schema.Schema({
        '--help': bool,
        '--xavier-on-scalars': bool,
        '--patience': schema.Or(None, schema.Use(int)),
        '--max-epochs': schema.Or(None, schema.Use(int)),
        '--model-name': schema.Or(None, schema.Use(str)),
        '--gradient-clip-val': schema.Use(int),
        '--processors': schema.And(schema.Use(lambda x: x.split(',')), lambda lst: all(x in _PROCESSSOR_DICT for x in lst)),
        '--RPHWM': bool,
        '--biased-gate': bool,
        '--update-edges-hidden': bool,
        '--use-LSTM': bool,
        '--load-proc-from': schema.Or(None, os.path.exists),
        '--freeze-proc': bool,
        '--double-process': bool,
        '--algorithms': schema.Use(lambda x: list(filter(lambda y: len(y) > 0, x.split(',')))),
        '--seed': schema.Use(int),
        '--test-with-val-ep': schema.Use(int),
        '--test-train-ep': schema.Use(int),
    })
    args = docopt(__doc__)
    args = schema.validate(args)
    name = args['--model-name'] if args['--model-name'] is not None else datetime.now().strftime('%b-%d-%Y-%H-%M')
    pl.utilities.seed.seed_everything(args['--seed'])

    lit_processor = LitAlgorithmProcessor(
        hidden_dim,
        ['VKC']+args['--algorithms'],
        dict((algo, {}) for algo in args['--algorithms']) | ({'VKC': {}}),
        dict((algo, LitAlgorithmReasoner) for algo in args['--algorithms']) | ({'VKC': LitVKCReasoner}),
        False, # That's for ensure permutation of TSP 
        use_gate=True,
        biased_gate=args['--biased-gate'],
        use_LSTM=args['--use-LSTM'],
        freeze_proc=args['--freeze-proc'],
        double_process=args['--double-process'],
        processors=args['--processors'],
        reduce_proc_hid_w_MLP=args['--RPHWM'],
        update_edges_hidden=args['--update-edges-hidden'],
        xavier_on_scalars=args['--xavier-on-scalars'],
        test_train_every_n_epoch=args['--test-train-ep'],
        test_with_val_every_n_epoch=args['--test-with-val-ep'],
    )

    if args['--load-proc-from'] is not None:
        stdc = torch.load(args['--load-proc-from'])['state_dict']
        striplen = len('processor_set.processors.0.')
        stdc = dict((k[striplen:], v) for k, v in stdc.items() if 'processor_set' in k)
        # lit_processor.processor_set.load_state_dict(stdc)
        lit_processor.processor_set.processors[0].load_state_dict(stdc)
        print("PROCESSOR LOADED FROM", args['--load-proc-from'])

    if args['--freeze-proc']:
        lit_processor.processor_set.processors[0].freeze()

    what_to_monitor = 'VKC'
    all_cbs = get_callbacks(name, serialised_models_dir, args['--patience'], monitor=f'val/acc/{what_to_monitor}/farthest_relative_error')
    maybe_remove(f'./serialised_models/best_{name}.ckpt')
    maybe_remove(f'./serialised_models/{name}-epoch_*.ckpt')
    wandb_logger = pl.loggers.WandbLogger(project='conar', entity='d-n-d', group='RERUN: 20-40 size, 100K data VKC, pick lowest err', log_model=True)
    wandb_logger.experiment.config.update({
        'batch_size': get_hyperparameters()['batch_size'],
        'load_proc': args['--load-proc-from'] is not None,
    })

    trainer = pl.Trainer(
        accelerator='cuda',
        max_epochs=args['--max-epochs'],
        callbacks=all_cbs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        gradient_clip_val=args['--gradient-clip-val'],
        logger=wandb_logger,
    )
    trainer.test(
        model=lit_processor,
    )
    trainer.fit(
        model=lit_processor,
    )
    trainer.test(
        ckpt_path='best',
    )
