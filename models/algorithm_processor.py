from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from models.gnns import LitProcessorSet
from hyperparameters import get_hyperparameters
from models.algorithm_reasoner import LitAlgorithmReasoner
from datasets.constants import _DATASET_CLASSES, _DATASET_ROOTS
from train_config import MODULE_CONFIG

class LitAlgorithmProcessor(pl.LightningModule):

    def __init__(self,
                 hidden_dim,
                 algorithm_names,
                 dataset_kwargs,
                 algo_classes,
                 ensure_permutation,
                 processors=['MPNN'],
                 bias=get_hyperparameters()['bias'],
                 reduce_proc_hid_w_MLP=False,
                 update_edges_hidden=False,
                 use_gate=False,
                 use_LSTM=False,
                 use_ln=False,
                 use_TF=False,
                 transferring=False,
                 freeze_proc=False,
                 double_process=False,
                 xavier_on_scalars=False,
                 biased_gate=False,
                 test_with_val=True,
                 test_with_val_every_n_epoch=20,
                 test_train_every_n_epoch=20,
                 lr=get_hyperparameters()['lr'],
                 weight_decay=get_hyperparameters()['weight_decay']):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.processors = processors
        self.bias = bias
        self.reduce_proc_hid_w_MLP = reduce_proc_hid_w_MLP
        self.use_gate = use_gate
        self.use_LSTM = use_LSTM
        self.use_ln = use_ln
        self.use_TF = use_TF
        self.update_edges_hidden = update_edges_hidden
        self.transferring = transferring
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.xavier_on_scalars = xavier_on_scalars
        self.biased_gate = biased_gate
        self.freeze_proc = freeze_proc
        self.double_process = double_process
        self.test_with_val = test_with_val
        self.test_with_val_every_n_epoch = test_with_val_every_n_epoch
        self.test_train_every_n_epoch = test_train_every_n_epoch
        self.val_dataloader = self.val_dataloader_normal
        if self.test_with_val:
            self.val_dataloader = self.val_dataloader_alt
            self.validation_step = self.validation_step_alt
        self.processor_set = LitProcessorSet(
            2*hidden_dim,
            hidden_dim,
            reduce_with_MLP=reduce_proc_hid_w_MLP,
            update_edges_hidden=update_edges_hidden,
            edge_dim=hidden_dim,
            bias=bias,
            use_gate=use_gate,
            use_LSTM=use_LSTM,
            use_ln=use_ln,
            biased_gate=biased_gate,
            processors=processors)
        self.algorithm_names = algorithm_names
        self.algorithms = nn.ModuleDict()
        for algo in algorithm_names:
            self.algorithms[algo] = algo_classes[algo](
                algorithm=algo,
                hidden_dim=hidden_dim,
                algo_processor=self.processor_set,
                dataset_class=_DATASET_CLASSES[algo],
                dataset_root=_DATASET_ROOTS[algo],
                dataset_kwargs=dataset_kwargs[algo],
                bias=bias,
                use_TF=use_TF,
                transferring=transferring,
                ensure_permutation=ensure_permutation,
                xavier_on_scalars=xavier_on_scalars,
                test_with_val=False, # ALWAYS FALSE
                test_with_val_every_n_epoch=test_with_val_every_n_epoch,
                test_train_every_n_epoch=test_train_every_n_epoch,
                double_process=self.double_process,
            )
        self.save_hyperparameters(ignore=[])
        self.debug_epoch = 1e9

    def train_dataloader(self):
        return [self.algorithms[algo].train_dataloader() for algo in self.algorithm_names]
        # return CombinedLoader(dict((name, algo.train_dataloader()) for name, algo in self.algorithms.items()), mode='max_size_cycle')

    def val_dataloader_normal(self):
        return CombinedLoader(dict((name, algo.val_dataloader()) for name, algo in self.algorithms.items()), mode='max_size_cycle')

    def val_dataloader_alt(self):
        return [self.val_dataloader_normal(), self.test_dataloader()]

    def test_dataloader(self, suffix=''):
        return CombinedLoader(dict((name, algo.test_dataloader(suffix=suffix)) for name, algo in self.algorithms.items()), mode='max_size_cycle')


    def forward(self, batch):
        return self.fwd_step(batch, 0)

    def fwd_step(self, batch, batch_idx):
        assert not self.freeze_proc or not any(k.requires_grad for k in self.processor_set.processors[0].parameters()), breakpoint()
        outputs = {}
        for name, algorithm in self.algorithms.items():
            outputs[name] = algorithm.fwd_step(batch[name], batch_idx)
        return outputs

    def on_train_epoch_start(self):
        for algorithm in self.algorithms.values():
            algorithm.current_epoch = self.current_epoch


    def training_step(self, batch, batch_idx):
        total_loss = 0
        for name, algo_batch in zip(self.algorithm_names, batch):
            algorithm = self.algorithms[name]
            output = algorithm.training_step(algo_batch, batch_idx)
            if isinstance(algo_batch, list):
                num_graphs = algo_batch[0].num_graphs
            else:
                num_graphs = algo_batch.num_graphs
            self.log_dict(dict((f'train/loss/{name}/{k}', v) for k, v in output['losses_dict'].items()), batch_size=num_graphs)
            self.log(f'train/loss/{name}/average_loss', output['loss'], on_step=True, on_epoch=True, batch_size=num_graphs)
            self.log_dict(dict((f'train/acc/{name}/{k}', v) for k, v in output['accuracies'].items()), batch_size=num_graphs, add_dataloader_idx=False, on_epoch=True, on_step=False)
            total_loss = total_loss + output['loss']
        total_loss = total_loss / len(self.algorithms)
        self.log('train/loss/average_loss', total_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=num_graphs)
        if self.current_epoch >= self.debug_epoch:
            breakpoint()
        return {'loss': total_loss}

    def valtest_step(self, batch, batch_idx, mode):
        output = {}
        total_loss = 0
        for name, algorithm in self.algorithms.items():
            output[name] = algorithm.valtest_step(batch[name], batch_idx, mode)
            self.log_dict(dict((f'{mode}/loss/{name}/{k}', v) for k, v in output[name]['losses'].items()), batch_size=batch[name].num_graphs, add_dataloader_idx=False)
            average_loss = sum(output[name]['losses'].values()) / len(output[name]['losses'])
            self.log(f'{mode}/loss/{name}/average_loss', average_loss, batch_size=batch[name].num_graphs, add_dataloader_idx=False, on_epoch=True)
            self.log_dict(dict((f'{mode}/acc/{name}/{k}', v) for k, v in output[name]['accuracies'].items()), batch_size=batch[name].num_graphs, add_dataloader_idx=False)
            total_loss = total_loss + average_loss
        total_loss = total_loss / len(self.algorithms)
        self.log(f'{mode}/loss/average_loss', total_loss, batch_size=batch[name].num_graphs, add_dataloader_idx=False)
        return output


    def validation_step_alt(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1 and not self.trainer.state.stage == 'sanity_check' and self.current_epoch % self.test_with_val_every_n_epoch == 0:
            return self.valtest_step(batch, batch_idx, 'periodic_test')
        if dataloader_idx == 0:
            return self.valtest_step(batch, batch_idx, 'val')

    def validation_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        term_params = []
        normal_params = []
        for name, param in self.named_parameters():
            if '_term' in name or 'termination' in name or 'predinet' in name:
                term_params.append(param)
            else:
                normal_params.append(param)
        lr = self.learning_rate
        optimizer = optim.Adam([
                                   {'params': term_params, 'lr': lr},
                                   {'params': normal_params, 'lr': lr}
                               ],
                               lr=lr,
                               weight_decay=self.weight_decay)
        return optimizer

if __name__ == '__main__':
    ...
