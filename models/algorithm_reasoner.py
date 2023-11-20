import time
from pprint import pprint
from collections import defaultdict
import copy
from torch_sparse import SparseTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

import torch_geometric
import torch_geometric.utils as tg_utils
from torch_geometric.loader import DataLoader
import torch_scatter
import pytorch_lightning as pl

from datasets.constants import _DATASET_CLASSES, _DATASET_ROOTS
from datasets._configs import CONFIGS
from hyperparameters import get_hyperparameters
from utils_execution import cross_entropy, check_edge_index_sorted, prepare_constants, edge_one_hot_encode_pointers, edge_one_hot_encode_pointers_edge
from clrs import Type, Location, Stage
from layers.gnns import TripletMPNN


def sinkhorn_normalize(batch, y, temperature, steps=10, add_noise=False):

    Inf = 1e5
    from_, to = batch.edge_index[0], batch.edge_index[1]

    if add_noise:
        eps = -torch.log(-torch.log(torch.rand_like(y) + 1e-12) + 1e-12)
        y = y + eps

    y = y.masked_fill(from_ == to, -Inf)
    y = y / temperature

    for _ in range(steps):
        y = torch_scatter.scatter_log_softmax(y, from_, dim_size=batch.num_nodes)
        y = torch_scatter.scatter_log_softmax(y, to, dim_size=batch.num_nodes)

    return y


class AlgorithmReasoner(nn.Module):
    @staticmethod
    def prepare_batch(batch):
        batch = batch.clone()
        for name, tensor in batch.items():
            if not torch.is_tensor(tensor):
                continue
            if name.endswith('_temporal') and 'index' not in name:
                tensor = tensor.transpose(1, 0)
                batch[name] = tensor
        return batch

    @staticmethod
    def get_masks(train, batch, continue_logits, enforced_mask):
        mask = continue_logits[batch.batch] > 0
        mask_cp = (continue_logits > 0.0).bool()
        mask_edges = mask[batch.edge_index[0]]
        if not train and enforced_mask is not None:
            enforced_mask_ids = enforced_mask[batch.batch]
            mask &= enforced_mask_ids
            mask_cp &= enforced_mask
        return mask_cp, mask, mask_edges

    def add_encoder(self, stage, name, loc, data_type, data_sample, bias):
        if name == 'adj': # we use edge indices
            return
        if data_type == Type.SCALAR or data_type == Type.MASK or data_type == Type.MASK_ONE:
            self.encoders[stage][name] = nn.Linear(1, self.latent_features, bias=bias)

        if data_type == Type.CATEGORICAL:
            in_shape = data_sample.shape[-1]
            self.encoders[stage][name] = nn.Linear(in_shape, self.latent_features, bias=bias)

        if loc == Location.NODE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]: # pointers are 1-hot encoded on the edges
            self.encoders[stage][name] = nn.Linear(1, self.latent_features, bias=bias)
        if loc == Location.EDGE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
            self.encoders[stage][name] = nn.ModuleList([
                nn.Linear(1, self.latent_features, bias=bias),
                nn.Linear(1, self.latent_features, bias=bias)
            ])

    def add_decoder(self, stage, name, loc, data_type, data_sample, bias):
        assert name != 'adj', 'Adjacency matrix should not be decoded'
        dec = None
        if loc == Location.NODE:
            if data_type in (Type.SCALAR, Type.MASK, Type.MASK_ONE):
                dec = nn.Linear(2*self.latent_features, 1, bias=bias)

            if data_type == Type.CATEGORICAL:
                in_shape = data_sample.shape[-1]
                dec = nn.Linear(2*self.latent_features, in_shape, bias=bias)

            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]: # pointers are decoded from both node and edge information
                dec = nn.ModuleList([
                        nn.Linear(2*self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(2*self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(self.latent_features, 1, bias=bias),
                ])
        if loc == Location.GRAPH:
            if data_type in [Type.MASK, Type.SCALAR, Type.CATEGORICAL, Type.MASK_ONE]:
                in_shape = data_sample.shape[-1] if data_type == Type.CATEGORICAL else 1
                dec = nn.ModuleList([
                        nn.Linear(2*self.latent_features, in_shape, bias=bias),
                        nn.Linear(self.latent_features, in_shape, bias=bias),
                ])

        if loc == Location.EDGE:
            if data_type in (Type.SCALAR, Type.MASK, Type.MASK_ONE):
                dec = nn.ModuleList([
                        nn.Linear(2*self.latent_features, 1, bias=bias),
                        nn.Linear(2*self.latent_features, 1, bias=bias),
                        nn.Linear(self.latent_features, 1, bias=bias),
                ])
            if data_type == Type.CATEGORICAL:
                in_shape = data_sample.shape[-1]
                dec = nn.ModuleList([
                        nn.Linear(2*self.latent_features, in_shape, bias=bias),
                        nn.Linear(2*self.latent_features, in_shape, bias=bias),
                        nn.Linear(self.latent_features, in_shape, bias=bias),
                ])
            if data_type == Type.POINTER:
                dec = nn.ModuleList([
                        nn.Linear(2*self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(2*self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(2*self.latent_features, self.latent_features, bias=bias),
                        nn.Linear(self.latent_features, 1, bias=bias),
                ])
        assert dec is not None, breakpoint()
        self.decoders[stage][name] = dec




    def __init__(self,
                 spec,
                 data,
                 latent_features,
                 algo_processor,
                 bias=True,
                 use_TF=False,
                 use_sinkhorn=True,
                 L1_loss=False,
                 xavier_on_scalars=True,
                 global_termination_pool='max', #'predinet',
                 get_attention=False,
                 use_batch_norm=False,
                 transferring=False,
                 timeit=True,
                 **kwargs):

        super().__init__()
        self.step_idx = 0
        self.latent_features = latent_features
        self.assert_checks = False
        self.timeit = timeit
        self.debug = False
        self.debug_epoch_threshold = 1e9
        self.L1_loss = L1_loss
        self.global_termination_pool = global_termination_pool
        self.next_step_pool = True
        self.processor = algo_processor
        self.triplet_reasoning = False
        if isinstance(self.processor.processors[0].processor, TripletMPNN):
            self.triplet_reasoning = True
            self.triplet_reductor = nn.Linear(2*latent_features, latent_features, bias=bias)
        self.use_TF = use_TF
        self.use_sinkhorn = use_sinkhorn
        self.get_attention = get_attention
        self.lambda_mul = 1  # 0.0001
        self.transferring = transferring
        self.node_encoder = nn.Sequential(
            nn.Linear(2*latent_features, latent_features, bias=bias),
        )
        self.encoders = nn.ModuleDict({
            'input': nn.ModuleDict({
            }),
            'hint': nn.ModuleDict({
            }),
        })
        self.decoders = nn.ModuleDict({
            'hint': nn.ModuleDict({
            }),
            'output': nn.ModuleDict({
            })
        })
        for name, (stage, loc, datatype) in spec.items():
            if name == 'adj': # we use edge indices
                continue
            if stage == 'input':
                self.add_encoder(stage, name, loc, datatype, getattr(data, name), bias)
            if stage == 'output':
                self.add_decoder(stage, name, loc, datatype, getattr(data, name), bias)
            if stage == 'hint':
                self.add_encoder(stage, name, loc, datatype, getattr(data, name), bias)
                self.add_decoder(stage, name, loc, datatype, getattr(data, name), bias)

        self.node_pointer_vec = nn.Parameter(torch.randn(latent_features))
        if xavier_on_scalars:
            assert False, "NEEDS REFACTORING"
            torch.nn.init.trunc_normal_(self.encoders['input']['edge_attr'].weight, std=1/torch.sqrt(torch.tensor(latent_features)))

        if global_termination_pool == 'attention':
            inp_dim = latent_features
            self.global_attn = GlobalAttentionPlusCoef(
                    nn.Sequential(
                        nn.Linear(inp_dim, latent_features, bias=bias),
                        nn.LeakyReLU(),
                        nn.Linear(latent_features, 1, bias=bias)
                    ),
                    nn=None)

        if global_termination_pool == 'predinet':
            lf = latent_features
            self.predinet = PrediNet(lf, 1, lf, lf, flatten_pooling=torch_geometric.nn.glob.global_max_pool)

        self.termination_network = nn.Sequential(
            nn.BatchNorm1d(latent_features) if use_batch_norm else nn.Identity(),
            nn.Linear(latent_features, 1, bias=bias),
        )

    def get_continue_logits(self, batch_ids, latent_nodes, sth_else=None):
        if self.global_termination_pool == 'mean':
            graph_latent = torch_geometric.nn.global_mean_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'max':
            graph_latent = torch_geometric.nn.global_max_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'attention':
            graph_latent, coef = self.global_attn(latent_nodes, batch_ids)
            if self.get_attention:
                self.attentions[self.step_idx] = coef.clone().detach()
                self.per_step_latent[self.step_idx] = sth_else

        if self.global_termination_pool == 'predinet':
            assert not torch.isnan(latent_nodes).any()
            graph_latent = self.predinet(latent_nodes, batch_ids)

        if self.get_attention:
            self.attentions[self.step_idx] = latent_nodes
        continue_logits = self.termination_network(graph_latent).view(-1)
        return continue_logits

    def zero_termination(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def zero_steps(self):
        self.sum_of_processed_nodes = 0
        self.sum_of_processed_edges = 0
        self.step_idx = 0
        self.sum_of_steps = 0
        self.cnt = 0

    @staticmethod
    def convert_logits_to_outputs(spec,
                                  logits,
                                  fr,
                                  to,
                                  num_nodes,
                                  batch_ids,
                                  include_probabilities=True,
                                  dbg=False):
        outs = defaultdict(dict)

        for stage in logits.keys():
            for name in logits[stage].keys():
                if name not in logits[stage] or name not in spec:
                    continue
                stage, loc, data_type = spec[name]
                assert stage != Stage.INPUT
                if data_type == Type.SOFT_POINTER:
                    assert False, f"Not yet added, please add {name}"
                if data_type in [Type.CATEGORICAL]:
                    indices = logits[stage][name].argmax(-1)
                    outshape = logits[stage][name].shape[-1]
                    outs[stage][name] = F.one_hot(indices, num_classes=outshape).float()
                if data_type == Type.MASK_ONE:
                    _, amax = torch_scatter.scatter_max(logits[stage][name], batch_ids, dim=0)
                    amax = amax.squeeze(-1)
                    outs[stage][name] = torch.zeros_like(logits[stage][name])
                    outs[stage][name][amax] = 1
                if data_type == Type.MASK:
                    outs[stage][name] = (logits[stage][name] > 0).float()
                if data_type == Type.SCALAR:
                    outs[stage][name] = logits[stage][name]
                if loc == Location.NODE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                    pointer_logits = logits[stage][name]
                    _, pointers = torch_scatter.scatter_max(pointer_logits, fr, dim_size=num_nodes)
                    pointers = to[pointers]
                    pointer_probabilities = torch_geometric.utils.softmax(pointer_logits, fr, num_nodes=num_nodes)
                    outs[stage][name] = pointers
                    if include_probabilities:
                        outs[stage][f'{name}_probabilities'] = pointer_probabilities
                if loc == Location.EDGE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                    pointer_logits = logits[stage][name]
                    pointers = pointer_logits.argmax(-1)
                    pointer_probabilities = F.softmax(pointer_logits, dim=-1)
                    outs[stage][name] = pointers
                    if include_probabilities:
                        outs[stage][f'{name}_probabilities'] = pointer_probabilities
        return outs

    def set_initial_states(self, batch, init_last_latent=None):
        self.processor.zero_lstm(batch.num_nodes) # NO-OP if processor(s) don't use LSTM
        self.last_latent = torch.zeros(batch.num_nodes, self.latent_features, device=batch.edge_index.device)
        if init_last_latent is not None:
            self.last_latent = init_last_latent
        self.last_latent_edges = torch.zeros(batch.num_edges, self.latent_features, device=batch.edge_index.device)
        self.last_continue_logits = torch.ones(batch.num_graphs, device=batch.edge_index.device)
        self.last_logits = defaultdict(dict)


        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage == Stage.INPUT:
                continue
            if name not in self.decoders[stage]:
                continue
            if stage == Stage.OUTPUT:

                if loc in [Location.NODE, Location.GRAPH]:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)
                    if data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name).unsqueeze(-1)
                    if data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(getattr(batch, name).bool(), 1e9, -1e9).unsqueeze(-1)
                    if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        self.last_logits[stage][name] = torch.where(batch.edge_index[0, :] == batch.edge_index[1, :], 1e9, -1e9).to(batch.edge_index.device) # self-loops

                if loc == Location.EDGE:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)
                    elif data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(getattr(batch, name).bool(), 1e9, -1e9).unsqueeze(-1)
                    elif data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        ptrs = getattr(batch, name).int()
                        starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                        ptrs = ptrs - starts_edge
                        self.last_logits[stage][name] = torch.full((batch.edge_index.shape[1], int(ptrs.max().item())+1), -1e9).to(batch.edge_index.device)
                        self.last_logits[stage][name][torch.arange(ptrs.shape[0]), ptrs] = 1e9
                    else:
                        assert False, breakpoint()

            if stage == Stage.HINT:

                if loc in [Location.NODE, Location.GRAPH]:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)[0]
                    elif data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name)[0].unsqueeze(-1)
                    elif data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(getattr(batch, name)[0, :].bool(), 1e9, -1e9).unsqueeze(-1)
                    elif data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        self.last_logits[stage][name] = torch.where(batch.edge_index[0, :] == batch.edge_index[1, :], 1e9, -1e9).to(batch.edge_index.device) # self-loops
                    else:
                        assert False, breakpoint()

                if loc == Location.EDGE:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)[0]
                    elif data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(getattr(batch, name)[0, :].bool(), 1e9, -1e9).unsqueeze(-1)
                    elif data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name)[0, :].unsqueeze(-1)
                    elif data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        ptrs = getattr(batch, name)[0, :].int()
                        starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                        ptrs = ptrs - starts_edge
                        self.max_nodes_in_graph = int(ptrs.max().item())+1 # FIXME try another way to infer
                        self.last_logits[stage][name] = torch.where(edge_one_hot_encode_pointers_edge(ptrs, batch, self.max_nodes_in_graph).bool(), 1e9, -1e9).to(batch.edge_index.device)
                    else:
                        assert False, breakpoint()

        self.all_hint_logits = []
        self.all_masks_graph = []

    def update_per_mask(self, before, after, mask=None):
        # NOTE: this does expansion of the mask, if you do
        # NOT use expansion, use torch.where
        if mask is None:
            mask = self.mask
        mask = mask.unsqueeze(-1).expand_as(before)
        return torch.where(mask, after, before)

    def update_state_dict(self, before, after):
        new_before = defaultdict(dict)
        for stage in after.keys():
            for name in after[stage].keys():
                _, loc, data_type = self.dataset_spec[name]
                if loc == Location.GRAPH:
                    new_before[stage][name] = self.update_per_mask(before[stage][name], after[stage][name], mask=self.mask_cp)
                if loc == Location.EDGE:
                    if data_type in [Type.MASK, Type.MASK_ONE, Type.SCALAR, Type.CATEGORICAL, Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        new_before[stage][name] = self.update_per_mask(before[stage][name], after[stage][name], mask=self.mask_edges)
                    else:
                        assert False, "Please implement"
                if loc == Location.NODE:
                    if data_type in [Type.MASK, Type.MASK_ONE, Type.SCALAR, Type.CATEGORICAL]:
                        new_before[stage][name] = self.update_per_mask(before[stage][name], after[stage][name])
                    elif data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                        new_before[stage][name] = torch.where(self.mask_edges, after[stage][name], before[stage][name])
                    else:
                        assert False, breakpoint()
        return new_before

    def update_states(self, batch, current_latent, edges_current_latent,
                      logits, continue_logits):
        self.last_continue_logits = torch.where(self.mask_cp, continue_logits,
                                                self.last_continue_logits)
        self.last_latent = self.update_per_mask(self.last_latent, current_latent)
        self.last_latent_edges = self.update_per_mask(self.last_latent_edges, edges_current_latent, mask=self.mask_edges)
        self.last_logits = self.update_state_dict(self.last_logits, logits)
        self.all_hint_logits.append(self.last_logits['hint'])
        self.all_masks_graph.append(self.mask_cp)
        preds = type(self).convert_logits_to_outputs(
            self.dataset_spec, self.last_logits, batch.edge_index[0],
            batch.edge_index[1], batch.num_nodes, batch.batch,
            self.epoch > self.debug_epoch_threshold)
        self.last_hint = preds['hint']
        self.last_output = preds['output']

    def prepare_initial_masks(self, batch):
        self.mask = torch.ones_like(batch.batch, dtype=torch.bool, device=batch.edge_index.device)
        self.mask_cp = torch.ones(batch.num_graphs, dtype=torch.bool, device=batch.edge_index.device)
        self.mask_edges = torch.ones_like(batch.edge_index[0], dtype=torch.bool, device=batch.edge_index.device)

    def loop_condition(self, termination, STEPS_SIZE):
        return (((not self.training and termination.any()) or
                 (self.training and termination.any())) and
                self.step_idx+1 < STEPS_SIZE)

    def loop_body(self,
                  batch,
                  node_fts,
                  edge_fts,
                  graph_fts,
                  hint_inp_curr,
                  hint_out_curr,
                  true_termination,
                  first_n_processors=1000):

        current_latent, edges_current_latent, preds, continue_logits =\
            self.forward(
                batch,
                node_fts,
                edge_fts,
                graph_fts,
                first_n_processors=first_n_processors,
            )
        termination = continue_logits

        self.debug_batch = batch
        self.debug_hint_out_curr = hint_out_curr
        if self.timeit:
            st = time.time()
        self.update_states(batch, current_latent, edges_current_latent, preds, termination)
        if self.timeit:
            print(f'updating states: {time.time()-st}')

    def get_step_input(self, x_curr, batch):
        if self.training and self.use_TF or self.hardcode_outputs:
            return x_curr
        return type(self).convert_logits_to_outputs(
            self.dataset_spec, self.last_logits, batch.edge_index[0],
            batch.edge_index[1], batch.num_nodes, batch.batch,
            self.epoch > self.debug_epoch_threshold)['hint']

    def encode_inputs(self, batch):
        node_fts = torch.zeros(batch.num_nodes, self.latent_features, device=batch.edge_index.device)
        edge_fts = torch.zeros(batch.num_edges, self.latent_features, device=batch.edge_index.device)
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.INPUT:
                continue
            if name not in self.encoders[stage]:
                continue
            data = getattr(batch, name)
            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                assert False, breakpoint() # we don't have it for now (B-F/MST), will figure out later
            if data_type != Type.CATEGORICAL:
                data = data.unsqueeze(-1)
            if loc == Location.EDGE:
                edge_fts += self.encoders[stage][name](data)
            if loc == Location.NODE:
                node_fts += self.encoders[stage][name](data)
        return node_fts, edge_fts

    def encode_hints(self, hints, batch):
        node_fts = torch.zeros(batch.num_nodes, self.latent_features, device=batch.edge_index.device)
        edge_fts = torch.zeros(batch.num_edges, self.latent_features, device=batch.edge_index.device)
        graph_fts = torch.zeros(batch.num_graphs, self.latent_features, device=batch.edge_index.device)

        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.HINT:
                continue
            if name not in self.encoders[stage]:
                continue
            hint = hints[name]
            if loc == Location.NODE and data_type in [Type.MASK, Type.MASK_ONE, Type.SCALAR, Type.CATEGORICAL]:
                node_fts = node_fts + self.encoders['hint'][name](hint)
            if loc == Location.EDGE and data_type in [Type.MASK, Type.MASK_ONE, Type.SCALAR, Type.CATEGORICAL]:
                edge_fts = edge_fts + self.encoders['hint'][name](hint)
            if loc == Location.NODE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                pred_gt_one_hot = edge_one_hot_encode_pointers(hint, batch.edge_index)
                edge_fts = edge_fts + self.encoders['hint'][name](pred_gt_one_hot.unsqueeze(-1))
            if loc == Location.EDGE and data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                pred_gt_one_hot = edge_one_hot_encode_pointers_edge(hint, batch, self.max_nodes_in_graph)
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                encoding = self.encoders['hint'][name][0](pred_gt_one_hot.unsqueeze(-1))
                encoding_2 = self.encoders['hint'][name][1](pred_gt_one_hot.unsqueeze(-1))
                encoding_sparse = SparseTensor(row=batch.edge_index[0], col=batch.edge_index[1], value=encoding)
                res_1 = encoding_sparse.mean(1)[batch.edge_index[0], batch.edge_index[1]-starts_edge]
                res_2 = encoding_2.mean(1)
                edge_fts += res_1 + res_2 # INPLACE
            if loc == Location.GRAPH and data_type in [Type.CATEGORICAL, Type.SCALAR, Type.MASK]:
                graph_fts = graph_fts + self.encoders['hint'][name](hint)
        return node_fts, edge_fts, graph_fts

    def get_input_output_hints(self, batch):
        hint_inp_curr = {}
        hint_out_curr = {}
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.HINT:
                continue
            hint_inp_curr[name] = getattr(batch, name)[self.step_idx]
            hint_out_curr[name] = getattr(batch, name)[self.step_idx+1]
            if 'mask' in data_type or data_type == Type.SCALAR:
                hint_inp_curr[name] = hint_inp_curr[name].unsqueeze(-1)
                hint_out_curr[name] = hint_out_curr[name].unsqueeze(-1)
        return hint_inp_curr, hint_out_curr

    def process(
            self,
            batch,
            EPSILON=0,
            enforced_mask=None,
            hardcode_outputs=False,
            debug=False,
            first_n_processors=1000,
            init_last_latent=None,
            **kwargs):

        SIZE, STEPS_SIZE = prepare_constants(batch)
        self.hardcode_outputs = hardcode_outputs

        # Pytorch Geometric batches along the node dimension, but we execute
        # along the temporal (step) dimension, hence we need to transpose
        # a few tensors. Done by `prepare_batch`.
        if self.assert_checks:
            check_edge_index_sorted(batch.edge_index)
        if self.epoch > self.debug_epoch_threshold:
            breakpoint()
        self.zero_steps()
        batch = type(self).prepare_batch(batch)
        # When we want to calculate last step metrics/accuracies
        # we need to take into account again different termination per graph
        # hence we save last step tensors (e.g. outputs) into their
        # corresponding tensor. The function below prepares these tensors
        # (all set to zeros, except masking for computation, which are ones)
        self.set_initial_states(batch, init_last_latent=init_last_latent)
        # Prepare masking tensors (each graph does at least 1 iteration of the algo)
        self.prepare_initial_masks(batch)
        # A flag if we had a wrong graph in the batch. Used for visualisation
        # of what went wrong
        self.wrong_flag = False
        assert self.mask_cp.all(), self.mask_cp
        if self.timeit:
            st = time.time()
        node_fts_inp, edge_fts_inp = self.encode_inputs(batch)
        if self.timeit:
            print(f'encoding inputs: {time.time()-st}')

        while True:
            hint_inp_curr, hint_out_curr = self.get_input_output_hints(batch)
            if not self.training:
                assert (self.last_continue_logits > 0).any() or True

            # Some algorithms output fewer values than they take
            # so if we reuse our last step outputs, they need to be fed back in.
            if self.timeit:
                st = time.time()
            hint_inp_curr = self.get_step_input(hint_inp_curr, batch)
            if self.timeit:
                print(f'getting step input : {time.time()-st}')
                st = time.time()
            node_fts_hint, edge_fts_hint, graph_fts = self.encode_hints(hint_inp_curr, batch)
            node_fts = node_fts_inp + node_fts_hint
            edge_fts = edge_fts_inp + edge_fts_hint
            if self.timeit:
                print(f'encoding hints: {time.time()-st}')

            true_termination = torch.where(self.step_idx+1 >= batch.lengths-1, -1e9, 1e9)

            # Does one iteration of the algo and accumulates statistics
            self.loop_body(batch,
                           node_fts,
                           edge_fts,
                           graph_fts,
                           hint_inp_curr,
                           hint_out_curr,
                           true_termination,
                           first_n_processors=first_n_processors)
            # And calculate what graphs would execute on the next step.
            self.mask_cp, self.mask, self.mask_edges = type(self).get_masks(self.training, batch, true_termination if self.training else self.last_continue_logits, enforced_mask)
            if not self.loop_condition(
                    self.mask_cp,
                    STEPS_SIZE):
                break
            assert self.mask_cp.any()
            self.step_idx += 1

        return self.all_hint_logits, self.last_logits, self.all_masks_graph

    def decode(self, batch, encoded_nodes, hidden, edge_fts, graph_fts):
        catted = torch.cat((encoded_nodes, hidden), dim=1)
        outs = defaultdict(dict)
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage == Stage.INPUT:
                continue

            if loc == Location.NODE:

                if data_type in [Type.MASK, Type.SCALAR, Type.CATEGORICAL, Type.MASK_ONE]:
                    outs[stage][name] = self.decoders[stage][name](catted)

                if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                    fr = self.decoders[stage][name][0](catted[batch.edge_index[0]])
                    to = self.decoders[stage][name][1](catted[batch.edge_index[1]])
                    edge = self.decoders[stage][name][2](edge_fts)
                    prod = self.decoders[stage][name][3](to.max(fr+edge)).squeeze(-1)
                    if data_type in [Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION] and self.use_sinkhorn:
                        prod = torch.maximum(prod, self.decoders[stage][name][3](fr.max(to+edge)).squeeze(-1))
                        prod = sinkhorn_normalize(batch, prod, temperature=0.1, steps=10 if self.training else 60, add_noise=self.training)
                    outs[stage][name] = prod

            if loc == Location.GRAPH:
                aggr_node_fts = torch_scatter.scatter_max(catted, batch.batch, dim=0)[0]
                if data_type in [Type.MASK, Type.SCALAR, Type.CATEGORICAL, Type.MASK_ONE]:
                    outs[stage][name] = self.decoders[stage][name][0](aggr_node_fts) + self.decoders[stage][name][1](graph_fts)
                else:
                    assert False

            if loc == Location.EDGE:
                fr = self.decoders[stage][name][0](catted[batch.edge_index[0]])
                to = self.decoders[stage][name][1](catted[batch.edge_index[1]])
                edge = self.decoders[stage][name][2](edge_fts)
                if data_type in (Type.CATEGORICAL, Type.MASK, Type.SCALAR):
                    outs[stage][name] = fr + to + edge
                elif data_type == Type.POINTER:
                    pred = fr + to + edge
                    pred_2 = self.decoders[stage][name][3](catted)
                    ebatch = batch.edge_index_batch
                    st = batch.ptr[ebatch]
                    en = batch.ptr[ebatch+1]
                    dense_pred_2, mask_pred_2 = tg_utils.to_dense_batch(pred_2, batch=batch.batch)
                    edge_pred_2 = dense_pred_2[ebatch]
                    mask_edge_pred_2 = mask_pred_2[ebatch]
                    probs_logits = self.decoders[stage][name][4](torch.maximum(pred[:, None, :], edge_pred_2)).squeeze(-1)
                    probs_logits[~mask_edge_pred_2] = -1e9
                    outs[stage][name] = probs_logits
                else:
                    assert False

        return outs

    def encode_nodes(self, current_input, last_latent):
        return torch.cat((current_input, last_latent), dim=1)

    def forward(self, batch, node_fts, edge_fts, graph_fts, first_n_processors=1000):
        if torch.isnan(node_fts).any():
            breakpoint()
        assert not torch.isnan(self.last_latent).any()
        assert not torch.isnan(node_fts).any()
        if self.timeit:
            st = time.time()
        if self.timeit:
            print(f'projecting nodes: {time.time()-st}')

        if self.timeit:
            st = time.time()
        edge_index = batch.edge_index
        hidden, edges_hidden = self.processor(node_fts, edge_fts, graph_fts, edge_index, self.last_latent, self.last_latent_edges, first_n_processors=first_n_processors, batch=batch)
        if self.timeit:
            print(f'message passing: {time.time()-st}')
        assert not torch.isnan(hidden).any()
        if self.timeit:
            st = time.time()
        if self.triplet_reasoning:
            edge_fts = self.triplet_reductor(torch.cat([edge_fts, edges_hidden], dim=-1))
        outs = self.decode(batch, node_fts, hidden, edge_fts, graph_fts)
        if self.timeit:
            print(f'decoding hints: {time.time()-st}')
        continue_logits = torch.where(self.step_idx+1 >= batch.lengths-1, -1e9, 1e9)
        return hidden, edges_hidden, outs, continue_logits


class LitAlgorithmReasoner(pl.LightningModule):
    def __init__(self,
                 hidden_dim,
                 algo_processor,
                 dataset_class,
                 dataset_root,
                 dataset_kwargs,
                 algorithm='mst_prim',
                 update_edges_hidden=False,
                 use_TF=False,
                 use_sinkhorn=True,
                 xavier_on_scalars=True,
                 learning_rate=get_hyperparameters()['lr'],
                 weight_decay=get_hyperparameters()['weight_decay'],
                 test_with_val=False,
                 test_with_val_every_n_epoch=20,
                 test_train_every_n_epoch=20,
                 **algorithm_base_kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.algorithm_base_kwargs = algorithm_base_kwargs
        self.dataset_class = dataset_class
        self.dataset_root = dataset_root
        self.dataset_kwargs = dataset_kwargs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.timeit = False
        self.update_edges_hidden = update_edges_hidden
        self.use_TF = use_TF
        self.use_sinkhorn = use_sinkhorn
        self.algorithm_base_kwargs = algorithm_base_kwargs
        self.algorithm = algorithm
        self.xavier_on_scalars = xavier_on_scalars
        self.test_with_val = test_with_val
        self.test_with_val_every_n_epoch = test_with_val_every_n_epoch
        self.test_train_every_n_epoch = test_train_every_n_epoch
        self._datasets = {}
        if self.test_with_val:
            self.val_dataloader = self.val_dataloader_alt
            self.validation_step = self.validation_step_alt
        self._current_epoch = 0
        self.load_dataset('train')

        self.algorithm_module = AlgorithmReasoner(self.dataset.spec,
                                                  self.dataset[0],
                                                  hidden_dim,
                                                  algo_processor,
                                                  update_edges_hidden=update_edges_hidden,
                                                  use_TF=use_TF,
                                                  use_sinkhorn=use_sinkhorn,
                                                  timeit=self.timeit,
                                                  xavier_on_scalars=xavier_on_scalars,
                                                  **algorithm_base_kwargs)
        self.save_hyperparameters(ignore=['algo_processor'])

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        return self.trainer.current_epoch if self._trainer else self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch) -> int:
        self._current_epoch = epoch

    def prepare_for_transfer(self):
        algo_processor = copy.deepcopy(self.algorithm_module.processor)
        self.algorithm_module = AlgorithmReasoner(self.hidden_dim,
                                                  self.node_features,
                                                  self.edge_features,
                                                  self.output_features,
                                                  algo_processor,
                                                  use_TF=False,
                                                  timeit=self.timeit,
                                                  **self.algorithm_base_kwargs)
        for p in self.algorithm_module.processor.parameters():
            p.requires_grad = False

    @staticmethod
    def pointer_loss(predecessor_pred, predecessor_gt_edge_1h,
                     softmax_idx, num_nodes):
        loss_unreduced = cross_entropy(predecessor_pred, softmax_idx, predecessor_gt_edge_1h, num_nodes)
        sum_loss = loss_unreduced.flatten().sum()
        cnt_loss = predecessor_gt_edge_1h.count_nonzero()
        return sum_loss / cnt_loss

    def single_prediction_loss(self, name, pred, pred_gt, batch, graph_mask,
                               node_mask, edge_mask):
        loss = None
        stage, loc, data_type = self.dataset.spec[name]
        if loc == Location.GRAPH:
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[graph_mask], pred_gt[graph_mask].argmax(-1))
            if data_type == Type.SCALAR:
                loss = F.mse_loss(
                        pred[graph_mask].squeeze(-1),
                        pred_gt[graph_mask])
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                        pred[graph_mask].squeeze(-1),
                        pred_gt[graph_mask])

        if loc == Location.NODE:
            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                pred_gt_one_hot = edge_one_hot_encode_pointers(pred_gt, batch.edge_index)
                loss = type(self).pointer_loss(
                    pred[edge_mask],
                    pred_gt_one_hot[edge_mask],
                    batch.edge_index[0][edge_mask], batch.num_nodes)
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                        pred[node_mask].squeeze(-1),
                        pred_gt[node_mask])
            if data_type == Type.MASK_ONE:
                lsms = torch_scatter.scatter_log_softmax(pred[node_mask], batch.batch[node_mask].unsqueeze(-1), dim=0)
                loss = (-lsms[(pred_gt[node_mask] == 1.)]).mean()
            if data_type == Type.SCALAR:
                loss = F.mse_loss(
                        pred[node_mask].squeeze(-1),
                        pred_gt[node_mask])
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[node_mask], pred_gt[node_mask].argmax(-1))
        if loc == Location.EDGE:
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                        pred[edge_mask].squeeze(-1),
                        pred_gt[edge_mask])
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[edge_mask], pred_gt[edge_mask].argmax(-1))
            if data_type == Type.SCALAR:
                loss = F.mse_loss(
                        pred[edge_mask].squeeze(-1),
                        pred_gt[edge_mask])
            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                pred_gt = pred_gt.int() - starts_edge
                loss = F.cross_entropy(
                    pred[edge_mask],
                    pred_gt[edge_mask])
        assert loss is not None, f'{stage}/{name}/{loc}/{data_type}'
        return loss

    def get_step_loss(self,
                      batch,
                      all_hint_logits,
                      output_logits,
                      all_masks_graph):

        if self.timeit:
            st = time.time()
        batch = self.algorithm_module.prepare_batch(batch)
        losses_dict = defaultdict(list)
        for i, (pred, graph_mask) in enumerate(zip(all_hint_logits, all_masks_graph)):
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            assert graph_mask.any()
            for name in pred:
                stage, loc, data_type = self.dataset.spec[name]
                pred_gt = getattr(batch, name)[i+1]
                losses_dict[name].append(
                    self.single_prediction_loss(name, pred[name], pred_gt,
                                                batch, graph_mask, node_mask,
                                                edge_mask))

        for name in output_logits:
            graph_mask = torch.ones(batch.num_graphs, dtype=torch.bool, device=self.device)
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            losses_dict[name].append(
                self.single_prediction_loss(name, output_logits[name],
                                            getattr(batch, name), batch,
                                            graph_mask, node_mask, edge_mask))

        for k, v in losses_dict.items():
            losses_dict[k] = torch.stack(v).mean()
        if self.timeit:
            print(f'loss calculation: {time.time()-st}')
            input()

        return losses_dict

    def single_prediction_acc(self, name, pred, pred_gt, batch, graph_mask,
                              node_mask, edge_mask):
        acc = None
        stage, loc, data_type = self.dataset.spec[name]
        if loc == Location.NODE:
            if data_type == Type.MASK_ONE:
                # try:
                acc = (pred[node_mask].squeeze(-1).nonzero() == pred_gt[node_mask].nonzero()).float().mean()
                # except Exception as e:
                #     breakpoint()
            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION, Type.MASK]:
                acc = (pred[node_mask].squeeze(-1) == pred_gt[node_mask]).float().mean()
            if data_type == Type.SCALAR:
                acc = ((pred[node_mask].squeeze(-1) - pred_gt[node_mask])**2).mean()
            if data_type == Type.CATEGORICAL:
                acc = (pred[node_mask].argmax(-1) == pred_gt[node_mask].argmax(-1)).float().mean()
            if data_type == Type.MASK:
                acc = multiclass_f1_score(pred[node_mask].squeeze(-1), pred_gt[node_mask])

        if loc == Location.GRAPH:
            if data_type == Type.CATEGORICAL:
                acc = (pred[graph_mask].argmax(-1) == pred_gt[graph_mask].argmax(-1)).float().mean()
            if data_type == Type.SCALAR:
                acc = ((pred[graph_mask].squeeze(-1) - pred_gt[graph_mask])**2).mean()
            if data_type == Type.MASK:
                acc = multiclass_f1_score(pred[graph_mask].squeeze(-1), pred_gt[graph_mask])

        if loc == Location.EDGE:
            if data_type == Type.CATEGORICAL:
                acc = (pred[edge_mask].argmax(-1) == pred_gt[edge_mask].argmax(-1)).float().mean()
            if data_type == Type.MASK:
                acc = multiclass_f1_score(pred[edge_mask].squeeze(-1), pred_gt[edge_mask])
            if data_type == Type.SCALAR:
                acc = ((pred[edge_mask].squeeze(-1) - pred_gt[edge_mask])**2).mean()
            if data_type in [Type.POINTER, Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]:
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                pred_gt = pred_gt.int() - starts_edge
                acc = (pred[edge_mask] == pred_gt[edge_mask]).float().mean()
        assert acc is not None, f"Please implement {name}"
        return acc

    def get_metrics(self,
                    batch,
                    all_hint_logits,
                    output_logits,
                    all_masks_graph):

        batch = self.algorithm_module.prepare_batch(batch)
        accs_dict = defaultdict(list)

        for i, (pred, graph_mask) in enumerate(zip(all_hint_logits, all_masks_graph)):
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            outputs = type(self.algorithm_module).convert_logits_to_outputs(
                self.dataset.spec, {'hint': pred},
                batch.edge_index[0],
                batch.edge_index[1],
                batch.num_nodes,
                batch.batch,
                include_probabilities=False)['hint']

            for name in outputs:
                acc = self.single_prediction_acc(
                        name,
                        outputs[name],
                        getattr(batch, name)[i+1],
                        batch,
                        graph_mask,
                        node_mask,
                        edge_mask)
                accs_dict[name].append(acc)

        outputs = type(self.algorithm_module).convert_logits_to_outputs(
            self.dataset.spec,
            output_logits,
            batch.edge_index[0],
            batch.edge_index[1],
            batch.num_nodes,
            batch.batch,
            include_probabilities=False)['output']
        for name in outputs:
            graph_mask = torch.ones(batch.num_graphs, dtype=torch.bool, device=self.device)
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            accs_dict[name].append(
                self.single_prediction_acc(
                    name,
                    outputs[name],
                    getattr(batch, name),
                    batch,
                    graph_mask,
                    node_mask,
                    edge_mask))

        for k, v in accs_dict.items():
            accs_dict[k] = torch.stack(v).mean()

        return accs_dict

    def fwd_step(self, batch, batch_idx):
        if self.timeit:
            st = time.time()
        self.algorithm_module.epoch = self.current_epoch
        all_hint_logits, output_logits, masks = self.algorithm_module.process(batch)
        if self.timeit:
            print(f'forward step: {time.time()-st}')
            input()
        return all_hint_logits, output_logits, masks

    def training_step(self, batch, batch_idx):
        all_hint_logits, output_logits, masks = self.fwd_step(batch, batch_idx)
        losses_dict = self.get_step_loss(batch, all_hint_logits, output_logits['output'], masks)
        self.log_dict(dict((f'train/loss/{k}', v) for k, v in losses_dict.items()), batch_size=batch.num_graphs)
        total_loss = sum(losses_dict.values()) / len(losses_dict)
        self.log('train/loss/average_loss', total_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch.num_graphs)
        accs_dict = {}
        if self.current_epoch % self.test_train_every_n_epoch == 0:
            accs_dict = self.get_metrics(batch, all_hint_logits, output_logits, masks)
        self.log_dict(dict((f'train/acc/{k}', v) for k, v in accs_dict.items()), batch_size=batch.num_graphs, add_dataloader_idx=False)
        # if sum(losses_dict.values()) > 1e5:
        #     breakpoint()
        return {'loss': total_loss, 'losses_dict': losses_dict, 'accuracies': accs_dict}

    def valtest_step(self, batch, batch_idx, mode):
        all_hint_logits, output_logits, masks = self.fwd_step(batch, batch_idx)
        losses_dict = self.get_step_loss(batch, all_hint_logits, output_logits['output'], masks)
        self.log_dict(dict((f'{mode}/loss/{k}', v) for k, v in losses_dict.items()), batch_size=batch.num_graphs, add_dataloader_idx=False)
        if torch.isnan(sum(losses_dict.values())).any():
            breakpoint()
        self.log(f'{mode}/loss/average_loss', sum(losses_dict.values()) / len(losses_dict), batch_size=batch.num_graphs, add_dataloader_idx=False)
        accs_dict = self.get_metrics(batch, all_hint_logits, output_logits, masks)
        self.log_dict(dict((f'{mode}/acc/{k}', v) for k, v in accs_dict.items()), batch_size=batch.num_graphs, add_dataloader_idx=False)
        return {'losses': losses_dict, 'accuracies': accs_dict}

    def validation_step_alt(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1 and not self.trainer.state.stage == 'sanity_check' and self.current_epoch % self.test_with_val_every_n_epoch == 0:
            return self.valtest_step(batch, batch_idx, 'periodic_test')
        if dataloader_idx == 0:
            return self.valtest_step(batch, batch_idx, 'val')

    def validation_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.fwd_step(batch, batch_idx)

    def load_dataset(self, split, suffix=''):
        split = split+suffix
        nn = CONFIGS[self.algorithm][split]['num_nodes']
        self.dataset_kwargs['split'] = split
        if (split, nn) not in self._datasets:
            self._datasets[(split, nn)] = self.dataset_class(
                self.dataset_root,
                nn,
                CONFIGS[self.algorithm][split]['num_samples'],
                algorithm=self.algorithm,
                **self.dataset_kwargs)
        self.dataset = self._datasets[(split, nn)]
        print(f'Loading {self.dataset=} (num nodes: {nn}) with kwargs')
        pprint(self.dataset_kwargs)
        print()

    def get_a_loader(self, split, suffix=''):
        self.load_dataset(split, suffix='')
        self.algorithm_module.dataset_spec = self.dataset.spec
        dl = DataLoader(self.dataset,
                        batch_size=get_hyperparameters()['batch_size'],
                        shuffle=True if split == 'train' else False,
                        drop_last=False,
                        follow_batch=['edge_index'],
                        num_workers=1,
                        persistent_workers=True)
        return dl

    def train_dataloader(self):
        return self.get_a_loader('train')

    def val_dataloader_alt(self):
        return [self.get_a_loader('val'), self.get_a_loader('test')]

    def val_dataloader(self):
        return self.get_a_loader('val')

    def test_dataloader(self, suffix=''):
        return self.get_a_loader('test'+suffix)

    def configure_optimizers(self):
        lr = self.learning_rate
        wd = self.weight_decay
        optimizer = optim.Adam(self.parameters(),
                               weight_decay=wd,
                               lr=lr)
        return optimizer

if __name__ == '__main__':
    hidden_dim = get_hyperparameters()['dim_latent']
    processor = MPNN(hidden_dim,
                     hidden_dim,
                     hidden_dim,
                     bias=True,
                     use_GRU=True)
    lit_reasoner = LitAlgorithmReasoner(get_hyperparameters()['dim_latent'], 1,
                                        1, 1, processor,
                                        _DATASET_CLASSES['mst_prim'],
                                        _DATASET_ROOTS['mst_prim'],
                                        {'num_nodes': 20},
                                        use_TF=False)
    train_dl = lit_reasoner.train_dataloader()
    train_dl_bs1 = DataLoader(lit_reasoner.algorithm_module.dataset, shuffle=False, follow_batch=['edge_index'])
    ba1 = list(iter(train_dl))[1]
    ba11 = list(iter(train_dl_bs1))[2]
    ba12 = list(iter(train_dl_bs1))[3]
    loss1 = lit_reasoner.training_step(ba1, 0)
    loss11 = lit_reasoner.training_step(ba11, 0)
    loss12 = lit_reasoner.training_step(ba12, 0)
    adict1 = lit_reasoner.valtest_step(ba1, 0)['mean_step_accuracy']
    adict11 = lit_reasoner.valtest_step(ba11, 0)['mean_step_accuracy']
    adict12 = lit_reasoner.valtest_step(ba12, 0)['mean_step_accuracy']
    from utils_execution import ReasonerZeroerCallback
    import math
    OVERFIT_BATCHES=1
    lit_reasoner.get_a_loader('train')
    dataset = list(lit_reasoner.dataset)[:OVERFIT_BATCHES*get_hyperparameters()['batch_size']]
    dl = DataLoader(dataset, get_hyperparameters()['batch_size'], shuffle=False)
    trainer = pl.Trainer(
        overfit_batches=OVERFIT_BATCHES,
        accelerator='cuda',
        max_epochs=int(1e9),
        callbacks=[ReasonerZeroerCallback()],
        check_val_every_n_epoch=10,
        log_every_n_steps=1000,
    )
    trainer.fit(
        model=lit_reasoner,
        train_dataloaders=dl,
        val_dataloaders=dl,
    )
