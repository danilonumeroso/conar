from collections import defaultdict
import copy
import itertools
import time
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
import torch_geometric
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from baselines.beam_search import vmapped_beam_search_rollout, BEAM_WIDTH
from models.algorithm_reasoner import AlgorithmReasoner, LitAlgorithmReasoner
from hyperparameters import get_hyperparameters

from torch_geometric.utils import k_hop_subgraph
from datasets._configs import CONFIGS
from utils_execution import cross_entropy, check_edge_index_sorted, prepare_constants, edge_one_hot_encode_pointers, get_number_of_nodes
from clrs import Type, Location, Stage

class TSPReasoner(AlgorithmReasoner):
    def __init__(self,
                 spec,
                 data,
                 latent_features,
                 algo_processor,
                 bias=True,
                 use_TF=False,
                 L1_loss=False,
                 global_termination_pool='max', #'predinet',
                 get_attention=False,
                 use_batch_norm=False,
                 transferring=False,
                 timeit=True,
                 double_process=False,
                 **algo_reasoner_kwargs):

        super().__init__(
            spec,
            data,
            latent_features,
            algo_processor,
            use_TF=use_TF,
            timeit=timeit,
            L1_loss=L1_loss,
            global_termination_pool=global_termination_pool,
            get_attention=get_attention,
            use_batch_norm=use_batch_norm,
            transferring=transferring,
            **algo_reasoner_kwargs,
        )
        self.step_idx = 0
        self.assert_checks = False
        self.debug = False
        self.debug_epoch_threshold = 1e9
        self.next_step_pool = True
        self.double_process = double_process
        self.lambda_mul = 1# 0.0001
        self.transferring = transferring

    def get_input_output_hints(self, batch):
        hint_inp_curr = dict()
        hint_out_curr = dict()
        return hint_inp_curr, hint_out_curr

    def process(
            self,
            *args,
            **kwargs):

        self.all_hint_logits, self.last_logits, self.all_masks_graph = super().process(
            *args,
            first_n_processors=1000 if not self.double_process else 1,
            **kwargs)

        if self.double_process:
            self.all_hint_logits, self.last_logits, self.all_masks_graph = super().process(
                *args,
                init_last_latent=self.last_latent,
                **kwargs)

        return self.all_hint_logits, self.last_logits, self.all_masks_graph

class LitTSPReasoner(LitAlgorithmReasoner):

    def __init__(self,
                 hidden_dim,
                 algo_processor,
                 dataset_class,
                 dataset_root,
                 dataset_kwargs,
                 bias=True,
                 use_TF=False,
                 ensure_permutation='greedy',
                 transferring=False,
                 learning_rate=get_hyperparameters()['lr'],
                 double_process=False,
                 **algo_reasoner_kwargs):
        super().__init__(hidden_dim,
                         algo_processor,
                         dataset_class,
                         dataset_root,
                         dataset_kwargs,
                         bias=bias,
                         use_TF=use_TF,
                         transferring=transferring,
                         learning_rate=learning_rate,
                         **algo_reasoner_kwargs)

        self.algorithm_module = TSPReasoner(self.dataset.spec,
                                            self.dataset[0],
                                            hidden_dim,
                                            algo_processor,
                                            bias=bias,
                                            use_TF=use_TF,
                                            transferring=transferring,
                                            timeit=self.timeit,
                                            double_process=double_process,
                                            **algo_reasoner_kwargs)
        self.ensure_permutation = ensure_permutation
        self.double_process = double_process
        self.save_hyperparameters(ignore=['algo_processor'])

    def training_step(self, batch, batch_idx):
        ret = {'loss': 0, 'losses_dict': defaultdict(list), 'accuracies': defaultdict(list)}
        for bb in batch:
            ans = super().training_step(bb, batch_idx)
            ret['loss'] += ans['loss']
            for name in ['losses_dict', 'accuracies']:
                for k, v in ans[name].items():
                    ret[name][k].append(v)
        ret['loss'] /= len(batch)
        for name in ['losses_dict', 'accuracies']:
            for k, v in ans[name].items():
                ret[name][k] = torch.tensor(v).mean()
        return ret

    def get_tour_metrics(self, output_logits, batch):

        def get_mask(edges):
            mask = torch.zeros_like(batch.edge_index[0])
            j = 0
            for i in range(batch.edge_index.shape[1]):
                u1, v1 = batch.edge_index[:, i]
                u2, v2 = edges[:, j]
                if u1 == u2 and v1 == v2:
                    mask[i] = 1
                    j += 1

                if j == edges.shape[1]:
                    break
            assert j == edges.shape[1]
            return mask

        def get_mask_v2(edges):
            dense_edges = torch_geometric.utils.to_dense_adj(edges, batch=batch.batch).bool()
            dense_edges_batch = torch_geometric.utils.to_dense_adj(batch.edge_index, batch=batch.batch).bool()
            edge_index, mask = torch_geometric.utils.dense_to_sparse(((dense_edges & dense_edges_batch).float()+1))
            mask = mask - 1
            return mask

        acc = None

        # st = time.time()
        outputs = type(self.algorithm_module).convert_logits_to_outputs(
            self.dataset.spec,
            output_logits,
            batch.edge_index[0],
            batch.edge_index[1],
            batch.num_nodes,
            batch.batch,
            include_probabilities=False)['output']
        for name in outputs:
            pred = outputs[name]
            pred_gt = getattr(batch, name)
            stage, loc, data_type = self.dataset.spec[name]
            if loc == Location.NODE:
                if name == 'predecessor_index':
                    tours = torch.stack([torch.arange(pred.shape[0]).to(pred), pred])
                    mask = get_mask_v2(tours).bool()

                    st = time.time()
                    mattr = batch.edge_attr[mask]
                    mbatch = batch.edge_index_batch[mask]
                    msrc, mdst = batch.edge_index[:, mask]
                    tour_len = torch_scatter.scatter_sum(mattr, mbatch)
                    tour_correctness = torch_scatter.scatter_sum((msrc == mdst.sort().values), mbatch)

        assert sum(tour_correctness)/len(tour_correctness) == 1
        return dict(tour_len=tour_len.mean(),
                    tour_len_gt=batch.optimal_value.mean().item(),
                    tour_correctness=sum(tour_correctness)/len(tour_correctness),
                    tour_relative_error=((tour_len-batch.optimal_value)/batch.optimal_value).mean())

    def process_TSP_tour_greedy(self, batch, output_logits):
        mask_active_nodes = torch.tensor(batch.start_route).bool()
        mask_edges_to_nodes_in_tour = torch.zeros_like(batch.edge_index[0]).bool()
        max_nodes_per_graph = batch.batch.unique(return_counts=True)[1].max()
        num_nodes_per_graph = batch.num_nodes // batch.num_graphs
        for _ in range(max_nodes_per_graph - 1):
            mask_active_edges = mask_active_nodes[batch.edge_index[0]] & ~mask_edges_to_nodes_in_tour # Any edge outwards of active nodes and not pointing to previously used node
            mask_edges_to_nodes_in_tour |= mask_active_nodes[batch.edge_index[1]] # any edge towards the active nodes should not be used in future iterations
            sloops = (batch.edge_index[0] == batch.edge_index[1])
            preds = output_logits['output']['predecessor_index'].clone()
            preds = preds.masked_fill(~mask_active_edges | sloops, -1e6)

            # nudge the max value to ensure there is a unique maximum
            max_idxs = preds.reshape(-1, num_nodes_per_graph).argmax(-1)
            max_idxs = F.one_hot(max_idxs, num_nodes_per_graph)
            preds[max_idxs.bool().flatten()] = (preds.reshape(-1, num_nodes_per_graph)[max_idxs.bool()] + 1e-4).flatten()
            output_logits['output']['predecessor_index'][mask_active_nodes[batch.edge_index[0]]] = preds[mask_active_nodes[batch.edge_index[0]]]
            new_active_nodes = preds.reshape(-1, num_nodes_per_graph).argmax(-1)[mask_active_nodes.bool()].unsqueeze(-1) # NOTE the reshape/flatten mechanic may not work if graphs in the same batch are of different sizes (consider using torch_scatter.scatter_max)
            mask_active_nodes = F.one_hot(new_active_nodes, num_nodes_per_graph).flatten().bool()
        final_pred_mask = mask_active_nodes[batch.edge_index[0]] & batch.start_route.bool()[batch.edge_index[1]]
        output_logits['output']['predecessor_index'] = output_logits['output']['predecessor_index'].masked_fill(final_pred_mask, 1e8)
        return output_logits

    def process_TSP_tour_BS(self, batch, output_logits):
        start_route = torch_geometric.utils.to_dense_batch(batch.start_route, batch=batch.batch)[0]
        dens_logits = torch_geometric.utils.to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=output_logits['output']['predecessor_index'])
        num_nodes = start_route.shape[1]
        # st = time.time()
        tours = torch.tensor(np.array(vmapped_beam_search_rollout(
            start_route.cpu().detach().numpy(),
            -dens_logits.cpu().detach().numpy(),
            num_nodes, BEAM_WIDTH)), device=start_route.device)
        # print('tours took', time.time()-st)
        # st = time.time()
        dens_logits_o = torch.full_like(dens_logits, -1e9)
        arranged = torch.arange(dens_logits_o.shape[0], device=dens_logits.device)
        fr = tours[arranged, 0]
        to = tours[arranged, 1]
        batch_id = arranged.unsqueeze(1).expand_as(fr)
        fr = fr.flatten()
        to = to.flatten()
        batch_id = batch_id.flatten()
        dens_logits_o[batch_id, fr, to] = 1e9
        edge_index, sparse_logits = torch_geometric.utils.dense_to_sparse(dens_logits_o)
        sparse_logits = sparse_logits.to(batch.edge_index.device)
        assert (edge_index == batch.edge_index).all()
        output_logits['output']['predecessor_index'] = sparse_logits
        # print('rest took', time.time()-st)
        return output_logits

    def process_TSP_tour(self, batch, output_logits):
        if self.ensure_permutation == "greedy":
            return self.process_TSP_tour_greedy(batch, output_logits)
        return self.process_TSP_tour_BS(batch, output_logits)

    def get_metrics(self,
                    batch,
                    all_hint_logits,
                    output_logits,
                    all_masks_graph):
        output_logits = self.process_TSP_tour(batch, output_logits)
        accs_dict = super().get_metrics(batch, all_hint_logits, output_logits,
                                        all_masks_graph)
        accs_dict.update(**self.get_tour_metrics(output_logits,
                                                 batch))
        return accs_dict

    def load_dataset(self, split, suffix=''):
        split = split+suffix
        nns = get_number_of_nodes(self.algorithm, split)
        for nn in nns:
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
        self.load_dataset(split, suffix=suffix)
        self.algorithm_module.dataset_spec = self.dataset.spec
        dl = DataLoader(self.dataset,
                        batch_size=get_hyperparameters()['batch_size'],
                        shuffle=True if split == 'train' else False,
                        drop_last=False,
                        follow_batch=['edge_index'],
                        num_workers=1,
                        persistent_workers=True)
        if split == 'train':
            nns = get_number_of_nodes(self.algorithm, split)
            dls = []
            for nn in nns:
                dl = DataLoader(self._datasets[(split, nn)],
                                batch_size=get_hyperparameters()['batch_size'],
                                shuffle=True if split == 'train' else False,
                                drop_last=False,
                                follow_batch=['edge_index'],
                                num_workers=1,
                                persistent_workers=True)
                dls.append(dl)
            dl = CombinedLoader(dls)
        return dl


if __name__ == '__main__':
    tsp = TSPReasoner(64, 1, 1, 1, None)
    ltsp = LitTSPReasoner(64, 1, 1, 1, None, None, None)
