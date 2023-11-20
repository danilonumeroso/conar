from collections import defaultdict
from pprint import pprint

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import torch_geometric.utils as tg_utils
import torch_scatter

from pytorch_lightning.trainer.supporters import CombinedLoader

import networkx as nx

from models.algorithm_reasoner import AlgorithmReasoner, LitAlgorithmReasoner
from utils_execution import get_number_of_nodes
from hyperparameters import get_hyperparameters
from datasets._configs import CONFIGS

class LitVKCReasoner(LitAlgorithmReasoner):

    def __init__(self,
                 hidden_dim,
                 algo_processor,
                 dataset_class,
                 dataset_root,
                 dataset_kwargs,
                 bias=True,
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
                         transferring=transferring,
                         learning_rate=learning_rate,
                         **algo_reasoner_kwargs)

        self.algorithm_module = AlgorithmReasoner(
            self.dataset.spec,
            self.dataset[0],
            hidden_dim,
            algo_processor,
            bias=bias,
            transferring=transferring,
            timeit=self.timeit,
            double_process=double_process,
            **algo_reasoner_kwargs)
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

    def get_VKC_metrics(self, batch, output_logits):
        selected_dense = torch_geometric.utils.to_dense_batch(output_logits['output']['selected'], batch=batch.batch)[0]
        selected_dense_topk = torch.sort(torch.topk(selected_dense.squeeze(-1), self.dataset.k, dim=-1).indices).values
        selected_topk = (selected_dense_topk+batch.ptr[:-1].unsqueeze(-1)).view(-1)
        selected_topk_gt = batch.selected.nonzero().squeeze(-1)
        selected_batch = batch.batch[selected_topk]

        acc_selected_topk = torch_scatter.scatter_mean((selected_topk == selected_topk_gt).float(), selected_batch).mean()
        G = tg_utils.to_networkx(batch, to_undirected=True, edge_attrs=['edge_attr'])
        mspl = nx.multi_source_dijkstra_path_length(G, sources=selected_topk.tolist(), weight='edge_attr')
        mspl = torch.tensor([mspl[i] for i in range(batch.num_nodes)]).to(selected_dense)
        farthest = torch_scatter.scatter_max(mspl, batch.batch)[0]
        assert (farthest + torch.finfo(torch.float32).eps >= batch.farthest).all()
        return {
            'acc_topk': acc_selected_topk,
            'farthest': farthest.mean(),
            'farthest_gt': batch.farthest.mean(),
            'farthest_relative_error': ((farthest-batch.farthest)/batch.farthest).mean(),
        }

    def get_metrics(self,
                    batch,
                    all_hint_logits,
                    output_logits,
                    all_masks_graph):

        accs_dict = super().get_metrics(batch, all_hint_logits, output_logits,
                                        all_masks_graph)
        accs_dict.update(**self.get_VKC_metrics(batch,
                                                output_logits))
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
