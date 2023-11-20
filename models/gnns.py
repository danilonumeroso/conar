import torch
import torch.nn as nn
import pytorch_lightning as pl
from layers.gnns import MPNN, GATv2, TripletMPNN, GPS

_PROCESSSOR_DICT = {
    'MPNN': MPNN,
    'GATv2': GATv2,
    'TriMPNN': TripletMPNN,
    'GPS': GPS,
}

class LitProcessorSet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, edge_dim, *args, processors=['MPNN'], reduce_with_MLP=False, update_edges_hidden=False, **kwargs):
        super().__init__()
        self.processors = nn.ModuleList([])
        for proc in processors:
            self.processors.append(LitGNN(in_channels, out_channels, edge_dim, *args, processor_type=proc, update_edges_hidden=update_edges_hidden, **kwargs))
        self.reduce_with_MLP = reduce_with_MLP
        self.update_edges_hidden = update_edges_hidden
        if reduce_with_MLP:
            
            self.reductor = nn.Sequential(
                nn.Linear(out_channels*len(self.processors), out_channels),
                nn.LayerNorm(out_channels),
                nn.LeakyReLU(),
                nn.Linear(out_channels, out_channels),
            )

            if update_edges_hidden:
                self.reductor_e = nn.Sequential(
                    nn.Linear(edge_dim*len(self.processors), out_channels),
                    nn.LayerNorm(out_channels),
                    nn.LeakyReLU(),
                    nn.Linear(out_channels, out_channels),
                )

    def zero_lstm(self, num_nodes):
        for proc in self.processors:
            proc.zero_lstm(num_nodes)

    def forward(self, *args, first_n_processors=1000, **kwargs):
        proco = [proc(*args, **kwargs) for proc in self.processors[:first_n_processors]]
        if self.reduce_with_MLP:
            re = self.reductor(torch.cat([proco[i][0] for i in range(len(self.processors))[:first_n_processors]], dim=-1))
            if self.update_edges_hidden:
                re_e = self.reductor_e(torch.cat([proco[i][1] for i in range(len(self.processors))[:first_n_processors]], dim=-1))
            else:
                re_e = sum([proco[i][1] for i in range(len(self.processors))[:first_n_processors]])/len(self.processors[:first_n_processors])
        else:
            re = sum([proco[i][0] for i in range(len(self.processors))[:first_n_processors]])/len(self.processors[:first_n_processors])
            re_e = sum([proco[i][1] for i in range(len(self.processors))[:first_n_processors]])/len(self.processors[:first_n_processors])

        return re, re_e

class LitGNN(pl.LightningModule):
    def __init__(self, in_channels, out_channels, *args, processor_type='MPNN', use_LSTM=False, **kwargs):
        super().__init__()
        self.processor = _PROCESSSOR_DICT[processor_type](in_channels, out_channels, *args, **kwargs)
        self.use_LSTM = use_LSTM
        self.out_channels = out_channels
        if use_LSTM:
            self.LSTMCell = nn.LSTMCell(out_channels, out_channels)

    def zero_lstm(self, num_nodes):
        if self.use_LSTM:
            self.lstm_state = (torch.zeros((num_nodes, self.out_channels), device=self.device),
                               torch.zeros((num_nodes, self.out_channels), device=self.device))

    def forward(self, *args, **kwargs):
        hidden = self.processor.forward(*args, **kwargs)
        if self.use_LSTM:
            self.lstm_state = self.LSTMCell(hidden, self.lstm_state)
            hidden = self.lstm_state[0]
        return hidden
