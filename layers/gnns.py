import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import to_dense_batch, to_dense_adj

class GATv2(nn.Module):

    def __init__(self, in_channels, out_channels, edge_dim, aggr='max', bias=False, flow='source_to_target', **unused_kwargs):
        super().__init__()
        self.gat = nng.GATv2Conv(in_channels,
                                 out_channels,
                                 edge_dim=edge_dim,
                                 aggr=aggr,
                                 bias=bias,
                                 flow=flow,
                                 add_self_loops=False)

    def forward(self, x, edge_attr, graph_fts, edge_index, hidden, edges_hidden, batch=None, **kwargs):
        x = x + graph_fts[batch.batch]
        edge_attr = edge_attr + graph_fts[edge_index[0]]
        z = torch.cat((x, hidden), dim=-1)
        gat_hidden = self.gat(z, edge_index, edge_attr=edge_attr)
        if not self.training:
            gat_hidden = torch.clamp(gat_hidden, -1e9, 1e9)
        return gat_hidden+hidden, edges_hidden

class MPNN(nng.MessagePassing):

    def __init__(self, in_channels, out_channels, edge_dim, aggr='max', bias=False, flow='source_to_target', use_gate=False, biased_gate=True, update_edges_hidden=False, num_layers=3, **unused_kwargs):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        if update_edges_hidden:
            modules = [nn.Linear(2*in_channels + 3*edge_dim, out_channels, bias=bias)]
        else:
            modules = [nn.Linear(2*in_channels + 2*edge_dim, out_channels, bias=bias)]
        for _ in range(num_layers):
            modules.extend([nn.LeakyReLU(),
                            nn.Linear(out_channels, out_channels, bias=bias)])
        self.M = nn.Sequential(*modules)
        self.update_edges_hidden = update_edges_hidden
        if self.update_edges_hidden:
            modules = [nn.Linear(2*in_channels + 2*edge_dim, out_channels, bias=bias)]
            for _ in range(num_layers):
                modules.extend([nn.LeakyReLU(),
                                nn.Linear(out_channels, out_channels, bias=bias)])
            self.M_e = nn.Sequential(*modules)
        self.use_gate = use_gate
        self.biased_gate = biased_gate
        self.U1 = nn.Linear(2*out_channels, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)
        if use_gate:
            self.gate1 = nn.Linear(2*out_channels, out_channels, bias=bias)
            self.gate2 = nn.Linear(out_channels, out_channels, bias=bias)
            self.gate3 = nn.Linear(out_channels, out_channels, bias=bias)
            if self.biased_gate:
                assert bias, "Bias has to be enabled"
                torch.nn.init.constant_(self.gate3.bias, -3)
            if self.update_edges_hidden:
                self.gate1_e = nn.Linear(out_channels, out_channels, bias=bias)
                self.gate2_e = nn.Linear(out_channels, out_channels, bias=bias)
                self.gate3_e = nn.Linear(out_channels, out_channels, bias=bias)
                if self.biased_gate:
                    assert bias, "Bias has to be enabled"
                    torch.nn.init.constant_(self.gate3_e.bias, -3)

        self.out_channels = out_channels

    def gps_forward(self, doesntmatter1, doesntmatter2, ei=None, btch=None, **kwargs):
        return self.forward_old(edge_index=ei, batch=btch, **kwargs)[0]

    def forward(self, node_fts, edge_attr, graph_fts, edge_index, hidden, edges_hidden, batch=None):
        z = torch.cat((node_fts, hidden), dim=-1)
        hidden = self.propagate(edge_index, x=z, hidden=hidden, edges_hidden=edges_hidden, edge_attr=edge_attr, graph_fts=graph_fts[batch.batch])
        if self.update_edges_hidden:
            edges_hidden = self.edge_updater(edge_index, x=z, hidden=hidden, edges_hidden=edges_hidden, edge_attr=edge_attr)
        if not self.training:
            hidden = torch.clamp(hidden, -1e9, 1e9)
        return hidden, edges_hidden

    def message(self, x_i, x_j, edge_attr, graph_fts_i, edges_hidden):
        if self.update_edges_hidden:
            return self.M(torch.cat((x_i, x_j, edge_attr, graph_fts_i, edges_hidden), dim=1))
        return self.M(torch.cat((x_i, x_j, edge_attr, graph_fts_i), dim=1))

    def edge_update(self, x_i, x_j, edge_attr, edges_hidden):
        m_e = self.M_e(torch.cat((x_i, x_j, edge_attr, edges_hidden), dim=1))
        gate = F.sigmoid(self.gate3_e(F.relu(self.gate1_e(edges_hidden) + self.gate2_e(m_e))))
        return m_e * gate + edges_hidden * (1-gate)

    def update(self, aggr_out, x, hidden):
        h_1 = self.U1(x)
        h_2 = self.U2(aggr_out)
        ret = h_1 + h_2
        if self.use_gate:
            gate = F.sigmoid(self.gate3(F.relu(self.gate1(x) + self.gate2(aggr_out))))
            ret = ret * gate + hidden * (1-gate)
        return ret


class TripletMPNN(nn.Module):

    def __init__(self, in_channels, out_channels, edge_dim, aggr='max', bias=False, flow='source_to_target', use_gate=False, biased_gate=True, update_edges_hidden=False, num_layers=2, use_ln=False):
        super(TripletMPNN, self).__init__()
        assert aggr == 'max', 'Max only mode, soz!'
        self.update_edges_hidden = update_edges_hidden
        self.use_ln = use_ln
        lst = []
        for in_dim in [in_channels, in_channels, in_channels, edge_dim, edge_dim, edge_dim, in_channels//2]:
            modules = [nn.Linear(in_dim, 8, bias=bias)]
            lst.append(nn.Sequential(*modules))
        self.M_tri = nn.ModuleList(lst)
        lst = []
        for in_dim in [in_channels, in_channels, edge_dim, edge_dim]:
            modules = [nn.Linear(in_dim, out_channels, bias=bias)]
            lst.append(nn.Sequential(*modules))

        modules = []
        for _ in range(num_layers):
            modules.extend([nn.ReLU(),
                            nn.Linear(out_channels, out_channels, bias=bias)])
        lst.append(nn.Sequential(*modules))
        self.M = nn.ModuleList(lst)
        self.use_gate = use_gate
        self.biased_gate = biased_gate
        self.U1 = nn.Linear(2*out_channels, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)
        self.U3 = nn.Linear(8, out_channels, bias=bias)
        if use_gate:
            self.gate1 = nn.Linear(2*out_channels, out_channels, bias=bias)
            self.gate2 = nn.Linear(out_channels, out_channels, bias=bias)
            self.gate3 = nn.Linear(out_channels, out_channels, bias=bias)
            if self.biased_gate:
                assert bias, "Bias has to be enabled"
                torch.nn.init.constant_(self.gate3.bias, -3)

        self.out_channels = out_channels
        self.trifd = torch.compile(self.triplet_forward_dense, disable=True)

    def triplet_forward_dense(self, z_dense, e_dense, graph_fts, mask, tri_msgs_mask, msgs_mask):
        tri_1 = self.M_tri[0](z_dense)
        tri_2 = self.M_tri[1](z_dense)
        tri_3 = self.M_tri[2](z_dense)
        tri_e_1 = self.M_tri[3](e_dense)
        tri_e_2 = self.M_tri[4](e_dense)
        tri_e_3 = self.M_tri[5](e_dense)
        tri_g = self.M_tri[6](graph_fts)
        tri_1[~mask] = 0
        tri_2[~mask] = 0
        tri_3[~mask] = 0

        tri_msgs = (
            tri_1[:, :, None, None, :] +  #   (B, N, 1, 1, H)
            tri_2[:, None, :, None, :] +  # + (B, 1, N, 1, H)
            tri_3[:, None, None, :, :] +  # + (B, 1, 1, N, H)
            tri_e_1[:, :, :, None, :]  +  # + (B, N, N, 1, H)
            tri_e_2[:, :, None, :, :]  +  # + (B, N, 1, N, H)
            tri_e_3[:, None, :, :, :]  +  # + (B, 1, N, N, H)
            tri_g[:, None, None, None, :] # + (B, 1, 1, 1, H)
        )                                 # = (B, N, N, N, H)
        msk_tri = mask[:, None, None, :] | mask[:, None, :, None] | mask[:, :, None, None]
        tri_msgs[~msk_tri] = -1e9
        tri_msgs = self.U3(tri_msgs.max(1).values) # B x N x N x H

        msg_1 = self.M[0](z_dense) # B x N x H
        msg_2 = self.M[1](z_dense) # B x N x H
        msg_e = self.M[2](e_dense) # B x N x N x H
        msg_g = self.M[3](graph_fts) # B x H
        msg_1[~mask] = 0
        msg_2[~mask] = 0
        msg_e[~msgs_mask] = 0
        msgs = (msg_1[:, None, :, :] + msg_2[:, :, None, :] + msg_e + msg_g[:, None, None, :]) # B x N x N x H
        msgs = self.M[-1](msgs)
        msgs[~msgs_mask] = -1e9
        msgs = msgs.max(1).values
        h_1 = self.U1(z_dense)
        h_2 = self.U2(msgs)
        ret = h_1 + h_2
        return ret, msgs, tri_msgs

    def forward(self, node_fts, edge_attr, graph_fts, edge_index, hidden, edges_hidden, batch=None):
        z = torch.cat((node_fts, hidden), dim=-1)
        hidden_dense, _ = to_dense_batch(hidden, batch=batch.batch) # BxNxH
        z_dense, mask = to_dense_batch(z, batch=batch.batch) # BxNxH
        e_dense = to_dense_adj(edge_index, batch=batch.batch, edge_attr=edge_attr) # BxNxNxH
        adj_mat = (e_dense != 0.).all(-1)
        fn = self.trifd if self.training else self.triplet_forward_dense
        ret, msgs, tri_msgs = fn(z_dense, e_dense, graph_fts, mask, mask[:, :, None] | mask[:, None, :], adj_mat)
        if self.use_gate:
            gate = F.sigmoid(self.gate3(F.relu(self.gate1(z_dense) + self.gate2(msgs))))
            ret = ret * gate + hidden_dense * (1-gate)
        ebatch = batch.edge_index_batch
        e1 = batch.edge_index[0]-batch.ptr[ebatch]
        e2 = batch.edge_index[1]-batch.ptr[ebatch]
        ret = ret[mask]
        assert (ret != -1e9).all(), breakpoint()
        if self.use_ln:
            ret = F.layer_norm(ret, ret.shape[1:])
        return ret, tri_msgs[ebatch, e1, e1]

class GPS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, update_edges_hidden=False, **kwargs):
        super().__init__()

        assert update_edges_hidden == False, "We don't support this here"
        pe_dim = 0
        self.node_emb = nn.Linear(in_channels, out_channels)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.edge_emb = nn.Linear(edge_dim, out_channels)
        self.attn_type = 'multihead'
        self.convs = nn.ModuleList()
        mpnn = MPNN(in_channels, out_channels, edge_dim, update_edges_hidden=update_edges_hidden, **kwargs)
        mpnn.forward_old = mpnn.forward
        mpnn.forward = mpnn.gps_forward
        conv = nng.GPSConv(out_channels, mpnn, heads=4,
                           attn_type=self.attn_type, attn_kwargs={'dropout': 0.5})
        self.convs.append(conv)

    def forward(self, node_fts, edge_attr, graph_fts, edge_index, hidden, edges_hidden, batch=None):
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(node_fts, edge_index, node_fts=node_fts, edge_attr=edge_attr, graph_fts=graph_fts, ei=edge_index, hidden=hidden, edges_hidden=edges_hidden, batch=batch.batch, btch=batch)
        return x, edges_hidden

if __name__ == '__main__':
    breakpoint()
