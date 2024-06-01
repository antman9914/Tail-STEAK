import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from model_hetero.layers import HGNConv

class HGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, id_emb, num_etype, edge_channels, num_layers, heads=4, aux_feat=False):
        super(HGN, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.num_etype = num_etype
        self.edge_channels = edge_channels
        self.heads = heads
        self.id_emb = nn.Parameter(id_emb)
        self.play_trans = nn.Parameter(torch.Tensor(in_channels, in_channels)) if aux_feat else None
        self.convs = nn.ModuleList()
        self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2*heads, hidden_channels), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(hidden_channels, 1))
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(HGNConv((in_channels, in_channels), hidden_channels, heads=heads, edge_channel=edge_channels, num_etype=num_etype, residual=True))
        self.epsilon = torch.FloatTensor([1e-12])
        self.reset_parameters()
    
    def reset_parameters(self, emb_init=None):
        if emb_init is not None:
            self.id_emb.data = emb_init
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.pred_head.apply(weight_reset)
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
        if self.play_trans is not None:
            glorot(self.play_trans)
    
    def forward(self, x, adjs, n_id, neg_sample_num, node_idx=None, out_deg=None, mode='train'):
        total_adj = []
        for adj in adjs:
            total_adj.append([adj[0], adj[1]])
        adjs = total_adj

        x = self.id_emb

        for i in range(self.num_layers):
            x, _ = self.convs[i](x, adjs, res_attn=None)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.3, training=self.training)

        splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
        out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:]), dim=0)   
        out, pos_out, neg_out = x[out_idx], x[pos_idx], x[neg_idx]
        
        tiled_out = out.repeat(1, neg_sample_num).reshape(-1, self.hidden_channel * self.heads)

        pos_logit = torch.cat([out, pos_out], dim=-1)
        neg_logits = torch.cat([tiled_out, neg_out], dim=-1)

        pos_logit = self.pred_head(pos_logit)
        neg_logits = self.pred_head(neg_logits)
        
        return pos_logit, neg_logits

    def embed(self, adjs, n_id, node_idx=None):
        total_adj = []
        for adj in adjs:
            tmp_adj = torch.cat([adj[0], adj[0]+self.id_emb.size(0) // 2], dim=-1)
            total_adj.append([tmp_adj, adj[1]])
        adjs = total_adj
        
        x = self.id_emb
        for i in range(self.num_layers):
            x, _ = self.convs[i](x, adjs, res_attn=None)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.3, training=self.training)
        
        n_id = torch.tensor(n_id, dtype=torch.int64).T.reshape(-1)
        return x[n_id]

