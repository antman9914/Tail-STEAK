from typing import Optional
from typing import Union, Tuple

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, dropout, nn
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model_hetero.layers import LGCConv


class LightGCN(torch.nn.Module):

    def __init__(self, in_channels, num_layers, id_embed):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = in_channels
        self.convs = nn.ModuleList()
        self.id_embed = Parameter(id_embed)
        self.pred_head = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(in_channels, 1))
        for i in range(num_layers):
            in_channels = in_channels
            self.convs.append(LGCConv(in_channels))
        self.reset_parameters()
    
    def reset_parameters(self, emb_init=None):
        if emb_init is not None:
            self.id_embed.data = emb_init
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.pred_head.apply(weight_reset)
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
    
    def forward(self, x, adjs, n_id, neg_sample_num, node_idx=None, out_deg=None, mode='train'):
        x = self.id_embed
        final_x = self.id_embed
        
        edge_index, edge_weight = adjs

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            final_x = final_x + x
        
        final_x = final_x / (self.num_layers + 1)
        splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
        out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:]), dim=0)
        out, pos_out, neg_out = final_x[out_idx], final_x[pos_idx], final_x[neg_idx]
        tiled_out = out.repeat(1, neg_sample_num).reshape(-1, self.hidden_channel)

        pos_logit = torch.cat([out, pos_out], dim=-1)
        neg_logits = torch.cat([tiled_out, neg_out], dim=-1)
        pos_logit = self.pred_head(pos_logit)
        neg_logits = self.pred_head(neg_logits)
        return pos_logit, neg_logits
    
    def embed(self, adjs, n_id, node_idx=None):
        x = self.id_embed
        final_x = self.id_embed
        edge_index, edge_weight = adjs[0]
        edge_index_ref = edge_index + self.id_embed.size(0)//2
        edge_index = torch.cat([edge_index, edge_index_ref], dim=-1)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            final_x = final_x + x
        
        final_x = final_x / (self.num_layers + 1)
        n_id = torch.tensor(n_id, dtype=torch.int64).T.reshape(-1)
        return final_x[n_id]

