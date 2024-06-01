import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import SAGEConv, GCNConv
from model_hetero.layers import HGNConv, LGCConv


class TailSTEAK(nn.Module):
    def __init__(self, in_channels, hidden_channels, id_emb, num_etype, edge_channels, num_layers,
                 heads=4, ssl_temp=0.5, deg_t_low=15, U=500, base='HGN'):
        super(TailSTEAK, self).__init__()
        self.base = base
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.num_etype = num_etype
        self.edge_channels = edge_channels
        self.heads = heads
        self.ssl_temp = ssl_temp
        self.U = U
        self.deg_t_low = int(deg_t_low)
        self.id_emb = nn.Parameter(id_emb)
        self.convs = nn.ModuleList()
        if base == 'HGN':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2*heads, hidden_channels), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(hidden_channels, 1))
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels * heads
                self.convs.append(HGNConv((in_channels, in_channels), hidden_channels, heads=heads, edge_channel=edge_channels, num_etype=num_etype, residual=True))        
        elif base == 'LightGCN':
            self.pred_head = nn.Sequential(nn.Linear(self.in_channel * 2, self.in_channel), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(self.in_channel, 1))
            for i in range(num_layers):
                self.convs.append(LGCConv(self.in_channel))
        elif base == 'SAGE':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2, hidden_channels), nn.LeakyReLU(0.2),nn.Dropout(0.3), nn.Linear(hidden_channels, 1))
            for i in range(num_layers):
                in_channels = in_channels if i == 0 else hidden_channels
                self.convs.append(SAGEConv(in_channels, hidden_channels))
        elif base == 'GCN':
            self.pred_head = nn.Sequential(nn.Linear(hidden_channels*2, hidden_channels), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(hidden_channels,1))
            for i in range(num_layers):
                in_channel = self.in_channel if i == 0 else hidden_channels
                self.convs.append(GCNConv(in_channel, hidden_channels))
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
    
    def forward(self, x, adjs, n_id, neg_sample_num=19, ssl_x=None, ssl_adjs=None, ssl_n_id=None, diff_adjs=None, 
                    out_deg=None, node_idx=None, ssl_node_idx=None, mode='train', device=None):
        id_emb =  self.id_emb

        if ssl_adjs is not None:
            ssl_adj = []
            for adj in ssl_adjs:
                ssl_adj.append([adj[0], adj[1]])
            ssl_adjs = ssl_adj
            ssl_x = self.id_emb.clone()

        x = id_emb
        
        x_backup = x
        if self.base == 'HGN':
            full_edge_index, _ = adjs[0]
            out_dim = self.heads * self.hidden_channel
            for i in range(self.num_layers):
                x, _ = self.convs[i](x, adjs, res_attn=None)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.3, training=self.training)
        elif self.base == 'LightGCN':
            out_dim = self.in_channel
            full_edge_index, _ = adjs[0]
            final_x = id_emb
            for i in range(self.num_layers):
                x = self.convs[i](x, full_edge_index)
                final_x = final_x + x
            x = final_x / (self.num_layers + 1)
        elif self.base == 'SAGE' or self.base == 'GCN':
            out_dim = self.hidden_channel
            full_edge_index, _ = adjs[0]
            for i in range(self.num_layers):
                x = self.convs[i](x, full_edge_index)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.3, training=self.training)
        
        if mode == 'train':
            flag = False
            if isinstance(out_deg, tuple):
                flag = True
                out_deg, train_deg, in_deg = out_deg
                splits = train_deg.split(train_deg.size(0) // (2 + neg_sample_num), dim=0)
                train_deg = splits[0]
                splits = train_deg.split(in_deg.size(0) // (2 + neg_sample_num), dim=0)
                in_deg = splits[0]
            splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
            out_deg = splits[0]
            splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
            out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:]), dim=0)

            out, pos_out, neg_out = x[out_idx], x[pos_idx], x[neg_idx]

            # # Uncomment this block when train only on head nodes
            out = out[out_deg > self.deg_t_low]
            pos_out = pos_out[out_deg > self.deg_t_low]
            neg_out = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg > self.deg_t_low].reshape(-1, out_dim)
            tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)

            pos_logit = torch.cat([out, pos_out], dim=-1)
            neg_logits = torch.cat([tiled_out, neg_out], dim=-1)
            pos_logit = self.pred_head(pos_logit)
            neg_logits = self.pred_head(neg_logits)

            # # Needed part
            out_tail, pos_tail, neg_tail = x[out_idx], x[pos_idx], x[neg_idx]
            mask_tail = (out_deg <= self.deg_t_low)
            out_tail = out_tail[mask_tail]
            pos_tail = pos_tail[mask_tail]

            pos_logit_tail = torch.cat([out_tail, pos_tail], dim=-1)

            ssl_eindex, _ = ssl_adjs[0]
            real_edge_index = ssl_node_idx[ssl_eindex]
            src_nid = torch.unique(out_idx, sorted=False)
            emask = None
            for i in range(src_nid.size(0)):
                cur_mask = full_edge_index[0] != src_nid[i]
                cur_mask_2 = full_edge_index[1] != src_nid[i]
                if emask is None:
                    emask = cur_mask & cur_mask_2
                else:
                    emask = emask & cur_mask & cur_mask_2
            ssl_eindex = full_edge_index[:, emask]
            if len(real_edge_index) != 0:
                inverted_eindex = torch.cat([real_edge_index[1], real_edge_index[0]]).view(2, -1)
                ssl_eindex = torch.cat([ssl_eindex, real_edge_index, inverted_eindex], dim=-1)
            ssl_adjs = [[ssl_eindex, None]]

            if self.base == 'HGN':
                for i in range(self.num_layers):
                    ssl_x, _ = self.convs[i](ssl_x, ssl_adjs, res_attn=None)
                    if i != self.num_layers - 1:
                        ssl_x = ssl_x.relu()
                        ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
            elif self.base == 'SAGE' or self.base == 'GCN':
                ssl_eindex, _ = ssl_adjs[0]
                for i in range(self.num_layers):
                    ssl_x = self.convs[i](ssl_x, ssl_eindex)
                    if i != self.num_layers - 1:
                        ssl_x = ssl_x.relu()
                        ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
            elif self.base == 'LightGCN':
                final_x = ssl_x
                for i in range(self.num_layers):
                    ssl_x = self.convs[i](ssl_x, ssl_eindex)
                    final_x = ssl_x + final_x
                ssl_x = final_x / (self.num_layers + 1)
            ssl_out = ssl_x[ssl_node_idx[ssl_n_id]]
            head_idx = out_idx[out_deg > self.deg_t_low]
            pos_head_idx = pos_idx[out_deg > self.deg_t_low]
            out_head = x[head_idx]

            pos_head = x[pos_head_idx]            
            tiled_ssl_out = ssl_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
            ssl_pos_logit = torch.cat([ssl_out, pos_head], dim=-1)
            ssl_neg_logits = torch.cat([tiled_ssl_out, neg_out], dim=-1)
            ssl_pos_logit = self.pred_head(ssl_pos_logit)
            ssl_neg_logits = self.pred_head(ssl_neg_logits)
            pos_logit = torch.cat([pos_logit, ssl_pos_logit], dim=0)
            neg_logits = torch.cat([neg_logits, ssl_neg_logits], dim=0)

            ssl_out_norm, out_head = F.normalize(ssl_out, p=2, dim=-1), F.normalize(out_head, p=2, dim=-1)
            out_tail = F.normalize(out_tail, p=2, dim=-1)
            ssl_loss_1 = self.infonce(out_head, ssl_out_norm)
            ssl_loss_1 = ssl_loss_1.mean()
            ssl_loss_1 = (ssl_loss_1 + self.infonce(ssl_out_norm, out_head).mean()) / 2

            return pos_logit, neg_logits, ssl_loss_1
        else:
            splits = n_id.split(n_id.size(0) // (2 + neg_sample_num), dim=0)
            out_idx, pos_idx, neg_idx = splits[0], splits[1], torch.cat(list(splits[2:]), dim=0)
            out, pos_out, neg_out = x[out_idx], x[pos_idx], x[neg_idx]

            if mode != 'ft' and mode != 'pseudo':
                splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
                out_deg = splits[0]

                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                pos_logit = torch.cat([out, pos_out], dim=-1)
                neg_logits = torch.cat([tiled_out, neg_out], dim=-1)
                pos_logit = self.pred_head(pos_logit)
                neg_logits = self.pred_head(neg_logits)
                out_head, pos_out_head = F.normalize(out[out_deg > 3], p=2, dim=-1), F.normalize(pos_out[out_deg > 3], p=2, dim=-1)
                out_tail, pos_out_tail = F.normalize(out[out_deg <= 3], p=2, dim=-1), F.normalize(pos_out[out_deg <= 3], p=2, dim=-1)
                out, pos_out = F.normalize(out, p=2, dim=-1), F.normalize(pos_out, p=2, dim=-1)
                return pos_logit, neg_logits

            elif mode == 'pseudo':
                out_idx = n_id
                out = x[out_idx]
                neg_sample_num = self.U
                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, neg_sample_num, out_dim)

                sampled_idx = []
                for idx in out_idx:
                    samples = np.random.choice(x.size(0), (neg_sample_num, ), replace=False)
                    while idx in samples:
                        samples = np.random.choice(x.size(0), (neg_sample_num, ), replace=False)
                    samples = torch.tensor(samples, dtype=torch.int64).unsqueeze(0)
                    sampled_idx.append(samples)
                sampled_idx = torch.cat(sampled_idx, dim=0).cuda()
                
                full_out = x[sampled_idx].reshape(-1, neg_sample_num, out_dim)
                logits = torch.cat([tiled_out, full_out], dim=-1)
                full_logits = self.pred_head(logits).squeeze()
                return sampled_idx.reshape(-1, neg_sample_num), full_logits

            elif mode == 'ft':
                splits = out_deg.split(out_deg.size(0) // (2 + neg_sample_num), dim=0)
                out_deg = splits[0]

                out_head = out[out_deg > self.deg_t_low]
                out_tail = out[out_deg <= self.deg_t_low]
                pos_out_head = pos_out[out_deg > self.deg_t_low]
                pos_out_tail = pos_out[out_deg <= self.deg_t_low]
                neg_out_head = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg > self.deg_t_low].reshape(-1, out_dim)
                neg_out_tail = neg_out.reshape(-1, out_dim * neg_sample_num)[out_deg <= self.deg_t_low].reshape(-1, out_dim)
                tiled_out = out.repeat(1, neg_sample_num).reshape(-1, out_dim)

                pos_logit = torch.cat([out, pos_out], dim=-1)
                neg_logits = torch.cat([tiled_out, neg_out], dim=-1)

                ssl_eindex, _ = ssl_adjs[0]
                real_edge_index = ssl_node_idx[ssl_eindex]
                src_nid = torch.unique(out_idx, sorted=False)
                emask = None
                for i in range(src_nid.size(0)):
                    cur_mask = full_edge_index[0] != src_nid[i]
                    cur_mask_2 = full_edge_index[1] != src_nid[i]
                    if emask is None:
                        emask = cur_mask & cur_mask_2
                    else:
                        emask = emask & cur_mask & cur_mask_2
                ssl_eindex = full_edge_index[:, emask]
                if len(real_edge_index) != 0:
                    inverted_eindex = torch.cat([real_edge_index[1], real_edge_index[0]]).view(2, -1)
                    ssl_eindex = torch.cat([ssl_eindex, real_edge_index, inverted_eindex], dim=-1)
                ssl_adjs = [[ssl_eindex, None]]
                
                if self.base == 'HGN':
                    for i in range(self.num_layers):
                        ssl_x, _ = self.convs[i](ssl_x, ssl_adjs, res_attn=None)
                        if i != self.num_layers - 1:
                            ssl_x = ssl_x.relu()
                            ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
                elif self.base == 'SAGE' or self.base == 'GCN':
                    ssl_eindex, _ = ssl_adjs[0]
                    for i in range(self.num_layers):
                        ssl_x = self.convs[i](ssl_x, ssl_eindex)
                        if i != self.num_layers - 1:
                            ssl_x = ssl_x.relu()
                            ssl_x = F.dropout(ssl_x, p=0.3, training=self.training)
                elif self.base == 'LightGCN':
                    final_x = ssl_x
                    for i in range(self.num_layers):
                        ssl_x = self.convs[i](ssl_x, ssl_eindex)
                        final_x = ssl_x + final_x
                    ssl_x = final_x / (self.num_layers + 1)
                ssl_out = ssl_x[ssl_node_idx[ssl_n_id]]

                tiled_ssl_out = ssl_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                ssl_pos_logit = torch.cat([ssl_out, pos_out_head], dim=-1)
                ssl_neg_logits = torch.cat([tiled_ssl_out, neg_out_head], dim=-1)
                pos_logit = torch.cat([pos_logit, ssl_pos_logit], dim=0)
                neg_logits = torch.cat([neg_logits, ssl_neg_logits], dim=0)
                pos_logit = self.pred_head(pos_logit)
                neg_logits = self.pred_head(neg_logits)

                ssl_out, out_head = F.normalize(ssl_out, p=2, dim=-1), F.normalize(out_head, p=2, dim=-1)
                out_tail = F.normalize(out_tail, p=2, dim=-1)

                ssl_loss_1 = self.infonce(out_head, ssl_out)# , out_tail)
                ssl_loss_1 = (ssl_loss_1 + self.infonce(ssl_out, out_head).mean()) / 2

                if diff_adjs is not None:
                    diff_x = self.id_emb
                    tail_idx = out_idx[out_deg <= self.deg_t_low]
                    if self.base == 'HGN':
                        for i in range(self.num_layers):
                            diff_x, _ = self.convs[i](diff_x, diff_adjs, res_attn=None)
                            if i != self.num_layers - 1:
                                diff_x = diff_x.relu()
                                diff_x = F.dropout(diff_x, p=0.3, training=self.training)
                    elif self.base == 'SAGE' or self.base == 'GCN':
                        diff_eindex, _ = diff_adjs[0]
                        for i in range(self.num_layers):
                            diff_x = self.convs[i](diff_x, diff_eindex)
                            if i != self.num_layers - 1:
                                diff_x = diff_x.relu()
                                diff_x = F.dropout(diff_x, 0.3, self.training)
                    elif self.base == 'LightGCN':
                        diff_eindex, _ = diff_adjs[0]
                        diff_x = id_emb + 0
                        final_diff_x = diff_x
                        for i in range(self.num_layers):
                            diff_x = self.convs[i](diff_x, diff_eindex)
                            final_diff_x = diff_x + final_diff_x
                        diff_x = final_diff_x / (self.num_layers + 1)
                    
                    diff_out = out[out_deg <= self.deg_t_low]
                    diff_out_dp = diff_x[tail_idx]
                    diff_out_tiled = diff_out.repeat(1, neg_sample_num).reshape(-1, out_dim)
                    pos_logit_tail = torch.cat([diff_out, pos_out_tail], dim=-1)
                    neg_logits_tail = torch.cat([diff_out_tiled, neg_out_tail], dim=-1)
                    pos_logit_tail = self.pred_head(pos_logit_tail)
                    neg_logits_tail = self.pred_head(neg_logits_tail)
                    pos_logit = torch.cat([pos_logit, pos_logit_tail], dim=0)
                    neg_logits = torch.cat([neg_logits, neg_logits_tail], dim=0)

                    diff_out, diff_out_dp = F.normalize(diff_out, p=2, dim=-1), F.normalize(diff_out_dp, p=2, dim=-1)
                    
                    diff_loss = self.infonce(diff_out, diff_out_dp)
                    diff_loss = (diff_loss + self.infonce(diff_out_dp, diff_out)) / 2
                    ssl_loss_1 = torch.cat([ssl_loss_1, diff_loss])
                ssl_loss_1 = ssl_loss_1.mean()

                return pos_logit, neg_logits, ssl_loss_1

    def infonce(self, out_1, out_2, out_tail=None):
        pos_score = (out_1 * out_2).sum(-1)
        total_score = torch.matmul(out_1, out_2.T)
        if out_tail is not None:
            total_score_2 = torch.matmul(out_1, out_tail.T)
            total_score = torch.cat([total_score, total_score_2], dim=-1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        total_score = (torch.exp(total_score / self.ssl_temp)).sum(-1)
        return -torch.log(pos_score / total_score)
    

    def embed(self, adjs, n_id, node_idx=None):
        total_adj = []
        for adj in adjs:
            total_adj.append([adj[0], adj[1]])
        adjs = total_adj
        
        if node_idx is None:
            x = self.id_emb
        else:
            x = self.id_emb[node_idx]
        if self.base == 'HGN':
            for i in range(self.num_layers):
                x, _ = self.convs[i](x, adjs, res_attn=None)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.3, training=self.training)
        elif self.base == 'LightGCN':
            full_edge_index, _ = adjs[0]
            final_x = x
            for i in range(self.num_layers):
                x = self.convs[i](x, full_edge_index)
                final_x = final_x + x
            x = final_x / (self.num_layers + 1)

        n_id = torch.tensor(n_id, dtype=torch.int64).T.reshape(-1)
        return x[n_id]
