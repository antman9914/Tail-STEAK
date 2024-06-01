import torch
import json, time
import copy
import random
import numpy as np
from typing import  Optional, Tuple, NamedTuple
from torch import Tensor
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from collections import Counter


class Adj(NamedTuple):
    edge_index: Tensor
    edge_weight: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        edge_weight = self.edge_weight.to(*args, **kwargs) if self.edge_weight is not None else None
        return Adj(edge_index, edge_weight, self.size)


class SocLoader(DataLoader):
    def __init__(self, node_feat, mode='train', debias_on=False, graph_path="./graph.txt", deg_t_low=15, gamma=2, pseudo_edge_index=None, **kwargs):
        
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        graph_info = json.load(open(graph_path, 'r'))

        edge_index_set = graph_info['base_graph']
        edge_w_set = []
        if mode == 'test':
            val_set = np.array(graph_info['val_set'])[:, :2]
            edge_index_set[0] += val_set.tolist()
        
        for i in range(len(edge_index_set)):
            edge_index_set[i] = torch.tensor(edge_index_set[i], dtype=torch.int64).t()
            # For undirected graph
            if 'gowalla' not in graph_path:
                inverted_eindex = torch.cat([edge_index_set[i][1], edge_index_set[i][0]], dim=-1).reshape(2, -1)
                edge_index_set[i] = torch.cat([edge_index_set[i], inverted_eindex], dim=-1)
            edge_w_set.append(torch.ones(edge_index_set[i].size(1)))

        self.node_feat, self.edge_weight, self.edge_index = node_feat, edge_w_set, edge_index_set
        num_nodes = self.node_feat.size(0)
        self.num_nodes = num_nodes
        
        # These array-based code here allow extensions to heterogeneous graph setting
        self.adj_t, self.in_deg, self.out_deg = [], [], []
        for i in range(len(edge_index_set)):
            edge_index = edge_index_set[i]
            value = torch.arange(edge_index.size(1))
            self.adj_t.append(SparseTensor(row=edge_index[0], col=edge_index[1],
                                        value=value,
                                        sparse_sizes=(num_nodes, num_nodes)).t())
            value_2 = torch.ones(edge_index.size(1))
            tmp_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                        value=value_2,
                                        sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
            tmp_adj += torch.sparse_coo_tensor((range(num_nodes), range(num_nodes)),
                                            [1.] * num_nodes)
            self.in_deg.append(torch.sparse.sum(tmp_adj, dim=0).values() - 1)
            self.out_deg.append(torch.sparse.sum(tmp_adj.t(), dim=0).values() - 1)

        homo_adj_t = None
        for i in range(len(edge_index_set)):
            edge_index = edge_index_set[i]
            value = edge_w_set[i]
            cur_adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes)).t().to_torch_sparse_coo_tensor()
            if homo_adj_t is None:
                homo_adj_t = cur_adj_t
            else:
                homo_adj_t = cur_adj_t + homo_adj_t
        homo_adj_t = homo_adj_t.coalesce()
        homo_edge_index = homo_adj_t.indices()
        self.homo_edge_index = homo_edge_index
        self.homo_edge_weight = homo_adj_t.values()

        # Calculate degree distribution over "original socgraph"
        value = torch.arange(homo_edge_index.size(1))
        self.homo_adj_t = SparseTensor(row=homo_edge_index[0], col=homo_edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes)).t()
        soc_edge_index = edge_index_set[0]
        value_2 = torch.ones(soc_edge_index.size(1))
        tmp_adj = SparseTensor(row=soc_edge_index[0], col=soc_edge_index[1], value=value_2, sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        tmp_adj += torch.sparse_coo_tensor((range(num_nodes), range(num_nodes)),
                                            [1.] * num_nodes)
        self.homo_in_deg = torch.sparse.sum(tmp_adj, dim=0).values() - 1
        self.homo_out_deg = torch.sparse.sum(tmp_adj.t(), dim=0).values() - 1

        self.mode = mode
        self.debias_on = debias_on
        self.deg_t_low = deg_t_low
        self.gamma = gamma

        if mode == 'train':
            self.pseudo_edge_index = None
            if pseudo_edge_index is None:
                self.dataset = torch.arange(self.edge_index[0].size(1)).reshape(-1, 1)
                self.diffusion_adj_t = None
            else:
                pseudo_edge_index = pseudo_edge_index.T
                inverted_pseudo_edge = torch.cat([pseudo_edge_index[1], pseudo_edge_index[0]]).reshape(2, -1)
                self.pseudo_edge_index = torch.cat([pseudo_edge_index, inverted_pseudo_edge], dim=-1)
                pseudo_full_eindex = torch.cat([self.edge_index[0], self.pseudo_edge_index], dim=-1)
                value = torch.arange(pseudo_full_eindex.size(1))
                self.diffusion_adj_t = SparseTensor(row=pseudo_full_eindex[0], col=pseudo_full_eindex[1], value=value, sparse_sizes=(num_nodes, num_nodes)).t()
                self.dataset = torch.arange(pseudo_full_eindex.size(1)).reshape(-1, 1)
        elif mode == 'val':
            self.dataset = torch.tensor(graph_info['val_set'], dtype=torch.int64)
        else:
            self.dataset = torch.tensor(graph_info['test_set'], dtype=torch.int64)

        super(SocLoader, self).__init__(self.dataset, collate_fn=self.sample, **kwargs)  

    def sample(self, batch):
        if self.mode == 'train':
            batch = torch.cat(batch)
            batch_orig = batch[batch < self.edge_weight[0].size(0)]

            backup_adj_t = copy.deepcopy(self.adj_t)
            backup_eweight = copy.deepcopy(self.edge_weight)
            neg_sample_num = 19
            full_edge_index = torch.cat(self.edge_index, dim=-1) if len(self.edge_index) > 1 else self.edge_index[0]
            pos_samples = full_edge_index.t()[batch_orig]
            flag = (neg_sample_num * pos_samples.size(0)) >= self.num_nodes
            neg_samples = np.random.choice(self.num_nodes, (neg_sample_num * pos_samples.size(0), ), replace=flag)
            neg_samples = torch.tensor(neg_samples).reshape(-1, neg_sample_num)

            if self.pseudo_edge_index is not None:
                batch_pseudo = batch[batch >= self.edge_weight[0].size(0)] - self.edge_weight[0].size(0)
                pos_sample_pseudo = self.pseudo_edge_index.t()[batch_pseudo]
                neg_sample_pseudo = np.random.choice(self.num_nodes, (neg_sample_num * pos_sample_pseudo.size(0), ), replace=False)
                neg_sample_pseudo = torch.tensor(neg_sample_pseudo).reshape(-1, neg_sample_num)

                pos_samples = torch.cat([pos_samples, pos_sample_pseudo], dim=0)
                neg_samples = torch.cat([neg_samples, neg_sample_pseudo], dim=0)
            
            idx_list = [batch_orig]
            for i in range(1):
                mask = torch.ones(self.edge_weight[i].size(0), dtype=torch.bool)
                mask[idx_list[i]] = False
                self.edge_weight[i] = self.edge_weight[i][mask]
                eindex = self.edge_index[i][:, mask]
                self.adj_t[i] = SparseTensor(row=eindex[0], col=eindex[1], value=torch.arange(eindex.size(1)), sparse_sizes=(self.num_nodes, self.num_nodes)).t()
            batch = torch.cat([pos_samples, neg_samples], dim=-1)
        else:
            sample_size = batch[0].size(0)
            batch = torch.cat(batch).reshape(-1, sample_size)

        orig_batch = batch
        batch_len = batch.size(0)

        if self.debias_on:
            out_deg = self.homo_out_deg[orig_batch[:, 0]]
            ssl_nid = [orig_batch[out_deg > self.deg_t_low, 0].reshape(-1) for _ in range(len(self.adj_t))]
            diff_nid = [orig_batch[out_deg <= self.deg_t_low, 0].reshape(-1) for _ in range(len(self.adj_t))]
            
            orig_ssl_nid = ssl_nid[0].numpy().tolist()
            orig_diff_nid = diff_nid[0].numpy().tolist() if self.diffusion_adj_t is not None else []

        out_deg = self.homo_out_deg[orig_batch[:, 0]]
        ref_batch = batch.T.reshape(-1)
        n_id = [ref_batch for _ in range(len(self.adj_t))]
        orig_n_id = n_id[0].numpy().tolist()
        
        total_adjs = []
        ssl_adjs = []
        diffusion_adjs = []

        if self.debias_on:
            ssl_adj = []
            sample_bound = np.random.randint(1, self.gamma)   
            for i in range(len(self.adj_t)):
                adj_t, ssl_nid[i] = self.adj_t[i].sample_adj(ssl_nid[i], sample_bound, replace=False)
                e_id = adj_t.storage.value()
                edge_weight = self.edge_weight[i][e_id]
                cur_size = adj_t.sparse_sizes()[::-1]
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                ssl_adj.append(Adj(edge_index, edge_weight, cur_size))
            ssl_adjs.append(ssl_adj)

            if len(orig_diff_nid) > 0:
                diffusion_adj = []
                sample_bound = 50
                for i in range(len(self.adj_t)):
                    adj_t, diff_nid[i] = self.diffusion_adj_t.sample_adj(diff_nid[i], sample_bound, replace=False)
                    cur_size = adj_t.sparse_sizes()[::-1]
                    row, col, _ = adj_t.coo()
                    edge_index = torch.stack([col, row], dim=0)
                    diffusion_adj.append(Adj(edge_index, None, cur_size))
                diffusion_adjs.append(diffusion_adj)
            
        if self.debias_on:
            total_ssl_nid = torch.unique(torch.cat(ssl_nid, dim=-1), sorted=False)
            node_idx = torch.full((self.num_nodes, ), -1)
            node_idx[total_ssl_nid] = torch.arange(total_ssl_nid.size(0))
            for j, adj in enumerate(ssl_adjs[-1]):
                orig_edge_index = ssl_nid[j][adj.edge_index]
                new_edge_index = node_idx[orig_edge_index]
                ssl_adjs[-1][j] = Adj(new_edge_index, adj.edge_weight, adj.size)
            ssl_adjs = ssl_adjs[-1]
            ssl_nid = total_ssl_nid
            transformed_ssl_nid = node_idx[orig_ssl_nid]

            if len(orig_diff_nid) > 0:
                total_diff_nid = torch.unique(torch.cat(diff_nid, dim=-1), sorted=False)
                node_idx = torch.full((self.num_nodes, ), -1)
                node_idx[total_diff_nid] = torch.arange(total_diff_nid.size(0))
                for j, adj in enumerate(diffusion_adjs[-1]):
                    orig_edge_index = diff_nid[j][adj.edge_index]
                    new_edge_index = node_idx[orig_edge_index]
                    diffusion_adjs[-1][j] = Adj(new_edge_index, adj.edge_index, adj.size)
                diffusion_adjs = diffusion_adjs[-1]
                diff_nid = total_diff_nid
                transformed_diff_nid = node_idx[orig_diff_nid]
               
        if self.debias_on:
            orig_n_id = torch.tensor(orig_n_id, dtype=torch.int64)
            orig_ssl_nid = torch.tensor(orig_ssl_nid, dtype=torch.int64)
            if len(orig_diff_nid) > 0:
                orig_diff_nid = torch.tensor(orig_diff_nid, dtype=torch.int64)
                out = (batch_len, self.node_feat[n_id], total_adjs, n_id, orig_n_id, 
                        self.node_feat[ssl_nid], ssl_adjs, ssl_nid, transformed_ssl_nid, 
                        self.node_feat[diff_nid], diffusion_adjs, diff_nid, transformed_diff_nid,
                        self.homo_out_deg[orig_n_id], self.homo_in_deg[orig_n_id])
            else:
                out = (batch_len, self.node_feat[n_id], total_adjs, n_id, orig_n_id, 
                        self.node_feat[ssl_nid], ssl_adjs, ssl_nid, transformed_ssl_nid, 
                        None, None, None, None,
                        self.homo_out_deg[orig_n_id], self.homo_in_deg[orig_n_id])
        else:
            homo_out_deg = torch.cat([self.homo_out_deg, self.homo_out_deg])
            homo_in_deg = torch.cat([self.homo_in_deg, self.homo_in_deg])
            orig_n_id = torch.tensor(orig_n_id, dtype=torch.int64)
            out = (batch_len, None, total_adjs, n_id, orig_n_id, None, None, None, homo_out_deg[orig_n_id], homo_in_deg[orig_n_id])

        if self.mode == 'train':
            self.adj_t = backup_adj_t
            self.edge_weight = backup_eweight
        return out


        