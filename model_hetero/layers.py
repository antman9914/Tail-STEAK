from typing import List, Optional
from typing import Union, Tuple, Callable
import torch

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F

class HGNConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        edge_channel: int = 32,
        num_etype: int = 6,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        beta: float = 0.05,
        residual: bool = False,
        activation = None,
        # add_self_loops: bool = True,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        # self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        self.edge_emb = nn.Parameter(Tensor(num_etype, edge_channel))
        self.beta = beta
        self.activation = activation
        self.residual = residual
        self.num_etype = num_etype

        # self.post_trans = nn.ModuleList()
        # for _ in range(num_etype):
        #     self.post_trans.append(Linear(out_channels, out_channels, bias=False, weight_initializer='glorot'))

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')
        self.lin_e = Linear(edge_channel, out_channels * heads, bias=False, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # if residual and self.in_channels != heads * out_channels:
        if residual:
            self.lin_res = Linear(self.in_channels[0], heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('lin_res', None)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()
        # self.message_args = inspect.getargspec(self.message)[0][1:]

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_e.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        # for i in range(self.num_etype):
        #     self.post_trans[i].reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_e)
        glorot(self.edge_emb)
        zeros(self.bias)

    def pre_propagate(self, edge_index: Adj, size: Size = None, **kwargs):

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index,
                                             size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        for hook in self._message_forward_pre_hooks.values():
            res = hook(self, (msg_kwargs, ))
            if res is not None:
                msg_kwargs = res[0] if isinstance(res, tuple) else res
        self.message(**msg_kwargs)
    

    def forward(self, x: Union[Tensor, OptPairTensor], adjs, res_attn: OptTensor = None, size: Size = None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            res_x = x
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else: 
            x_src, x_dst = x
            res_x = x_dst
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        num_nodes = x_src.size(0)

        total_edge_index = []
        for type_id, (edge_index, edge_weight) in enumerate(adjs):
            if edge_index == []:
                continue
            # if self.add_self_loops:
            #     if isinstance(edge_index, Tensor):
            #         if x_dst is not None:
            #             num_nodes = min(num_nodes, x_dst.size(0))
            #         num_nodes = min(size) if size is not None else num_nodes
            #         edge_index, edge_attr = remove_self_loops(
            #             edge_index, edge_attr)
            #         edge_index, edge_attr = add_self_loops(
            #             edge_index, edge_attr, fill_value=self.fill_value,
            #             num_nodes=num_nodes)
            self.pre_propagate(edge_index, x=x, alpha=alpha, res_attn=res_attn, edge_type=type_id, size=size, obtain_attn=True, bound=None, edge_weight=None)
            total_edge_index.append(edge_index[1])

        assert self._alpha is not None
        bound = [0]
        idx = 0
        # for ts in total_edge_index:
        for i, (edge_index, _) in enumerate(adjs):
            if edge_index == []:
                bound.append(bound[-1])
            else:
                bound.append(bound[-1] + total_edge_index[idx].size(0))
                idx += 1
        index = torch.cat(total_edge_index, dim=-1)
        alpha_backup = alpha
        alpha = torch.cat(self._alpha, dim=0)
        alpha = softmax(alpha, index, None, num_nodes)
        self._alpha = alpha
        out = None
        for type_id, (edge_index, edge_weight) in enumerate(adjs):
            if edge_index == []:
                continue
            
            cur_out = self.propagate(edge_index, x=x, alpha=alpha_backup, res_attn=res_attn, edge_type=type_id, size=size, obtain_attn=False, bound=bound, edge_weight=None)
            # if out is None:
            #     out = self.post_trans[type_id](cur_out)
            # else:
            #     out = out + self.post_trans[type_id](cur_out)
            if out is None:
                out = cur_out
            else:
                out = out + cur_out
        self._alpha = None
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # if self.residual:
        if self.lin_res is not None:
            res_val = self.lin_res(res_x)
        # else:
        #     res_val = res_x
            out = res_val + out

        if self.bias is not None:
            out += self.bias
        if self.activation:
            out = self.activation(out)

        # if isinstance(return_attention_weights, bool):
        #     if isinstance(edge_index, Tensor):
        #         return out, (edge_index, alpha)
        #     elif isinstance(edge_index, SparseTensor):
        #         return out, edge_index.set_value(alpha, layout='coo')
        # else:
        return out, alpha


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, res_attn: OptTensor, edge_type: int, obtain_attn: bool, bound: Optional[List[int]], edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        if obtain_attn:        
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

            e_emb = self.edge_emb[edge_type]
            x_e = self.lin_e(e_emb).view(-1, self.heads, self.out_channels)
            alpha_e = (x_e * self.att_e).sum(-1)
            alpha = alpha + alpha_e
            if res_attn is not None:
                alpha = alpha * (1 - self.beta) + res_attn * self.beta

            # alpha = alpha * torch.log(edge_weight+1).unsqueeze(-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self._alpha is None:
                self._alpha = [alpha]
            else:
                self._alpha.append(alpha)
            return x_j
        else:
            lower, upper = bound[edge_type], bound[edge_type + 1]
            alpha = self._alpha[lower:upper]
            # alpha = alpha * torch.log(edge_weight+1).unsqueeze(-1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            # return x_j * self.edge_emb[edge_type].unsqueeze(0) * alpha.unsqueeze(-1)
            return x_j * alpha.unsqueeze(-1)


class LGCConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 normalize: bool = False,
                 bias: bool = True, **kwargs):  
        super(LGCConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.normalize = normalize
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.improved = False
        self.add_self_loops = True
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
