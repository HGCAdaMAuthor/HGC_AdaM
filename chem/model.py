import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
# from utils2 import filter_attri_from_batch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from batch import BatchMaskedData
from loader import graph_data_obj_to_mol_simple
# import environment as env
# from rdkit import Chem
import os

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, add_self_loop=True):
        #add self loops in the edge space
        if add_self_loop == True:
            edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

            #edge_index = torch.LongTensor(edge_index)
            #print(type(edge_index))
            #print(edge_index)
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index=edge_index, size=[None, None], x=x, edge_attr=edge_embeddings)

        #return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # print(x_j.)
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)


import numpy as np
class MaskedGINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(MaskedGINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        # self.mlp1 = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
        #                                 torch.nn.Linear(2 * emb_dim, emb_dim))
        # self.mlp2 = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
        #                                 torch.nn.Linear(2 * emb_dim, emb_dim))
        self.epsilon = torch.nn.Parameter(torch.zeros(2,), requires_grad=True)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, masked_indices=None, mask=None):
        # add self loops in the edge space
        if masked_indices == None and mask == None:
            num_nodes = x.size(0)
            num_masked_nodes = int(0.5 * num_nodes + 1)
            masked_indices = np.random.choice(np.arange(0, num_nodes), size=num_masked_nodes)
            masked_indices = torch.from_numpy(masked_indices).to(x.device).long()
        if mask == None:
            mask = torch.ones_like(x).to(x.device).float()
            mask[masked_indices, :] = 0.
        to_propagate_x = mask * x

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # edge_index = torch.LongTensor(edge_index)
        # print(type(edge_index))
        # print(edge_index)
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        propagated_x = self.propagate(edge_index=edge_index, size=[None, None], x=to_propagate_x, edge_attr=edge_embeddings)
        propagated_x = propagated_x * (1. - mask)
        propagated_x = propagated_x + ((1. + self.epsilon[0]) * self.mlp(x * (1. - mask)) * (1. - mask) + x * mask)

        # mask = 1. - mask
        # to_propagate_x = propagated_x * mask
        # propagated_x_new = self.propagate(edge_index=edge_index, size=[None, None], x=to_propagate_x, edge_attr=edge_embeddings)
        # propagated_x_new = propagated_x_new * (1. - mask)
        # propagated_x_new = propagated_x_new + ((1 + self.epsilon[1]) * self.mlp2(propagated_x * (1 - mask)) + propagated_x * mask)

        return propagated_x, mask

        # return self.propagate(edge_index=edge_index, size=[None, None], x=x, edge_attr=edge_embeddings)

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr, add_self_loop=None):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        # print(edge_index.size(), x.size(), edge_attr.size(), norm.size())

        return self.propagate(edge_index=edge_index, size=[None, None], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, add_self_loop=None):

        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # print(edge_index.size())
        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, add_self_loop=None):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnn_type = gnn_type
        ### List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            elif gnn_type == "masked_gin":
                self.gnns.append(MaskedGINConv(emb_dim))
            else:
                raise NotImplementedError("Not implemented gnn type!")

        ### List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))


    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv, mask_indices=None, add_self_loop=True, one_hot_attr=None):
        has_filter = False
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 6:
            x, edge_index, edge_attr, num_nodes, num_edges, filter_k = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]
            has_filter = True
        else:
            raise ValueError("unmatched number of arguments.")

        if one_hot_attr == None:
            if x.size(1) < 5:
                x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = torch.matmul(one_hot_attr, self.x_embedding1.weight) + self.x_embedding2(x[:, 1])

        if mask_indices != None:
            x[mask_indices, :] = 0.

        if has_filter == True:
            x = filter_attri_from_batch(x, edge_index, num_nodes, num_edges, filter_k)

        h_list = [x]
        if self.gnn_type != "masked_gin":
            for layer in range(self.num_layer):
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr, add_self_loop=add_self_loop)
                h = self.batch_norms[layer](h)
                #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                h_list.append(h)
        else:  #  different propagation process for masked gnns
            mask = None
            for layer in range(self.num_layer):
                if layer % 2 == 0:
                    h, mask = self.gnns[layer](h_list[layer], edge_index, edge_attr, add_self_loop=add_self_loop)
                else:
                    assert mask != None
                    h, mask = self.gnns[layer](h_list[layer], edge_index, edge_attr, mask=(1. - mask), add_self_loop=add_self_loop)
                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                h_list.append(h)


        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list[len(h_list) - 2: ], dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

    def reset_parameterss(self):
        for layer in self.gnns:
            layer.reset_parameters()

    def loss(self, batch, device, num_samples, num_neg_samples, args, T):
        num_mols = batch.x_list_slices.size(0)
        for i in range(num_mols): # range(len(batch.x_list)):
            x = batch.x_list[batch.x_list_slices[i, 0]: batch.x_list_slices[i, 1], :]
            edge_index = batch.edge_index_list[:, batch.edge_index_slices[i, 0]: batch.edge_index_slices[i, 1]]
            edge_attr = batch.edge_attr_list[batch.edge_attr_slices[i, 0]: batch.edge_attr_slices[i, 1], :]
            batch_batch = batch.batch_list[batch.batch_slices[i, 0]: batch.batch_slices[i, 1]]
            all_rep = self(x.to(device), edge_index.to(device), edge_attr.to(device))
            all_glb_rep = pool_func(all_rep, batch_batch.to(device), args.pooling)
            all_glb_rep = all_glb_rep / (torch.norm(all_glb_rep, p=2, dim=1, keepdim=True) + 1e-9)
            # assert all_glb_rep.size(0) == 1 + num_samples + num_neg_samples
            target_rep = all_glb_rep[0, :]
            pos_rep = all_glb_rep[1: num_samples + 1, :].view(num_samples, -1)
            # if len(pos_rep.size()) == 1:
            #     pos_rep = pos_rep.view(1, -1)
            # pos_rep = pos_rep
            neg_rep = all_glb_rep[num_samples + 1:, :].view(num_neg_samples, -1)
            # if len(neg_rep.size()) == 1:
            #     neg_rep = neg_rep.view(1, -1)
            # num_pos_samples x 1
            pos_score = torch.sum(torch.mul(target_rep.unsqueeze(0), pos_rep), dim=-1, keepdim=True) / T
            # num_neg_samples
            neg_score = torch.sum(torch.mul(target_rep.unsqueeze(0), neg_rep), dim=-1, keepdim=False) / T
            # num_pos_samples x num_neg_samples
            neg_score = neg_score.unsqueeze(0).repeat(num_samples, 1)
            scores_cat = torch.cat([pos_score, neg_score], dim=-1)
            labels = torch.zeros((num_samples,), dtype=torch.long).to(device)
            loss_tmp = F.nll_loss(F.log_softmax(scores_cat, dim=-1), labels)
            if i == 0:
                loss = loss_tmp
            else:
                loss += loss_tmp
        return loss / num_mols# len(batch.x_list)

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location='cpu'))

    def reset_node_embedding(self):
        torch.nn.init.xavier_uniform_(self.gnn.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.gnn.x_embedding2.weight.data)

    def reset_parameterss(self):
        self.gnn.reset_parameterss()

    def forward(self, *argv):
        has_filter = False
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif len(argv) == 7:
            x, edge_index, edge_attr, batch, num_nodes, num_edges, filter_k = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]
            has_filter = True
        else:
            raise ValueError("unmatched number of arguments.")

        if has_filter == False:
            node_representation = self.gnn(x, edge_index, edge_attr)
        else:
            node_representation = self.gnn(x, edge_index, edge_attr, num_nodes, num_edges, filter_k)

        return self.graph_pred_linear(self.pool(node_representation, batch))



class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))
        # print("init rescale weight ", self.weight)

    def forward(self, x):
        # print(self.weight)
        if torch.isnan(torch.exp(self.weight)).any():
            # print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x


class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.scale_weight_norm = scale_weight_norm

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
        else:
            self.rescale1 = Rescale()

        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        # if self.apply_batch_norm:
        #     x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        return s, t


class Mole_Flow_Cond(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_labels, num_flow_layer=6, JK = "last", drop_ratio = 0, gnn_type = "gin",
                 learn_prior=True, deq_coeff=0.5, deq_type="random", num_node_type=118, num_edge_type=4):
        super(Mole_Flow_Cond, self).__init__()
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.emb_dim = emb_dim

        # note that we only have 118 different nodes and 4 different edges: since we do not need to consider
        # the self-loop edge, the masked node type and the no-edge type....
        # which is different from num_node_type and num_edge_type in GNN, but it doesn't matter....

        # self.edge_unroll = 12

        self.GNN = GNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        self.learn_prior = learn_prior
        if self.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]))  # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var, 0.)

            self.prior_ln_var_edge = nn.Parameter(torch.zeros([1]))  # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var_edge, 0.)
        else:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)
            self.prior_ln_var_edge = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.deq_type = deq_type
        self.deq_coeff = deq_coeff
        self.num_flow_layer = num_flow_layer
        self.divide_loss = True

        # where do these values come from?
        self.num_labels = num_labels
        max_node_unroll = 38
        max_edge_unroll = 12
        self.max_size = 38
        self.max_edge_num = 38
        num_masks = int(
            max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (
                max_edge_unroll))
        self.latent_node_length = self.max_size * self.num_node_type

        # self.latent_edge_length = (num_masks - self.max_size) * self.num_edge_type
        self.latent_edge_length = self.max_edge_num * self.num_edge_type

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)
        self.labels_embeddings = nn.Embedding(num_labels, self.emb_dim)

        self.node_st_net = nn.ModuleList([ST_Net_Exp(emb_dim * 2, self.num_node_type, hid_dim=emb_dim, bias=True,
                                                     scale_weight_norm=False) for i in range(num_flow_layer)])
        self.edge_st_net = nn.ModuleList(
            [ST_Net_Exp(emb_dim * 4, self.num_edge_type + 1, hid_dim=emb_dim * 2, bias=True,
                        scale_weight_norm=False) for i in range(num_flow_layer)])

    def pool_func(self, x, batch, mode="sum"):
        if mode == "sum":
            return global_add_pool(x, batch)
        elif mode == "mean":
            return global_mean_pool(x, batch)
        elif mode == "max":
            return global_max_pool(x, batch)

    def flow_forward(self, data, node_labels_deq, edge_labels_deq):

        x = data.split_x
        edge_index = data.split_edge_index
        edge_attr = data.split_edge_attr
        # print("split_pred_nodes_labels", data.split_pred_nodes_labels.size())
        pred_nodes_labels = data.split_pred_nodes_labels
        pred_edges_labels = data.split_pred_edges_labels
        # pred_nodes_labels_rep = self.labels_embeddings[pred_nodes_labels]
        # print("self.labels_embedding.weight.size = ", self.labels_embeddings.weight.size())
        pred_nodes_labels_rep = torch.matmul(pred_nodes_labels.float(), self.labels_embeddings.weight)
        # print("node_labels_after_embedding", pred_nodes_labels_rep.size())

        pred_edges_labels_rep = torch.matmul(pred_edges_labels.float(), self.labels_embeddings.weight)


        rep_node = self.GNN(x, edge_index, edge_attr)
        # each node to a subgraph
        pred_node_nodes_rep = rep_node[data.pred_nodes_node_idx, :]
        rep_subgraph = self.pool_func(pred_node_nodes_rep, data.pred_nodes_to_node_idx, "sum")  # num_nodes x emb_dim
        assert rep_subgraph.size(0) == data.x.size(0), "the pooled subgraph size should be consistent with number of nodes."
        rep_subgraph = torch.cat([rep_subgraph, pred_nodes_labels_rep], dim=-1)

        # print("rep_subgraph.size() = ", rep_subgraph.size())
        # print("rep_node.size() =", rep_node.size(), "rep_subgraph.size() = ", rep_subgraph.size(),
        #       "batch_subgraph_node_map.size() = ", batch_subgraph_node_map.size(), "node_deq.size() = ", node_labels_deq.size())
        if torch.isnan(rep_node).any():
            raise RuntimeError(
                'rep_node has NaN entries.')

        to_pred_edge_idx = data.pred_edge_st_ed_idx
        st_node_idx = to_pred_edge_idx[0, :]
        ed_node_idx = to_pred_edge_idx[1, :]
        rep_st_node = rep_node[st_node_idx, :]
        rep_ed_node = rep_node[ed_node_idx, :]

        pred_edge_nodes_rep = rep_node[data.pred_edge_node_idx, :]
        pred_edge_subgraph_rep = self.pool_func(pred_edge_nodes_rep, data.pred_edge_nodes_to_edge_idx, "sum")
        assert pred_edge_subgraph_rep.size(0) == rep_st_node.size(0) == rep_ed_node.size(0), "Reps to pred edges should be consistent with each other."

        # rep_subgraph_edges = rep_subgraph[to_pred_subgraph_idx, :]
        # print("connecting rep_edges", rep_subgraph_edges.size(), rep_st_node.size(), rep_ed_node.size())
        # rep_edges = torch.cat([rep_subgraph_edges, rep_st_node, rep_ed_node], dim=-1)
        rep_edges = torch.cat([pred_edge_subgraph_rep, rep_st_node, rep_ed_node, pred_edges_labels_rep], dim=-1)

        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](rep_subgraph)
            node_s = node_s.exp()
            # print(i, node_s.size(), node_t.size(), node_labels_deq.size())
            node_labels_deq = (node_labels_deq + node_t) * node_s

            if torch.isnan(node_labels_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()

            edge_s, edge_t = self.edge_st_net[i](rep_edges)
            edge_s = edge_s.exp()
            edge_labels_deq = (edge_labels_deq + edge_t) * edge_s

            if torch.isnan(edge_labels_deq).any():
                raise RuntimeError(
                    'adj_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                adj_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                adj_log_jacob += (torch.abs(edge_s) + 1e-20).log()

        # need we transform it into the batch form?
        assert node_labels_deq.size(0) == x_log_jacob.size(0)  # = total_node_num --- each node should be predicted
        assert edge_labels_deq.size(0) == adj_log_jacob.size(0) # = pred_edge_num

        return [node_labels_deq, edge_labels_deq], [x_log_jacob, adj_log_jacob]

    def _get_node_latent(self, x, edge_index, edge_attr, label_emb, latent):   # one moleculer please...
        # x.size() = num_nodes x 2
        # edge_index.size() = 2 x num_edges
        # edge_attr.size() = num_edges x 2
        # label_emb.size() = 1 x emb_size
        node_embs = self.GNN(x, edge_index, edge_attr)
        node_embs = torch.sum(node_embs, dim=0, keepdim=True)
        node_embs = torch.cat([node_embs, label_emb], dim=-1)
        # not coherent with the calculation in the forward pass, where we just use the node embeddings to calculate
        # change the calculation method in the forward pass... ---- add pooling indexes is ok?

        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](node_embs)
            inv_node_s = torch.exp(-node_s)
            latent = latent * inv_node_s - node_t

        # latent = 1 x self.num_nodes
        node_type = torch.argmax(latent, dim=1, keepdim=False)
        return int(node_type[0].item())

    def _get_edge_latent(self, x, edge_index, edge_attr, label_emb, st_ed_node, latent):
        node_embs = self.GNN(x, edge_index, edge_attr)
        st_node, ed_node = st_ed_node
        graph_emb = torch.sum(node_embs, keepdim=True, dim=0)
        edge_emb = torch.cat([graph_emb, node_embs[st_node, :].view(1, -1), node_embs[ed_node, :].view(1, -1), label_emb], dim=-1).view(1, -1)
        # edge_emb.size() = 1 x 2emb_dim

        for i in range(self.num_flow_layer):
            edge_s, edge_t = self.edge_st_net[i](edge_emb)
            inv_edge_s = torch.exp(-edge_s)
            latent = latent * inv_edge_s - edge_t

        edge_type = torch.argmax(latent, dim=1, keepdim=False)
        return int(edge_type[0].item())

    def reverse_flow(self, disturbed_latent, disturbed_latent_edges, current_nodes_attr, current_edges_index,
                     current_edges_attr, target_labels_emb):
        device = disturbed_latent.device
        # please only one molecular...
        # distrubed_laten = 1 x num_nodes
        # disturbed_edges_latent = num_edges x num_edges

        if (target_labels_emb.size(1) == self.num_labels):
            target_labels_emb = torch.matmul(target_labels_emb.float(), self.labels_embeddings.weight)

        # current_nodes_attr = torch.cat([current_nodes_attr, torch.tensor([self.num_node_type + 1, 0], dtype=torch.long,
        #                                                                  device=device).view(1, -1)], dim=0)
        latent_node = self._get_node_latent(current_nodes_attr, current_edges_index,
                                            current_edges_attr, target_labels_emb, disturbed_latent)
        current_nodes_attr[-1, 0] = latent_node
        num_nodes = current_nodes_attr.size(0)
        is_connected = False
        for i in range(num_nodes - 1):
            st_node_idx = i
            ed_node_idx = num_nodes - 1
            latent_edge = self._get_edge_latent(current_nodes_attr, current_edges_index, current_edges_attr, target_labels_emb,
                                                [st_node_idx, ed_node_idx], disturbed_latent_edges[i, :].view(1, -1))
            if latent_edge != self.num_edge_type:
                current_edges_index = torch.cat([current_edges_index, torch.tensor([st_node_idx, ed_node_idx], dtype=torch.long,
                                                                                   device=device).view(-1, 1)], dim=1)
                current_edges_index = torch.cat(
                    [current_edges_index, torch.tensor([ed_node_idx, st_node_idx], dtype=torch.long,
                                                       device=device).view(-1, 1)], dim=1)
                current_edges_attr = torch.cat([current_edges_attr, torch.tensor([latent_edge, 0], device=device, dtype=torch.long).view(1, -1)], dim=0)
                current_edges_attr = torch.cat([current_edges_attr, torch.tensor([latent_edge, 0], device=device, dtype=torch.long).view(1, -1)], dim=0)
                is_connected = True
        if not is_connected:
            current_nodes_attr = current_nodes_attr[: -1, :]

        return current_nodes_attr, current_edges_index, current_edges_attr

    def generate_one_mole_given_original_mole(self, data, target_label_one_hot, disturb_factor=0.1):
        edge_labels = data.pred_edge_attr[:, 0]
        node_labels = data.x[:, 0]

        x_deq = self.dequantinization(node_labels, one_hot=False, hot_dim=self.num_node_type)
        edge_labels_deq = self.dequantinization(edge_labels, one_hot=False, hot_dim=self.num_edge_type + 1)

        z, log_det = self.flow_forward(data, x_deq, edge_labels_deq)

        node_latent = z[0]  # 1 x self.num_nodes_type
        edge_latent = z[1]  # num_edges x self.num_edges_type

        disturbed_latent = node_latent + disturb_factor * torch.randn(node_latent.size(), device=data.x.device)
        disturbed_latent_edge = edge_latent + disturb_factor * torch.randn(edge_latent.size(), device=data.x.device)


        x, edge_index, edge_attr = self.reverse_flow(disturbed_latent, disturbed_latent_edge, data.prev_nodes_attr,
                                                     data.prev_edge_index, data.prev_edge_attr, target_label_one_hot)

        return x, edge_index, edge_attr

    def dequantinization(self, x, one_hot=False, hot_dim=None):
        assert one_hot or hot_dim is not None
        if one_hot == False:
            one_hoe = torch.zeros((x.size(0), hot_dim)).to(x.device)
            one_hoe[torch.arange(x.size(0)).to(x.device), x] = 1.0
            x = one_hoe

        return x + self.deq_coeff * torch.rand(x.size()).to(x.device)

    def forward(self, data):

        device = data.x.device

        to_pred_node_labels = data.x
        num_nodes = to_pred_node_labels.size(0)

        # non_split_node_x = data.x
        # num_nodes = non_split_node_x.size(0)

        node_labels_one_hot = torch.zeros((num_nodes, self.num_node_type)).to(device)
        node_labels_one_hot[torch.arange(num_nodes).to(device), to_pred_node_labels[:, 0] - 1] = 1.0  # assum that...
        node_labels_deq = node_labels_one_hot + self.deq_coeff * torch.rand(node_labels_one_hot.size()).to(device)

        to_pred_edge_attr = data.pred_edge_attr
        num_to_pred_edge = to_pred_edge_attr.size(0)
        edge_labels_one_hot = torch.zeros((num_to_pred_edge, self.num_edge_type + 1)).to(device)  # include the non-type
        edge_labels_one_hot[torch.arange(num_to_pred_edge).to(device), to_pred_edge_attr[:, 0]] = 1.0
        edge_labels_deq = edge_labels_one_hot + self.deq_coeff * torch.rand(edge_labels_one_hot.size()).to(device)

        z, log_det = \
            self.flow_forward(data, node_labels_deq, edge_labels_deq)

        batch_node = data.batch_node
        batch_edge = data.batch_edge

        loss = self.log_prob(z, log_det, batch_node, batch_edge)
        return loss

    def log_prob(self, z, log_det, batch_node, batch_edge):

        log_det[0] = self.pool_func(log_det[0], batch_node, "sum")
        bs = log_det[0].size(0)
        log_det[0] = log_det[0].view(bs, -1).sum(dim=-1)

        log_det[1] = self.pool_func(log_det[1], batch_edge, "sum")
        log_det[1] = log_det[1].view(bs, -1).sum(dim=-1)

        log_det[0] = log_det[0] - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        log_det[1] = log_det[1] - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = self.pool_func(ll_node, batch_node, "sum")
        ll_node = ll_node.view(bs, -1).sum(-1)
        # ll_node = ll_node.sum(-1)  # (B)

        ll_edge = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var_edge + torch.exp(-self.prior_ln_var_edge) * (z[1] ** 2))
        ll_edge = self.pool_func(ll_edge, batch_edge, "sum")
        ll_edge = ll_edge.view(bs, -1).sum(-1)
        # ll_edge = ll_edge.sum(-1)  # (B)
        print("loss to fit into prior = ", -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length)))
        ll_node += log_det[0]  # ([B])
        ll_edge += log_det[1]  # ([B])

        if self.deq_type == 'random':
            if self.divide_loss:

                return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))
            else:
                # ! useless
                return -torch.mean(ll_node + ll_edge)  # scalar
        else:
            raise NotImplementedError("deq_type must be random...")


class Mole_Flow(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, num_flow_layer=6, JK = "last", drop_ratio = 0, gnn_type = "gin",
                 learn_prior=True, deq_coeff=0.5, deq_type="random", num_node_type=118, num_edge_type=4):
        super(Mole_Flow, self).__init__()
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.emb_dim = emb_dim

        # note that we only have 118 different nodes and 4 different edges: since we do not need to consider
        # the self-loop edge, the masked node type and the no-edge type....
        # which is different from num_node_type and num_edge_type in GNN, but it doesn't matter....

        # self.edge_unroll = 12

        self.GNN = GNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        self.learn_prior = learn_prior
        if self.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros([1])) # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var, 0.)

            self.prior_ln_var_edge = nn.Parameter(torch.zeros([1]))  # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var_edge, 0.)
        else:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)
            self.prior_ln_var_edge = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.deq_type = deq_type
        self.deq_coeff = deq_coeff
        self.num_flow_layer = num_flow_layer
        self.divide_loss = True

        # where do these values come from?
        max_node_unroll = 38
        max_edge_unroll = 12
        self.max_size = 38
        num_masks = int(
            max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (
                max_edge_unroll))
        self.latent_node_length = self.max_size * self.num_node_type
        self.latent_edge_length = (num_masks - self.max_size) * self.num_edge_type

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)

        self.node_st_net = nn.ModuleList([ST_Net_Exp(emb_dim, self.num_node_type, hid_dim=emb_dim, bias=True,
                                                 scale_weight_norm=False) for i in range(num_flow_layer)])
        self.edge_st_net = nn.ModuleList([ST_Net_Exp(emb_dim * 3, self.num_edge_type + 1, hid_dim=emb_dim * 2, bias=True,
                                                 scale_weight_norm=False) for i in range(num_flow_layer)])
        # pred edge should include the non-type edge

    def dequantinization(self, x, one_hot=False, hot_dim=None):
        assert one_hot or hot_dim is not None
        if one_hot == False:
            one_hoe = torch.zeros((x.size(0), hot_dim)).to(x.device)
            one_hoe[torch.arange(x.size(0)).to(x.device), x] = 1.0
            x = one_hoe

        return x + self.deq_coeff * torch.rand(x.size()).to(x.device)

    def forward(self, data):
        
        device = data.x.device

        non_split_node_x = data.x
        num_nodes = non_split_node_x.size(0)

        node_labels_one_hot = torch.zeros((num_nodes, self.num_node_type)).to(device)
        node_labels_one_hot[torch.arange(num_nodes).to(device), non_split_node_x[:, 0] - 1] = 1.0  # assum that...
        node_labels_deq = node_labels_one_hot + self.deq_coeff * torch.rand(node_labels_one_hot.size()).to(device)

        to_pred_edge_attr = data.pred_edge_attr
        num_to_pred_edge = to_pred_edge_attr.size(0)
        edge_labels_one_hot = torch.zeros((num_to_pred_edge, self.num_edge_type + 1)).to(device)  # include the non-type
        edge_labels_one_hot[torch.arange(num_to_pred_edge).to(device), to_pred_edge_attr[:, 0]] = 1.0
        edge_labels_deq = edge_labels_one_hot + self.deq_coeff * torch.rand(edge_labels_one_hot.size()).to(device)

        z, log_det = \
            self.flow_forward(data, node_labels_deq, edge_labels_deq)

        batch_node = data.batch_node
        batch_edge = data.batch_edge

        loss = self.log_prob(z, log_det, batch_node, batch_edge)
        return loss

    def reinforce_forward(self, device, temperature=0.75, mute=False, batch_size=32, max_size_rl=48,
                          in_baseline=None, cur_iter=None, penalty=True):

        edge_unroll = 12
        prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.num_node_type]).to(device),
                                                            temperature * torch.ones([self.num_node_type]).to(device))
        prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.num_edge_type + 1]).to(device),
                                                            temperature * torch.ones([self.num_edge_type + 1]).to(device))
        node_inputs = dict()
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []

        adj_inputs = dict()
        adj_inputs['node_features'] = []  # node features --- features for each generated nodes
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []

        reward_baseline = torch.zeros([max_size_rl + 5, 2]).to(device)
        max_action_size = 25 * \
                          (int(max_size_rl + (edge_unroll - 1) * edge_unroll / 2 + (
                                      max_size_rl - edge_unroll) * edge_unroll))
        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        # print("max_action_size = ", max_action_size)

        # reward value of each mol..
        per_mol_reward = []
        per_mol_property_score = []

        self.eval()

        data_list = list()

        tot_edge_index = 0

        #### TODO: change the node sampling strategy since we still wish that the sampling process can be continued!
        with torch.no_grad():
            while (tot_edge_index == 0) or (total_node_step + total_edge_step < max_action_size and batch_length < batch_size):
                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs["adj_connections"] = []
                traj_node_inputs['adj_features'] = []  # edge features in each generated subgraph
                traj_node_inputs['node_features_cont'] = []  # nodes to be predicted
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_node_inputs["node_to_subgraph"] = []
                traj_node_inputs["pred_nodes_node_idx"] = []
                traj_node_inputs["pred_nodes_to_node_idx"] = []

                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs["adj_connections"] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs["to_pred_edge_st_ed_index"] = []
                traj_adj_inputs["to_pred_edge_subgraph_index"] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []
                traj_adj_inputs["pred_edge_st_ed_idx"] = []
                traj_adj_inputs["pred_edge_nodes_idx"] = []
                traj_adj_inputs["pred_edge_nodes_to_edge_idx"] = []
                traj_adj_inputs["pred_edge_attr"] = []

                step_cnt = 1.0

                mol = None

                current_node_features = torch.tensor([self.num_node_type, 0], dtype=torch.long, device=device).view(
                    1, -1)
                # current_node_features = torch.empty((0, 2), dtype=torch.long, device=device)
                current_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                current_edge_attr = torch.empty((0, 2), dtype=torch.long, device=device)

                current_edge_index_split = torch.empty((2, 0), dtype=torch.long, device=device)
                current_edge_attr_split = torch.empty((0, 2), dtype=torch.long, device=device)

                is_continue = True
                total_resample = 0
                each_node_resample = np.zeros([max_size_rl])

                step_num_data_edge = 0

                cnt = -1

                edge_idx = 0

                for i in range(max_size_rl):
                    new_added_edge_index = [torch.empty((2, 0), dtype=torch.long, device=device)]
                    new_added_edge_attr = [torch.empty((0, 2), dtype=torch.long, device=device)]
                    new_added_rewards = [torch.empty((0, 1), device=device)]
                    new_added_baseline_index = [torch.empty((0,), dtype=torch.long, device=device)]
                    new_added_node_fea = [torch.empty((0, 2), dtype=torch.long, device=device)]
                    new_pred_edge_nodes_idx = [torch.empty((0,), dtype=torch.long, device=device)]
                    new_pred_edge_nodes_to_edge_idx = [torch.empty((0,), dtype=torch.long, device=device) ]
                    new_pred_edge_st_ed_idx = [torch.empty((2, 0), dtype=torch.long, device=device)]
                    new_pred_edge_attr = [torch.empty((0, 2), dtype=torch.long, device=device)]

                    step_num_data_edge = 0
                    if not is_continue:
                        break

                    if i < edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = edge_unroll
                        start = i - edge_unroll

                    latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)


                        # current_node_features = torch.tensor([self.num_node_type - 1, 0], dtype=torch.long, device=device).view(1, -1)
                    # print("getting node type for {:d}th node.".format(i + 1), current_node_features.size(), current_edge_index.size(),
                    #       current_edge_attr.size())
                    # print(current_edge_index)
                    # print(current_edge_attr)
                    latent_node = self._get_node_latent(current_node_features, current_edge_index, current_edge_attr,
                                                        latent_node) + 1
                    print("current_latent_node = ", latent_node)
                    # if i == 0:
                    #     current_node_features = torch.empty((0, 2), dtype=torch.long, device=device)
                    #
                    ##### non-type node will not be generated...
                    # if latent_node >= self.num_node_type - 2:
                    #     is_continue = False
                    #     if len(traj_node_inputs["node_features"]) == 0:
                    #         print("traj_node_inputs[...] = 0")
                    #         current_node_features = torch.empty((0, 2), dtype=torch.long, device=device)
                    #     else:
                    #         print("going back..")
                    #         current_node_features = traj_node_inputs["node_features"][-1].clone()
                    #     # pop out the dummy feature for the next node
                    #     continue

                    cnt_l = cnt + 1
                    cnt_r = cnt_l + (i + 1) - 1


                    total_node_step += 1
                    node_feature_cond = torch.tensor([latent_node, 0], dtype=torch.long, device=device).view(1, -1)

                    tmp_edge_index = current_edge_index.clone() + cnt_l
                    tmp_edge_attr = current_edge_attr.clone()

                    cnt += i + 1
                    # current_node_idx = cnt

                    # add one node to the future input array lists, but we do not know whether this node is valid
                    # update node trajactory
                    traj_node_inputs['node_features'].append(current_node_features.clone())  # node_feas to pred this node
                    traj_adj_inputs["adj_connections"].append(tmp_edge_index)
                    traj_adj_inputs['adj_features'].append(tmp_edge_attr)
                    # traj_node_inputs['adj_connections'].append(current_edge_index.clone()) # edge_index to pred this node
                    # traj_node_inputs["adj_features"].append(current_edge_attr.clone()) # edge_attrs to pred this node
                    traj_node_inputs['node_features_cont'].append(node_feature_cond)  # ground truth

                    # (1,) for the reward of each node????
                    traj_node_inputs['rewards'].append(
                        torch.full(size=(1, 1), fill_value=step_cnt).to(device))  # (1, 1)

                    # (1,) each node a baseline_index ??????
                    traj_node_inputs['baseline_index'].append(
                        torch.full(size=(1,), fill_value=step_cnt).long().to(device))  # (1, 1)
                    traj_node_inputs["node_to_subgraph"].append(torch.full(
                        (current_node_features.size(0), ), i, dtype=torch.long, device=device
                    ))
                    # cnt_r = cnt
                    # cnt_l = cnt - (i + 1) + 1
                    # still need to pop out...
                    traj_node_inputs["pred_nodes_node_idx"].append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long,
                                                                                device=device))
                    traj_node_inputs["pred_nodes_to_node_idx"].append(torch.full((cnt_r - cnt_l + 1, ), i, dtype=torch.long,
                                                                                 device=device))

                    current_node_features[-1, :] = node_feature_cond.clone()
                    # current_node_features = torch.cat([current_node_features,
                    #                                    torch.tensor([self.num_node_type + 1, 0], dtype=torch.long, device=device).view(1, -1)], dim=0)
                    # current_node_features = torch.cat([current_node_features, node_feature_cond], dim=0)
                    # how to add atoms... especially we should use env to check the validaty of the generated molecular

                    #### TODO: use the generated node type and edge type to determine whether the node or the edge is
                    #### TODO: valid or just suppose that we can always generate the correct node and try to generate
                    #### TODO: edges for it? which is better?
                    if i == 0:
                        is_connected = True
                    else:
                        is_connected = False

                    for j in range(start, i):
                        valid = False
                        resample_edge = 0
                        edge_discrete_id = 0
                        invalid_bond_type_set = set()
                        while not valid:

                            # we have total 4 types of edges
                            if len(invalid_bond_type_set) < 4 and resample_edge <= 50:
                                resample_edge_per_step = 0
                                # while edge_discrete_id == 0 and resample_edge_per_step <= 30:
                                st_node = j
                                ed_node = i
                                latent_edge = prior_edge_dist.sample().view(1, -1)
                                edge_discrete_id = self._get_edge_latent(current_node_features, current_edge_index,
                                                                current_edge_attr, [st_node, ed_node], latent_edge)
                                resample_edge_per_step += 1
                            else:
                                if not mute:
                                    print('have tried all possible bond type, use virtual bond.')
                                assert resample_edge > 50 or len(invalid_bond_type_set) == self.num_edge_type
                                edge_discrete_id = self.num_edge_type   # TODO: check the mask edge part, is the masked edge_type set to 0 or other values?
                            total_edge_step += 1

                            step_num_data_edge += 1

                            # edge_attr_cont = torch.tensor([edge_discrete_id, 0], dtype=torch.long, device=device).view(1, -1)
                            #
                            # traj_adj_inputs['node_features'].append(current_node_features.clone())  # 1, max_size_rl, self.node_dim
                            # traj_adj_inputs['adj_features'].append(current_edge_attr.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                            # traj_adj_inputs["adj_connections"].append(current_edge_index.clone())
                            # traj_adj_inputs['edge_features_cont'].append(edge_attr_cont)  # 1, self.bond_dim
                            # traj_adj_inputs['index'].append(
                            #     torch.Tensor([[j + start, i]]).long().to(device).view(1, -1))  # (1, 2)
                            # step_num_data_edge += 1  # add one edge data, not sure if this should be added to the final train data
                            #
                            # if edge_discrete_id != 0:
                            #     current_edge_index = torch.cat([current_edge_index,
                            #                                     torch.tensor([st_node, ed_node], dtype=torch.long, device=device).view(-1, 1)], dim=1)
                            #     current_edge_index = torch.cat([current_edge_index,
                            #                                     torch.tensor([ed_node, st_node], dtype=torch.long,
                            #                                                  device=device).view(-1, 1)], dim=1)
                            #     current_edge_attr = torch.cat([current_edge_attr, torch.tensor([edge_discrete_id, 0], dtype=torch.long, device=device).view(1, -1)], dim=0)
                            #     current_edge_attr = torch.cat([current_edge_attr,
                            #                                    torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                            #                                                 device=device).view(1, -1)], dim=0)
                            if edge_discrete_id == self.num_edge_type:
                                valid = True
                            else:
                                current_edge_index = torch.cat([current_edge_index, torch.tensor([j, i], dtype=torch.long,
                                                                                                device=device).view(-1, 1)], dim=1)
                                current_edge_index = torch.cat([current_edge_index, torch.tensor([i, j], dtype=torch.long,
                                                                      device=device).view(-1, 1)], dim=1)
                                current_edge_attr = torch.cat([current_edge_attr, torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                                                                                              device=device).view(1, -1)], dim=0)
                                current_edge_attr = torch.cat([current_edge_attr, torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                                                                                               device=device).view(1, -1)], dim=0)

                                # print("checking validity")
                                # print(current_node_features)
                                # print(current_edge_index)
                                # print(current_edge_attr)
                                mol = graph_data_obj_to_mol_simple(current_node_features, current_edge_index, current_edge_attr)


                                # new_added_edge_index.append(torch.tensor([j, i], dtype=torch.long, device=device).view(-1, 1))
                                # new_added_edge_attr.append(torch.tensor([edge_discrete_id, 0], dtype=torch.long, device=device).view(1, -1))
                                # nei = torch.cat(new_added_edge_index, dim=1)
                                # nea = torch.cat(new_added_edge_attr, dim=0)
                                # print("generating mol adj with num_nodes =", current_node_features.size(0))
                                # mol = graph_data_obj_to_mol_simple(current_node_features[:-1, :],
                                #                                    torch.cat([current_edge_index, nei], dim=1),
                                #                                    torch.cat([current_edge_attr, nea], dim=0))

                                valid = env.check_valency(mol)
                                # print(valid)
                                if valid:
                                    is_connected = True
                                    # print(num2bond_symbol[edge_discrete_id])
                                else:  # backtrack
                                    current_edge_attr = current_edge_attr[:-2, :]
                                    current_edge_index = current_edge_index[:, :-2]

                                    # new_added_edge_index.pop()
                                    # new_added_edge_attr.pop()
                                    total_resample += 1.0
                                    each_node_resample[i] += 1.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)

                                pass
                                #### TODO: check the validaty of the generated molecular

                            if not valid:
                                step_num_data_edge -= 1

                        if i > 0:
                            # cnt += (i + 1)
                            # traj_adj_inputs['rewards'].append(
                            #     torch.full(size=(1, 1), fill_value=step_cnt).to(device))  # (1, 1)
                            # traj_adj_inputs['baseline_index'].append(
                            #     torch.full(size=(1,), fill_value=step_cnt).long().to(device))  # (1)

                            new_added_rewards.append(torch.full((1, 1), fill_value=step_cnt).to(device))
                            new_added_baseline_index.append(torch.full(size=(1,), fill_value=step_cnt).long().to(device))
                            cnt_l = cnt + 1
                            cnt_r = cnt + 1 + (i + 1) - 1

                            tmp_edge_index = current_edge_index.clone() + cnt_l
                            tmp_edge_attr = current_edge_attr.clone()

                            cnt += (i + 1)
                            ed_idx = cnt
                            st_idx = cnt - (i - j)

                            new_added_node_fea.append(current_node_features.clone())
                            new_pred_edge_nodes_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long,
                                                                                       device=device))
                            new_pred_edge_nodes_to_edge_idx.append(torch.full((cnt_r - cnt_l + 1, ), edge_idx,
                                                                                             dtype=torch.long, device=device))
                            edge_idx += 1
                            # tot_edge_index += 1
                            new_pred_edge_st_ed_idx.append(torch.tensor([st_idx, ed_idx], dtype=torch.long,
                                                                                       device=device).view(-1, 1))
                            new_pred_edge_attr.append(torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                                                                                  device=device).view(1, -1))
                            new_added_edge_index.append(tmp_edge_index)
                            new_added_edge_attr.append(tmp_edge_attr)

                            # traj_node_inputs['node_features'].append(current_node_features.clone())
                            # traj_adj_inputs["pred_edge_nodes_idx"].append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long,
                            #                                                            device=device))
                            # traj_adj_inputs["pred_edge_nodes_to_edge_idx"].append(torch.full((cnt_r - cnt_l + 1, ), edge_idx,
                            #                                                                  dtype=torch.long, device=device))
                            # edge_idx += 1
                            # traj_adj_inputs["pred_edge_st_ed_idx"].append(torch.tensor([st_idx, ed_idx], dtype=torch.long,
                            #                                                            device=device).view(-1, 1))
                            # traj_adj_inputs["pred_edge_attr"].append(torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                            #                                                       device=device).view(1, -1))
                            # traj_adj_inputs["adj_connections"].append(tmp_edge_index)
                            # traj_adj_inputs['adj_features'].append(tmp_edge_attr)


                            # if valid and edge_discrete_id != 0:
                            #     traj_adj_inputs['rewards'].append(
                            #         torch.full(size=(1, 1), fill_value=step_cnt).to(device))  # (1, 1)
                            #     traj_adj_inputs['baseline_index'].append(
                            #         torch.full(size=(1,), fill_value=step_cnt).long().to(device))  # (1)
                            #     if edge_discrete_id != 0:
                            #         st_node_idx = current_node_idx - (i - j)
                            #         have_connections_node_st_idx.append(st_node_idx)
                            #         have_connections_edge_labels.append(edge_discrete_id)
                            #         edge_attr_cont = torch.tensor([edge_discrete_id, 0], dtype=torch.long,
                            #                                       device=device).view(1, -1)
                            #         traj_adj_inputs["edge_features_cont"].append(edge_attr_cont)
                            # else:
                            #     step_num_data_edge -= 1
                                # if penalty:
                                #     # todo: the step_reward can be tuned here
                                #     traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                #     traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                #     #TODO: check baselien of invalid step, maybe we do not add baseline for invalid step
                                # else:
                                #     traj_adj_inputs['node_features'].pop(-1)
                                #     traj_adj_inputs["adj_connections"].pop(-1)
                                #     traj_adj_inputs['adj_features'].pop(-1)
                                #     traj_adj_inputs['edge_features_cont'].pop(-1)
                                #     traj_adj_inputs['index'].pop(-1)
                                #     step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1

                    if is_connected:
                        is_continue = True
                        current_node_features = torch.cat(
                            [current_node_features, torch.tensor([self.num_node_type, 0], dtype=torch.long, device=device).view(1, -1)], dim=0)
                        traj_node_inputs['node_features'].append(torch.cat(new_added_node_fea, dim=0))
                        traj_adj_inputs["pred_edge_nodes_idx"].append(torch.cat(new_pred_edge_nodes_idx, dim=-1))
                        traj_adj_inputs["pred_edge_nodes_to_edge_idx"].append(torch.cat(new_pred_edge_nodes_to_edge_idx, dim=-1))
                        # edge_idx += 1
                        tot_edge_index += len(new_pred_edge_st_ed_idx)
                        traj_adj_inputs["pred_edge_st_ed_idx"].append(torch.cat(new_pred_edge_st_ed_idx, dim=1))
                        traj_adj_inputs["pred_edge_attr"].append(torch.cat(new_pred_edge_attr, dim=0))
                        traj_adj_inputs["adj_connections"].append(torch.cat(new_added_edge_index, dim=1))
                        traj_adj_inputs['adj_features'].append(torch.cat(new_added_edge_attr, dim=0))

                        traj_adj_inputs['rewards'].append(torch.cat(new_added_rewards, dim=0))  # (1, 1)
                        traj_adj_inputs['baseline_index'].append(torch.cat(new_added_baseline_index, dim=-1))  # (1)

                    else:
                        is_continue = False
                        traj_node_inputs['node_features'].pop()
                        traj_adj_inputs["adj_connections"].pop()
                        traj_adj_inputs['adj_features'].pop()
                        traj_node_inputs['node_features_cont'].pop()
                        traj_node_inputs["node_to_subgraph"].pop()
                        traj_node_inputs['rewards'].pop()
                        traj_node_inputs['baseline_index'].pop()
                        traj_node_inputs["pred_nodes_to_node_idx"].pop()
                        traj_node_inputs["pred_nodes_node_idx"].pop()
                        current_node_features = current_node_features[:-1, :]

                    step_cnt += 1

                    # for j, st_node in enumerate(have_connections_node_st_idx):
                    #     connected_edge_index.append(torch.tensor([st_node, current_node_idx], dtype=torch.long, device=device).view(-1, 1))
                    #     connected_edge_attr.append(torch.tensor([have_connections_edge_labels[j], 0], dtype=torch.long,
                    #                                             device=device).view(1, -1))
                    #
                    # if is_connected and i != 0:
                    #     connected_edge_index = torch.cat(connected_edge_index, dim=1)
                    #     connected_edge_attr = torch.cat(connected_edge_attr, dim=0)
                    #
                    #     current_edge_index_split = torch.cat([current_edge_index, connected_edge_index], dim=1)
                    #     current_edge_attr_split = torch.cat([current_edge_attr, connected_edge_attr], dim=0)
                    #
                    #     traj_adj_inputs["node_features"].append(current_node_features.clone())
                    #     traj_adj_inputs["adj_connections"].append(current_edge_index_split.clone())
                    #     traj_adj_inputs["adj_features"].append(current_edge_attr_split.clone())
                    #     traj_adj_inputs["to_pred_edge_st_ed_index"].append(connected_edge_index.clone())
                    #     traj_adj_inputs["to_pred_edge_subgraph_index"].append(torch.full((len(have_connections_node_st_idx), ), i,
                    #                                                                  dtype=torch.long, device=device))
                    #     connected_edge_index = list()
                    #     connected_edge_attr = list()
                    #     # for i, st_node in enumerate(have_connections_node_st_idx):
                    #     #     connected_edge_index.append(
                    #     #         torch.tensor([current_node_idx, st_node], dtype=torch.long, device=device).view(-1, 1))
                    #     #     connected_edge_attr.append(
                    #     #         torch.tensor([have_connections_edge_labels[i], 0], dtype=torch.long,
                    #     #                      device=device).view(1, -1))
                    #     # connected_edge_index = torch.cat(connected_edge_index, dim=1)
                    #     # connected_edge_attr = torch.cat(connected_edge_attr, dim=0)
                    #
                    #     for j in range(len(new_added_edge_index)):
                    #         a, b = int(new_added_edge_index[j][0]), int(new_added_edge_index[j][1])
                    #         connected_edge_index.append(torch.tensor([a, b], dtype=torch.long, device=device).view(-1, 1))
                    #         connected_edge_index.append(torch.tensor([b, a], dtype=torch.long, device=device).view(-1, 1))
                    #         connected_edge_attr.append(new_added_edge_attr[j])
                    #         connected_edge_attr.append(new_added_edge_attr[j])
                    #     connected_edge_index = torch.cat(connected_edge_index, dim=1)
                    #     connected_edge_attr = torch.cat(connected_edge_attr, dim=0)
                    #
                    #
                    #     current_edge_index = torch.cat([current_edge_index, connected_edge_index], dim=1)
                    #     current_edge_attr = torch.cat([current_edge_attr, connected_edge_attr], dim=0)
                    #
                    #     connected_edge_index = list()
                    #     connected_edge_attr = list()
                    #     for i, st_node in enumerate(have_connections_node_st_idx):
                    #         connected_edge_index.append(
                    #             torch.tensor([current_node_idx, st_node], dtype=torch.long, device=device).view(-1, 1))
                    #         connected_edge_attr.append(
                    #             torch.tensor([have_connections_edge_labels[i], 0], dtype=torch.long,
                    #                          device=device).view(1, -1))
                    #     connected_edge_index = torch.cat(connected_edge_index, dim=1)
                    #     connected_edge_attr = torch.cat(connected_edge_attr, dim=0)
                    #
                    #     current_edge_index_split = torch.cat([current_edge_index_split, connected_edge_index], dim=1)
                    #     current_edge_attr_split = torch.cat([current_edge_attr_split, connected_edge_attr], dim=0)
                    #
                    # elif i != 0:
                    #     traj_node_inputs["node_features"].pop()
                    #     traj_node_inputs['adj_connections'].pop()
                    #     traj_node_inputs["adj_features"].pop()
                    #     traj_node_inputs['node_features_cont'].pop()
                    #     traj_node_inputs['rewards'].pop()
                    #     traj_node_inputs['baseline_index'].pop()
                    #     traj_node_inputs["node_to_subgraph"].pop()

                            # num_atoms = current_node_features.size(0)
                #### TODO: those invalid but already added nodes or edges?
                ####

                # reward for the valid noleculer
                # first check whether the mole is valid, then check the property...
                reward_valid = 2
                reward_property = 0
                reward_length = 0

                batch_length += 1

                # print("generated final mol with node num =", current_node_features.size() )
                # print(current_node_features)
                # print(current_edge_index)
                # print(current_edge_attr)
                mol = graph_data_obj_to_mol_simple(current_node_features[:, :], current_edge_index, current_edge_attr)

                assert mol is not None, "the final mol is None.. "
                final_valid = env.check_chemical_validity(mol)
                # assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'
                if not final_valid:
                    print("use valency check during generation but the final molecule is invalid!!")

                if not final_valid:
                    reward_valid -= 5  # this is useless, because this case will not occur in our implementation
                else:
                    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                    # print(s)  # print the final generated molecular
                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not env.steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1  # TODO: check the magnitude of this reward.
                        flag_steric_strain_filter = False
                    if not env.zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1
                        flag_zinc_molecule_filter = False

                    # todo: add arg for property_type here
                    property_type = "qed"
                    assert property_type in ['qed',
                                             'plogp'], 'unsupported property optimization, choices are [qed, plogp]'

                    try:
                        save_path = "./saved_moles"
                        output_model_file= "./temp/good_mol_point"
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        qed_coeff = 1.0
                        if property_type == 'qed':
                            score = env.qed(final_mol)
                            reward_property += (score * qed_coeff)
                            if score > 0.945:
                                print(s)
                                save_one_mol_path = os.path.join(save_path, 'good_mol_qed_{}.txt'.format(str(s)))
                                env.save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)
                                torch.save(self.state_dict(), output_model_file)

                        elif property_type == 'plogp':
                            score = env.penalized_logp(final_mol)

                            # TODO: design stable reward....
                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias)
                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.plogp_coeff)

                            if score > 4.0:
                                save_one_mol_path = os.path.join(save_path, 'good_mol_plogp.txt')
                                env.save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)
                    except:
                        print('generated mol does not pass env.qed/plogp')
                        # reward_property -= 2.0
                        # TODO: check what we should do if the model do not pass qed/plogp test
                        # workaround1: add a extra penalty reward
                        # workaround2: discard the molecule.

                reward_final_total = reward_valid + reward_property + reward_length
                # reward_final_total = reward_property
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)

                reward_decay = 0.90

                data = Data()
                data.x = torch.cat(traj_node_inputs["node_features_cont"], dim=0)
                data.edge_index = torch.cat(traj_adj_inputs["pred_edge_st_ed_idx"], dim=1)  ## dummy
                data.edge_attr = torch.cat(traj_adj_inputs["pred_edge_attr"], dim=0)  ## dummy
                data.split_x = torch.cat(traj_node_inputs['node_features'], dim=0)
                data.split_edge_index = torch.cat(traj_adj_inputs["adj_connections"], dim=1)
                data.split_edge_attr = torch.cat(traj_adj_inputs['adj_features'], dim=0)

                data.pred_nodes_node_idx = torch.cat(traj_node_inputs["pred_nodes_node_idx"], dim=-1)
                data.pred_nodes_to_node_idx = torch.cat(traj_node_inputs["pred_nodes_to_node_idx"], dim=-1)

                data.pred_edge_st_ed_idx = torch.cat(traj_adj_inputs["pred_edge_st_ed_idx"], dim=1)
                data.pred_edge_node_idx = torch.cat(traj_adj_inputs["pred_edge_nodes_idx"], dim=-1)
                data.pred_edge_nodes_to_edge_idx = torch.cat(traj_adj_inputs["pred_edge_nodes_to_edge_idx"], dim=-1)
                data.pred_edge_attr = torch.cat(traj_adj_inputs["pred_edge_attr"], dim=0)

                # data.node_to_subgraph = torch.cat(traj_node_inputs["node_to_subgraph"], dim=-1)
                # data.to_pred_edge_st_ed_idx = torch.cat(traj_adj_inputs["to_pred_edge_st_ed_index"], dim=1)
                # data.to_pred_edge_attr = torch.cat(traj_adj_inputs['edge_features_cont'], dim=0)
                # data.to_pred_edge_to_subgraph = torch.cat(traj_adj_inputs["to_pred_edge_subgraph_index"], dim=-1)

                data_list.append(data)
                batch_length += 1

                traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0)  # (max_size_rl)
                traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'],
                                                     dim=0)  # tensor of shape (max_size_rl, 1)
                traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[
                        traj_node_inputs_rewards > 0])
                node_inputs['rewards'].append(
                    traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)
                node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)

                for ss in range(traj_node_inputs_rewards.size(0)):
                    reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]

                # adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0))  # (step, 2)

                traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'],
                                                           dim=0)  # (step)
                traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay,
                                                   step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                for ss in range(traj_adj_inputs_rewards.size(0)):
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        batch = BatchMaskedData.from_data_list(data_list)

        batch = batch.to(device)

        self.train()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0]  # (max_size_rl, )
        # TODO: check the baseline for invalid edge penalty step....
        # panelty step do not have property reward. So its magnitude may be quite different from others.

        if in_baseline is not None:
            moving_coeff = 0.95
            #    #print(reward_baseline_per_step.size())
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (
                        1. - moving_coeff) + in_baseline * moving_coeff
            # print('calculating moving baseline per step')

        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1)  # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long()  # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0,
                                                  index=node_inputs_baseline_index)  # (total_size, )

        # adj_inputs_index = torch.cat(adj_inputs['index'], dim=0)  # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1)  # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long()  # (total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0,
                                                 index=adj_inputs_baseline_index)  # (total_size, )

        print("batch.x.size() = ", batch.x.size())
        node_labels = batch.x[:, 0] - 1  ## guess..
        node_label_one_hot = torch.zeros((batch.x.size(0), self.num_node_type), device=device)
        node_label_one_hot[torch.arange(batch.x.size(0)).to(device), node_labels] = 1.0
        node_labels_deq = node_label_one_hot + self.deq_coeff * torch.rand_like(node_label_one_hot)

        edge_labels = batch.pred_edge_attr[:, 0]
        edge_label_one_hot = torch.zeros((batch.pred_edge_attr.size(0), self.num_edge_type + 1), device=device)
        edge_label_one_hot[torch.arange(batch.pred_edge_attr.size(0)).to(device), edge_labels] = 1.0
        edge_labels_deq = edge_label_one_hot + self.deq_coeff * torch.rand_like(edge_label_one_hot)


        # print(batch.split_x.size(), batch.split_edge_index.size(), batch.split_edge_attr.size(), batch.pred_edge_st_ed_idx.size())
        z, log_det = self.flow_forward(batch, node_labels_deq, edge_labels_deq)
        z_node, z_edge = z
        logdet_node, logdet_edge = log_det


        print("node_labels_deq.size() = ", node_labels_deq.size())
        print("logdet_node.size() = ", logdet_node.size(), "logdet_edge.size() = ", logdet_edge.size(), batch.batch_node.size(),
              batch.batch_edge.size(), batch.batch_node.max(), batch.batch_edge.max())


        logdet_node = self.pool_func(logdet_node, batch.batch_node, "sum")
        logdet_edge = self.pool_func(logdet_edge, batch.batch_edge, "sum")

        if (logdet_node.size(0) != logdet_edge.size(0)):
            bs = max(logdet_node.size(0), logdet_edge.size(0))
            if logdet_node.size(0) < bs:
                logdet_node = torch.cat([logdet_node, torch.zeros((bs - logdet_node.size(0), logdet_node.size(1)), device=device)], dim=0)
            else:
                logdet_edge = torch.cat([logdet_edge, torch.zeros((bs - logdet_edge.size(0), logdet_edge.size(1)), device=device)], dim=0)
        else:
            bs = logdet_node.size(0)
        logdet_node = logdet_node.view(bs, -1).sum(dim=-1)
        logdet_edge = logdet_edge.view(bs, -1).sum(dim=-1)


        node_total_length = z_node.size(0) * float(self.num_node_type)
        edge_total_length = z_edge.size(0) * float(self.num_edge_type)

        # logdet_node = logdet_node - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        # logdet_edge = logdet_edge - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node ** 2))
        ll_node = self.pool_func(ll_node, batch.batch_node, "sum")

        if (ll_node.size(0) != bs):
            ll_node = torch.cat([ll_node, torch.zeros((bs - ll_node.size(0), ll_node.size(1)), device=device)], dim=0)
        ll_node = ll_node.view(bs, -1).sum(-1)

        ll_edge = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge ** 2))
        ll_edge = self.pool_func(ll_edge, batch.batch_edge, "sum")

        if (ll_edge.size(0) != bs):
            ll_edge = torch.cat([ll_edge, torch.zeros((bs - ll_edge.size(0), ll_edge.size(1)), device=device)], dim=0)
        ll_edge = ll_edge.view(bs, -1).sum(-1)

        ll_node += logdet_node  # ([B])
        ll_edge += logdet_edge  # ([B])

        # TODO: check baseline of penalty step(invalid edge)
        # TODO: check whether moving baseline is better than batch average.
        node_inputs_rewards = self.pool_func(node_inputs_rewards, batch.batch_node, "sum")
        node_inputs_baseline = self.pool_func(node_inputs_baseline, batch.batch_node, "sum")

        adj_inputs_rewards = self.pool_func(adj_inputs_rewards, batch.batch_edge, "sum")
        adj_inputs_baseline = self.pool_func(adj_inputs_baseline, batch.batch_edge, "sum")

        if node_inputs_rewards.size(0) != bs:
            node_inputs_rewards = torch.cat([node_inputs_rewards, torch.zeros((bs - node_inputs_rewards.size(0), node_inputs_rewards.size(1)), device=device)], dim=0)
            node_inputs_baseline = torch.cat([node_inputs_baseline, torch.zeros((bs - node_inputs_baseline.size(0), node_inputs_baseline.size(1)), device=device)], dim=0)

        if adj_inputs_rewards.size(0) != bs:
            adj_inputs_rewards = torch.cat([adj_inputs_rewards, torch.zeros(
                (bs - adj_inputs_rewards.size(0), adj_inputs_rewards.size(1)), device=device)], dim=0)
            adj_inputs_baseline = torch.cat([adj_inputs_baseline, torch.zeros(
                (bs - adj_inputs_baseline.size(0), adj_inputs_baseline.size(1)), device=device)], dim=0)

        ll_node = ll_node * (node_inputs_rewards - node_inputs_baseline)   # rewards are used here!
        ll_edge = ll_edge * (adj_inputs_rewards - adj_inputs_baseline)

        if self.deq_type == 'random':
            if self.divide_loss:
                # print(ll_node.size())
                # print(ll_edge.size())
                return -((ll_node.sum() + ll_edge.sum()) / (
                            node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step
            else:
                # ! useless
                return -torch.sum(ll_node + ll_edge) / batch_length  # scalar


    def flow_forward(self, data, node_labels_deq, edge_labels_deq):

        x = data.split_x
        edge_index = data.split_edge_index
        edge_attr = data.split_edge_attr
        rep_node = self.GNN(x, edge_index, edge_attr)
        # each node to a subgraph
        pred_node_nodes_rep = rep_node[data.pred_nodes_node_idx, :]
        rep_subgraph = self.pool_func(pred_node_nodes_rep, data.pred_nodes_to_node_idx, "sum")  # num_nodes x emb_dim
        assert rep_subgraph.size(0) == data.x.size(0), "the pooled subgraph size should be consistent with number of nodes."
        # print("rep_subgraph.size() = ", rep_subgraph.size())
        # print("rep_node.size() =", rep_node.size(), "rep_subgraph.size() = ", rep_subgraph.size(),
        #       "batch_subgraph_node_map.size() = ", batch_subgraph_node_map.size(), "node_deq.size() = ", node_labels_deq.size())
        if torch.isnan(rep_node).any():
            raise RuntimeError(
                'rep_node has NaN entries.')

        to_pred_edge_idx = data.pred_edge_st_ed_idx
        st_node_idx = to_pred_edge_idx[0, :]
        ed_node_idx = to_pred_edge_idx[1, :]
        rep_st_node = rep_node[st_node_idx, :]
        rep_ed_node = rep_node[ed_node_idx, :]

        pred_edge_nodes_rep = rep_node[data.pred_edge_node_idx, :]
        pred_edge_subgraph_rep = self.pool_func(pred_edge_nodes_rep, data.pred_edge_nodes_to_edge_idx, "sum")
        assert pred_edge_subgraph_rep.size(0) == rep_st_node.size(0) == rep_ed_node.size(0), "Reps to pred edges should be consistent with each other."

        # rep_subgraph_edges = rep_subgraph[to_pred_subgraph_idx, :]
        # print("connecting rep_edges", rep_subgraph_edges.size(), rep_st_node.size(), rep_ed_node.size())
        # rep_edges = torch.cat([rep_subgraph_edges, rep_st_node, rep_ed_node], dim=-1)
        rep_edges = torch.cat([pred_edge_subgraph_rep, rep_st_node, rep_ed_node], dim=-1)

        # st_node_idx = edge_index[0, :]
        # to_node_idx = edge_index[1, :]
        # rep_st_node = rep_node[st_node_idx, :]
        # rep_to_node = rep_node[to_node_idx, :]
        # rep_edges = torch.cat([rep_st_node, rep_to_node], dim=1)

        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](rep_subgraph)
            node_s = node_s.exp()
            # print(i, node_s.size(), node_t.size(), node_labels_deq.size())
            node_labels_deq = (node_labels_deq + node_t) * node_s

            if torch.isnan(node_labels_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()

            edge_s, edge_t = self.edge_st_net[i](rep_edges)
            edge_s = edge_s.exp()
            edge_labels_deq = (edge_labels_deq + edge_t) * edge_s

            if torch.isnan(edge_labels_deq).any():
                raise RuntimeError(
                    'adj_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                adj_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                adj_log_jacob += (torch.abs(edge_s) + 1e-20).log()

        # need we transform it into the batch form?
        assert node_labels_deq.size(0) == x_log_jacob.size(0)  # = total_node_num --- each node should be predicted
        assert edge_labels_deq.size(0) == adj_log_jacob.size(0) # = pred_edge_num

        return [node_labels_deq, edge_labels_deq], [x_log_jacob, adj_log_jacob]

    def pool_func(self, x, batch, mode="sum"):
        if mode == "sum":
            return global_add_pool(x, batch)
        elif mode == "mean":
            return global_mean_pool(x, batch)
        elif mode == "max":
            return global_max_pool(x, batch)

    def log_prob(self, z, log_det, batch_node, batch_edge):

        log_det[0] = self.pool_func(log_det[0], batch_node, "sum")
        bs = log_det[0].size(0)
        log_det[0] = log_det[0].view(bs, -1).sum(dim=-1)

        log_det[1] = self.pool_func(log_det[1], batch_edge, "sum")
        log_det[1] = log_det[1].view(bs, -1).sum(dim=-1)

        log_det[0] = log_det[0] - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        log_det[1] = log_det[1] - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = self.pool_func(ll_node, batch_node, "sum")
        ll_node = ll_node.view(bs, -1).sum(-1)
        # ll_node = ll_node.sum(-1)  # (B)

        ll_edge = -1 / 2 * (
                    torch.log(2 * self.constant_pi) + self.prior_ln_var_edge + torch.exp(-self.prior_ln_var_edge) * (z[1] ** 2))
        ll_edge = self.pool_func(ll_edge, batch_edge, "sum")
        ll_edge = ll_edge.view(bs, -1).sum(-1)
        # ll_edge = ll_edge.sum(-1)  # (B)
        print("loss to fit into prior = ", -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length)))
        ll_node += log_det[0]  # ([B])
        ll_edge += log_det[1]  # ([B])

        if self.deq_type == 'random':
            if self.divide_loss:

                return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))
            else:
                # ! useless
                return -torch.mean(ll_node + ll_edge)  # scalar
        else:
            raise NotImplementedError("deq_type must be random...")

    def _get_node_latent(self, x, edge_index, edge_attr, latent):   # one moleculer please...
        # x.size() = num_nodes x 2
        # edge_index.size() = 2 x num_edges
        # edge_attr.size() = num_edges x 2
        node_embs = self.GNN(x, edge_index, edge_attr)
        node_embs = torch.sum(node_embs, dim=0, keepdim=True)
        # not coherent with the calculation in the forward pass, where we just use the node embeddings to calculate
        # change the calculation method in the forward pass... ---- add pooling indexes is ok?

        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](node_embs)
            inv_node_s = torch.exp(-node_s)
            latent = latent * inv_node_s - node_t

        # latent = 1 x self.num_nodes
        node_type = torch.argmax(latent, dim=1, keepdim=False)
        return int(node_type[0].item())

    def _get_edge_latent(self, x, edge_index, edge_attr, st_ed_node, latent):
        node_embs = self.GNN(x, edge_index, edge_attr)
        st_node, ed_node = st_ed_node
        graph_emb = torch.sum(node_embs, keepdim=False, dim=0)
        edge_emb = torch.cat([graph_emb, node_embs[st_node, :], node_embs[ed_node, :]], dim=-1).view(1, -1)
        # edge_emb.size() = 1 x 2emb_dim

        for i in range(self.num_flow_layer):
            edge_s, edge_t = self.edge_st_net[i](edge_emb)
            inv_edge_s = torch.exp(-edge_s)
            latent = latent * inv_edge_s - edge_t

        edge_type = torch.argmax(latent, dim=1, keepdim=False)
        return int(edge_type[0].item())

    def reverse_flow(self, disturbed_latent, disturbed_latent_edges, max_nodes=38, T=0.75):
        device = disturbed_latent.device
        # please only one molecular...
        ori_nodes = disturbed_latent.size(0)

        # and we can only random sample the features for those nodes and edges exceeeded the max limit!!!!
        rand_first_node_emb = torch.distributions.normal.Normal(torch.zeros((1, self.emb_dim)).to(device),
                                         T * torch.ones((1, self.emb_dim)).to(device))
        first_node_latent = disturbed_latent[0, :].view(1, -1)

        current_node_features = torch.tensor([119, 0], dtype=torch.long, device=device).view(1, -1)
        current_edge_features = torch.empty((0, 2), dtype=torch.long, device=device)
        current_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        max_resample_num = 10

        for i in range(max_nodes):
            if (i < ori_nodes):
                node_latent = disturbed_latent[i, :].view(1, -1)
            else:
                node_latent = torch.distributions.normal.Normal(torch.zeros((1, self.num_node_type)).to(device),
                                         T * torch.ones((1, self.num_node_type)).to(device)).sample().view(1, -1)
            latent_node = self._get_node_latent(current_node_features, current_edge_index, current_edge_features, node_latent)
            current_node_features[-1, 0] = latent_node
            if i == 0:
                current_node_features = torch.cat([current_node_features, torch.tensor([119, 0], dtype=torch.long,
                                                                                       device=device).view(1, -1)], dim=0)
                continue
            is_connected = False
            for j in range(i):
                num_used_edges = ((i - 1) * i) // 2 + j

                valid = False
                num_resample = 0
                invalid_edge_type = set()
                while (num_resample < max_resample_num and (not valid) and len(invalid_edge_type) < self.num_edge_type) :
                    if (i < ori_nodes and num_resample == 0):
                        edge_latent = disturbed_latent_edges[num_used_edges, :]
                    else:
                        edge_latent = torch.distributions.normal.Normal(torch.zeros((1, self.num_edge_type + 1)).to(device),
                                             T * torch.ones((1, self.num_edge_type + 1)).to(device)).sample().view(1, -1)
                    latent_edge = self._get_edge_latent(current_node_features, current_edge_index, current_edge_features, [j, i], edge_latent)
                    if latent_edge == self.num_edge_type:
                        valid = True
                    else:
                        current_edge_index = torch.cat([current_edge_index, torch.tensor([j, i], device=device, dtype=torch.long).view(-1, 1)], dim=1)
                        current_edge_index = torch.cat([current_edge_index, torch.tensor([i, j], device=device, dtype=torch.long).view(-1, 1)], dim=1)
                        current_edge_features = torch.cat([current_edge_features, torch.tensor([latent_edge, 0], device=device, dtype=torch.long).view(1, -1)], dim=0)
                        current_edge_features = torch.cat([current_edge_features, torch.tensor([latent_edge, 0], device=device, dtype=torch.long).view(1, -1)], dim=0)
                        mol = graph_data_obj_to_mol_simple(current_node_features, current_edge_index, current_edge_features)

                        # new_added_edge_index.append(torch.tensor([j, i], dtype=torch.long, device=device).view(-1, 1))
                        # new_added_edge_attr.append(torch.tensor([edge_discrete_id, 0], dtype=torch.long, device=device).view(1, -1))
                        # nei = torch.cat(new_added_edge_index, dim=1)
                        # nea = torch.cat(new_added_edge_attr, dim=0)
                        # print("generating mol adj with num_nodes =", current_node_features.size(0))
                        # mol = graph_data_obj_to_mol_simple(current_node_features[:-1, :],
                        #                                    torch.cat([current_edge_index, nei], dim=1),
                        #                                    torch.cat([current_edge_attr, nea], dim=0))

                        valid = env.check_valency(mol)
                        # print(valid)
                        if valid:
                            is_connected = True
                            # print(num2bond_symbol[edge_discrete_id])
                        else:  # backtrack
                            current_edge_features = current_edge_features[:-2, :]
                            current_edge_index = current_edge_index[:, :-2]

                            # new_added_edge_index.pop()
                            # new_added_edge_attr.pop()
                            num_resample += 1.0

                            invalid_edge_type.add(latent_edge)
                if valid == False:
                    latent_edge = self.num_edge_type
            if is_connected == False:
                current_node_features = current_node_features[:-1, :]
                break
            current_node_features = torch.cat([current_node_features, torch.tensor([119, 0], dtype=torch.long,
                                                                                       device=device).view(1, -1)], dim=0)
        print(current_node_features.size(0), disturbed_latent.size(0))
        # data = Data()
        # data.x = current_node_features
        # data.edge_index = current_edge_index
        # data.edge_attr = current_edge_features
        # return data
        return current_node_features, current_edge_index, current_edge_features

        # use the model in cpu and add the generated data to the current data's attributes...
        # but... run the model in cpu?????

        # for i in range(self.num_flow_layer):
        #     node_s, node_t = self.node_st_net[i](rand_first_node_emb)
        #     inv_node_s = torch.exp(-node_s)
        #     first_node_latent = first_node_latent * inv_node_s - node_t
        # first_node_type = torch.argmax(first_node_latent, dim=1, keepdim=False)[0].item()
        # if first_node_type == self.num_node_type - 1:
        #     return None # the first node is no node...
        # x = torch.zeros((1, 2), dtype=torch.long, device=device)
        # x[0, 0] = first_node_type
        # edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        # edge_attr = torch.empty((0, 2), dtype=torch.long, device=device)
        #
        # # start generate nodes
        # num_atom = 1
        # for i in range(1, max_nodes + 1):
        #     if i < disturbed_latent.size(0):
        #         latent_node = disturbed_latent[i, :].view(1, -1)
        #     else:
        #         latent_node = torch.distributions.normal.Normal(torch.zeros((1, self.emb_dim)).to(device),
        #                                  T * torch.ones((1, self.emb_dim)).to(device))
        #     x = torch.cat([x, torch.tensor([self.num_node_type, 0], dtype=torch.long, device=device).view(1, -1)], dim=0)
        #     node_type = self._get_node_latent(x, edge_index, edge_attr, latent_node)
        #     if (node_type == self.num_node_type - 1):
        #         break # get a no-type node
        #     # x[i, 0] = node_type
        #     num_atom += 1
        #
        #     latent_edge = torch.distributions.normal.Normal(torch.zeros((1, 2 * self.emb_dim)).to(device),
        #                                                     T * torch.ones((1, 2 * self.emb_dim)).to(device))
        #
        #     # generate edges
        #     to_be_added_edge_index_list = list()
        #     to_be_added_edge_attr_list = list()
        #     has_cont = False
        #     for prev_idx in range(i):
        #         edge_type = self._get_edge_latent(x, edge_index, edge_attr, [prev_idx, i], latent_edge)
        #         if edge_type != 0:
        #             has_cont = True
        #             edge_index = torch.cat([edge_index,
        #                                     torch.tensor([prev_idx, i], dtype=torch.long, device=device).view(-1, 1)], dim=1)
        #             edge_attr = torch.cat([edge_attr,
        #                                    torch.tensor([edge_type, 0], dtype=torch.long, device=device).view(1, -1)], dim=0)
        #             to_be_added_edge_index_list.append(torch.tensor([i, prev_idx], dtype=torch.long, device=device).view(-1, 1))
        #             to_be_added_edge_attr_list.append(torch.tensor([edge_type, 0], dtype=torch.long, device=device).view(1, -1))
        #     to_be_added_edge_index = torch.cat(to_be_added_edge_index_list, dim=1)
        #     to_be_added_edge_attr = torch.cat(to_be_added_edge_attr_list, dim=0)
        #     edge_index = torch.cat([edge_index, to_be_added_edge_index], dim=1)
        #     edge_attr = torch.cat([edge_attr, to_be_added_edge_attr], dim=0)
        #     x[i, 0] = node_type
        #
        #     if has_cont == False:
        #         break # node has no adjacent nodes break;
        # return x, edge_index, edge_attr

    def generate_one_mole_given_original_mole(self, data, disturb_factor=0.1):
        edge_labels = data.pred_edge_attr[:, 0]
        node_labels = data.x[:, 0]

        x_deq = self.dequantinization(node_labels, one_hot=False, hot_dim=self.num_node_type)
        edge_labels_deq = self.dequantinization(edge_labels, one_hot=False, hot_dim=self.num_edge_type + 1)

        z, log_det = self.flow_forward(data, x_deq, edge_labels_deq)

        node_latent = z[0]
        edge_latent = z[1]

        disturbed_latent = node_latent + disturb_factor * torch.randn(node_latent.size(), device=data.x.device)
        disturbed_latent_edge = edge_latent + disturb_factor * torch.randn(edge_latent.size(), device=data.x.device)


        x, edge_index, edge_attr = self.reverse_flow(disturbed_latent, disturbed_latent_edge)
        ##### already dual??

        # dual_edge_index = list()
        # dual_edge_attr = list()
        # for i in range(edge_index.size(1)):
        #     a, b = int(edge_index[0, i]), int(edge_index[1, i])
        #     dual_edge_index.append(torch.tensor([a, b], device=ori_x.device, dtype=torch.long).view(-1, 1))
        #     dual_edge_index.append(torch.tensor([b, a], device=ori_x.device, dtype=torch.long).view(-1, 1))
        #     dual_edge_attr.append(torch.tensor(edge_attr[i, :].view(1, -1)))
        #     dual_edge_attr.append(torch.tensor(edge_attr[i, :].view(1, -1)))
        # edge_index = torch.cat(dual_edge_index, dim=1)
        # edge_attr = torch.cat(dual_edge_attr, dim=0)
        return x, edge_index, edge_attr



if __name__ == "__main__":
    pass

