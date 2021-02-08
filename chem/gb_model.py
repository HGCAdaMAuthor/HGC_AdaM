import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from utils2 import filter_attri_from_batch
import torch.nn as nn

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

# add node embedding but... ignore
# todo how can we utilize edge embeddings ?
# context bs x k : indices; wl embedding bs x wl_dim; relative position embedding k x rela_dim; and raw features, cat!
# G-transformers

class graph_transformer_layer(nn.Module):
    def __init__(self, feature_dim, residue="raw"):
        super(graph_transformer_layer, self).__init__()
        self.wq = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.wk = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.wv = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.fea_dim = feature_dim
        self.res = residue
    def forward(self, x, raw_x):
        # x: bs x (k + 1) x fea_dim
        qx = torch.matmul(x, self.wq)
        kx = torch.matmul(x, self.wk)
        vx = torch.matmul(x, self.wv)
        if self.res == "raw":
            res_x = raw_x
        elif self.res == "ori":
            res_x = x
        else:
            raise NotImplementedError("aa")
        logists = torch.matmul(qx, kx.t()) / torch.sqrt(self.fea_dim)
        logists = F.softmax(logists)
        transfered_x = logists * vx
        return transfered_x + res_x

class graph_bert_model(nn.Module):
    def __init__(self, feature_dim, residue="raw", n_layers=7, k=5):
        super(graph_bert_model, self).__init__()

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, feature_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, feature_dim)
        self.relative_pos_emb = torch.nn.Embedding((k+1), feature_dim)

        self.raw_trans = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.wl_trans = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.pos_trans = nn.Parameter(torch.randn(feature_dim, feature_dim))
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.relative_pos_emb.weight.data)
        torch.nn.init.xavier_uniform_(self.raw_trans.weight.data)
        torch.nn.init.xavier_uniform_(self.wl_trans.weight.data)
        torch.nn.init.xavier_uniform_(self.pos_trans.weight.data)


        self.transformer_stack = nn.ModuleList()
        for j in range(n_layers):
            self.transformer_stack.append(graph_transformer_layer(feature_dim, residue))

    def forward(self, indicies_idx, indices_pos, raw_fea, wl_fea):
        batch_raw = raw_fea[indicies_idx]
        print(batch_raw.size()) # can view also
        batch_pos_emb = self.relative_pos_emb[indices_pos]
        batch_wl_emb = wl_fea[indicies_idx]
        batch_x = torch.matmul(batch_raw, self.raw_trans) + torch.matmul(batch_pos_emb, self.pos_trans) + torch.matmul(batch_wl_emb, self.wl_trans)
        for layer in self.transformer_stack:
            batch_x = layer(batch_x, batch_raw)
            batch_x = F.layer_norm(batch_x, batch_x.size()[-1:])
        return batch_x


