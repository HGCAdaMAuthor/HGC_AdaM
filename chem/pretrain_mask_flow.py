import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderMaskSubstruct  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
# from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskSubstruct

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
from torch_geometric.nn.inits import uniform

criterion = nn.CrossEntropyLoss()

import timeit

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


import time
def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_atoms, linear_pred_bonds, model_dis = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_dis = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    model_dis.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep = model(batch.x, batch.edge_indices_masked, batch.edge_attr_masked, add_self_loop=False)
        # block self loop information propagation

        # pooled_masked_node_rep = []
        #
        # num_masked_nodes = batch.cut_order_indices.size(0)
        # st_tim = time.time()

        # a = node_rep[batch.cut_order_indices[:, 0]: batch.cut_order_indices[:, 1], :]
        # print(a.size())
        # for i in range(num_masked_nodes):
        #     fr, to = int(batch.cut_order_indices[i, 0]), int(batch.cut_order_indices[i, 1])
        #     subgraph_rep = node_rep[fr: to, :]
        #     if len(subgraph_rep.size()) > 1:
        #         pooled_subgraph_rep = torch.sum(subgraph_rep, dim=0, keepdim=True)
        #     else:
        #         pooled_subgraph_rep = subgraph_rep.unsqueeze(0)
        #     pooled_masked_node_rep.append(pooled_subgraph_rep)
        # ed_tim = time.time()
        # print("pool time", ed_tim - st_tim)
        # pooled_masked_node_rep = torch.cat(pooled_masked_node_rep, dim=0)

        pooled_masked_node_rep = node_rep[batch.mask_node_indices, :]

        assert pooled_masked_node_rep.size(0) == batch.masked_atom_labels.size(0)
        pred_node = linear_pred_atoms(pooled_masked_node_rep)
        loss = criterion(pred_node.double(), batch.masked_atom_labels[:, 0])

        # todo add use infomax ?
        if args.use_infomax:
            summary_emb = torch.sigmoid(global_mean_pool(node_rep, batch.batch))

            positive_expanded_summary_emb = summary_emb[batch.batch]

            shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)]
            negative_expanded_summary_emb = shifted_summary_emb[batch.batch]

            positive_score = model_dis(node_rep, positive_expanded_summary_emb)
            negative_score = model_dis(node_rep, negative_expanded_summary_emb)
            loss = loss + F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score)) + \
                F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))

            ## loss for nodes
        # pred_node = linear_pred_atoms(node_rep[batch.masked_atom_indices])
        # loss = criterion(pred_node.double(), batch.mask_node_label[:, 0])
        #
        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:, 0])
        # acc_node_accum += acc_node

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = linear_pred_bonds(edge_rep)
            loss += criterion(pred_edge.double(), batch.masked_edge_labels[:, 0])

            # acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:, 0])
            # acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()
        optimizer_dis.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()
        optimizer_dis.step()

        print(loss.item())
        loss_accum += float(loss.cpu().item())

    return loss_accum / step, acc_node_accum / step, acc_edge_accum / step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--use_infomax', default=False, action="store_true", help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    args.output_model_file = "mask_flow_use_infomax_{}_mask_edge_{}".format(str(args.use_infomax), str(args.mask_edge))
    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=MaskSubstruct(num_atom_type=119, num_edge_type=5, mask_rate=args.mask_rate,
                                                 mask_edge=args.mask_edge))

    loader = DataLoaderMaskSubstruct(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)
    model_dis = Discriminator(args.emb_dim).to(device)

    model_list = [model, linear_pred_atoms, linear_pred_bonds, model_dis]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_dis]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
