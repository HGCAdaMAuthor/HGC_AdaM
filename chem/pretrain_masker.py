import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderMasking  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import mask_strategies

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(MLPLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )
        self.init_weight()

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.net(x).view(-1)


class Makser(nn.Module):
    def __init__(self, args):
        super(Makser, self).__init__()
        self.gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        self.mlp = MLPLayer(args.emb_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        node_rep = self.gnn(x, edge_index, edge_attr)
        prob = self.mlp(node_rep)
        return prob

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


import time
def train(args, model_list, loader, optimizer_list, device):
    model, masker, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_masker, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list

    model.train()
    # model_dis.train()
    linear_pred_atoms.train()
    # linear_pred_atoms_dis.train()
    linear_pred_bonds.train()
    masker.train()

    loss_accum = 0
    # loss_accum_dis = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    # todo edges are not masked yet..
    # todo add global information alignment?
    # todo directly change one type to another type is not that reasonable? since we the let it to predict the true
    #  label, too much disturbation of the graph structure?

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        num_nodes = batch.x.size(0)
        masked_num_nodes = int(num_nodes * args.mask_rate + 1)

        # batch num_moleculor moleculor_nodes moleculor_edges

        # batch.masked_nodes = batch.x
        # node_labels = batch.x[:, 0].copy()
        node_labels = torch.zeros((batch.x.size(0), ), dtype=torch.long).to(batch.x.device)
        node_labels[:] = batch.x[:, 0]
        # print(torch.max(node_labels))

        st_tim = time.time()

        with torch.no_grad():
            bef_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
            bef_node_pred_prob = torch.softmax(linear_pred_atoms(bef_node_rep), dim=-1)


        if args.moleculor_mask == True:
            for j in range(batch.num_moleculor):
                st_nodes, ed_nodes = int(batch.moleculor_nodes[j]), int(batch.moleculor_nodes[j + 1])
                st_edges, ed_edges = int(batch.moleculor_edges[j]), int(batch.moleculor_edges[j + 1])
                moleculor_x = batch.x[st_nodes: ed_nodes, :]
                moleculor_edges = batch.edge_index[:, st_edges: ed_edges] - st_nodes
                moleculor_edge_attr = batch.edge_attr[st_edges: ed_edges, :]
                nodes_masked_prob = masker(moleculor_x, moleculor_edges, moleculor_edge_attr)
                # num_nodes
                num_masked = int((ed_nodes - st_nodes) * args.mask_rate + 1)
                masked_nodes = torch.argsort(nodes_masked_prob, dim=0, descending=True)[: num_masked]
                batch.x[st_nodes + masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=batch.x.device)
        else:
            num_masked = int(batch.x.size(0) * args.mask_rate + 1)

            nodes_masked_prob = masker(batch.x, batch.edge_index, batch.edge_attr)
            # print(nodes_masked_prob)
            masked_nodes = torch.argsort(nodes_masked_prob, dim=0, descending=True)[: num_masked]
            # print(masked_nodes)
            batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=batch.x.device)

        ed_time = time.time()
        # print("time", ed_time - st_tim)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)
        # print(torch.max(node_labels))
        loss = criterion(pred_node.double(), node_labels)

        # so the masker aims to maximize the entropy between the prediction distribution between previous probability and
        # post mask probability; which is equal to minimize the negative of the entropy
        aft_node_prob = torch.softmax(pred_node, dim=-1)
        neg_entropy = bef_node_pred_prob * torch.log(torch.clamp(aft_node_prob, min=1e-12))
        neg_entropy = torch.mean(torch.sum(neg_entropy, dim=-1, keepdim=False), dim=0)

        if args.masker_loss_type == "entropy":
            masker_loss = neg_entropy
            print("masker loss = ", neg_entropy)
        else:
            masker_loss = -loss
        # acc_node = compute_accuracy(pred_node, batch.mask_node_label[:, 0])
        # acc_node_accum += acc_node
        #
        # if args.mask_edge:
        #     masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
        #     edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        #     pred_edge = linear_pred_bonds(edge_rep)
        #     loss += criterion(pred_edge.double(), batch.mask_edge_label[:, 0])
        #
        #     acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:, 0])
        #     acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_masker.zero_grad()
        # optimizer_model_dis.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        # optimizer_linear_pred_atoms_dis.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        st_time = time.time()
        loss.backward(retain_graph=True)
        ed_time = time.time()
        # print("first_backward_time", ed_time - st_time)

        optimizer_model.step()
        # optimizer_model_dis.step()
        optimizer_linear_pred_atoms.step()
        # optimizer_linear_pred_atoms_dis.step()
        optimizer_linear_pred_bonds.step()

        st_time = time.time()
        masker_loss.backward()
        ed_time = time.time()
        # print("second_backward_time", ed_time - st_time)
        # loss_dis.backward()

        optimizer_masker.step()

        loss_accum += float(loss.cpu().item())
        # loss_accum_dis += float(loss_dis.cpu().item())
        print(loss.item())

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
    parser.add_argument('--use_infomax', action="store_true", default=False, help='number of workers for dataset loading')
    parser.add_argument('--minus', type=int, default=2, help='number of workers for dataset loading')
    parser.add_argument('--moleculor_mask', type=bool, default=False, help='number of workers for dataset loading')
    parser.add_argument('--masker-loss-type', type=str, default="entropy", help='number of workers for dataset loading')
    args = parser.parse_args()

    args.output_model_file = "temp/masker_moleculor_mask_{}_masker_type_{}".format(str(args.moleculor_mask),
                                                                                   args.masker_loss_type)
    print(args.output_model_file)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.


    # set up models, one for pre-training and one for context embeddings
    # model_gen = GNN(max(args.num_layer - args.minus, 1), args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
    #     device)
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    masker = Makser(args).to(device)

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_atoms_dis = torch.nn.Linear(args.emb_dim, 2).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=None)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model_list = [model, masker, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_masker = optim.Adam(masker.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_masker, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

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
