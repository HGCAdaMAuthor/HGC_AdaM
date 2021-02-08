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
        self.gnn_2 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        self.mlp = MLPLayer(args.emb_dim, 1)
        self.mlp_2 = MLPLayer(args.emb_dim, 1)

    def forward_a(self, x, edge_index, edge_attr):
        node_rep = self.gnn(x, edge_index, edge_attr)
        prob = self.mlp(node_rep)
        return prob

    def forward_b(self, x, edge_index, edge_attr):
        node_rep = self.gnn_2(x, edge_index, edge_attr)
        prob = self.mlp_2(node_rep)
        return prob

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


import time
def train(args, model_list, loader, optimizer_list, device, epoch):
    model, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list

    LARGE_NUM = 1e12
    T = 0.07
    model.train()
    # model_dis.train()
    linear_pred_atoms.train()
    # linear_pred_atoms_dis.train()
    linear_pred_bonds.train()
    # masker.train()

    loss_accum = 0
    # loss_accum_dis = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    st_idx = 0
    ed_idx = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        num_nodes = batch.x.size(0)
        device = batch.x.device
        ed_idx = st_idx + num_nodes

        node_labels = torch.zeros((batch.x.size(0), ), dtype=torch.long).to(batch.x.device)
        node_labels[:] = batch.x[:, 0]
        # print(torch.max(node_labels))

        tot_masked = int(batch.x.size(0) * args.mask_rate + 1)

        with torch.no_grad():
            node_rep_no_mask = model(batch.x, batch.edge_index, batch.edge_attr)
            node_pred_prob_no_mask = torch.softmax(linear_pred_atoms(node_rep_no_mask).double(), dim=-1)
        num_masked = masked_times[st_idx: ed_idx].clone()
        profit_masked = masked_score[st_idx: ed_idx].clone()
        profit_masked = profit_masked + 0.95 * torch.sqrt(
            torch.log(torch.tensor([epoch], device=device, dtype=torch.float64)) / num_masked)
        masked_prob = torch.softmax(profit_masked / T, dim=0)
        masked_nodes = torch.multinomial(masked_prob, tot_masked, replacement=False)
        batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)

        # print(torch.max(node_labels))
        loss = criterion(pred_node.double(), node_labels)

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward(retain_graph=True)

        optimizer_model.step()
        # optimizer_model_dis.step()
        optimizer_linear_pred_atoms.step()
        # optimizer_linear_pred_atoms_dis.step()
        optimizer_linear_pred_bonds.step()

        with torch.no_grad():
            node_pred_prob = torch.softmax(pred_node.double(), dim=-1)
            entropy = -torch.sum(node_pred_prob_no_mask * torch.log(torch.clamp(
                node_pred_prob, min=1e-12
            )), dim=-1)
            mean = torch.mean(entropy)
            entropy -= mean
            masked_score[st_idx: ed_idx] += entropy
            masked_times[st_idx: ed_idx] += 1.0

        loss_accum += float(loss.cpu().item())
        # loss_accum_dis += float(loss_dis.cpu().item())
        print(loss.item(), mean)
        st_idx = ed_idx

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
    parser.add_argument('--mask_times', type=int, default=3, help='number of workers for dataset loading')
    parser.add_argument('--several-times-mask', type=bool, default=False, help='number of workers for dataset loading')

    args = parser.parse_args()

    args.output_model_file = "temp/masker_prob_{:d}".format(args.mask_times)
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
    # masker = Makser(args).to(device)
    # masker_2 = Makser(args).to(device)

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_atoms_dis = torch.nn.Linear(args.emb_dim, 2).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)


    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=None)

    global masked_times
    masked_times= torch.ones((len(dataset) * 50, ), dtype=torch.float64, device=device, requires_grad=False)

    global masked_score
    masked_score= torch.zeros((len(dataset) * 50, ), dtype=torch.float64, device=device, requires_grad=False)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_masker = optim.Adam(masker.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device, epoch)
        print(train_loss, train_acc_atom, train_acc_bond)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
