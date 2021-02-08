import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderMasking  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import structural_pair_extract

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_atoms, linear_pred_bonds, linear_pred_bonds_exi = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_bonds_exi = optimizer_list

    model.train()
    # model_dis.train()
    linear_pred_atoms.train()
    # linear_pred_atoms_dis.train()
    linear_pred_bonds.train()

    loss_accum = 0
    # loss_accum_dis = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)
        # todo add chil-pred?
        # todo add denoise reconstruct --- the whole graph
        loss = criterion(pred_node.double(), batch.x[:, 0])

        edge_rep_connected = node_rep[batch.sample_connected_indices[0, :], :] + \
                             node_rep[batch.sample_connected_indices[1, :], :]
        edge_rep_unconnected = node_rep[batch.sample_unconnected_indices[0, :], :] + \
                             node_rep[batch.sample_unconnected_indices[1, :], :]
        # edge_rep = node_rep[batch.edge_index[0, :], :] + node_rep[batch.edge_index[1, :], :]
        pred_edge = linear_pred_bonds(edge_rep_connected)
        connected_labels = batch.sample_connected_edge_attr[:, 0]
        loss += F.cross_entropy(pred_edge.double(), connected_labels)

        pred_edge_connected = linear_pred_bonds_exi(edge_rep_connected)
        pred_edge_unconnected = linear_pred_bonds_exi(edge_rep_unconnected)
        loss += F.binary_cross_entropy_with_logits(pred_edge_connected, torch.ones_like(pred_edge_connected)) + \
                F.binary_cross_entropy_with_logits(pred_edge_unconnected, torch.zeros_like(pred_edge_unconnected))
        # loss += criterion(pred_edge.double(), batch.edge_attr[:, 0])

        if args.mask_atom == True:
            masked_node_rep = node_rep[batch.masked_atom_indices, :]
            pred_atom = linear_pred_atoms(masked_node_rep)
            loss = loss + F.cross_entropy(pred_atom, batch.mask_node_label[:, 0])

        optimizer_model.zero_grad()
        # optimizer_model_dis.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        # optimizer_linear_pred_atoms_dis.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()
        optimizer_linear_pred_bonds_exi.zero_grad()

        loss.backward()
        # loss_dis.backward()

        optimizer_model.step()
        # optimizer_model_dis.step()
        optimizer_linear_pred_atoms.step()
        # optimizer_linear_pred_atoms_dis.step()
        optimizer_linear_pred_bonds.step()
        optimizer_linear_pred_bonds_exi.step()

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
    parser.add_argument('--sample_connected_edge_ratio', type=float, default=0.15, help='number of workers for dataset loading')
    parser.add_argument('--sample_unconnected_edge_ratio', type=float, default=0.30, help='number of workers for dataset loading')
    parser.add_argument('--mask_atom', action="store_true", default=False, help='number of workers for dataset loading')

    args = parser.parse_args()

    args.output_model_file = "temp/no_mask_struct_reconstruct_2_mask_atom_{}".format(str(args.mask_atom))
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
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_atoms_dis = torch.nn.Linear(args.emb_dim, 2).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)
    linear_pred_bonds_exi = torch.nn.Sequential(
        torch.nn.Linear(args.emb_dim, 1),
        nn.Sigmoid()
    ).to(device)

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=structural_pair_extract(pos_sample_ratio=args.sample_connected_edge_ratio,
                                                                neg_sample_ratio=args.sample_unconnected_edge_ratio, mask_atom=args.mask_atom))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model_list = [model, linear_pred_atoms, linear_pred_bonds, linear_pred_bonds_exi]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds_exi = optim.Adam(linear_pred_bonds_exi.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_bonds_exi]

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
