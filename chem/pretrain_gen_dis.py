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

from util import MaskAtom7

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def train(args, model_list, loader, optimizer_list, device):
    model_gen, model_dis, linear_pred_atoms_gen, linear_pred_atoms_dis, linear_pred_bonds = model_list
    optimizer_model_gen, optimizer_model_dis, optimizer_linear_pred_atoms_gen, optimizer_linear_pred_atoms_dis, \
    optimizer_linear_pred_bonds = optimizer_list

    model_gen.train()
    model_dis.train()
    linear_pred_atoms_gen.train()
    linear_pred_atoms_dis.train()
    linear_pred_bonds.train()

    loss_accum_gen = 0
    loss_accum_dis = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    # todo edges are not masked yet..
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep = model_gen(batch.masked_x, batch.edge_index, batch.edge_attr)

        ## loss for nodes
        pred_node = linear_pred_atoms_gen(node_rep[batch.masked_atom_indices])
        loss_gen = criterion(pred_node.double(), batch.mask_node_label[:, 0])

        prob = F.softmax(pred_node, dim=1)
        samples = torch.multinomial(prob, 1).squeeze(-1)
        # print(samples.size())

        batch.masked_x = batch.x
        batch.masked_x[batch.masked_atom_indices, 0] = samples
        # target = batch.masked_x[:, 0] == batch.x[:, 0]
        target = torch.where(batch.masked_x[:, 0] == batch.x[:, 0], torch.ones(batch.x.size(0)).to(batch.x.device),
                             torch.zeros(batch.x.size(0)).to(batch.x.device))
        target = target.long()
        # print(target.size())
        # print(target[:100])

        node_rep = model_dis(batch.masked_x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms_dis(node_rep)
        # loss_dis = F.binary_cross_entropy_with_logits(pred_node.double(), target) # output == 1?
        loss_dis = criterion(pred_node.double(), target) # equal to the BCEWithLogistLoss
        pred_node_typ = linear_pred_atoms_gen(node_rep)
        loss_dis += criterion(pred_node_typ, batch.x[:, 0])

        # indices ?

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

        optimizer_model_gen.zero_grad()
        optimizer_model_dis.zero_grad()
        optimizer_linear_pred_atoms_gen.zero_grad()
        optimizer_linear_pred_atoms_dis.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss_gen.backward()
        loss_dis.backward()

        optimizer_model_gen.step()
        optimizer_model_dis.step()
        optimizer_linear_pred_atoms_gen.step()
        optimizer_linear_pred_atoms_dis.step()
        optimizer_linear_pred_bonds.step()

        loss_accum_gen += float(loss_gen.cpu().item())
        loss_accum_dis += float(loss_dis.cpu().item())
        print("generator loss: {}, discriminator loss: {}".format(str(loss_gen.item()), str(loss_dis.item())))

    return loss_accum_gen / step, loss_accum_dis / step, acc_node_accum / step, acc_edge_accum / step


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
    args = parser.parse_args()

    args.output_model_file = "gen_dis_gnn_use_infomax_{}_mask_edge_{}_minus_{:d}".format(str(args.use_infomax), str(args.mask_edge), args.minus)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=MaskAtom7(num_atom_type=119, num_edge_type=5, mask_rate=args.mask_rate,
                                                 mask_edge=args.mask_edge))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model_gen = GNN(max(args.num_layer - args.minus, 1), args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    model_dis = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    linear_pred_atoms_gen = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_atoms_dis = torch.nn.Linear(args.emb_dim, 2).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    model_list = [model_gen, model_dis, linear_pred_atoms_gen, linear_pred_atoms_dis, linear_pred_bonds]

    # set up optimizers
    optimizer_model_gen = optim.Adam(model_gen.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms_gen = optim.Adam(linear_pred_atoms_gen.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model_gen, optimizer_model_dis, optimizer_linear_pred_atoms_gen,
                      optimizer_linear_pred_atoms_dis, optimizer_linear_pred_bonds]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss_gen, train_loss_dis, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        print(train_loss_gen, train_loss_dis, train_acc_atom, train_acc_bond)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model_dis.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model_dis.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()
