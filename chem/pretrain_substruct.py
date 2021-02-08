import argparse

from loader import MoleculeDataset, SubstructDataset
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

from util import MaskAtom, SubstructMask

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()
criterion_structure = nn.BCEWithLogitsLoss()

import timeit

def pool_func(x, batch, mode = "sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def train(args, model_list, loader, optimizer_list, device, epoch):
    model, model_corupt, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_model_corupt, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list
    mask_obj = 0
    loader.dataset.mask_obj = 0
    #if epoch % 2 == 0:
    #
    #else:
    #    loader.dataset.mask_obj = 1
    #    mask_obj = 1
    model.train()
    model_corupt.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        corupt_rep = model_corupt(batch.mask_x, batch.edge_index, batch.mask_edge_attr)

        ## loss for nodes
        if mask_obj == 0:
            pred_node = linear_pred_atoms(corupt_rep[batch.masked_atom_indices])
            loss = criterion(pred_node.double(), batch.mask_node_label[:, 0])

            acc_node = compute_accuracy(pred_node, batch.mask_node_label[:, 0])
            acc_node_accum += acc_node
        else:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = corupt_rep[masked_edge_index[0]] + corupt_rep[masked_edge_index[1]]
            pred_edge = linear_pred_bonds(edge_rep)
            loss = criterion(pred_edge.double(), batch.mask_edge_label[:, 0])

            acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:, 0])
            acc_edge_accum += acc_edge

        corupt_global_rep = pool_func(corupt_rep, batch.node_idx_tensor, mode=args.pooling)

        ## loss for structure
        if args.mode == "cbow":
            # positive context representation
            complete_rep = pool_func(node_rep, batch.node_idx_tensor, mode=args.pooling)

            # context_rep = pool_func(overlapped_node_rep, batch.batch_overlapped_context, mode=args.context_pooling)
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_complete_rep = torch.cat(
                [complete_rep[cycle_index(len(complete_rep), i + 1)] for i in range(args.neg_samples)], dim=0)

            pred_pos = torch.sum(corupt_global_rep * complete_rep, dim=1)
            pred_neg = torch.sum(corupt_global_rep.repeat((args.neg_samples, 1)) * neg_complete_rep, dim=1)

        elif args.mode == "skipgram":

            expanded_substruct_rep = torch.cat(
                [corupt_global_rep[i].repeat((batch.num_atoms_size[i], 1)) for i in range(len(corupt_global_rep))],
                dim=0)
            pred_pos = torch.sum(expanded_substruct_rep * node_rep, dim=1)

            # shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = corupt_global_rep[cycle_index(len(corupt_global_rep), i + 1)]
                shifted_expanded_substruct_rep.append(torch.cat(
                    [shifted_substruct_rep[i].repeat((batch.num_atoms_size[i], 1)) for i in
                     range(len(shifted_substruct_rep))], dim=0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim=0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * node_rep.repeat((args.neg_samples, 1)),
                                 dim=1)
        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion_structure(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion_structure(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        #loss += loss_pos + args.neg_samples*loss_neg

        #optimizer_model.zero_grad()
        #optimizer_model_corupt.zero_grad()
        if mask_obj == 0:
            optimizer_linear_pred_atoms.zero_grad()
        else:
            optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        #optimizer_model.step()
        #optimizer_model_corupt.step()
        if mask_obj == 0:
            optimizer_linear_pred_atoms.step()
        else:
            optimizer_linear_pred_bonds.step()

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
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--mode', type=str, default="cbow", help="cbow or skipgram")
    parser.add_argument('--output_model_file', type=str, default='temp/substruct', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.
    dataset = SubstructDataset("dataset/" + args.dataset, dataset=args.dataset,
                               transform=SubstructMask(k=2, num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    # dataset_corupt = SubstructDataset("dataset/" + args.dataset, dataset=args.dataset,
                                     # transform=SubstructMask(k=2, num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))

    # about shuffle?? --- how to shuffle those two dataset??
    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # loader_corupt = DataLoaderMasking(dataset_corupt, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    model_corupt = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    model_list = [model, model_corupt, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_model_corupt = optim.Adam(model_corupt.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_model_corupt, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device, epoch)
        print(train_loss, train_acc_atom, train_acc_bond)
# 1.62 0.842
    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
