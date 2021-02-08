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

from util import mask_strategies

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit

num_type_atoms = 119

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


# num_nodes = batch.x.size(0)
#         masked_num_nodes = int(num_nodes * args.mask_rate + 1)
#         x_split, edge_index_split, edge_attr_split = int(batch.x_split), int(batch.edge_index_split), int(batch.edge_attr_split)
#         # train masker
#         with torch.no_grad():

def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list

    model.train()
    # model_dis.train()
    linear_pred_atoms.train()
    # linear_pred_atoms_dis.train()
    linear_pred_bonds.train()

    loss_accum = 0
    # loss_accum_dis = 0
    acc_node_accum = 0
    T = 0.07
    acc_edge_accum = 0

    # todo edges are not masked yet..
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        num_nodes = batch.x.size(0)
        masked_num_nodes = int(num_nodes * args.mask_rate + 1)

        if args.mask_stra == "variance_based":
            with torch.no_grad():
                node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                pred_node = linear_pred_atoms(node_rep)
                var_pred = torch.var(pred_node, dim=-1)
                var_pred_ordered = torch.argsort(var_pred, dim=0, descending=True)
                masked_nodes_indices = var_pred_ordered[: masked_num_nodes].long()
                masked_labels = batch.x[masked_nodes_indices, 0]
                batch.x[masked_nodes_indices, :] = torch.tensor([119, 0], dtype=torch.long, device=batch.x.device)

        elif args.mask_stra == "meta_based":
            one_hot_attr = torch.zeros((batch.x.size(0), num_type_atoms + 1), requires_grad=False, device=batch.x.device)
            one_hot_attr[torch.arange(batch.x.size(0), device=batch.x.device), batch.x[:, 0]] = 1.
            one_hot_attr.requires_grad = True
            node_rep = model(batch.x, batch.edge_index, batch.edge_attr, one_hot_attr=one_hot_attr)
            pred_node = linear_pred_atoms(node_rep)
            loss = criterion(pred_node.double(), batch.x[:, 0])

            loss.backward()
            delta_x_one_hot = one_hot_attr.grad
            s = (-one_hot_attr * 2.0 + 1.0) * delta_x_one_hot
            s = s + s[torch.arange(batch.x.size(0), device=batch.x.device), batch.x[:, 0]].view(-1, 1)
            s[torch.arange(batch.x.size(0), device=batch.x.device), batch.x[:, 0]] = -args.inf
            # index_max = torch.argmax(s, dim=1)
            s, index_max = torch.max(s, dim=1)
            # print(index_max)
            # print(s[torch.arange(batch.x.size(0), device=batch.x.device)])
            ordered_s_index = s.argsort(dim=0, descending=True)
            num_masked_nodes = int(batch.x.size(0) * args.mask_rate + 1)
            masked_nodes_indices = ordered_s_index[: num_masked_nodes]
            masked_labels = batch.x[masked_nodes_indices, 0]
            batch.x[masked_nodes_indices, 0] = index_max[masked_nodes_indices]
            batch.x[masked_nodes_indices, 1] = 0

        elif args.mask_stra == "meta_based_v2":
            mask_iters = 5 if args.mask_times is None else args.mask_times
            with torch.no_grad():
                prev_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                prev_node_prob = torch.softmax(linear_pred_atoms(prev_node_rep).double(), dim=-1)

            one_hot_attr = torch.zeros((batch.x.size(0), num_type_atoms + 1), requires_grad=False,
                                       device=batch.x.device)
            one_hot_attr[torch.arange(batch.x.size(0), device=batch.x.device), batch.x[:, 0]] = 1.
            one_hot_attr.requires_grad = True

            # num_mask_nodes_per_step = list()

            num_mask_per_iter = masked_num_nodes // mask_iters
            num_mask_nodes_per_step = [num_mask_per_iter for _ in range(mask_iters)]
            if mask_iters * num_mask_per_iter < masked_num_nodes:
                num_mask_nodes_per_step.append(masked_num_nodes - mask_iters * num_mask_per_iter)

            for j, num_mask_iter in enumerate(num_mask_nodes_per_step):
                node_rep = model(batch.x, batch.edge_index, batch.edge_attr, one_hot_attr=one_hot_attr)
                pred_node = linear_pred_atoms(node_rep)
                pred_node_prob = torch.softmax(pred_node.double(), dim=-1)
                entropy = -prev_node_prob * torch.log(torch.clamp(pred_node_prob, min=1e-12))
                entropy = torch.sum(entropy)
                # print("entropy's size = ", entropy.size())
                entropy.backward()

                delta_x_one_hot = one_hot_attr.grad
                delta_target_one_hot = -torch.sum(delta_x_one_hot * one_hot_attr, dim=-1, keepdim=False)
                # print(delta_target_one_hot)
                ordered_delta_index = delta_target_one_hot.argsort(dim=0, descending=True)
                masked_nodes = ordered_delta_index[: num_mask_iter]

                # delta_target_one_hot = delta_target_one_hot
                # delta_target_one_hot = delta_target_one_hot - torch.min(delta_target_one_hot)
                # # print(delta_target_one_hot)
                # delta_target_one_hot = torch.softmax(delta_target_one_hot, dim=0)
                # # print("after possi", delta_target_one_hot)
                # masked_nodes = torch.multinomial(delta_target_one_hot, num_mask_iter, replacement=False)
                _, index_max = torch.max(one_hot_attr, dim=1)
                index_max = index_max[masked_nodes]
                non_ty_node_idx = one_hot_attr.size(1) - 1
                ori_ty = batch.x[masked_nodes, 0]
                altered_ty = torch.where(index_max != non_ty_node_idx, torch.full((num_mask_iter, ), non_ty_node_idx,
                                                                                  device=device, dtype=torch.long), ori_ty)
                one_hot_attr.requires_grad = False
                one_hot_attr[masked_nodes, :] *= 0.0
                one_hot_attr[masked_nodes, altered_ty] = 1.0
                one_hot_attr.requires_grad = True

            _, altered_ty = torch.max(one_hot_attr, dim=-1)
            true_masked_num = torch.sum(altered_ty == non_ty_node_idx)
            print(true_masked_num, masked_num_nodes)
            masked_labels = batch.x[:, 0].clone()
            batch.x[:, 0] = altered_ty.detach()
            masked_nodes_indices = torch.arange(num_nodes, dtype=torch.long, device=device)

        else:
            raise NotImplementedError("mask_stra must be variance_based or meta_based")

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep[masked_nodes_indices, :])
        loss = criterion(pred_node.double(), masked_labels)

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()

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
    parser.add_argument('--mask_times', type=int, default=5, help='number of workers for dataset loading')
    parser.add_argument('--inf', type=float, default=1e12, help='number of workers for dataset loading')
    parser.add_argument('--mask_stra', type=str, default="meta_based_v2", help='number of workers for dataset loading')
    args = parser.parse_args()

    args.output_model_file = "temp/masked_strategies_{}".format(args.mask_stra)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.

    # set up models, one for pre-training and one for context embeddings
    # model_gen = GNN(max(args.num_layer - args.minus, 1), args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
    # gnn_type=args.gnn_type).to(device)
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    # linear_pred_atoms_dis = torch.nn.Linear(args.emb_dim, 2).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=None)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr,
    # weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        if epoch % 1 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
