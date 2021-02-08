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

from util import mask_strategies, gen_fake_mask_idxes

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
import pickle
def train(args, model_list, loader, optimizer_list, device):
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

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        num_nodes = batch.x.size(0)
        device = batch.x.device

        node_labels = torch.zeros((batch.x.size(0), ), dtype=torch.long).to(batch.x.device)
        node_labels[:] = batch.x[:, 0]
        # print(torch.max(node_labels))

        tot_masked = int(batch.x.size(0) * args.mask_rate + 1)
        num_nodes_per_mask = int(tot_masked // args.mask_times)
        mask_times = args.mask_times
        if num_nodes_per_mask == 0:
            num_masked_nodes = [tot_masked]
        else:
            num_masked_nodes = [num_nodes_per_mask for _ in range(mask_times)]
            if sum(num_masked_nodes) < tot_masked:
                num_masked_nodes.append(tot_masked - sum(num_masked_nodes))
        masked_nodes_idx = torch.empty((0, ), dtype=torch.long, device=batch.x.device)
        init_masked_prob = torch.ones((num_nodes, ), device=device)
        init_masked_prob = init_masked_prob / num_nodes

        unmasked_nodes_prob_distri = None

        num_has_masked_nodes = 0

        ori_batch_x = batch.x.clone()

        for i, num_per_mask in enumerate(num_masked_nodes):
            with torch.no_grad():
                if i == 0:
                    prev_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    prev_node_prob = torch.softmax(linear_pred_atoms(prev_node_rep).double(), dim=-1)
                    unmasked_nodes_prob_distri = prev_node_prob.clone()
                    masked_nodes = torch.multinomial(init_masked_prob, num_per_mask, replacement=False)
                    # print(masked_nodes.size())
                    batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    num_has_masked_nodes += num_per_mask
                else:
                    now_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    now_node_prob = torch.softmax(linear_pred_atoms(now_node_rep).double(), dim=-1)
                    entropy = -torch.sum(prev_node_prob * torch.log(torch.clamp(
                        now_node_prob, min=1e-12
                    )), dim=-1)
                    # print(entropy)
                    if args.reverse:
                        # print("in reversing")
                        entropy = entropy - torch.min(entropy)
                    else:
                        # then nodes with larger entropy are more likely to be chosen
                        entropy = torch.max(entropy) + torch.min(entropy) - entropy
                        # if args.dy_by_epoch:
                        #     sort_dims = torch.argsort(entropy, dim=0, descending=True)
                        #     if (args.epoch - 1) % args.loop_epoch != 0:
                        #         st_idx = int((float((args.epoch - 1) % args.loop_epoch) / float(args.loop_epoch)) * \
                        #                      entropy.size(0))
                        #         block_idx = sort_dims[: st_idx]
                        #     else:
                        #         block_idx = None

                    entropy = torch.softmax(entropy / T, dim=0)
                    entropy[masked_nodes_idx] = 0.0

                    # masked_prob = entropy / torch.sum(entropy, dim=0) # normalize the masking probability distribution
                    permute_idxes = np.arange(entropy.size(0))
                    np.random.shuffle(permute_idxes)
                    permute_idxes = torch.from_numpy(permute_idxes).to(entropy.device)
                    permute_entropys = entropy[permute_idxes]
                    sort_dims = torch.argsort(permute_entropys, dim=0, descending=True)
                    masked_nodes = sort_dims[:num_per_mask]
                    masked_nodes = permute_idxes[masked_nodes]
                    # assert abs(torch.sum(masked_prob, dim=0).item() - 1.0) < 1e-9, \
                    #     "expect sample prob summation to be 1, got {:.4f}" .format(torch.sum(masked_prob, dim=0).item())

                    # masked_nodes = torch.multinomial(masked_prob, num_per_mask, replacement=False)
                    batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    prev_node_prob = now_node_prob.clone()
                    num_has_masked_nodes += num_per_mask
        # masked_infos = {"edge_index": batch.edge_index.cpu(),
        #                 "masked_nodes": masked_nodes_idx.cpu(),
        #                 "batch": batch.batch.cpu(),
        #                 "x": batch.x.cpu()}

        # if args.epoch == 2 and step <= 200:
        #     fw = open("masked_info_exp/nll_type_2_epoch_{:d}_batch_{:d}".format(args.epoch, step), 'wb')
        #     pickle.dump(masked_infos, fw)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)

        # entropy = -torch.sum(prev_node_prob * torch.log(torch.clamp(
        #     now_node_prob, min=1e-12
        # )), dim=-1)

        with torch.no_grad():
            cross_entro = -(torch.sum(unmasked_nodes_prob_distri * torch.log(torch.clamp(pred_node, min=1e-12)), dim=1) \
                            + torch.sum((1. - unmasked_nodes_prob_distri) * torch.log(torch.clamp((1. - pred_node), min=1e-12)), dim=1))
            sum_cross_entro = torch.sum(cross_entro, dim=0)
            print("sum_cross_entropy_dy", sum_cross_entro)

            ori_batch_x[batch.masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
            node_rep_random = model(ori_batch_x, batch.edge_index, batch.edge_attr)
            pred_node_random = linear_pred_atoms(node_rep_random)
            cross_entro_random = -(torch.sum(unmasked_nodes_prob_distri * torch.log(torch.clamp(pred_node_random, min=1e-12)), dim=1) \
                            + torch.sum(
                        (1. - unmasked_nodes_prob_distri) * torch.log(torch.clamp((1. - pred_node_random), min=1e-12)), dim=1))
            sum_cross_entro_random = torch.sum(cross_entro_random, dim=0)
            print("sum_cross_entropy_random", sum_cross_entro_random)

        # print(torch.max(node_labels))
        loss = criterion(pred_node.double(), node_labels)

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        # optimizer_model_dis.step()
        optimizer_linear_pred_atoms.step()
        # optimizer_linear_pred_atoms_dis.step()
        optimizer_linear_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        # loss_accum_dis += float(loss_dis.cpu().item())
        print(loss.item())

    return loss_accum / step, acc_node_accum / step, acc_edge_accum / step


def main():
    # Training settings
    # params --- batch_size & mask_times
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--epoch', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--loop_epoch', type=int, default=20,
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
    parser.add_argument('--reverse', default=False, action='store_true', help='number of workers for dataset loading')
    parser.add_argument('--dy_by_epoch', default=False, action='store_true', help='number of workers for dataset loading')

    args = parser.parse_args()

    print(args.reverse)

    args.output_model_file = "temp/masker_entropy_based_2_gnn_{}_mask_times_v2_{:d}_batch_size_{:d}_reverse_{}".format(args.gnn_type, args.mask_times, args.batch_size, str(args.reverse))
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
                              transform=gen_fake_mask_idxes(0.15))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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
        args.epoch = epoch
        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        # if epoch >= 10:
        #     args.reverse = True
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
