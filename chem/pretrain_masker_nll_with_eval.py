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

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

# from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

# from util import mask_strategies

# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from torch_geometric.data import DataLoader

# from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

# import timeit

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

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        if args.epoch == 1 and step == 10:
            print("just for debug val")
            break
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

        num_has_masked_nodes = 0

        for i, num_per_mask in enumerate(num_masked_nodes):
            with torch.no_grad():
                if i == 0:
                    prev_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    prev_node_prob = torch.softmax(linear_pred_atoms(prev_node_rep).double(), dim=-1)
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

                    # entropy[masked_nodes_idx] = -LARGE_NUM
                    # print(entropy)  #
                    entropy = torch.softmax(entropy / T, dim=0)
                    entropy[masked_nodes_idx] = 0.0
                    if args.dy_by_epoch: # choose nodes to be masked based on epochs
                        permute_idxes = np.arange(entropy.size(0))
                        np.random.shuffle(permute_idxes)
                        permute_idxes = torch.from_numpy(permute_idxes).to(entropy.device)
                        permute_entropys = entropy[permute_idxes]
                        sort_dims = torch.argsort(permute_entropys, dim=0, descending=True)  # sort entropy in descending order
                        if (args.epoch - 1) % args.loop_epoch != 0:  # blocked nodes is not zero
                            st_idx = int((float((args.epoch - 1) % args.loop_epoch) / float(args.loop_epoch)) * \
                                         (entropy.size(0) - num_has_masked_nodes))  # those nodes should be blocked
                            block_idx = sort_dims[: st_idx]  # blocked nodes idxes
                            block_idx = permute_idxes[block_idx]
                            entropy[block_idx] = 0.0  # set their entropy (now the probabilities) to zero

                        # sort_dims = torch.argsort(entropy, dim=0, descending=True) # sort entropy in descending order
                        # if (args.epoch - 1) % args.loop_epoch != 0:  # blocked nodes is not zero
                        #     st_idx = int((float((args.epoch - 1) % args.loop_epoch) / float(args.loop_epoch)) * \
                        #                  (entropy.size(0) - num_has_masked_nodes)) # those nodes should be blocked
                        #     block_idx = sort_dims[: st_idx] # blocked nodes idxes
                        #     entropy[block_idx] = 0.0 # set their entropy (now the probabilities) to zero
                    #     else:
                    #         block_idx = None
                    # if args.dy_by_epoch and block_idx is not None:

                    masked_prob = entropy / torch.sum(entropy, dim=0) # normalize the masking probability distribution
                    assert abs(torch.sum(masked_prob, dim=0).item() - 1.0) < 1e-9, \
                        "expect sample prob summation to be 1, got {:.4f}" .format(torch.sum(masked_prob, dim=0).item())

                    masked_nodes = torch.multinomial(masked_prob, num_per_mask, replacement=False)
                    batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    prev_node_prob = now_node_prob.clone()
                    num_has_masked_nodes += num_per_mask

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)
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
        # print(loss.item())

    return loss_accum / step, acc_node_accum / step, acc_edge_accum / step


import pickle
import os
import datetime
def get_val_dataloader(args):
    if args.env == "jizhi":
        dataset_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/" + args.val_dataset
    else:
        dataset_path = "./dataset/" + args.val_dataset
    dataset = MoleculeDataset(dataset_path, dataset=args.val_dataset)
    # smiles_list = pd.read_csv(dataset_path + '/processed/smiles.csv', header=None)[0].tolist()
    processed_dataset_path = dataset_path + "/processed"
    with open(os.path.join(processed_dataset_path, "split_idx_train.pkl"), "rb") as f:
        train_idx_tsr = pickle.load(f)
        f.close()

    with open(os.path.join(processed_dataset_path, "split_idx_valid.pkl"), "rb") as f:
        valid_idx_tsr = pickle.load(f)
        f.close()

    with open(os.path.join(processed_dataset_path, "split_idx_test.pkl"), "rb") as f:
        test_idx_tsr = pickle.load(f)
        f.close()
    train_dataset = dataset[train_idx_tsr]
    valid_dataset = dataset[valid_idx_tsr]
    test_dataset = dataset[test_idx_tsr]

    train_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers)
    # torch_geometric.data.DataListLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader

def get_val_graph_pred_model(args):
    if args.val_dataset == "tox21":
        num_tasks = 12
    elif args.val_dataset == "hiv":
        num_tasks = 1
    elif args.val_dataset == "pcba":
        num_tasks = 128
    elif args.val_dataset == "muv":
        num_tasks = 17
    elif args.val_dataset == "bace":
        num_tasks = 1
    elif args.val_dataset == "bbbp":
        num_tasks = 1
    elif args.val_dataset == "toxcast":
        num_tasks = 617
    elif args.val_dataset == "sider":
        num_tasks = 27
    elif args.val_dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")
    model = GNN_graphpred(args.num_layer,
                          args.emb_dim,
                          num_tasks,
                          JK=args.JK,
                          drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling,
                          gnn_type=args.gnn_type)
    # todo: remember to set the input_model_file to the latest trained and saved model file
    if not args.input_model_file == "":
        print("from pretrained")
        model.from_pretrained(args.input_model_file)
    # model = torch_geometric.nn.DataParallel(model.cuda())
    model = model.to(torch.device("cuda:" + str(args.device)))

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    return model, optimizer

def load_test_model(args, load_from_file):
    if args.val_dataset == "tox21":
        num_tasks = 12
    elif args.val_dataset == "hiv":
        num_tasks = 1
    elif args.val_dataset == "pcba":
        num_tasks = 128
    elif args.val_dataset == "muv":
        num_tasks = 17
    elif args.val_dataset == "bace":
        num_tasks = 1
    elif args.val_dataset == "bbbp":
        num_tasks = 1
    elif args.val_dataset == "toxcast":
        num_tasks = 617
    elif args.val_dataset == "sider":
        num_tasks = 27
    elif args.val_dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")
    model = GNN_graphpred(args.num_layer,
                          args.emb_dim,
                          num_tasks,
                          JK=args.JK,
                          drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling,
                          gnn_type=args.gnn_type)
    model.load_state_dict(torch.load(load_from_file, map_location='cpu'))
    model = model.to(torch.device("cuda:" + str(args.device)))
    return model

criterion_val = nn.BCEWithLogitsLoss(reduction = "none")
def train_val(args, model, train_loader, optimizer):
    model.train()
    device = torch.device("cuda:" + str(args.device))

    # for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    for step, batch in enumerate(train_loader):
        # batch = batch.to()
        # print(type(batch.edge_index))
        # batch.edge_index = torch.LongTensor(batch.edge_index.numpy())
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion_val(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def eval_val(args, model, loader):
    model.eval()
    y_true = []
    y_scores = []

    device = torch.device("cuda:" + str(args.device))

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/max(len(roc_list), 1e-12) #y_true.shape[1]


def start_val(dataloaders, args, epoch):
    train_loader, valid_loader, test_loader = dataloaders
    args.input_model_file = args.output_model_file + "_{:d}.pth".format(epoch)
    train_epoch = epoch
    model, optimizer = get_val_graph_pred_model(args)
    best_test_acc = 0
    best_epoch = 0
    best_val_acc = 0

    if args.env == "jizhi":
        save_eval_model_folder = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/saved_model/"
    else:
        save_eval_model_folder = "saved_model/"
    if not os.path.exists(save_eval_model_folder):
        os.mkdir(save_eval_model_folder)

    for epoch in range(1, args.eval_epochs + 1):
        print("====eval epoch " + str(epoch))

        train_val(args, model, train_loader, optimizer)

        print("====Evaluation")
        val_acc = eval_val(args, model, valid_loader)
        # val_acc = 0.0
        test_acc = eval_val(args, model, test_loader)

        # print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

        if test_acc > best_test_acc:
        # if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_test_acc = test_acc
            if test_acc > args.besttestacc:
                torch.save(model.state_dict(), os.path.join(save_eval_model_folder,
                                                           "masker_entropy_gnn_{}_mask_times_{:d}_batch_size_{:d}_reverse_{}_drprate_{:.4f}_num_layers_{:d}_lr_{:.4f}_mask_ratio_{:.4f}_{}_tr_ep_{:d}.pth".format(args.gnn_type,
                                                                                                          args.mask_times,
                                                                                                          args.batch_size,
                                                                                                          str(args.reverse),
                                                                                                          args.dropout_ratio,
                                                                                                          args.num_layer,
                                                                                                          args.lr,
                                                                                                          args.mask_rate,
                                                                                                          args.val_dataset,
                                                                                                          train_epoch)))

    return best_test_acc, best_epoch

import json
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

    parser.add_argument('--val_dataset', type=str, default='bbbp',
                        help='root directory of dataset for pretraining')

    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
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
    parser.add_argument('--env', type=str, default="normal",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--eval_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')

    args = parser.parse_args()

    print(args.reverse)

    worker_path = os.getenv("JIZHI_WORKSPACE_PATH")
    # job_param_path = './job_param.json'
    job_param_path = os.path.join(worker_path, "job_param.json")
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        mask_times = hyper_params["mask_times"]
        batch_size = hyper_params["batch_size"]
        dropout_ratio = hyper_params["dropout_ratio"]
        num_layer = hyper_params["num_layer"]
        lr = hyper_params["lr"]

    args.mask_times = mask_times
    args.batch_size = batch_size
    args.dropout_ratio = dropout_ratio
    args.num_layer = num_layer
    args.lr = lr

    args.output_model_file = \
        "temp/masker_entropy_based_rever_exp_gnn_{}_mask_times_v2_{:d}_batch_size_{:d}_reverse_{}_drprate_{:.4f}_num_layers_{:d}_lr_{:.4f}_mask_ratio_{:.4f}".format(args.gnn_type,
                                                                                                          args.mask_times,
                                                                                                          args.batch_size,
                                                                                                          str(args.reverse),
                                                                                                          args.dropout_ratio,
                                                                                                          args.num_layer,
                                                                                                          args.lr,
                                                                                                          args.mask_rate)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.env == "jizhi":
        args.output_model_file = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/temp_model/" + args.output_model_file.split("/", 1)[1]

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    dataset_root_path = "dataset/" + args.dataset
    if args.env == "jizhi":
        dataset_root_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/" + args.dataset
    dataset = MoleculeDataset(dataset_root_path, dataset=args.dataset,
                              transform=None)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_loader, valid_loader, test_loader = get_val_dataloader(args)

    print("got val loaders!")

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_masker = optim.Adam(masker.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_model_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    # optimizer_linear_pred_atoms_dis = optim.Adam(linear_pred_atoms_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    best_test_acc = 0.0
    best_test_epoch = 0
    args.besttestacc = best_test_acc

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

        if epoch % 5 == 0:
            test_acc, val_epoch = start_val([train_loader, valid_loader, test_loader], args, epoch)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch
                args.besttestacc = test_acc
            print('time: %s, epoch = %d, best_test_epoch = %d, auc = %.4f' % (
            datetime.datetime.now(), epoch, best_test_epoch, best_test_acc))

    # if not args.output_model_file == "":
    #     torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
