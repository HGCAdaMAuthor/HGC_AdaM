import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # print(type(batch.edge_index))
        # batch.edge_index = torch.LongTensor(batch.edge_index.numpy())
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_scores = []

    y_pred = list()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()
    # input_dict = {"y_true": y_true, "y_pred": y_pred}
    # print(y_true)
    #
    # return evaluator.eval(input_dict)

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / max(len(roc_list), 1e-12)  # y_true.shape[1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--reset', default=False, action="store_true", help='number of workers for dataset loading')
    parser.add_argument('--resetp', default=False, action="store_true", help='number of workers for dataset loading')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='number of workers for dataset loading')
    parser.add_argument('--test_ratio', type=float, default=0.8, help='number of workers for dataset loading')
    parser.add_argument('--output_file', type=str, default='', help='number of workers for dataset loading')

    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # set up dataset
    # dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    args.dataset = "ogbg-molpcba"
    dataset = PygGraphPropPredDataset(name=args.dataset)

    dataset.data.x = dataset.data.x[:, :2]
    dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    dataset.data.y = torch.where(dataset.data.y > 0, dataset.data.y, torch.zeros_like(dataset.data.y))
    print(dataset.data.y)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, dataset.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    if args.reset == True:
        model.reset_node_embedding()

    if args.resetp == True:
        model.reset_parameterss()

    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    best_test_acc = 0
    best_epoch = 0

    test_acc = eval(args, model, device, test_loader, evaluator)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, evaluator)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        # val_acc = eval(args, model, device, val_loader)
        val_acc = 0.0
        test_acc = eval(args, model, device, test_loader, evaluator)

        # print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))
        print(test_acc)
        # test_acc = test_acc["ap"]

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    print("best_acc", best_test_acc, best_epoch)
    if not args.filename == "":
        writer.close()

    if args.input_model_file.find("/") != -1:
        ssl = args.input_model_file.split("/")[1]
    else:
        ssl = args.input_model_file
    if ssl.find(".") != -1:
        ssl = ssl.split(".")[0]

    if not args.input_model_file == "":
        if not os.path.exists("./record"):
            os.mkdir("./record")
        with open(os.path.join("./record", ssl), "a") as wf:
            wf.write("dataset: {}, train_ratio: {:.4f}, best epoch: {:d}, best acc: {:.4f}.\n".format(args.dataset,
                                                                                                      args.train_ratio,
                                                                                                      best_epoch,
                                                                                                      best_test_acc))
            wf.close()


if __name__ == "__main__":
    main()
