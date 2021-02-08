import argparse

from loader import MoleculeDataset, GCC_GRAPH_CLASSIFICATION_DATASETS, GCC_NODE_CLASSIFICATION_DATASETS
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# from splitters import scaffold_split
# import pandas as pd

import os
# import shutil

# from tensorboardX import SummaryWriter
# import sklearn
# print("sklearn", sklearn.__version__) 0.23.2
# import torch_geometric
# print("pgy", torch_geometric.__version__) 1.6.1 python 3.6.8  torch 1.5.0 cu 101

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
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


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    tlab = []
    prdlab = []
    totb = 0
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

        preds = pred.argmax(dim=1)
        y = batch.y.view(pred.shape).argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")
        tlab.append(f1 * batch.x.size(0))
        totb += batch.x.size(0)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        elif (args.dataset not in GCC_GRAPH_CLASSIFICATION_DATASETS
            and args.dataset not in GCC_NODE_CLASSIFICATION_DATASETS):
            print(y_true)

    if len(roc_list) < y_true.shape[1] and (args.dataset not in GCC_GRAPH_CLASSIFICATION_DATASETS
            and args.dataset not in GCC_NODE_CLASSIFICATION_DATASETS):
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    if args.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS \
            or args.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
        return sum(tlab) / totb # len(tlab)
    else:
        return sum(roc_list) / max(len(roc_list), 1e-12)  # y_true.shape[1]


import pickle
import datetime
import json
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
    parser.add_argument('--env', type=str, default='normal', help='number of workers for dataset loading')
    parser.add_argument('--no_automl', default=False, action="store_true", help='number of workers for dataset loading')
    parser.add_argument('--no_val', default=False, action="store_true", help='number of workers for dataset loading')
    parser.add_argument('--othera', default=False, action="store_true", help='number of workers for dataset loading')

    args = parser.parse_args()

    args.runseed = 0
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if not args.no_automl:
        worker_path = os.getenv("JIZHI_WORKSPACE_PATH")
        # job_param_path = './job_param.json'
        job_param_path = os.path.join(worker_path, "job_param.json")
        with open(job_param_path, 'r') as f:
            hyper_params = json.load(f)
            batch_size = hyper_params["batch_size"]
            dropout_ratio = hyper_params["dropout_ratio"]
            lr = hyper_params["lr"]
            if args.dataset == "hiv":
                if "pooling" in hyper_params:
                    args.graph_pooling = "mean" if hyper_params["pooling"] == 1 else "sum"
                args.lr_scale = hyper_params["lr_scale"]

        args.batch_size = batch_size
        args.dropout_ratio = dropout_ratio
        args.lr = lr

    dataset_root_path = "dataset/" + args.dataset
    if args.env == "jizhi":
        dataset_root_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/" + args.dataset
        if args.othera:
            dataset_root_path = "/apdcephfs/share_1142145/meowliu/gnn_pretraining/data/" + args.dataset

    dataset = MoleculeDataset(dataset_root_path, dataset=args.dataset)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == "collab":
        num_tasks = 3
    elif args.dataset == "rdt-b":
        num_tasks = 2
    elif args.dataset == "imdb-binary":
        num_tasks = 2
    elif args.dataset == "imdb-multi":
        num_tasks = 3
    elif args.dataset == "rdt-5k":
        num_tasks = 5
    elif args.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
        num_tasks = dataset.num_labels
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset

    # dataset_root_path = "dataset/" + args.dataset
    # if args.env == "jizhi":
    #     dataset_root_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/" + args.dataset
    #     if args.othera:
    #         dataset_root_path = "/apdcephfs/share_1142145/meowliu/gnn_pretraining/data/" + args.dataset
    #
    # dataset = MoleculeDataset(dataset_root_path, dataset=args.dataset)

    print(dataset)

    assert args.split == "scaffold"

    processed_dataset_path = dataset_root_path + "/processed"

    if args.dataset not in GCC_GRAPH_CLASSIFICATION_DATASETS and \
            args.dataset not in GCC_NODE_CLASSIFICATION_DATASETS:
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
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        idx_list = []
        # dataset_len = len(dataset.graphs)
        for idx in skf.split(np.zeros(len(dataset.labels)), dataset.labels):
            idx_list.append(idx)
        trainn_idx, test_idx = idx_list[0]

        # train_dataset = torch.utils.data.Subset(dataset, trainn_idx)

        print("trainn_idx_type", type(trainn_idx))
        # trainn_dataset = torch.utils.data.Subset(dataset, trainn_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        # valid_dataset = torch.utils.data.Subset(dataset, test_idx)


        trainn_labels = dataset.labels[torch.tensor(trainn_idx, dtype=torch.long)]
        skfs = StratifiedKFold(n_splits=9, shuffle=True, random_state=args.seed)
        idx_listt = []
        for idx in skfs.split(np.zeros(len(trainn_labels)), trainn_labels):
            idx_listt.append(idx)
        train_idx, valid_idx = idx_listt[0]
        trainn_idx_tsr = torch.tensor(trainn_idx, dtype=torch.long)
        train_idx = trainn_idx_tsr[train_idx]
        valid_idx = trainn_idx_tsr[valid_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

        # print("original split train_len = %d, test_len = %d" % (len(train_idx), len(test_idx)))
        # train_offset = (len(dataset) // 10) * 8 + 1
        # train_idx, valid_idx = train_idx[: train_offset], train_idx[train_offset: ]
        # print("train_len = %d, valid_len = %d, test_len = %d" % (len(train_idx), len(valid_idx), len(test_idx)))
        # train_dataset = torch.utils.data.Subset(dataset, train_idx)
        # valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
        # test_dataset = torch.utils.data.Subset(dataset, test_idx)

        # skf_2 = StratifiedKFold(n_splits=9, shuffle=False, random_state=args.seed)
        # idx_list_2 = []
        # for idx in skf_2.split(np.zeros(len(train_dataset.labels)), train_dataset.labels):
        #     idx_list_2.append(idx)
        # train_idx, valid_idx = idx_list_2[0]
        # train_dataset = torch.utils.data.Subset(train_datasetd)
        # train_dataset = torch.utils.data.Subset(dataset, train_idx)
        # test_dataset = torch.utils.data.Subset(dataset, test_idx)
        # valid_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # torch_geometric.data.DataListLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # if args.split == "scaffold":
    #     smiles_list = pd.read_csv(dataset_root_path + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
    #                                                                 frac_train=args.train_ratio,
    #                                                                 frac_valid=1. - args.train_ratio - args.test_ratio,
    #                                                                 frac_test=args.test_ratio)
    #     print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
    #                                                               frac_test=0.1, seed=args.seed)
    #     print("random")
    # elif args.split == "random_scaffold":
    #     smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #     train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
    #                                                                        frac_train=0.8, frac_valid=0.1,
    #                                                                        frac_test=0.1, seed=args.seed)
    #     print("random scaffold")
    # else:
    #     raise ValueError("Invalid split option.")

    # print(train_dataset[0])

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred(args.num_layer,
                          args.emb_dim,
                          num_tasks,
                          JK=args.JK,
                          drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling,
                          gnn_type=args.gnn_type)
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

    # if not args.filename == "":
    #     fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
    #     # delete the directory if there exists one
    #     if os.path.exists(fname):
    #         shutil.rmtree(fname)
    #         print("removed the existing file.")
    #     writer = SummaryWriter(fname)

    best_test_acc = 0
    best_epoch = 0
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        # if args.eval_train:
        #     train_acc = eval(args, model, device, train_loader)
        # else:
        #     print("omit the training accuracy computation")
        #     train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        # val_acc = 0.0
        test_acc = eval(args, model, device, test_loader)

        print("time: %s, val: %.4f, test: %.4f" % (datetime.datetime.now(), val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        # train_acc_list.append(train_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_test_acc = test_acc

        # if not args.filename == "":
        #     writer.add_scalar('data/train auc', train_acc, epoch)
        #     writer.add_scalar('data/val auc', val_acc, epoch)
        #     writer.add_scalar('data/test auc', test_acc, epoch)

    # print("best_val_acc = ", best_val_acc, "best_test_acc = ", best_test_acc, best_epoch)

    print('time: %s, auc = %.4f, dataset: %s' % (
        datetime.datetime.now(), best_val_acc, args.dataset))
    print("test_auc = %.4f" % best_test_acc)
    # print(best_test_acc)
    # if not args.filename == "":
    #     writer.close()

    # if args.input_model_file.find("/") != -1:
    #     ssl = args.input_model_file.split("/")[1]
    # else:
    #     ssl = args.input_model_file
    # if ssl.find(".") != -1:
    #     ssl = ssl.split(".")[0]

    # if not args.input_model_file == "":
    #     if not os.path.exists("./record"):
    #         os.mkdir("./record")
    #     with open(os.path.join("./record", ssl), "a") as wf:
    #         wf.write("gnn_type: {}, dataset: {}, train_ratio: {:.4f}, best epoch: {:d}, best acc: {:.4f}.\n".format(
    #             args.gnn_type, args.dataset, args.train_ratio, best_epoch, best_test_acc))
    #         wf.close()


if __name__ == "__main__":
    main()
