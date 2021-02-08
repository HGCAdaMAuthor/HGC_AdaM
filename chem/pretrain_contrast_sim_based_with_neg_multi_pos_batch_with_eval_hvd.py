import argparse


import sys
print(sys.path)

from loader import MoleculeDataset, ContrastDataset, MoleculeDatasetForContrast

from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

# from model import GNN
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
# from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
import torch_geometric
import datetime

from torch_geometric.data import DataLoader

# from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask

from dataloader import DataLoaderContrastSimBasedMultiPosPer, DataLoaderContrastSimBasedMultiPosComNeg, \
    DataLoaderContrastSimBasedMultiPosComNegPyg, DataLoaderContrastList

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# from tensorboardX import SummaryWriter
# from contrastive_model import MemoryMoCo
# from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform
import torch.utils.data.distributed
import horovod.torch as hvd



LARGE_NUM = 1e9
criterion_dis = nn.BCEWithLogitsLoss()

import sys
print(sys.path)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class atten_module(nn.Module):
    def __init__(self, input_dim, value_dim=None, query_dim=None, key_dim=None):
        super(atten_module, self).__init__()
        if value_dim == None:
            value_dim = input_dim
        if query_dim == None:
            query_dim = value_dim
        if key_dim == None:
            key_dim = value_dim

        self.value_weight = nn.Parameter(torch.Tensor(input_dim, value_dim))
        self.query_weight = nn.Parameter(torch.Tensor(input_dim, query_dim))
        self.key_weight = nn.Parameter(torch.Tensor(input_dim, key_dim))
        self.temperature = 8
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.value_weight.size(0)
        uniform(size, self.value_weight)
        size = self.query_weight.size(0)
        uniform(size, self.query_weight)
        size = self.key_weight.size(0)
        uniform(size, self.key_weight)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.query_weight)
        k = torch.matmul(x, self.key_weight)
        v = torch.matmul(x, self.value_weight)
        # print("q_shape ", q.size())
        # print("k_shape ", k.size())
        # print("v_shape ", v.size())
        # if mask != None:
        #     print("mask_shape ", mask.size())
        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))
        if mask is not None:
            attn = attn.masked_fill_(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        return out


def pool_func(x, batch, mode="sum"):
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


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


# criterion = nn.BCEWithLogitsLoss()


def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def compute_accuracy(pred, target, bs):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / bs

los_batch = []

sampled_neg_idx_to_count = dict()

import os

import time
def train(args, model_q, loader, optimizer_q, device, epoch):
    # model_k.eval()
    model_q.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    # stt_time = time.time()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # print(step)
        if (epoch == 1 and step == 10):
            print("just for debug eval")
            break
        # print("time for fetching data: ", time.time() - stt_time)
        # num_samples = len(batch.sim_x)
        # batch = batch.to(device)
        batch = batch.cuda()
        # batch = batch.cuda()
        num_samples = args.num_samples
        num_neg_samples = args.num_neg_samples * num_samples
        # num_neg_samples = 32
        num_neg_samples = args.num_com_negs
        # print("in main hhh")

        # loss = model_q.loss(batch, device, num_samples, num_neg_samples, args, T)

        # neg_data_list = batch.neg_data_list
        # sim_data_list = batch.sim_data_list
        # data_list = batch.data_list

        # neg_rep = model_q(neg_data_list)

        neg_rep = model_q(batch.neg_x, batch.neg_edge_index, batch.neg_edge_attr)
        neg_rep_glb = pool_func(neg_rep, batch.neg_batch, args.pooling)

        node_rep = model_q(batch.x, batch.edge_index, batch.edge_attr)
        # node_rep = model_q(data_list)
        rep_glb = pool_func(node_rep, batch.batch, args.pooling)

        sim_rep = model_q(batch.sim_x, batch.sim_edge_index, batch.sim_edge_attr)
        # sim_rep = model_q(sim_data_list)

        sim_rep_glb = pool_func(sim_rep, batch.sim_batch, args.pooling)

        bs = rep_glb.size(0)

        rep_glb = rep_glb / torch.clamp(torch.norm(rep_glb, p=2, dim=1, keepdim=True), min=1e-9)
        sim_rep_glb = sim_rep_glb / torch.clamp(torch.norm(sim_rep_glb, p=2, dim=1, keepdim=True), min=1e-9)
        sim_rep_glb = sim_rep_glb.view(rep_glb.size(0), num_samples, -1) # bs x num_samples x rep_size
        neg_rep_glb = neg_rep_glb / torch.clamp(torch.norm(neg_rep_glb, p=2, dim=1, keepdim=True), min=1e-9)

        self_neg_mult_scores = torch.matmul(rep_glb, neg_rep_glb.transpose(1, 0)) / T
        # print(self_neg_mult_scores.size())
        self_neg_mult_scores_expand = torch.cat([torch.zeros((rep_glb.size(0), 1), dtype=torch.float32).cuda(), self_neg_mult_scores],
                                                dim=-1)
        for i in range(num_samples):
            masked_node_idx = batch.masked_pos_idx[i, :] + 1
            self_neg_mult_scores_expand[torch.arange(rep_glb.size(0)), masked_node_idx] = -LARGE_NUM

        self_neg_mult_scores = self_neg_mult_scores_expand[:, 1:] # remove dim 0
        # bs x num_samples x num_neg_samples
        self_neg_mult_scores = self_neg_mult_scores.unsqueeze(1).repeat(1, num_samples, 1)
        # bs x 1 x rep_size

        self_pos_mult_scores = torch.sum(torch.mul(rep_glb.unsqueeze(1), sim_rep_glb), dim=-1, keepdim=True) / T
        # bs x num_samples x 1
        self_pos_neg_mult_scores_cat = torch.cat([self_pos_mult_scores, self_neg_mult_scores], dim=-1)
        # bs x num_samples x (1 + num_neg_samples)
        self_pos_neg_mult_scores_cat_expand = self_pos_neg_mult_scores_cat.view(-1, 1 + num_neg_samples)
        labels = torch.zeros((self_pos_neg_mult_scores_cat_expand.size(0), ), dtype=torch.long).cuda()
        loss = F.nll_loss(F.log_softmax(self_pos_neg_mult_scores_cat_expand, dim=-1), labels)

        sim_neg_mult_scores = torch.sum(torch.mul(sim_rep_glb.unsqueeze(2), neg_rep_glb.unsqueeze(0).unsqueeze(0)),
                                        dim=-1, keepdim=False) / T
        # bs x num_sampleds x num_neg_samples
        sim_self_mult_scores = torch.sum(torch.mul(sim_rep_glb, rep_glb.unsqueeze(1)), dim=-1, keepdim=True) / T
        # bs x num_samples x 1
        sim_neg_mult_scores_expand = torch.cat([torch.zeros((bs, num_samples, 1)).cuda(), sim_neg_mult_scores],
                                               dim=-1)
        for i in range(num_samples):
            masked_node_idx = batch.masked_pos_idx[i, :] + 1
            sim_neg_mult_scores_expand[torch.arange(bs).cuda(), :, masked_node_idx] = -LARGE_NUM
        sim_neg_mult_scores = sim_neg_mult_scores_expand[:, :, 1:]
        sim_self_neg_mult_scores_cat = torch.cat([sim_self_mult_scores, sim_neg_mult_scores], dim=-1)
        sim_self_neg_mult_scores_cat_expand = sim_self_neg_mult_scores_cat.view(-1, 1 + num_neg_samples)
        sim_self_labels = torch.zeros((sim_self_neg_mult_scores_cat_expand.size(0), ), dtype=torch.long).cuda()
        loss += F.nll_loss(F.log_softmax(sim_self_neg_mult_scores_cat_expand, dim=-1), sim_self_labels)

        optimizer_q.zero_grad()
        # loss = criterion(out)
        loss.backward()
        clip_grad_norm(model_q.parameters(), args.clip_norm)
        global_step = (epoch - 1) * args.batch_size + step
        lr_this_step = args.lr * warmup_linear(
            global_step / (args.epochs * args.batch_size), 0.1
        )
        for param_group in optimizer_q.param_groups:
            param_group["lr"] = lr_this_step

        balanced_loss_accum += loss.detach().cpu().item()

        optimizer_q.step()
        # moment_update(model_q, model_k, args.alpha)
        torch.cuda.synchronize()
        # stt_time = time.time()

        # print(loss.detach().cpu().item(), batch.tot_in.cpu())
        # with open(args.log_loss_file, "a") as wf:
        #     wf.write("{}\n".format(str(loss.detach().cpu().item())))
        print('time: %s, epoch = %d, step = %d, loss = %.4f, totin = %d' % (
            datetime.datetime.now(), epoch, step, loss.detach().cpu().item(), batch.tot_in.cpu().item()))
        # if len(args.los_batch) < 100:
        #     args.los_batch.append(loss.detach().cpu().item())
        # else:
        #     with open(args.log_loss_file, "a") as wf:
        #         wf.write("{}\n".format(str(sum(args.los_batch) / len(args.los_batch))))
        #     args.los_batch = [loss.detach().cpu().item()]
        # for sampled_neg_idx in batch.sampled_neg_idxes:
        #     if sampled_neg_idx not in sampled_neg_idx_to_count:
        #         sampled_neg_idx_to_count[sampled_neg_idx] = 1
        #     else:
        #         sampled_neg_idx_to_count[sampled_neg_idx] += 1
        # if step % 100 == 0:
        #     with open(args.log_neg_samples_idx_file + "_epoch_{:d}_step_{:d}".format(epoch, step), "a") as wf:
        #         for neg_idx in sampled_neg_idx_to_count:
        #             wf.write("{:d}: {:d}\n".format(neg_idx, sampled_neg_idx_to_count[neg_idx]))
        #         wf.close()

    return balanced_loss_accum / step, acc_accum / (step * 2)


import pickle

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
    # model = GNN_graphpred(args.num_layer,
    #                       args.emb_dim,
    #                       num_tasks,
    #                       JK=args.JK,
    #                       drop_ratio=args.dropout_ratio,
    #                       graph_pooling=args.graph_pooling,
    #                       gnn_type=args.gnn_type)
    # # todo: remember to set the input_model_file to the latest trained and saved model file
    # if not args.input_model_file == "":
    #     print("from pretrained")
    #     model.from_pretrained(args.input_model_file)
    # # model = torch_geometric.nn.DataParallel(model.cuda())
    # model = model.to(torch.device("cuda:" + str(args.device)))
    #
    # model_param_group = []
    # model_param_group.append({"params": model.gnn.parameters()})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    model_gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                    gnn_type=args.gnn_type)  # .to(device)

    # todo: remember to set the input_model_file to the latest trained and saved model file

    if not args.input_model_file == "":
        # model.from_pretrained(args.input_model_file)
        model_gnn.load_state_dict(torch.load(args.input_model_file, map_location='cpu'))
    # model_gnn = torch_geometric.nn.DataParallel(model_gnn.cuda())
    model_gnn = model_gnn.cuda()
    # model_gnn = torch_geometric.nn.DataParallel(model_gnn)
    optimizer_gnn = optim.Adam(model_gnn.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_gnn = hvd.DistributedOptimizer(optimizer_gnn,
                                             named_parameters=model_gnn.named_parameters(),
                                             compression=hvd.Compression.none,
                                             op=hvd.Adasum,  # if args.use_adasum else hvd.Average,
                                             gradient_predivide_factor=args.gradient_predivide_factor
                                             )
    hvd.broadcast_parameters(model_gnn.state_dict(), root_rank=0)

    mult = 1
    if args.JK == "concat":
        graph_pred_linear = torch.nn.Linear(mult * (args.num_layer + 1) * args.emb_dim, num_tasks)
    else:
        graph_pred_linear = torch.nn.Linear(mult * args.emb_dim, num_tasks)
    # graph_pred_linear = DataParallel(graph_pred_linear.cuda())
    # move it to cuda
    graph_pred_linear = graph_pred_linear.cuda()
    optimizer_pred = optim.Adam(graph_pred_linear.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_pred = hvd.DistributedOptimizer(optimizer_pred,
                                              named_parameters=graph_pred_linear.named_parameters(),
                                              compression=hvd.Compression.none,
                                              op=hvd.Adasum,  # if args.use_adasum else hvd.Average,
                                              gradient_predivide_factor=args.gradient_predivide_factor
                                              )
    hvd.broadcast_parameters(graph_pred_linear.state_dict(), root_rank=0)

    # model_param_group = []
    # model_param_group.append({"params": model_gnn.parameters()})
    # model_param_group.append({"params": graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    return [model_gnn, graph_pred_linear], [optimizer_gnn, optimizer_pred]

    # return model, optimizer


criterion_val = nn.BCEWithLogitsLoss(reduction = "none")
def train_val(args, model, train_loader, optimizer):
    # model.train()
    optimizer_gnn, optimizer_pred = optimizer
    model_gnn, model_pred = model
    model_gnn.train()
    model_pred.train()
    # device = torch.device("cuda:" + str(args.device))

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        # batch = batch.to()
        # print(type(batch.edge_index))
        # batch.edge_index = torch.LongTensor(batch.edge_index.numpy())
        # batch = batch.to(device)
        batch = batch.cuda()

        node_rep = model_gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = model_pred(pool_func(node_rep, batch.batch, args.graph_pooling))

        # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion_val(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        # optimizer.zero_grad()
        optimizer_gnn.zero_grad()
        optimizer_pred.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer_gnn.step()
        optimizer_pred.step()


def eval_val(args, model, loader):
    # model.eval()
    model_gnn, model_pred = model
    model_gnn.eval()
    model_pred.eval()
    # optimizer_gnn, optimizer_pred =
    y_true = []
    y_scores = []

    # device = torch.device("cuda:" + str(args.device))

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch = batch.to(device)
        batch = batch.cuda()

        with torch.no_grad():
            # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            node_rep = model_gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model_pred(pool_func(node_rep, batch.batch, args.graph_pooling))

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
    model, optimizer = get_val_graph_pred_model(args)
    best_test_acc = 0
    best_epoch = 0
    best_val_acc = 0

    for epoch in range(1, args.eval_epochs + 1):
        print("====eval epoch " + str(epoch))

        train_val(args, model, train_loader, optimizer)

        print("====Evaluation")
        val_acc = eval_val(args, model, valid_loader)
        # val_acc = 0.0
        test_acc = eval_val(args, model, test_loader)

        # print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

        if test_acc > best_test_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_test_acc = test_acc
    return best_test_acc, best_epoch


import json
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')

    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--eval_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--csize', type=int, default=3,
                        help='context size (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--mode', type=str, default="cbow", help="cbow or skipgram")
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')

    parser.add_argument('--val_dataset', type=str, default='bbbp',
                        help='root directory of dataset for pretraining')

    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--output_model_file', type=str, default='temp/contrast2', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--num_hops', type=int, default=2, help='number of workers for dataset loading')
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--restart_prob", type=float, default=0.2)
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--p", type=float, default=0.6, help="exponential moving average weight")
    parser.add_argument("--q", type=float, default=1.4, help="exponential moving average weight")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")
    parser.add_argument("--num_samples", type=int, default=1, help="clip norm")
    parser.add_argument("--num_neg_samples", type=int, default=7, help="clip norm") # --num_neg_samples n2v_walk_pos_neg
    parser.add_argument("--T", type=int, default=4, help="clip norm")
    parser.add_argument("--neg_sample_stra", type=str, default="uniform_neg", help="clip norm")
    parser.add_argument("--env", type=str, default="normal", help="clip norm")
    parser.add_argument("--construct_big_graph", default=False, action='store_true', help="clip norm")
    parser.add_argument("--select_other_node_stra", default="last_one", type=str, help="clip norm")
    parser.add_argument("--select_other_node_hop", default=5, type=int, help="clip norm")
    parser.add_argument("--rw_hops", default=64, type=int, help="clip norm")
    parser.add_argument("--num_path", default=7, type=int, help="clip norm")
    parser.add_argument("--num_com_negs", default=32, type=int, help="clip norm")
    parser.add_argument("--num_devices", default=4, type=int, help="clip norm")
    parser.add_argument("--data_para", default=False, action='store_true', help="clip norm")
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')


    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    hvd.init()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(hvd.local_rank())

    # l1 = args.num_layer - 1
    # l2 = l1 + args.csize
    # args.gpu = args.device
    # print(args.mode)
    # print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))
    # args.num_samples = 1

    # for automl only
    # worker_path = os.getenv("JIZHI_WORKSPACE_PATH")
    # # job_param_path = './job_param.json'
    # job_param_path = os.path.join(worker_path, "job_param.json")
    # with open(job_param_path, 'r') as f:
    #     hyper_params = json.load(f)
    #     rw_hops = hyper_params["rw_hops"]
    #     num_path = hyper_params["num_path"]
    #     lr = hyper_params["lr"]
    #     # hidden2 = hyper_params["hidden2"]
    # # l1 = args.num_layer - 1
    # # l2 = l1 + args.csize
    # # args.gpu = args.device
    # # print(args.mode)
    # # print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))
    # # args.num_samples = 1
    # args.num_path = num_path
    # args.rw_hops = rw_hops
    # args.lr = lr

    # hvd: scale the learning rate by number workers.
    args.lr = args.lr * args.num_devices

    args.output_model_file = "temp/contrast_sim_based_multi_pos_v3_sample_stra_{}_{:d}_num_neg_samples_{:d}_T_{:d}_sel_other_stra_{}_hop_{:d}_rstprob_{}_p_{:.1f}_q_{:.1f}_rw_hops_{:d}_num_path_{:d}_dl_other_p_w2v_neg_p_with_eval".format(
        args.neg_sample_stra,
        args.num_samples,
        args.num_neg_samples,
        args.T,
        args.select_other_node_stra,
        args.select_other_node_hop,
        str(args.restart_prob),
        args.p,
        args.q,
        args.rw_hops,
        args.num_path)

    args.log_loss_file = "log_loss/" + args.output_model_file.split("/")[-1]
    args.log_neg_samples_idx_file = "log_loss/" + args.output_model_file.split("/")[-1] + "_neg_sampled_idx"
    args.los_batch = []

    if args.env == "jizhi":
        args.output_model_file = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/temp_model/" + args.output_model_file.split("/", 1)[1]
        args.log_loss_file = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/temp_model/" + args.log_loss_file.split("/", 1)[1]
        args.log_neg_samples_idx_file = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/temp_model/" + args.log_neg_samples_idx_file.split("/", 1)[1]

    print("num_com_negs", args.num_com_negs)

    #### form of the model_file's name: "temp/contrast_sim_based_vallina_rwr_with_neg_{:d}_num_neg_samples_{:d}".format(
        # args.num_samples, args.num_neg_samples) --- want to explore the relationship between number of negative samples and the performance.
    ### "temp/contrast_sim_based_neg_{}_{:d}_num_neg_samples_{:d}".format(args.neg_sample_stra,
        # args.num_samples, args.num_neg_samples) --- want to explore the relationship between sampling strategy and performace.

    # set up dataset and transform function.
    dataset_root_path = "dataset/" + args.dataset
    if args.env == "jizhi":
        dataset_root_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/" + args.dataset
    dataset = MoleculeDatasetForContrast(dataset_root_path, dataset=args.dataset, transform=None,
                                         extract_sim=True, num_samples=args.num_samples, k=args.num_neg_samples,
                                         extract_sim_type=args.neg_sample_stra, with_neg=True, T=args.T,
                                         construct_big_graph=args.construct_big_graph,
                                         select_other_node_stra=args.select_other_node_stra,
                                         restart_prob=args.restart_prob,
                                         num_hops=args.num_hops,
                                         neg_p_q=[args.p, args.q],
                                         rw_hops=args.rw_hops,
                                         num_path=args.num_path,
                                         args=args)
    loader = DataLoaderContrastSimBasedMultiPosComNeg(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, args=args)

    # val_dataset = MoleculeDataset("dataset/" + args.val_dataset, dataset=args.val_dataset)

    train_loader, valid_loader, test_loader = get_val_dataloader(args)
    print("have got data loaders!")

    # set up models, one for pre-training and one for context embeddings
    # model_k = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
    #               gnn_type=args.gnn_type).to(device)
    model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                  gnn_type=args.gnn_type) #.to(device)
    # model_q = torch_geometric.nn.DataParallel(model_q.cuda())
    # model_q = model_q.to(device)
    model_q = model_q.cuda()
    # model_q = DataParallel(model_q)
    # model_q = model_q.cuda()
    # model_dis = Discriminator(args.emb_dim if args.JK != "concat" else args.emb_dim * (1 + 1)).to(device)
    # model_attn = atten_module(input_dim=args.emb_dim).to(device)
    # pred_fiel = "temp/contrast_sim_based_neg_rwr_neg_1_num_neg_samples_64_2.pth"
    # model_q.load_state_dict(torch.load(pred_fiel))
    # print("loading model succ!")

    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)

    # for horovod: broadcast parameters for model and broadcast optimizer state
    hvd.broadcast_parameters(model_q.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer_q, root_rank=0)

    optimizer_q = hvd.DistributedOptimizer(optimizer_q,
                                         named_parameters=model_q.named_parameters(),
                                         compression=hvd.Compression.none,
                                         op=hvd.Adasum,# if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    best_test_acc = 0.0
    best_test_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_q, loader, optimizer_q, device, epoch)
        # print(train_loss, train_acc)
        if epoch % 1 == 0 and hvd.rank() == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))
        if epoch % 1 == 0 and os.path.exists(args.output_model_file + "_{}.pth".format(str(epoch))):
            test_acc, val_epoch = start_val([train_loader, valid_loader, test_loader], args, epoch)
            if test_acc > best_test_acc:
                best_test_epoch = epoch
                best_test_acc = test_acc
            print('time: %s, epoch = %d, best_test_epoch = %d, auc = %.4f' % (
            datetime.datetime.now(), epoch, best_test_epoch, best_test_acc))

    if not args.output_model_file == "" and hvd.rank() == 0:
        torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
