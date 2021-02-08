#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import copy
import os
import time
import warnings
import datetime

import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import json

import data_util
import dgl
from graph_dataset import (
    GRAPH_CLASSIFICATION_DSETS,
    CogDLGraphClassificationDataset,
    CogDLGraphClassificationDatasetLabeled,
    CogDLGraphDataset,
    CogDLGraphDatasetLabeled,
    GraphDataset,
    LoadBalanceGraphDataset,
    worker_init_fn,
    GraphDatasetOtherContrast,
    GraphDatasetOtherContrastOtherSample,
    GraphDatasetOtherContrastMolDataset,
    worker_init_fn_mol_data,
    GraphDatasetOtherContrastSimReSam,
    GraphDatasetOtherContrastSimReSamL
)
from models.graph_encoder import GraphEncoder
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from util import AverageMeter, adjust_learning_rate, warmup_linear

# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume h-index-rand-1
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="con_dgl_la", choices=["con_dgl_resam_l", "con_dgl_resam", "con_dgl_mol_data", "con_dgl_other_sample", "con_dgl_la", "dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index-rand20intop200"] + GRAPH_CLASSIFICATION_DSETS)

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    # parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")
    parser.add_argument("--degree-input", action="store_true", help="concat node degree embeddings to input features")

    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw_hops", type=int, default=256)
    parser.add_argument("--num_path", type=int, default=7)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model-path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb-path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    # finetune setting
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, nargs='+', help="GPU id to use.")

    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--env", type=str, default="normal")
    parser.add_argument("--automl", default=False, action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp,
        opt.moco,
        opt.dataset,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.hidden_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.rw_hops,
        opt.restart_prob,
        opt.aug,
        opt.finetune,
        opt.degree_embedding_size if opt.degree_input else 0,
        opt.positional_embedding_size,
        opt.alpha
    )

    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        # opt.model_folder = os.path.join("saved", opt.model_name)

        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # opt.tb_folder = os.path.join("tensorboard", opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.num_path = 7
    opt.lr = 0.001 # what is 0.00824??
    opt.rw_hops = 14
    opt.num_path = 12
    if opt.automl:
        worker_path = os.getenv("JIZHI_WORKSPACE_PATH")
        # job_param_path = './job_param.json'
        job_param_path = os.path.join(worker_path, "job_param.json")
        with open(job_param_path, 'r') as f:
            hyper_params = json.load(f)
            # batch_size = hyper_params["batch_size"]
            # # dropout_ratio = hyper_params["dropout_ratio"]
            lr = hyper_params["lr"]
            rw_hops = hyper_params["rw_hops"]
            num_path = hyper_params["num_path"]

        # opt.batch_size = batch_size
        # # args.dropout_ratio = dropout_ratio
        opt.lr = lr
        opt.rw_hops = rw_hops
        opt.num_path = num_path
    opt.learning_rate = 0.001
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_finetune(
    epoch,
    train_loader,
    model,
    output_layer,
    criterion,
    optimizer,
    output_layer_optimizer,
    sw,
    opt,
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, y = batch

        graph_q = graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
        out = output_layer(feat_q)

        loss = criterion(out, y)

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        f1_meter.update(f1, bsz)
        epoch_f1_meter.update(f1, bsz)
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(graph_q.number_of_nodes() / bsz, bsz)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "f1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    f1=f1_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            sw.add_scalar("ft_loss", loss_meter.avg, global_step)
            sw.add_scalar("ft_f1", f1_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("lr", lr_this_step, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            #  sw.add_scalar(
            #      "learning_rate", optimizer.param_groups[0]["lr"], global_step
            #  )
            loss_meter.reset()
            f1_meter.reset()
            graph_size.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def test_finetune(epoch, valid_loader, model, output_layer, criterion, sw, opt):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()

    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()

    for idx, batch in enumerate(valid_loader):
        graph_q, y = batch

        graph_q = graph_q.to(torch.device(opt.gpu))
        y = y.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size

        # ===================forward=====================

        with torch.no_grad():
            feat_q = model(graph_q)
            assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
            out = output_layer(feat_q)
        loss = criterion(out, y)

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        epoch_loss_meter.update(loss.item(), bsz)
        epoch_f1_meter.update(f1, bsz)

    global_step = (epoch + 1) * n_batch
    sw.add_scalar("ft_loss/valid", epoch_loss_meter.avg, global_step)
    sw.add_scalar("ft_f1/valid", epoch_f1_meter.avg, global_step)
    print(
        f"Epoch {epoch}, loss {epoch_loss_meter.avg:.3f}, f1 {epoch_f1_meter.avg:.3f}"
    )
    return epoch_loss_meter.avg, epoch_f1_meter.avg


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def train_moco(
    epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    print("start training ")
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch
        # print("get dta")
        if graph_q is None or graph_k is None:
            print("bad data batch")
            continue
        graph_q = graph_q.to(torch.device(opt.gpu))
        graph_k = graph_k.to(torch.device(opt.gpu))

        bsz = graph_q.batch_size


        if opt.moco:
            # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
            # if args.dataset == "con_dgl_resam_l":
            #     sim_scores = (feat_q * feat_k).sum(dim=-1) / (torch.norm(feat_q, p=2, dim=-1) *
            #                                                   torch.norm(feat_k, p=2, dim=-1))
            #     train_loader.dataset.renew_sims(graph_q, sim_scores)
            # try:
            #     # print("using moco, in forwarding")
            #     feat_q = model(graph_q)
            #     with torch.no_grad():
            #         feat_k = model_ema(graph_k)
            #
            #     out = contrast(feat_q, feat_k)
            #     prob = out[:, 0].mean()
            # except:
            #     print("bad forwarding")
            #     continue
        else:
            # ===================Negative sampling forward=====================
            feat_q = model(graph_q)
            feat_k = model(graph_k)

            out = torch.matmul(feat_k, feat_q.t()) / opt.nce_t
            prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg


# def main(args, trial):
def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            pretrain_args = checkpoint["opt"]
            pretrain_args.fold_idx = args.fold_idx
            pretrain_args.gpu = args.gpu
            pretrain_args.finetune = args.finetune
            pretrain_args.resume = args.resume
            pretrain_args.cv = args.cv
            pretrain_args.dataset = args.dataset
            pretrain_args.epochs = args.epochs
            pretrain_args.num_workers = args.num_workers
            if args.dataset in GRAPH_CLASSIFICATION_DSETS:
                # HACK for speeding up finetuning on graph classification tasks
                pretrain_args.num_workers = 0
            pretrain_args.batch_size = args.batch_size
            pretrain_args.automl = args.automl
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    args = option_update(args)
    print(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)
    args.env = "jizhi"
    if args.finetune:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            dataset = CogDLGraphClassificationDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
                env=args.env
            )
            if args.dataset in ["imdb-multi", "rdt-5k", "imdb-binary", "rdt-b", "collab"]:
                labels = dataset.labels.tolist()
            else:
                labels = dataset.dataset.data.y.tolist()
        else:
            dataset = CogDLGraphDatasetLabeled(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                num_path=args.num_path,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            labels = dataset.data.y.argmax(dim=1).tolist()

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        assert (
            0 <= args.fold_idx and args.fold_idx < 10
        ), "fold_idx must be from 0 to 9."
        trainn_idx, test_idx = idx_list[args.fold_idx]
        # train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, test_idx)

        trainn_labels = torch.tensor(labels)[torch.tensor(trainn_idx, dtype=torch.long)]
        skfs = StratifiedKFold(n_splits=9, shuffle=True, random_state=args.seed)
        idx_listt = []
        for idx in skfs.split(np.zeros(len(trainn_labels)), trainn_labels):
            idx_listt.append(idx)
        train_idx, valid_real_idx = idx_listt[0]
        trainn_idx_tsr = torch.tensor(trainn_idx, dtype=torch.long)
        train_idx = trainn_idx_tsr[train_idx]
        valid_idx = trainn_idx_tsr[valid_real_idx]
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_real_dataset = torch.utils.data.Subset(dataset, valid_idx)

    elif args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file="./data_bin/dgl/small.bin",
            num_copies=args.num_copies,
        )
    elif args.dataset == "con_dgl":
        train_dataset = GraphDatasetOtherContrast(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    elif args.dataset == "con_dgl_resam":
        train_dataset = GraphDatasetOtherContrastSimReSam(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    elif args.dataset == "con_dgl_resam_l":
        train_dataset = GraphDatasetOtherContrastSimReSamL(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    elif args.dataset == "con_dgl_other_sample":
        train_dataset = GraphDatasetOtherContrastOtherSample(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_path=args.num_path,
            env=args.env
        )
        # train_dataset = GraphDataset(
        #     rw_hops=args.rw_hops,
        #     restart_prob=args.restart_prob,
        #     positional_embedding_size=args.positional_embedding_size,
        # )
    elif args.dataset == "con_dgl_mol_data":
        train_dataset = GraphDatasetOtherContrastMolDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    elif args.dataset == "con_dgl_la":
        train_dataset = GraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    else:
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = CogDLGraphClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
        else:
            train_dataset = CogDLGraphDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )


    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_util.labeled_batcher()
        if args.finetune
        else data_util.batcher(),
        shuffle=True if args.finetune else False,
        num_workers=args.num_workers,
        worker_init_fn=None
        if args.finetune or (args.dataset != "dgl" and args.dataset != "con_dgl_mol_data")
        else (worker_init_fn if args.dataset == "dgl" else worker_init_fn_mol_data),
    )
    if args.finetune:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=data_util.labeled_batcher(),
            num_workers=args.num_workers,
        )
        valid_real_loader = torch.utils.data.DataLoader(
            dataset=valid_real_dataset,
            batch_size=args.batch_size,
            collate_fn=data_util.labeled_batcher(),
            num_workers=args.num_workers,
        )
    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None

    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=args.degree_input,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
    ).cuda(args.gpu)

    if args.finetune:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
        criterion = criterion.cuda(args.gpu)

    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)

    if args.finetune:
        output_layer = nn.Linear(
            in_features=args.hidden_size, out_features=dataset.num_classes
        )
        output_layer = output_layer.cuda(args.gpu)
        output_layer_optimizer = torch.optim.Adam(
            output_layer.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

        def clear_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.reset_running_stats()

        model.apply(clear_bn)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        # print("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume, map_location="cpu")
        # checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        contrast.load_state_dict(checkpoint["contrast"])
        if args.moco:
            model_ema.load_state_dict(checkpoint["model_ema"])

        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
        del checkpoint
        torch.cuda.empty_cache()

    # tensorboard
    #  logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    sw = SummaryWriter(args.tb_folder)
    #  plots_q, plots_k = zip(*[train_dataset.getplot(i) for i in range(5)])
    #  plots_q = torch.cat(plots_q)
    #  plots_k = torch.cat(plots_k)
    #  sw.add_images('images/graph_q', plots_q, 0, dataformats="NHWC")
    #  sw.add_images('images/graph_k', plots_k, 0, dataformats="NHWC")

    # routine
    best_acc = 0.0
    best_val_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        if args.finetune:
            loss, _ = train_finetune(
                epoch,
                train_loader,
                model,
                output_layer,
                criterion,
                optimizer,
                output_layer_optimizer,
                sw,
                args,
            )
        else:
            loss = train_moco(
                epoch,
                train_loader,
                model,
                model_ema,
                contrast,
                criterion,
                optimizer,
                sw,
                args,
            )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        if epoch % 1 == 0:
            if args.finetune:
                valid_loss, valid_f1 = test_finetune(
                    epoch, valid_loader, model, output_layer, criterion, sw, args
                )
                valid_real_loss, valid_real_f1 = test_finetune(
                    epoch, valid_real_loader, model, output_layer, criterion, sw, args
                )
                # best_acc = max(best_acc, valid_f1)
                if valid_real_f1 > best_val_acc:
                    best_val_acc = valid_real_f1
                    best_acc = valid_f1
                print("{:d}\tloss:{:.4f}\tvalid_f1:{:.4f}\ttest_f1:{:.4f}".format(epoch,
                                                                                  valid_loss,
                                                                                  valid_real_f1,
                                                                                  valid_f1))
        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if args.moco:
                state["model_ema"] = model_ema.state_dict()
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print("==> Saving...")
        state = {
            "opt": args,
            "model": model.state_dict(),
            "contrast": contrast.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if args.moco:
            state["model_ema"] = model_ema.state_dict()
        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()

    if args.finetune:
        valid_loss, valid_f1 = test_finetune(
            epoch, valid_loader, model, output_layer, criterion, sw, args
        )
        # print(best_acc)
        ppa = best_acc if args.automl else best_val_acc
        print('time: %s, acc = %.4f' % (
            datetime.datetime.now(), ppa))
        return valid_f1



if __name__ == "__main__":

    warnings.simplefilter("once", UserWarning)
    args = parse_option()

    if args.cv:
        gpus = args.gpu

        def variant_args_generator():
            for fold_idx in range(10):
                args.fold_idx = fold_idx
                args.num_workers = 0
                args.gpu = gpus[fold_idx % len(gpus)]
                yield copy.deepcopy(args)

        # f1 = Parallel(n_jobs=5)(
        #     delayed(main)(args) for args in variant_args_generator()
        # )
        f1 = [main(args) for args in variant_args_generator()]
        print(f1)

        print(f"Mean = {np.mean(f1)}; Std = {np.std(f1)}")
    else:
        args.gpu = args.gpu[0]
        main(args)
    # import optuna
    # def objective(trial):
    #     args.epochs = 50
    #     args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    #     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    #     args.restart_prob = trial.suggest_uniform('restart_prob', 0.5, 1)
    #     # args.alpha = 1 - trial.suggest_loguniform('alpha', 1e-4, 1e-2)
    #     return main(args, trial)

    # study = optuna.load_study(study_name='cat_prone', storage="sqlite:///example.db")
    # study.optimize(objective, n_trials=20)