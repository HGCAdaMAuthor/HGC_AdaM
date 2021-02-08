#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import os
import time

import numpy as np
# import tensorboard_logger as tb_logger
import torch

from data_util import batcher
from graph_dataset import (
    CogDLGraphDataset,
    CogDLGraphClassificationDataset,
    GraphDataset,
)
# from models.gat import UnsupervisedGAT
# from models.gcn import UnsupervisedGCN
from models.graph_encoder import GraphEncoder
# from models.mpnn import UnsupervisedMPNN
# from NCE.NCEAverage import MemoryMoCo
# from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss
# from train_graph_moco import option_update, parse_option
from graph_dataset import GRAPH_CLASSIFICATION_DSETS
from util import AverageMeter, adjust_learning_rate


def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """

    model.eval()

    emb_list = []
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q = graph_q.to(torch.device(opt.gpu))
        graph_k = graph_k.to(torch.device(opt.gpu))

        with torch.no_grad():
            feat_q, all_outputs_q = model(graph_q, return_all_outputs=True)
            feat_k, all_outputs_k = model(graph_k, return_all_outputs=True)
            if opt.return_all_outputs:
                all_outputs_q = torch.cat(all_outputs_q, dim=1)
                all_outputs_k = torch.cat(all_outputs_k, dim=1)

        assert feat_q.shape == (bsz, opt.hidden_size)
        if opt.return_all_outputs:
            emb_list.append(((all_outputs_q + all_outputs_k) / 2).detach().cpu())
        else:
            emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def main():
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off  CUDA_VISIBLE_DIVICES="2,3,4,5" taskset -c 15-18 python test_graph_moco.py --gpu 3 --load-path /home/lxyww7/graph-self-learn/graph-self-learn/saved/Pretrain_moco_True_con_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_98.pth
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--dataset", type=str, default="usa_airport", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index-rand20intop200"] + GRAPH_CLASSIFICATION_DSETS)
    parser.add_argument("--return-all-outputs", action="store_true", help="concat all layer's pooled output as final embedding")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    args_test = parser.parse_args()

    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    assert args_test.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args_test.gpu))

    if args_test.dataset == "dgl":
        train_dataset = GraphDataset(
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
        )
    else:
        if args_test.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = CogDLGraphClassificationDataset(
                dataset=args_test.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
        else:
            train_dataset = CogDLGraphDataset(
                dataset=args_test.dataset,
                rw_hops=args.rw_hops,
                num_path=12,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
    args.batch_size=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
    n_data = len(train_dataset)

    model = GraphEncoder(
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
        gnn_model=args.model,
        norm=args.norm,
        degree_input=args.degree_input,
    )

    model = model.cuda(args_test.gpu)

    model.load_state_dict(checkpoint["model"])

    del checkpoint
    torch.cuda.empty_cache()

    args.gpu = args_test.gpu
    args.return_all_outputs = args_test.return_all_outputs
    emb = test_moco(train_loader, model, args)
    model_folder = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/con_dgl"
    np.save(os.path.join(model_folder, args_test.dataset), emb.numpy())


if __name__ == "__main__":
    main()
