import argparse

from loader import MoleculeDataset, ContrastDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

# from model import GNN
from model import GNN, Mole_Flow
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask, ExtractFlowContrastData

from dataloader import DataLoaderSubstructContext, DataLoaderMasking, DataLoaderContrastFlow

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
from contrastive_model import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform

LARGE_NUM = 1e9
criterion_dis = nn.BCEWithLogitsLoss()

def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)


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


def train(args, model_k, model_q, loader, optimizer_q, optimizer_k, device, epoch):
    model_k.train()
    model_q.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # hard to define the supervised loss for nodes or edges since if we corrupt it randomly it is hard to
        # get the mask type for each graph in the batch....

        # with torch.no_grad():
        #     rep_k = model_k(batch.xa, batch.edge_index, batch.edgea_attr)
        #     rep_k_global = pool_func(rep_k, batch.node_idx_tensor, args.pooling)
        rep_real = model_q(batch.x, batch.edge_index, batch.edge_attr)
        rep_sim = model_k(batch.dist_x, batch.dist_edge_index, batch.dist_edge_attr)
        rep_real_global = pool_func(rep_real, batch.batch, args.pooling)
        rep_sim_global = pool_func(rep_sim, batch.dist_batch, args.pooling)

        rep_k_global = rep_real_global / (torch.norm(rep_real_global, p=2, dim=1, keepdim=True) + 1e-9)
        rep_q_global = rep_sim_global / (torch.norm(rep_sim_global, p=2, dim=1, keepdim=True) + 1e-9)
        bs = rep_k_global.shape[0]

        mask = torch.zeros((bs, bs)).cuda(args.device)
        # labels = torch.zeros((bs, 2 * bs)).cuda(args.device)
        mask[torch.arange(bs).cuda(args.device), torch.arange(bs).cuda(args.device)] = 1
        # labels[torch.arange(bs).cuda(args.device), torch.arange(bs).cuda(args.device) + bs] = 1
        labels = torch.arange(bs, 2 * bs).cuda(args.device)
        logits_kk = torch.matmul(rep_k_global, rep_k_global.t()) / T
        logits_kk = logits_kk - mask * LARGE_NUM
        logits_qq = torch.matmul(rep_q_global, rep_q_global.t()) / T
        logits_qq = logits_qq - mask * LARGE_NUM
        logits_kq = torch.matmul(rep_k_global, rep_q_global.t()) / T
        logits_qk = torch.matmul(rep_q_global, rep_k_global.t()) / T

        pred_k = torch.cat((logits_kk, logits_kq), dim=1)
        pred_q = torch.cat((logits_qq, logits_qk), dim=1)

        loss_k = F.nll_loss(F.log_softmax(pred_k, dim=1), labels)
        loss_q = F.nll_loss(F.log_softmax(pred_q, dim=1), labels)
        loss = (loss_k + loss_q)

        target = torch.arange(bs).cuda(args.device) + bs
        acc_k_num = compute_accuracy(pred_k, target, bs)
        acc_q_num = compute_accuracy(pred_q, target, bs)
        acc_accum += (acc_k_num + acc_q_num)

        # pred_k_logits = torch.argmax(pred_k, dim=1, keepdim=False)
        # pred_k_logits = pred_k_logits == torch.arange(bs).cuda(args.device) + bs
        # pred_q_logits = torch.argmax(pred_q, dim=1, keepdim=False)
        # pred_q_logits = pred_q_logits == torch.arange(bs).cuda(args.device) + bs

        # out = contrast(rep_q_global, rep_k_global)
        # out.to(args.device)
        # Contexts are represented by
        optimizer_q.zero_grad()
        optimizer_k.zero_grad()
        # loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model_q.parameters(), args.clip_norm)
        global_step = (epoch - 1) * args.batch_size + step
        lr_this_step = args.lr * warmup_linear(
            global_step / (args.epochs * args.batch_size), 0.1
        )
        for param_group in optimizer_q.param_groups:
            param_group["lr"] = lr_this_step

        balanced_loss_accum += loss.detach().cpu().item()

        optimizer_q.step()
        optimizer_k.step()
        # moment_update(model_q, model_k, args.alpha)
        # torch.cuda.synchronize()
        print(loss.detach().cpu().item())

        if not args.output_model_file == "":
            torch.save(model_q.state_dict(), args.output_model_file + "_{}_{}.pth".format(str(epoch), str(step)))

    return balanced_loss_accum / step, acc_accum / (step * 2)


import os
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--output_model_file', type=str, default='temp/contrast2', help='filename to output the model')
    parser.add_argument('--input_model_file', type=str, default='temp/contrast2', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    args = parser.parse_args()

    args.output_model_file = "temp_flow_fake/flow_GNN_model"
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize
    args.gpu = args.device
    print(args.mode)
    print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))

    # set up dataset and transform function.
    flow_model = Mole_Flow(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                           gnn_type=args.gnn_type)
    if not args.input_model_file.endswith(".pth"):
        print(type(torch.load(args.input_model_file)["model_state_dict"]))
        print(type(torch.load("temp/masked_strategies_variances_2_55.pth")))
        flow_model.load_state_dict(torch.load(args.input_model_file)["model_state_dict"])
    else:
        flow_model.load_state_dict(torch.load(args.input_model_file))

    ### what if we just train the flow model during the contrast training process?
    flow_model.eval()

    ### is is really ok to use model in cpu just to sample moleculars?

    dataset = ContrastDataset("dataset/" + args.dataset, args=args, dataset=args.dataset,
                              transform=ExtractFlowContrastData(flow_model))
    loader = DataLoaderContrastFlow(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model_k = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                          gnn_type=args.gnn_type).to(device)
    model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                        gnn_type=args.gnn_type).to(device)
    # flow_model = flow_model.to(device)

    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_k = optim.Adam(model_k.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_k, model_q, loader, optimizer_q, optimizer_k, args.device, epoch)
        print(train_loss, train_acc)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
