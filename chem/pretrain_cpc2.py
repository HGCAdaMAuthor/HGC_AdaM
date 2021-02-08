import argparse

from loader import MoleculeDataset, ContrastDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

# from model import GNN
from model import GNN
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import  ContrastMask6

from dataloader import DataLoaderContrast6

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
from contrastive_model import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform
from utils2 import from_rep_to_subgraph_rep

LARGE_NUM = 1e9
criterion_dis = nn.BCEWithLogitsLoss()


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, dis_type=1):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()
        self.dis_type = dis_type

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary, aa=None):
        if self.dis_type == 1:
            h = torch.matmul(summary, self.weight)
            return torch.sum(x*h, dim = 1)
        elif self.dis_type == 2:
            assert aa != None and aa.size(0) == x.size(0)
            h = torch.matmul(summary, self.weight)
            logists = torch.matmul(x, h.t())
            # x.size(0) x summary.size(0)
            target = torch.zeros(logists.size(), device=x.device)
            target[torch.arange(0, target.size(0), device=x.device), aa] = 1.
            return logists.view(-1), target.view(-1)


class Discriminator_ip(nn.Module):
    def __init__(self, hidden_dim, dis_type=1):
        super(Discriminator_ip, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()
        self.dis_type = dis_type

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary, aa=None):
        h = torch.matmul(summary, self.weight)
        logists = torch.matmul(x, h.t())
        return logists


class atten_module(nn.Module):
    def __init__(self, input_dim, value_dim=None, query_dim=None, key_dim=None):
        super(atten_module, self).__init__()
        if value_dim == None:
            value_dim  = input_dim
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

#criterion = nn.BCEWithLogitsLoss()


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

# patch global repel and attract ; patch and patch ---- instance discrimination


def train(args, model, model_dis, loader, optimizer, optimizer_dis, device, criterion, epoch, model_dis_ip, optimizer_dis_ip):
    # model_k.eval()
    # model_q.train()
    model.train()
    model_dis.train()
    model_dis_ip.train()

    kk = args.kk   # num of node-context pairs
    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    offsets = []
    for j in range(1, kk):
        art = torch.LongTensor([j if (ii + j) < kk else j - kk for ii in range(kk)]).to(device)
        offsets.append(art)

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        rep = model(batch.x, batch.edge_index, batch.edge_attr)

        rep_global = pool_func(rep, batch.batch, args.pooling)
        loss = 0.0

        for j in range(batch.k):
            rep_fr = pool_func(rep[batch.dis_to_nodes_fr[j].to(rep.device)], batch.nodes_fr_to_subgraph[j].to(rep.device), args.pooling)
            rep_to = pool_func(rep[batch.dis_to_nodes_to[j].to(rep.device)], batch.nodes_to_to_subgraph[j].to(rep.device), args.pooling)
            j_bs = rep_fr.size(0)
            assert rep_fr.size(0) == rep_to.size(0)
            logits = model_dis_ip(rep_fr, rep_to)
            logits = logits / T
            target = torch.arange(0, j_bs, device=rep_fr.device)
            loss = loss + F.cross_entropy(logits, target)

        # rep_context = pool_func(rep[batch.context_nodes], batch.context_nodes_to_graph_idx, args.pooling)
        # rep_center_nodes = rep[batch.center_nodes]
        # bs = rep_global.size(0)
        # logits = model_dis_ip(rep_center_nodes, rep_context)
        # num_pairs = rep_center_nodes.size(0)
        # arr = torch.arange(0, num_pairs, device=rep_global.device)
        # for j, offset in enumerate(offsets):
        #     logits[arr, arr + offset.repeat(bs)] = -args.inf
        # logits = logits / T
        # target = arr
        # loss = F.cross_entropy(logits, target=target)

        if args.use_infomax == True:
            summary_emb = F.sigmoid(rep_global)
            logits, target = model_dis(rep, summary_emb, batch.batch)
            loss = loss + criterion_dis(logits, target)

        optimizer.zero_grad()

        optimizer_dis_ip.zero_grad()
        if args.use_infomax == True:
            optimizer_dis.zero_grad()
        # loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
        grad_norm = clip_grad_norm(model_dis.parameters(), args.clip_norm)
        global_step = (epoch - 1) * args.batch_size + step
        lr_this_step = args.lr * warmup_linear(
            global_step / (args.epochs * args.batch_size), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        balanced_loss_accum += loss.detach().cpu().item()

        optimizer.step()
        optimizer_dis_ip.step()
        if args.use_infomax == True:
            optimizer_dis.step()
        # moment_update(model_q, model_k, args.alpha)
        torch.cuda.synchronize()
        print(loss.detach().cpu().item())

    return balanced_loss_accum / step, acc_accum / (step * 2)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
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
    parser.add_argument('--output_model_file', type=str, default='temp/contrast79', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--clip-norm", type=float, default=100.0, help="clip norm")
    parser.add_argument("--eps", type=float, default=1e-12, help="epsilon, help with the numerical stability.")
    parser.add_argument("--inf", type=float, default=1e12, help="epsilon, help with the numerical stability.")
    parser.add_argument("--kk", type=int, default=5, help="epsilon, help with the numerical stability.")
    parser.add_argument("--use_infomax", default=False, action="store_true", help="epsilon, help with the numerical stability.")
    parser.add_argument("--step", type=int, default=1,
                        help="epsilon, help with the numerical stability.")

    args = parser.parse_args()

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
    args.output_model_file = "temp/con_context_cpc2_k_{:d}_step_{:d}_infomax_{}_gnn_type_{}".format(args.kk, args.step, str(args.use_infomax), str(args.gnn_type))

    # set up dataset and transform function.
    dataset = ContrastDataset("dataset/" + args.dataset, args=args, dataset=args.dataset,
                              transform=ContrastMask6(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge, k=args.kk, step=args.step))
    print(len(dataset))
    loader = DataLoaderContrast6(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                          gnn_type=args.gnn_type).to(device)
    # model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
    #                     gnn_type=args.gnn_type).to(device)
    model_dis = Discriminator(args.emb_dim if args.JK != "concat" else args.emb_dim * (1 + 1), dis_type=2).to(device)
    model_dis_ip = Discriminator_ip(args.emb_dim if args.JK != "concat" else args.emb_dim * (1 + 1)).to(device)


    # model_attn = atten_module(input_dim=args.emb_dim).to(device)

    n_data = dataset.total
    # contrast = MemoryMoCo(inputSize=args.emb_dim, outputSize=n_data, K=args.nce_k, T=args.nce_t, use_softmax=True, device=args.device)
    # contrast = contrast.cuda(args.device)
    # criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    # criterion = criterion.cuda(args.device)
    criterion = nn.NLLLoss()
    criterion = criterion.cuda(args.device)
    # set up optimizer for the two GNNs
    # optimizer_k = optim.Adam(model_k.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dis_ip = optim.Adam(model_dis_ip.parameters(), lr=args.lr, weight_decay=args.decay)

    # optimizer_attn = optim.Adam(model_attn.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model, model_dis, loader, optimizer, optimizer_dis, device, criterion, epoch, model_dis_ip, optimizer_dis_ip)
        print(train_loss, train_acc)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()