import argparse

from loader import MoleculeDataset, ContrastDataset, MoleculeDatasetForContrast

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

from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask

from dataloader import DataLoaderSubstructContext, DataLoaderMasking, DataLoaderContrastSimBased

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
from contrastive_model import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform

LARGE_NUM = 1e9
criterion_dis = nn.BCEWithLogitsLoss()


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


def train(args, model_k, model_q, loader, optimizer_q, device, epoch):
    model_k.eval()
    model_q.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        num_samples = len(batch.sim_x)
        rep_q = model_q(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
        rep_q_global = pool_func(rep_q, batch.batch.to(device), args.pooling)
        rep_q_global = rep_q_global / (torch.norm(rep_q_global, p=2, dim=1, keepdim=True) + 1e-9)
        loss = 0.0
        assert num_samples >= 1

        for j in range(num_samples):
            sim_x, sim_edge_index, sim_edge_attr = batch.sim_x[j], batch.sim_edge_index[j], batch.sim_edge_attr[j]
            rep_k = model_q(sim_x.to(device), sim_edge_index.to(device), sim_edge_attr.to(device))
            rep_k_global = pool_func(rep_k, batch.batch_sim[j].to(device), args.pooling) # n x fea_dim
            rep_k_global = rep_k_global / (torch.norm(rep_k_global, p=2, dim=1, keepdim=True) + 1e-9) # n x fea_dim
            bs = rep_k_global.shape[0]
            mask = torch.zeros((bs, bs)).cuda(args.device) # bs x bs
            mask[torch.arange(bs).cuda(args.device), torch.arange(bs).cuda(args.device)] = 1 # set self to 1
            labels = torch.arange(0, bs).cuda(args.device)
            logits_kq = torch.matmul(rep_k_global, rep_q_global.t()) / T
            logits_qk = torch.matmul(rep_q_global, rep_k_global.t()) / T
            loss_kq = F.nll_loss(F.log_softmax(logits_kq, dim=1), labels)
            loss_qk = F.nll_loss(F.log_softmax(logits_qk, dim=1), labels)
            loss += (loss_kq + loss_qk)


        # batch = batch.to(device)
        # # hard to define the supervised loss for nodes or edges since if we corrupt it randomly it is hard to
        # # get the mask type for each graph in the batch....
        #
        # # with torch.no_grad():
        # #     rep_k = model_k(batch.xa, batch.edge_index, batch.edgea_attr)
        # #     rep_k_global = pool_func(rep_k, batch.node_idx_tensor, args.pooling)
        #
        # rep_q = model_q(batch.x, batch.edge_index, batch.edge_attr)
        # # rep_q = model_attn(rep_q, batch.mask)
        #
        # rep_k = model_q(batch.sim_x, batch.sim_edge_index, batch.sim_edge_attr)
        # # rep_k = model_attn(rep_k, batch.mask)
        #
        #
        # rep_k_global = rep_k_global / (torch.norm(rep_k_global, p=2, dim=1, keepdim=True) + 1e-9)
        # rep_q_global = rep_q_global / (torch.norm(rep_q_global, p=2, dim=1, keepdim=True) + 1e-9)
        # bs = rep_k_global.shape[0]
        #
        # mask = torch.zeros((bs, bs)).cuda(args.device)
        # mask[torch.arange(bs).cuda(args.device), torch.arange(bs).cuda(args.device)] = 1
        # # labels[torch.arange(bs).cuda(args.device), torch.arange(bs).cuda(args.device) + bs] = 1
        # labels = torch.arange(bs, 2 * bs).cuda(args.device)
        # logits_kk = torch.matmul(rep_k_global, rep_k_global.t()) / T
        # logits_kk = logits_kk - mask * LARGE_NUM
        # logits_qq = torch.matmul(rep_q_global, rep_q_global.t()) / T
        # logits_qq = logits_qq - mask * LARGE_NUM
        # logits_kq = torch.matmul(rep_k_global, rep_q_global.t()) / T
        # logits_qk = torch.matmul(rep_q_global, rep_k_global.t()) / T
        #
        # pred_k = torch.cat((logits_kk, logits_kq), dim=1)
        # pred_q = torch.cat((logits_qq, logits_qk), dim=1)
        #
        # loss_k = F.nll_loss(F.log_softmax(pred_k, dim=1), labels)
        # loss_q = F.nll_loss(F.log_softmax(pred_q, dim=1), labels)
        # loss = (loss_k + loss_q)

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
    parser.add_argument('--output_model_file', type=str, default='temp/contrast2', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")
    parser.add_argument("--num_samples", type=int, default=7, help="clip norm")
    parser.add_argument("--num_neg_samples", type=int, default=7, help="clip norm")
    parser.add_argument("--T", type=int, default=4, help="clip norm")
    parser.add_argument("--num_hops", type=int, default=2, help="clip norm")
    parser.add_argument("--neg_sample_stra", type=str, default="uniform_neg", help="clip norm")
    parser.add_argument("--shuffle", type=bool, default=True, help="clip norm")
    parser.add_argument("--with_neg", default=False, action='store_true', help="clip norm")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # l1 = args.num_layer - 1
    # todo: 或者应该如何 正例构造出来的graph其实是可以看作是相似度在一层层地传递 但负例的graph可能会有负负得正的结果？
    # l2 = l1 + args.csize
    # args.gpu = args.device
    # print(args.mode)
    # print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))

    args.output_model_file = "temp/contrast_sim_shuffle_{}_hops_{:d}_{}_{:d}_gnn_{}_no_prob_with_neg_{}".format(str(args.shuffle), args.num_hops, args.neg_sample_stra, args.num_samples, args.gnn_type, str(args.with_neg))

    dataset = MoleculeDatasetForContrast("dataset/" + args.dataset, dataset=args.dataset,
                                         transform=None, extract_sim=True, num_samples=args.num_samples,
                                         k=args.num_neg_samples, num_hops=args.num_hops,
                                         extract_sim_type=args.neg_sample_stra, with_neg=args.with_neg, T=args.T)
    # todo: change shuffle to true and compare the performance
    loader = DataLoaderContrastSimBased(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    #
    model_k = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                  gnn_type=args.gnn_type).to(device)
    model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                  gnn_type=args.gnn_type).to(device)

    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_k, model_q, loader, optimizer_q, device, epoch)
        print(train_loss, train_acc)
        if epoch % 1 == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
