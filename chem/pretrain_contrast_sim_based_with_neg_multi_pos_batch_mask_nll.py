import argparse


import sys
print(sys.path)

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
# from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

# from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask

from dataloader import DataLoaderContrastSimBasedMultiPosPer, DataLoaderContrastSimBasedMultiPosComNeg

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

# from tensorboardX import SummaryWriter
# from contrastive_model import MemoryMoCo
# from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform

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
def train(args, model_q, linear_pred_atoms, loader, optimizer_q, optimizer_linear_pred_atoms, device, epoch):
    # model_k.eval()
    model_q.train()
    linear_pred_atoms.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    stt_time = time.time()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        print(step, time.time() - stt_time)
        # print("time for fetching data: ", time.time() - stt_time)
        # num_samples = len(batch.sim_x)
        batch = batch.to(device)
        num_samples = args.num_samples
        num_neg_samples = args.num_neg_samples * num_samples
        # num_neg_samples = 32
        num_neg_samples = args.num_com_negs
        # print("in main hhh")


        # loss = model_q.loss(batch, device, num_samples, num_neg_samples, args, T)

        neg_rep = model_q(batch.neg_x, batch.neg_edge_index, batch.neg_edge_attr)
        neg_rep_glb = pool_func(neg_rep, batch.neg_batch, args.pooling)
        node_rep = model_q(batch.x, batch.edge_index, batch.edge_attr)
        rep_glb = pool_func(node_rep, batch.batch, args.pooling)
        sim_rep = model_q(batch.sim_x, batch.sim_edge_index, batch.sim_edge_attr)

        sim_rep_glb = pool_func(sim_rep, batch.sim_batch, args.pooling)

        bs = rep_glb.size(0)

        rep_glb = rep_glb / torch.clamp(torch.norm(rep_glb, p=2, dim=1, keepdim=True), min=1e-9)
        sim_rep_glb = sim_rep_glb / torch.clamp(torch.norm(sim_rep_glb, p=2, dim=1, keepdim=True), min=1e-9)
        sim_rep_glb = sim_rep_glb.view(rep_glb.size(0), num_samples, -1) # bs x num_samples x rep_size
        neg_rep_glb = neg_rep_glb / torch.clamp(torch.norm(neg_rep_glb, p=2, dim=1, keepdim=True), min=1e-9)

        self_neg_mult_scores = torch.matmul(rep_glb, neg_rep_glb.transpose(1, 0)) / T
        # print(self_neg_mult_scores.size())
        self_neg_mult_scores_expand = torch.cat([torch.zeros((rep_glb.size(0), 1), dtype=torch.float32).to(device), self_neg_mult_scores],
                                                dim=-1)
        for i in range(num_samples):
            masked_node_idx = batch.masked_pos_idx[i, :] + 1
            self_neg_mult_scores_expand[torch.arange(rep_glb.size(0)).to(device), masked_node_idx] = -LARGE_NUM

        self_neg_mult_scores = self_neg_mult_scores_expand[:, 1:] # remove dim 0
        # bs x num_samples x num_neg_samples
        self_neg_mult_scores = self_neg_mult_scores.unsqueeze(1).repeat(1, num_samples, 1)
        # bs x 1 x rep_size

        self_pos_mult_scores = torch.sum(torch.mul(rep_glb.unsqueeze(1), sim_rep_glb), dim=-1, keepdim=True) / T
        # bs x num_samples x 1
        self_pos_neg_mult_scores_cat = torch.cat([self_pos_mult_scores, self_neg_mult_scores], dim=-1)
        # bs x num_samples x (1 + num_neg_samples)
        self_pos_neg_mult_scores_cat_expand = self_pos_neg_mult_scores_cat.view(-1, 1 + num_neg_samples)
        labels = torch.zeros((self_pos_neg_mult_scores_cat_expand.size(0), ), dtype=torch.long, device=device)
        loss = F.nll_loss(F.log_softmax(self_pos_neg_mult_scores_cat_expand, dim=-1), labels)

        sim_neg_mult_scores = torch.sum(torch.mul(sim_rep_glb.unsqueeze(2), neg_rep_glb.unsqueeze(0).unsqueeze(0)),
                                        dim=-1, keepdim=False) / T
        # bs x num_sampleds x num_neg_samples
        sim_self_mult_scores = torch.sum(torch.mul(sim_rep_glb, rep_glb.unsqueeze(1)), dim=-1, keepdim=True) / T
        # bs x num_samples x 1
        sim_neg_mult_scores_expand = torch.cat([torch.zeros((bs, num_samples, 1), device=device), sim_neg_mult_scores],
                                               dim=-1)
        for i in range(num_samples):
            masked_node_idx = batch.masked_pos_idx[i, :] + 1
            sim_neg_mult_scores_expand[torch.arange(bs, device=device), :, masked_node_idx] = -LARGE_NUM
        sim_neg_mult_scores = sim_neg_mult_scores_expand[:, :, 1:]
        sim_self_neg_mult_scores_cat = torch.cat([sim_self_mult_scores, sim_neg_mult_scores], dim=-1)
        sim_self_neg_mult_scores_cat_expand = sim_self_neg_mult_scores_cat.view(-1, 1 + num_neg_samples)
        sim_self_labels = torch.zeros((sim_self_neg_mult_scores_cat_expand.size(0), ), dtype=torch.long, device=device)
        loss += F.nll_loss(F.log_softmax(sim_self_neg_mult_scores_cat_expand, dim=-1), sim_self_labels)

        # inint perpare for node maksing
        num_nodes = batch.x.size(0)
        node_labels = torch.zeros((batch.x.size(0),), dtype=torch.long, requires_grad=False).to(batch.x.device)
        node_labels[:] = batch.x[:, 0]
        # print(torch.max(node_labels))

        tot_masked = int(num_nodes * args.mask_rate + 1)
        num_nodes_per_mask = int(tot_masked // args.mask_times)
        mask_times = args.mask_times
        if num_nodes_per_mask == 0:
            num_masked_nodes = [tot_masked]
        else:
            num_masked_nodes = [num_nodes_per_mask for _ in range(mask_times)]
            if sum(num_masked_nodes) < tot_masked:
                num_masked_nodes.append(tot_masked - sum(num_masked_nodes))
        masked_nodes_idx = torch.empty((0,), dtype=torch.long, device=device)
        init_masked_prob = torch.ones((num_nodes,), device=device)
        init_masked_prob = init_masked_prob / num_nodes

        num_has_masked_nodes = 0
        batch.x.requires_grad = False
        xan = torch.zeros_like(batch.x, requires_grad=False)
        xan[:, :] = batch.x[:, :]
        for i, num_per_mask in enumerate(num_masked_nodes):
            with torch.no_grad():
                if i == 0:
                    prev_node_rep = model_q(xan, batch.edge_index, batch.edge_attr)
                    prev_node_prob = torch.softmax(linear_pred_atoms(prev_node_rep).double(), dim=-1)
                    masked_nodes = torch.multinomial(init_masked_prob, num_per_mask, replacement=False)
                    xan[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    num_has_masked_nodes += num_per_mask
                else:
                    now_node_rep = model_q(xan, batch.edge_index, batch.edge_attr)
                    now_node_prob = torch.softmax(linear_pred_atoms(now_node_rep).double(), dim=-1)
                    entropy = -torch.sum(prev_node_prob * torch.log(torch.clamp(
                        now_node_prob, min=1e-12
                    )), dim=-1)

                    # not reserved
                    entropy = torch.max(entropy) + torch.min(entropy) - entropy

                    entropy = torch.softmax(entropy / T, dim=0)
                    entropy[masked_nodes_idx] = 0.0

                    masked_prob = entropy / torch.sum(entropy, dim=0) # normalize the masking probability distribution
                    assert abs(torch.sum(masked_prob, dim=0).item() - 1.0) < 1e-9, \
                        "expect sample prob summation to be 1, got {:.4f}" .format(torch.sum(masked_prob, dim=0).item())

                    masked_nodes = torch.multinomial(masked_prob, num_per_mask, replacement=False)
                    xan[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    prev_node_prob = now_node_prob.clone()
                    num_has_masked_nodes += num_per_mask


        node_rep = model_q(xan, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)
        loss += F.cross_entropy(pred_node.double(), node_labels)

        optimizer_linear_pred_atoms.zero_grad()
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
        optimizer_linear_pred_atoms.step()
        # moment_update(model_q, model_k, args.alpha)
        torch.cuda.synchronize()

        print(loss.detach().cpu().item(), batch.tot_in.cpu())

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
    parser.add_argument('--mask_times', type=int, default=3, help='number of workers for dataset loading')
    # parser.add_argument('--mask_rate', type=float, default=0.15,
    #                     help='dropout ratio (default: 0.15)')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    args.gpu = args.device

    args.output_model_file = "temp/contrast_sim_based_multi_pos_v3_sample_stra_{}_{:d}_num_neg_samples_{:d}_T_{:d}_sel_other_stra_{}_hop_{:d}_rstprob_{}_p_{:.1f}_q_{:.1f}_rw_hops_{:d}_num_path_{:d}_dl_other_p_w2v_neg_p_mask_times_{:d}_mask_rate_{:.4f}_gnn_{}".format(
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
        args.num_path,
        args.mask_times,
        args.mask_rate,
        args.gnn_type)


    if args.env == "jizhi":
        args.output_model_file = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/temp_model/" + args.output_model_file.split("/", 1)[1]

    print("num_com_negs", args.num_com_negs)

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

    # set up models, one for pre-training and one for context embeddings
    # model_k = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
    #               gnn_type=args.gnn_type).to(device)
    model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                  gnn_type=args.gnn_type).to(device)
    # model_dis = Discriminator(args.emb_dim if args.JK != "concat" else args.emb_dim * (1 + 1)).to(device)
    # model_attn = atten_module(input_dim=args.emb_dim).to(device)
    # pred_fiel = "temp/contrast_sim_based_neg_rwr_neg_1_num_neg_samples_64_2.pth"
    # model_q.load_state_dict(torch.load(pred_fiel))
    # print("loading model succ!")
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)

    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_q, linear_pred_atoms, loader, optimizer_q,
                                      optimizer_linear_pred_atoms, device, epoch)
        print(train_loss, train_acc)
        if epoch % 1 == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    # if not args.output_model_file == "":
    #     torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
