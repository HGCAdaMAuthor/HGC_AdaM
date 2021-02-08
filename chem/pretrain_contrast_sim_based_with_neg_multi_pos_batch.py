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
def train(args, model_q, loader, optimizer_q, device, epoch):
    # model_k.eval()
    model_q.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    stt_time = time.time()

    for step, batch in enumerate(loader):
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

        # for i, data in enumerate(batch.data_list):
        #     print(i)
        #     all_rep = model_q(data.all_x.to(device), data.all_edge_index.to(device), data.all_edge_attr.to(device))
        #     all_glb_rep = pool_func(all_rep, data.all_batch.to(device), args.pooling)
        #     all_glb_rep = all_glb_rep / (torch.norm(all_glb_rep, p=2, dim=1, keepdim=True) + 1e-9)
        #     # assert all_glb_rep.size(0) == 1 + num_samples + num_neg_samples
        #     target_rep = all_glb_rep[0, :]
        #     pos_rep = all_glb_rep[1: num_samples + 1, :].view(num_samples, -1)
        #     # if len(pos_rep.size()) == 1:
        #     #     pos_rep = pos_rep.view(1, -1)
        #     # pos_rep = pos_rep
        #     neg_rep = all_glb_rep[num_samples + 1:, :].view(num_neg_samples, -1)
        #     # if len(neg_rep.size()) == 1:
        #     #     neg_rep = neg_rep.view(1, -1)
        #     # num_pos_samples x 1
        #     pos_score = torch.sum(torch.mul(target_rep.unsqueeze(0), pos_rep), dim=-1, keepdim=True) / T
        #     # num_neg_samples
        #     neg_score = torch.sum(torch.mul(target_rep.unsqueeze(0), neg_rep), dim=-1, keepdim=False) / T
        #     # num_pos_samples x num_neg_samples
        #     neg_score = neg_score.unsqueeze(0).repeat(num_samples, 1)
        #     scores_cat = torch.cat([pos_score, neg_score], dim=-1)
        #     labels = torch.zeros((num_samples, ), dtype=torch.long)
        #     loss_tmp = F.nll_loss(F.log_softmax(scores_cat, dim=-1), labels)
        #     if i == 0:
        #         loss = loss_tmp
        #     else:
        #         loss += loss_tmp

        # rep_q = model_q(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
        # rep_q_global = pool_func(rep_q, batch.batch.to(device), args.pooling)
        # rep_q_global = rep_q_global / (torch.norm(rep_q_global, p=2, dim=1, keepdim=True) + 1e-9)
        # # loss = 0.0
        # assert num_samples >= 1
        # # print(num_samples)
        # # st_time = time.time()
        #
        # bs = rep_q_global.size(0)
        # # for j in range(num_neg_samples):
        # sim_x, sim_edge_index, sim_edge_attr = batch.sim_x, batch.sim_edge_index, batch.sim_edge_attr
        # rep_sim = model_q(sim_x.to(device), sim_edge_index.to(device), sim_edge_attr.to(device))
        # rep_sim_global = pool_func(rep_sim, batch.batch_sim.to(device), args.pooling)
        #
        # rep_sim_global = rep_sim_global / (torch.norm(rep_sim_global, p=2, dim=1, keepdim=True) + 1e-9)
        # ### bs x rep_size
        # ### sim_rep_global_all.size() = bs x num_samples x rep_size
        # sim_rep_global_all = rep_sim_global.view(bs, num_samples, -1)
        # # print("rep_sim_global.size() = ", rep_sim_global.size())
        #
        # # pos_global_reps = list()
        # #### todo: from sim_xs to a big batch and view them to the expected dim just as neg_xs do...
        # # for j in range(num_samples):
        # #     sim_x, sim_edge_index, sim_edge_attr = batch.sim_x[j], batch.sim_edge_index[j], batch.sim_edge_attr[j]
        # #     rep_k = model_q(sim_x.to(device), sim_edge_index.to(device), sim_edge_attr.to(device))
        # #     rep_k_global = pool_func(rep_k, batch.batch_sim[j].to(device), args.pooling)
        # #     rep_sim_global = rep_k_global / (torch.norm(rep_k_global, p=2, dim=1, keepdim=True) + 1e-9)
        # #     #### rep_sim_global.size() = bs x rep_size
        # #     pos_global_reps.append(rep_sim_global.unsqueeze(1).clone())
        # #### todo: sample positive samples and negative ones at first and use them during the training...
        # # pos_global_reps = torch.cat(pos_global_reps, dim=1)
        # #### pos_global_reps.size() = bs x n_sim x rep_size
        # # sim_rep_global_all = sim_rep_global_all / (torch.norm(sim_rep_global_all, p=2, dim=-1, keepdim=True) + 1e-9)
        # mult_self_pos_score = torch.sum(torch.mul(rep_q_global.unsqueeze(1), sim_rep_global_all), dim=-1, keepdim=True) / T
        # #### bs x n_sim x 1
        #
        # # bs = rep_q_global.size(0)
        # # for j in range(num_neg_samples):
        # #
        # neg_x, neg_edge_index, neg_edge_attr = batch.neg_x, batch.neg_edge_index, batch.neg_edge_attr
        # rep_neg = model_q(neg_x.to(device), neg_edge_index.to(device), neg_edge_attr.to(device))
        # rep_neg_global = pool_func(rep_neg, batch.batch_neg.to(device), args.pooling)
        #
        # rep_neg_global = rep_neg_global / (torch.norm(rep_neg_global, p=2, dim=1, keepdim=True) + 1e-9)
        # ### bs x rep_size
        # # print(rep_neg_global.size())
        # neg_rep_global_all = rep_neg_global.view(bs, num_neg_samples, -1)
        # #### neg_rep_global.size() = bs x n_neg_samples x rep_size
        # mult_self_neg_score = torch.sum(torch.mul(rep_q_global.unsqueeze(1), neg_rep_global_all), dim=-1, keepdim=False) / T
        # # bs x n_neg
        # mult_self_neg_score = mult_self_neg_score.unsqueeze(1).repeat(1, num_samples, 1)
        # # print("mult_self_neg_score after repeated", mult_self_neg_score.size())
        # mult_self_pos_neg_score_cat = torch.cat([mult_self_pos_score, mult_self_neg_score], dim=-1)
        # #### bs x n_sim x (n_neg + 1) and label is 0
        # mult_self_pos_neg_score_cat_expanded = mult_self_pos_neg_score_cat.view(-1, (num_neg_samples + 1))
        # labels = torch.zeros((mult_self_pos_neg_score_cat_expanded.size(0), ), dtype=torch.long, device=device)
        # loss = F.nll_loss(F.log_softmax(mult_self_pos_neg_score_cat_expanded, dim=-1), labels)
        #
        # # print(mult_self_pos_neg_score_cat_expanded)
        # # loss (another direction)
        # # bs x n_pos x 1 x rep_size @ bs x 1 x n_neg x rep_size = bs x n_pos x n_neg x rep_size
        # mult_sim_neg_score = torch.sum(torch.mul(sim_rep_global_all.unsqueeze(2), neg_rep_global_all.unsqueeze(1)),
        #                                dim=-1, keepdim=False) / T
        # # mult_sim_neg_score.size() = bs x n_pos x n_neg
        # # mult_self_pos_score.size() = bs x n_pos x 1
        # mult_sim_self_neg_score_cat = torch.cat([mult_self_pos_score, mult_sim_neg_score], dim=-1)
        # mult_sim_self_neg_score_expand = mult_sim_self_neg_score_cat.view(-1, (num_neg_samples + 1))
        #
        # if "div_importance" in batch:
        #     loss_tmp = F.nll_loss(F.log_softmax(mult_sim_self_neg_score_expand, dim=-1), labels, reduce=False)
        #     loss_tmp_unexpand = loss_tmp.view(bs, -1)
        #     loss_tmp_unexpand /= batch.div_importance.to(device).view(bs, -1)
        #     loss += torch.mean(loss_tmp_unexpand)
        # else:
        #     loss += F.nll_loss(F.log_softmax(mult_sim_self_neg_score_expand, dim=-1), labels)

        # bs = rep_neg_global.size(0)
        # neg_rep_global_all_list.append(rep_neg_global.view(bs, 1, -1))
        # # pos_rep_global_all_list.append(rep_sim_global.view(bs, 1, -1))
        # q_rep_global_all_list.append(rep_q_global.view(bs, 1, -1))
        # neg_rep_global_all = torch.cat(neg_rep_global_all_list, dim=1)
        # pos_rep_global_all = torch.cat(pos_rep_global_all_list, dim=1)
        # q_rep_global_all_list = [rep_q_global.view(bs, 1, -1) for _ in range(num_neg_samples)]
        # q_rep_global_all = torch.cat(q_rep_global_all_list, dim=1)
        # sim_scores = torch.sum(rep_q_global * rep_sim_global, dim=-1, keepdim=True) / T ### bs x 1
        # neg_scores = torch.sum(q_rep_global_all * neg_rep_global_all, dim=-1, keepdim=False) / T ### bs x k
        # cat_scores = torch.cat([sim_scores, neg_scores], dim=-1)
        # labels = torch.zeros((rep_q_global.size(0), ), dtype=torch.long, device=device)
        # loss = F.nll_loss(F.log_softmax(cat_scores, dim=-1), labels)

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
        # print(time.time() - st_time)

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

        print(loss.detach().cpu().item(), batch.tot_in.cpu())
        # with open(args.log_loss_file, "a") as wf:
        #     wf.write("{}\n".format(str(loss.detach().cpu().item())))

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

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # l1 = args.num_layer - 1
    # l2 = l1 + args.csize
    args.gpu = args.device
    # print(args.mode)
    # print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))
    # args.num_samples = 1

    args.output_model_file = "temp/contrast_sim_based_multi_pos_v3_sample_stra_{}_{:d}_num_neg_samples_{:d}_T_{:d}_sel_other_stra_{}_hop_{:d}_rstprob_{}_p_{:.1f}_q_{:.1f}_rw_hops_{:d}_num_path_{:d}_dl_other_p_w2v_neg_p_no_automl_gnn_{}".format(
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
        args.gnn_type)

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

    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_q, loader, optimizer_q, device, epoch)
        print(train_loss, train_acc)
        if epoch % 1 == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    # if not args.output_model_file == "":
    #     torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
