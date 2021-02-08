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

from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask

from dataloader import DataLoaderSubstructContext, DataLoaderMasking

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
        return torch.sum(x*h, dim = 1)


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

class two_attn_distri(nn.Module):
    def __init__(self, input_dim_A, input_dim_B, output_dim_A=None, output_dim_B=None, key_dim=None):
        super(two_attn_distri, self).__init__()
        if output_dim_A == None:
            output_dim_A = input_dim_A
        if output_dim_B == None:
            output_dim_B = input_dim_B
        # if key_dim == None:
        #     key_dim = value_dim
        self.transfer_A_weight = nn.Parameter(torch.Tensor(input_dim_A, output_dim_A))
        self.transfer_B_weight = nn.Parameter(torch.Tensor(input_dim_B, output_dim_B))
        self.temperature = 0.07

        # self.value_weight = nn.Parameter(torch.Tensor(input_dim_A, value_dim))
        # self.query_weight = nn.Parameter(torch.Tensor(input_dim_B, query_dim))
        # # self.key_weight = nn.Parameter(torch.Tensor(input_dim, key_dim))
        # self.temperature = 8
        # self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.transfer_A_weight.size(0)
        uniform(size, self.transfer_A_weight)
        size = self.transfer_B_weight.size(0)
        uniform(size, self.transfer_B_weight)
        # size = self.key_weight.size(0)
        # uniform(size, self.key_weight)

    def forward(self, A, B, mask=None):
        A = torch.matmul(A, self.transfer_A_weight)
        B = torch.matmul(B, self.transfer_B_weight)
        A_attn_A = F.softmax(torch.matmul(A / self.temperature, A.transpose(0, 1)), dim=1)
        B_attn_B = F.softmax(torch.matmul(B / self.temperature, B.transpose(0, 1)), dim=1)
        loss = F.mse_loss(A_attn_A, B_attn_B, reduction="mean")

        return loss

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


def train(args, model_k, model_q, model_dis, model_attn, loader, optimizer_q, optimizer_dis, optimizer_attn, device,
          contrast, criterion, epoch):
    model_k.eval()
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
        rep_q = model_q(batch.xb, batch.edge_index, batch.edgeb_attr)
        # rep_q = model_attn(rep_q, batch.mask)
        # num_of_nodes = rep_q.size(0)
        loss_q_attn = model_attn(batch.wave_emb, rep_q)
        rep_q_global = pool_func(rep_q, batch.batch, args.pooling)
        rep_k = model_q(batch.xa, batch.edge_index, batch.edgea_attr)
        loss_k_attn = model_attn(batch.wave_emb, rep_k)
        # rep_k = model_attn(rep_k, batch.mask)
        rep_k_global = pool_func(rep_k, batch.batch, args.pooling)

        summary_emb_q = F.sigmoid(rep_q_global)
        summary_emb_k = F.sigmoid(rep_k_global)

        rep_k_global = rep_k_global / (torch.norm(rep_k_global, p=2, dim=1, keepdim=True) + 1e-9)
        rep_q_global = rep_q_global / (torch.norm(rep_q_global, p=2, dim=1, keepdim=True) + 1e-9)
        bs = rep_k_global.shape[0]

        positive_expanded_summary_emb_q = summary_emb_q[batch.batch]
        positive_expanded_summary_emb_k = summary_emb_k[batch.batch]

        shifted_summary_emb_k = summary_emb_k[cycle_index(len(summary_emb_k), 1)]
        shifted_summary_emb_q = summary_emb_q[cycle_index(len(summary_emb_q), 1)]
        negative_expanded_summary_emb_k = shifted_summary_emb_k[batch.batch]
        negative_expanded_summary_emb_q = shifted_summary_emb_q[batch.batch]

        positive_score_k = model_dis(rep_k, positive_expanded_summary_emb_k)
        positive_score_q = model_dis(rep_q, positive_expanded_summary_emb_q)
        positive_score_kq = model_dis(rep_k, positive_expanded_summary_emb_q)
        positive_score_qk = model_dis(rep_q, positive_expanded_summary_emb_k)
        negative_score_k = model_dis(rep_k, negative_expanded_summary_emb_k)
        negative_score_q = model_dis(rep_q, negative_expanded_summary_emb_q)
        negative_score_kq = model_dis(rep_k, negative_expanded_summary_emb_q)
        negative_score_qk = model_dis(rep_q, negative_expanded_summary_emb_k)

        loss = criterion_dis(positive_score_k, torch.ones_like(positive_score_k)) + \
               criterion_dis(negative_score_k, torch.zeros_like(negative_score_k)) + \
               criterion_dis(positive_score_q, torch.ones_like(positive_score_q)) + \
               criterion_dis(negative_score_q, torch.zeros_like(negative_score_q)) + \
               criterion_dis(positive_score_kq, torch.ones_like(positive_score_kq)) + \
               criterion_dis(positive_score_qk, torch.ones_like(positive_score_qk)) + \
               criterion_dis(negative_score_kq, torch.zeros_like(negative_score_kq)) + \
               criterion_dis(negative_score_qk, torch.zeros_like(negative_score_qk))
        # loss.backward()

        # print(batch.node_idx_tensor.shape)

        # print(batch.num_moleculor)
        # print(rep_q.shape)
        # print(rep_k_global.shape)
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
        loss += (loss_k + loss_q) + 0.1 * (loss_k_attn + loss_q_attn) # / num_of_nodes

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
        optimizer_dis.zero_grad()
        optimizer_attn.zero_grad()
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
        optimizer_dis.step()
        optimizer_attn.step()
        # moment_update(model_q, model_k, args.alpha)
        torch.cuda.synchronize()
        print(loss.detach().cpu().item())

    return balanced_loss_accum / step, acc_accum / (step * 2)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
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
    parser.add_argument('--output_model_file', type=str, default='temp/attn', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

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

    # set up dataset and transform function.
    dataset = ContrastDataset("dataset/" + args.dataset, args=args, dataset=args.dataset,
                              transform=PairwiseNodeEdgeMask(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge, add_wave=True))
    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model_k = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                          gnn_type=args.gnn_type).to(device)
    model_q = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                        gnn_type=args.gnn_type).to(device)
    model_dis = Discriminator(args.emb_dim if args.JK != "concat" else args.emb_dim * (1 + 1)).to(device)
    model_attn = two_attn_distri(input_dim_A=30, input_dim_B=args.emb_dim, output_dim_A=args.emb_dim, output_dim_B=args.emb_dim).to(device)

    n_data = dataset.total
    contrast = MemoryMoCo(inputSize=args.emb_dim, outputSize=n_data, K=args.nce_k, T=args.nce_t, use_softmax=True, device=args.device)
    contrast = contrast.cuda(args.device)
    criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    criterion = criterion.cuda(args.device)
    criterion = nn.NLLLoss()
    criterion = criterion.cuda(args.device)
    # set up optimizer for the two GNNs
    # optimizer_k = optim.Adam(model_k.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_q = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=args.lr, weight_decay=args.decay)


    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_k, model_q, model_dis, model_attn, loader, optimizer_q, optimizer_dis,
                                      optimizer_attn, device, contrast, criterion, epoch)
        print(train_loss, train_acc)
        if epoch % 5 == 0:
            if not args.output_model_file == "":
                torch.save(model_q.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model_q.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
