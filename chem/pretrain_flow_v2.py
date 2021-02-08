import argparse

from loader import MoleculeDataset, ContrastDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from time import time

# from model import GNN
from model import GNN, Mole_Flow, Mole_Flow_Cond
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import ExtractSubstructureContextPair, PairwiseNodeEdgeMask, ExtractMaskedData, ExtractMaskedDataVTwo

from dataloader import DataLoaderSubstructContext, DataLoaderMasking, DataLoaderMaskedFlow, DataLoaderContrastFlowVTwo

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
from contrastive_model import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, NCESoftmaxLossNS
from torch_geometric.nn.inits import uniform

LARGE_NUM = 1e9
criterion_dis = nn.BCEWithLogitsLoss()

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

def adjust_learning_rate(optimizer, cur_iter, init_lr, warm_up_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if warm up step is 0, no warm up actually.
    if cur_iter < warm_up_step:
        lr = init_lr * (1. / warm_up_step + 1. / warm_up_step * cur_iter)  # [0.1lr, 0.2lr, 0.3lr, ..... 1lr]
    else:
        lr = init_lr
        return lr
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


import os
import json
def save_model(model, optimizer, args, var_list, epoch=None):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )

##### 太假。。。。 figure out how to calculate the edge embeddings --- what embeddings do we need to calculate the edge
##### embeddings
##### and how to use rl to sample nodes???

def train(args, model, loader, optimizer, device, epoch):

    model.train()

    balanced_loss_accum = 0
    acc_accum = 0
    T = 0.07

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        loss = model(batch)
        # loss = model(batch.x, batch.split_x, batch.split_edge_index, batch.split_edge_attr, batch.split_node_map,
        #              batch.to_pred_edge_st_ed_idx, batch.to_pred_edge_subgraph_idx, batch.to_pred_edge_attr,
        #              batch.batch_node, batch.batch_edge)

        optimizer.zero_grad()
        # loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
        global_step = (epoch - 1) * args.batch_size + step
        lr_this_step = args.lr * warmup_linear(
            global_step / (args.epochs * args.batch_size), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step

        balanced_loss_accum += loss.detach().cpu().item()

        optimizer.step()
        # moment_update(model_q, model_k, args.alpha)
        torch.cuda.synchronize()
        print(loss.detach().cpu().item())

    return balanced_loss_accum / step, acc_accum / (step * 2)


def reinforce(args, model, optimizer, scheduler, device, epoch):
    best_reward = -100.0
    start_iter = 0
    model.load_state_dict(torch.load(args.input_model_file))
    t_total = time()
    total_loss = []
    total_reward = []
    total_score = []
    # moving_baseline = np.zeros([self.args.max_size_rl])
    moving_baseline = None
    print('start finetuning model(reinforce)')
    for cur_iter in range(args.reinforce_iters):
        if cur_iter == 0:
            iter_loss, iter_reward, iter_score, moving_baseline = reinforce_one_iter(args, model, optimizer,
                                                                                     scheduler, cur_iter + start_iter, device,
                                                                                          in_baseline=None)
        else:
            iter_loss, iter_reward, iter_score, moving_baseline = reinforce_one_iter(args, model, optimizer,
                                                                                     scheduler, cur_iter + start_iter, device,
                                                                                          in_baseline=moving_baseline)

        total_loss.append(iter_loss)
        total_reward.append(iter_reward)
        total_score.append(iter_score)
        # save_one_reward(os.path.join(self.args.save_path, 'iter_rewards.txt'), iter_reward, iter_score, iter_loss,
        #                 cur_iter + start_iter)  # append the iter reward to file
        print(moving_baseline)

        # then save the model...
        if iter_reward > best_reward:
            best_reward = iter_reward
            if args.save:
                var_list = {'cur_iter': cur_iter + start_iter,
                            'best_reward': best_reward,
                            }
                save_model(model, optimizer, args, var_list, epoch=cur_iter + start_iter)

    print("Finetuning(Reinforce) Finished!")
    print("Total time elapsed: {:.4f}s".format(time() - t_total))


def reinforce_one_iter(args, model, optimizer, scheduler, iter_cnt, device, in_baseline=None):
    t_start = time()
    #self._model.train() we will manually set train/eval mode in self._model.reinforce_forward()
    #if iter_cnt % self.args.accumulate_iters == 0:
    optimizer.zero_grad()

    loss, per_mol_reward, per_mol_property_score, out_baseline = model.reinforce_forward(device, temperature=args.rl_sample_temperature,
                                            max_size_rl=args.max_size_rl, batch_size=args.batch_size, in_baseline=in_baseline, cur_iter=iter_cnt)

    num_mol = len(per_mol_reward)
    avg_reward = sum(per_mol_reward) / num_mol
    avg_score = sum(per_mol_property_score) / num_mol
    max_cur_reward = max(per_mol_reward)
    max_cur_score = max(per_mol_property_score)
    loss.backward()
    #if (iter_cnt + 1) % self.args.accumulate_iters == 0:
    #    print('update parameter at iter %d' % iter_cnt)
    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    cur_lr = adjust_learning_rate(optimizer, iter_cnt, args.lr, args.warm_up)
    optimizer.step()
    if args.lr_decay:
        scheduler.step(avg_reward)

    print('Iter: {: d}, num_mol {:d}, loss {:5.5f}, lr {:5.5f}, reward {:5.5f}, score {:5.5f}, max_reward {:5.5f}, max_score {:5.5f}, iter time {:.5f}'.format(iter_cnt,
                                num_mol, loss.item(), cur_lr,
                                avg_reward, avg_score, max_cur_reward, max_cur_score, time()-t_start))
    return loss.item(), avg_reward, avg_score, out_baseline


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
    parser.add_argument('--dataset', type=str, default='chembl_filtered',
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
    parser.add_argument('--reinforce', action='store_true', default=False, help='reinforce')
    parser.add_argument('--reinforce_iters', type=int, default=5000, help='number of iters for reinforce')
    parser.add_argument('--save', action='store_true', default=False, help='Save model.')
    parser.add_argument('--max_size_rl', type=int, default=48, help='maximal #atoms of generated molecule')
    parser.add_argument('--rl_sample_temperature', type=float, default=0.75,
                        help='maximal #atoms of generated molecule')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='reinforce')
    parser.add_argument('--warm_up', type=int, default=0, help='linearly learning rate warmup')
    parser.add_argument('--num_labels', type=int, default=1310, help='linearly learning rate warmup')
    parser.add_argument('--save-path', type=str, default="./tmp_flow_model_v2/")
#### to start reinforce finetune --lr_decay --save --reinforce --input_model_file ./temp/base_flow_model_15.pth

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

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

    args.output_model_file = "temp/base_flow_model_split_edge_selected_v2"

    # set up dataset and transform function.
    dataset = ContrastDataset("dataset/" + args.dataset, args=args, dataset=args.dataset,
                              transform=ExtractMaskedDataVTwo())
    loader = DataLoaderMaskedFlow(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings

    model = Mole_Flow_Cond(args.num_layer, args.emb_dim, args.num_labels, JK=args.JK, drop_ratio=args.dropout_ratio,
                          gnn_type=args.gnn_type).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, min_lr=1e-6)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        if not args.reinforce:
            train_loss, train_acc = train(args, model, loader, optimizer, device, epoch)
            print(train_loss, train_acc)
            if epoch % 1 == 0:
                if not args.output_model_file == "":
                    torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))
        else:

            reinforce(args, model, optimizer, scheduler, device, epoch)
            if epoch % 1 == 0:
                if not args.output_model_file == "":
                    torch.save(model.state_dict(), args.output_model_file + "_reinforce_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10, 2)
    main()
