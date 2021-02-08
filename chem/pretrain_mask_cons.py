import argparse

# TODO implement this, the reconstruct error and the adversarial loss refer to L7 P30
from loader import MoleculeDataset
from dataloader import DataLoaderMasking  # , DataListLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskAtom2

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit
from torch_geometric.nn.inits import uniform


def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)

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
        assert aa != None and aa.size(0) == x.size(0)
        h = torch.matmul(summary, self.weight)
        logists = torch.matmul(x, h.t())
        # x.size(0) x summary.size(0)
        target = torch.zeros(logists.size(), device=x.device)
        target[torch.arange(0, target.size(0), device=x.device), aa] = 1.
        return logists.view(-1), target.view(-1)



class Ad_dis(nn.Module):
    def __init__(self, in_dim):
        super(Ad_dis, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
    def init_weight(self):
        for i in self.net:
            if isinstance(i, nn.Linear):
                torch.nn.init.xavier_uniform_(i.weight.data)
    def forward(self, x):
        logists = self.net(x)
        return logists

def train(args, model_list, loader, optimizer_list, device):

    model, model_dis, model_dis_infomax = model_list
    optim_model, optim_dis, optim_infomax = optimizer_list

    model.train()
    model_dis.train()
    if model_dis_infomax != None:
        model_dis_infomax.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)[batch.masked_atom_indices]
        node_rep_cor = model(batch.x, batch.edge_index, batch.edge_attr, mask_indices=batch.masked_atom_indices)[batch.masked_atom_indices]

        loss = F.mse_loss(node_rep, node_rep_cor)

        logists_real = model_dis(node_rep)
        logists_fake = model_dis(node_rep_cor)
        # loss = loss + 0.5 * (F.binary_cross_entropy_with_logits(logists_real, torch.ones_like(logists_real)) +
        #        F.binary_cross_entropy_with_logits(logists_fake, torch.zeros_like(logists_fake)))

        if args.infomax == True:
            global_rep = pool_func(node_rep, batch.batch, args.pooling)
            logists, target = model_dis_infomax(node_rep, global_rep, batch.batch)
            loss += F.binary_cross_entropy_with_logits(logists, target)


        optim_model.zero_grad()
        optim_dis.zero_grad()
        optim_infomax.zero_grad()

        loss.backward()

        optim_model.step()
        optim_dis.step()
        optim_infomax.step()
        print(loss.item())
        loss_accum += float(loss.cpu().item())

    return loss_accum / step, acc_node_accum / step, acc_edge_accum / step

# auto regressive representations from the nearby nodes can be used to predict the representation of the dropped node
# naive predictive method ? ----- how can we predict ? ---- mostly just still by the discrimination between those
# two kind of representations
# really dropped the nodes rather than just set its feature to zero ---- and use the context ndoes to predict ist atom
# type and other things
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--infomax', default=False, action="store_true", help='number of workers for dataset loading')
    parser.add_argument('--pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max)')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    args.output_model_file = "masking_adversarial_infomas_{}".format(str(args.infomax))
    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    # set up dataset and transform function.
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              transform=MaskAtom2(num_atom_type=119, num_edge_type=5, mask_rate=args.mask_rate,
                                                 mask_edge=args.mask_edge))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
