import argparse

from loader import MoleculeDataset
from dataloader import DataLoaderMasking  # , DataListLoader

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

from tqdm import tqdm
import numpy as np

from model import GNN
from MolVocab import MolVocab
#
from util import mask_strategies, MaskAtomGetFea

criterion = nn.CrossEntropyLoss()


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(MLPLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )
        self.init_weight()

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.net(x).view(-1)


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


# import time
def train(args, model_list, loader, optimizer_list, device):
    model, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list

    # LARGE_NUM = 1e12
    T = 0.07
    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        num_nodes = batch.x.size(0)
        device = batch.x.device

        # set node_labels to labels extracted from mol features
        node_labels = batch.atom_fea_idx

        tot_masked = int(batch.x.size(0) * args.mask_rate + 1)
        num_nodes_per_mask = int(tot_masked // args.mask_times)
        mask_times = args.mask_times
        if num_nodes_per_mask == 0:
            num_masked_nodes = [tot_masked]
        else:
            num_masked_nodes = [num_nodes_per_mask for _ in range(mask_times)]
            if sum(num_masked_nodes) < tot_masked:
                num_masked_nodes.append(tot_masked - sum(num_masked_nodes))
        masked_nodes_idx = torch.empty((0, ), dtype=torch.long, device=batch.x.device)
        init_masked_prob = torch.ones((num_nodes, ), device=device)
        init_masked_prob = init_masked_prob / num_nodes
        # len_fea = batch.fea_x.size(1)

        for i, num_per_mask in enumerate(num_masked_nodes):
            with torch.no_grad():
                if i == 0:
                    # prev_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    prev_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    prev_node_prob = torch.softmax(linear_pred_atoms(prev_node_rep).double(), dim=-1)
                    masked_nodes = torch.multinomial(init_masked_prob, num_per_mask, replacement=False)
                    # print(masked_nodes.size())
                    batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    # batch.fea_x[masked_nodes, :] = torch.zeros((len_fea, ), dtype=torch.float32, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                else:
                    now_node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
                    now_node_prob = torch.softmax(linear_pred_atoms(now_node_rep).double(), dim=-1)
                    entropy = -torch.sum(prev_node_prob * torch.log(torch.clamp(
                        now_node_prob, min=1e-12
                    )), dim=-1)
                    # print(entropy)
                    if args.reverse:
                        # print("in reversing")
                        entropy = entropy - torch.min(entropy)
                    else:
                        entropy = torch.max(entropy) + torch.min(entropy) - entropy
                    # entropy[masked_nodes_idx] = -LARGE_NUM
                    # print(entropy)  #
                    entropy = torch.softmax(entropy / T, dim=0)
                    entropy[masked_nodes_idx] = 0.0
                    masked_prob = entropy / torch.sum(entropy, dim=0)
                    assert abs(torch.sum(masked_prob, dim=0).item() - 1.0) < 1e-9, "expect sample prob summation to be 1, got {:.4f}" .format(torch.sum(masked_prob, dim=0).item())

                    masked_nodes = torch.multinomial(masked_prob, num_per_mask, replacement=False)
                    batch.x[masked_nodes, :] = torch.tensor([119, 0], dtype=torch.long, device=device)
                    # batch.fea_x[masked_nodes, :] = torch.zeros((len_fea,), dtype=torch.float32, device=device)
                    masked_nodes_idx = torch.cat([masked_nodes_idx, masked_nodes], dim=0)
                    prev_node_prob = now_node_prob.clone()

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = linear_pred_atoms(node_rep)
        # print(torch.max(node_labels))
        loss = criterion(pred_node.double(), node_labels)

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        print(loss.item())

    return loss_accum / step, acc_node_accum / step, acc_edge_accum / step


def main():
    # Training settings
    # params --- batch_size & mask_times
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
    parser.add_argument('--use_infomax', action="store_true", default=False, help='number of workers for dataset loading')
    parser.add_argument('--minus', type=int, default=2, help='number of workers for dataset loading')
    parser.add_argument('--moleculor_mask', type=bool, default=False, help='number of workers for dataset loading')
    parser.add_argument('--mask_times', type=int, default=3, help='number of workers for dataset loading')
    parser.add_argument('--several-times-mask', type=bool, default=False, help='number of workers for dataset loading')
    parser.add_argument('--reverse', default=False, action='store_true', help='number of workers for dataset loading')
    parser.add_argument('--DEBUG', default=False, action='store_true', help='number of workers for dataset loading')
    parser.add_argument('--vocab_path', type=str, help='number of workers for dataset loading')
    parser.add_argument('--max_size', type=int, default=966, help='number of workers for dataset loading')

    # --vocab_path ./dataset/zinc_standard_agent/processed/atom_vocab.pkl

    args = parser.parse_args()

    # print(args.reverse)

    args.output_model_file = "temp/masker_entropy_based_gnn_{}_mask_times_{:d}_batch_size_{:d}_reverse_{}_other_fea_in_also_max_size_{:d}".format(
        args.gnn_type,
        args.mask_times,
        args.batch_size,
        str(args.reverse),
        args.max_size)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    mol_vocab = MolVocab.load_vocab(args.vocab_path)
    print("vocal.size() = ", len(mol_vocab.stoi))

    # args.emb_dim = sum([119 + 1, 6 + 1, 6, 5, 6, 6, 2, 18])
    print("emb_dim = ", args.emb_dim)

    model = GNN(args.num_layer,
                args.emb_dim,
                JK=args.JK,
                drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, args.max_size).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset,
                              after_transform=MaskAtomGetFea(mol_vocab=mol_vocab, DEBUG=args.DEBUG))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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
        if epoch % 1 == 0:
            if not args.output_model_file == "":
                torch.save(model.state_dict(), args.output_model_file + "_{}.pth".format(str(epoch)))

    if not args.output_model_file == "":
        torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
