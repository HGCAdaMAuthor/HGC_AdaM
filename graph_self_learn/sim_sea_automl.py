import argparse
# import copy
# import random
# import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn.functional as F
# from scipy import sparse as sp
# from tqdm import tqdm

from data_util_2 import SSDataset
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
from models.graph_encoder import GraphEncoder
from graph_dataset import GRAPH_CLASSIFICATION_DSETS
import datetime
import json


class FromNumpyAlign(object):
    def __init__(self, hidden_size, emb_path_1, emb_path_2, **kwargs):
        self.hidden_size = hidden_size
        self.emb_1 = np.load(emb_path_1)
        self.emb_2 = np.load(emb_path_2)
        self.t1, self.t2 = False, False

    def train(self, G):
        if G.number_of_nodes() == self.emb_1.shape[0] and not self.t1:
            emb = self.emb_1
            self.t1 = True
        elif G.number_of_nodes() == self.emb_2.shape[0] and not self.t2:
            emb = self.emb_2
            self.t2 = True
        else:
            raise NotImplementedError

        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([emb[id2node[i]] for i in range(len(id2node))])

        return embeddings

def build_model(name, hidden_size, **model_args):
    return {
        "from_numpy_align": FromNumpyAlign,
    }[name](hidden_size, **model_args)

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

class get_emb_numpy(object):
    def __init__(self, dsn, load_path, gpu, rw_hops, num_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path, map_location="cpu")
            print(
                "=> loaded successfully '{}' (epoch {})".format(
                    load_path, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
        args = checkpoint["opt"]

        assert gpu is not None and torch.cuda.is_available()
        print("Use GPU: {} for training".format(gpu))

        if dsn == "dgl":
            train_dataset = GraphDataset(
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                hidden_size=args.hidden_size,
            )
        else:
            if dsn in GRAPH_CLASSIFICATION_DSETS:
                train_dataset = CogDLGraphClassificationDataset(
                    dataset=dsn,
                    rw_hops=args.rw_hops,
                    subgraph_size=args.subgraph_size,
                    restart_prob=args.restart_prob,
                    positional_embedding_size=args.positional_embedding_size,
                )
            else:
                train_dataset = CogDLGraphDataset(
                    dataset=dsn,
                    rw_hops=rw_hops,
                    num_path=num_path,
                    subgraph_size=args.subgraph_size,
                    restart_prob=args.restart_prob,
                    positional_embedding_size=args.positional_embedding_size,
                )
        args.batch_size = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=batcher(),
            shuffle=False,
            num_workers=args.num_workers,
        )

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

        model = model.cuda(gpu)

        model.load_state_dict(checkpoint["model"])

        del checkpoint
        torch.cuda.empty_cache()

        args.gpu = gpu
        args.return_all_outputs = False
        self.emb = test_moco(train_loader, model, args)
        print("get emb for %s" % dsn)
    def get_emb(self):
        return self.emb



class SimilaritySearch(object):
    def __init__(self, dataset_1, dataset_2, model, hidden_size, emb_path_1, emb_path_2):
        self.data = SSDataset("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/panther",
                              dataset_1, dataset_2).data
        # self.model = build_model(model, hidden_size, **model_args)
        self.mn = model
        self.hz = hidden_size
        self.empa = emb_path_1
        self.empb = emb_path_2
        self.hidden_size = hidden_size

    def _train_wrap(self, data):
        G = nx.MultiGraph()
        G.add_edges_from(data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((G.number_of_nodes(), self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
        return features_matrix

    def train(self):
        self.model = build_model(self.mn, self.hz, emb_path_1=self.empa, emb_path_2=self.empb)
        emb_1 = self._train_wrap(self.data[0])
        emb_2 = self._train_wrap(self.data[1])
        return self._evaluate(emb_1, emb_2, self.data[0].y, self.data[1].y)

    def train_with_emb(self, emba, embb):
        return self._evaluate(emba, embb, self.data[0].y, self.data[1].y)

    def _evaluate(self, emb_1, emb_2, dict_1, dict_2):
        shared_keys = set(dict_1.keys()) & set(dict_2.keys())
        shared_keys = list(
            filter(
                lambda x: dict_1[x] < emb_1.shape[0] and dict_2[x] < emb_2.shape[0],
                shared_keys,
            )
        )
        emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
        emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)
        reindex = [dict_2[key] for key in shared_keys]
        reindex_dict = dict([(x, i) for i, x in enumerate(reindex)])
        emb_2 = emb_2[reindex]
        k_list = [20, 40]
        id2name = dict([(dict_2[k], k) for k in dict_2])

        all_results = defaultdict(list)
        for key in shared_keys:
            v = emb_1[dict_1[key]]
            scores = emb_2.dot(v)

            idxs = scores.argsort()[::-1]
            for k in k_list:
                all_results[k].append(int(reindex_dict[dict_2[key]] in idxs[:k]))
        res = dict(
            (f"Recall @ {k}", sum(all_results[k]) / len(all_results[k])) for k in k_list
        )
        return sum(all_results[20]) / len(all_results[20])
        # return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rw_hops", type=int, default=14)
    parser.add_argument("--num_path", type=int, default=12)
    parser.add_argument("--emb-path-1", type=str, default="")
    parser.add_argument("--emb-path-2", type=str, default="")
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    args = parser.parse_args()

    worker_path = os.getenv("JIZHI_WORKSPACE_PATH")
    # job_param_path = './job_param.json'
    job_param_path = os.path.join(worker_path, "job_param.json")
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        rw_hops = hyper_params["rw_hops"]
        num_path = hyper_params["num_path"]

    args.rw_hops = rw_hops
    args.num_path = num_path

    dsa = args.dataset.split("_")[0]
    dsb = args.dataset.split("_")[1]

    dsaa = get_emb_numpy(dsa, args.load_path, args.gpu, args.rw_hops, args.num_path)
    dsa_emb = dsaa.get_emb().detach().cpu().numpy()

    dsbb = get_emb_numpy(dsb, args.load_path, args.gpu, args.rw_hops, args.num_path)
    dsb_emb = dsbb.get_emb().detach().cpu().numpy()

    print(dsa_emb.shape, dsb_emb.shape)

    task = SimilaritySearch(
        args.dataset.split("_")[0],
        args.dataset.split("_")[1],
        args.model,
        args.hidden_size,
        emb_path_1=args.emb_path_1,
        emb_path_2=args.emb_path_2,
    )
    # ret = task.train()
    ret = task.train_with_emb(dsa_emb, dsb_emb)
    # print(ret)

    print('time: %s, auc = %.4f, dataset: %s' % (
        datetime.datetime.now(), ret, args.dataset))
