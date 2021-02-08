#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import math
import numpy as np
import operator
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
from dgl.data import AmazonCoBuy, Coauthor
import dgl.data
# from dgl.nodeflow import NodeFlow

# from cogdl.datasets import build_dataset
import data_util
#import horovod.torch as hvd

GRAPH_CLASSIFICATION_DSETS = ["collab", "IMDB-B", "imdb-multi", "rdt-b", "rdt-5k", "ppi", "imdb-binary"]


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
            dataset.dgl_graphs_file,
            dataset.jobs[worker_id]
            )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))

def worker_init_fn_mol_data(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.rwr_sub_graphs, _ = dgl.data.utils.load_graphs(
            dataset.dgl_graphs_file,
            # dataset.jobs[worker_id]
            )
    dataset.idx_to_sim_idxs = np.load(os.path.join(dataset.processed_dir, "node_idx_to_adj_idx_list_dict.npy"),
                                       allow_pickle=True).item()
    dataset.idx_to_sim_scores = np.load(os.path.join(dataset.processed_dir, "idx_to_sim_scores_array_dict.npy"),
                                         allow_pickle=True).item()
    dataset.length = len(dataset.rwr_sub_graphs)
    dataset.total = dataset.length
    np.random.seed(worker_info.seed % (2 ** 32))

def hvd_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
            dataset.dgl_graphs_file,
            dataset.jobs[worker_id]
            )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed((worker_info.seed % (2 ** 32) + hvd.rank()) % (2 ** 32))
    #  print("hvd.rank=%d, hvd.local_rank=%d, worker_id=%d, dataset.length=%d" % (hvd.rank(), hvd.local_rank(), worker_id, dataset.length))

class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self, rw_hops=64, restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            num_workers=1,
            dgl_graphs_file="./data_bin/dgl/small.bin",
            num_samples=10000,
            num_copies=1,
            graph_transform=None,
            aug="rwr",
            num_neighbors=5):
        super(LoadBalanceGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size > 1)
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)['graph_sizes'].tolist()
        print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True)
        # Drop top 2 largest graphs
        # graph_sizes = graph_sizes[2:]
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def __len__(self):
        return self.num_samples * num_workers

    def __iter__(self):
        #  samples = np.random.randint(low=0, high=self.length, size=self.num_samples)
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(self.length, size=self.num_samples, replace=True, p=prob.numpy())
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        #  worker_info = torch.utils.data.get_worker_info()
        #  print("hvd.rank=%d, hvd.local_rank=%d, worker_id=%d, seed=%d, idx=%d" % (hvd.rank(), hvd.local_rank(), worker_info.id, worker_info.seed, idx))
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                    g=self.graphs[graph_idx],
                    seeds=[node_idx],
                    num_traces=1,
                    num_hops=step
                    )[0][0][-1].item()

        if self.aug == 'rwr':
            max_nodes_per_seed = max(self.rw_hops,
                    int(((self.graphs[graph_idx].in_degree(node_idx) ** 0.75) * math.e / (math.e-1) / self.restart_prob) + 0.5)
                    )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed)
        elif self.aug == 'ns':
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                    self.graphs[graph_idx]._graph,
                    dgl.utils.toindex([node_idx]).todgltensor(),
                    0, # batch_start_id
                    1, # batch_size
                    1, # workers
                    self.num_neighbors, # expand_factor
                    self.rw_hops, # num_hops
                    'out',
                    False,
                    prob)[0]
            nf1 = NodeFlow(
                    self.graphs[graph_idx],
                    nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                    self.graphs[graph_idx]._graph,
                    dgl.utils.toindex([other_node_idx]).todgltensor(),
                    0, # batch_start_id
                    1, # batch_size
                    1, # workers
                    self.num_neighbors, # expand_factor
                    self.rw_hops, # num_hops
                    'out',
                    False,
                    prob)[0]
            nf2 = NodeFlow(
                    self.graphs[graph_idx],
                    nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            traces = [trace1, trace2]

        graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                positional_embedding_size=self.positional_embedding_size,
                )
        graph_k = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=traces[1],
                positional_embedding_size=self.positional_embedding_size,
                )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k

# from to dgl graph
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, 
                 num_path=12,
                 subgraph_size=64, 
                 restart_prob=0.8, 
                 positional_embedding_size=32, 
                 step_dist=[1.0, 0.0, 0.0]):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.num_path = num_path
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size  > 1)
        graphs = []
        # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs

        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                    g=self.graphs[graph_idx],
                    seeds=[node_idx],
                    num_traces=1,
                    num_hops=step
                    )[0][0][-1].item()

        # max_nodes_per_seed = max(self.rw_hops,
        #         int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e-1) / self.restart_prob) + 0.5)
        #         )
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=64)

        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            [node_idx for _ in range(self.num_path)],
            length=self.rw_hops)

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces,
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=False,
            notCat=True)

        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            [other_node_idx for _ in range(self.num_path)],
            length=self.rw_hops)

        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces,
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=False,
            notCat=True)
        #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling

        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     length=64)
        # print(traces, _)

        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     dgl_graph,
        #     seeds=[seed],
        #     restart_prob=restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed)
        # print(traces)
        #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling

        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #     g=dgl_graph,
        #     seed=seed,
        #     trace=traces,
        #     entire_graph=False,
        #     notCat=True
        # )

        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [node_idx for __ in range(7)],
        #     length=64)

        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=node_idx,
        #         trace=traces[0],
        #         positional_embedding_size=self.positional_embedding_size,
        #         entire_graph=False,
        #         notCat=True
        #         )  # really goods?  # positional embedding_size....

        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [other_node_idx for __ in range(7)],
        #     length=64)
        # graph_k = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=other_node_idx,
        #         trace=traces[1],
        #         positional_embedding_size=self.positional_embedding_size,
        #         entire_graph=False,
        #         notCat=True
        #         )
        return graph_q, graph_k

import os
class GraphDatasetOtherContrast(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0]):
        super(GraphDatasetOtherContrast).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size  > 1)
        processed_dir = "data_bin/dgl/subgraphs2"
        processed_dir2 = "data_bin/dgl/subgraphs"

        graphs = []
        # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

        self.rwr_sub_graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}.bin".format(3))
        assert len(self.rwr_sub_graphs) == self.length
        self.idx_to_sim_idxs = np.load(os.path.join(processed_dir2, "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(3))).item()
        assert len(self.idx_to_sim_idxs) == self.length
        self.idx_to_sim_scores = np.load(os.path.join(processed_dir, "idx_to_sim_score_dict_{:d}.npy".format(3))).item()
        print(type(self.idx_to_sim_scores))
        assert len(self.idx_to_sim_scores) == self.length

    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def rwr_sample_pos_graphs_path_dis(self, dgl_bg, idx, num_samples=5, num_path=7):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(num_path)],
            # prob="pos_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        # todo: count the frequency and choose top k ones?
        # subv = torch.unique(traces).tolist()
        # subvs = list()
        # assert traces.size(0) ==

        candi_to_count = {}
        candi_to_dis_sum = {}
        for i in range(traces.size(0)):
            subv = traces[i, :].tolist()
            candi_to_min_dis = {}
            candi_to_cnt_tmp = {}
            for ic in range(len(subv) - 1, -1, -1):
                if subv[ic] != idx and subv[ic] != -1:
                    candi_to_min_dis[subv[ic]] = ic
                    if subv[ic] not in candi_to_cnt_tmp:
                        candi_to_cnt_tmp[subv[ic]] = 1
                    else:
                        candi_to_cnt_tmp[subv[ic]] += 1
            for candi in candi_to_min_dis:
                if candi not in candi_to_count:
                    candi_to_count[candi] = candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] = candi_to_min_dis[candi]
                else:
                    candi_to_count[candi] += candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] += candi_to_min_dis[candi]
        candi_to_mean_dis = {candi: float(candi_to_dis_sum[candi]) / float(candi_to_count[candi]) \
                             for candi in candi_to_count}

        return candi_to_mean_dis, candi_to_count

    def __getitem__(self, idx):
        graph_q = self.rwr_sub_graphs[idx]
        graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        if len(self.idx_to_sim_idxs[idx]) == 0:
            graph_k = graph_q.clone()
        else:
            sim_socres = torch.from_numpy(self.idx_to_sim_scores[idx])
            print(type(self.idx_to_sim_scores[idx]), type(sim_socres), sim_socres.size())
            if torch.max(sim_socres).item() > 1e-9:
                sim_socres = sim_socres / torch.max(sim_socres).item()
            chosen_prob = torch.softmax(sim_socres, dim=-1)
            chosen_idx = int(torch.multinomial(chosen_prob, 1)[0])
            graph_k = self.rwr_sub_graphs[self.idx_to_sim_idxs[idx][chosen_idx]]
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)

        # graph_idx, node_idx = self._convert_idx(idx)
        #
        # step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        # if step == 0:
        #     other_node_idx = node_idx
        # else:
        #     other_node_idx = dgl.contrib.sampling.random_walk(
        #             g=self.graphs[graph_idx],
        #             seeds=[node_idx],
        #             num_traces=1,
        #             num_hops=step
        #             )[0][0][-1].item()
        #
        # max_nodes_per_seed = max(self.rw_hops,
        #         int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e-1) / self.restart_prob) + 0.5)
        #         )
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed)
        # #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling
        #
        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=node_idx,
        #         trace=traces[0],
        #         positional_embedding_size=self.positional_embedding_size,
        #         engire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )  # really goods?  # positional embedding_size....
        # graph_k = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=other_node_idx,
        #         trace=traces[1],
        #         positional_embedding_size=self.positional_embedding_size,
        #         entire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )
        return graph_q, graph_k


class GraphDatasetOtherContrastSimReSam(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0]):
        super(GraphDatasetOtherContrastSimReSam).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size  > 1)
        # processed_dir = "data_bin/dgl/subgraphs2"
        # processed_dir2 = "data_bin/dgl/subgraphs"

        processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs2"
        processed_dir2 = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs"

        # graphs = []
        graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir2, "some_small_graphs.bin"))
        # for name in ["cs", "physics"]:
        #     g = Coauthor(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # for name in ["computers", "photo"]:
        #     g = AmazonCoBuy(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

        self.rwr_sub_graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir,
                                                                         "subgraphs_rwr_{:d}.bin".format(3)))
        assert len(self.rwr_sub_graphs) == self.length
        self.idx_to_sim_idxs = np.load(os.path.join(processed_dir2,
                                                    "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(3)),
                                       allow_pickle=True).item()
        assert len(self.idx_to_sim_idxs) == self.length
        self.idx_to_sim_scores = np.load(os.path.join(processed_dir,
                                                      "idx_to_sim_score_dict_{:d}.npy".format(3)),
                                         allow_pickle=True).item()
        print(type(self.idx_to_sim_scores))
        assert len(self.idx_to_sim_scores) == self.length

    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def rwr_sample_pos_graphs_path_dis(self, dgl_bg, idx, num_samples=5, num_path=7):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(num_path)],
            # prob="pos_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        # todo: count the frequency and choose top k ones?
        # subv = torch.unique(traces).tolist()
        # subvs = list()
        # assert traces.size(0) ==

        candi_to_count = {}
        candi_to_dis_sum = {}
        for i in range(traces.size(0)):
            subv = traces[i, :].tolist()
            candi_to_min_dis = {}
            candi_to_cnt_tmp = {}
            for ic in range(len(subv) - 1, -1, -1):
                if subv[ic] != idx and subv[ic] != -1:
                    candi_to_min_dis[subv[ic]] = ic
                    if subv[ic] not in candi_to_cnt_tmp:
                        candi_to_cnt_tmp[subv[ic]] = 1
                    else:
                        candi_to_cnt_tmp[subv[ic]] += 1
            for candi in candi_to_min_dis:
                if candi not in candi_to_count:
                    candi_to_count[candi] = candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] = candi_to_min_dis[candi]
                else:
                    candi_to_count[candi] += candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] += candi_to_min_dis[candi]
        candi_to_mean_dis = {candi: float(candi_to_dis_sum[candi]) / float(candi_to_count[candi]) \
                             for candi in candi_to_count}

        return candi_to_mean_dis, candi_to_count

    def get_rwr_sampled_graphs(self, idx, num_path=7, num_samples=5):
        gra_idx, nod_idx = self._convert_idx(idx)
        traces, _ = dgl.sampling.random_walk(
            self.graphs[gra_idx],
            [nod_idx for __ in range(num_path)],
            # prob="pos_sample_p",
            length=16)
        graph_rw = data_util._rwr_trace_to_dgl_graph(self.graphs[gra_idx], nod_idx, traces, notCat=True,
                                                     use_g=False, entire_graph=False,
                                                     positional_embedding_size=self.positional_embedding_size)
        return graph_rw

    def __getitem__(self, idx):
        # graph_q = self.rwr_sub_graphs[idx]
        # graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        graph_q = self.get_rwr_sampled_graphs(idx)
        if len(self.idx_to_sim_idxs[idx]) == 0:
            # graph_k = graph_q.clone()
            graph_k = self.get_rwr_sampled_graphs(idx)
        else:
            sim_socres = torch.from_numpy(self.idx_to_sim_scores[idx])
            # print(type(self.idx_to_sim_scores[idx]), type(sim_socres), sim_socres.size())
            if torch.max(sim_socres).item() > 1e-9:
                sim_socres = sim_socres / torch.max(sim_socres).item()
            chosen_prob = torch.softmax(sim_socres, dim=-1)
            chosen_idx = int(torch.multinomial(chosen_prob, 1)[0])
            raw_idx = self.idx_to_sim_idxs[idx][chosen_idx]
            graph_k = self.get_rwr_sampled_graphs(raw_idx)
            # graph_k = self.rwr_sub_graphs[self.idx_to_sim_idxs[idx][chosen_idx]]
            # graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
        try:
            graph_q.ndata.pop("feat")
        except:
            pass

        try:
            graph_k.ndata.pop("feat")
        except:
            pass
        # graph_idx, node_idx = self._convert_idx(idx)
        #
        # step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        # if step == 0:
        #     other_node_idx = node_idx
        # else:
        #     other_node_idx = dgl.contrib.sampling.random_walk(
        #             g=self.graphs[graph_idx],
        #             seeds=[node_idx],
        #             num_traces=1,
        #             num_hops=step
        #             )[0][0][-1].item()
        #
        # max_nodes_per_seed = max(self.rw_hops,
        #         int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e-1) / self.restart_prob) + 0.5)
        #         )
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed)
        # #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling
        #
        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=node_idx,
        #         trace=traces[0],
        #         positional_embedding_size=self.positional_embedding_size,
        #         engire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )  # really goods?  # positional embedding_size....
        # graph_k = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=other_node_idx,
        #         trace=traces[1],
        #         positional_embedding_size=self.positional_embedding_size,
        #         entire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )
        return graph_q, graph_k


class GraphDatasetOtherContrastSimReSamL(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0]):
        super(GraphDatasetOtherContrastSimReSamL).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size  > 1)
        # processed_dir = "data_bin/dgl/subgraphs2"
        # processed_dir2 = "data_bin/dgl/subgraphs"

        processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs2"
        processed_dir2 = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs"
        processed_dir = "/apdcephfs/private_meowliu/ft_local/GCC/data"

        # graphs = []
        graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "small.bin"))
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

        root_path = "/apdcephfs/private_meowliu/ft_local/GCC/data"
        saved_nodes_edges_dir = os.path.join(root_path, "nodes_edges")
        self.idx_to_sim_idxs_arr = np.load(os.path.join(saved_nodes_edges_dir, "idx_to_candi_idx_list_array_pos.npy"),
                                           allow_pickle=True)
        print(self.idx_to_sim_idxs_arr.shape)
        self.idx_to_sim_idxs_num = np.load(os.path.join(saved_nodes_edges_dir, "idx_to_candi_idx_num_pos.npy"),
                                           allow_pickle=True).item()
        print(len(self.idx_to_sim_idxs_num), self.length)
        assert len(self.idx_to_sim_idxs_num) == self.length
        assert self.idx_to_sim_idxs_arr.shape[0] == self.length
        self.idx_to_sim_scores = torch.full((self.length, self.idx_to_sim_idxs_arr.shape[1]), 0.5,
                                            requires_grad=False)
        self.idx_to_sim_idxs_arr = torch.from_numpy(self.idx_to_sim_idxs_arr)


    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def rwr_sample_pos_graphs_path_dis(self, dgl_bg, idx, num_samples=5, num_path=7):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(num_path)],
            # prob="pos_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        # todo: count the frequency and choose top k ones?
        # subv = torch.unique(traces).tolist()
        # subvs = list()
        # assert traces.size(0) ==

        candi_to_count = {}
        candi_to_dis_sum = {}
        for i in range(traces.size(0)):
            subv = traces[i, :].tolist()
            candi_to_min_dis = {}
            candi_to_cnt_tmp = {}
            for ic in range(len(subv) - 1, -1, -1):
                if subv[ic] != idx and subv[ic] != -1:
                    candi_to_min_dis[subv[ic]] = ic
                    if subv[ic] not in candi_to_cnt_tmp:
                        candi_to_cnt_tmp[subv[ic]] = 1
                    else:
                        candi_to_cnt_tmp[subv[ic]] += 1
            for candi in candi_to_min_dis:
                if candi not in candi_to_count:
                    candi_to_count[candi] = candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] = candi_to_min_dis[candi]
                else:
                    candi_to_count[candi] += candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] += candi_to_min_dis[candi]
        candi_to_mean_dis = {candi: float(candi_to_dis_sum[candi]) / float(candi_to_count[candi]) \
                             for candi in candi_to_count}

        return candi_to_mean_dis, candi_to_count

    def get_rwr_sampled_graphs(self, idx, num_path=7, num_samples=5):
        gra_idx, nod_idx = self._convert_idx(idx)
        traces, _ = dgl.sampling.random_walk(
            self.graphs[gra_idx],
            [nod_idx for __ in range(num_path)],
            # prob="pos_sample_p",
            length=16)
        graph_rw = data_util._rwr_trace_to_dgl_graph(self.graphs[gra_idx], nod_idx, traces, notCat=True,
                                                     use_g=False, entire_graph=False,
                                                     positional_embedding_size=self.positional_embedding_size)
        return graph_rw

    def renew_sims(self, batch_q, sim_scores):
        self_idx = batch_q.ndata["self_idx"]
        idxes = batch_q.ndata["sim_idx"]
        self.idx_to_sim_scores[self_idx, idxes] = sim_scores

    def __getitem__(self, idx):
        # graph_q = self.rwr_sub_graphs[idx]
        # graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        # print("getting item 0", idx)
        graph_q = self.get_rwr_sampled_graphs(idx)
        # print("in get item 1", idx)
        if self.idx_to_sim_idxs_num[idx] <= 0:
            # graph_k = graph_q.clone()
            # print("in get item 2", idx)
            graph_k = self.get_rwr_sampled_graphs(idx)
            # graph_q.ndata["self_idx"] = torch.tensor([idx], dtype=torch.long)
            # graph_q.ndata["sim_idx"] = torch.tensor([0], dtype=torch.long)
        else:
            # print("in get item 3", idx)
            sim_socres = self.idx_to_sim_scores[idx][: self.idx_to_sim_idxs_num[idx]]
            # print(type(self.idx_to_sim_scores[idx]), type(sim_socres), sim_socres.size())
            # sim_socres = sim_socres / (torch.sum(sim_socres) + 1e-9)
            if torch.max(torch.abs(sim_socres)).item() > 1e-9:
                sim_socres = sim_socres / torch.max(torch.abs(sim_socres)).item()
            chosen_prob = torch.softmax(sim_socres, dim=-1)
            chosen_idx = int(torch.multinomial(chosen_prob, 1)[0])
            raw_idx = self.idx_to_sim_idxs_arr[idx][chosen_idx].item()
            # print(raw_idx)
            graph_k = self.get_rwr_sampled_graphs(raw_idx)
            # graph_q.ndata["self_idx"] = torch.tensor([idx], dtype=torch.long)
            # graph_q.ndata["sim_idx"] = torch.tensor([chosen_idx], dtype=torch.long)
            # graph_k = self.rwr_sub_graphs[self.idx_to_sim_idxs[idx][chosen_idx]]
            # graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
        try:
            graph_q.ndata.pop("feat")
        except:
            pass

        try:
            graph_k.ndata.pop("feat")
        except:
            pass
        # graph_idx, node_idx = self._convert_idx(idx)
        #
        # step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        # if step == 0:
        #     other_node_idx = node_idx
        # else:
        #     other_node_idx = dgl.contrib.sampling.random_walk(
        #             g=self.graphs[graph_idx],
        #             seeds=[node_idx],
        #             num_traces=1,
        #             num_hops=step
        #             )[0][0][-1].item()
        #
        # max_nodes_per_seed = max(self.rw_hops,
        #         int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e-1) / self.restart_prob) + 0.5)
        #         )
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=max_nodes_per_seed)
        # #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling
        #
        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=node_idx,
        #         trace=traces[0],
        #         positional_embedding_size=self.positional_embedding_size,
        #         engire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )  # really goods?  # positional embedding_size....
        # graph_k = data_util._rwr_trace_to_dgl_graph(
        #         g=self.graphs[graph_idx],
        #         seed=other_node_idx,
        #         trace=traces[1],
        #         positional_embedding_size=self.positional_embedding_size,
        #         entire_graph=hasattr(self, "entire_graph") and self.entire_graph
        #         )
        return graph_q, graph_k


class GraphDatasetOtherContrastMolDataset(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64,
                 subgraph_size=64,
                 restart_prob=0.8,
                 positional_embedding_size=32,
                 step_dist=[1.0, 0.0, 0.0],
                 env="normal"):
        super(GraphDatasetOtherContrastMolDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(positional_embedding_size  > 1)
        processed_dir = "data_bin/dgl/subgraphs2"
        processed_dir2 = "data_bin/dgl/subgraphs"
        processed_dir = "/apdcephfs/share_1142145/meowliu/gnn_pretraining/data/zinc_standard_agent/processed"
        self.processed_dir = processed_dir
        # graphs = []
        # # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        # for name in ["cs", "physics"]:
        #     g = Coauthor(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # for name in ["computers", "photo"]:
        #     g = AmazonCoBuy(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # # more graphs are comming ...
        # print("load graph done")
        # self.graphs = graphs
        # self.length = sum([g.number_of_nodes() for g in self.graphs])
        # self.total = self.length
        self.dgl_graphs_file = os.path.join(processed_dir, "dgl_graphs.bin")
        self.total = 2000000
        self.length = 2000000

        # num_copies = 1
        # assert num_workers % num_copies == 0
        # jobs = [list() for i in range(num_workers // num_copies)]
        # workloads = [0] * (num_workers // num_copies)
        # graph_sizes = sorted(
        #     enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        # )
        # for idx, size in graph_sizes:
        #     argmin = workloads.index(min(workloads))
        #     workloads[argmin] += size
        #     jobs[argmin].append(idx)
        # self.jobs = jobs * num_copies
        # self.total = self.num_samples * num_workers
        # self.rwr_sub_graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "dgl_graphs.bin"))
        # # assert len(self.rwr_sub_graphs) == self.length
        # self.length = len(self.rwr_sub_graphs)
        # self.total = self.length
        # self.idx_to_sim_idxs = np.load(os.path.join(processed_dir, "node_idx_to_adj_idx_list_dict.npy"),
        #                                allow_pickle=True).item()
        # print(type(self.idx_to_sim_idxs), len(self.idx_to_sim_idxs), self.length)
        # assert len(self.idx_to_sim_idxs) == self.length
        # self.idx_to_sim_scores = np.load(os.path.join(processed_dir, "idx_to_sim_scores_array_dict.npy"),
        #                                  allow_pickle=True).item()
        # print(type(self.idx_to_sim_scores))
        # assert len(self.idx_to_sim_scores) == self.length

    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_q = self.rwr_sub_graphs[idx]
        graph_q = graph_q.subgraph(graph_q.nodes())
        node_idx = self.rwr_sub_graphs[idx].out_degrees().argmax().item()

        # graph_q = data_util._rwr_trace_to_dgl_graph(graph_q, node_idx)
        graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        graph_q.ndata['seed'] = torch.zeros(graph_q.number_of_nodes(), dtype=torch.long)
        graph_q.ndata['seed'][0] = 1
        if len(self.idx_to_sim_idxs[idx]) == 0:
            # print("is zero")
            # graph_k = graph_q.clone()
            # graph_k = graph_q.clone()
            graph_k = self.rwr_sub_graphs[idx]
            graph_k = graph_k.subgraph(graph_k.nodes())
            node_idx = self.rwr_sub_graphs[idx].out_degrees().argmax().item()
            # graph_q = data_util._rwr_trace_to_dgl_graph(graph_q, node_idx)
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
            graph_k.ndata['seed'] = torch.zeros(graph_k.number_of_nodes(), dtype=torch.long)
            graph_k.ndata['seed'][0] = 1
        else:
            sim_socres = torch.from_numpy(self.idx_to_sim_scores[idx])
            # print(type(self.idx_to_sim_scores[idx]), type(sim_socres), sim_socres.size())
            if torch.max(sim_socres).item() > 1e-9:
                sim_socres = sim_socres / torch.max(sim_socres).item()
            chosen_prob = torch.softmax(sim_socres, dim=-1)
            assert abs(float(torch.sum(chosen_prob)) - 1.0) < 1e-5
            chosen_idx = int(torch.multinomial(chosen_prob, 1)[0])
            assert self.idx_to_sim_idxs[idx][chosen_idx] >= 0 and self.idx_to_sim_idxs[idx][chosen_idx] < len(self.rwr_sub_graphs)
            graph_k = self.rwr_sub_graphs[self.idx_to_sim_idxs[idx][chosen_idx]]
            graph_k = graph_k.subgraph(graph_k.nodes())
            node_idx = graph_k.out_degrees().argmax().item()
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
            graph_k.ndata['seed'] = torch.zeros(graph_k.number_of_nodes(), dtype=torch.long)
            assert node_idx >= 0 and node_idx < graph_k.number_of_nodes()
            graph_k.ndata['seed'][0] = 1
        # print(graph_q.number_of_nodes(), graph_k.number_of_nodes())
        return graph_q, graph_k


import pickle
class GraphDatasetOtherContrastOtherSample(torch.utils.data.Dataset):
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, positional_embedding_size=32,
                 step_dist=[1.0, 0.0, 0.0],
                 num_path=7,
                 env="normal"):
        super(GraphDatasetOtherContrastOtherSample).__init__()
        self.rw_hops = rw_hops
        self.num_path = num_path
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert (positional_embedding_size > 1)
        processed_dir = "data_bin/dgl/subgraphs2"
        processed_dir2 = "data_bin/dgl/subgraphs"

        if env == "jizhi":
            processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs2"
            processed_dir2 = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/data_bin/dgl/subgraphs"

        # graphs = []
        # # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
        # for name in ["cs", "physics"]:
        #     g = Coauthor(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # for name in ["computers", "photo"]:
        #     g = AmazonCoBuy(name)[0]
        #     g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        #     g.readonly()
        #     graphs.append(g)
        # # more graphs are comming ...
        # print("load graph done")
        # self.graphs = graphs
        # self.length = sum([g.number_of_nodes() for g in self.graphs])
        # self.total = self.length
        self.num_samples = 1

        self.rwr_sub_graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "subgraphs_rwr_{:d}.bin".format(3)))
        self.length = len(self.rwr_sub_graphs)
        self.total = self.length
        assert len(self.rwr_sub_graphs) == self.length
        self.idx_to_sim_idxs = np.load(
            os.path.join(processed_dir2, "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(3)), allow_pickle=True).item()
        assert len(self.idx_to_sim_idxs) == self.length
        # self.idx_to_sim_scores = np.load(os.path.join(processed_dir, "idx_to_sim_score_dict_{:d}.npy".format(3))).item()
        # print(type(self.idx_to_sim_scores))
        # assert len(self.idx_to_sim_scores) == self.length
        with open(os.path.join(processed_dir2, "big_graph_dgl_so_network.pkl"), "rb") as f:
            self.dgl_big_gra = pickle.load(f)

    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = data_util.plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = data_util.plot_to_image(figure_k)
        return image_q, image_k

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def rwr_sample_pos_graphs_path_dis(self, dgl_bg, idx, num_samples=5, num_path=7):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(num_path)],
            # prob="pos_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        # todo: count the frequency and choose top k ones?
        # subv = torch.unique(traces).tolist()
        # subvs = list()
        # assert traces.size(0) ==

        candi_to_count = {}
        candi_to_dis_sum = {}
        for i in range(traces.size(0)):
            subv = traces[i, :].tolist()
            candi_to_min_dis = {}
            candi_to_cnt_tmp = {}
            for ic in range(len(subv) - 1, -1, -1):
                if subv[ic] != idx and subv[ic] != -1:
                    candi_to_min_dis[subv[ic]] = ic
                    if subv[ic] not in candi_to_cnt_tmp:
                        candi_to_cnt_tmp[subv[ic]] = 1
                    else:
                        candi_to_cnt_tmp[subv[ic]] += 1
            for candi in candi_to_min_dis:
                if candi not in candi_to_count:
                    candi_to_count[candi] = candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] = candi_to_min_dis[candi]
                else:
                    candi_to_count[candi] += candi_to_cnt_tmp[candi]
                    candi_to_dis_sum[candi] += candi_to_min_dis[candi]
        candi_to_mean_dis = {candi: float(candi_to_dis_sum[candi]) / float(candi_to_count[candi]) \
                             for candi in candi_to_count}

        return candi_to_mean_dis, candi_to_count

    def __getitem__(self, idx):
        graph_q = self.rwr_sub_graphs[idx]
        graph_q = data_util._add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        if len(self.idx_to_sim_idxs[idx]) == 0:
            graph_k = graph_q.clone()
        else:
            _, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(self.dgl_big_gra, idx, self.rw_hops, self.num_path)
            sampled_nodes = list(sampled_idx_to_count.keys())
            # sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])
            sorted_sampled_idx = sorted(sampled_nodes, key=lambda idxx: sampled_idx_to_count[idxx], reverse=True)

            num_pos_samples = self.num_samples
            pos_sampled_idx = sorted_sampled_idx

            if len(pos_sampled_idx) == 0:
                pos_sampled_idx = [idx for __ in range(num_pos_samples)]
            else:
                pos_idx_to_cnt = [sampled_idx_to_count[pos_idx] for pos_idx in pos_sampled_idx]
                pos_idx_to_cnt_tsr = torch.tensor(pos_idx_to_cnt, dtype=torch.float64)
                pos_idx_to_prob = torch.softmax(pos_idx_to_cnt_tsr, dim=0)
                pos_sampled_idxs = torch.multinomial(pos_idx_to_prob,
                                                     replacement=True, num_samples=self.num_samples)
                pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[i_pos])] for i_pos \
                                   in range(pos_sampled_idxs.size(0))]
                # sampled_idx_count = [sampled_idx_to_count[sam_idx] for sam_idx in pos_sampled_idx]

            chosen_idx = pos_sampled_idx[0]
            graph_k = self.rwr_sub_graphs[chosen_idx]
            graph_k = data_util._add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)

        return graph_q, graph_k


from data_util_2 import create_node_classification_dataset
class CogDLGraphDataset(GraphDataset):
    def __init__(self, dataset,
                 rw_hops=64, num_path=12, subgraph_size=64, restart_prob=0.8,
                 positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0],
                 env="normal"):
        self.env = env
        self.rw_hops = rw_hops
        self.num_path = num_path
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert(positional_embedding_size > 1)

        class tmp():
            # HACK
            pass
        args = tmp()
        args.dataset = dataset
        if dataset == "imdb-multi":
            self._create_graphs_from_local_file("IMDB-MULTI")
        else:
            # self.data = build_dataset(args)[0]
            # self.graphs = [self._create_dgl_graph(self.data)]
            self.data = create_node_classification_dataset(dataset).data
            self.graphs = [self._create_dgl_graph(self.data)]
            # self.length = sum([g.number_of_nodes() for g in self.graphs])
            # self.total = self.length

        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length


    def _create_graphs_from_local_file(self, dataset_name):
        raw_path = "./datasets/{}".format(dataset_name)
        if self.env == "jizhi":
            raw_path = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/{}".format(dataset_name)
        graph_idx_to_edge_pairs = np.load(os.path.join(raw_path, "node_to_edge_pairs.npy")).item()
        print(len(graph_idx_to_edge_pairs))
        self.graphs = []
        for j in graph_idx_to_edge_pairs:
            edge_tensor = torch.tensor(graph_idx_to_edge_pairs[j], dtype=torch.long)
            max_nodes = edge_tensor.max() + 1
            graph = dgl.DGLGraph()
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
            graph.add_nodes(max_nodes)
            graph.add_edges(src, dst)
            graph.readonly()
            self.graphs.append(graph)
        print(len(self.graphs))

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        # assert all(graph.out_degrees() != 0)
        graph.readonly()
        return graph

from dgl.data.tu import TUDataset

class CogDLGraphClassificationDataset(CogDLGraphDataset):
    def __init__(self, dataset, rw_hops=64, subgraph_size=64,
                 restart_prob=0.8,
                 positional_embedding_size=32,
                 step_dist=[1.0, 0.0, 0.0],
                 env="normal"):
        self.rw_hops = rw_hops
        self.env = env
        print("env = ", self.env)
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = True
        assert(positional_embedding_size > 1)

        class tmp():
            # HACK
            pass
        args = tmp()
        args.dataset = dataset
        args.task = 'unsupervised_graph_classification'

        if dataset == "imdb-multi":
            self._create_graphs_from_local_file("IMDB-MULTI")
        elif dataset == "rdt-5k":
            self._create_graphs_from_local_file("REDDIT-MULTI-5K")
        elif dataset == "imdb-binary":
            self._create_graphs_from_local_file("IMDB-BINARY")
        elif dataset == "rdt-b":
            self._create_graphs_from_local_file("REDDIT-BINARY")
        elif dataset == "collab":
            self._create_graphs_from_local_file("COLLAB")
        else:
            # self.data = build_dataset(args)[0]
            # self.graphs = [self._create_dgl_graph(self.data)]

            # self.dataset = build_dataset(args)
            # self.graphs = [self._create_dgl_graph(data) for data in self.dataset]
            print("creating datasets from tu datasets %s" % dataset)
            self.dataset = self._create_graph_datasets_from_tu(dataset)
            self.graphs = self.dataset.graph_lists
            print("created! tot_number: %d" % len(self.graphs))

        self.length = len(self.graphs)
        self.total = self.length

    def _create_graph_datasets_from_tu(self, dataset_name):
        name = {
            "imdb-binary": "IMDB-BINARY",
            "imdb-multi": "IMDB-MULTI",
            "rdt-b": "REDDIT-BINARY",
            "rdt-5k": "REDDIT-MULTI-5K",
            "collab": "COLLAB",
        }[dataset_name]
        dataset = TUDataset(name)
        dataset.num_labels = dataset.num_labels[0]
        dataset.graph_labels = dataset.graph_labels.squeeze()
        return dataset

    def _create_graphs_from_local_file(self, dataset_name):
        raw_path = "./datasets/{}".format(dataset_name)
        # if self.env == "jizhi":
        raw_path = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/{}".format(dataset_name)
        graph_idx_to_edge_pairs = np.load(os.path.join(raw_path, "node_to_edge_pairs.npy"), allow_pickle=True).item()
        print(len(graph_idx_to_edge_pairs))
        self.graphs = []
        labels = np.load(os.path.join(raw_path, "graph_idx_to_label.npy"), allow_pickle=True).item()
        self.labels = labels
        max_lab = 0
        min_key = min(list(self.labels.keys()))
        labels_list = [labels[i + min_key] for i in range(len(labels))]
        if min(labels_list) == 1:
            self.labels = torch.tensor(labels_list, dtype=torch.long) - 1
        else:
            self.labels = torch.tensor(labels_list, dtype=torch.long)
        if dataset_name == "REDDIT-BINARY":
            for i in range(self.labels.size(0)):
                if int(self.labels[i]) < 0:
                    self.labels[i] = 0
        for j in graph_idx_to_edge_pairs:
            edge_tensor = torch.tensor(graph_idx_to_edge_pairs[j], dtype=torch.long)
            max_nodes = edge_tensor.max() + 1
            graph = dgl.DGLGraph()
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
            graph.add_nodes(max_nodes)
            graph.add_edges(src, dst)
            graph.readonly()
            self.graphs.append(graph)
        print(len(self.graphs))

    def _convert_idx(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()
        return graph_idx, node_idx

class CogDLGraphClassificationDatasetLabeled(CogDLGraphClassificationDataset):
    def __init__(self, dataset,
                 rw_hops=64, subgraph_size=64, restart_prob=0.8,
                 positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0], env="normal"):
        super(CogDLGraphClassificationDatasetLabeled, self).__init__(dataset,
                                                                     rw_hops, subgraph_size,
                                                                     restart_prob, positional_embedding_size,
                                                                     step_dist, env=env)
        if dataset in ["imdb-multi", "rdt-5k", "imdb-binary", "rdt-b", "collab"]:
            self.num_classes = self.labels.max().item() + 1
            self.dataset_name = dataset
        else:
            # self.num_classes = self.dataset.data.y.max().item() + 1
            self.num_classes = self.dataset.num_labels
            self.dataset_name = dataset
        self.entire_graph = True
        self.dict = [self.getitem(idx) for idx in range(len(self))]
    
    def __getitem__(self, idx):
        return self.dict[idx]
    
    def getitem(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()

        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            [node_idx for _ in range(self.rw_hops * 6)],
            restart_prob=self.restart_prob,
            length=self.rw_hops)

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces,
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=False,
            notCat=True,
            use_g=True)
        # print("using g")
        # print("total graph size = ", self.graphs[graph_idx].number_of_nodes())
        # print("sampled graph size = ", graph_q.number_of_nodes())

        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[graph_idx],
        #     seeds=[node_idx],
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=self.rw_hops)
        #
        # graph_q = data_util._rwr_trace_to_dgl_graph(
        #     g=self.graphs[graph_idx],
        #     seed=node_idx,
        #     trace=traces[0],
        #     positional_embedding_size=self.positional_embedding_size,
        #     entire_graph=True,
        # )
        # print(self.dataset_name)
        if self.dataset_name in ["imdb-multi", "rdt-5k", "imdb-binary", "rdt-b", "collab"]:
            lab = self.labels[graph_idx].item()
        else:
            # lab = self.dataset.data.y[graph_idx].item()
            lab = self.dataset.graph_labels[graph_idx].item()

        return graph_q, lab

class CogDLGraphDatasetLabeled(CogDLGraphDataset):
    def __init__(self, dataset, rw_hops=64, subgraph_size=64, restart_prob=0.8,
                 positional_embedding_size=32, step_dist=[1.0, 0.0, 0.0], cat_prone=False,
                 env="normal",
                 num_path=7): # gin ?
        super(CogDLGraphDatasetLabeled, self).__init__(dataset, rw_hops,
                                                       subgraph_size, restart_prob,
                                                       positional_embedding_size,
                                                       step_dist,
                                                       env=env)
        assert(len(self.graphs) == 1)
        self.num_classes = self.data.y.shape[1]
        self.rw_hops = rw_hops
        self.num_paths= num_path
        print("rw_hops, num_paths", self.rw_hops, self.num_paths)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [node_idx for _ in range(7)],
        #     restart_prob=self.restart_prob,
        #     length=16)
        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [node_idx for _ in range(7)],
        #     length=64)

        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            [node_idx for _ in range(self.num_paths)],
            length=self.rw_hops)

        # traces, _ = dgl.sampling.random_walk(
        #     self.graphs[graph_idx],
        #     [node_idx, other_node_idx],
        #     restart_prob=self.restart_prob,
        #     length=max_nodes_per_seed)  # train graph_moco   # 图没有取全。。

        graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces,
                positional_embedding_size=self.positional_embedding_size,
                entire_graph=False,
                notCat=True)
        return graph_q, self.data.y[idx].argmax().item()

if __name__ == '__main__':
    #  import horovod.torch as hvd
    #  hvd.init()
    num_workers=1
    import psutil
    mem = psutil.virtual_memory()
    print(mem.used/1024**3)
    graph_dataset = LoadBalanceGraphDataset(num_workers=num_workers, aug='ns', rw_hops=4, num_neighbors=5)
    mem = psutil.virtual_memory()
    print(mem.used/1024**3)
    graph_loader = torch.utils.data.DataLoader(
            graph_dataset,
            batch_size=1,
            collate_fn=data_util.batcher(),
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
            )
    mem = psutil.virtual_memory()
    print(mem.used/1024**3)
    for step, batch in enumerate(graph_loader):
        print("bs", batch[0].batch_size)
        print("n=", batch[0].number_of_nodes())
        print("m=", batch[0].number_of_edges())
        mem = psutil.virtual_memory()
        print(mem.used/1024**3)
        #  print(batch.graph_q)
        #  print(batch.graph_q.ndata['pos_directed'])
        print(batch[0].ndata['pos_undirected'])
    exit(0)
    graph_dataset = CogDLGraphDataset(dataset="wikipedia")
    pq, pk = graph_dataset.getplot(0)
    graph_loader = torch.utils.data.DataLoader(
            dataset=graph_dataset,
            batch_size=20,
            collate_fn=data_util.batcher(),
            shuffle=True,
            num_workers=4)
    for step, batch in enumerate(graph_loader):
        print(batch.graph_q)
        print(batch.graph_q.ndata['x'].shape)
        print(batch.graph_q.batch_size)
        print("max", batch.graph_q.edata['efeat'].max())
        break
