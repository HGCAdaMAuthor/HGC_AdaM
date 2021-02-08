import math
import numpy as np
import operator
import dgl
# import networkx as nx
# import matplotlib.pyplot as plt
import torch
from dgl.data import AmazonCoBuy, Coauthor
import dgl.data
# from dgl.nodeflow import NodeFlow

# from cogdl.datasets import build_dataset
# import data_util
# import horovod.torch as hvd
import os
# import data_util


# when we compute similarity between those graphs we should also consider the position of ego-user-nodes.


def extract_rhop_ego_networks(networkx_graph, dgl_graph, seed, cutoff=5):
    # node_to_dis = nx.single_source_shortest_path_length(networkx_graph, seed, cutoff)
    # ego_graph_idx = list(node_to_dis.keys())
    # dgl_subgraph = dgl.node_subgraph(dgl_graph, ego_graph_idx)
    rw_hops = 64
    restart_prob = 0.8
    # dgl_graph = dgl.graph(dgl_graph.edges())
    # dgl_graph = dgl.from_networkx(networkx_graph)
    max_nodes_per_seed = max(rw_hops,
                             int((dgl_graph.out_degree(seed) * math.e / (
                                         math.e - 1) / restart_prob) + 0.5)
                             )
    # print

    # print(dgl_graph.edges())
    # print("out_degree", dgl_graph.out_degree(seed))
    # print(len(dgl_graph.nodes()), len(dgl_graph.edges()[0]))

    traces, _ = dgl.sampling.random_walk(
        dgl_graph,
        [seed for __ in range(max_nodes_per_seed * 6)],
        restart_prob=0.8,
        length=max_nodes_per_seed)
    # print(traces, _)

    # traces = dgl.contrib.sampling.random_walk_with_restart(
    #     dgl_graph,
    #     seeds=[seed],
    #     restart_prob=restart_prob,
    #     max_nodes_per_seed=max_nodes_per_seed)
    # print(traces)
    #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling

    graph_q = data_util._rwr_trace_to_dgl_graph(
        g=dgl_graph,
        seed=seed,
        trace=traces,
        entire_graph=False,
        notCat=True
    )  # really goods?  # positional embedding_size....
    # print("num_of_sampled_graph_nodes", len(graph_q.nodes()))
    # print("num_nodes = ", len(dgl_subgraph.nodes()))
    return graph_q


def get_similarity(idx_interval=None, cutoff=3):
    if idx_interval is not None:
        l, r = idx_interval
    else:
        l, r = 0, 73832
    # if os.path.exists("data_bin/dgl/subgraphs/subgraphs_rwr_{:d}_{:d}.bin".format(cutoff, l)):
    #     print("saved!")
    #     return

    if not os.path.exists("data_bin/dgl/subgraphs2"):
        os.mkdir("data_bin/dgl/subgraphs2")
    graphs = list()
    # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/graphs.bin", [0, 1, 2])
    for name in ["cs", "physics"]:
        g = Coauthor(name)[0]
        g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        g.readonly() # read only graphs...
        graphs.append(g)
    for name in ["computers", "photo"]:
        g = AmazonCoBuy(name)[0]
        g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        g.readonly()
        graphs.append(g)
    # more graphs are comming ...
    print("load graph done")
    # graphs = graphs
    length = sum([g.number_of_nodes() for g in graphs])

    print("total length = ", length)


    networkx_graphs = list()
    for i, graph in enumerate(graphs):
        print(i)
        # networkx_graphs.append(dgl.to_networkx(graph))
        networkx_graphs.append(graph)
    node_to_graph_idx_dict, node_to_self_idx_dict = dict(), dict()
    cumsum_node_idx = 0
    rebuild_graphs = list()
    for i, graph in enumerate(graphs):
        print(i)
        dgl_g = dgl.DGLGraph()
        dgl_g.add_nodes(len(graph.nodes()))
        dgl_g.add_edges(graph.edges()[0], graph.edges()[1])
        # edge_dict = dict()
        # exi_edges = 0
        # for edge_idx in range(len(graph.edges()[0])):
        #     st_idx, ed_idx = int(graph.edges()[0][edge_idx]), int(graph.edges()[1][edge_idx])
        #     if (st_idx, ed_idx) not in edge_dict:
        #         edge_dict[(st_idx, ed_idx)] = 1
        #         edge_dict[(ed_idx, st_idx)] = 1
        #     else:
        #         exi_edges += 1
        #         # print(exi_edges)
        # print("exi_edges", exi_edges, len(graph.edges()[0]))
        # print(graph.edges()[0])
        # print(graph.edges()[1])
        rebuild_graphs.append(dgl_g)
        for j in range(graph.number_of_nodes()):
            node_to_graph_idx_dict[cumsum_node_idx] = i
            node_to_self_idx_dict[cumsum_node_idx] = j
            cumsum_node_idx += 1
    assert cumsum_node_idx == length
    dgl_subgraphs = list()
    graph_idx_labels = list()
    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        seed = node_to_self_idx_dict[i]
        graph_idx = node_to_graph_idx_dict[i]
        dgl_subgraph = extract_rhop_ego_networks(networkx_graphs[graph_idx], rebuild_graphs[graph_idx], seed, cutoff=cutoff)
        dgl_subgraphs.append(dgl_subgraph)
        graph_idx_labels.append(i)
    assert len(dgl_subgraphs) == r - l
    graph_idx_labels = torch.tensor(graph_idx_labels)
    dgl.data.utils.save_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}_{:d}.bin".format(cutoff, l), dgl_subgraphs, {"graph_idx_labels": graph_idx_labels})
    print("saved!")
    # return dgl_subgraphs

import multiprocessing
def get_similarity_all(num_cpu=20, cutoff=3, DEBUGE=True):
    if not DEBUGE:
        all_len = 80000
        actual_len = 73832
    else:
        all_len = 400
        actual_len = 400

    len_per = all_len // num_cpu
    print(all_len, actual_len, len_per)
    lis = [[i * len_per, min((i + 1) * len_per, actual_len)] for i in range(num_cpu)]
    pool = multiprocessing.Pool(processes=num_cpu)
    results = list()
    for i in range(len(lis)):
        # get_similarity([lis[i][0], lis[i][1]])
        results.append(pool.apply_async(get_similarity, ([lis[i][0], lis[i][1]],)))
    pool.close()
    pool.join()
    all_dgl_subgraphs = list()
    # sim_idx_to_score_dict_dict = dict()

    # for res in results:
    #     res = res.get()
    #     all_dgl_subgraphs.extend(res)
    #     print(len(all_dgl_subgraphs))
    # assert len(all_dgl_subgraphs) == actual_len
    # graph_idx_labels = torch.tensor(range(actual_len))
    # dgl.data.utils.save_graphs("data_bin/dgl/subgraphs/subgraphs_{:d}.bin".format(cutoff), all_dgl_subgraphs, {"graph_idx_labels": graph_idx_labels})
    # print("saved!")


def merge_graphs_parts(num_cpu=10, cutoff=3, DEBUGE=True):
    all_len = 80000
    actual_len = 73832
    len_per = all_len // num_cpu
    print(all_len, actual_len, len_per)
    lis = [[i * len_per, min((i + 1) * len_per, actual_len)] for i in range(num_cpu)]
    all_dgl_graphs = list()
    # all_graph_labels = dict()
    for i, interval in enumerate(lis):
        if interval[0] >= actual_len:
            continue
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}_{:d}.bin".format(cutoff, interval[0]))
        all_dgl_graphs.extend(graphs)
    all_graph_labels = {"graph_idx_labels": torch.tensor(range(actual_len))}
    dgl.data.utils.save_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}.bin".format(cutoff), all_dgl_graphs,
                               all_graph_labels)
    assert len(all_dgl_graphs) == actual_len
    print("all_graphs saved!")


### number of nodes; number of edges; max_degree; min_degree and other things



def get_candidate_one_way(dgl_graphs, target_actual_idx, idx_to_actual_idx, start_idx, end_idx, direction, max_candidate=70):
    candidates = list()
    cumsum_candi = 0
    target_dgl_graph = dgl_graphs[target_actual_idx]
    target_num_nodes = len(target_dgl_graph.nodes())
    target_num_edges = len(target_dgl_graph.edges()[0])

    for other_idx in range(start_idx, end_idx, direction):
        other_actual_idx = idx_to_actual_idx[other_idx]
        other_dgl_graph = dgl_graphs[other_actual_idx]
        ### now we have dgl graphs
        other_num_nodes = len(other_dgl_graph.nodes())
        other_num_edges = len(other_dgl_graph.edges()[0])
        if abs(float(target_num_nodes) - float(other_num_nodes)) / float(target_num_nodes) > 0.2:
            break

        # if target_num_edges == 0:

        if (target_num_edges == 0 and abs(other_num_edges - target_num_edges) <= 5) or (target_num_edges > 0 and abs(float(target_num_edges) - float(other_num_edges)) / float(target_num_edges) <= 0.2):
            if direction == -1:
                candidates.insert(0, other_actual_idx)
            else:
                candidates.append(other_actual_idx)
            cumsum_candi += 1
            if (cumsum_candi >= max_candidate):
                break
    return candidates


def order_graph_list(cutoff=3, max_candidate=70, DEBUGE=True):
    processed_dir = "data_bin/dgl/subgraphs2"
    if DEBUGE:
        print("Aaaa")
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs2/subgraphs_{:d}_160.bin".format(cutoff))
        print("Bbb")
    else:
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}.bin".format(cutoff))

    print("len_graphs", len(graphs))
    # paixu --- just store the ordered indexes?
    idx_to_num_nodes = dict()
    for i, graph in enumerate(graphs):
        idx_to_num_nodes[i] = len(graph.nodes())
    ordered_items = sorted(idx_to_num_nodes.items(), key=lambda i: i[1], reverse=False)
    print(ordered_items)
    order_idx_to_idx = {i: item[0] for i, item in enumerate(ordered_items)} # order_idx to actual idx
    # how to get number of cycles or other information?
    idx_to_candi_idx_list = dict()
    tot_len = len(graphs)
    assert len(ordered_items) == tot_len
    for i in range(tot_len):
        actual_idx = ordered_items[i][0]
        candi = get_candidate_one_way(graphs, actual_idx, order_idx_to_idx, i - 1, -1, -1, max_candidate=max_candidate)
        candi_right = get_candidate_one_way(graphs, actual_idx, order_idx_to_idx, i + 1, tot_len, 1, max_candidate=max_candidate)
        candi.extend(candi_right)
        if len(candi) > max_candidate:
            left_half_st_idx = (len(candi) - max_candidate) // 2
            candi = candi[left_half_st_idx: left_half_st_idx + max_candidate]
        # print(len(candi))
        if (i % 100 == 0):
            print(i, len(candi))
        idx_to_candi_idx_list[actual_idx] = candi
    if not DEBUGE:
        np.save(os.path.join(processed_dir, "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(cutoff)), idx_to_candi_idx_list)
    else:
        np.save(os.path.join(processed_dir, "idx_to_candi_idx_list_dict_{:d}_160.npy".format(cutoff)), idx_to_candi_idx_list)


from grakel import Graph
# from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import GraphKernel
def from_dgl_graph_to_kel_graph(dgl_graph):
    edges_fr, edges_to = dgl_graph.edges()
    edge_list = [(int(edges_fr[i]) + 1, int(edges_to[i]) + 1) for i in range(len(edges_fr))]
    num_nodes = len(dgl_graph.nodes())
    node_labels = {i + 1 : str(i + 1) for i in range(num_nodes)}
    kel_graph = Graph(edge_list, node_labels=node_labels)
    return kel_graph

# for different kernels to calculate similarity -- 写一个自动测试的脚本
# 还有哪些grpahs可以去计算？
# 思考一下现在的方法的问题iii

def compute_similarity_pair(s_dgl_graph, t_dgl_graph):
    # s_edges_fr, s_edges_to = s_dgl_graph.edges()
    # t_edges_fr, t_edges_to = t_dgl_graph.edges()
    # s_edge_list = [(int(s_edges_fr[i]), int(s_edges_to[i])) for i in range(len(s_edges_fr))]
    # t_edge_list = [(int(t_edges_fr[i]), int(t_edges_to[i])) for i in range(len(t_edges_fr))]
    # s_kel_graph = Graph(s_edge_list)
    # t_kel_graph = Graph(t_edge_list)
    s_kel_graph = from_dgl_graph_to_kel_graph(s_dgl_graph)
    t_kel_graph = from_dgl_graph_to_kel_graph(t_dgl_graph)
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], Nystroem=20)
    # wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
    K_train = gk.fit_transform([s_kel_graph])
    K_test = gk.fit_transform([t_kel_graph])
    # print(type(K_test))
    sim_score = np.sum(K_train[0] * K_test[0], axis=0)
    return float(sim_score)

def compute_similarity_batch(s_dgl_graph, t_dgl_graphs):
    s_kel_graph = from_dgl_graph_to_kel_graph(s_dgl_graph)
    t_kel_graphs = [from_dgl_graph_to_kel_graph(t_dgl_graph) for t_dgl_graph in t_dgl_graphs]
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], Nystroem=20)
    # wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
    K_train = gk.fit_transform([s_kel_graph])
    if len(t_kel_graphs) > 0:
        K_test = gk.fit_transform(t_kel_graphs)
        compute_sims = np.matmul(K_train, K_test.T)
        compute_sims = compute_sims[0]
        # print(compute_sims)
        # print(type(compute_sims))
    else:
        compute_sims = np.array([])
    # 1 x num_tests

    return compute_sims

def compute_similarity_batch_norm(s_dgl_graph, t_dgl_graphs, is_dgl=True):
    if is_dgl is True:
        s_kel_graph = from_dgl_graph_to_kel_graph(s_dgl_graph)
        t_kel_graphs = [from_dgl_graph_to_kel_graph(t_dgl_graph) for t_dgl_graph in t_dgl_graphs]
    else:
        s_kel_graph = s_dgl_graph
        t_kel_graphs = t_dgl_graphs
    gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}, "subtree_wl"], normalize=True)
    # wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
    K_train = gk.fit_transform([s_kel_graph])
    if len(t_kel_graphs) > 0:
        K_test = gk.transform(t_kel_graphs)
        compute_sims = K_test[:, 0]
    else:
        compute_sims = np.array([])
    # 1 x num_tests
    return compute_sims


def compute_similarity(interval=None, cutoff=3, DEBUGE=False):
    if interval is not None:
        l, r = interval
    else:
        l, r = 0, 73832
    processed_dir = "data_bin/dgl/subgraphs2"
    processed_dir2 = "data_bin/dgl/subgraphs"
    if DEBUGE:
        idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir, "idx_to_candi_idx_list_dict_rwr_{:d}_160.npy".format(cutoff))).item()
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs/subgraphs_{:d}_160.bin".format(cutoff))
    else:
        idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir2, "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(cutoff))).item()
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs2/subgraphs_rwr_{:d}.bin".format(cutoff))

    # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs/subgraphs_{:d}.bin".format(cutoff))
    # if DEBUGE:
    #     graphs = graphs[160: 200]
    print(len(graphs))
    idx_to_sim_scores = dict()
    # for i, s_graph in enumerate(graphs):
    #     if i % 100 == 0:
    #         print(i)
    #     candi_idx = idx_to_candi_idx_list_dict[i]
    #     candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
    #     sim_array = compute_similarity_batch(s_graph, candi_dgl_graphs)
    #     idx_to_sim_scores[i] = sim_array
    # np.save(os.path.join(processed_dir, "idx_to_sim_score_dict_{:d}.npy".format(cutoff)), idx_to_sim_scores)
    # print("saved!")

    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        s_graph = graphs[i]
        candi_idx = idx_to_candi_idx_list_dict[i]
        candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
        sim_array = compute_similarity_batch(s_graph, candi_dgl_graphs)
        idx_to_sim_scores[i] = sim_array
    np.save(os.path.join(processed_dir, "idx_to_sim_score_dict_{:d}_{:d}.npy".format(cutoff, l)), idx_to_sim_scores)
    print("saved!")


def compute_similarity_all(num_cpu=10, cutoff=3, DEBUGE=True):
    if not DEBUGE:
        all_len = 80000
        actual_len = 73832
    else:
        all_len = 400
        actual_len = 400

    len_per = all_len // num_cpu
    print(all_len, actual_len, len_per)
    lis = [[i * len_per, min((i + 1) * len_per, actual_len)] for i in range(num_cpu)]
    pool = multiprocessing.Pool(processes=num_cpu)
    results = list()
    for i in range(len(lis)):
        # get_similarity([lis[i][0], lis[i][1]])
        results.append(pool.apply_async(compute_similarity, ([lis[i][0], lis[i][1]],)))
    pool.close()
    pool.join()

def parse_dataset(dataset_name="IMDB-MULTI"):
    raw_path = "./datasets/{}".format(dataset_name)

    edge_path = os.path.join(raw_path, "{}.edges".format(dataset_name))
    label_path = os.path.join(raw_path, "{}.graph_labels".format(dataset_name))
    graph_idx_path = os.path.join(raw_path, "{}.graph_idx".format(dataset_name))
    graph_idx_to_label = dict()
    node_idx_to_graph_idx = dict()
    with open(label_path, "r") as rf:
        for i, line in enumerate(rf):
            label = int(line.strip())
            graph_idx_to_label[i] = label
            if i < 10:
                print(graph_idx_to_label[i])
    print("tot_graph", len(graph_idx_to_label))
    np.save(os.path.join(raw_path, "graph_idx_to_label.npy"), graph_idx_to_label)

    node_cumsum = dict()

    with open(graph_idx_path, "r") as rf:
        for i, line in enumerate(rf):
            gi = int(line.strip())
            assert gi >= 1
            node_idx_to_graph_idx[i] = gi - 1
            if i < 10:
                print(node_idx_to_graph_idx[i])
    print("tot_nodes", len(node_idx_to_graph_idx))

    graph_idx_to_node_idx = dict()

    node_to_edge_pairs = dict()
    for i in range(len(graph_idx_to_label)):
        node_to_edge_pairs[i] = list()
        graph_idx_to_node_idx[i] = list()
    for node_idx in node_idx_to_graph_idx:
        gi = node_idx_to_graph_idx[node_idx]
        graph_idx_to_node_idx[gi].append(node_idx)
    for gi in graph_idx_to_node_idx:
        graph_idx_to_node_idx[gi] = sorted(graph_idx_to_node_idx[gi], reverse=False)
    node_idx_to_new_node_idx = dict()
    for gi in graph_idx_to_node_idx:
        for j, ni in enumerate(graph_idx_to_node_idx[gi]):
            assert ni not in node_idx_to_new_node_idx
            node_idx_to_new_node_idx[ni] = j
    totnot = 0
    tot = 0
    with open(edge_path, "r") as rf:
        for line in rf:
            qq = line.strip().split(",")
            a, b = int(qq[0]) - 1, int(qq[1]) - 1
            gia, gib = node_idx_to_graph_idx[a], node_idx_to_graph_idx[b]
            tot += 1
            if gia == gib:
                # assert gia == gib
                node_to_edge_pairs[gia].append((node_idx_to_new_node_idx[a], node_idx_to_new_node_idx[b]))
            else:
                totnot += 1
    print(totnot, tot)

    np.save(os.path.join(raw_path, "node_to_edge_pairs.npy"), node_to_edge_pairs)
    print("saved")


import pickle
def get_big_gra():
    processed_dir = "data_bin/dgl/subgraphs"
    idx_to_sim_idxs = np.load(
        os.path.join(processed_dir, "idx_to_candi_idx_list_dict_rwr_{:d}.npy".format(3))).item()
    fr_idx_list = list()
    to_idx_list = list()
    for fr_idx in idx_to_sim_idxs:
        to_idxes = idx_to_sim_idxs[fr_idx]
        # for to_idx in to_idxes:
        #     fr_idx_list.append(fr_idx)
        #     to_idx_list.append(to_idx)
        fr_idx_list += [fr_idx for __ in range(len(to_idxes))]
        to_idx_list += to_idxes
    dgl_big_gra = dgl.graph((torch.tensor(fr_idx_list, dtype=torch.long),
                            torch.tensor(to_idx_list, dtype=torch.long)))
    with open(os.path.join(processed_dir, "big_graph_dgl_so_network.pkl"), "wb") as f:
        pickle.dump(dgl_big_gra, f)
    print("big graph dumped!")

def _convert_idx(graphs, idx):
    graph_idx = 0
    node_idx = idx
    for i in range(len(graphs)):
        if  node_idx < graphs[i].number_of_nodes():
            graph_idx = i
            break
        else:
            node_idx -= graphs[i].number_of_nodes()
    return graph_idx, node_idx

def gcc_rw_sample(node_idx, graph, rw_hops=64, restart_prob=0.0, step_dist=[1.0, 0.0, 0.0]):

    step = np.random.choice(len(step_dist), 1, p=step_dist)[0]
    if step == 0:
        other_node_idx = node_idx
    else:
        other_node_idx = dgl.sampling.random_walk(
                g=graph,
                seeds=[node_idx],
                num_traces=1,
                num_hops=step
                )[0][0][-1].item()

    # max_nodes_per_seed = max(rw_hops,
    #         int((self.graphs[graph_idx].out_degree(node_idx) * math.e / (math.e-1) / restart_prob) + 0.5)
    #         )
    # traces, _ = dgl.sampling.random_walk(
    #     dgl_bg,
    #     [idx for __ in range(num_path)],
    #     prob="pos_sample_p",
    #     restart_prob=self.restart_prob,
    #     length=num_samples)
    max_nodes_per_seed = 16
    # traces, _ = dgl.sampling.random_walk(
    #     graph,
    #     [node_idx, other_node_idx],
    #     restart_prob=restart_prob,
    #     length=max_nodes_per_seed)

    traces, _ = dgl.sampling.random_walk(
        graph,
        [node_idx for __ in range(1)],
        restart_prob=restart_prob,
        length=max_nodes_per_seed)
    #  dgl.contrib.sampling.sampler._CAPI_NeighborSampling

    graph_q = data_util._rwr_trace_to_dgl_graph(
            g=graph,
            seed=node_idx,
            trace=traces,
            positional_embedding_size=64,
            entire_graph=False,
            notCat=True
            )  # really goods?  # positional embedding_size....

    traces, _ = dgl.sampling.random_walk(
        graph,
        [other_node_idx for __ in range(1)],
        restart_prob=restart_prob,
        length=max_nodes_per_seed)
    graph_k = data_util._rwr_trace_to_dgl_graph(
            g=graph,
            seed=other_node_idx,
            trace=traces,
            positional_embedding_size=64,
            entire_graph=False,
            notCat=True
            )
    return graph_q, graph_k

def rebuild_dgl_graph(dgl_gra):
    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(len(dgl_gra.nodes()))
    dgl_g.add_edges(dgl_gra.edges()[0], dgl_gra.edges()[1])
    return dgl_g

# the restart?
def get_sims(sampled_keys=None):
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
    # self.graphs = graphs
    lenn = sum([g.number_of_nodes() for g in graphs])
    print(lenn)
    gcc_sim = []
    if sampled_keys is None:
        sampled_keys = np.random.choice(range(lenn), 100, replace=False)
    sampled_keys = [int(sampled_keys[i]) for i in range(100)]
    for node_raw_idx in sampled_keys:
        # if node_raw_idx >= 10:
        #     break
        graph_idx, node_idx = _convert_idx(graphs, node_raw_idx)
        gra_q, gra_k = gcc_rw_sample(node_idx, rebuild_dgl_graph(graphs[graph_idx]), 128, 0.0)
        gra_q = rebuild_dgl_graph(gra_q)
        gra_k = rebuild_dgl_graph(gra_k)
        gcc_sim.append(compute_similarity_pair(gra_q, gra_k))
        # if node_raw_idx % 100 == 0:
        #     print(gcc_sim[-1])
    print(max(gcc_sim), min(gcc_sim), sum(gcc_sim) / (len(gcc_sim)))


def get_sims_calcu(sampled_keys=None):
    processed_dir = "data_bin/dgl/subgraphs2"
    sim_array_dict = np.load(os.path.join(processed_dir, "idx_to_sim_score_dict_{:d}.npy".format(3))).item()

    print(type(sim_array_dict))
    print(len(sim_array_dict))

    keys = list(sim_array_dict.keys())

    sim_per_array = sim_array_dict[keys[0]]
    print(len(sim_per_array))
    maxx = list()
    if sampled_keys is None:
        sampled_keys = np.random.choice(keys, 100, replace=False)
    sampled_keys = [int(sampled_keys[i]) for i in range(100)]
    for j in sampled_keys:
        maxx.append(np.max(sim_array_dict[j]).item())
    print("all_max_mean:", sum(maxx) / len(maxx))
    print(np.max(sim_per_array), np.min(sim_per_array), np.mean(sim_per_array), keys[0])
    print(sim_per_array)


# 83.92684911835453 (16) 77.77774058021008(32) 85.02853275225702(64) 82.36022071290078(128)
# 101.96628650999438 (pre_compute)
# 286.1845520277264 v.s. 91.33255441887677 (16)
# 205.02792454131296 v.s. 89.54076011492765 (32)
# 191.7264287651423 v.s. 89.99143356032481 (64)
# 272.1194740618265 v.s. 89.07308052184806 (128)

def raw_data_to_processed_data(dataset_name):
    raw_path = "./datasets/{}".format(dataset_name)
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
    data_set_dir = "/home/lxyww7/.dgl/%s/%s" % (dataset_name, dataset_name)
    node_idx_to_gra_idx = dict()
    gra_idx_to_node_list = dict()
    with open(os.path.join(data_set_dir, "%s_graph_indicator.txt" % dataset_name), "r") as rf:
        for i, line in enumerate(rf):
            gra_idx = int(line.strip())
            node_idx_to_gra_idx[i + 1] = gra_idx
            if gra_idx not in gra_idx_to_node_list:
                gra_idx_to_node_list[gra_idx] = [i + 1]
            else:
                gra_idx_to_node_list[gra_idx].append(i + 1)
    print("node_idx_to_gra_idx processed!")
    gra_idx_to_edge_list_dict = dict()
    with open(os.path.join(data_set_dir, "%s_A.txt" % dataset_name), "r") as rf:
        for i, line in enumerate(rf):
            ssl = line.strip().split(", ")
            a, b = int(ssl[0]), int(ssl[1])
            assert node_idx_to_gra_idx[a] == node_idx_to_gra_idx[b]
            gra_idx = node_idx_to_gra_idx[a]
            if gra_idx not in gra_idx_to_edge_list_dict:
                gra_idx_to_edge_list_dict[gra_idx] = [(a, b)]
            else:
                gra_idx_to_edge_list_dict[gra_idx].append((a, b))
    print("gra idx to edge list processed!!")
    for gra_idx in gra_idx_to_edge_list_dict:
        minn_node_idx = min(gra_idx_to_node_list[gra_idx])
        gra_idx_to_edge_list_dict[gra_idx] = [(node_idx[0] - minn_node_idx, node_idx[1] - minn_node_idx) for node_idx \
                                              in gra_idx_to_edge_list_dict[gra_idx]]
    print("gra idx to edge list transferred!")
    np.save(os.path.join(raw_path, "node_to_edge_pairs.npy"), gra_idx_to_edge_list_dict)
    print("node pairs saved")

    gra_idx_to_label = dict()
    with open(os.path.join(data_set_dir, "%s_graph_labels.txt" % dataset_name), "rf") as rf:
        for i, line in enumerate(rf):
            lab = int(line.strip())
            gra_idx_to_label[i + 1] = lab
    np.save(os.path.join(raw_path, "graph_idx_to_label.npy"), gra_idx_to_label)
    print("gra_idx_to_label saved!")

def transfer_sim_infos(DEBUG=True):
    processed_dir = "/apdcephfs/share_1142145/meowliu/gnn_pretraining/data/zinc_standard_agent/processed"
    sim_to_score_dict_list = np.load(os.path.join(processed_dir, "sim_to_score_dict_list_wei_nc_natom.npy"),
                                     allow_pickle=True)
    idx_to_adjs = np.load(os.path.join(processed_dir, "node_idx_to_adj_idx_list_dict.npy"), allow_pickle=True).item()
    idx_to_sim_scores_array_dict = dict()
    for idx in idx_to_adjs:
        adj_idxes = idx_to_adjs[idx]
        sim_scores_dict = sim_to_score_dict_list[idx]
        assert isinstance(sim_scores_dict, dict)
        scores = list()
        for j, adj_idx in enumerate(adj_idxes):
            scores.append(sim_scores_dict[adj_idx])
        sim_scores_array = np.array(scores)
        idx_to_sim_scores_array_dict[idx] = sim_scores_array
        if DEBUG:
            print(idx, sim_scores_array.shape, len(adj_idxes), len(sim_scores_dict))
    np.save(os.path.join(processed_dir, "idx_to_sim_scores_array_dict.npy"), idx_to_sim_scores_array_dict)
    print("all saved")

# from data_util_2 import _create_graphs_from_local_file
def build_graph_pretraning_datasets():
    gra_datasets_name = list({
            "imdb-binary": "IMDB-BINARY",
            "imdb-multi": "IMDB-MULTI",
            "rdt-b": "REDDIT-BINARY",
            "rdt-5k": "REDDIT-MULTI-5K",
            "collab": "COLLAB",
        }.values())
    tot_dgl_gras = list()
    raw_path = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/ALL_GRA"
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
    for i, dataset in enumerate(gra_datasets_name):
        print(i, dataset)
        gras = _create_graphs_from_local_file(dataset)
        tot_dgl_gras += gras
    sorted_gras = sorted(tot_dgl_gras, key=lambda gra: gra.number_of_nodes(), reverse=False)

    gra_labels = torch.tensor(range(len(sorted_gras)), dtype=torch.long)
    all_graph_labels = {"graph_idx_labels": gra_labels}
    dgl.data.utils.save_graphs(os.path.join(raw_path, "tot_gra_cls_gras_sorted.bin"), sorted_gras,
                               all_graph_labels)
    print("all saved!")

def get_candidate_gras(max_candidate=70):
    # processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/ALL_GRA"

    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    # graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))
    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))

    print("len_graphs", len(graphs))
    order_idx_to_idx = {i: i for i in range(len(graphs))}
    # how to get number of cycles or other information?
    idx_to_candi_idx_list = dict()
    tot_len = len(graphs)
    for i in range(tot_len):
        actual_idx = i
        candi = get_candidate_one_way(graphs, actual_idx, order_idx_to_idx, i - 1, -1, -1,
                                      max_candidate=max_candidate)
        candi_right = get_candidate_one_way(graphs, actual_idx, order_idx_to_idx, i + 1, tot_len, 1,
                                            max_candidate=max_candidate)
        candi.extend(candi_right)
        if len(candi) > max_candidate:
            left_half_st_idx = (len(candi) - max_candidate) // 2
            candi = candi[left_half_st_idx: left_half_st_idx + max_candidate]
        # print(len(candi))
        if (i % 100 == 0):
            print(i, len(candi))
        idx_to_candi_idx_list[actual_idx] = candi

    np.save(os.path.join(processed_dir, "social_idx_to_candi_idx_list_dict.npy"),
            idx_to_candi_idx_list)


def compute_similarity_another():
    # if interval is not None:
    #     l, r = interval
    # else:
    #     l, r = 0, 73832
    processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/ALL_GRA"

    idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir, "idx_to_candi_idx_list_dict.npy"),
                                         allow_pickle=True).item()
    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))

    # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/subgraphs/subgraphs_{:d}.bin".format(cutoff))
    # if DEBUGE:
    #     graphs = graphs[160: 200]
    print(len(graphs))
    idx_to_sim_scores = dict()
    l, r = 0, len(graphs)

    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        s_graph = graphs[i]
        candi_idx = idx_to_candi_idx_list_dict[i]
        candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
        sim_array = compute_similarity_batch(s_graph, candi_dgl_graphs)
        idx_to_sim_scores[i] = sim_array
    np.save(os.path.join(processed_dir, "idx_to_sim_score_dict.npy"), idx_to_sim_scores)
    print("saved!")

def get_smaller_graphs():
    v = []
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
    length = sum([g.number_of_nodes() for g in graphs])
    print(length)
    graph_idx_labels = torch.tensor(range(len(graphs)))
    dgl.data.utils.save_graphs("data_bin/dgl/subgraphs2/some_small_graphs.bin", graphs,
                               {"graph_idx_labels": graph_idx_labels})
    print("saved!")

def get_nodes_edges_number():
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    # graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))
    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))
    gra_idx_to_nodes_number = dict()
    gra_idx_to_edges_number = dict()
    for i, gra in enumerate(graphs):
        gra_idx_to_edges_number[i] = len(gra.edges()[0])
        gra_idx_to_nodes_number[i] = gra.number_of_nodes()
    np.save(os.path.join(processed_dir, "social_dgl_gra_idx_to_nodes_number.npy"), gra_idx_to_nodes_number)
    np.save(os.path.join(processed_dir, "social_dgl_gra_idx_to_edges_number.npy"), gra_idx_to_edges_number)
    print("saved!")

def get_candidate_one_way_dict(gra_idx_to_nodes_num, gra_idx_to_edges_num,
                               target_actual_idx, idx_to_actual_idx, start_idx, end_idx, direction, max_candidate=70):
    candidates = list()
    cumsum_candi = 0
    # target_dgl_graph = dgl_graphs[target_actual_idx]
    # target_num_nodes = len(target_dgl_graph.nodes())
    target_num_nodes = gra_idx_to_nodes_num[target_actual_idx]
    # target_num_edges = len(target_dgl_graph.edges()[0])
    target_num_edges = gra_idx_to_edges_num[target_actual_idx]

    for other_idx in range(start_idx, end_idx, direction):
        other_actual_idx = idx_to_actual_idx[other_idx]
        # other_dgl_graph = dgl_graphs[other_actual_idx]
        ### now we have dgl graphs
        # other_num_nodes = len(other_dgl_graph.nodes())
        other_num_nodes = gra_idx_to_nodes_num[other_actual_idx]
        # other_num_edges = len(other_dgl_graph.edges()[0])
        other_num_edges = gra_idx_to_edges_num[other_actual_idx]
        if abs(float(target_num_nodes) - float(other_num_nodes)) / float(target_num_nodes) > 0.2:
            break

        # if target_num_edges == 0:

        if (target_num_edges == 0 and abs(other_num_edges - target_num_edges) <= 5) or (target_num_edges > 0 and abs(float(target_num_edges) - float(other_num_edges)) / float(target_num_edges) <= 0.2):
            if direction == -1:
                candidates.insert(0, other_actual_idx)
            else:
                candidates.append(other_actual_idx)
            cumsum_candi += 1
            if (cumsum_candi >= max_candidate):
                break
    return candidates

def get_candidate_gras_dict(max_candidate=70):
    # processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/ALL_GRA"
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    # graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))
    # graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))
    gra_idx_to_nodes_num = np.load(os.path.join(processed_dir, "social_dgl_gra_idx_to_nodes_number.npy"),
                                   allow_pickle=True).item()
    gra_idx_to_edges_num = np.load(os.path.join(processed_dir, "social_dgl_gra_idx_to_edges_number.npy"),
                                   allow_pickle=True).item()

    # print("len_graphs", len(graphs))
    # order_idx_to_idx = {i: i for i in range(len(graphs))}
    # # how to get number of cycles or other information?
    idx_to_candi_idx_list = dict()
    # tot_len = len(graphs)
    tot_len = len(gra_idx_to_edges_num)
    order_idx_to_idx = {i: i for i in range(tot_len)}
    for i in range(tot_len):
        actual_idx = i
        candi = get_candidate_one_way_dict(gra_idx_to_nodes_num, gra_idx_to_edges_num,
                                           actual_idx, order_idx_to_idx, i - 1, -1, -1,
                                      max_candidate=max_candidate)
        candi_right = get_candidate_one_way_dict(gra_idx_to_nodes_num, gra_idx_to_edges_num,
                                                 actual_idx, order_idx_to_idx, i + 1, tot_len, 1,
                                            max_candidate=max_candidate)
        candi.extend(candi_right)
        if len(candi) > max_candidate:
            left_half_st_idx = (len(candi) - max_candidate) // 2
            candi = candi[left_half_st_idx: left_half_st_idx + max_candidate]
        # print(len(candi))
        if (i % 100 == 0):
            print(i, len(candi))
        idx_to_candi_idx_list[actual_idx] = candi

    np.save(os.path.join(processed_dir, "social_idx_to_candi_idx_list_dict.npy"),
            idx_to_candi_idx_list)

def compute_similarity_norm(interval=None, cutoff=3, DEBUGE=False):
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir, "social_idx_to_candi_idx_list_dict.npy"),
                                         allow_pickle=True).item()
    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))

    print(len(graphs))
    print(type(idx_to_candi_idx_list_dict))
    idx_to_sim_scores = dict()
    if interval is None:
        l, r = 0, len(graphs)
    else:
        l, r = interval[0], interval[1]

    # l, r = 0, len(graphs)
    r = min(r, len(graphs))
    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        print(i)
        s_graph = graphs[i]
        candi_idx = idx_to_candi_idx_list_dict[i]
        candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
        sim_array = compute_similarity_batch_norm(s_graph, candi_dgl_graphs)
        idx_to_sim_scores[i] = sim_array
    np.save(os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict_{:d}.npy".format(l)), idx_to_sim_scores)
    print("saved!")

def from_dgl_gra_to_grakel_gra():
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))
    gra_kel_gra = list()
    for i, gra in enumerate(graphs):
        kel_gra = from_dgl_graph_to_kel_graph(gra)
        gra_kel_gra.append(kel_gra)
        if i % 100 == 0:
            print(i)
    np.save(os.path.join(processed_dir, "social_kel_gra_sorted.npy"), gra_kel_gra)


def compute_similarity_norm_gra_kel(interval=None, cutoff=3, DEBUGE=False):
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir, "social_idx_to_candi_idx_list_dict.npy"),
                                         allow_pickle=True).item()
    # graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))
    graphs = np.load(os.path.join(processed_dir, "social_kel_gra_sorted.npy"), allow_pickle=True)

    print(len(graphs))
    print(type(graphs))
    print(type(idx_to_candi_idx_list_dict))
    idx_to_sim_scores = dict()

    l, r = 0, len(graphs)
    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        s_graph = graphs[i]
        candi_idx = idx_to_candi_idx_list_dict[i]
        candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
        sim_array = compute_similarity_batch_norm(s_graph, candi_dgl_graphs, is_dgl=False)
        idx_to_sim_scores[i] = sim_array
    np.save(os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict.npy"), idx_to_sim_scores)
    print("saved!")

def compute_similarity_norm_other_set(interval=None, cutoff=3, DEBUGE=False):
    processed_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/ALL_GRA"

    idx_to_candi_idx_list_dict = np.load(os.path.join(processed_dir, "idx_to_candi_idx_list_dict.npy"),
                                         allow_pickle=True).item()
    graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))

    print(len(graphs))
    print(type(idx_to_candi_idx_list_dict))
    idx_to_sim_scores = dict()
    if interval is None:
        l, r = 0, len(graphs)
    else:
        l, r = interval[0], interval[1]

    # l, r = 0, len(graphs)
    r = min(r, len(graphs))
    for i in range(l, r):
        if i % 100 == 0:
            print(i)
        print(i)
        s_graph = graphs[i]
        candi_idx = idx_to_candi_idx_list_dict[i]
        candi_dgl_graphs = [graphs[jj] for jj in candi_idx]
        sim_array = compute_similarity_batch_norm(s_graph, candi_dgl_graphs)
        idx_to_sim_scores[i] = sim_array
    np.save(os.path.join(processed_dir, "combined_social_gra_idx_to_sim_score_dict_{:d}.npy".format(l)), idx_to_sim_scores)
    print("saved!")

def test_sims():
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"
    aa = np.load(os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict_100000.npy"),
                 allow_pickle=True).item()
    print(len(aa))
    # keys = list(aa.keys())
    # print(aa[keys[0]])
    # for a in keys:
    #     print(aa[a])



def test_graphs():
    graphs = []
    # graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
    for name in ["cs", "physics"]:
        g = Coauthor(name)[0]
        g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        g.readonly()
        graphs.append(g)
        print('load %s'% name, len(graphs))
    for name in ["computers", "photo"]:
        g = AmazonCoBuy(name)[0]
        g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
        g.readonly()
        graphs.append(g)
        print('load %s' % name, len(graphs))
    # more graphs are comming ...
    print("load graph done")

if __name__ == "__main__":
    # get_similarity(cutoff=5)

    # get_similarity_all(DEBUGE=False, num_cpu=10)
    # merge_graphs_parts()
    # order_graph_list(cutoff=3, DEBUGE=False)
    # compute_similarity(DEBUGE=False)
    # compute_similarity_all(DEBUGE=False)
    # parse_dataset()
    # get_big_gra()

    # sampled_keys = np.random.choice(range(73832), 100, replace=False)
    # get_sims(sampled_keys)
    # get_sims_calcu(sampled_keys)
    # raw_data_to_processed_data("REDDIT-MULTI-5K")
    # transfer_sim_infos(DEBUG=False)
    # get_sims()
    # build_graph_pretraning_datasets()
    # get_candidate_gras()
    # get_sims_calcu()
    # compute_similarity_another()
    # get_smaller_graphs()
    # get_candidate_gras()
    # get_nodes_edges_number()
    # get_candidate_gras_dict()
    # compute_similarity_batch_norm()
    # H2O_adjacency = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    # H2O_node_labels = {0: 'O', 1: 'H', 2: 'H'}
    # H2O = Graph(initialization_object=H2O_adjacency, node_labels=H2O_node_labels)
    # el = [[0, 1, 0, 2],
    #       [1, 0, 2, 0]]
    #
    # # H3O_adjacency = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    # # H3O_node_labels = {0: 'O', 1: 'H', 2: 'H', 3: 'H'}
    # # H3O = Graph(initialization_object=H3O_adjacency, node_labels=H3O_node_labels)
    # el3 = [[0, 1, 0, 2, 0, 3],
    #       [1, 0, 2, 0, 3, 0]]
    # ga = dgl.graph((el[0], el[1]))
    # gb = dgl.graph((el3[0], el3[1]))
    # compute_similarity_batch_norm(ga, [ga, gb])
    # compute_similarity_norm([0, 50000])
    # compute_similarity_norm([0, 25000])
    # compute_similarity_norm([25000, 50000])
    # compute_similarity_norm([50000, 75000])
    # compute_similarity_norm([75000, 90000])
    # compute_similarity_norm([90000, 100000])

    # compute_similarity_norm([100000, 105000])
    # compute_similarity_norm([105000, 110000])
    # compute_similarity_norm([110000, 115000])

    # compute_similarity_norm([115000, 120000])
    # compute_similarity_norm([120000, 125000])

    # compute_similarity_norm([125000, 130000])
    # compute_similarity_norm([130000, 135000])
    # compute_similarity_norm([135000, 140000])
    # compute_similarity_norm([140000, 145000])
    # compute_similarity_norm([145000, 150000])
    # compute_similarity_norm([150000, 153000])
    # compute_similarity_norm([153000, 155000])
    # compute_similarity_norm([155000, 156000])
    # compute_similarity_norm([156000, 157000])
    # test_sims()
    test_graphs()

    # compute_similarity_norm([125000, 150000])
    # compute_similarity_norm([150000, 200000])
    # compute_similarity_norm([50000, 100000])
    # compute_similarity_norm([100000, 150000])
    # compute_similarity_norm([150000, 200000])
    # compute_similarity_norm_other_set([0, 5000])
    # compute_similarity_norm_other_set([5000, 7500])
    # compute_similarity_norm_other_set([7500, 10000])
    # compute_similarity_norm_other_set([10000, 12500])
    # compute_similarity_norm_other_set([12500, 15000])
    # from_dgl_gra_to_grakel_gra()
    # compute_similarity_norm_gra_kel()
    # g1 = dgl.graph(([0, 1, 1, 2, 3, 1], [1, 2, 3, 0, 0, 0]))
    # traces, _ = dgl.sampling.random_walk(g1, [0, 0, 0, 0, 0], length=4, restart_prob=0.8)
    # print(traces)





