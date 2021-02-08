import os
import numpy as np

def raw_data_to_processed_data(dataset_name):
    # raw_path = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/{}".format(dataset_name)
    # if not os.path.exists(raw_path):
    #     os.mkdir(raw_path)
    # data_set_dir = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/%s" % (dataset_name)
    raw_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/%s" % dataset_name
    data_set_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/%s" % dataset_name
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
    with open(os.path.join(data_set_dir, "%s_graph_labels.txt" % dataset_name), "r") as rf:
        for i, line in enumerate(rf):
            lab = int(line.strip())
            gra_idx_to_label[i + 1] = lab
    np.save(os.path.join(raw_path, "graph_idx_to_label.npy"), gra_idx_to_label)
    print("gra_idx_to_label saved!")


import dgl
import torch
def merge_social_small_graphs():
    datasets = ["REDDIT-MULTI-12K", "github_stargazers", "tumblr_ct1", "tumblr_ct2", "twitch_egos"]
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"

    if os.path.exists(os.path.join(processed_dir, "social_dgl_graphs.bin")):
        dgl_graphs, _ = dgl.data.utils.load_graphs(
            os.path.join(processed_dir, "social_dgl_graphs.bin")
        )
        tot_gra_idx = len(dgl_graphs)
    else:
        tot_gra_idx = 0
        dgl_graphs = list()
    for ds in datasets:
        raw_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/%s" % ds
        gra_idx_to_node_list = np.load(os.path.join(raw_path, "node_to_edge_pairs.npy"),
                                       allow_pickle=True).item()
        for gra_idx in gra_idx_to_node_list:
            edge_tensor = torch.tensor(gra_idx_to_node_list[gra_idx], dtype=torch.long)
            max_nodes = edge_tensor.max() + 1
            graph = dgl.DGLGraph()
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
            graph.add_nodes(max_nodes)
            graph.add_edges(src, dst)
            graph.readonly()
            dgl_graphs.append(graph)
            tot_gra_idx += 1
    print(tot_gra_idx)

    labels = torch.tensor(range(tot_gra_idx), dtype=torch.long)
    dgl.data.utils.save_graphs(processed_dir + "/social_dgl_graphs.bin", dgl_graphs,
                               {"graph_idx_labels": labels})
    print("saved!")

def sort_graphs():
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"
    dgl_graphs, _ = dgl.data.utils.load_graphs(
        os.path.join(processed_dir, "social_dgl_graphs.bin")
    )
    dgl_graphs = sorted(dgl_graphs, key=lambda gra: gra.number_of_nodes(), reverse=False)
    labels = torch.tensor(range(len(dgl_graphs)), dtype=torch.long)
    dgl.data.utils.save_graphs(processed_dir + "/social_dgl_graphs_sorted.bin", dgl_graphs,
                               {"graph_idx_labels": labels})
    print("saved!")

if __name__ == "__main__":
    # raw_data_to_processed_data("COLLAB")
    # datasets = ["REDDIT-MULTI-12K", "github_stargazers", "tumblr_ct1", "tumblr_ct2", "twitch_egos"]
    # for ds in datasets:
    #     raw_data_to_processed_data(ds)
    # merge_social_small_graphs()
    sort_graphs()