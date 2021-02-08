import torch
# import copy
import random
import networkx as nx
import numpy as np
# from torch_geometric.utils import convert
from loader import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple
from rdkit import Chem
from rdkit.Chem import AllChem
from loader import mol_to_graph_data_obj_simple, \
    graph_data_obj_to_mol_simple

# from loader import MoleculeDataset
import scipy.sparse as sparse
from scipy.sparse import linalg
# import karateclub as kc
import torch.nn.functional as F
# from karateclub.node_embedding.structural import graphwave
import sklearn.preprocessing as preprocessing
import dgl
from utils2 import gen_bfs_order


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)

def add_wavelet_emb(g):
    G = graph_data_obj_to_nx_simple(g)
    gv = graphwave.GraphWave()
    gv.fit(G)
    emb = gv.get_embedding()
    emb = torch.FloatTensor(emb)
    # print("getting emb")
    return emb

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype('float64')
    ncv=min(n, max(2*k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype('float64')
    for i in range(retry):
        try:
            s, u = linalg.eigsh(
                    laplacian,
                    k=k,
                    which='LA',
                    ncv=ncv,
                    v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            print("arpack error, retry=", i)
            ncv = min(ncv*2, n)
            if i + 1 == retry:
                sparse.save_npz('arpack_error_sparse_matrix.npz', laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm='l2')
    x = torch.from_numpy(x.astype('float32'))
    x = F.pad(x, (0, hidden_size-k), 'constant', 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
            dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5,
            dtype=float)
    laplacian = norm * adj * norm
    k=min(n-2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    return x
    # g.ndata['pos_undirected'] = x.float()
    # return g

class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0, i].cpu().item()) + "," + str(
            data.edge_index[1, i].cpu().item()) for i in
                        range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data


class ExtractSubstructureContextPairWithType:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

        self.num_atom_type = 119
        self.mask_edge = True
        self.num_edge_type = 5

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]

        data.num_nodes = num_atoms

        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        node_type = int(data.x[root_idx, 0])

        # first_two_layer_nodes_num = len(nx.single_source_shortest_path_length(G, root_idx, 2).keys())
        #
        # dgl_graph = dgl.DGLGraph(G)
        # traces = dgl.sampling.random_walk(dgl_graph, [root_idx], length=first_two_layer_nodes_num)[0]
        # traces = traces[0][1:]
        # rwr_sampled_nodes = []

        mask_node_labels_list = list()

        for atom_idx in range(num_atoms):
            mask_node_labels_list.append(torch.tensor([self.num_atom_type, 0], dtype=torch.long).view(1, -1))
        data.x_masked = torch.cat(mask_node_labels_list, dim=0)
        # print("x_masked", data.x_masked.size())

        if self.mask_edge:
            data.edge_attr_masked = []
            for bond_idx in range(data.edge_index.size(1)):
                data.edge_attr_masked.append(torch.tensor(
                    [self.num_edge_type, 0]
                ).view(1, -1))
            data.edge_attr_masked = torch.cat(data.edge_attr_masked, dim=0)
            # print(data.edge_attr_masked.size())

        first_approx_node_idxes = list(nx.single_source_shortest_path_length(G, root_idx, 1).keys())

        data.first_approx_node_idxes = torch.tensor(first_approx_node_idxes, dtype=torch.long)

        data.node_type = torch.tensor([node_type], dtype=torch.long)

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0

            for i in range(data.first_approx_node_idxes.size(0)):
                a = int(data.first_approx_node_idxes[i])
                data.first_approx_node_idxes[i] = substruct_node_map[a]

            # for i in range(traces.size(0)):
            #     a = int(traces[i])
            #     if a not in substruct_node_map:
            #         continue
            #     rwr_sampled_nodes.append(substruct_node_map[a])

            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # data.first_approx_node_idxes = torch.tensor(rwr_sampled_nodes, dtype=torch.long)

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


class ExtractFlowContrastData:
    def __init__(self, flow_model, disturb_type="Gaussian", disturb_ratio=0.15):
        self.flow = flow_model
        self.disturb_type = disturb_type
        self.disturb_ratio = disturb_ratio
        self.num_edge = 4

    def __call__(self, data, node_order=None, seed=None, mask_type=None):
        num_atom = data.x.size(0)
        if node_order is None:
            if seed == None:
                seed = random.sample(range(num_atom), 1)[0]
            ordered_nodes_dict = gen_bfs_order(data.edge_index, seed)
        elif isinstance(node_order, list):
            ordered_nodes_dict = {nod: i for i, nod in enumerate(node_order)}
        else:
            ordered_nodes_dict = node_order

        ordered_items = sorted(ordered_nodes_dict.items(), key=lambda x: x[1])
        ordered_nodes = [i[0] for i in ordered_items]
        ordered_nodes_dict_inv = {j: i for i, j in enumerate(ordered_nodes_dict)}

        # print("ordered nodes = ", ordered_nodes)

        edge_dict = dict()

        node_to_adj_nodes = dict()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            edge_dict[(a, b)] = i
            if ordered_nodes_dict[b] < ordered_nodes_dict[a]:
                if a not in node_to_adj_nodes:
                    node_to_adj_nodes[a] = [b]
                else:
                    node_to_adj_nodes[a].append(b)

        cnt = 0
        xx = list()  # the first subgraph
        xx.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))
        eil = list()
        eit = list()
        # batch = list()
        # batch.append(torch.full((1, ), 0, dtype=torch.long))
        dual_connected_edges = torch.empty((2, 0), dtype=torch.long)
        dual_connected_edges_attr = [torch.empty((0, 2), dtype=torch.long)]

        to_pred_edge_st_ed_idx = list()
        to_pred_edge_attr = list()
        to_pred_edge_subgraph_idx = list()

        # batch.append(torch.full((1, ), 0, dtype=torch.long))
        subgraph_idx = 0

        # xx[-1][0, :] = data.x[seed, :]

        pred_node_nodes_idx = list()
        pred_node_nodes_idx.append(torch.tensor([0], dtype=torch.long))  # 0 th node is used to pred the first node
        pred_nodes_to_node_idx = list()
        pred_nodes_to_node_idx.append(torch.tensor([0], dtype=torch.long))  # 0 th node

        split_edge_index = [torch.empty((2, 0), dtype=torch.long)]
        split_edge_attr = [torch.empty((0, 2), dtype=torch.long)]

        pred_edge_st_ed_idx = list()
        pred_edge_node_idx = list()
        pred_edge_nodes_to_edge_idx = list()
        edge_idx = 0  # 0 th edge
        pred_edge_attr = list()

        first_real_idx = ordered_nodes[0]
        to_pred_node_attr = [data.x[first_real_idx, :].view(1, -1)]

        current_connected_edges = torch.empty((2, 0), dtype=torch.long)
        current_edge_attr = torch.empty((0, 2), dtype=torch.long)

        connected_edges = torch.empty((2, 0), dtype=torch.long)
        connected_edges_attr = torch.empty((0, 2), dtype=torch.long)

        for i in range(1, len(ordered_nodes)):
            subgraph_idx += 1
            # batch.append(torch.full((i + 1, ), subgraph_idx, dtype=torch.long))  # node subgraph, with i + 1 nodes; and the last node unknown
            tmp_x = torch.zeros_like(xx[-1])
            cnt_l = cnt + 1
            cnt_r = cnt + (i + 1)

            if i == 1:
                assert int(xx[-1][-1, 0]) == 119, "at this time the previous node's attribute must not be assigned."
                # xx[-1][-1, :] = data.x[ordered_nodes[i - 1], :]
                tmp_x[:, :] = xx[-1][-1, :]
                tmp_x[-1, :] = data.x[ordered_nodes[i - 1], :]
            else:
                # print("going into i != 1")
                # cur_edge_index += cur_edge_index.clone() + i
                # split_edge_attr.append(cur_edge_attr.clone())
                # split_edge_index.append(cur_edge_index.clone())
                split_edge_index.append(current_connected_edges.clone() + cnt_l)
                split_edge_attr.append(current_edge_attr.clone())
                tmp_x[:, :] = xx[-1][:, :]
                assert int(xx[-1][-1, 0]) != 119, "at this time the previous node's attribute must be assgined!"
            # tmp_x[:, :] = xx[-1][:, :]

            # add one new node and set its featuers to masked node feature
            tmp_x = torch.cat((tmp_x, torch.tensor([119, 0], dtype=torch.long).view(1, -1)), dim=0)
            xx.append(tmp_x)
            to_pred_node_attr.append(data.x[ordered_nodes[i], :].view(1, -1))  # add prediction for current node

            pred_node_nodes_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
            pred_nodes_to_node_idx.append(
                torch.full((len(pred_node_nodes_idx[-1]),), i, dtype=torch.long))  # the i'th node

            cnt = cnt_r  # update cnt

            cur_pred_edges_x = torch.zeros_like(xx[-1])  # add the current node information
            cur_pred_edges_x[:, :] = xx[-1][:, :]
            cur_pred_edges_x[-1, :] = data.x[ordered_nodes[i], :]
            # cur_edge_index = torch.zeros_like(split_edge_index[-1])
            # cur_edge_index[:, :] = split_edge_index[-1][:, :]
            # cur_edge_attr = torch.zeros_like(split_edge_attr[-1])
            # cur_edge_attr[:, :] = split_edge_attr[-1][:, :]

            for j in range(i):
                # add edges
                xx.append(cur_pred_edges_x.clone())

                cnt_l = cnt + 1
                cnt_r = cnt + cur_pred_edges_x.size(0)

                cnt += cur_pred_edges_x.size(0)

                split_edge_index.append(current_connected_edges.clone() + cnt_l)
                split_edge_attr.append(current_edge_attr.clone())
                # cur_edge_index = cur_edge_index + cur_pred_edges_x.size(0)
                # split_edge_index.append(cur_edge_index.clone())
                # split_edge_attr.append(cur_edge_attr.clone())

                ed_idx = cnt
                st_idx = ed_idx - (i - j)

                # cnt_r = cnt
                # cnt_l = cnt - cur_pred_edges_x.size(0) + 1
                pred_edge_node_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
                pred_edge_nodes_to_edge_idx.append(torch.full((cnt_r - cnt_l + 1,), edge_idx, dtype=torch.long))
                edge_idx += 1
                pred_edge_st_ed_idx.append(torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1))

                ori_st_idx = ordered_nodes[j]
                ori_ed_idx = ordered_nodes[i]
                if ((ori_st_idx, ori_ed_idx) in edge_dict) or ((ori_ed_idx, ori_st_idx) in edge_dict):
                    ori_edge_idx = edge_dict[(ori_st_idx, ori_ed_idx)]
                    pred_edge_attr.append(data.edge_attr[ori_edge_idx, :].view(1, -1))
                    current_connected_edges = torch.cat(
                        [current_connected_edges, torch.tensor([j, i], dtype=torch.long).view(-1, 1)], dim=1)
                    current_connected_edges = torch.cat(
                        [current_connected_edges, torch.tensor([i, j], dtype=torch.long).view(-1, 1)], dim=1)
                    current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)],
                                                  dim=0)
                    current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)],
                                                  dim=0)
                    # cur_edge_index = torch.cat([cur_edge_index, torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1)], dim=1)
                    # cur_edge_index = torch.cat([cur_edge_index, torch.tensor([ed_idx, st_idx], dtype=torch.long).view(-1, 1)], dim=1)
                    # cur_edge_attr = torch.cat([cur_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                    # cur_edge_attr = torch.cat([cur_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                else:
                    pred_edge_attr.append(torch.tensor([self.num_edge, 0], dtype=torch.long).view(1, -1))

        data.x = torch.cat(to_pred_node_attr, dim=0)
        data.split_x = torch.cat(xx, dim=0)
        data.split_edge_index = torch.cat(split_edge_index, dim=1)
        data.split_edge_attr = torch.cat(split_edge_attr, dim=0)

        data.pred_nodes_node_idx = torch.cat(pred_node_nodes_idx, dim=-1)
        data.pred_nodes_to_node_idx = torch.cat(pred_nodes_to_node_idx, dim=-1)

        data.pred_edge_st_ed_idx = torch.cat(pred_edge_st_ed_idx, dim=1)
        data.pred_edge_node_idx = torch.cat(pred_edge_node_idx, dim=-1)
        data.pred_edge_nodes_to_edge_idx = torch.cat(pred_edge_nodes_to_edge_idx, dim=-1)
        data.pred_edge_attr = torch.cat(pred_edge_attr, dim=0)

        dist_x, dist_edge_index, dist_edge_attr = self.flow.generate_one_mole_given_original_mole(data, self.disturb_ratio)
        data.dist_x = dist_x
        data.dist_edge_index = dist_edge_index
        data.dist_edge_attr = dist_edge_attr
        data.edge_index = current_connected_edges # data.split_edge_index.clone()
        data.edge_attr = current_edge_attr # data.edge_attr.clone()
        return data


from copy import deepcopy
#### TODO: Unit Test...
class ExtractMaskedData:
    def __init__(self):
        self.num_edge = 4
        pass

    def __call__(self, data, node_order=None, mask_type=None, seed=None):
        num_atom = data.x.size(0)
        if node_order is None:
            if seed == None:
                seed = random.sample(range(num_atom), 1)[0]
            ordered_nodes_dict = gen_bfs_order(data.edge_index, seed)
        elif isinstance(node_order, list):
            ordered_nodes_dict = {nod: i for i, nod in enumerate(node_order)}
        else:
            ordered_nodes_dict = node_order

        ordered_items = sorted(ordered_nodes_dict.items(), key=lambda x: x[1])
        ordered_nodes = [i[0] for i in ordered_items]
        ordered_nodes_dict_inv = {j: i for i, j in enumerate(ordered_nodes_dict)}

        # print("ordered nodes = ", ordered_nodes)

        edge_dict = dict()

        node_to_adj_nodes = dict()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            edge_dict[(a, b)] = i
            if ordered_nodes_dict[b] < ordered_nodes_dict[a]:
                if a not in node_to_adj_nodes:
                    node_to_adj_nodes[a] = [b]
                else:
                    node_to_adj_nodes[a].append(b)

        cnt = 0
        xx = list()  # the first subgraph
        xx.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))
        eil = list()
        eit = list()
        # batch = list()
        # batch.append(torch.full((1, ), 0, dtype=torch.long))
        dual_connected_edges = torch.empty((2, 0), dtype=torch.long)
        dual_connected_edges_attr = [torch.empty((0, 2), dtype=torch.long)]

        to_pred_edge_st_ed_idx = list()
        to_pred_edge_attr = list()
        to_pred_edge_subgraph_idx = list()

        # batch.append(torch.full((1, ), 0, dtype=torch.long))
        subgraph_idx = 0

        # xx[-1][0, :] = data.x[seed, :]

        pred_node_nodes_idx = list()
        pred_node_nodes_idx.append(torch.tensor([0], dtype=torch.long)) # 0 th node is used to pred the first node
        pred_nodes_to_node_idx = list()
        pred_nodes_to_node_idx.append(torch.tensor([0], dtype=torch.long)) # 0 th node

        split_edge_index = [torch.empty((2, 0), dtype=torch.long)]
        split_edge_attr = [torch.empty((0, 2), dtype=torch.long)]

        pred_edge_st_ed_idx = list()
        pred_edge_node_idx = list()
        pred_edge_nodes_to_edge_idx = list()
        edge_idx = 0 # 0 th edge
        pred_edge_attr = list()

        first_real_idx = ordered_nodes[0]
        to_pred_node_attr = [data.x[first_real_idx, :].view(1, -1)]

        current_connected_edges = torch.empty((2, 0), dtype=torch.long)
        current_edge_attr = torch.empty((0, 2), dtype=torch.long)

        for i in range(1, len(ordered_nodes)):
            subgraph_idx += 1
            # batch.append(torch.full((i + 1, ), subgraph_idx, dtype=torch.long))  # node subgraph, with i + 1 nodes; and the last node unknown
            tmp_x = torch.zeros_like(xx[-1])
            cnt_l = cnt + 1
            cnt_r = cnt + (i + 1)

            if i == 1:
                assert int(xx[-1][-1, 0]) == 119, "at this time the previous node's attribute must not be assigned."
                # xx[-1][-1, :] = data.x[ordered_nodes[i - 1], :]
                tmp_x[:, :] = xx[-1][-1, :]
                tmp_x[-1, :] = data.x[ordered_nodes[i - 1], :]
            else:
                # print("going into i != 1")
                # cur_edge_index += cur_edge_index.clone() + i
                # split_edge_attr.append(cur_edge_attr.clone())
                # split_edge_index.append(cur_edge_index.clone())
                split_edge_index.append(current_connected_edges.clone() + cnt_l)
                split_edge_attr.append(current_edge_attr.clone())
                tmp_x[:, :] = xx[-1][:, :]
                assert int(xx[-1][-1, 0]) != 119, "at this time the previous node's attribute must be assgined!"
            # tmp_x[:, :] = xx[-1][:, :]

            # add one new node and set its featuers to masked node feature
            tmp_x = torch.cat((tmp_x, torch.tensor([119, 0], dtype=torch.long).view(1, -1)), dim=0)
            xx.append(tmp_x)
            to_pred_node_attr.append(data.x[ordered_nodes[i], :].view(1, -1))  # add prediction for current node

            pred_node_nodes_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
            pred_nodes_to_node_idx.append(torch.full((len(pred_node_nodes_idx[-1]), ), i, dtype=torch.long)) # the i'th node

            cnt = cnt_r # update cnt

            cur_pred_edges_x = torch.zeros_like(xx[-1])  # add the current node information
            cur_pred_edges_x[:, :] = xx[-1][:, :]
            cur_pred_edges_x[-1, :] = data.x[ordered_nodes[i], :]
            # cur_edge_index = torch.zeros_like(split_edge_index[-1])
            # cur_edge_index[:, :] = split_edge_index[-1][:, :]
            # cur_edge_attr = torch.zeros_like(split_edge_attr[-1])
            # cur_edge_attr[:, :] = split_edge_attr[-1][:, :]

            for j in range(i):
                # add edges
                xx.append(cur_pred_edges_x.clone())

                cnt_l = cnt + 1
                cnt_r = cnt + cur_pred_edges_x.size(0)

                cnt += cur_pred_edges_x.size(0)

                split_edge_index.append(current_connected_edges.clone() + cnt_l)
                split_edge_attr.append(current_edge_attr.clone())
                # cur_edge_index = cur_edge_index + cur_pred_edges_x.size(0)
                # split_edge_index.append(cur_edge_index.clone())
                # split_edge_attr.append(cur_edge_attr.clone())

                ed_idx = cnt
                st_idx = ed_idx - (i - j)

                # cnt_r = cnt
                # cnt_l = cnt - cur_pred_edges_x.size(0) + 1
                pred_edge_node_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
                pred_edge_nodes_to_edge_idx.append(torch.full((cnt_r - cnt_l + 1, ), edge_idx, dtype=torch.long))
                edge_idx += 1
                pred_edge_st_ed_idx.append(torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1))

                ori_st_idx = ordered_nodes[j]
                ori_ed_idx = ordered_nodes[i]
                if ((ori_st_idx, ori_ed_idx) in edge_dict) or ((ori_ed_idx, ori_st_idx) in edge_dict):
                    ori_edge_idx = edge_dict[(ori_st_idx, ori_ed_idx)]
                    pred_edge_attr.append(data.edge_attr[ori_edge_idx, :].view(1, -1))
                    current_connected_edges = torch.cat([current_connected_edges, torch.tensor([j, i], dtype=torch.long).view(-1, 1)], dim=1)
                    current_connected_edges = torch.cat([current_connected_edges, torch.tensor([i, j], dtype=torch.long).view(-1, 1)], dim=1)
                    current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                    current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                    # cur_edge_index = torch.cat([cur_edge_index, torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1)], dim=1)
                    # cur_edge_index = torch.cat([cur_edge_index, torch.tensor([ed_idx, st_idx], dtype=torch.long).view(-1, 1)], dim=1)
                    # cur_edge_attr = torch.cat([cur_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                    # cur_edge_attr = torch.cat([cur_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                else:
                    pred_edge_attr.append(torch.tensor([self.num_edge, 0], dtype=torch.long).view(1, -1))

        data.x = torch.cat(to_pred_node_attr, dim=0)
        data.split_x = torch.cat(xx, dim=0)
        data.split_edge_index = torch.cat(split_edge_index, dim=1)
        data.split_edge_attr = torch.cat(split_edge_attr, dim=0)

        data.pred_nodes_node_idx = torch.cat(pred_node_nodes_idx, dim=-1)
        data.pred_nodes_to_node_idx = torch.cat(pred_nodes_to_node_idx, dim=-1)

        data.pred_edge_st_ed_idx = torch.cat(pred_edge_st_ed_idx, dim=1)
        data.pred_edge_node_idx = torch.cat(pred_edge_node_idx, dim=-1)
        data.pred_edge_nodes_to_edge_idx = torch.cat(pred_edge_nodes_to_edge_idx, dim=-1)
        data.pred_edge_attr = torch.cat(pred_edge_attr, dim=0)

        return data

    def __repr__(self):
        return '{}'.format(
            self.__class__.__name__)


class ExtractMaskedDataVTwo:
    def __init__(self):
        self.num_edge = 4
        pass

    def __call__(self, data, node_order=None, mask_type=None, seed=None):
        num_atom = data.x.size(0)
        if node_order is None:
            if seed == None:
                seed = random.sample(range(num_atom), 1)[0]
            ordered_nodes_dict = gen_bfs_order(data.edge_index, seed)
        elif isinstance(node_order, list):
            ordered_nodes_dict = {nod: i for i, nod in enumerate(node_order)}
        else:
            ordered_nodes_dict = node_order

        ordered_items = sorted(ordered_nodes_dict.items(), key=lambda x: x[1])
        ordered_nodes = [i[0] for i in ordered_items]
        ordered_nodes_dict_inv = {j: i for i, j in enumerate(ordered_nodes_dict)}

        # print("ordered nodes = ", ordered_nodes)
        edge_dict = dict()

        # node_to_adj_nodes = dict()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            edge_dict[(a, b)] = i
            # if ordered_nodes_dict[b] < ordered_nodes_dict[a]:
            #     if a not in node_to_adj_nodes:
            #         node_to_adj_nodes[a] = [b]
            #     else:
            #         node_to_adj_nodes[a].append(b)

        last_node_idx = ordered_nodes[-1]
        to_pred_node_attr = [data.x[last_node_idx, :].view(1, -1)]
        altered_x_list = list()
        for i in range(data.x.size(0) - 1):
            now_node_idx = ordered_nodes[i]
            altered_x_list.append(data.x[now_node_idx, :].view(1, -1))
        altered_x_list.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))
        # data.split_x = torch.cat(altered_x_list, dim=0)
        altered_x = torch.cat(altered_x_list, dim=0)

        cnt = len(ordered_nodes) - 1
        xx = list()  # the first subgraph
        # xx.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))

        xx.append(altered_x)
        ccel = list()
        ccea = list()
        for i in range(0, data.edge_index.size(1), 2):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if a != last_node_idx and b != last_node_idx:
                ccel.append(torch.tensor([ordered_nodes_dict_inv[a], ordered_nodes_dict_inv[b]], dtype=torch.long).view(-1, 1))
                ccel.append(torch.tensor([ordered_nodes_dict_inv[b], ordered_nodes_dict_inv[a]], dtype=torch.long).view(-1, 1))
                ccea.append(data.edge_attr[i, :].view(1, -1))
                ccea.append(data.edge_attr[i, :].view(1, -1))

        current_connected_edges = torch.cat(ccel, dim=1)
        current_edge_attr = torch.cat(ccea, dim=0)

        subgraph_idx = 0

        # xx[-1][0, :] = data.x[seed, :]

        pred_node_nodes_idx = list()
        pred_node_nodes_idx.append(torch.arange(len(ordered_nodes), dtype=torch.long))
        pred_nodes_to_node_idx = list()
        pred_nodes_to_node_idx.append(torch.full((len(ordered_nodes), ), 0, dtype=torch.long)) # 0 th node

        split_edge_index = [current_connected_edges.clone()]
        split_edge_attr = [current_edge_attr.clone()]

        pred_edge_st_ed_idx = list()
        pred_edge_node_idx = list()
        pred_edge_nodes_to_edge_idx = list()
        edge_idx = 0 # 0 th edge
        pred_edge_attr = list()
        split_pred_edge_labels = list()

        # first_real_idx = ordered_nodes[0]
        # to_pred_node_attr = [data.x[first_real_idx, :].view(1, -1)]
        cur_pred_edges_x = xx[-1].clone()
        cur_pred_edges_x[-1, :] = data.x[last_node_idx, :]

        for j in range(len(ordered_nodes) - 1):
            # add edges

            xx.append(cur_pred_edges_x.clone())

            cnt_l = cnt + 1
            cnt_r = cnt + cur_pred_edges_x.size(0)

            cnt += cur_pred_edges_x.size(0)

            split_edge_index.append(current_connected_edges.clone() + cnt_l)
            split_edge_attr.append(current_edge_attr.clone())

            ed_idx = cnt
            st_idx = ed_idx - (len(ordered_nodes) - 1 - j)

            # cnt_r = cnt
            # cnt_l = cnt - cur_pred_edges_x.size(0) + 1
            pred_edge_node_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
            pred_edge_nodes_to_edge_idx.append(torch.full((cnt_r - cnt_l + 1, ), edge_idx, dtype=torch.long))
            edge_idx += 1
            pred_edge_st_ed_idx.append(torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1))

            ori_st_idx = ordered_nodes[j]
            ori_ed_idx = last_node_idx
            if ((ori_st_idx, ori_ed_idx) in edge_dict) or ((ori_ed_idx, ori_st_idx) in edge_dict):
                ori_edge_idx = edge_dict[(ori_st_idx, ori_ed_idx)]
                pred_edge_attr.append(data.edge_attr[ori_edge_idx, :].view(1, -1))
                split_pred_edge_labels.append(data.y.clone().view(1, -1))
                current_connected_edges = torch.cat([current_connected_edges, torch.tensor([j, len(ordered_nodes) - 1], dtype=torch.long).view(-1, 1)], dim=1)
                current_connected_edges = torch.cat([current_connected_edges, torch.tensor([len(ordered_nodes) - 1, j], dtype=torch.long).view(-1, 1)], dim=1)
                current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)

            else:
                pred_edge_attr.append(torch.tensor([self.num_edge, 0], dtype=torch.long).view(1, -1))
                split_pred_edge_labels.append(data.y.clone().view(1, -1))

        data.x = torch.cat(to_pred_node_attr, dim=0)
        data.split_x = torch.cat(xx, dim=0)
        data.split_edge_index = torch.cat(split_edge_index, dim=1)
        data.split_edge_attr = torch.cat(split_edge_attr, dim=0)

        data.pred_nodes_node_idx = torch.cat(pred_node_nodes_idx, dim=-1)
        data.pred_nodes_to_node_idx = torch.cat(pred_nodes_to_node_idx, dim=-1)

        data.pred_edge_st_ed_idx = torch.cat(pred_edge_st_ed_idx, dim=1)
        data.pred_edge_node_idx = torch.cat(pred_edge_node_idx, dim=-1)
        data.pred_edge_nodes_to_edge_idx = torch.cat(pred_edge_nodes_to_edge_idx, dim=-1)
        data.pred_edge_attr = torch.cat(pred_edge_attr, dim=0)
        data.split_pred_nodes_labels = data.y.clone().view(1, -1)
        # print("data.y.size() = ", data.y.size())
        data.split_pred_edges_labels = torch.cat(split_pred_edge_labels, dim=0)

        return data

    def __repr__(self):
        return '{}'.format(
            self.__class__.__name__)

class ExtractMaskedContrastDataVTwo:
    def __init__(self, flow_model):
        self.num_edge = 4
        self.k = 5
        self.flow = flow_model
        pass

    def __call__(self, data, node_order=None, mask_type=None, seed=None):
        num_atom = data.x.size(0)

        if node_order is None:
            if seed == None:
                seed = random.sample(range(num_atom), 1)[0]
            ordered_nodes_dict = gen_bfs_order(data.edge_index, seed)
        elif isinstance(node_order, list):
            ordered_nodes_dict = {nod: i for i, nod in enumerate(node_order)}
        else:
            ordered_nodes_dict = node_order

        ordered_items = sorted(ordered_nodes_dict.items(), key=lambda x: x[1])
        ordered_nodes = [i[0] for i in ordered_items]
        ordered_nodes_dict_inv = {j: i for i, j in enumerate(ordered_nodes_dict)}


        # print("ordered nodes = ", ordered_nodes)
        edge_dict = dict()

        # node_to_adj_nodes = dict()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            edge_dict[(a, b)] = i
            # if ordered_nodes_dict[b] < ordered_nodes_dict[a]:
            #     if a not in node_to_adj_nodes:
            #         node_to_adj_nodes[a] = [b]
            #     else:
            #         node_to_adj_nodes[a].append(b)

        last_node_idx = ordered_nodes[-1]
        to_pred_node_attr = [data.x[last_node_idx, :].view(1, -1)]
        altered_x_list = list()
        for i in range(data.x.size(0) - 1):
            now_node_idx = ordered_nodes[i]
            altered_x_list.append(data.x[now_node_idx, :].view(1, -1))
        altered_x_list.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))
        # data.split_x = torch.cat(altered_x_list, dim=0)
        altered_x = torch.cat(altered_x_list, dim=0)

        cnt = len(ordered_nodes) - 1
        xx = list()  # the first subgraph
        # xx.append(torch.tensor([119, 0], dtype=torch.long).view(1, -1))

        xx.append(altered_x)
        ccel = list()
        ccea = list()
        for i in range(0, data.edge_index.size(1), 2):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if a != last_node_idx and b != last_node_idx:
                ccel.append(torch.tensor([ordered_nodes_dict_inv[a], ordered_nodes_dict_inv[b]], dtype=torch.long).view(-1, 1))
                ccel.append(torch.tensor([ordered_nodes_dict_inv[b], ordered_nodes_dict_inv[a]], dtype=torch.long).view(-1, 1))
                ccea.append(data.edge_attr[i, :].view(1, -1))
                ccea.append(data.edge_attr[i, :].view(1, -1))

        current_connected_edges = torch.cat(ccel, dim=1)
        current_edge_attr = torch.cat(ccea, dim=0)

        subgraph_idx = 0

        # xx[-1][0, :] = data.x[seed, :]

        pred_node_nodes_idx = list()
        pred_node_nodes_idx.append(torch.arange(len(ordered_nodes), dtype=torch.long))
        pred_nodes_to_node_idx = list()
        pred_nodes_to_node_idx.append(torch.full((len(ordered_nodes), ), 0, dtype=torch.long)) # 0 th node

        split_edge_index = [current_connected_edges.clone()]
        split_edge_attr = [current_edge_attr.clone()]

        pred_edge_st_ed_idx = list()
        pred_edge_node_idx = list()
        pred_edge_nodes_to_edge_idx = list()
        edge_idx = 0 # 0 th edge
        pred_edge_attr = list()
        split_pred_edge_labels = list()

        # first_real_idx = ordered_nodes[0]
        # to_pred_node_attr = [data.x[first_real_idx, :].view(1, -1)]

        for j in range(len(ordered_nodes) - 1):
            # add edges
            cur_pred_edges_x = xx[-1].clone()
            cur_pred_edges_x[-1, :] = data.x[last_node_idx, :]
            xx.append(cur_pred_edges_x.clone())

            cnt_l = cnt + 1
            cnt_r = cnt + cur_pred_edges_x.size(0)

            cnt += cur_pred_edges_x.size(0)

            split_edge_index.append(current_connected_edges.clone() + cnt_l)
            split_edge_attr.append(current_edge_attr.clone())

            ed_idx = cnt
            st_idx = ed_idx - (len(ordered_nodes) - 1 - j)

            # cnt_r = cnt
            # cnt_l = cnt - cur_pred_edges_x.size(0) + 1
            pred_edge_node_idx.append(torch.arange(cnt_l, cnt_r + 1, dtype=torch.long))
            pred_edge_nodes_to_edge_idx.append(torch.full((cnt_r - cnt_l + 1, ), edge_idx, dtype=torch.long))
            edge_idx += 1
            pred_edge_st_ed_idx.append(torch.tensor([st_idx, ed_idx], dtype=torch.long).view(-1, 1))

            ori_st_idx = ordered_nodes[j]
            ori_ed_idx = last_node_idx
            if ((ori_st_idx, ori_ed_idx) in edge_dict) or ((ori_ed_idx, ori_st_idx) in edge_dict):
                ori_edge_idx = edge_dict[(ori_st_idx, ori_ed_idx)]
                pred_edge_attr.append(data.edge_attr[ori_edge_idx, :].view(1, -1))
                split_pred_edge_labels.append(data.y.clone().view(1, -1))
                current_connected_edges = torch.cat([current_connected_edges, torch.tensor([j, len(ordered_nodes) - 1], dtype=torch.long).view(-1, 1)], dim=1)
                current_connected_edges = torch.cat([current_connected_edges, torch.tensor([len(ordered_nodes) - 1, j], dtype=torch.long).view(-1, 1)], dim=1)
                current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)
                current_edge_attr = torch.cat([current_edge_attr, data.edge_attr[ori_edge_idx, :].view(1, -1)], dim=0)

            else:
                pred_edge_attr.append(torch.tensor([self.num_edge, 0], dtype=torch.long).view(1, -1))
                split_pred_edge_labels.append(data.y.clone().view(1, -1))


        data.x = torch.cat(to_pred_node_attr, dim=0)
        data.split_x = torch.cat(xx, dim=0)
        data.split_edge_index = torch.cat(split_edge_index, dim=1)
        data.split_edge_attr = torch.cat(split_edge_attr, dim=0)

        data.pred_nodes_node_idx = torch.cat(pred_node_nodes_idx, dim=-1)
        data.pred_nodes_to_node_idx = torch.cat(pred_nodes_to_node_idx, dim=-1)

        data.pred_edge_st_ed_idx = torch.cat(pred_edge_st_ed_idx, dim=1)
        data.pred_edge_node_idx = torch.cat(pred_edge_node_idx, dim=-1)
        data.pred_edge_nodes_to_edge_idx = torch.cat(pred_edge_nodes_to_edge_idx, dim=-1)
        data.pred_edge_attr = torch.cat(pred_edge_attr, dim=0)
        data.split_pred_nodes_labels = data.y.clone().view(1, -1)
        # print("data.y.size() = ", data.y.size())
        data.split_pred_edges_labels = torch.cat(split_pred_edge_labels, dim=0)

        data.prev_nodes_attr = xx[0]
        data.prev_edge_index = split_edge_index[0]
        data.prev_edge_attr = split_edge_attr[0]

        sim_x, sim_edge_index, sim_edge_attr = self.flow.generate_one_mole_given_original_mole(data, data.y.clone().view(1, -1))
        data.x = xx[1]
        data.edge_index = current_connected_edges.clone()
        data.edge_attr = current_edge_attr.clone()

        data.sim_x = sim_x
        data.sim_edge_index = sim_edge_index
        data.sim_edge_attr = sim_edge_attr

        #### global contrast over; and the below aims to extract the first similar nodes to perform the local similar

        # root_idx = random.sample(range(num_atom), 1)[0]
        #
        # G = graph_data_obj_to_nx_simple(data)
        #
        # node_type = int(data.x[root_idx, 0])
        #
        # first_approx_node_idxes = list(nx.single_source_shortest_path_length(G, root_idx, 1).keys())
        #
        # data.first_approx_node_idxes = torch.tensor(first_approx_node_idxes, dtype=torch.long)
        #
        # data.node_type = torch.tensor([node_type], dtype=torch.long)
        #
        # # Get k-hop subgraph rooted at specified atom idx
        # substruct_node_idxes = nx.single_source_shortest_path_length(G,
        #                                                              root_idx,
        #                                                              self.k).keys()
        # if len(substruct_node_idxes) > 0:
        #     substruct_G = G.subgraph(substruct_node_idxes)
        #     substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
        #     # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
        #     # make sense, since the node indices in data obj must start at 0
        #
        #     for i in range(data.first_approx_node_idxes.size(0)):
        #         a = int(data.first_approx_node_idxes[i])
        #         data.first_approx_node_idxes[i] = substruct_node_map[a]
        #
        #     # for i in range(traces.size(0)):
        #     #     a = int(traces[i])
        #     #     if a not in substruct_node_map:
        #     #         continue
        #     #     rwr_sampled_nodes.append(substruct_node_map[a])
        #
        #     substruct_data = nx_to_graph_data_obj_simple(substruct_G)
        #     data.x_substruct = substruct_data.x
        #     data.edge_attr_substruct = substruct_data.edge_attr
        #     data.edge_index_substruct = substruct_data.edge_index
        #     data.center_substruct_idx = torch.tensor([substruct_node_map[
        #                                                   root_idx]])  # need
        #     # to convert center idx from original graph node ordering to the
        #     # new substruct node ordering
        #
        # # data.first_approx_node_idxes = torch.tensor(rwr_sampled_nodes, dtype=torch.long)
        #
        # # Get subgraphs that is between l1 and l2 hops away from the root node

        return data

    def __repr__(self):
        return '{}'.format(
            self.__class__.__name__)


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, introduce_random=False, mask_edge=True, mask_stra="random"):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.introduce_random = introduce_random
        self.mask_stra = mask_stra

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            if self.mask_stra == "random":
                masked_atom_indices = random.sample(range(num_atoms), sample_size)
            elif self.mask_stra == "substruct":
                seed = random.sample(range(num_atoms), 1)[0]
                G = graph_data_obj_to_nx_simple(data)
                node_to_dis = nx.single_source_shortest_path_length(seed, G)
                ordered_nodes = sorted(node_to_dis.items(), key=lambda i: i[1])
                masked_atom_indices = [ordered_nodes[i][0] for i in range(sample_size)]
            elif self.mask_stra == "hop":
                seeds = random.sample(range(num_atoms), sample_size)


        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            if not self.introduce_random:
                data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
            else:
                idx = random.sample(range(10), 1)[0]
                if idx < 8:
                    data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
                elif idx == 8:
                    random_atom_idx = random.sample(range(self.num_atom_type), 1)[0]
                    data.x[atom_idx] = torch.tensor([random_atom_idx, 0])


        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

# class structural_properties_extract:
#     def __init__(self, num, property_type="pair_wise_shortest_path"):
#         self.property_type = property_type
#         self.num = num
#
#     def __call__(self, data, mask_type=None):
#         G = graph_data_obj_to_nx_simple(data)
#         spl = nx.all_pairs_shortest_path_length(G)
#         chosen_node_indices = list()
#         num_atom = data.x.size(0)
#         labels = list()
#
#         if self.property_type == "pair_wise_shortest_path":  # abandoned...
#             for i in range(self.num):
#                 pair = np.random.choice(range(num_atom), size=2)
#                 pair = torch.from_numpy(pair)
#                 chosen_node_indices.append(pair.view(1, -1))
#
#         elif self.property_type == "centrality":
#
#
#         else:
#             raise NotImplementedError("aaaa")

# from mol2features import atom_to_vocab, bond_to_vocab, atom_features
class MaskAtomGetFea:
    def __init__(self, mol_vocab, DEBUG=False):
        self.mol_vocab = mol_vocab
        self.DEBUG = DEBUG

        # mol = Chem.MolFromSmiles(smiles)
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __call__(self, data, smile=None, masked_atom_indices=None):
        # mol = graph_data_obj_to_mol_simple(data.x, data.edge_index, data.edge_attr)
        if smile is not None:
            # if self.DEBUG:
            #     print("original data obj.x: ")
            #     print(data.x)
            mol = Chem.MolFromSmiles(smile)
            self.hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
            self.hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
            self.acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
            self.basic_match = sum(mol.GetSubstructMatches(self.basic), ())
            self.ring_info = mol.GetRingInfo()
            tmp_y = data.y
            data = mol_to_graph_data_obj_simple(mol)
            data.y = tmp_y
            # if self.DEBUG:
            #     for i in range(data.x.size(0)):
            #         atom = mol.GetAtomWithIdx(i)
            #         print(atom.GetAtomicNum(), int(data.x[i, 0]))
                # print("from mol data obj.x: ")
                # print(data.x)
        else:
            mol = graph_data_obj_to_mol_simple(data.x, data.edge_index, data.edge_attr)
        data.atom_fea_idx = list()
        num_atoms = data.x.size()[0]
        data.fea_x = []
        for j in range(num_atoms):
            atom = mol.GetAtomWithIdx(j)
            fea_idx = self.mol_vocab.stoi.get(atom_to_vocab(mol, atom), self.mol_vocab.other_index)
            data.atom_fea_idx.append(fea_idx)
            data.fea_x.append(atom_features(self, atom))
        data.atom_fea_idx = torch.LongTensor(data.atom_fea_idx)
        data.fea_x = torch.FloatTensor(data.fea_x)
        # if self.DEBUG:
            # print("atom_fea_idx = ", data.atom_fea_idx, torch.max(data.atom_fea_idx), torch.min(data.atom_fea_idx))
            # print("data.feature_x = ", data.fea_x)

        return data

    def __repr__(self):
        return 'transformerWithMolFeatures'

class MaskAtomKeptOrigin:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, introduce_random=False, mask_edge=True, mask_stra="random"):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.introduce_random = introduce_random
        self.mask_stra = mask_stra

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            if self.mask_stra == "random":
                masked_atom_indices = random.sample(range(num_atoms), sample_size)
            elif self.mask_stra == "substruct":
                seed = random.sample(range(num_atoms), 1)[0]
                G = graph_data_obj_to_nx_simple(data)
                node_to_dis = nx.single_source_shortest_path_length(seed, G)
                ordered_nodes = sorted(node_to_dis.items(), key=lambda i: i[1])
                masked_atom_indices = [ordered_nodes[i][0] for i in range(sample_size)]
            elif self.mask_stra == "hop":
                seeds = random.sample(range(num_atoms), sample_size)


        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node

        data.masked_x = data.x.clone()
        for atom_idx in masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        # print(data.masked_x)
        # print(len(masked_atom_indices))

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)




def GSIteration(L, init_vecs, k=5):

    D = L.diagonal()
    D_inv = 1. / D
    low_diag = np.tril(L, -1)
    high_diag = np.triu(L, 1)
    for kk in range(k):
        # init_vecs = np.linalg.solve(-np.dot(high_diag, init_vecs), D - low_diag)
        init_vecs = np.linalg.solve(D + low_diag, -np.dot(high_diag, init_vecs))
    return init_vecs


import sklearn
from sklearn import cluster
import time
from sklearn import decomposition

class MaskWholeAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_edge=True, fusion_rate=None):
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_edge = mask_edge
        self.fusion_rate = fusion_rate if fusion_rate is not None else 0.7

    def __call__(self, data, masked_atom_indices=None):
        print("getting data.")
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        num_atoms = data.x.size(0)

        G = graph_data_obj_to_nx_simple(data)
        adj = nx.to_numpy_matrix(G)
        # print(adj.shape)
        D = np.sum(adj, axis=1)
        if len(D.shape) == 1:
            D = D.reshape((-1, 1))
        # print("D.shape = ", D.shape)
        L = D - adj
        aft_nodes = int(num_atoms * self.fusion_rate + 1)

        low_freq_reps = list()

        st_time = time.time()
        low_freq_reps = decomposition.TruncatedSVD(n_components=aft_nodes).fit_transform(L)
        print("SVD time = ", time.time() - st_time)
        # print("low_freq_reps.shape = ", low_freq_reps.shape)
        #
        # for i in range(aft_nodes):
        #     init_vec = np.random.rand(num_atoms) #.reshape([-1, 1])
        #     low_freq = GSIteration(L, init_vec)
        #     low_freq_reps.append(low_freq.reshape([-1, 1]))

        # low_freq_reps = torch.cat(low_freq_reps, dim=0)
        # low_freq_reps = np.concatenate(low_freq_reps, axis=1)

        st_time = time.time()
        KMeans_model = cluster.KMeans(n_clusters=aft_nodes, max_iter=10)
        labels = KMeans_model.fit_predict(low_freq_reps)
        print("K-Means time = ", time.time() - st_time)

        # print("KMeans finished.")

        # init_vecs = np.random.rand([num_atoms, aft_nodes])
        # low_freq_reps = GSIteration(L, init_vecs)

        data.coarsed_x = list() # all no type
        data.coarsed_edge_index = list()
        data.coarsed_edge_attr = list()
        mapping = dict()
        # print(data.x.size(0), labels.shape)
        for i in range(labels.shape[0]):
            lab = int(labels[i])
            mapping[i] = lab
        for i in range(aft_nodes):
            data.coarsed_x.append(torch.tensor([self.num_atom_type, 0], dtype=torch.long).view(1, -1))
        exit_edges = dict()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if (a, b) in exit_edges or (b, a) in exit_edges:
                continue
            else:
                a, b = mapping[a], mapping[b]
                data.coarsed_edge_index.append(torch.tensor([a, b], dtype=torch.long).view(-1, 1))
                data.coarsed_edge_index.append(torch.tensor([b, a], dtype=torch.long).view(-1, 1))
                data.coarsed_edge_attr.append(torch.tensor([self.num_edge_type, 0], dtype=torch.long).view(1, -1))
                data.coarsed_edge_attr.append(torch.tensor([self.num_edge_type, 0], dtype=torch.long).view(1, -1))
                exit_edges[(a, b)] = 1
                exit_edges[(b, a)] = 1
        data.coarsed_x = torch.cat(data.coarsed_x, dim=0)
        data.coarsed_edge_attr = torch.cat(data.coarsed_edge_attr, dim=0)
        data.coarsed_edge_index = torch.cat(data.coarsed_edge_index, dim=1)


        for atom_idx in range(num_atoms):
            mask_node_labels_list.append(torch.tensor([self.num_atom_type, 0], dtype=torch.long).view(1, -1))
        data.x_masked = torch.cat(mask_node_labels_list, dim=0)

        if self.mask_edge:
            data.edge_attr_masked = []
            for bond_idx in range(data.edge_index.size(1)):
                data.edge_attr_masked.append(torch.tensor(
                    [self.num_edge_type, 0]
                ).view(1, -1))
            data.edge_attr_masked = torch.cat(data.edge_attr_masked, dim=0)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class MultiTimeMask:
    def __init__(self, mask_edge=True, num_mask_version=5):
        # self.num_atom_type = num_atom_type
        # self.num_edge_type = num_edge_type
        self.mask_edge = mask_edge
        self.mask_rate_int_min = 13
        self.mask_rate_int_max = 20
        self.num_mask_version = num_mask_version
        # self.fusion_rate = fusion_rate if fusion_rate is not None else 0.7

    def __call__(self, data, masked_atom_indices=None):
        # print("getting data.")
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        num_atoms = data.x.size(0)

        mask_nodes_idx_list = []
        for j in range(self.num_mask_version):
            mask_rate = random.sample(range(self.mask_rate_int_min, self.mask_rate_int_max), 1)[0]
            mask_rate = (mask_rate + 0.0) / 100
            num_mask_nodes = int(mask_rate * num_atoms + 1)
            mask_nodes_idx = random.sample(range(num_atoms), num_mask_nodes)
            mask_nodes_idx_list.append(torch.tensor(mask_nodes_idx, dtype=torch.long))
        data.mask_nodes_idx_list = mask_nodes_idx_list

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class gen_fake_mask_idxes:
    def __init__(self, mask_rate):
        self.mask_rate = mask_rate
        # self.num_atom_type = num_atom_type
        # self.mask_edge = mask_edge
        # self.num_edge_type = num_edge_type

    def __call__(self, data, mask_type=None):
        num_atoms = data.x.size(0)
        masked_num_atoms = int(self.mask_rate * num_atoms + 1)
        masked_indices_a = np.random.choice(range(num_atoms), masked_num_atoms)
        masked_indices_a = torch.from_numpy(masked_indices_a)

        data.masked_nodes = masked_indices_a
        return data

# todo: only random mask can reveal the inner charactersictics of the moleculer graph?
class contrast_mask_atom_pair:
    def __init__(self, mask_rate, num_atom_type, num_edge_type, mask_edge=False):
        self.mask_rate = mask_rate
        self.num_atom_type = num_atom_type
        self.mask_edge = mask_edge
        self.num_edge_type = num_edge_type

    def __call__(self, data, mask_type=None):
        num_atoms = data.x.size(0)
        masked_num_atoms = int(self.mask_rate * num_atoms + 1)
        masked_indices_a = np.random.choice(range(num_atoms), masked_num_atoms)
        masked_indices_b = np.random.choice(range(num_atoms), masked_num_atoms)
        masked_indices_a = torch.from_numpy(masked_indices_a)
        masked_indices_b = torch.from_numpy(masked_indices_b)
        data.x_a = torch.zeros_like(data.x)
        data.x_a[:, :] = data.x[:, :]
        data.x_a[masked_indices_a, :] = torch.tensor([self.num_atom_type, 0], dtype=torch.long)
        data.x_b = torch.zeros_like(data.x)
        data.x_b[:, :] = data.x[:, :]
        data.x_b[masked_indices_b, :] = torch.tensor([self.num_atom_type, 0], dtype=torch.long)
        data.masked_label_a = data.x[masked_indices_a, 0]
        data.masked_label_b = data.x[masked_indices_b, 0]
        data.mask_indices_a = masked_indices_a
        data.mask_indices_b = masked_indices_b

        if self.mask_edge == True:
            connected_edge_indices_a = []
            connected_edge_indices_b = []
            masked_nodes_dict_a = {node: i for i, node in enumerate(masked_indices_a)}
            masked_nodes_dict_b = {node: i for i, node in enumerate(masked_indices_b)}
            for i in range(data.edge_index.size(1)):
                a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
                if a in masked_indices_a or b in masked_nodes_dict_a:
                    connected_edge_indices_a.append(i)
            for i in range(data.edge_index.size(1)):
                a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
                if a in masked_indices_b or b in masked_nodes_dict_b:
                    connected_edge_indices_b.append(i)
            data.connected_edge_indices_a = torch.tensor(connected_edge_indices_a, dtype=torch.long)
            data.connected_edge_indices_b = torch.tensor(connected_edge_indices_b, dtype=torch.long)
            data.edge_labels_a = data.edge_attr[data.connected_edge_indices_a, 0]
            data.edge_labels_b = data.edge_attr[data.connected_edge_indices_b, 0]
            data.edge_attr_a = torch.zeros_like(data.edge_attr)
            data.edge_attr_a[:, :] = data.edge_attr[:, :]
            data.edge_attr_b = torch.zeros_like(data.edge_attr)
            data.edge_attr_b[:, :] = data.edge_attr[:, :]
            data.edge_attr_a[data.connected_edge_indices_a, :] = torch.tensor([self.num_edge_type, 0], dtype=torch.long)
            data.edge_attr_b[data.connected_edge_indices_b, :] = torch.tensor([self.num_edge_type, 0], dtype=torch.long)
        else:
            data.edge_attr_a = torch.zeros_like(data.edge_attr)
            data.edge_attr_a[:, :] = data.edge_attr[:, :]
            data.edge_attr_b = torch.zeros_like(data.edge_attr)
            data.edge_attr_b[:, :] = data.edge_attr[:, :]
        return data

class DropNodeT:
    def __init__(self, mask_rate, num_atom_type, num_edge_type, mask_edge=False):
        self.mask_rate = mask_rate
        self.num_atom_type = num_atom_type
        self.mask_edge = mask_edge
        self.num_edge_type = num_edge_type

    def __call__(self, data, mask_type=None):

        num_atoms = data.x.size(0)
        masked_num_atoms = int(self.mask_rate * num_atoms + 1)
        masked_indices_a = np.random.choice(range(num_atoms), masked_num_atoms)

        old_idx_to_new_idx = dict()

        print("num_atoms = ", num_atoms)
        print("mased_indices_a.shape = ", masked_indices_a.shape)
        masked_indices = dict()
        for i in range(masked_indices_a.shape[0]):
            a = int(masked_indices_a[i])
            masked_indices[a] = 1
        print("masked_indices.size() = ", len(masked_indices))
        new_node_num = num_atoms - masked_indices_a.shape[0]
        x_list = list()

        cnt = 0
        for j in range(num_atoms):
            if j not in masked_indices:
                x_list.append(data.x[j, :].view(1, -1))
                old_idx_to_new_idx[j] = cnt
                cnt += 1
        x_list = torch.cat(x_list, dim=0)
        data.x = x_list
        print("data.x.size() = ", data.x.size())

        edge_list = list()
        edge_attr_list = list()
        for i in range(data.edge_index.size(1)):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if (a not in masked_indices) and (b not in masked_indices):
                a = old_idx_to_new_idx[a]
                b = old_idx_to_new_idx[b]
                data.edge_index[0, i] = a
                data.edge_index[1, i] = b
                edge_list.append(data.edge_index[:, i].view(-1, 1))
                edge_attr_list.append(data.edge_attr[i, :].view(1, -1))
        edge_list = torch.cat(edge_list, dim=1)
        edge_attr_list = torch.cat(edge_attr_list, dim=0)
        data.edge_index = edge_list
        data.edge_attr = edge_attr_list
        return data


from utils2 import  get_tree_edges
class DropEdgePair:
    def __init__(self, num_atom_type, num_edge_type, drop_edge_rate=0.15, introduce_random=False, mask_edge=True, mask_stra="random"):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        # self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.introduce_random = introduce_random
        self.mask_stra = mask_stra
        self.drop_edge_rate = drop_edge_rate

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        te, nte = get_tree_edges(data)
        if len(nte) == 0:
            data.edge_droped_index = data.edge_index
            data.edge_droped_attr = data.edge_attr
        else:
            num_droped_edges = min(int(data.edge_index.size(1) * self.drop_edge_rate + 1), len(nte))
            droped_edge_index_half = np.random.choice(list(nte), num_droped_edges)
            dict_droped_edge_index_half = {i: 1 for i in droped_edge_index_half}
            edge_droped_index = []
            edge_droped_attr = []
            for i in range(0, data.edge_index.size(1), 2):
                half_index = i // 2
                if half_index not in dict_droped_edge_index_half:
                    edge_droped_index.append(data.edge_index[:, i].view(-1, 1))
                    edge_droped_index.append(data.edge_index[:, i + 1].view(-1, 1))
                    edge_droped_attr.append(data.edge_attr[i, :].view(1, -1))
                    edge_droped_attr.append(data.edge_attr[i + 1, :].view(1, -1))
            data.edge_droped_index = torch.cat(edge_droped_index, dim=1)
            data.edge_droped_attr = torch.cat(edge_droped_attr, dim=0)
        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class MaskAtom_hop:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, introduce_random=False, mask_edge=True,
                 mask_stra="random", hop=1):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.introduce_random = introduce_random
        self.mask_stra = mask_stra
        self.hop = hop

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """


            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
        num_atoms = data.x.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        context_atoms = set()
        G = graph_data_obj_to_nx_simple(data)

        for atom in masked_atom_indices:
            context = nx.single_source_shortest_path_length(G, atom, self.hop)
            context_atoms |= set(context.keys())
        context_masked_atom = context_atoms - set(masked_atom_indices)
        central_atoms_not_masked = set(masked_atom_indices) - context_atoms
        assert len(context_masked_atom & central_atoms_not_masked) == 0


        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        to_be_pred_labels = []
        masked_atom_indices = []
        for atom_idx in context_masked_atom:
            to_be_pred_labels.append(data.x[atom_idx].view(1, -1))
            masked_atom_indices.append(atom_idx)
        for atom_idx in central_atoms_not_masked:
            to_be_pred_labels.append(data.x[atom_idx].view(1, -1))
            masked_atom_indices.append(atom_idx)
        # masked_atom_indices =
        data.mask_node_label = torch.cat(to_be_pred_labels, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in context_masked_atom:
            if not self.introduce_random:
                data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
            else:
                idx = random.sample(range(10), 1)[0]
                if idx < 8:
                    data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
                elif idx == 8:
                    random_atom_idx = random.sample(range(self.num_atom_type), 1)[0]
                    data.x[atom_idx] = torch.tensor([random_atom_idx, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in context_masked_atom:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class Extract_Substruct_Contrast:
    def __init__(self, num_atom_type, num_edge_type, mask_edge=True, hop=1):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        # self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        # self.introduce_random = introduce_random
        # self.mask_stra = mask_stra
        self.hop = hop

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """


            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
        num_atoms = data.x.size()[0]
        seed = random.sample(range(num_atoms), 1)[0]
        G = graph_data_obj_to_nx_simple(data)
        nodes_to_dis = nx.single_source_shortest_path_length(G, seed)
        ordered_nodes_pair = sorted(nodes_to_dis.items(), key=lambda i: i[1])
        seed = ordered_nodes_pair[-1][0]
        nodes_to_dis = nx.single_source_shortest_path_length(G, seed)
        ordered_nodes_pair = sorted(nodes_to_dis.items(), key=lambda i: i[1])
        ordered_nodes = [par[0] for i, par in enumerate(ordered_nodes_pair)]
        num_2 = num_atoms // 2
        part_a_nodes_indices = ordered_nodes[: num_2]
        part_b_nodes_indices = ordered_nodes[num_2: ]
        data.part_a_nodes_indices = torch.tensor(part_a_nodes_indices, dtype=torch.long)
        data.part_b_nodes_indices = torch.tensor(part_b_nodes_indices, dtype=torch.long)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class SubstructMask(MaskAtom):
    def __init__(self, k, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        super(SubstructMask, self).__init__(num_atom_type=num_atom_type,
                                            num_edge_type=num_edge_type, mask_rate=mask_rate, mask_edge=mask_edge)
        self.k = k

    def __call__(self, data, root_idx=None, mask_type=0): # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
        G = graph_data_obj_to_nx_simple(data)
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()

        data.node_idx_tensor = torch.tensor(range(num_atoms))
        data.num_atoms_size = torch.tensor([num_atoms], dtype=torch.long)
        masked_atom_indices = list(substruct_node_idxes)
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        data.mask_x = data.x
        data.mask_edge_attr = data.edge_attr
        # method that is more robust?
        if mask_type == 0:  # mask node
            for atom_idx in masked_atom_indices:
                data.mask_x[atom_idx] = torch.tensor([self.num_atom_type, 0])
        else:  # mask edge
            # if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.mask_edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

from utils2 import sampled_subgraph_gcc, get_subgraph_data



class PairwiseNodeGCC(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False,
                 step_dist=[1.0, 0.0, 0.0],
                 length=64,
                 rsprob=0.8):
        super(PairwiseNodeGCC, self).__init__(num_atom_type=num_atom_type,
                                            num_edge_type=num_edge_type, mask_rate=mask_rate, mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.step_dist = step_dist
        self.length = length
        self.rsprob = rsprob

    def __call__(self, data, masked_atom_indices=None, mask_type=0): # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        num_types = 2

        node_a, node_b = sampled_subgraph_gcc(data, step_dist=self.step_dist, length=self.length, rsprob=self.rsprob)
        # old_to_new_idx_a = {node_idx: i for i, node_idx in enumerate(node_a)}
        # old_to_new_idx_b = {node_idx: i for i, node_idx in enumerate(node_b)}
        x_a, ei_a, ea_a = get_subgraph_data(data, node_a)
        x_b, ei_b, ea_b = get_subgraph_data(data, node_b)
        data.x_a = x_a
        data.edge_index_a = ei_a
        data.edge_attr_a = ea_a

        data.x_b = x_b
        data.edge_index_b = ei_b
        data.edge_attr_b = ea_b
        # print(len(node_a), len(node_b))
        return data


class PairwiseNodeEdgeMask(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False):
        super(PairwiseNodeEdgeMask, self).__init__(num_atom_type=num_atom_type,
                                            num_edge_type=num_edge_type, mask_rate=mask_rate, mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave

    def __call__(self, data, masked_atom_indices=None, mask_type=0): # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        num_types = 2
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)


        data.xa = data.x
        data.edgea_attr = data.edge_attr
        data.edgeb_attr = data.edge_attr
        data.xb = data.x
        if self.add_wave == True:
            dgl_graph = dgl.DGLGraph(graph_data=graph_data_obj_to_nx_simple(data))
            data.wave_emb = _add_undirected_graph_positional_embedding(dgl_graph, hidden_size=30)
            # data.wave_emb = add_wavelet_emb(data)
        # mask_types = random.sample([0, 0, 1, 1, 2], num_types)

        # G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj
        #
        # # Get k-hop subgraph rooted at specified atom idx
        # substruct_node_idxes = nx.single_source_shortest_path_length(G,
        #                                                              root_idx,
        #                                                              self.k).keys()
        num_atoms = data.x.size()[0]
        G = graph_data_obj_to_nx_simple(data)
        sample_size = int(num_atoms * self.mask_rate + 1)
        sampled_nodes = list()
        dis = self.dis
        if self.keep_dis == True:
            remained_atoms = set(range(num_atoms))
            while len(sampled_nodes) < sample_size and len(remained_atoms) > 0:
                now_nodes = random.sample(remained_atoms, 1)[0]
                sampled_nodes.append(now_nodes)
                remained_atoms.discard(now_nodes)
                nearest_k_nodes = nx.single_source_shortest_path_length(G, now_nodes, dis)
                for near_node in nearest_k_nodes:
                    remained_atoms.discard(near_node)
            masked_atom_indices = sampled_nodes
            # print(len(sampled_nodes), sample_size)

        # mask_nodes_all = None
        for j in range(num_types):
            # mask_type = mask_types[j]
            # mask_type = random.sample(range(num_types + 1), 1)[0]
            mask_type = random.sample([0, 0, 1, 1], 1)[0]
            if mask_type == 0:
                if masked_atom_indices == None:
                    print("here get no dis nodes")
                    # sample x distinct atoms to be masked, based on mask rate. But
                    # will sample at least 1 atom
                    num_atoms = data.x.size()[0]
                    sample_size = int(num_atoms * self.mask_rate + 1)
                    masked_atom_indices = random.sample(range(num_atoms), sample_size)
                    # mask_nodes_all = masked_atom_indices.copy()
                # if mask_nodes_all == None:
                #     if masked_atom_indices == None:
                #         # sample x distinct atoms to be masked, based on mask rate. But
                #         # will sample at least 1 atom
                #         num_atoms = data.x.size()[0]
                #         sample_size = int(num_atoms * self.mask_rate + 1)
                #         masked_atom_indices = random.sample(range(num_atoms), sample_size)
                #         mask_nodes_all = masked_atom_indices.copy()
                # else:
                #     masked_atom_indices = mask_nodes_all.copy()

                # create mask node label by copying atom feature of mask atom

                mask_node_labels_list = []
                for atom_idx in masked_atom_indices:
                    mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
                # data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
                # data.masked_atom_indices = torch.tensor(masked_atom_indices)
                for atom_idx in masked_atom_indices:
                    if num_types == 0:
                        data.xa[atom_idx] = torch.tensor([self.num_atom_type, 0])
                    else:
                        data.xb[atom_idx] = torch.tensor([self.num_atom_type, 0])
                if num_types == 0:
                    data.masked_atom_indices_a = torch.tensor(masked_atom_indices, dtype=torch.long)
                else:
                    data.masked_atom_indices_b = torch.tensor(masked_atom_indices, dtype=torch.long)
            elif mask_type == 1:
                if masked_atom_indices == None:
                    # sample x distinct atoms to be masked, based on mask rate. But
                    # will sample at least 1 atom
                    num_atoms = data.x.size()[0]
                    sample_size = int(num_atoms * self.mask_rate + 1)
                    masked_atom_indices = random.sample(range(num_atoms), sample_size)
                    # mask_nodes_all = masked_atom_indices.copy()
                # if mask_nodes_all == None:
                #     if masked_atom_indices == None:
                #         # sample x distinct atoms to be masked, based on mask rate. But
                #         # will sample at least 1 atom
                #         num_atoms = data.x.size()[0]
                #         sample_size = int(num_atoms * self.mask_rate + 1)
                #         masked_atom_indices = random.sample(range(num_atoms), sample_size)
                #         mask_nodes_all = masked_atom_indices.copy()
                # else:
                #     masked_atom_indices = mask_nodes_all.copy()

                if num_types == 0:
                    data.masked_atom_indices_a = torch.tensor(masked_atom_indices, dtype=torch.long)
                else:
                    data.masked_atom_indices_b = torch.tensor(masked_atom_indices, dtype=torch.long)

                connected_edge_indices = []
                for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                    for atom_idx in masked_atom_indices:
                        if atom_idx in set((u, v)) and \
                                bond_idx not in connected_edge_indices:
                            connected_edge_indices.append(bond_idx)

                if len(connected_edge_indices) > 0:
                    # create mask edge labels by copying bond features of the bonds connected to
                    # the mask atoms
                    mask_edge_labels_list = []
                    for bond_idx in connected_edge_indices[::2]:  # because the
                        # edge ordering is such that two directions of a single
                        # edge occur in pairs, so to get the unique undirected
                        # edge indices, we take every 2nd edge index from list
                        mask_edge_labels_list.append(
                            data.edge_attr[bond_idx].view(1, -1))

                    # data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                    # modify the original bond features of the bonds connected to the mask atoms
                    for bond_idx in connected_edge_indices:
                        if num_types == 0:
                            data.edgea_attr[bond_idx] = torch.tensor(
                                [self.num_edge_type, 0])
                        else:
                            data.edgeb_attr[bond_idx] = torch.tensor(
                                [self.num_edge_type, 0])

        """
        if mask_type == 0:
            if masked_atom_indices == None:
                # sample x distinct atoms to be masked, based on mask rate. But
                # will sample at least 1 atom
                num_atoms = data.x.size()[0]
                sample_size = int(num_atoms * self.mask_rate + 1)
                masked_atom_indices = random.sample(range(num_atoms), sample_size)

            # create mask node label by copying atom feature of mask atom
            mask_node_labels_list = []
            for atom_idx in masked_atom_indices:
                mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
            # data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
            # data.masked_atom_indices = torch.tensor(masked_atom_indices)

            data.xa = data.x
            data.edgea_attr = data.edge_attr
            data.edgeb_attr = data.edge_attr
            data.xb = data.x

            # modify the original node feature of the masked node
            for atom_idx in masked_atom_indices:
                data.xa[atom_idx] = torch.tensor([self.num_atom_type, 0])

            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                #data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edgeb_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])
        else:
            data.xa = data.x
            data.edgea_attr = data.edge_attr
            data.edgeb_attr = data.edge_attr
            data.xb = data.x
            num_atoms = data.x.size()[0]
            if mask_type == 1:
                for atom_idx in range(num_atoms):
                    data.xa[atom_idx] = torch.tensor([self.num_atom_type, 0])
            else:
                for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                    data.edgea_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])
        """
        return data

from utils2 import sample_subgraph, filter_attri, sample_subgraph_only_node

class ContrastMask1(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_node=False, use_filter=False, mask_edge=True, dis=3, keep_dis=True, add_wave=False):
        super(ContrastMask1, self).__init__(num_atom_type=num_atom_type,
                                                   num_edge_type=num_edge_type, mask_rate=mask_rate,
                                                   mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.mask_node = mask_node
        self.use_filter = use_filter

    def __call__(self, data, masked_atom_indices=None, mask_type=0):  # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        num_types = 2
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)

        # ops = np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=2)
        # ops = np.random.choice(np.arange(0, 3), p=[0.2, 0.4, 0.4], size=2)
        if self.use_filter == False:
            ops = [np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=1)[0]] + [np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=1)[0]]
        else:
            ops = np.random.choice(np.arange(0, 3), p=[0.2, 0.4, 0.4], size=2)

        for j in range(2):
            op = ops[j]
            if op == 0:
                if j == 0:
                    data.x_a = data.x
                    data.edge_attr_a = data.edge_attr
                    data.edge_index_a = data.edge_index
                    data.filter_k_a = 0
                else:
                    data.x_b = data.x
                    data.edge_attr_b = data.edge_attr
                    data.edge_index_b = data.edge_index
                    data.filter_k_b = 0
                # data.filter_k = 0
            elif op == 1:
                ratio = float(np.random.choice(np.arange(3, 9), size=1)[0]) / 10.0
                if j == 0:
                    data.x_a, data.edge_index_a, data.edge_attr_a = sample_subgraph(data, ratio)
                    data.x_a = data.x_a.long()
                    data.filter_k_a = 0
                else:
                    data.x_b, data.edge_index_b, data.edge_attr_b = sample_subgraph(data, ratio)
                    data.x_b = data.x_b.long()
                    data.filter_k_b = 0
                # data.filter_k = 0
            else:
                k = np.random.choice(np.arange(2, 4), size=1)[0]
                if j == 0:
                    data.x_a = data.x
                    data.edge_attr_a = data.edge_attr
                    data.edge_index_a = data.edge_index
                    data.filter_k_a = k
                else:
                    data.x_b = data.x
                    data.edge_attr_b = data.edge_attr
                    data.edge_index_b = data.edge_index
                    data.filter_k_b = k
                # data.filter_k = k
                # if j == 0:
                #     data.x_a, data.edge_index_a, data.edge_attr_a = filter_attri(data, k), data.edge_index, data.edge_attr
                # else:
                #     data.x_b, data.edge_index_b, data.edge_attr_b = filter_attri(data, k), data.edge_index, data.edge_attr
        # print(data.x_a.size(), ops[1])

        if self.mask_node == True:
            if masked_atom_indices == None:
                # sample x distinct atoms to be masked, based on mask rate. But
                # will sample at least 1 atom
                num_atoms = data.x_a.size()[0]
                sample_size = int(num_atoms * self.mask_rate + 1)
                masked_atom_indices = random.sample(range(num_atoms), sample_size)

            # create mask node label by copying atom feature of mask atom
            mask_node_labels_list = []
            for atom_idx in masked_atom_indices:
                mask_node_labels_list.append(data.x_a[atom_idx].view(1, -1))
            data.mask_node_label_a = torch.cat(mask_node_labels_list, dim=0)
            data.masked_atom_indices_a = torch.tensor(masked_atom_indices)

            # modify the original node feature of the masked node
            for atom_idx in masked_atom_indices:
                data.x_a[atom_idx] = torch.tensor([self.num_atom_type, 0])

            # if masked_atom_indices == None:
                # sample x distinct atoms to be masked, based on mask rate. But
                # will sample at least 1 atom
            num_atoms = data.x_b.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

            # create mask node label by copying atom feature of mask atom
            mask_node_labels_list = []
            for atom_idx in masked_atom_indices:
                mask_node_labels_list.append(data.x_b[atom_idx].view(1, -1))
            data.mask_node_label_b = torch.cat(mask_node_labels_list, dim=0)
            data.masked_atom_indices_b = torch.tensor(masked_atom_indices)

            # modify the original node feature of the masked node
            for atom_idx in masked_atom_indices:
                data.x_b[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if data.x_a.size(1) != 2:
            print(data.x_a.size(), ops[1])
        if data.x_b.size(1) != 2:
            print(data.x_b.size(), ops[1])
        return data


class ContrastMask2(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False, k=5):
        super(ContrastMask2, self).__init__(num_atom_type=num_atom_type,
                                                   num_edge_type=num_edge_type, mask_rate=mask_rate,
                                                   mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.k = k

    def __call__(self, data, masked_atom_indices=None, k=5, mask_type=0):  # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        num_types = 2
        k = self.k
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)

        # ops = np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=2)
        n_nodes = data.x.size(0)
        seeds = np.random.choice(np.arange(n_nodes), size=k)
        num_sub_nodes = list()
        s_nodes = list()
        for i, seed in enumerate(seeds):
            ratio = float(np.random.choice(np.arange(2, 7), size=1)[0]) / 10.0
            sample_nodes = sample_subgraph_only_node(data, ratio, seed=seed)
            s_nodes = s_nodes + sample_nodes
            num_sub_nodes.append(len(sample_nodes))
        data.s_nodes = torch.LongTensor(s_nodes)
        data.num_sub_nodes = torch.LongTensor(num_sub_nodes)
        data.k = k

        return data

class ContrastMask4(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False, k=5):
        super(ContrastMask4, self).__init__(num_atom_type=num_atom_type,
                                                   num_edge_type=num_edge_type, mask_rate=mask_rate,
                                                   mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.k = k

    def __call__(self, data, masked_atom_indices=None, k=5, mask_type=0):  # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        num_types = 2
        k = self.k
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)

        # ops = np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=2)
        n_nodes = data.x.size(0)
        seed = np.random.choice(np.arange(n_nodes), size=1)[0]
        sample_nodes = sample_subgraph_only_node(data, 0.5, seed=seed)
        all_nodes = [i for i in range(n_nodes)]
        b_nodes = list(set(all_nodes) - set(sample_nodes))
        data.a_nodes = torch.LongTensor(sample_nodes)
        data.b_nodes = torch.LongTensor(b_nodes)

        return data


from utils2 import node_context_extract_with_step
class ContrastMask5(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False, k=5, step=1):
        super(ContrastMask5, self).__init__(num_atom_type=num_atom_type,
                                                   num_edge_type=num_edge_type, mask_rate=mask_rate,
                                                   mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.k = k
        self.step = step

    def __call__(self, data, masked_atom_indices=None, k=5, mask_type=0):  # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        # num_types = 2
        # k = self.k
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)
        num_nodes = data.x.size(0)
        seeds = np.random.choice(np.arange(0, num_nodes), size=self.k)

        nodes = []
        context_nodes = []
        context_nodes_to_graph_idx = []
        for i, seed in enumerate(seeds):
            context = node_context_extract_with_step(data, seed, self.step)
            context = list(set(context) - {seed})
            if len(context) == 0:
                tmp_step = self.step - 1
                while len(context) == 0 and tmp_step >= 1:
                    context = node_context_extract_with_step(data, seed, tmp_step)
                    context = list(set(context) - {seed})
                    tmp_step -= 1
            context_nodes += context
            context_to_idx = torch.full((len(context),), i)
            context_nodes_to_graph_idx.append(context_to_idx)
            nodes.append(seed)
        data.center_nodes = torch.LongTensor(nodes)
        data.context_nodes = torch.LongTensor(context_nodes)
        data.context_nodes_to_graph_idx = torch.cat(context_nodes_to_graph_idx, dim=0)
        data.k = self.k  # number of node-context pairs

        return data

class ContrastMask6(MaskAtom):
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, dis=3, keep_dis=True, add_wave=False, k=5, step=1):
        super(ContrastMask6, self).__init__(num_atom_type=num_atom_type,
                                                   num_edge_type=num_edge_type, mask_rate=mask_rate,
                                                   mask_edge=mask_edge)
        # self.k = k
        self.dis = dis
        self.keep_dis = keep_dis
        self.add_wave = add_wave
        self.k = k
        self.step = step

    def __call__(self, data, masked_atom_indices=None, k=5, mask_type=0):  # 0 is mask node and 1 is mask edge #assume
        # the molecule graph is a connected graph ??
        # # substruct...
        # num_atoms = data.x.size()[0]
        # data.node_idx_tensor = torch.tensor(range(num_atoms)).view(1, -1)
        # num_types = 2
        # k = self.k
        # 0 node mask; 1 is edge mask; 3 is no mask; supervised information is added in the training process...
        # data.mask_node_label = torch.empty((0, 0), dtype=torch.long)
        # data.masked_atom_indices = torch.empty((0,), dtype=torch.long)
        num_nodes = data.x.size(0)
        seed = np.random.choice(np.arange(0, num_nodes), size=1)[0]

        k_to_nodes = dict()
        G = graph_data_obj_to_nx_simple(data)
        nodes_to_dis = nx.single_source_shortest_path_length(G, seed, self.k)
        dis_to_nodes = dict()
        for node in nodes_to_dis:
            nowk = nodes_to_dis[node]
            if nowk not in dis_to_nodes:
                dis_to_nodes[nowk] = [node]
            else:
                dis_to_nodes[nowk].append(node)
        for dis in dis_to_nodes:
            dis_to_nodes[dis] = torch.LongTensor(dis_to_nodes[dis])
        data.dis_to_nodes = dis_to_nodes
        data.k = self.k

        return data


class MaskAtom2:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        # for atom_idx in masked_atom_indices:
        #     data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
        masked_x = data.x
        for atom_idx in masked_atom_indices:
            masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])
        data.masked_x = masked_x

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class MaskAtom3:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, step=1):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.step = step

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        num_atoms = data.x.size(0)

        sampled_size = int(num_atoms * self.mask_rate + 1)
        res_nodes = set([i for i in range(num_atoms)])
        masked_atom_indices = []
        G = graph_data_obj_to_nx_simple(data)
        context_nodes_old_idx = list()
        context_nodes_to_idx = []
        cumsum = 0
        mask_node_labels_list = []
        while (len(masked_atom_indices) < sampled_size and len(res_nodes) > 0):
            now_sample = random.sample(list(res_nodes), 1)[0]

            context_nodes = nx.single_source_shortest_path_length(G, now_sample, self.step)
            context_nodes = set(list(context_nodes.keys()))
            if len(context_nodes) == 1:
                res_nodes -= {now_sample}
                continue
            mask_node_labels_list.append(data.x[now_sample].view(1, -1))
            res_nodes -= context_nodes
            context_nodes -= {now_sample}
            context_nodes_old_idx = context_nodes_old_idx + list(context_nodes)
            context_nodes_to_idx.append(torch.full((len(context_nodes),), cumsum))
            cumsum += 1
            masked_atom_indices.append(now_sample)


        # create mask node label by copying atom feature of mask atom

        idx_to_new_idx = dict()
        res_num_nodes = 0
        res_nodes_old_idx = []
        for i in range(num_atoms):
            if i not in masked_atom_indices:
                res_nodes_old_idx.append(i)
                idx_to_new_idx[i] = res_num_nodes
                res_num_nodes += 1
        assert len(res_nodes_old_idx) == res_num_nodes

        # mask_node_labels_list = []
        # context_nodes_list = []
        # # context_nodes_num = []
        # context_nodes_to_idx = []
        # cumsum = 0
        # data.masked_atom_num = len(masked_atom_indices)
        # for atom_idx in masked_atom_indices:
        #     mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        #     node_to_dis = nx.single_source_shortest_path_length(G, atom_idx, self.step)
        #     nodes = set(node_to_dis.keys()) - {atom_idx}
        #     real_nodes =
        #
        #     context_nodes_list += list(nodes)
        #     # context_nodes_num.append(len(nodes))
        #     node_to_idx = torch.full((len(nodes),), cumsum)
        #     cumsum += 1
        #     context_nodes_to_idx.append(node_to_idx)

        context_nodes_new_idx = [idx_to_new_idx[i] for i in context_nodes_old_idx]

        data.context_nodes = torch.LongTensor(context_nodes_new_idx)
        data.context_nodes_to_idx = torch.cat(context_nodes_to_idx, dim=0).long()
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)
        data.masked_atom_num = len(masked_atom_indices)

        edge_attr = []
        edge_indices = []
        for j in range(data.edge_index.size(1)):
            fr = int(data.edge_index[0, j])
            to = int(data.edge_index[1, j])
            if fr in masked_atom_indices or to in masked_atom_indices:
                continue
            else:
                edge_indices.append([idx_to_new_idx[fr], idx_to_new_idx[to]])
                # edge_indices.append([to, fr])
                edge_attr.append(data.edge_attr[j, :].view(1, -1))

        if len(edge_indices) == 0:
            data.context_nodes = torch.LongTensor(context_nodes_old_idx)
        else:
            edge_indices = torch.LongTensor(edge_indices).t()
            edge_attr = torch.cat(edge_attr, dim=0).long()
            # edge_attr = torch.LongTensor(edge_attr)
            # data.x = res_x
            data.edge_index = edge_indices
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

from utils2 import get_bfs_order

class MaskSubstruct:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True, step=1):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.step = step

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        num_atoms = data.x.size(0)

        seed = random.sample(range(num_atoms), 1)[0]
        node_ordered_list = get_bfs_order(data, seed)
        # block information broadcasting ways
        edge_indices_masked = []
        edge_attr_masked = []
        connected_edge_indices = []
        masked_edge_labels = []
        node_to_order_dict = {node: i for i, node in enumerate(node_ordered_list)}
        # todo check the usage of self loop in the GNN models ? in this masking strategy, there should be no
        #  self loop in the propagation process
        num_edges = data.edge_index.size(1)

        for i in range(num_edges):
            fr, to = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if node_to_order_dict[fr] < node_to_order_dict[to]:
                edge_indices_masked.append(data.edge_index[:, i].unsqueeze(1))
                edge_attr_masked.append(data.edge_attr[i, :].unsqueeze(0))
            else:
                connected_edge_indices.append(i)
                masked_edge_labels.append(data.edge_attr[i, :].unsqueeze(0))
        edge_indices_masked = torch.cat(edge_indices_masked, dim=1)
        edge_attr_masked = torch.cat(edge_attr_masked, dim=0)
        connected_edge_indices = torch.LongTensor(connected_edge_indices)
        masked_edge_labels = torch.cat(masked_edge_labels, dim=0)
        assert edge_indices_masked.size(0) == 2, edge_attr_masked.size(1) == data.edge_attr.size(1)
        node_ordered = torch.LongTensor(node_ordered_list)

        cut_order_indices_list = []
        masked_atom_labels_list = []
        for i in range(1, num_atoms):
            masked_atom_labels_list.append(data.x[node_ordered_list[i]].view(1, -1))
            cut_order_indices_list.append(torch.tensor([[0, i]], dtype=torch.long))
        cut_order_indices = torch.cat(cut_order_indices_list, dim=0)
        # need add num_atom in the batch transformation
        masked_atom_labels = torch.cat(masked_atom_labels_list, dim=0)

        data.edge_indices_masked = edge_indices_masked
        data.edge_attr_masked = edge_attr_masked
        data.node_ordered = node_ordered
        data.cut_order_indices = cut_order_indices
        data.masked_atom_labels = masked_atom_labels
        data.mask_node_indices = node_ordered[1:]
        data.connected_edge_indices = connected_edge_indices
        data.masked_edge_labels = masked_edge_labels

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class MaskAtom7:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        data.masked_x = data.x
        for atom_idx in masked_atom_indices:
            data.masked_x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


from utils2 import get_node_context, get_intimacy_matrix, get_wl_position_embedding
class AttentionBased:
    def __init__(self, args, num_atom_type, num_edge_type, mask_rate, k=5, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.fea_dim = args.emb_dim
        self.k = args.num_context
        self.num_layer = args.num_layer

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        num_atoms = data.x.size(0)
        assert self.fea_dim % 2 == 0
        S = get_intimacy_matrix(data, self.num_layer)   # num_layer's power iteration for the approximation calculation
        top_k_context = get_node_context(S, self.k + 1)
        wl_embedding = get_wl_position_embedding(data, self.fea_dim // 2, max_iter=self.num_layer)
        data.context_pos_indices = torch.arange(self.k + 1).repeat(num_atoms, 1)
        print(data.context_pos_indices.size())
        data.context_node_indices = top_k_context
        data.wl_embedding = wl_embedding
        print("wl_embedding shape", data.wl_embedding.size())

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)

class mask_strategies:
    def __init__(self, args, model_list, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        # self.num_atom_type = num_atom_type
        # self.num_edge_type = num_edge_type
        # self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        # self.fea_dim = args.emb_dim
        # # self.k = args.num_context
        # self.num_layer = args.num_layer
        self.model, self.pred_model = model_list
        self.device = args.device
        self.mask_rate = args.mask_rate


    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        with torch.no_grad():
            num_atoms = data.x.size(0)
            varis = []
            for i in range(num_atoms):
                masked_x = data.x
                masked_x[i, :] = torch.tensor([0, 0], dtype=torch.long).to(self.device)
                node_rep = self.model(masked_x)
                pred_node = self.pred_model(node_rep)
                varis.append(torch.mean(torch.var(pred_node, dim=-1)).item())
            varis = torch.FloatTensor(varis)
            top_k_varis_nodes = torch.argsort(varis, dim=0)
            num_masked = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = top_k_varis_nodes[:num_masked].cpu()

        mask_node_labels_list = []
        num_masked = masked_atom_indices.size(0)
        for i in range(num_masked):
            atom_idx = masked_atom_indices[i].item()
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = masked_atom_indices

        # modify the original node feature of the masked node
        for i in range(num_masked):
            atom_idx = masked_atom_indices[i].item()
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for i in range(num_masked):
                    atom_idx = masked_atom_indices[i].item()
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)


        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class structural_pair_extract:
    def __init__(self, pos_sample_ratio, neg_sample_ratio, mask_node=False, mask_ratio=0.15):
        self.num_pos_sample_ratio = pos_sample_ratio
        self.num_neg_sample_ratio = neg_sample_ratio
        self.mask_node = mask_node
        self.mask_ratio = mask_ratio

    def __call__(self, data, mask_type=None):
        num_atom = data.x.size(0)
        connected_pairs_dict = dict()
        num_edge = data.edge_index.size(1)
        for i in range(0, num_edge, 2):
            a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
            if (a, b) not in connected_pairs_dict:
                connected_pairs_dict[(a, b)] = 1
        unconnected_edge_list = []
        for a_idx in range(num_atom - 1):
            for b_idx in range(a_idx + 1, num_atom):
                if (a_idx, b_idx) not in connected_pairs_dict and (b_idx, a_idx) not in connected_pairs_dict:
                    unconnected_edge_list.append(torch.tensor([a_idx, b_idx], dtype=torch.long).view(-1, 1))
        unconnected_edge = torch.cat(unconnected_edge_list, dim=1)
        num_unconnected = unconnected_edge.size(1)
        num_connected = num_edge // 2
        sample_connected_indices = np.random.choice(range(num_connected), int(num_connected * self.num_pos_sample_ratio + 1))
        sample_connected_indices = sample_connected_indices * 2
        data.sample_connected_indices = data.edge_index[:, sample_connected_indices]
        # print(data.sample_connected_indices.size())
        data.sample_connected_edge_attr = data.edge_attr[sample_connected_indices, :]
        if len(data.sample_connected_indices.size()) == 1:
            data.sample_connected_indices = data.sample_connected_indices.view(-1, 1)
        if len(data.sample_connected_edge_attr.size()) == 1:
            data.sample_connected_edge_attr = data.sample_connected_edge_attr.view(1, -1)
        sample_unconnected_indices = np.random.choice(range(num_unconnected), int(num_unconnected * self.num_neg_sample_ratio + 1))
        data.sample_unconnected_indices = unconnected_edge[:, sample_unconnected_indices]
        if len(data.sample_unconnected_indices.size()) == 1:
            data.sample_unconnected_indices = data.sample_unconnected_indices.view(-1, 1)

        if self.mask_node == True:
            num_atom = data.x.size(0)
            num_masked = int(self.mask_ratio * num_atom + 1)
            masked_atom_indices = np.random.choice(range(num_atom), num_masked)
            masked_atom_indices = torch.from_numpy(masked_atom_indices)
            masked_atom_labels = data.x[masked_atom_indices, :]
            data.mask_node_label = masked_atom_labels
            data.masked_atom_indices = masked_atom_indices


        return data




if __name__ == "__main__":
    # transform = NegativeEdge()
    # dataset = MoleculeDataset("dataset/tox21", dataset="tox21")
    # transform(dataset[0])

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(m)
    root_idx = 13
    G = graph_data_obj_to_nx_simple(data)

    for i in range(data.x.size(0)):
        print(data.x[i, :])

    print("\n")
    for i in range(data.edge_index.size(1)):
        print(data.edge_index[:, i])
    print("\n")
    for i in range(data.edge_attr.size(0)):
        print(data.edge_attr[i, :])

    print(G.edges())
    print(G.nodes())
    et = ExtractMaskedData()
    data_tr = et(data, seed=4)

    with open("tmp_graph_info.txt", "w") as wf:
        for j in range(data_tr.split_x.size(0)):
            wf.write(str(data_tr.split_x[j, :]) + "\n")
        wf.write("\n")
        for j in range(data_tr.split_edge_index.size(1)):
            wf.write(str(data_tr.split_edge_index[:, j]) + "\n")
        wf.write("\n")
        for j in range(data_tr.split_edge_attr.size(0)):
            wf.write(str(data_tr.split_edge_attr[j, :]) + "\n")
    #
    # print(data_tr.split_x)
    # print(data_tr.split_edge_index)
    # print(data_tr.split_edge_attr)

    """
    # TODO(Bowen): more unit tests
    # test ExtractSubstructureContextPair

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(m)
    root_idx = 13

    # 0 hops: no substructure or context. We just test the absence of x attr
    transform = ExtractSubstructureContextPair(0, 0, 0)
    transform(data, root_idx)
    assert not hasattr(data, 'x_substruct')
    assert not hasattr(data, 'x_context')

    # k > n_nodes, l1 = 0 and l2 > n_nodes: substructure and context same as
    # molecule
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(100000, 0, 100000)
    transform(data, root_idx)
    substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
                                                 data.edge_index_substruct,
                                                 data.edge_attr_substruct)
    context_mol = graph_data_obj_to_mol_simple(data.x_context,
                                               data.edge_index_context,
                                               data.edge_attr_context)
    assert check_same_molecules(AllChem.MolToSmiles(substruct_mol),
                                AllChem.MolToSmiles(context_mol))

    transform = ExtractSubstructureContextPair(1, 1, 10000)
    transform(data, root_idx)

    # increase k from 0, and increase l1 from 1 while keeping l2 > n_nodes: the
    # total number of atoms should be n_atoms
    for i in range(len(m.GetAtoms())):
        data = mol_to_graph_data_obj_simple(m)
        print('i: {}'.format(i))
        transform = ExtractSubstructureContextPair(i, i, 100000)
        transform(data, root_idx)
        if hasattr(data, 'x_substruct'):
            n_substruct_atoms = data.x_substruct.size()[0]
        else:
            n_substruct_atoms = 0
        print('n_substruct_atoms: {}'.format(n_substruct_atoms))
        if hasattr(data, 'x_context'):
            n_context_atoms = data.x_context.size()[0]
        else:
            n_context_atoms = 0
        print('n_context_atoms: {}'.format(n_context_atoms))
        assert n_substruct_atoms + n_context_atoms == len(m.GetAtoms())

    # l1 < k and l2 >= k, so an overlap exists between context and substruct
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(2, 1, 3)
    transform(data, root_idx)
    assert hasattr(data, 'center_substruct_idx')

    # check correct overlap atoms between context and substruct


    # m = AllChem.MolFromSmiles('COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(C)C(OC)=C2C)C=C1')
    # data = mol_to_graph_data_obj_simple(m)
    # root_idx = 9
    # k = 1
    # l1 = 1
    # l2 = 2
    # transform = ExtractSubstructureContextPaidata = mol_to_graph_data_obj_simple(m)r(k, l1, l2)
    # transform(data, root_idx)
    pass

    # TODO(Bowen): more unit tests
    # test MaskAtom
    from loader import mol_to_graph_data_obj_simple, \
        graph_data_obj_to_mol_simple

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    original_data = mol_to_graph_data_obj_simple(m)
    num_atom_type = 118
    num_edge_type = 5

    # manually specify masked atom indices, don't mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=False)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    assert not hasattr(data, 'mask_edge_label')
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()

    # manually specify masked atom indices, mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=True)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()
    # check that the correct rows in edge_attr have been modified to be mask edge
    # type, and the mask_edge_label are correct
    rdkit_bonds = []
    for atom_idx in masked_atom_indices:
        bond_indices = list(AllChem.FindAtomEnvironmentOfRadiusN(m, radius=1,
                                                                 rootedAtAtom=atom_idx))
        for bond_idx in bond_indices:
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetBeginAtomIdx(), m.GetBonds()[
                    bond_idx].GetEndAtomIdx()))
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetEndAtomIdx(), m.GetBonds()[
                    bond_idx].GetBeginAtomIdx()))
    rdkit_bonds = set(rdkit_bonds)
    connected_edge_indices = []
    for i in range(data.edge_index.size()[1]):
        if tuple(data.edge_index.numpy().T[i].tolist()) in rdkit_bonds:
            connected_edge_indices.append(i)
    assert (data.edge_attr[connected_edge_indices] ==
            torch.tensor(([num_edge_type, 0]))).all()
    assert (data.mask_edge_label == original_data.edge_attr[
        connected_edge_indices[::2]]).all() # data.mask_edge_label contains
    # the unique edges (ignoring direction). The data obj has edge ordering
    # such that two directions of a single edge occur in pairs, so to get the
    # unique undirected edge indices, we take every 2nd edge index from list
    """

