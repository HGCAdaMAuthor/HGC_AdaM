import torch
import copy
import random
import networkx as nx
import numpy as np
from torch_geometric.utils import convert
from loader import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from loader import mol_to_graph_data_obj_simple, \
#     graph_data_obj_to_mol_simple
#
# from loader import MoleculeDataset
# import scipy.sparse as sparse
# from scipy.sparse import linalg
# import karateclub as kc
# import torch.nn.functional as F
# from karateclub.node_embedding.structural import graphwave
# import sklearn.preprocessing as preprocessing
from rdkit import Chem
from torch_geometric.data import Data
# from rdkit.Chem import Descriptors
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import dgl

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)


        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def mol_to_dgl_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_atoms = len(mol.GetAtoms())
    if len(mol.GetBonds()) > 0:
        edge_fr_list = list()
        edge_to_list = list()
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_fr_list += [i, j]
            edge_to_list += [j, i]
        edge_fr = torch.tensor(edge_fr_list, dtype=torch.long)
        edge_to = torch.tensor(edge_to_list, dtype=torch.long)
    else:
        edge_fr = torch.empty((0, ), dtype=torch.long)
        edge_to = torch.empty((0, ), dtype=torch.long)
    dglg = dgl.DGLGraph()
    dglg.add_nodes(num_atoms)
    dglg.add_edges(edge_fr, edge_to)
    return dglg


def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        # print(atomic_num_idx)
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx - 1]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol

def gen_bfs_order(edge_index, seed):
    nodes_to_neighs = dict()
    for i in range(edge_index.size(1)):
        a, b = int(edge_index[0, i]), int(edge_index[1, i])
        if a not in nodes_to_neighs:
            nodes_to_neighs[a] = [b]
        else:
            nodes_to_neighs[a].append(b)
        if b not in nodes_to_neighs:
            nodes_to_neighs[b] = [a]
        else:
            nodes_to_neighs[b].append(a)
    ordered_nodes = {seed: 0}
    nodes_to_exi = {seed: 1}
    que = [seed]
    cnt = 0
    l = 0
    while (l < len(que)):
        now = que[l]
        l += 1
        for i in nodes_to_neighs[now]:
            if i in nodes_to_exi:
                continue
            else:
                que.append(i)
                nodes_to_exi[i] = 1
                cnt += 1
                ordered_nodes[i] = cnt
    return ordered_nodes


def graph_data_part_to_nx_simple(edge_index, num_nodes):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    # for i in range(num_nodes):
    G.add_nodes_from([j for j in range(num_nodes)])


    #
    # atom_features = data.x.cpu().numpy()
    # num_atoms = atom_features.shape[0]
    # for i in range(num_atoms):
    #     atomic_num_idx, chirality_tag_idx = atom_features[i]
    #     G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
    #     pass
    #
    # # bonds
    # edge_index = data.edge_index.cpu().numpy()
    # edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        # bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    return nx.to_numpy_matrix(G)


    # return G

def graph_data_part_to_nx_simple_no_loop(edge_index, num_nodes):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    # for i in range(num_nodes):
    G.add_nodes_from([j for j in range(num_nodes)])


    #
    # atom_features = data.x.cpu().numpy()
    # num_atoms = atom_features.shape[0]
    # for i in range(num_atoms):
    #     atomic_num_idx, chirality_tag_idx = atom_features[i]
    #     G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
    #     pass
    #
    # # bonds
    # edge_index = data.edge_index.cpu().numpy()
    # edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        # bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx) and begin_idx != end_idx:
            G.add_edge(begin_idx, end_idx)
    return nx.to_numpy_matrix(G)

def from_edge_index_to_adj(edge_index, num_nodes):
    A = torch.zeros(num_nodes, num_nodes)
    num_edges = edge_index.shape[1]
    for j in range(num_edges):
        a, b = edge_index[0, j], edge_index[1, j]
        if a != b:
            try:
                A[a, b] = A[b, a] = 1.
            except:
                print(num_edges)
                print(edge_index)
                raise RuntimeError("aaa")
    return A

def get_subgraph_data(data, node_list):
    n_sample = len(node_list)
    idxnew_to_idx = {i: node_idx for i, node_idx in enumerate(node_list)}
    idx_to_idxnew = {node_idx: i for i, node_idx in enumerate(node_list)}
    x = torch.zeros(n_sample, data.x.size(1), dtype=torch.long)
    for j in range(n_sample):
        x[j, :] = data.x[idxnew_to_idx[j], :]
    edge_idx = []
    edge_attri = []
    for i in range(data.edge_index.size(1)):
        a = int(data.edge_index[0, i])
        b = int(data.edge_index[1, i])

        if a in idx_to_idxnew and b in idx_to_idxnew:
            a = idx_to_idxnew[a]
            b = idx_to_idxnew[b]
            edge_idx.append([a, b])
            edge_attri.append(data.edge_attr[i, :].unsqueeze(0))
    # print(len(edge_attri), data.edge_attr.size(), n_sample, edge_attri[0])

    # if len(edge_attri) == 0:
    #     print(len(edge_idx), n_sample, ratio, data.x.size(0))
    #     raise RuntimeError("len(edge_Attri) == 0!")
    edge_idx = np.array(edge_idx).T
    edge_idx = torch.from_numpy(edge_idx).long()
    # edge_attri = torch.FloatTensor(edge_attri)
    if len(edge_attri) == 0:
        edge_attri = torch.empty((0, 2), dtype=torch.long)
    else:
        edge_attri = torch.cat(edge_attri, dim=0)
    return x, edge_idx, edge_attri


def sample_subgraph(data, ratio, seed=None):
    assert ratio > 0 and ratio < 1
    num_atoms = data.x.size(0)
    n_sample = int(ratio * num_atoms)
    if n_sample <= 1:
        n_sample = 2
    if seed == None:
        seed = np.random.choice(np.arange(0, num_atoms), size=1)[0]
    G = graph_data_obj_to_nx_simple(data)

    node_to_dis = nx.single_source_shortest_path_length(G, seed)
    node_to_dis = sorted(node_to_dis.items(), key=lambda x: x[1])
    sampled_nodes = [x[0] for x in node_to_dis[:n_sample]]
    n_sample = len(sampled_nodes)
    idx_to_idxnew = {j: i for i, j in enumerate(sampled_nodes)}
    idxnew_to_idx = {i: j for i, j in enumerate(sampled_nodes)}
    x = torch.zeros(n_sample, data.x.size(1), dtype=torch.long)
    for j in range(n_sample):
        x[j, :] = data.x[idxnew_to_idx[j], :]
    edge_idx = []
    edge_attri = []
    for i in range(data.edge_index.size(1)):
        a = int(data.edge_index[0, i])
        b = int(data.edge_index[1, i])

        if a in idx_to_idxnew and b in idx_to_idxnew:
            a = idx_to_idxnew[a]
            b = idx_to_idxnew[b]
            edge_idx.append([a, b])
            edge_attri.append(data.edge_attr[i, :].unsqueeze(0))
    # print(len(edge_attri), data.edge_attr.size(), n_sample, edge_attri[0])

    if len(edge_attri) == 0:
        print(len(edge_idx), n_sample, ratio, data.x.size(0))
        raise RuntimeError("len(edge_Attri) == 0!")
    edge_idx = np.array(edge_idx).T
    edge_idx = torch.from_numpy(edge_idx).long()
    # edge_attri = torch.FloatTensor(edge_attri)
    edge_attri = torch.cat(edge_attri, dim=0)

    # edge_attri = torch.from_numpy(np.array(edge_attri))

    return x, edge_idx, edge_attri


def sampled_subgraph_gcc(data, seed=None, step_dist=[1.0, 0.0, 0.0], length=64, rsprob=0.8):
    # assert ratio > 0 and ratio < 1
    num_atoms = data.x.size(0)
    # n_sample = int(ratio * num_atoms)
    if seed == None:
        seed = np.random.choice(np.arange(0, num_atoms), size=1)[0]

    # step_dist = [1.0, 0.0, 0.0]
    # print(seed)
    step = np.random.choice(len(step_dist), 1, p=step_dist)[0] + 1
    G = graph_data_obj_to_nx_simple(data)
    dgl_bg = dgl.DGLGraph(G)
    # edges = torch.tensor(dgl_bg.edges(), dtype=torch.long)
    # dgl_bg = dgl.graph((edges[0], edges[1]))
    # print(seed)
    length = 64
    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [seed],
        restart_prob=0.0,
        length=step)
    # print(len(traces[0]))
    other_node = int(traces[0][-1].item())
    # traces, _ = dgl.sampling.random_walk(
    #     dgl_bg,
    #     [seed, other_node],
    #     # prob="pos_sample_p",
    #     restart_prob=rsprob,
    #     length=length * 19)
    # traces = dgl.contrib.sampling.random_walk_with_restart(
    #     dgl_bg,
    #     seeds=[seed, other_node],
    #     restart_prob=rsprob,
    #     max_nodes_per_seed=64,
    # )

    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [seed for __ in range(1)],
        # prob="pos_sample_p",
        restart_prob=0.0,
        length=length)
    # todo: count the frequency and choose top k ones?
    # subv = torch.unique(traces).tolist()
    subv_a = torch.unique(traces).tolist()
    # subv_a = torch.unique(traces[0]).tolist()

    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [other_node for __ in range(1)],
        # prob="pos_sample_p",
        restart_prob=0.0,
        length=length)
    subv_b = torch.unique(traces).tolist()
    # subv_b = torch.unique(traces[1]).tolist()
    # print("calculated...", subv)
    # try:
    #     # subv.remove(seed)
    #     subv_a.remove(seed)
    #     subv_a.remove(seed)
    # except:
    #     pass
    try:
        subv_a.remove(-1)
        subv_b.remove(-1)
    except:
        pass
    if len(subv_a) == 1:
        subv_a.append(other_node)
    if len(subv_b) == 1:
        subv_b.append(seed)
    # print(len(subv_a), len(subv_b), seed, other_node)
    return [subv_a, subv_b]


def sample_subgraph_only_node(data, ratio, seed=None):
    assert ratio > 0 and ratio < 1
    num_atoms = data.x.size(0)
    n_sample = int(ratio * num_atoms)
    if seed == None:
        seed = np.random.choice(np.arange(0, num_atoms), size=1)[0]
    G = graph_data_obj_to_nx_simple(data)

    node_to_dis = nx.single_source_shortest_path_length(G, seed)
    node_to_dis = sorted(node_to_dis.items(), key=lambda x: x[1])
    sampled_nodes = [x[0] for x in node_to_dis[:n_sample]]
    # n_sample = len(sampled_nodes)
    return sampled_nodes


def filter_attri(data, k=3, sigma=1):
    G = graph_data_obj_to_nx_simple(data)
    # G.remove_edges_from(G.selfloop_edges())
    adj = nx.to_numpy_matrix(G)
    N = adj.shape[0]
    adj = torch.from_numpy(adj) + (sigma - 1) * torch.eye(N)
    D = adj.sum(dim=1, keepdim=True)
    A_rw = adj / D
    filtered_attri = data.x
    for j in range(k):
        filtered_attri = torch.matmul(A_rw.float(), filtered_attri.float())
    return filtered_attri


def filter_attri_from_batch(x, edge_index, num_nodes, num_edges, filter_k, sigma=1):
    filtered_x = list()
    num_graphs = len(num_nodes)
    cum_nodes = 0
    cum_edges = 0
    for i in range(num_graphs):
        now_graph_nodes = num_nodes[i]
        now_graph_edges = num_edges[i]
        x_now = x[cum_nodes: cum_nodes + now_graph_nodes, :]
        if filter_k[i] == 0:
            filtered_x.append(x_now)
            cum_nodes += now_graph_nodes
            cum_edges += now_graph_edges
            continue
        # print(i, now_graph_nodes, x_now.size(0))
        # print(filter_k)
        edge_index_now = edge_index[:, cum_edges: cum_edges + now_graph_edges] - cum_nodes
        # A = from_edge_index_to_adj(edge_index_now, now_graph_nodes) + sigma * torch.eye(now_graph_nodes)
        A = torch.from_numpy(graph_data_part_to_nx_simple(edge_index_now, now_graph_nodes)) + sigma * torch.eye(now_graph_nodes)
        D = A.sum(dim=1, keepdim=True)
        A_rw = A / D
        A_rw = A_rw.to(x.device).float()
        x_filtered = x_now
        for j in range(filter_k[i]):
            x_filtered = torch.matmul(A_rw, x_filtered)
        filtered_x.append(x_filtered)
        cum_nodes += now_graph_nodes
        cum_edges += now_graph_edges

    filtered_x = torch.cat(filtered_x, dim=0)
    assert filtered_x.size() == x.size(), "must keep the same dimension!"
    return filtered_x


def from_rep_to_subgraph_rep(batch, x, pool_func="sum"):
    tot_num_sub = batch.num_sub_nodes.size(0)
    subgraph_reps = list()
    cusum_nodes = 0
    for i in range(tot_num_sub):
        now_sub_num_nodes = int(batch.num_sub_nodes[i])
        now_x_range = batch.s_nodes[cusum_nodes: cusum_nodes + now_sub_num_nodes]
        now_x = x[now_x_range, :]
        assert now_x.size(0) == now_sub_num_nodes
        if pool_func == "sum":
            now_sub_rep = now_x.sum(dim=0, keepdims=True)
        elif pool_func == "mean":
            now_sub_rep = now_x.mean(dim=0, keepdims=True)
        else:
            raise NotImplementedError("pool_func should be sum or mean")
        subgraph_reps.append(now_sub_rep)
        cusum_nodes += now_sub_num_nodes
        assert now_sub_rep.size(0) == 1
    subgraph_reps = torch.cat(subgraph_reps, dim=0)
    assert subgraph_reps.size(0) == tot_num_sub
    return subgraph_reps

def node_context_extract_with_step(data, seed, step):
    G = graph_data_obj_to_nx_simple(data)
    seed_in_step = nx.single_source_shortest_path_length(G, seed, step)
    context_nodes = list(seed_in_step.keys())
    return context_nodes

def get_bfs_order(data, seed):
    G = graph_data_obj_to_nx_simple(data)
    node_to_step = nx.single_source_shortest_path_length(G, seed)
    node_step_list = sorted(node_to_step.items(), key=lambda i: i[0])
    node_ordered = [par[0] for par in node_step_list]
    return node_ordered

def get_adj_dict(data):
    node_to_adj = dict()

    num_edges = data.edge_index.size(1)
    for i in range(num_edges):
        a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
        if a not in node_to_adj:
            node_to_adj[a] = dict()
        if b not in node_to_adj:
            node_to_adj[b] = dict()
        node_to_adj[a][b] = 1
        node_to_adj[b][a] = 1
    return node_to_adj

import hashlib
import math

def positional_embedding(node_to_data, dh_2):
    num_nodes = len(node_to_data)
    all_pos_emb = list()
    for node in range(num_nodes):
        color = node_to_data[node]
        wl_emb = [torch.tensor([math.sin(color / math.pow(10000, 2 * i / (dh_2 * 2))),
                                math.cos(color / math.pow(10000, (2 * i + 1) / (dh_2 * 2)))]) for i in range(dh_2)]
        wl_emb = torch.cat(wl_emb, dim=0)
        all_pos_emb.append(wl_emb.unsqueeze(0))
    all_pos_emb = torch.cat(all_pos_emb, dim=0)
    return all_pos_emb


def get_wl_position_embedding(data, dh_2, max_iter=2):
    assert dh_2 > 0
    node_to_adj = get_adj_dict(data)
    node_to_color = dict()
    num_nodes = data.x.size(0)
    for i in range(num_nodes):
        node_to_color[i] = 1
    iter_count = 0
    while True:
        node_to_color_new = dict()
        for node in range(num_nodes):
            neis_color_list = [node_to_color[nei] for nei in node_to_adj[node]]
            color_string_list = [str(node_to_color[node])] + sorted([str(color) for color in neis_color_list])
            color_string = "_".join(color_string_list)
            hash_obj = hashlib.md5(color_string.encode())
            hashing = hash_obj.hexdigest()
            node_to_color_new[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(node_to_color_new.values())))}
        for node in range(num_nodes):
            node_to_color_new[node] = color_index_dict[node_to_color_new[node]]
        if node_to_color_new == node_to_color or iter_count == max_iter:
            break
        else:
            node_to_color = node_to_color_new
            iter_count += 1
    all_wl_emb = positional_embedding(node_to_color, dh_2)
    # all_wl_emb = list()
    # for node in range(num_nodes):
    #     color = node_to_color[node]
    #     wl_emb = [torch.tensor([math.sin(color / math.pow(10000, 2 * i / (dh_2 * 2))),
    #                              math.cos(color / math.pow(10000, (2 * i + 1) / (dh_2 * 2)))]) for i in range(dh_2)]
    #     wl_emb = torch.cat(wl_emb, dim=0)
    #     all_wl_emb.append(wl_emb.unsqueeze(0))
    # all_wl_emb = torch.cat(all_wl_emb, dim=0)
    assert all_wl_emb.size(0) == num_nodes
    return all_wl_emb

# todo for intimacy embedding, should use the embedding layer?4
# todo it seems that calculating the hop distance between two nodes is time consuming... then how can we solve it?
def get_node_context(S, k):
    # S --- torch tensor n x n
    num_nodes = S.size(0)
    S[torch.arange(num_nodes), torch.arange(num_nodes)] = 1000000.0
    tok_k_node_idx = torch.argsort(S, descending=True)[:, :k]
    assert tok_k_node_idx[:, 0] == torch.arange(num_nodes)



    # for node in range(num_nodes):
    #     adjs = S[node, :]
    #     adjs[node] = 1000000
    #     top_k_node = torch.argsort(adjs, descending=True)[: k]
    return tok_k_node_idx

#

def get_intimacy_matrix(data, k, alpha=0.15, approx_mode="power"):
    assert k > 1
    # G = graph_data_obj_to_nx_simple(data)
    # adj = nx.to_numpy_matrix(G)
    # n = adj.shape[0]
    # adj[]
    adj = graph_data_part_to_nx_simple_no_loop(data.edge_index, data.x.size(0))
    adj = torch.from_numpy(adj)
    n = adj.size(0)
    D = adj.sum(dim=1, keepdim=True)
    tilde_adj = adj / torch.clamp(D, min=1e-12)
    inner_mat = torch.eye(n) + (1 - alpha) * tilde_adj
    if approx_mode == "power":
        S = alpha * torch.pow(inner_mat, k)
    else:
        raise NotImplementedError("approx_mode error")
    return S

def get_tree_edges(data):
    tree_edges = list()
    queue = list()
    vis = dict()
    num_atoms = data.x.size(0)
    vis = {i: 0 for i in range(num_atoms)}
    node_to_adjs = dict()
    for i in range(0, data.edge_index.size(1), 2):
        a, b = int(data.edge_index[0, i]), int(data.edge_index[1, i])
        if a not in node_to_adjs:
            node_to_adjs[a] = {}
        if b not in node_to_adjs:
            node_to_adjs[b] = {}
        node_to_adjs[a][b] = i // 2
        node_to_adjs[b][a] = i // 2
    seed = np.random.choice(range(num_atoms), 1)[0]
    queue.append(seed)
    vis[seed] = 1
    l = 0

    while l != len(queue):
        now = queue[l]
        for nei in node_to_adjs[now]:
            if vis[nei] == 0:
                vis[nei] = 1
                queue.append(nei)
                tree_edges.append(node_to_adjs[now][nei])
        l += 1

    all_edges = range(data.edge_index.size(1) // 2)
    no_tree_edges = set(all_edges) - set(tree_edges)
    return tree_edges, no_tree_edges

