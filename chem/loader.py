import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
# import networkx as nx

# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import dgl
#from util import MaskAtom, PairwiseNodeEdgeMask


# allowable node and edge features
# allowable_features = {
#     'possible_atomic_num_list' : list(range(1, 119)),
#     'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list' : [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list' : [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds' : [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs' : [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }

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

def graph_data_obj_to_nx_simple(data):
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
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G

def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
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

def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges

def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

# from compute_similarity import compute_similarity_1vn
import time
import random

GCC_GRAPH_CLASSIFICATION_DATASETS = ["imdb-multi", "rdt-5k", "imdb-binary", "rdt-b", "collab"]
GCC_NODE_CLASSIFICATION_DATASETS = ["usa_airport", "brazil_airport", "europe_airport",
                                    "h-index-rand-1", "h-index-top-1", "h-index",
                                    "kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]

NAME_TO_RAW_NAME = {
    "imdb-binary": "IMDB-BINARY",
    "imdb-multi": "IMDB-MULTI",
    "rdt-b": "REDDIT-BINARY",
    "rdt-5k": "REDDIT-MULTI-5K",
    "collab": "COLLAB",
}

NAME_TO_NUM_TASK = {
    "IMDB-BINARY": 2,
    "IMDB-MULTI": 3,
    "REDDIT-BINARY": 2,
    "REDDIT-MULTI-5K": 5,
    "COLLAB": 3,
}

# class GCCDatasets(torch.utils.data.IterableDataset):
#     def __init__(self, dataset):
#         super(GCCDatasets, self).__init__()
#         self.dataset = dataset
#
from data_util import create_node_classification_dataset, _create_dgl_graph

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False,
                 extract_sim=False,
                 num_samples=3,
                 k=7,
                 with_neg=False,
                 extract_sim_type="precomputed_rwr",
                 after_transform=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.after_transform = after_transform
        self.total = 0
        self.extract_sim = extract_sim
        self.num_samples = num_samples
        self.k = k
        self.with_neg = with_neg
        self.calculated_sim_score = list()
        self.calculated_sim_list = list()
        self.num_calculate_sim_samples = 20
        self.idx_to_adj_list_dict = dict()
        self.restart_prob = 0.8
        self.num_hops = 3
        self.extract_sim_type = extract_sim_type

        self.num_epochs = 100
        if not empty:
            if self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
                self.data = create_node_classification_dataset(dataset).data
                self.graphs = [_create_dgl_graph(self.data)]
                self.length = sum([g.number_of_nodes() for g in self.graphs])
                self.total = self.length
                if self.dataset in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]:
                    self.num_labels = 2
                else:
                    self.num_labels = self.data.y.shape[1]
                    self.labels = torch.argmax(self.data.y, dim=1, keepdim=False)
                self.idx_to_gra_data = dict()
                # for i in range(self.graphs[0].number_of_nodes()):
                #     self.idx_to_gra_data[i] = self.rw_sample_dgl_subgraph(i)
                # print("proprocessed done!")

            elif self.dataset not in GCC_GRAPH_CLASSIFICATION_DATASETS:
                self.data, self.slices = torch.load(self.processed_paths[0])
            else:
                self._create_graphs_from_local_file(NAME_TO_RAW_NAME[self.dataset])
            # input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
            # input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
            # self.smiles_list = list(input_df['smiles'])
            if self.dataset == "zinc_standard_agent" and extract_sim:
                input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
                input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
                self.smiles_list = list(input_df['smiles'])

                lenn = 2000000
                assert len(self.smiles_list) == lenn
                # self.calculated_sim_list = [torch.empty((0, ), dtype=torch.long) for i in range(lenn)]
                # self.calculated_sim_score = [torch.empty((0, ), dtype=torch.float64) for i in range(lenn)]
                #
                # len_per = lenn // 20
                # ls = [i * len_per for i in range(20)]
                self.pos_idxes = list()
                # for l in ls:  # not only one positive sample?
                processed_path = "./dataset/zinc_standard_agent/processed"
                sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))
                    # print(sim_to_score_dict_list)
                for j in range(len(sim_to_score_dict_list)):
                    assert isinstance(sim_to_score_dict_list[j], dict)
                    self.pos_idxes.append(sim_to_score_dict_list[j])
                        # self.calculated_sim_list.append(torch.empty((0, ), dtype=torch.long))
                        # self.calculated_sim_score.append(torch.empty((0, ), dtype=torch.float64))
                    # self.pos_idxes += sim_to_score_dict_list
                assert len(self.pos_idxes) == lenn

                idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
                assert isinstance(idx_to_adjs, dict) and len(idx_to_adjs) == lenn
                self.idx_to_adj_list_dict = idx_to_adjs

                # if self.extract_sim_type == "sim" and self.with_neg:
                #     per_neg_num_samples = self.k * self.num_samples * self.num_epochs
                #     self.idx_to_neg_samples_list = dict()
                #     print("Calculating negative samples.")
                #     for j in range(lenn):
                #         print(j)
                #         if j % 1000 == 0:
                #             print(j)
                #         idx_to_sim_score_dict = self.pos_idxes[j]
                #         keys = list()
                #         values = list()
                #         for idxbb in idx_to_sim_score_dict:
                #             keys.append(idxbb)
                #             values.append(idx_to_sim_score_dict[idxbb])
                #         sim_scores_all = torch.zeros((len(self.smiles_list),), dtype=torch.float64)
                #         if len(keys) > 0:
                #             calculated_sim_idx = torch.tensor(keys, dtype=torch.long)
                #             calculated_sim_scores = torch.tensor(values, dtype=torch.float64)
                #             sim_scores_all[calculated_sim_idx] = calculated_sim_scores / 4
                #         normalized_neg_sample_rates = torch.softmax(sim_scores_all, dim=0)
                #         neg_samples_idx = torch.multinomial(normalized_neg_sample_rates, per_neg_num_samples)
                #         self.idx_to_neg_samples_list[j] = [int(neg_samples_idx[neg_sample_idx]) for neg_sample_idx in range(neg_samples_idx.size(0))]


                if self.extract_sim_type == "precomputed_rwr":
                    self.idx_to_sampled_pos_idx = dict()
                    for j in range(lenn):
                        # print(j)
                        if j % 1000 == 0:
                            print(j)
                        if len(self.idx_to_adj_list_dict[j]) == 0:
                            self.idx_to_sampled_pos_idx[j] = [j for _ in range(self.num_samples * self.num_epochs)]
                            continue
                        # print(len(self.idx_to_adj_list_dict[j]))
                        r_ego_graph, new_idx_to_old_idx, _ = self.get_rhop_ego_graph(j, num_hops=self.num_hops + 1)
                        # print(r_ego_graph)
                        sampled_idx = self.rwr_sample_pos_graphs(r_ego_graph, 0, num_samples=self.num_samples * self.num_epochs * 2)
                        # print(sampled_idx)
                        sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                        sampled_num_idx = self.num_samples * self.num_epochs
                        if len(sampled_idx) > sampled_num_idx:
                            sampled_idx = sampled_idx[: sampled_num_idx]
                        elif len(sampled_idx) < sampled_num_idx:
                            sampled_idx += [j for _ in range(sampled_num_idx - len(sampled_idx))]
                        self.idx_to_sampled_pos_idx[j] = sampled_idx

    def _create_graphs_from_local_file(self, dataset_name):
        # raw_path = "./datasets/{}".format(dataset_name)
        # if self.env == "jizhi":
        raw_path = "/apdcephfs/private_meowliu/ft_local/graph_self_learn/datasets/{}".format(dataset_name)
        graph_idx_to_edge_pairs = np.load(os.path.join(raw_path, "node_to_edge_pairs.npy"), allow_pickle=True).item()
        print(len(graph_idx_to_edge_pairs))
        graph_idxs = list(graph_idx_to_edge_pairs.keys())
        min_graph_idx = min(graph_idxs)
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
            data = Data()
            data.x = torch.zeros((max_nodes, 2), dtype=torch.long)
            data.x[:, 0] = 119
            data.x[:, 1] = 0
            data.edge_index = edge_tensor.t()
            data.edge_attr = torch.zeros((data.edge_index.size(1), 2), dtype=torch.long)
            # data.y = torch.tensor([int(self.labels[j - min_graph_idx])], dtype=torch.long)
            # data.y = torch.zeros((NAME_TO_NUM_TASK[dataset_name], ), dtype=torch.long)
            data.y = torch.full((NAME_TO_NUM_TASK[dataset_name],), fill_value=-1, dtype=torch.long)
            data.y[int(self.labels[j - min_graph_idx])] = 1
            self.graphs.append(data)
        print(len(self.graphs))

    def get_rhop_ego_graph(self, idx, num_hops=3):
        idx_to_dis = dict()
        que = [idx]
        idx_to_dis[idx] = 0
        edge_fr, edge_to = list(), list()
        new_idx = 0
        old_to_new_idx = {idx: new_idx}
        while len(que) > 0:
            now_idx = que[-1]
            que.pop()
            new_now_idx = old_to_new_idx[now_idx]
            if idx_to_dis[now_idx] >= num_hops:
                break
            for other_idx in self.idx_to_adj_list_dict[now_idx]:
                if other_idx not in idx_to_dis:
                    new_idx += 1
                    old_to_new_idx[other_idx] = new_idx
                    idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                    edge_fr.append(new_now_idx)
                    edge_to.append(new_idx)
                    edge_fr.append(new_idx)
                    edge_to.append(new_now_idx)
                    que.insert(0, other_idx)
        edge_fr, edge_to = torch.tensor(edge_fr, dtype=torch.long), torch.tensor(edge_to, dtype=torch.long)
        dgl_g = dgl.DGLGraph()
        dgl_g.add_nodes(new_idx + 1)
        dgl_g.add_edges(edge_fr, edge_to)
        # dgl_g = dgl.graph((edge_fr, edge_to))
        # print(dgl_g.nodes(), dgl_g.edges())
        # print(dgl_g)
        new_idx_to_old_idx = {old_to_new_idx[i]: i for i in old_to_new_idx}
        # bg = dgl.to_bidirected(dgl_g)
        return dgl_g, new_idx_to_old_idx

    def rwr_sample_pos_graphs(self, dgl_bg, idx, num_samples=5):
        # print(type(dgl_bg))
        traces = dgl.contrib.sampling.random_walk_with_restart(
            dgl_bg,
            seeds=[idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=num_samples + 1)
        subv = torch.unique(torch.cat(traces[0])).tolist()
        try:
            subv.remove(idx)
        except ValueError:
            pass
        return subv

    def get_data_simple(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    def rw_sample_dgl_subgraph(self, idx):
        # if idx in self.idx_to_gra_data:
        #     return self.idx_to_gra_data[idx]
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        num_pat = 7 if self.dataset not in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"] else 8
        length = 16 if self.dataset not in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"] else 5
        traces, _ = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            [idx for __ in range(num_pat)],
            restart_prob=0,
            length=length)
        # print(traces)
        subv = torch.unique(traces).tolist()
        # print(len(subv))
        try:
            subv.remove(-1)
        except ValueError:
            pass
        graph_q = self.graphs[graph_idx].subgraph(subv)
        edge_fr = graph_q.edges()[0]
        edge_to = graph_q.edges()[1]
        edge_index = torch.cat([edge_fr.view(1, -1), edge_to.view(1, -1)], dim=0)
        data = Data()
        data.x = torch.zeros((len(subv), 2), dtype=torch.long)
        data.x[:, 0] = 119
        data.x[:, 1] = 0
        data.edge_index = edge_index
        data.edge_attr = torch.zeros((data.edge_index.size(1), 2), dtype=torch.long)
        # data.y = torch.tensor([int(self.labels[j - min_graph_idx])], dtype=torch.long)
        # data.y = torch.zeros((NAME_TO_NUM_TASK[dataset_name], ), dtype=torch.long)
        if self.dataset not in ["kdd", "icdm", "sigir", "cikm", "sigmod", "icde"]:
            data.y = torch.full((self.num_labels,), fill_value=-1, dtype=torch.long)
            data.y[int(self.data.y[idx].argmax().item())] = 1
        self.idx_to_gra_data[idx] = data
        return data

    def __getitem__(self, item):
        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS:
            return self.graphs[item]
        elif self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return self.rw_sample_dgl_subgraph(item)
        return super().__getitem__(item)

    def get(self, idx):
        # print(idx)

        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS:
            return self.graphs[idx]
        elif self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return self.rw_sample_dgl_subgraph(idx)
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        # each epoc ?
        if self.dataset == "zinc_standard_agent" and self.extract_sim:
            # num_all_data = len(self.smiles_list)
            if self.extract_sim_type == "precomputed_rwr":
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                # sampled_idx = self.idx_to_sampled_pos_idx[idx]
                prev_len = len(self.idx_to_sampled_pos_idx[idx])
                for i in range(self.num_samples):

                    real_idx = int(self.idx_to_sampled_pos_idx[idx][i])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)
                    self.idx_to_sampled_pos_idx[idx].pop(0)
                assert prev_len - len(self.idx_to_sampled_pos_idx[idx]) == self.num_samples

            if self.extract_sim_type == "rwr":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx = self.get_rhop_ego_graph(idx, num_hops=self.num_hops)
                    sampled_idx = self.rwr_sample_pos_graphs(dgl_bg, 0, num_samples=self.num_samples * 2)
                    sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                else:
                    sampled_idx = [idx for _ in range(self.num_samples)]
                if len(sampled_idx) > self.num_samples:
                    sampled_idx = sampled_idx[: self.num_samples]
                elif len(sampled_idx) < self.num_samples:
                    sampled_idx += [idx for _ in range(self.num_samples - len(sampled_idx))]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    real_idx = int(sampled_idx[i])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

            # base_score = torch.ones((num_all_data, ), dtype=torch.float64)
            # calculated_idx = self.calculated_sim_list[idx]
            # calculated_score = self.calculated_sim_score[idx]
            # mean_score = torch.mean(calculated_score)
            # base_score[calculated_idx] += (calculated_score - mean_score)
            # base_score[idx] = -1e9
            # sampled_rates = torch.softmax(base_score, dim=0)
            #
            # calculate_sim_samples = torch.multinomial(sampled_rates, self.num_calculate_sim_samples)
            # calculate_sim_smiles = list()
            # for i in range(len(calculate_sim_samples)):
            #     now_idx = int(calculate_sim_samples[i])
            #     calculate_sim_smiles.append(self.smiles_list[now_idx])
            # sim_scores = compute_similarity_1vn(self.smiles_list[idx], calculate_sim_smiles)
            # sim_scores = torch.tensor(sim_scores, dtype=torch.float64)
            # sampled_rates = torch.softmax(sim_scores, dim=0)
            # choose_idx = torch.multinomial(sampled_rates, self.num_samples)
            # self.calculated_sim_list[idx] = torch.cat([self.calculated_sim_list[idx], calculate_sim_samples], dim=0)
            # self.calculated_sim_score[idx] = torch.cat([self.calculated_sim_score[idx], sim_scores], dim=0)
            if self.extract_sim_type == "sim":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples, ), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]
                # print(choose_idx, keys[choose_idx])
                # print(choose_idx, keys[choose_idx], idx_to_sim_score_dict[keys[choose_idx]])

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                st_time = time.time()
                if self.with_neg:
                    data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                    num_neg_samples = self.k * self.num_samples
                    # stt_len = len(self.idx_to_neg_samples_list[idx])
                    # for i in range(num_neg_samples):
                    #     neg_sample_id = self.idx_to_neg_samples_list[idx][0]
                    #     neg_data = self.get_data_simple(neg_sample_id)
                    #     data.neg_x.append(neg_data.x)
                    #     data.neg_edge_index.append(neg_data.edge_index)
                    #     data.neg_edge_attr.append(neg_data.edge_attr)
                    #     self.idx_to_neg_samples_list[idx].pop(0)
                    # assert stt_len - len(self.idx_to_neg_samples_list[idx]) == num_neg_samples

                    # for neg samples:
                #### on-fly compute negative samples
                    # random_chosen_idx = random.sample(list(set(range(len(self.smiles_list))) - set(keys)),
                    #                                   num_neg_samples * 10)
                    #
                    # all_involved_idx = keys + random_chosen_idx

                    sim_scores_all = torch.zeros((len(self.smiles_list), ), dtype=torch.float64)
                    if len(idx_to_sim_score_dict) > 0:
                        calculated_sim_idx = torch.tensor(keys, dtype=torch.long)
                        calculated_sim_scores = ori_values
                        # sim_scores_all[: len(keys)] = ori_values / 4
                        sim_scores_all[calculated_sim_idx] = calculated_sim_scores / 4
                    normalized_neg_sample_rates = torch.softmax(sim_scores_all, dim=0)
                    neg_samples_idx = torch.multinomial(normalized_neg_sample_rates, num_neg_samples)
                    # data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                    # neg_samples_idx = [all_involved_idx[int(neg_samples_idx[jj])] for jj in range(neg_samples_idx.size(0))]
                    # print(neg_samples_idx)


                    for i in range(num_neg_samples):
                        sample_idx = int(neg_samples_idx[i])
                        # real_idx = int(calculate_sim_samples[sampled_idx])
                        # real_idx = int(keys[sample_idx])
                        neg_data = self.get_data_simple(sample_idx)
                        data.neg_x.append(neg_data.x)
                        data.neg_edge_index.append(neg_data.edge_index)
                        data.neg_edge_attr.append(neg_data.edge_attr)
                # print("time for calculating negative samples = ", time.time() - st_time)
            # sim_data = self.get_data_simple(keys[choose_idx])
            # data.sim_x, data.sim_edge_index, data.sim_edge_attr = sim_data.x, sim_data.edge_index, sim_data.edge_attr

        if self.after_transform is not None:
            return self.after_transform(data, self.smiles_list[idx])
        else:
            return data


    @property
    def raw_file_names(self):
        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS or \
                self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return []
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS or \
                self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')
    def __len__(self):
        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS:
            return len(self.graphs)
        elif self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return self.graphs[0].number_of_nodes()
        else:
            return super(MoleculeDataset, self).__len__()

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset in GCC_GRAPH_CLASSIFICATION_DATASETS or \
                self.dataset in GCC_NODE_CLASSIFICATION_DATASETS:
            return

        if self.dataset == 'zinc_standard_agent':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            for i in range(len(smiles_list)):
                print(i)
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor(
                            [id])  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue

        elif self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            from splitters import scaffold_split

            ### 
            downstream_dir = [
            'dataset/bace',
            'dataset/bbbp',
            'dataset/clintox',
            'dataset/esol',
            'dataset/freesolv',
            'dataset/hiv',
            'dataset/lipophilicity',
            'dataset/muv',
            # 'dataset/pcba/processed/smiles.csv',
            'dataset/sider',
            'dataset/tox21',
            'dataset/toxcast'
            ]

            downstream_inchi_set = set()
            for d_path in downstream_dir:
                print(d_path)
                dataset_name = d_path.split('/')[1]
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)

                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset, downstream_smiles, task_idx=None, null_value=0,
                                   frac_train=0.8,frac_valid=0.1, frac_test=0.1,
                                   return_smiles=True)

                ### remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split('.')
                    for s in species_list:  # record inchi for all species, not just
                     # largest (by default in create_standardized_mol_id if input has
                     # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))

            print('processing')
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #Chem.SanitizeMol(rdkit_mol,
                                 #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'pcba':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'pcba_pretrain':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            downstream_inchi = set(pd.read_csv(os.path.join(self.root,
                                                            'downstream_mol_inchi_may_24_2019'),
                                               sep=',', header=None)[0])
            for i in range(len(smiles_list)):
                print(i)
                if '.' not in smiles_list[i]:   # remove examples with
                    # multiples species
                    rdkit_mol = rdkit_mol_objs[i]
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi:
                            # # convert aromatic bonds to double bonds
                            # Chem.SanitizeMol(rdkit_mol,
                            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        # elif self.dataset == ''

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'ptc_mr':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', header=None, names=['id', 'label', 'smiles'])
            smiles_list = input_df['smiles']
            labels = input_df['label'].values
            for i in range(len(smiles_list)):
                print(i)
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'mutag':
            smiles_path = os.path.join(self.root, 'raw', 'mutag_188_data.can')
            # smiles_path = 'dataset/mutag/raw/mutag_188_data.can'
            labels_path = os.path.join(self.root, 'raw', 'mutag_188_target.txt')
            # labels_path = 'dataset/mutag/raw/mutag_188_target.txt'
            smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
            labels = pd.read_csv(labels_path, header=None)[0].values
            for i in range(len(smiles_list)):
                print(i)
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Invalid dataset name')

        self.total = len(smiles_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain

import networkx as nx
from n2v_graph import Graph
from multiprocessing import Pool
# from compute_similarity impor

class MoleculeDatasetForContrast(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False,
                 extract_sim=False,
                 num_samples=3,
                 num_hops=2,
                 k=7,
                 with_neg=False,
                 pos_p_q=[0.4, 1.6],
                 # neg_p_q=[1.4, 0.6],
                 neg_p_q=[0.2, 1.8],
                 extract_sim_type="precomputed_rwr",
                 T=4,
                 construct_big_graph=False,
                 select_other_node_stra="last_one",
                 restart_prob=0.2,
                 rw_hops=64,
                 num_path=7,
                 args=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDatasetForContrast, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.total = 0
        self.extract_sim = extract_sim
        self.num_samples = num_samples
        self.select_other_node_stra = select_other_node_stra
        self.k = k
        self.with_neg = with_neg
        self.calculated_sim_score = list()
        self.calculated_sim_list = list()
        self.num_calculate_sim_samples = 20
        self.idx_to_adj_list_dict = dict()
        self.restart_prob = restart_prob
        self.num_hops = num_hops
        self.extract_sim_type = extract_sim_type
        self.num_epochs = 100
        self.T = T # set T to 1
        self.pos_p_q = pos_p_q
        self.neg_p_q = neg_p_q
        self.rw_hops = rw_hops
        self.num_path = num_path
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self.dataset == "zinc_standard_agent" and extract_sim:
                # input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
                input_path = root + "/raw/zinc_combined_apr_8_2019.csv.gz"
                input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
                self.smiles_list = list(input_df['smiles'])

                lenn = 2000000
                assert len(self.smiles_list) == lenn

                self.pos_idxes = list()
                self.data_para = True if "data_para" in args and args.data_para == True else False
                # for l in ls:  # not only one positive sample?
                # if args is not None and args.env == "jizhi":
                #     processed_path = "/apdcephfs/private_meowliu/ft_local/mnist_trainer_torch/data/zinc_standard_agent"
                # else:
                #     processed_path = "./dataset/zinc_standard_agent/processed"
                processed_path = root + "/processed"

                # sim to score idx to sim {fri_idx: sim_score} dict

                # TODO: we need to get a pre_computed result which is the sumation
                #  of the similarity between nodes (having edges.)
                #  and assume it is the "self.idx_to_sim_sum" which is a dict.
                #  we need to define a T --- the

                if extract_sim_type != "rwr_hop_pos_neg_on_big_graph_bat" and \
                        extract_sim_type != "biased_rwr_hop_pos_neg_on_big_graph_bat":

                    # pos_idxes is loaded just for positive sampling by similarity?
                    sim_to_score_dict_list = np.load(
                        os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

                    for j in range(len(sim_to_score_dict_list)):
                        assert isinstance(sim_to_score_dict_list[j], dict)
                        self.pos_idxes.append(sim_to_score_dict_list[j])
                    assert len(self.pos_idxes) == lenn

                    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
                    assert isinstance(idx_to_adjs, dict) and len(idx_to_adjs) == lenn
                    # idx to self adjacency nodes --- selected by similarity
                    self.idx_to_adj_list_dict = idx_to_adjs


                if self.extract_sim_type in ["neg_graph_n2v", "neg_graph_n2v_no_graph_pos", "no_neg_pos_graph"]:
                    self.idx_to_unsim_node_list_dict = np.load(os.path.join(processed_path, "unsim_node_adj_list_cutoff_0.1.npy")).item()

                if construct_big_graph:
                    if self.extract_sim_type == "n2v_neg_pos_one_walk_on_big_graph":
                        if not os.path.exists(os.path.join(processed_path, "big_graph.pkl")):
                            self.nxg = nx.Graph()
                            self.nxg.add_nodes_from([inode for inode in range(0, len(self.idx_to_adj_list_dict))])
                            num_cpu = 20
                            num_per_cpu = lenn // num_cpu
                            for ii in range(num_cpu):
                                print(ii)
                                st_node_idx = ii * num_per_cpu
                                tmp_edge_list = np.load(os.path.join(processed_path, "sim_edge_list_{:d}.npy".format(st_node_idx)))
                                print(len(tmp_edge_list))
                                self.nxg.add_weighted_edges_from(tmp_edge_list)
                            print(self.nxg.number_of_edges(), self.nxg.number_of_nodes())
                            with open(os.path.join(processed_path, "big_graph.pkl"), "wb") as f:
                                pickle.dump(self.nxg, f)
                        else:
                            print("going to load pickle!")
                            with open(os.path.join(processed_path, "big_graph.pkl"), "rb") as f:
                                self.nxg = pickle.load(f)
                            print("loaded!")
                            print(self.nxg.number_of_nodes(), self.nxg.number_of_edges())
                    if self.extract_sim_type in ["rwr_hop_pos_neg_on_big_graph",
                                                 "rwr_hop_pos_neg_on_big_graph_no_pos_graph",
                                                 "rwr_pos_big_gra_degree_neg",
                                                 "rwr_hop_pos_neg_on_big_graph_bat",
                                                 "biased_rwr_hop_pos_neg_on_big_graph_bat"]:
                        if (not os.path.exists(os.path.join(processed_path, "big_graph_dgl.pkl"))) and \
                                (not os.path.exists(os.path.join(processed_path, "big_graph_dgl_with_edge_p_normed.pkl"))):
                            self.dgl_big_gra = dgl.from_networkx(self.nxg)
                            with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "wb") as f:
                                pickle.dump(self.dgl_big_gra, f)
                        else:
                            # with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "rb") as f:
                            #     self.dgl_big_gra = pickle.load(f)

                            with open(os.path.join(processed_path, "big_graph_dgl_with_edge_p_normed.pkl"), "rb") as f:
                                self.dgl_big_gra = pickle.load(f)
                            print("dgl big graph loaded!")
                            if self.extract_sim_type == "biased_rwr_hop_pos_neg_on_big_graph_bat":

                                print("going to transfer dgl graph to networkx graph!")
                                self.nxg = dgl.to_networkx(self.dgl_big_gra)
                                print("from dgl graph to networkx graph transferred!")
                                self.biased_nvg = Graph(self.nxg, False, 1.0, neg_p_q[1])
                                print("from nx graph to n2v graph transferred!")
                                self.biased_nvg.preprocess_transition_probs()
                                print("biased random walk possibility transferred!")

                        if os.path.exists(os.path.join(processed_path, "big_graph_degree_array_np_array.npy")):
                            degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"))
                            self.degree_unnormalized_tensor = torch.from_numpy(degree_np_array)
                            print("degree array loaded", self.degree_unnormalized_tensor.size())
                        if os.path.exists(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy")):
                            degree_np_array = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy"))
                            self.degree_unnormalized_tensor = torch.from_numpy(degree_np_array)
                            print("degree array loaded", self.degree_unnormalized_tensor.size())

                        # if not os.path.exists(os.path.join(processed_path, "big_graph_degree_exp_tensor.pkl")):
                        #     print("going to calculate degrees exp")
                        #     degrees = list()
                        #     for i_node in range(self.dgl_big_gra.num_nodes()):
                        #         if i_node % 1000 == 0:
                        #             print(i_node)
                        #         degrees.append(math.exp(self.dgl_big_gra.out_degree(i_node)))
                        #     self.degree_unnormalized_tensor = torch.tensor(degrees, dtype=torch.float64)
                        #     with open(os.path.join(processed_path, "big_graph_degree_exp_tensor.pkl"), "wb") as f:
                        #         pickle.dump(self.degree_unnormalized_tensor, f)
                        # else:
                        #     with open(os.path.join(processed_path, "big_graph_degree_exp_tensor.pkl"), "rb") as f:
                        #         self.degree_unnormalized_tensor = pickle.load(f)
                        #     print("degree_unnormalized_tensor_loaded!")
                        # with open(os.path.join(processed_path, "big_graph.pkl"), "rb") as f:
                        #     self.nxg = pickle.load(f)

                    # if not os.path.exists(os.path.join(processed_path, "big_graph_n2v_graph_p_{}_q_{}.pkl".format(str(neg_p_q[0]), str(neg_p_q[1])))):
                    #     self.big_graph_n2v_graph = Graph(self.nxg, False, neg_p_q[0], neg_p_q[1])
                    #     print("not exist, going to process transition probs")

                    #     self.big_graph_n2v_graph.preprocess_transition_probs()
                    #     print("processed going to save n2v graph")
                    #     with open(os.path.join(processed_path, "big_graph_n2v_graph_p_{}_q_{}.pkl".format(str(neg_p_q[0]), str(neg_p_q[1]))), "wb") as f:
                    #         pickle.dump(self.big_graph_n2v_graph, f)
                    # else:
                    #     with open(os.path.join(processed_path, "big_graph_n2v_graph_p_{}_q_{}.pkl".format(str(neg_p_q[0]), str(neg_p_q[1]))), "rb") as f:
                    #         self.big_graph_n2v_graph = pickle.load(f)
                    #     print("n2v graph loaded!")


                    # if not os.path.exists(os.path.join(processed_path, "sim_edge_list.npy")):
                    #     print("calculating edge list")
                    #     edge_list_with_weight = []
                    #     for st_node in self.idx_to_adj_list_dict:
                    #         print(st_node, len(self.idx_to_adj_list_dict[st_node]))
                    #         edge_list_with_weight.extend([(st_node, ed_node, self.pos_idxes[st_node][ed_node])
                    #         for ed_node in self.idx_to_adj_list_dict[st_node]])
                    #     print("calculated")
                    #     self.nxg.add_weighted_edges_from(edge_list_with_weight)
                    #     np.save(os.path.join(processed_path, "sim_edge_list.npy"), edge_list_with_weight)
                    #     print(len(edge_list_with_weight))
                    # else:
                    #     edge_list_with_weight = np.load(os.path.join(processed_path, "sim_edge_list.npy")).item()
                    #     self.nxg.add_weighted_edges_from(edge_list_with_weight)

                if self.extract_sim_type in ["rwr_neg", "pos_rwr", "rwr_hop_pos_neg", "rwr_hop_sample_only_neg"] :
                    # self.idx_to_sim_sum = np.load(os.path.join(processed_path, "idx_to_sim_sum.npy")).item()
                    if not os.path.exists(os.path.join(processed_path, "idx_to_sim_sum_T_{:d}.npy".format(self.T))):
                        print("calculating idx to sum with T = {:d}.".format(self.T))
                        idx_to_normalized_factor_dict = dict()
                        for now_idx in idx_to_adjs:
                            cumsum_normalized_factor = 0.0
                            for other_idx in self.idx_to_adj_list_dict[now_idx]:
                                cumsum_normalized_factor += math.exp(self.pos_idxes[now_idx][other_idx] / self.T)
                            idx_to_normalized_factor_dict[now_idx] = cumsum_normalized_factor
                        np.save(os.path.join(processed_path, "idx_to_sim_sum_T_{:d}.npy".format(self.T)), idx_to_normalized_factor_dict)
                        print(len(idx_to_normalized_factor_dict))
                        assert len(idx_to_normalized_factor_dict) == lenn
                        self.idx_to_sim_sum = idx_to_normalized_factor_dict
                        print("calculated!")
                    else:
                        self.idx_to_sim_sum = np.load(os.path.join(processed_path, "idx_to_sim_sum_T_{:d}.npy".format(self.T))).item()

                if self.extract_sim_type in ["precomputed_neg"]:
                    unsim_to_score_dict_list = np.load(os.path.join(processed_path, "unsim_to_score_dict_list_wei_nc_natom.npy"))
                    self.neg_idxes = list()
                    assert len(unsim_to_score_dict_list) == lenn
                    for j in range(len(unsim_to_score_dict_list)):
                        assert isinstance(unsim_to_score_dict_list[j], dict)
                        self.neg_idxes.append(unsim_to_score_dict_list[j])
                    assert len(self.neg_idxes) == lenn
                if self.extract_sim_type == "no_neg_pos_graph":
                    self.idx_to_unsim_to_score_dict_list = np.load(os.path.join(processed_path, "unsim_to_score_list_dict.npy"))
                    print("idx_to_unsim_score_dict_list loaded!", len(self.idx_to_unsim_to_score_dict_list))
                # if self.extract_sim_type == "sim" and self.with_neg:
                #     per_neg_num_samples = self.k * self.num_samples * self.num_epochs
                #     self.idx_to_neg_samples_list = dict()
                #     print("Calculating negative samples.")
                #     for j in range(lenn):
                #         print(j)
                #         if j % 1000 == 0:
                #             print(j)
                #         idx_to_sim_score_dict = self.pos_idxes[j]
                #         keys = list()
                #         values = list()
                #         for idxbb in idx_to_sim_score_dict:
                #             keys.append(idxbb)
                #             values.append(idx_to_sim_score_dict[idxbb])
                #         sim_scores_all = torch.zeros((len(self.smiles_list),), dtype=torch.float64)
                #         if len(keys) > 0:
                #             calculated_sim_idx = torch.tensor(keys, dtype=torch.long)
                #             calculated_sim_scores = torch.tensor(values, dtype=torch.float64)
                #             sim_scores_all[calculated_sim_idx] = calculated_sim_scores / 4
                #         normalized_neg_sample_rates = torch.softmax(sim_scores_all, dim=0)
                #         neg_samples_idx = torch.multinomial(normalized_neg_sample_rates, per_neg_num_samples)
                #         self.idx_to_neg_samples_list[j] = [int(neg_samples_idx[neg_sample_idx]) for neg_sample_idx in range(neg_samples_idx.size(0))]

                if self.extract_sim_type == "precomputed_rwr":
                    self.idx_to_sampled_pos_idx = dict()
                    for j in range(lenn):
                        # print(j)
                        if j % 1000 == 0:
                            print(j)
                        if len(self.idx_to_adj_list_dict[j]) == 0:
                            self.idx_to_sampled_pos_idx[j] = [j for _ in range(self.num_samples * self.num_epochs)]
                            continue
                        # print(len(self.idx_to_adj_list_dict[j]))
                        r_ego_graph, new_idx_to_old_idx, _ = self.get_rhop_ego_graph(j, num_hops=self.num_hops + 1)
                        # print(r_ego_graph)
                        sampled_idx = self.rwr_sample_pos_graphs(r_ego_graph, 0, num_samples=self.num_samples * self.num_epochs * 2)
                        # print(sampled_idx)
                        sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                        sampled_num_idx = self.num_samples * self.num_epochs
                        if len(sampled_idx) > sampled_num_idx:
                            sampled_idx = sampled_idx[: sampled_num_idx]
                        elif len(sampled_idx) < sampled_num_idx:
                            sampled_idx += [j for _ in range(sampled_num_idx - len(sampled_idx))]
                        self.idx_to_sampled_pos_idx[j] = sampled_idx

                if self.extract_sim_type == "precomputed_n2v_walk_pos_neg":
                    num_pos_samples_10_epochs = self.num_samples * 10
                    num_neg_samples_10_epochs = self.num_samples * self.k * 10
                    print("getting positive and negative samples for 10 epochs")
                    # self.get_n2v_pos_neg_samples(num_pos_samples_10_epochs, num_neg_samples_10_epochs)
                    num_workers = 20
                    # pool = Pool(20)
                    num_per_pool = len(self.smiles_list) // num_workers
                    if num_workers * num_per_pool < len(self.smiles_list):
                        num_workers += 1
                    pool = Pool(num_workers)
                    # if num_per_pool * 20 < len(self.
                    # smiles_list):
                    #     num_per_pool += 1
                    res = []
                    for j in range(num_workers):
                        st, ed = j * num_per_pool, min((j + 1) * num_per_pool, len(self.smiles_list))
                        res.append(pool.apply_async(self.get_n2v_pos_neg_samples,
                                                    args=(st, ed, num_pos_samples_10_epochs, num_neg_samples_10_epochs)))
                    pool.close()
                    pool.join()
                    self.n2v_idx_to_pos_samples = dict()
                    self.n2v_idx_to_neg_samples = dict()
                    for r in res:
                        part_ans = r.get()
                        for idx in part_ans[0]:
                            self.n2v_idx_to_pos_samples[idx] = part_ans[0][idx]
                            self.n2v_idx_to_neg_samples[idx] = part_ans[1][idx]


    # todo: is multi-processing possible?
    def get_n2v_pos_neg_samples(self, st, ed, num_pos_samples, num_neg_samples):
        # self.n2v_idx_to_pos_samples = dict()
        # self.n2v_idx_to_neg_samples = dict()
        part_n2v_idx_to_pos_samples = dict()
        part_n2v_idx_to_neg_samples = dict()
        # for idx in range(len(self.smiles_list)):
        for idx in range(st, ed):
            if idx % 1000 == 0:
                print("getting samples for {:d} th node.".format(idx))
            if len(self.idx_to_adj_list_dict[idx]) > 0:
                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                pos_n2v_G = Graph(r_hop_nx_G, False, self.pos_p_q[0], self.pos_p_q[1])
                neg_n2v_G = Graph(r_hop_nx_G, False, self.neg_p_q[0], self.neg_p_q[1])
                pos_n2v_G.preprocess_transition_probs()
                neg_n2v_G.preprocess_transition_probs()
                pos_walk_nodes = pos_n2v_G.node2vec_walk(num_pos_samples * 7, 0)
                neg_walk_nodes = neg_n2v_G.node2vec_walk(num_neg_samples * 7, 0)
                ## todo: start the biased random walk from the root node only once; --- we should try to start from this node several times!!!
                pos_walk_nodes_dict = {nei_idx: nei_pos for (nei_pos, nei_idx) in enumerate(reversed(pos_walk_nodes))}
                neg_walk_nodes_dict = {nei_idx: nei_pos for (nei_pos, nei_idx) in enumerate(reversed(neg_walk_nodes))}

                pos_walk_nodes_dict_items_sorted = sorted(pos_walk_nodes_dict.items(), key=lambda ii: ii[1], reverse=True)
                neg_walk_nodes_dict_items_sorted = sorted(neg_walk_nodes_dict.items(), key=lambda ii: ii[1], reverse=True)
                pos_walk_unique_nodes = [pos_item[0] for pos_item in pos_walk_nodes_dict_items_sorted]
                neg_walk_unique_nodes = [neg_item[0] for neg_item in neg_walk_nodes_dict_items_sorted]
                # print("pos_walk_unique_nodes = ", len(pos_walk_unique_nodes), "neg_walk_unique_nodes = ", len(neg_walk_unique_nodes))
                try:
                    pos_walk_unique_nodes.remove(0)
                except ValueError:
                    pass
                try:
                    neg_walk_unique_nodes.remove(0)
                except ValueError:
                    pass
                pos_sampled_idx = [new_idx_to_old_idx[pos_idx] for pos_idx in pos_walk_unique_nodes]
                neg_sampled_idx = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
                # num_pos_samples = selfnum_samples
                # num_neg_samples = self.k * self.num_samples
                if idx % 1000 == 0:
                    print("numpos", num_pos_samples, num_neg_samples, len(pos_sampled_idx), len(neg_sampled_idx)) # only for debug
                if len(pos_sampled_idx) < num_pos_samples:
                    pos_sampled_idx = pos_sampled_idx + [idx for _ in range(num_pos_samples - len(pos_sampled_idx))]
                elif len(pos_sampled_idx) > num_pos_samples:
                    pos_sampled_idx = pos_sampled_idx[: num_pos_samples]

                if len(neg_sampled_idx) < num_neg_samples:
                    neg_sampled_idx = neg_sampled_idx + [(idx + _) % len(self.smiles_list) for _ in
                                                         range(num_neg_samples - len(neg_sampled_idx))]
                elif len(neg_sampled_idx) > num_neg_samples:
                    neg_sampled_idx = neg_sampled_idx[-num_neg_samples:]

            else:
                # sampled_idx = [idx for _ in range(self.num_samples)]
                pos_sampled_idx = [idx for _ in range(self.num_samples)]
                neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(self.num_samples * self.k)]
            part_n2v_idx_to_pos_samples[idx] = pos_sampled_idx
            part_n2v_idx_to_neg_samples[idx] = neg_sampled_idx
        return [part_n2v_idx_to_pos_samples, part_n2v_idx_to_neg_samples]

    def get_n2v_pos_neg_samples_big_nvg(self, idx, num_samples, num_paths):
        node_to_cnt = dict()
        for i in range(num_paths):
            walk = self.biased_nvg.node2vec_walk(num_samples, idx)
            for nod in walk:
                if nod not in node_to_cnt:
                    node_to_cnt[nod] = 1
                else:
                    node_to_cnt[nod] += 1
        # sorted_items = sorted(node_to_cnt, key=lambda i: i[1], reverse=True)
        return node_to_cnt
    # num_nhops = 3... ?
    def get_rhop_ego_graph(self, idx, num_hops=3, neg_sampling=False, pos_sampling=True): # assume "T" is equal to 4 and self.idx_to_sim_sum[idx] = sum(exp(sim_score / T))
        idx_to_dis = dict()
        que = [idx]
        idx_to_dis[idx] = 0
        edge_fr, edge_to = list(), list()
        edge_sample_prob = list()
        # pos_sample_prob = list()
        pos_edge_sample_prob = list()
        new_idx = 0

        old_to_new_idx = {idx: new_idx}
        T = self.T
        while len(que) > 0:
            now_idx = que[-1]
            que.pop()
            new_now_idx = old_to_new_idx[now_idx]
            if idx_to_dis[now_idx] >= num_hops:
                break
            for other_idx in self.idx_to_adj_list_dict[now_idx]:
                if other_idx not in idx_to_dis:
                    new_idx += 1
                    old_to_new_idx[other_idx] = new_idx
                    idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                    edge_fr.append(new_now_idx)
                    edge_to.append(new_idx)
                    edge_fr.append(new_idx)
                    edge_to.append(new_now_idx)
                    if neg_sampling:
                        ori_sim_score = self.pos_idxes[now_idx][other_idx]
                        div_sim_score = ori_sim_score / T
                        neg_sample_prob = math.exp(div_sim_score) / self.idx_to_sim_sum[now_idx]
                        edge_sample_prob.append(neg_sample_prob)
                        edge_sample_prob.append(neg_sample_prob)
                    if pos_sampling:
                        assert self.T == 1 # need sim score to be equal to 1
                        ori_sim_score = self.pos_idxes[now_idx][other_idx]
                        # div_sim_score = ori_sim_score / T
                        pos_sample_prob = math.exp(ori_sim_score) / self.idx_to_sim_sum[now_idx]
                        pos_edge_sample_prob.append(pos_sample_prob)
                        pos_edge_sample_prob.append(pos_sample_prob)
                    que.insert(0, other_idx)
        edge_fr, edge_to = torch.tensor(edge_fr, dtype=torch.long), torch.tensor(edge_to, dtype=torch.long)

        dgl_g = dgl.DGLGraph()
        dgl_g.add_nodes(new_idx + 1)
        dgl_g.add_edges(edge_fr, edge_to)
        # dgl_g = dgl.graph(edge_fr, edge_to)

        if neg_sampling:
            edge_sample_prob = torch.tensor(edge_sample_prob, dtype=torch.float64)
            dgl_g.edata["neg_sample_p"] = edge_sample_prob
        if pos_sampling:
            pos_edge_sample_prob = torch.tensor(pos_edge_sample_prob, dtype=torch.float64)
            dgl_g.edata["pos_sample_p"] = pos_edge_sample_prob
            ### if we cannot sample enough negative samples -- just try to add the next moleculer to the negative samples list?
        # dgl_g = dgl.graph((edge_fr, edge_to))
        # print(dgl_g.nodes(), dgl_g.edges())
        # print(dgl_g)
        new_idx_to_old_idx = {old_to_new_idx[i]: i for i in old_to_new_idx}
        # bg = dgl.to_bidirected(dgl_g)
        return dgl_g, new_idx_to_old_idx, idx_to_dis

    def get_rhop_ego_nx_graph(self, idx, num_hops=3, max_nodes_num=100): # assume "T" is equal to 4 and self.idx_to_sim_sum[idx] = sum(exp(sim_score / T))
        ## remember that the idx of the root node to be sampled from is always equal to zero!!!!!!
        idx_to_dis = dict()
        que = [idx]
        idx_to_dis[idx] = 0
        new_idx = 0
        edge_list_with_weight = list()

        old_to_new_idx = {idx: new_idx}
        # T = self.T
        while len(que) > 0:
            now_idx = que[-1]
            que.pop()
            new_now_idx = old_to_new_idx[now_idx]
            if idx_to_dis[now_idx] >= num_hops:
                break
            for other_idx in self.idx_to_adj_list_dict[now_idx]:
                if other_idx not in idx_to_dis:
                    new_idx += 1
                    # map other_idx (old idx) to new_idx (new idx)
                    old_to_new_idx[other_idx] = new_idx
                    # set the distance of newly added node
                    idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                    edge_list_with_weight.append((new_now_idx, new_idx, self.pos_idxes[now_idx][other_idx]))
                    que.insert(0, other_idx)
                    if (new_idx + 1) >= max_nodes_num:
                        break
            if (new_idx + 1) >= max_nodes_num:
                break
        nx_G = nx.Graph()
        nx_G.add_nodes_from([_ for _ in range(0, new_idx + 1)])
        nx_G.add_weighted_edges_from(edge_list_with_weight)

        new_idx_to_old_idx = {old_to_new_idx[i]: i for i in old_to_new_idx}
        # bg = dgl.to_bidirected(dgl_g)
        return nx_G, new_idx_to_old_idx, idx_to_dis

    def get_rhop_neg_ego_nx_graph(self, idx, num_hops=3, max_nodes_num=100): # assume "T" is equal to 4 and self.idx_to_sim_sum[idx] = sum(exp(sim_score / T))
        ## remember that the idx of the root node to be sampled from is always equal to zero!!!!!!
        idx_to_dis = dict()
        que = [idx]
        idx_to_dis[idx] = 0
        new_idx = 0
        edge_list_with_weight = list()

        old_to_new_idx = {idx: new_idx}
        # T = self.T
        while len(que) > 0:
            now_idx = que[-1]
            que.pop()
            new_now_idx = old_to_new_idx[now_idx]
            if idx_to_dis[now_idx] >= num_hops:
                break
            for other_idx in self.idx_to_unsim_node_list_dict[now_idx]:
                if other_idx not in idx_to_dis:
                    new_idx += 1
                    # map other_idx (old idx) to new_idx (new idx)
                    old_to_new_idx[other_idx] = new_idx
                    # set the distance of newly added node
                    idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                    edge_list_with_weight.append((new_now_idx, new_idx, 1.))
                    que.insert(0, other_idx)
                    if (new_idx + 1) >= max_nodes_num:
                        break
            if (new_idx + 1) >= max_nodes_num:
                break
        nx_G = nx.Graph()
        nx_G.add_nodes_from([_ for _ in range(0, new_idx + 1)])
        nx_G.add_weighted_edges_from(edge_list_with_weight)

        new_idx_to_old_idx = {old_to_new_idx[i]: i for i in old_to_new_idx}
        # bg = dgl.to_bidirected(dgl_g)
        return nx_G, new_idx_to_old_idx, idx_to_dis

    def rwr_sample_pos_graphs(self, dgl_bg, idx, num_samples=5):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(9)],
            # prob="pos_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        # todo: count the frequency and choose top k ones?
        subv = torch.unique(traces).tolist()
        # print("calculated...", subv)
        try:
            subv.remove(idx)
        except:
            pass
        try:
            subv.remove(-1)
        except:
            pass
        return subv

    def rwr_sample_pos_graphs_path_dis(self, dgl_bg, idx, num_samples=5, num_path=7):
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(num_path)],
            prob="pos_sample_p",
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

    def rwr_sample_neg_graphs(self, dgl_bg, idx, num_samples=5, exc_idx=None):
        # print(type(dgl_bg))
        print("calculating rwr negative samples....")
        traces, _ = dgl.sampling.random_walk(
            dgl_bg,
            [idx for __ in range(18)],
            prob="neg_sample_p",
            restart_prob=self.restart_prob,
            length=num_samples)
        subv = torch.unique(traces).tolist()

        # newsubv = list()
        # for ii in subv:
        #     if ii >= 0 and ii != idx:
        #         newsubv.append(ii)
        try:
            subv.remove(idx)
        except ValueError:
            pass
        try:
            subv.remove(-1)
        except ValueError:
            pass
        # if exc_idx is not None:
        #     try:
        #         subv.remove(exc_idx)
        #     except ValueError:
        #         pass
        # return subv
        # print(len(subv))
        return subv

    def get_data_simple(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    def get(self, idx):
        # print(idx)
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        # each epoc ?
        if self.dataset == "zinc_standard_agent" and self.extract_sim:
            # num_all_data = len(self.smiles_list)
            if self.extract_sim_type == "no_pos_graph_seq_neg":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                num_neg_samples = self.num_samples * self.k

                seq_sample_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, num_neg_samples + 1)]
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for i in range(num_neg_samples):
                    neg_data = self.get_data_simple(seq_sample_idx[i])
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "no_neg_pos_graph":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                # for neg samples

                idx_to_sim_score_dict = self.idx_to_unsim_to_score_dict_list[idx]
                # keys = list()
                keys = self.idx_to_unsim_node_list_dict[idx]
                values = list()
                # for idxbb in idx_to_sim_score_dict:
                #     keys.append(idxbb)
                #     values.append(idx_to_sim_score_dict[idxbb])
                for idxbb in keys:
                    values.append(idx_to_sim_score_dict[idxbb])

                num_neg_samples = self.num_samples * self.k
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    values = values.max() + values.min() - values # reverse the similarity
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = num_neg_samples if num_neg_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < num_neg_samples:
                        # keys.append(idx)
                        keys = keys + [(idx + _) % len(self.smiles_list) for _ in range(1, num_neg_samples - num_samples + 1)]
                        # choose_idx = torch.cat([choose_idx, torch.full((num_neg_samples - num_samples,), len(keys) - 1,
                        #                                                dtype=torch.long)], dim=0)
                        choose_idx = torch.cat([choose_idx, torch.arange(num_samples, num_neg_samples)], dim=0)
                else:
                    choose_idx = [j for j in range(num_neg_samples)]
                    keys = [(idx + j) % len(self.smiles_list) for j in range(1, num_neg_samples + 1)]

                # sim_datas = list()
                # neg_samples_idx = [int(keys[int(choose_idx[i])]) for i in range(num_neg_samples)]
                # print("neg samples idx = ", idx, neg_samples_idx)
                # neg_samples_sim_scores = compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[neg_idx] \
                #                                                                         for neg_idx in neg_samples_idx])
                # seq_samples_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, num_neg_samples + 1)]
                # seq_samples_sim_scores = compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[neg_idx] \
                #                                                                         for neg_idx in seq_samples_idx])
                # print("seq_samples_idx = ", seq_samples_idx)
                # print("neg_samples_sim = ", neg_samples_sim_scores)
                # print("seq_samples_sim = ", seq_samples_sim_scores)
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for i in range(num_neg_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    neg_data = self.get_data_simple(real_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "neg_graph_n2v_no_graph_pos":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    # ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)
                r_hop_nx_G_neg, new_idx_to_old_idx_neg, idx_to_dis_neg = self.get_rhop_neg_ego_nx_graph(idx,
                                                                                                        num_hops=self.num_hops)
                neg_n2v_G = Graph(r_hop_nx_G_neg, False, self.neg_p_q[0], self.neg_p_q[1])
                neg_n2v_G.preprocess_transition_probs()
                neg_walk_nodes = neg_n2v_G.node2vec_walk(self.num_samples * self.k * 2, 0)
                neg_walk_unique_nodes = list(set(neg_walk_nodes))
                neg_walk_unique_nodes = [new_idx_to_old_idx_neg[neg_idx] for neg_idx in neg_walk_unique_nodes if
                                         new_idx_to_old_idx_neg[neg_idx] not in idx_to_sim_score_dict]
                try:
                    neg_walk_unique_nodes.remove(idx)
                except ValueError:
                    pass
                # print(len(neg_walk_unique_nodes))
                if len(neg_walk_unique_nodes) < self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % len(self.smiles_list) for _ in range(1,
                        self.num_samples * self.k - len(neg_walk_unique_nodes) + 1)]
                elif len(neg_walk_unique_nodes) > self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes[-self.num_samples * self.k:]
                # print("neg samples idx = ", neg_walk_unique_nodes)
                # sampled_nodes_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                                   [self.smiles_list[neg_idx] for neg_idx in neg_walk_unique_nodes])
                # print("n2v_neg_gra_sampled_idx = ", idx, neg_walk_unique_nodes)
                # print("n2v_neg_gra_sampled_sim_scores = ", sampled_nodes_sim_scores)
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_walk_unique_nodes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "neg_graph_n2v":
                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                r_hop_nx_G_neg, new_idx_to_old_idx_neg, idx_to_dis_neg = self.get_rhop_neg_ego_nx_graph(idx, num_hops=self.num_hops)
                pos_n2v_G = Graph(r_hop_nx_G, False, self.pos_p_q[0], self.pos_p_q[1])
                neg_n2v_G = Graph(r_hop_nx_G_neg, False, self.neg_p_q[0], self.neg_p_q[1])
                pos_n2v_G.preprocess_transition_probs()
                neg_n2v_G.preprocess_transition_probs()
                pos_walk_nodes = pos_n2v_G.node2vec_walk(self.num_samples * 7, 0)
                neg_walk_nodes = neg_n2v_G.node2vec_walk(self.num_samples * self.k * 7, 0)
                pos_walk_unique_nodes = list(set(pos_walk_nodes))
                neg_walk_unique_nodes = list(set(neg_walk_nodes))

                try:
                    pos_walk_unique_nodes.remove(0)
                except ValueError:
                    pass
                pos_walk_unique_nodes = [new_idx_to_old_idx[pos_idx] for pos_idx in pos_walk_unique_nodes]
                neg_walk_unique_nodes = [new_idx_to_old_idx_neg[neg_idx] for neg_idx \
                                         in neg_walk_unique_nodes if new_idx_to_old_idx_neg[neg_idx] not in idx_to_dis]
                # print(len(pos_walk_unique_nodes), len(neg_walk_unique_nodes))
                if len(pos_walk_unique_nodes) < self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes + [idx for _ in range(self.num_samples - len(pos_walk_unique_nodes))]
                elif len(pos_walk_unique_nodes) > self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes[: self.num_samples]
                if len(neg_walk_unique_nodes) < self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % len(self.smiles_list) for _ in range(
                        self.num_samples * self.k - len(neg_walk_unique_nodes))]
                elif len(neg_walk_unique_nodes) > self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes[-self.num_samples * self.k:]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_walk_unique_nodes:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_walk_unique_nodes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_hop_pos_neg":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx, old_idx_to_dis = self.get_rhop_ego_graph(idx,
                                                                                         num_hops=self.num_hops,
                                                                                         pos_sampling=True,
                                                                                         neg_sampling=False)
                    sampled_idx = self.rwr_sample_pos_graphs(dgl_bg, 0, num_samples=(self.num_samples * (self.k + 1)) * 2)
                    # convert smapled idx from new idx to old idx
                    sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                    # sort the sampled nodes by their distance from the source node;
                    # then i is the old_idx of each sampled node
                    sorted_sampled_idx = sorted(sampled_idx, key=lambda i: old_idx_to_dis[i])
                    pos_idx_offset = int(len(sorted_sampled_idx) / 8)
                    pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                    neg_sampled_idx = sorted_sampled_idx[pos_idx_offset: ]
                    num_pos_samples = self.num_samples
                    num_neg_samples = self.k * self.num_samples
                    # print("numpos", num_pos_samples, num_neg_samples) # only for debug
                    if len(pos_sampled_idx) < num_pos_samples:
                        pos_sampled_idx = pos_sampled_idx + [idx for _ in range(num_pos_samples - len(pos_sampled_idx))]
                    elif len(pos_sampled_idx) > num_pos_samples:
                        pos_sampled_idx = pos_sampled_idx[: num_pos_samples]

                    if len(neg_sampled_idx) < num_neg_samples:
                        neg_sampled_idx = neg_sampled_idx + [(idx + _) % len(self.smiles_list) for _ in range(1, num_neg_samples - len(neg_sampled_idx) + 1)]
                    elif len(neg_sampled_idx) > num_neg_samples:
                        neg_sampled_idx = neg_sampled_idx[-num_neg_samples: ]

                else:
                    # sampled_idx = [idx for _ in range(self.num_samples)]
                    pos_sampled_idx = [idx for _ in range(self.num_samples)]
                    neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, self.num_samples * self.k + 1)]

                # print(len(pos_sampled_idx), len(neg_sampled_idx))
                # print("pos", idx, pos_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx]))
                # print("neg", idx, neg_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in
                #                                                            neg_sampled_idx]))
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_sampled_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_sampled_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_hop_pos_neg_on_big_graph":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    # dgl_bg, new_idx_to_old_idx, old_idx_to_dis = self.get_rhop_ego_graph(idx,
                    #                                                                      num_hops=self.num_hops,
                    #                                                                      pos_sampling=True,
                    #                                                                      neg_sampling=False)
                    # rw_hops = 64
                    divv = 0.2
                    # max_nodes_per_seed = max(self.rw_hops,
                    #                          int((self.dgl_big_gra.out_degree(idx) * math.e / (
                    #                                      math.e - 1) / (self.restart_prob + 0.2)) + 0.5)
                    #                          )
                    max_nodes_per_seed = self.rw_hops
                    # num_path = max(7, int(self.dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                    # num_samples = self.rw_hops
                    # print("max_nodes_per_seed", max_nodes_per_seed) # for debug only
                    sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                        self.dgl_big_gra, idx, num_samples=max_nodes_per_seed)
                    # sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                    #     self.dgl_big_gra, idx, num_samples=num_samples, num_path=num_path)
                    # convert smapled idx from new idx to old idx
                    # sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                    # sort the sampled nodes by their distance from the source node;
                    # then i is the old_idx of each sampled node samidx sampled idx
                    # sorted_sampled_idx = sorted(sampled_idx, key=lambda i: old_idx_to_dis[i])
                    sampled_nodes = list(sampled_idx_to_mean_dis.keys())
                    # sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])
                    sorted_sampled_idx = sorted(sampled_nodes, key=lambda idxx: sampled_idx_to_count[idxx], reverse=True)
                    # print("all sampled nodes idxes", len(sorted_sampled_idx))
                    # node_idx_to_dis = {sam_idx: nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in sampled_idx}
                    # sorted_sampled_idx = sampled_idx
                    # sorted_sampled_idx = sorted(sampled_idx, key=lambda i: node_idx_to_dis[i])
                    # print("nodes sampled in total:", len(sorted_sampled_idx))
                    # pos_candi = sorted_sampled_idx[: int(len(sorted_sampled_idx) // 7 + 1)]
                    # pos_sampled_prob = [1.0 + (-0.9) * (x * x / ((len(pos_candi) - 1) * (len(pos_candi) - 1))) \
                    #                     for x in range(len(pos_candi))]
                    # neg_candi = sorted_sampled_idx[int(len(sorted_sampled_idx) // 8 + 1): ]
                    # neg_sampled_prob = [1.0 + (-0.9) * ((len(neg_candi) - 1 - x) * (len(neg_candi) - 1 - x) / ((len(neg_candi) - 1) * (len(neg_candi) - 1))) \
                    #                     for x in range(len(neg_candi))]
                    # pos_sampled_prob_tensor = torch.tensor(pos_sampled_prob, dtype=torch.float64) / sum(pos_sampled_prob)
                    # neg_sampled_prob_tensor = torch.tensor(neg_sampled_prob, dtype=torch.float64) / sum(neg_sampled_prob)
                    # pos_sampled_idxs = torch.multinomial(pos_sampled_prob_tensor,
                    #                                      replacement=True, num_samples=self.num_samples)
                    # neg_sampled_idxs = torch.multinomial(neg_sampled_prob_tensor,
                    #                                      replacement=True, num_samples=self.num_samples * self.k)
                    # pos_sampled_idx = [pos_candi[int(pos_sampled_idxs[ipos])] for ipos in range(pos_sampled_idxs.size(0))]
                    # neg_sampled_idx = [neg_candi[int(neg_sampled_idxs[ineg])] for ineg in range(neg_sampled_idxs.size(0))]

                    # self.k = 7

                    num_pos_samples = self.num_samples
                    num_neg_samples = self.k * self.num_samples
                    pos_idx_offset = int(len(sorted_sampled_idx) / (self.k + 1) + 1)
                    #

                    #
                    # pos_idx_offset = min(pos_idx_offset, self.num_samples + 2)
                    # neg_idx_offset = max(pos_idx_offset, len(sorted_sampled_idx) - num_neg_samples - 2)
                    #
                    # print("all sampled nodes idxes", len(sorted_sampled_idx), pos_idx_offset)
                    #
                    pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                    neg_sampled_idx = sorted_sampled_idx[pos_idx_offset: ]
                    # neg_sampled_idx = sorted_sampled_idx[neg_idx_offset: ]


                    # num_pos_samples = self.num_samples
                    # num_neg_samples = self.k * self.num_samples

                    importance_for_div = float(self.num_samples)

                    if len(pos_sampled_idx) == 0:
                        pos_sampled_idx = [idx for _ in range(num_pos_samples)]
                    else:
                        importance_for_div = max(1.0, float(self.num_samples) / float(len(pos_sampled_idx)))
                        pos_sampled_prob = [1.0 + (-0.9) * (x * x / ((len(pos_sampled_idx)) * (len(pos_sampled_idx)))) \
                                            for x in range(1, len(pos_sampled_idx) + 1)] # generates possibilities to be sampled for each candidate
                        # uniform distribution
                        # pos_sampled_prob = [1.0 / len(pos_sampled_idx) for __ in range(len(pos_sampled_idx))]
                        pos_sampled_prob_tensor = torch.tensor(pos_sampled_prob, dtype=torch.float64) / sum(
                            pos_sampled_prob)
                        pos_sampled_idxs = torch.multinomial(pos_sampled_prob_tensor,
                                                             replacement=True, num_samples=self.num_samples)
                        pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[ipos])] for ipos in
                                           range(pos_sampled_idxs.size(0))]

                    if len(neg_sampled_idx) == 0:
                        neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, 1 + num_neg_samples)]
                    else:
                        len_neg = len(neg_sampled_idx)
                        neg_sampled_prob = [1.0 + (-0.9) * ((len_neg - x) * (len_neg - x) / ((len_neg) * (len_neg))) \
                                                    for x in range(1, len_neg + 1)]
                        # uniform distribution
                        # neg_sampled_prob = [1.0 / len_neg for __ in range(len_neg)]
                        neg_sampled_prob_tensor = torch.tensor(neg_sampled_prob, dtype=torch.float64) / sum(neg_sampled_prob)
                        neg_sampled_idxs = torch.multinomial(neg_sampled_prob_tensor,
                                                             replacement=True, num_samples=num_neg_samples)
                        neg_sampled_idx = [neg_sampled_idx[int(neg_sampled_idxs[ineg])] for ineg in
                                           range(neg_sampled_idxs.size(0))]

                    # print("numpos", num_pos_samples, num_neg_samples) # only for debug
                    # print(len(pos_sampled_idx), len(neg_sampled_idx))
                    # if len(pos_sampled_idx) < num_pos_samples:
                    #     pos_sampled_idx = pos_sampled_idx + [idx for _ in range(num_pos_samples - len(pos_sampled_idx))]
                    # elif len(pos_sampled_idx) > num_pos_samples:
                    #     pos_sampled_idx = pos_sampled_idx[: num_pos_samples]
                    #
                    # if len(neg_sampled_idx) < num_neg_samples:
                    #     neg_sampled_idx = neg_sampled_idx + [(idx + _) % len(self.smiles_list) for _ in range(1, num_neg_samples - len(neg_sampled_idx) + 1)]
                    # elif len(neg_sampled_idx) > num_neg_samples:
                    #     neg_sampled_idx = neg_sampled_idx[-num_neg_samples: ]

                else:
                    # sampled_idx = [idx for _ in range(self.num_samples)]
                    importance_for_div = float(self.num_samples)
                    pos_sampled_idx = [idx for _ in range(self.num_samples)]
                    neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, self.num_samples * self.k + 1)]

                # print("importance_for_div", idx, importance_for_div)
                #
                # pos_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] \
                #                                                            for pos_idx in pos_sampled_idx])
                # pos_sim_path_length = [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in pos_sampled_idx]
                # print("pos_sam_infos", idx, [(pos_sampled_idx[jj], pos_sim_scores[jj], pos_sim_path_length[jj]) \
                #                              for jj in range(len(pos_sampled_idx))])
                #
                # neg_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # neg_sim_path_length = [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in neg_sampled_idx]
                # print("pos_sam_infos", idx, [(neg_sampled_idx[jj], neg_sim_scores[jj], neg_sim_path_length[jj]) \
                #                              for jj in range(len(neg_sampled_idx))])
                #
                # print("pos", idx, pos_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx]),
                #       )
                # print("neg", idx, neg_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in
                #                                                            neg_sampled_idx]),
                #       [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in neg_sampled_idxs])

                # pos_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx])
                # print("pos_sam_infos", idx, [(pos_sampled_idx[jj], pos_sim_scores[jj]) for \
                #                              jj in range(len(pos_sampled_idx))])
                #
                # neg_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # print("neg_sam_infos", idx, [(neg_sampled_idx[jj], neg_sim_scores[jj]) for \
                #                              jj in range(len(neg_sampled_idx))])

                data.div_importance = torch.tensor([importance_for_div], dtype=torch.float64)
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_sampled_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_sampled_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_hop_pos_neg_on_big_graph_bat":
                # if len(self.idx_to_adj_list_dict[idx]) > 0:
                if int(self.degree_unnormalized_tensor[idx]) > 0.1:
                    # dgl_bg, new_idx_to_old_idx, old_idx_to_dis = self.get_rhop_ego_graph(idx,
                    #                                                                      num_hops=self.num_hops,
                    #                                                                      pos_sampling=True,
                    #                                                                      neg_sampling=False)
                    # rw_hops = 64
                    # divv = 0.2
                    # max_nodes_per_seed = max(self.rw_hops,
                    #                          int((self.dgl_big_gra.out_degree(idx) * math.e / (
                    #                                      math.e - 1) / (self.restart_prob + 0.2)) + 0.5)
                    #                          )
                    max_nodes_per_seed = self.rw_hops
                    # num_path = max(7, int(self.dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                    # num_samples = self.rw_hops
                    # print("max_nodes_per_seed", max_nodes_per_seed) # for debug only
                    sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                        self.dgl_big_gra, idx, num_samples=max_nodes_per_seed, num_path=self.num_path)
                    sampled_nodes = list(sampled_idx_to_mean_dis.keys())
                    # sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])
                    sorted_sampled_idx = sorted(sampled_nodes, key=lambda idxx: sampled_idx_to_count[idxx], reverse=True)

                    num_pos_samples = self.num_samples
                    # num_neg_samples = self.k * self.num_samples
                    # pos_idx_offset = int(len(sorted_sampled_idx) / (self.k + 1) + 1)

                    # print("all sampled nodes idxes", len(sorted_sampled_idx), num_pos_samples)
                    #
                    # pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                    pos_sampled_idx = sorted_sampled_idx
                    # print(len(pos_sampled_idx))
                    # neg_sampled_idx = sorted_sampled_idx[pos_idx_offset: ]

                    # importance_for_div = float(self.num_samples)

                    if len(pos_sampled_idx) == 0:
                        pos_sampled_idx = [idx for _ in range(num_pos_samples)]
                    else:
                        pos_idx_to_cnt = [sampled_idx_to_count[pos_idx] for pos_idx in pos_sampled_idx]
                        pos_idx_to_cnt_tsr = torch.tensor(pos_idx_to_cnt, dtype=torch.float64)
                        pos_idx_to_prob = torch.softmax(pos_idx_to_cnt_tsr, dim=0)
                        pos_sampled_idxs = torch.multinomial(pos_idx_to_prob,
                                                             replacement=True, num_samples=self.num_samples)
                        pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[i_pos])] for i_pos \
                                           in range(pos_sampled_idxs.size(0))]
                        sampled_idx_count = [sampled_idx_to_count[sam_idx] for sam_idx in pos_sampled_idx]
                        # print(len(sampled_idx_to_count), idx, pos_sampled_idx, sampled_idx_count)

                    # if len(neg_sampled_idx) == 0:
                    #     neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, 1 + num_neg_samples)]
                    # else:
                    #     # len_neg = len(neg_sampled_idx)
                    #     neg_idx_to_cnt = [sampled_idx_to_count[neg_idx] for neg_idx in neg_sampled_idx]
                    #     max_neg_cnt = max(neg_idx_to_cnt)
                    #     neg_idx_to_cnt_tsr = torch.tensor(neg_idx_to_cnt, dtype=torch.float64)
                    #     neg_idx_to_cnt_tsr = max_neg_cnt - neg_idx_to_cnt_tsr
                    #     neg_idx_to_prob = torch.softmax(neg_idx_to_cnt_tsr, dim=0)
                    #     neg_sampled_idxs = torch.multinomial(neg_idx_to_prob,
                    #                                          replacement=True, num_samples=self.num_samples * self.k)
                    #     neg_sampled_idx = [neg_sampled_idx[int(neg_sampled_idxs[i_neg])] for i_neg \
                    #                        in range(neg_sampled_idxs.size(0))]
                else:
                    # sampled_idx = [idx for _ in range(self.num_samples)]
                    # importance_for_div = float(self.num_samples)
                    pos_sampled_idx = [idx for _ in range(self.num_samples)]
                    # neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in
                    # range(1, self.num_samples * self.k + 1)]

                # pos_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx])
                # print("pos_sam_infos", idx, [(pos_sampled_idx[jj], pos_sim_scores[jj]) for \
                #                              jj in range(len(pos_sampled_idx))])
                #
                # neg_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # print("neg_sam_infos", idx, [(neg_sampled_idx[jj], neg_sim_scores[jj]) for \
                #                              jj in range(len(neg_sampled_idx))])

                # data.div_importance = torch.tensor([importance_for_div], dtype=torch.float64)

                # data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                if self.data_para:
                    data.sim_list = []
                else:
                    data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_sampled_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    if self.data_para:
                        data.sim_list.append(sim_data)
                    else:
                        data.sim_x.append(sim_data.x)
                        data.sim_edge_index.append(sim_data.edge_index)
                        data.sim_edge_attr.append(sim_data.edge_attr)
                data.sim_node_idx = torch.tensor(pos_sampled_idx, dtype=torch.long)
                # data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                # for neg_idx in neg_sampled_idx:
                #     neg_data = self.get_data_simple(neg_idx)
                #     data.neg_x.append(neg_data.x)
                #     data.neg_edge_index.append(neg_data.edge_index)
                #     data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "biased_hop_pos_neg_on_big_graph_bat":
                if int(self.degree_unnormalized_tensor[idx]) > 0.1:
                    max_nodes_per_seed = self.rw_hops
                    # sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                    #     self.dgl_big_gra, idx, num_samples=max_nodes_per_seed, num_path=self.num_path)
                    sampled_idx_to_count = self.get_n2v_pos_neg_samples_big_nvg(idx,
                                                                                max_nodes_per_seed,
                                                                                self.num_path)
                    sampled_nodes = list(sampled_idx_to_count.keys())
                    # sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])
                    sorted_sampled_idx = sorted(sampled_nodes, key=lambda idxx: sampled_idx_to_count[idxx], reverse=True)

                    num_pos_samples = self.num_samples
                    pos_sampled_idx = sorted_sampled_idx

                    if len(pos_sampled_idx) == 0:
                        pos_sampled_idx = [idx for _ in range(num_pos_samples)]
                    else:
                        pos_idx_to_cnt = [sampled_idx_to_count[pos_idx] for pos_idx in pos_sampled_idx]
                        pos_idx_to_cnt_tsr = torch.tensor(pos_idx_to_cnt, dtype=torch.float64)
                        pos_idx_to_prob = torch.softmax(pos_idx_to_cnt_tsr, dim=0)
                        pos_sampled_idxs = torch.multinomial(pos_idx_to_prob,
                                                             replacement=True, num_samples=self.num_samples)
                        pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[i_pos])] for i_pos \
                                           in range(pos_sampled_idxs.size(0))]
                        sampled_idx_count = [sampled_idx_to_count[sam_idx] for sam_idx in pos_sampled_idx]

                else:
                    pos_sampled_idx = [idx for _ in range(self.num_samples)]

                if self.data_para:
                    data.sim_list = []
                else:
                    data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_sampled_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    if self.data_para:
                        data.sim_list.append(sim_data)
                    else:
                        data.sim_x.append(sim_data.x)
                        data.sim_edge_index.append(sim_data.edge_index)
                        data.sim_edge_attr.append(sim_data.edge_attr)
                data.sim_node_idx = torch.tensor(pos_sampled_idx, dtype=torch.long)

            if self.extract_sim_type == "rwr_pos_big_gra_degree_neg":
                num_pos_samples = self.num_samples
                num_neg_samples = self.k * self.num_samples
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    max_nodes_per_seed = max(self.rw_hops,
                                             int((self.dgl_big_gra.out_degree(idx) * math.e / (
                                                     math.e - 1) / (self.restart_prob + 0.2)) + 0.5)
                                             )
                    sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                        self.dgl_big_gra, idx, num_samples=max_nodes_per_seed)
                    # num_path = max(self.rw_hops, int(self.dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                    # num_samples = self.rw_hops
                    # sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                    #     self.dgl_big_gra, idx, num_samples=num_samples, num_path=num_path)
                    sampled_nodes = list(sampled_idx_to_mean_dis.keys())
                    # sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])
                    # print(len(sampled_nodes))
                    sorted_sampled_idx = sorted(sampled_nodes, key=lambda idxx: sampled_idx_to_count[idxx], reverse=True)
                    weights = [sampled_idx_to_count[idxx] for idxx in sorted_sampled_idx]
                    weights = torch.tensor(weights, dtype=torch.float64)
                    weights = torch.softmax(weights, dim=0)
                    sampled_pos_idx_tensor = torch.multinomial(weights, num_pos_samples)
                    sampled_pos_idx = [sorted_sampled_idx[int(sampled_pos_idx_tensor[ii])] for ii in \
                                       range(sampled_pos_idx_tensor.size(0))]
                    # print(len(sampled_pos_idx))
                else:
                    sampled_pos_idx = [idx for __ in range(num_pos_samples)]

                pos_samples_idxs = list(set(sampled_pos_idx + [idx]))
                pos_samples_idxs_tensor = torch.tensor(pos_samples_idxs, dtype=torch.long)
                now_degree_tensor = self.degree_unnormalized_tensor.clone()
                now_degree_tensor[pos_samples_idxs_tensor] = 0.0
                now_degree_tensor = now_degree_tensor / torch.sum(now_degree_tensor, dim=0, keepdim=True)
                neg_samples_idx_tensor = torch.multinomial(now_degree_tensor, num_neg_samples)
                neg_samples_idx = neg_samples_idx_tensor.tolist()

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in sampled_pos_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_samples_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_hop_pos_neg_on_big_graph_no_pos_graph":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    # divv = 0.2
                    max_nodes_per_seed = max(self.rw_hops,
                                             int((self.dgl_big_gra.out_degree(idx) * math.e / (
                                                         math.e - 1) / (self.restart_prob + 0.2)) + 0.5)
                                             )
                    # num_path = max(7, int(self.dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                    # num_samples = self.rw_hops
                    # print("max_nodes_per_seed", max_nodes_per_seed) # for debug only
                    sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                        self.dgl_big_gra, idx, num_samples=max_nodes_per_seed)
                    # num_path = max(self.rw_hops, int(self.dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                    # num_samples = self.rw_hops
                    # sampled_idx_to_mean_dis, sampled_idx_to_count = self.rwr_sample_pos_graphs_path_dis(
                    #     self.dgl_big_gra, idx, num_samples=num_samples, num_path=num_path)
                    sampled_nodes = list(sampled_idx_to_mean_dis.keys())
                    sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])

                    # num_pos_samples = self.num_samples
                    num_neg_samples = self.k * self.num_samples
                    pos_idx_offset = int(len(sorted_sampled_idx) / (self.k + 1) + 1)
                    #
                    # pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                    neg_sampled_idx = sorted_sampled_idx[pos_idx_offset: ]

                    # importance_for_div = float(self.num_samples)

                    # if len(pos_sampled_idx) == 0:
                    #     pos_sampled_idx = [idx for _ in range(num_pos_samples)]
                    # else:
                    #     importance_for_div = max(1.0, float(self.num_samples) / float(len(pos_sampled_idx)))
                    #     pos_sampled_prob = [1.0 + (-0.9) * (x * x / ((len(pos_sampled_idx)) * (len(pos_sampled_idx)))) \
                    #                         for x in range(1, len(pos_sampled_idx) + 1)] # generates possibilities to be sampled for each candidate
                    #     # uniform distribution
                    #     # pos_sampled_prob = [1.0 / len(pos_sampled_idx) for __ in range(len(pos_sampled_idx))]
                    #     pos_sampled_prob_tensor = torch.tensor(pos_sampled_prob, dtype=torch.float64) / sum(
                    #         pos_sampled_prob)
                    #     pos_sampled_idxs = torch.multinomial(pos_sampled_prob_tensor,
                    #                                          replacement=True, num_samples=self.num_samples)
                    #     pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[ipos])] for ipos in
                    #                        range(pos_sampled_idxs.size(0))]
                    # print(len(neg_sampled_idx))
                    if len(neg_sampled_idx) == 0:
                        neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, 1 + num_neg_samples)]
                    else:
                        len_neg = len(neg_sampled_idx)
                        neg_sampled_prob = [1.0 + (-0.9) * ((len_neg - x) * (len_neg - x) / ((len_neg) * (len_neg))) \
                                                    for x in range(1, len_neg + 1)]
                        neg_sampled_prob_tensor = torch.tensor(neg_sampled_prob, dtype=torch.float64) / sum(neg_sampled_prob)
                        neg_sampled_idxs = torch.multinomial(neg_sampled_prob_tensor,
                                                             replacement=True, num_samples=num_neg_samples)
                        neg_sampled_idx = [neg_sampled_idx[int(neg_sampled_idxs[ineg])] for ineg in
                                           range(neg_sampled_idxs.size(0))]

                else:
                    # sampled_idx = [idx for _ in range(self.num_samples)]
                    # importance_for_div = float(self.num_samples)
                    # pos_sampled_idx = [idx for _ in range(self.num_samples)]
                    neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, self.num_samples * self.k + 1)]

                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                pos_sampled_idx = []
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    pos_sampled_idx.append(real_idx)
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)
                # print("importance_for_div", idx, importance_for_div)
                #
                # pos_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] \
                #                                                            for pos_idx in pos_sampled_idx])
                # pos_sim_path_length = [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in pos_sampled_idx]
                # print("pos_sam_infos", idx, [(pos_sampled_idx[jj], pos_sim_scores[jj], pos_sim_path_length[jj]) \
                #                              for jj in range(len(pos_sampled_idx))])
                #
                # neg_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # neg_sim_path_length = [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in neg_sampled_idx]
                # print("pos_sam_infos", idx, [(neg_sampled_idx[jj], neg_sim_scores[jj], neg_sim_path_length[jj]) \
                #                              for jj in range(len(neg_sampled_idx))])
                #
                # print("pos", idx, pos_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx]),
                #       )
                # print("neg", idx, neg_sampled_idx, compute_similarity_1vn(self.smiles_list[idx],
                #                                                           [self.smiles_list[pos_idx] for pos_idx in
                #                                                            neg_sampled_idx]),
                #       [nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in neg_sampled_idxs])

                # pos_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[pos_idx] for pos_idx in pos_sampled_idx])
                # print("pos_sam_infos", idx, [(pos_sampled_idx[jj], pos_sim_scores[jj]) for \
                #                              jj in range(len(pos_sampled_idx))])
                #
                # neg_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                         [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # print("neg_sam_infos", idx, [(neg_sampled_idx[jj], neg_sim_scores[jj]) for \
                #                              jj in range(len(neg_sampled_idx))])

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_sampled_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_hop_sample_only_neg":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                # sampled_pos_idx = [int(keys[int(choose_idx[i])]) for i in range(self.num_samples)]
                # print("pos", sampled_pos_idx)
                # print("pos_sim", compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[pos_idx] for \
                #                                                                 pos_idx in sampled_pos_idx]))
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx, old_idx_to_dis = self.get_rhop_ego_graph(idx, num_hops=self.num_hops, pos_sampling=True, neg_sampling=False)
                    sampled_idx = self.rwr_sample_pos_graphs(dgl_bg, 0, num_samples=(self.num_samples * (self.k)))
                    # convert smapled idx from new idx to old idx
                    sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                    # sorted_sampled_idx = sorted(sampled_idx, key=lambda i: old_idx_to_dis[i])
                    # sorted by similarity
                    sorted_sampled_idx = sorted(sampled_idx, key=lambda i: (1.0 - self.pos_idxes[idx][i]) \
                        if i in self.pos_idxes[idx] else old_idx_to_dis[i])
                    # pos_idx_offset = int(len(sorted_sampled_idx) / 8)
                    # pos_sampled_idx = sampled_idx[: pos_idx_offset]
                    neg_sampled_idx = [neg_idx for neg_idx in sorted_sampled_idx if old_idx_to_dis[neg_idx] >= 2]
                    # num_pos_samples = self.num_samples
                    num_neg_samples = self.k * self.num_samples
                    # print(len(sorted_sampled_idx))

                    if len(neg_sampled_idx) < num_neg_samples:
                        neg_sampled_idx = neg_sampled_idx + [(idx + _) % len(self.smiles_list) \
                                                             for _ in range(1, num_neg_samples - len(neg_sampled_idx) + 1)]
                    elif len(neg_sampled_idx) > num_neg_samples:
                        neg_sampled_idx = neg_sampled_idx[-num_neg_samples: ]
                else:
                    neg_sampled_idx = [(idx + _) % len(self.smiles_list) for _ in range(1, self.num_samples * self.k + 1)]
                # neg_samples_sim_scores = compute_similarity_1vn(self.smiles_list[idx],
                #                                                 [self.smiles_list[neg_idx] for neg_idx in neg_sampled_idx])
                # print("sampled neg idx = ", idx, neg_sampled_idx)
                # print("sampled neg scores = ", neg_samples_sim_scores)
                # data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_sampled_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "n2v_neg_pos_one_walk":
                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                neg_n2v_G = Graph(r_hop_nx_G, False, self.neg_p_q[0], self.neg_p_q[1])
                neg_n2v_G.preprocess_transition_probs()
                num_neg_samples = self.k * self.num_samples
                neg_walk_nodes = neg_n2v_G.node2vec_walk((self.num_samples + num_neg_samples) * 7, 0)
                neg_walk_unique_nodes = list(set(neg_walk_nodes))
                # try to remove self
                try:
                    neg_walk_unique_nodes.remove(0)
                except ValueError:
                    pass

                # transfer to old node idx
                neg_walk_unique_nodes = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
                neg_idx_to_index = {neg_idx: i for i, neg_idx in enumerate(neg_walk_unique_nodes)}
                neg_walk_unique_nodes = sorted(neg_walk_unique_nodes, key=lambda neg_idx: neg_idx_to_index[neg_idx] \
                    if idx_to_dis[neg_idx] == 1 else idx_to_dis[neg_idx])
                pos_offset = len(neg_walk_unique_nodes) // 8
                pos_candi_idxes = neg_walk_unique_nodes[: pos_offset + 1]
                neg_candi_idxes = neg_walk_unique_nodes[pos_offset + 1:]

                if len(pos_candi_idxes) < self.num_samples:
                    pos_candi_idxes = pos_candi_idxes + [idx for _ in range(self.num_samples - len(pos_candi_idxes))]
                elif len(pos_candi_idxes) > self.num_samples:
                    pos_candi_idxes = pos_candi_idxes[: self.num_samples]

                if len(neg_candi_idxes) < num_neg_samples:
                    neg_candi_idxes = neg_candi_idxes + [(idx + _) % len(self.smiles_list) \
                                                         for _ in range(1, num_neg_samples - len(neg_candi_idxes) + 1)]
                elif len(neg_candi_idxes) > num_neg_samples:
                    neg_candi_idxes = neg_candi_idxes[-num_neg_samples:]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_candi_idxes:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_candi_idxes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "n2v_neg_pos_one_walk_on_big_graph":
                # r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                # neg_n2v_G = Graph(r_hop_nx_G, False, self.neg_p_q[0], self.neg_p_q[1])
                # neg_n2v_G.preprocess_transition_probs()
                num_neg_samples = self.k * self.num_samples
                neg_walk_nodes = self.big_graph_n2v_graph.node2vec_walk((self.num_samples + num_neg_samples) * 7, idx)
                neg_walk_nodes = [int(neg_walk_idx) for neg_walk_idx in neg_walk_nodes]
                neg_walk_unique_nodes = list(set(neg_walk_nodes))
                # try to remove self
                try:
                    neg_walk_unique_nodes.remove(idx) # remoe self
                except ValueError:
                    pass

                # transfer to old node idx
                # neg_walk_unique_nodes = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
                neg_idx_to_index = {neg_idx: i for i, neg_idx in enumerate(neg_walk_unique_nodes)}
                node_idx_to_dis = {sam_idx: nx.shortest_path_length(self.nxg, idx, sam_idx) for sam_idx in neg_walk_unique_nodes}
                neg_walk_unique_nodes = sorted(neg_walk_unique_nodes, key=lambda neg_idx: neg_idx_to_index[neg_idx] \
                    if node_idx_to_dis[neg_idx] == 1 else node_idx_to_dis[neg_idx])
                pos_offset = len(neg_walk_unique_nodes) // 8
                pos_candi_idxes = neg_walk_unique_nodes[: pos_offset + 1]
                neg_candi_idxes = neg_walk_unique_nodes[pos_offset + 1:]

                if len(pos_candi_idxes) < self.num_samples:
                    pos_candi_idxes = pos_candi_idxes + [idx for _ in range(self.num_samples - len(pos_candi_idxes))]
                elif len(pos_candi_idxes) > self.num_samples:
                    pos_candi_idxes = pos_candi_idxes[: self.num_samples]

                if len(neg_candi_idxes) < num_neg_samples:
                    neg_candi_idxes = neg_candi_idxes + [(idx + _) % len(self.smiles_list) \
                                                         for _ in range(1, num_neg_samples - len(neg_candi_idxes) + 1)]
                elif len(neg_candi_idxes) > num_neg_samples:
                    neg_candi_idxes = neg_candi_idxes[-num_neg_samples:]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_candi_idxes:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_candi_idxes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "n2v_walk_neg_more_hop":
                if self.select_other_node_stra == "independent":
                    node_to_dis = nx.single_target_shortest_path(self.nxg, idx, 5)
                    node_to_dis_item = sorted(node_to_dis.items(), key=lambda i: i[1])
                    dis_node_idx = int(node_to_dis_item[-1][0])
                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                if self.select_other_node_stra != "independent":
                    idx_to_dis_items = sorted(idx_to_dis.items(), key= lambda i: i[1])
                    dis_node_idx = int(idx_to_dis_items[-1][0])
                r_hop_nx_G_dis, new_idx_to_old_idx_dis, idx_to_dis_dis = self.get_rhop_ego_nx_graph(dis_node_idx, num_hops=self.num_hops)
                pos_n2v_G = Graph(r_hop_nx_G, False, self.pos_p_q[0], self.pos_p_q[1])
                pos_n2v_G_dis = Graph(r_hop_nx_G_dis, False, self.pos_p_q[0], self.pos_p_q[1])
                pos_n2v_G.preprocess_transition_probs()
                pos_n2v_G_dis.preprocess_transition_probs()
                pos_walk_nodes = pos_n2v_G.node2vec_walk(self.num_samples * 17, 0)
                pos_walk_unique_nodes = list(set(pos_walk_nodes))
                try:
                    pos_walk_unique_nodes.remove(0)
                except ValueError:
                    pass
                pos_walk_unique_nodes = [new_idx_to_old_idx[pos_idx] for pos_idx in pos_walk_unique_nodes]
                pos_walk_unique_nodes = sorted(pos_walk_unique_nodes, key=lambda pos_idx: idx_to_dis[pos_idx])

                if len(pos_walk_unique_nodes) < self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes + [idx for _ in range(self.num_samples - len(pos_walk_unique_nodes))]
                elif len(pos_walk_unique_nodes) > self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes[: self.num_samples]

                neg_walk_nodes = pos_n2v_G_dis.node2vec_walk(self.num_samples * self.k * 17, 0)
                neg_walk_unique_nodes = list(set(neg_walk_nodes))
                # discount those nodes who are near to the target node
                neg_walk_unique_nodes = [new_idx_to_old_idx_dis[neg_idx] for neg_idx in neg_walk_unique_nodes if new_idx_to_old_idx_dis[neg_idx] not in idx_to_dis]
                # print(len(neg_walk_unique_nodes))

                if len(neg_walk_unique_nodes) < self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % len(self.smiles_list) for _ in range(
                        self.num_samples * self.k - len(neg_walk_unique_nodes))]
                elif len(neg_walk_unique_nodes) > self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes[-self.num_samples * self.k:]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_walk_unique_nodes:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_walk_unique_nodes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "n2v_walk_neg_no_graph_pos":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    # ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                # sim_datas = list()
                sampled_pos_idx = [int(keys[int(choose_idx[i])]) for i in range(self.num_samples)]
                print("pos", sampled_pos_idx)
                print("pos_sim", compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[pos_idx] for \
                                                                                pos_idx in sampled_pos_idx]))
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops + 1)
                neg_n2v_G = Graph(r_hop_nx_G, False, self.neg_p_q[0], self.neg_p_q[1])
                neg_n2v_G.preprocess_transition_probs()
                neg_walk_nodes = neg_n2v_G.node2vec_walk(self.k * self.num_samples * 7, 0)
                neg_walk_unique_nodes = list(set(neg_walk_nodes))
                try:
                    neg_walk_unique_nodes.remove(0)
                except ValueError:
                    pass

                neg_walk_unique_nodes = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
                neg_walk_unique_nodes = sorted(neg_walk_unique_nodes, key=lambda neg_idx: idx_to_dis[neg_idx])
                neg_walk_unique_nodes = [neg_idx for neg_idx in neg_walk_unique_nodes if neg_idx not in idx_to_sim_score_dict]

                if len(neg_walk_unique_nodes) < self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % len(self.smiles_list) for _ in range(1,
                                                                                                                      self.num_samples * self.k - len(
                                                                                                                          neg_walk_unique_nodes) + 1)]
                elif len(neg_walk_unique_nodes) > self.num_samples * self.k:
                    neg_walk_unique_nodes = neg_walk_unique_nodes[-self.num_samples * self.k:]
                print("neg", idx, neg_walk_unique_nodes)
                print("neg_sim", compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[neg_idx] \
                                                                                for neg_idx in neg_walk_unique_nodes]))
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_walk_unique_nodes:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "n2v_walk_pos_neg":
                r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = self.get_rhop_ego_nx_graph(idx, num_hops=self.num_hops)
                pos_n2v_G = Graph(r_hop_nx_G, False, self.pos_p_q[0], self.pos_p_q[1])
                pos_n2v_G.preprocess_transition_probs()
                pos_walk_nodes = pos_n2v_G.node2vec_walk(self.num_samples * 7, 0)
                pos_walk_unique_nodes = list(set(pos_walk_nodes))
                try:
                    pos_walk_unique_nodes.remove(0)
                except ValueError:
                    pass
                pos_walk_unique_nodes = [new_idx_to_old_idx[pos_idx] for pos_idx in pos_walk_unique_nodes]
                pos_walk_unique_nodes = sorted(pos_walk_unique_nodes, key=lambda pos_idx: idx_to_dis[pos_idx])
                # print("pos", len(pos_walk_unique_nodes))
                if len(pos_walk_unique_nodes) < self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes + [idx for _ in range(self.num_samples - len(pos_walk_unique_nodes))]
                elif len(pos_walk_unique_nodes) > self.num_samples:
                    pos_walk_unique_nodes = pos_walk_unique_nodes[: self.num_samples]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_walk_unique_nodes:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)
                # print("pos_sim", compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[neg_idx] for neg_idx in pos_walk_unique_nodes]))
                if self.with_neg:
                    neg_n2v_G = Graph(r_hop_nx_G, False, self.neg_p_q[0], self.neg_p_q[1])
                    neg_n2v_G.preprocess_transition_probs()
                    neg_walk_nodes = neg_n2v_G.node2vec_walk(self.k * self.num_samples * 7, 0)
                    ## todo: start the biased random walk from the root node only once; --- we should try to start from this node several times!!!
                    # pos_walk_nodes_dict = {nei_idx: nei_pos for (nei_pos, nei_idx) in enumerate(reversed(pos_walk_nodes))}
                    # neg_walk_nodes_dict = {nei_idx: nei_pos for (nei_pos, nei_idx) in enumerate(reversed(neg_walk_nodes))}
                    # pos_walk_nodes_dict_items_sorted = sorted(pos_walk_nodes_dict.items(), key=lambda ii: ii[1], reverse=True)
                    # neg_walk_nodes_dict_items_sorted = sorted(neg_walk_nodes_dict.items(), key=lambda ii: ii[1], reverse=True)
                    # # pos_walk_nodes_dict =
                    # pos_walk_unique_nodes = [pos_item[0] for pos_item in pos_walk_nodes_dict_items_sorted]
                    # neg_walk_unique_nodes = [neg_item[0] for neg_item in neg_walk_nodes_dict_items_sorted]
                    neg_walk_unique_nodes = list(set(neg_walk_nodes))
                    # print("pos_walk_unique_nodes = ", len(pos_walk_unique_nodes), "neg_walk_unique_nodes = ", len(neg_walk_unique_nodes))
                    try:
                        neg_walk_unique_nodes.remove(0)
                    except ValueError:
                        pass

                    neg_walk_unique_nodes = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
                    neg_walk_unique_nodes = sorted(neg_walk_unique_nodes, key=lambda neg_idx: idx_to_dis[neg_idx])
                    # print(pos_walk_unique_nodes, neg_walk_unique_nodes)
                    # print(len(pos_walk_unique_nodes), len(neg_walk_unique_nodes))
                    # print("neg", len(neg_walk_unique_nodes))

                    if len(neg_walk_unique_nodes) < self.num_samples * self.k:
                        neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % len(self.smiles_list) for _ in range(1, self.num_samples * self.k - len(neg_walk_unique_nodes) + 1)]
                    elif len(neg_walk_unique_nodes) > self.num_samples * self.k:
                        neg_walk_unique_nodes = neg_walk_unique_nodes[-self.num_samples * self.k: ]
                    # print("neg_sim", compute_similarity_1vn(self.smiles_list[idx], [self.smiles_list[neg_idx] for neg_idx in neg_walk_unique_nodes]))
                    data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                    for neg_idx in neg_walk_unique_nodes:
                        neg_data = self.get_data_simple(neg_idx)
                        data.neg_x.append(neg_data.x)
                        data.neg_edge_index.append(neg_data.edge_index)
                        data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "precomputed_n2v_walk_pos_neg":
                pos_sampled_idx = self.n2v_idx_to_pos_samples[idx][: self.num_samples]
                self.n2v_idx_to_pos_samples[idx] = self.n2v_idx_to_pos_samples[idx][self.num_samples: ]
                neg_sampled_idx = self.n2v_idx_to_neg_samples[idx][: self.num_samples * self.k]
                self.n2v_idx_to_neg_samples[idx] = self.n2v_idx_to_neg_samples[idx][self.num_samples * self.k: ]
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for pos_idx in pos_sampled_idx:
                    sim_data = self.get_data_simple(pos_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for neg_idx in neg_sampled_idx:
                    neg_data = self.get_data_simple(neg_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "precomputed_rwr":
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                # sampled_idx = self.idx_to_sampled_pos_idx[idx]
                prev_len = len(self.idx_to_sampled_pos_idx[idx])
                for i in range(self.num_samples):

                    real_idx = int(self.idx_to_sampled_pos_idx[idx][i])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)
                    self.idx_to_sampled_pos_idx[idx].pop(0)
                assert prev_len - len(self.idx_to_sampled_pos_idx[idx]) == self.num_samples

            if self.extract_sim_type == "rwr":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx, _ = self.get_rhop_ego_graph(idx, num_hops=self.num_hops)
                    sampled_idx = self.rwr_sample_pos_graphs(dgl_bg, 0, num_samples=self.num_samples * 2)
                    sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                else:
                    sampled_idx = [idx for _ in range(self.num_samples)]
                if len(sampled_idx) > self.num_samples:
                    sampled_idx = sampled_idx[: self.num_samples]
                elif len(sampled_idx) < self.num_samples:
                    sampled_idx += [idx for _ in range(self.num_samples - len(sampled_idx))]

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    real_idx = int(sampled_idx[i])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

            # base_score = torch.ones((num_all_data, ), dtype=torch.float64)
            # calculated_idx = self.calculated_sim_list[idx]
            # calculated_score = self.calculated_sim_score[idx]
            # mean_score = torch.mean(calculated_score)
            # base_score[calculated_idx] += (calculated_score - mean_score)
            # base_score[idx] = -1e9
            # sampled_rates = torch.softmax(base_score, dim=0)
            #
            # calculate_sim_samples = torch.multinomial(sampled_rates, self.num_calculate_sim_samples)
            # calculate_sim_smiles = list()
            # for i in range(len(calculate_sim_samples)):
            #     now_idx = int(calculate_sim_samples[i])
            #     calculate_sim_smiles.append(self.smiles_list[now_idx])
            # sim_scores = compute_similarity_1vn(self.smiles_list[idx], calculate_sim_smiles)
            # sim_scores = torch.tensor(sim_scores, dtype=torch.float64)
            # sampled_rates = torch.softmax(sim_scores, dim=0)
            # choose_idx = torch.multinomial(sampled_rates, self.num_samples)
            # self.calculated_sim_list[idx] = torch.cat([self.calculated_sim_list[idx], calculate_sim_samples], dim=0)
            # self.calculated_sim_score[idx] = torch.cat([self.calculated_sim_score[idx], sim_scores], dim=0)

            if self.extract_sim_type == "uniform_neg":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                num_neg_samples = self.num_samples * self.k

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                ##### comput engative samples by performing rwr on the ego-network graph.  -- related params -- num_hops -- number of hops to samples from
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                sampled_idx = random.sample(range(len(self.smiles_list)), num_neg_samples)

                # data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(num_neg_samples):
                    real_idx = int(sampled_idx[i])
                    neg_data = self.get_data_simple(real_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "rwr_neg":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]

                num_neg_samples = self.num_samples * self.k

                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                ##### comput engative samples by performing rwr on the ego-network graph.  -- related params -- num_hops -- number of hops to samples from
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()

                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx, _ = self.get_rhop_ego_graph(idx, num_hops=self.num_hops, neg_sampling=True)
                    sampled_idx = self.rwr_sample_neg_graphs(dgl_bg, 0, num_samples=num_neg_samples, exc_idx=None)
                    sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                    try:
                        sampled_idx.remove(choose_idx[0])
                    except ValueError:
                        pass
                else:
                    sampled_idx = [(idx + 1) % len(self.smiles_list) for _ in range(num_neg_samples)]
                if len(sampled_idx) > num_neg_samples:
                    sampled_idx = sampled_idx[: num_neg_samples]
                elif len(sampled_idx) < num_neg_samples:
                    sampled_idx += [(idx + 1 + _) % len(self.smiles_list) for _ in range(num_neg_samples - len(sampled_idx))]

                # data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(num_neg_samples):
                    real_idx = int(sampled_idx[i])
                    neg_data = self.get_data_simple(real_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)

            if self.extract_sim_type == "precomputed_neg":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]
                # print(choose_idx, keys[choose_idx])
                # print(choose_idx, keys[choose_idx], idx_to_sim_score_dict[keys[choose_idx]])

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                idx_to_unsim_score_dict = self.neg_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_unsim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_unsim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                num_neg_samples = self.num_samples * self.k
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = num_neg_samples if num_neg_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < num_neg_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((num_neg_samples - num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(num_neg_samples)]
                    keys = [idx for j in range(num_neg_samples)]
                # print(choose_idx, keys[choose_idx])
                # print(choose_idx, keys[choose_idx], idx_to_sim_score_dict[keys[choose_idx]])

                # sim_datas = list()
                data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                for i in range(num_neg_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    neg_data = self.get_data_simple(real_idx)
                    data.neg_x.append(neg_data.x)
                    data.neg_edge_index.append(neg_data.edge_index)
                    data.neg_edge_attr.append(neg_data.edge_attr)
            if self.extract_sim_type == "pos_rwr":
                if len(self.idx_to_adj_list_dict[idx]) > 0:
                    dgl_bg, new_idx_to_old_idx, _ = self.get_rhop_ego_graph(idx, num_hops=self.num_hops, neg_sampling=False, pos_sampling=True)
                    pos_samples = self.rwr_sample_pos_graphs(dgl_bg=dgl_bg, idx=0, num_samples=self.num_samples)
                    pos_samples = [new_idx_to_old_idx[new_idx] for new_idx in pos_samples]
                else:
                    pos_samples = []
                if len(pos_samples) < self.num_samples:
                    pos_samples += [idx for __ in range(self.num_samples - len(pos_samples))]
                elif len(pos_samples) > self.num_samples:
                    pos_samples = pos_samples[: self.num_samples]
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    real_idx = int(pos_samples[i])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

            if self.extract_sim_type == "sim":
                idx_to_sim_score_dict = self.pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                # for idxbb, score in enumerate(idx_to_sim_score_dict):
                #     keys.append(idxbb)
                #     values.append(score)
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    ori_values = values.clone()
                    values = torch.softmax(values, dim=0)
                    num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, num_samples)

                    if num_samples < self.num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples, ), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(self.num_samples)]
                    keys = [idx for j in range(self.num_samples)]
                # print(choose_idx, keys[choose_idx])
                # print(choose_idx, keys[choose_idx], idx_to_sim_score_dict[keys[choose_idx]])

                # sim_datas = list()
                data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                for i in range(self.num_samples):
                    sample_idx = int(choose_idx[i])
                    # real_idx = int(calculate_sim_samples[sampled_idx])
                    real_idx = int(keys[sample_idx])
                    sim_data = self.get_data_simple(real_idx)
                    data.sim_x.append(sim_data.x)
                    data.sim_edge_index.append(sim_data.edge_index)
                    data.sim_edge_attr.append(sim_data.edge_attr)

                st_time = time.time()
                if self.with_neg:
                    data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                    num_neg_samples = self.k * self.num_samples
                    # stt_len = len(self.idx_to_neg_samples_list[idx])
                    # for i in range(num_neg_samples):
                    #     neg_sample_id = self.idx_to_neg_samples_list[idx][0]
                    #     neg_data = self.get_data_simple(neg_sample_id)
                    #     data.neg_x.append(neg_data.x)
                    #     data.neg_edge_index.append(neg_data.edge_index)
                    #     data.neg_edge_attr.append(neg_data.edge_attr)
                    #     self.idx_to_neg_samples_list[idx].pop(0)
                    # assert stt_len - len(self.idx_to_neg_samples_list[idx]) == num_neg_samples

                    # for neg samples:
                #### on-fly compute negative samples
                    # random_chosen_idx = random.sample(list(set(range(len(self.smiles_list))) - set(keys)),
                    #                                   num_neg_samples * 10)
                    #
                    # all_involved_idx = keys + random_chosen_idx

                    sim_scores_all = torch.zeros((len(self.smiles_list), ), dtype=torch.float64)
                    if len(idx_to_sim_score_dict) > 0:
                        calculated_sim_idx = torch.tensor(keys, dtype=torch.long)
                        calculated_sim_scores = ori_values
                        # sim_scores_all[: len(keys)] = ori_values / 4
                        sim_scores_all[calculated_sim_idx] = calculated_sim_scores / 4
                    normalized_neg_sample_rates = torch.softmax(sim_scores_all, dim=0)
                    neg_samples_idx = torch.multinomial(normalized_neg_sample_rates, num_neg_samples)
                    # data.neg_x, data.neg_edge_index, data.neg_edge_attr = list(), list(), list()
                    # neg_samples_idx = [all_involved_idx[int(neg_samples_idx[jj])] for jj in range(neg_samples_idx.size(0))]
                    # print(neg_samples_idx)


                    for i in range(num_neg_samples):
                        sample_idx = int(neg_samples_idx[i])
                        # real_idx = int(calculate_sim_samples[sampled_idx])
                        # real_idx = int(keys[sample_idx])
                        neg_data = self.get_data_simple(sample_idx)
                        data.neg_x.append(neg_data.x)
                        data.neg_edge_index.append(neg_data.edge_index)
                        data.neg_edge_attr.append(neg_data.edge_attr)
                # print("time for calculating negative samples = ", time.time() - st_time)
            # sim_data = self.get_data_simple(keys[choose_idx])
            # data.sim_x, data.sim_edge_index, data.sim_edge_attr = sim_data.x, sim_data.edge_index, sim_data.edge_attr

        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')


class MoleculeDatasetForContrastOth(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 num_samples=3,
                 extract_sim_type="precomputed_rwr",
                 rw_hops=64,
                 num_path=7):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDatasetForContrastOth, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.total = 0
        self.num_samples = num_samples
        self.idx_to_adj_list_dict = dict()
        self.extract_sim_type = extract_sim_type
        self.rw_hops = rw_hops
        self.num_path = num_path
        processed_dir = "./dataset"

        if dataset == "other_soc":
            processed_dir = "./dataset/ALL_GRA"
            self.graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))
            # self.idx_to_sim_scores_ary = np.load(
            #     os.path.join(processed_dir, "idx_to_sim_score_dict.npy"), allow_pickle=True).item()
            self.idx_to_sim_scores_ary = np.load(
                os.path.join(processed_dir, "combined_social_gra_idx_to_sim_score_dict_0.npy"), allow_pickle=True).item()
            self.idx_to_candi_idxes = np.load(
                os.path.join(processed_dir, "idx_to_candi_idx_list_dict.npy"), allow_pickle=True
            ).item()
        else:
            # self.graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "tot_gra_cls_gras_sorted.bin"))
            self.graphs, _ = dgl.data.utils.load_graphs(os.path.join(processed_dir, "social_dgl_graphs_sorted.bin"))
            # self.idx_to_sim_scores_ary = np.load(
            #     os.path.join(processed_dir, "idx_to_sim_score_dict.npy"), allow_pickle=True).item()
            # self.idx_to_sim_scores_ary = np.load(
            #     os.path.join(processed_dir, "combined_social_gra_idx_to_sim_score_dict_0.npy"), allow_pickle=True).item()
            self.idx_to_sim_scores_ary = np.load(
                os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict.npy"), allow_pickle=True).item()
            # self.idx_to_candi_idxes = np.load(
            #     os.path.join(processed_dir, "idx_to_candi_idx_list_dict.npy"), allow_pickle=True
            # ).item()
            self.idx_to_candi_idxes = np.load(
                os.path.join(processed_dir, "social_idx_to_candi_idx_list_dict.npy"), allow_pickle=True
            ).item()
        print(type(self.idx_to_sim_scores_ary), type(self.idx_to_candi_idxes))
        print("pres loaded~")
        self.gra_datas = list()
        for idx, gra in enumerate(self.graphs):
            self.gra_datas.append(self.from_dgl_graph_to_graph_data(idx))


    def from_dgl_graph_to_graph_data(self, idx):
        edges_fr_tst, edge_to_tsr = self.graphs[idx].edges()
        edge_index = torch.cat([edges_fr_tst.view(1, -1), edge_to_tsr.view(1, -1)], dim=0)
        x = torch.zeros((self.graphs[idx].number_of_nodes(), 2), dtype=torch.long)
        x[:, 0] = 119
        edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def get_data_simple(self, idx):
        return self.gra_datas[idx]

    def get(self, idx):
        # idx_to_sim_score_dict = self.pos_idxes[idx]
        keys = list()
        values = list()
        # for idxbb in idx_to_sim_score_dict:
        #     keys.append(idxbb)
        #     values.append(idx_to_sim_score_dict[idxbb])
        # for idxbb, score in enumerate(idx_to_sim_score_dict):
        #     keys.append(idxbb)
        #     values.append(score)
        keys = self.idx_to_candi_idxes[idx]
        data = self.gra_datas[idx]

        if len(keys) > 0:

            values = torch.from_numpy(self.idx_to_sim_scores_ary[idx])
            values = torch.softmax(values, dim=0)
            # values = values / (torch.sum(values) + 1e-12)
            # num_samples = self.num_samples if self.num_samples <= values.size(0) else values.size(0)
            # choose_idx = torch.multinomial(values, num_samples, replacement=True)
            choose_idx = torch.multinomial(values, self.num_samples, replacement=True)

            # if num_samples < self.num_samples:
            #     keys.append(idx)
            #     choose_idx = torch.cat([choose_idx, torch.full((self.num_samples - num_samples,), len(keys) - 1,
            #                                                    dtype=torch.long)], dim=0)
        else:
            choose_idx = [0 for j in range(self.num_samples)]
            keys = [idx for j in range(self.num_samples)]
        # print(choose_idx, keys[choose_idx])
        # print(choose_idx, keys[choose_idx], idx_to_sim_score_dict[keys[choose_idx]])

        # sim_datas = list()
        data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
        for i in range(self.num_samples):
            sample_idx = int(choose_idx[i])
            # real_idx = int(calculate_sim_samples[sampled_idx])
            real_idx = int(keys[sample_idx])
            sim_data = self.get_data_simple(real_idx)
            data.sim_x.append(sim_data.x)
            data.sim_edge_index.append(sim_data.edge_index)
            data.sim_edge_attr.append(sim_data.edge_attr)

        # st_time = time.time()
        return data

    def __len__(self):
        return len(self.graphs)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return
        # raise NotImplementedError('Must indicate valid location of raw data. '
        #                           'No download allowed')

class SplitLabelMoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False,
                 k=5):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(SplitLabelMoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.total = 0

        self.is_a_label = dict()
        self.not_a_label = dict()
        self.all_labels = dict()
        self.k = k

        self.process()
        if not empty:
            print("abc loading data from processed paths[0]")
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        print("processing....")
        data_smiles_list = []
        data_list = []

        data_list_realy = list()

        if self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            from splitters import scaffold_split

            ###
            downstream_dir = [
                'dataset/bace',
                'dataset/bbbp',
                'dataset/clintox',
                'dataset/esol',
                'dataset/freesolv',
                'dataset/hiv',
                'dataset/lipophilicity',
                'dataset/muv',
                # 'dataset/pcba/processed/smiles.csv',
                'dataset/sider',
                'dataset/tox21',
                'dataset/toxcast'
            ]

            downstream_inchi_set = set()
            for d_path in downstream_dir:
                print(d_path)
                dataset_name = d_path.split('/')[1]
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)

                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset,
                                                                                    downstream_smiles, task_idx=None,
                                                                                    null_value=0,
                                                                                    frac_train=0.8, frac_valid=0.1,
                                                                                    frac_test=0.1,
                                                                                    return_smiles=True)

                ### remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split('.')
                    for s in species_list:  # record inchi for all species, not just
                        # largest (by default in create_standardized_mol_id if input has
                        # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))

            print('processing')
            for i in range(len(rdkit_mol_objs)):
                if i % 1000 == 0:
                    print("i = ", i)
                # if i == 100:
                #     break
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            # print("data.y = ", data.y, data.y.size())

                            realy = -1
                            for q in range(data.y.size(0)):
                                if int(data.y[q]) != 0:
                                    realy = q
                                    print(realy)
                                    break
                            assert realy != -1
                            self.all_labels[realy] = 1

                            data_list_realy.append(realy)
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name for split label dataset for training.')

        import random

        for i in range(len(data_list)):
            # tmp_y = data_list[i].y

            tmp_y = data_list_realy[i]

            if tmp_y not in self.is_a_label:
                self.is_a_label[tmp_y] = [i]
            else:
                self.is_a_label[tmp_y].append(i)

            for j in self.all_labels:
                if j == tmp_y:
                    continue
                if j not in self.not_a_label:
                    self.not_a_label[j] = [i]
                else:
                    self.not_a_label[j].append(i)

        data_list_triplet = []
        data_smiles_list_triplet = []

        for k in range(self.k):
            for i in range(len(data_list)):
                if i % 100 == 0:
                    print("i = ", i, " data_list.length = ", len(data_list))
                # tmp_y = data_list[i].y
                tmp_y = data_list_realy[i]
                pos_len = len(self.is_a_label[tmp_y])
                pos_idx = -1
                while True:
                    pos_idx = self.is_a_label[tmp_y][random.randint(0, pos_len - 1)]
                    if pos_idx != i:
                        break
                neg_len = len(self.not_a_label[tmp_y])
                neg_idx = -1
                while True:
                    neg_idx = self.not_a_label[tmp_y][random.randint(0, neg_len - 1)]
                    if neg_idx != i:
                        break
                data_list_triplet.append(data_list[i])
                data_list_triplet.append(data_list[pos_idx])
                data_list_triplet.append(data_list[neg_idx])

                data_smiles_list_triplet.append(data_smiles_list[i])
                data_smiles_list_triplet.append(data_smiles_list[pos_idx])
                data_smiles_list_triplet.append(data_smiles_list[neg_idx])

        self.total = len(data_smiles_list_triplet)
        if self.pre_filter is not None:
            data_list_triplet = [data for data in data_list_triplet if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_triplet = [self.pre_transform(data) for data in data_list_triplet]

        data_smiles_series = pd.Series(data_smiles_list_triplet)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, "smiles_triplet.csv"), index=False, header=False)

        data, slices = self.collate(data_list_triplet)
        torch.save((data, slices), self.processed_paths[0])

        # self.total = len(smiles_list)
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        #
        # # write data_smiles_list in processed paths
        # data_smiles_series = pd.Series(data_smiles_list)
        # data_smiles_series.to_csv(os.path.join(self.processed_dir,
        #                                        'smiles.csv'), index=False,
        #                           header=False)
        #
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])



class SubstructDataset(MoleculeDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        super(SubstructDataset, self).__init__(root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=pre_filter,
                 dataset=dataset,
                 empty=empty)
        self.mask_obj = 0
        self.pree_transform = transform

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                               slices[idx + 1])
            data[key] = item[s]
        data = self.pree_transform(data, mask_type=self.mask_obj)
        # data_corupt = data.clone()
        # data_corupt = self.pre_transform(data_corupt)
        # data = self.pre_transform(data, self.mask_obj)

        return data


class ContrastDataset(MoleculeDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 args,  ## add args!!! todo
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        super(ContrastDataset, self).__init__(root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=pre_filter,
                 dataset=dataset,
                 empty=empty)
        # mask node, mask edge, mask substruct, mask the total node or edge ---- advanced----drop nodes or edges ---
        # can be useful ??
        self.mask_obj = 0
        self.pree_transform =transform
        #self.pre_transform = PairwiseNodeEdgeMask(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge)


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                               slices[idx + 1])
            data[key] = item[s]
        # data_corupt = data.clone()
        # data_corupt = self.pre_transform(data_corupt)

        data = self.pree_transform(data, mask_type=self.mask_obj)
        self.mask_obj = (self.mask_obj + 1) % 3

        return data


class ContrastDataset2(MoleculeDataset):
    def __init__(self,
                 root,
                 args,  ## add args!!! todo
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        super(ContrastDataset2, self).__init__(root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=pre_filter,
                 dataset=dataset,
                 empty=empty)
        # mask node, mask edge, mask substruct, mask the total node or edge ---- advanced----drop nodes or edges ---
        # can be useful ??
        self.mask_obj = 0
        self.pree_transform =transform

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        data = self.pree_transform(data, mask_type=self.mask_obj)
        self.mask_obj = (self.mask_obj + 1) % 3

        return data


def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

def create_circular_fingerprint(mol, radius, size, chirality):
    """

    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)

class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
        processed_fp dir. The raw dir should contain the file containing the
        smiles, and the processed_fp dir can either be empty or a
        previously processed file
        :param dataset: name of dataset. Currently only implemented for
        tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information
        """
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality

        self._load()

    def _process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'chembl_with_labels':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
            print('processing')
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    fp_arr = create_circular_fingerprint(rdkit_mol,
                                                         self.radius,
                                                         self.size, self.chirality)
                    fp_arr = torch.tensor(fp_arr)
                    # manually add mol id
                    id = torch.tensor([i])  # id here is the index of the mol in
                    # the dataset
                    y = torch.tensor(labels[i, :])
                    # fold information
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y,
                                      'fold': fold})
                    data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(os.path.join(self.root, 'raw/tox21.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                        self.radius,
                                                        self.size,
                                                        self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor(labels[i, :])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(os.path.join(self.root, 'raw/HIV.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                        self.radius,
                                                        self.size,
                                                        self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor([labels[i]])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, 'processed_fp')
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False,
                                  header=False)
        with open(os.path.join(processed_dir,
                                    'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(processed_dir,
                                   'fingerprint_data_processed.pkl'),
                      'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ## if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(self.root, self.dataset, self.radius, self.size, chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)   # 0 -> train
    folds = folds.replace('Valid', 1)   # 1 -> valid
    folds = folds.replace('Test', 2)    # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                                          rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values
# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values
# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
       'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
       'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.value

def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f=open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds=pickle.load(f)
    f.close()

    f=open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat=pickle.load(f)
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)
    f.close()

    targetMat=targetMat
    targetMat=targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd=targetAnnInd
    targetAnnInd=targetAnnInd-targetAnnInd.min()

    folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed=targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData=targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f=open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr=pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        # if i == 1000:
        #     break
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]   # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData
# root_path = 'dataset/chembl_with_labels'

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def create_all_datasets():
    #### create dataset
    downstream_dir = [
            'bace',
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'muv',
            'sider',
            'tox21',
            'toxcast'
            ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeDataset(root, dataset=dataset_name)
        print(dataset)


    dataset = MoleculeDataset(root = "dataset/chembl_filtered", dataset="chembl_filtered")
    print(dataset)
    dataset = MoleculeDataset(root = "dataset/zinc_standard_agent", dataset="zinc_standard_agent")
    print(dataset)


# test MoleculeDataset object
if __name__ == "__main__":

    create_all_datasets()

