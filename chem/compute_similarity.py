from rdkit import Chem, DataStructs
from typing import List
import numpy as np
import torch
import pandas as pd
import random
import os
import multiprocessing
from rdkit.Chem import Descriptors, PandasTools
# # from rdkit import Chem
# from rdkit.Chem import Draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_similarity_dataset(idx_range=None, dataset="zinc_standard_agent"):
    print(multiprocessing.current_process().name + " started!")
    print(idx_range)
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    # print(zinc_id_list)
    num_atoms_per_sample = 20

    sim_idx_to_score_dict_list = list()
    DEBUG = False

    l, r = 0, len(smiles_list)
    if idx_range is not None:
        l, r = idx_range

    # idx = l

    for i in range(l, r):
        smile_a = smiles_list[i]
        # for i, smile_a in enumerate(smiles_list):
        sampled_idx = random.sample(range(lenn), num_atoms_per_sample)
        if DEBUG:
            print(i)
        if i % 100 == 0:
            print(i)
        if DEBUG and i >= 100:
            break
        assert isinstance(smile_a, str)
        if DEBUG and i == 0:
            print("sampled_idx")
            print(sampled_idx)
        sim_idx_to_score_dict = {}
        for smile_b_idx in sampled_idx:
            smile_b = smiles_list[smile_b_idx]
            assert isinstance(smile_b, str)

            sim_score = compute_similarity([smile_a, smile_b])
            sim_idx_to_score_dict[smile_b_idx] = sim_score
        sim_idx_to_score_dict_list.append(sim_idx_to_score_dict.copy())
    print(len(sim_idx_to_score_dict_list))
    np.save(os.path.join(processed_path, "sim_to_score_dict_list_{}.npy".format(str(l))), sim_idx_to_score_dict_list)


"""
compute similarities 
"""
def compute_similarity_dataset_based_on_wei_all(num_cpu=20, dataset="zinc_standard_agent", DEBUGE=True):
    print(multiprocessing.current_process().name + " started!")
    # print(idx_range)
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    # print(zinc_id_list)
    # num_atoms_per_sample = 20

    idx_wei_items_sorted_list = np.load(os.path.join(processed_path, "idx_to_weis_list_wei_nc_natom_sorted.npy"))
    # sim_idx_to_score_dict_dict = dict()
    assert len(idx_wei_items_sorted_list) == lenn

    len_per = lenn // num_cpu

    if DEBUGE:
        len_per = 10

    if num_cpu > 1:
        lis = [[i * len_per, (i + 1) * len_per] for i in range(num_cpu)]
        pool = multiprocessing.Pool(processes=num_cpu)
        results = list()
        for i in range(len(lis)):
            results.append(pool.apply_async(compute_sim_based_on_wei_per_sec, ([lis[i][0], lis[i][1]],)))
        pool.close()
        pool.join()
        sim_idx_to_score_dict_dict = dict()

        #### TODO: check the order?
        for res in results:
            res_dict = res.get()
            for now_idx in res_dict:
                assert now_idx not in sim_idx_to_score_dict_dict
                sim_idx_to_score_dict_dict[now_idx] = res_dict[now_idx]
            print(len(sim_idx_to_score_dict_dict))
    else:
        print("In main process.")
        sim_idx_to_score_dict_dict = compute_sim_based_on_wei_per_sec([0, lenn])

    # ch = list()
    chosen_idx = range(lenn)
    if DEBUGE:
        lenn = len_per * num_cpu
        chosen_idx = [int(idx_wei_items_sorted_list[i][0]) for i in range(lenn)]

    assert len(sim_idx_to_score_dict_dict) == lenn

    sim_idx_to_score_dict_list = list()
    for i in chosen_idx:
        sim_idx_to_score_dict_list.append(sim_idx_to_score_dict_dict[i])

    print(len(sim_idx_to_score_dict_list))
    np.save(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"), sim_idx_to_score_dict_list)

def compute_unsimilarity_dataset_based_on_wei_all(num_cpu=20, dataset="zinc_standard_agent", DEBUGE=True):
    print(multiprocessing.current_process().name + " started!")
    # print(idx_range)
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    # print(zinc_id_list)
    # num_atoms_per_sample = 20

    idx_wei_items_sorted_list = np.load(os.path.join(processed_path, "idx_to_weis_list_wei_nc_natom_sorted.npy"))
    # sim_idx_to_score_dict_dict = dict()
    assert len(idx_wei_items_sorted_list) == lenn

    len_per = lenn // num_cpu

    if DEBUGE:
        len_per = 10

    if num_cpu > 1:
        lis = [[i * len_per, (i + 1) * len_per] for i in range(num_cpu)]
        pool = multiprocessing.Pool(processes=num_cpu)
        results = list()
        for i in range(len(lis)):
            results.append(pool.apply_async(compute_unsim_based_on_wei_per_sec, ([lis[i][0], lis[i][1]],)))
        pool.close()
        pool.join()
        sim_idx_to_score_dict_dict = dict()

        #### TODO: check the order?
        for res in results:
            res_dict = res.get()
            for now_idx in res_dict:
                assert now_idx not in sim_idx_to_score_dict_dict
                sim_idx_to_score_dict_dict[now_idx] = res_dict[now_idx]
            print(len(sim_idx_to_score_dict_dict))
    else:
        print("In main process.")
        sim_idx_to_score_dict_dict = compute_unsim_based_on_wei_per_sec([0, lenn])

    # ch = list()
    chosen_idx = range(lenn)
    if DEBUGE:
        lenn = len_per * num_cpu
        chosen_idx = [int(idx_wei_items_sorted_list[i][0]) for i in range(lenn)]

    assert len(sim_idx_to_score_dict_dict) == lenn

    sim_idx_to_score_dict_list = list()
    for i in chosen_idx:
        sim_idx_to_score_dict_list.append(sim_idx_to_score_dict_dict[i])

    print(len(sim_idx_to_score_dict_list))
    np.save(os.path.join(processed_path, "unsim_to_score_dict_list_wei_nc_natom_dis_wei.npy"), sim_idx_to_score_dict_list)


def check_cond(s_cond, t_cond):
    # 如果环数量相差1 则返回False
    if abs(int(s_cond[1]) - int(t_cond[1])) > 1:
        return False
    # 如果原子数量相差大于7 则返回False
    if abs(int(s_cond[2]) - int(t_cond[2])) > 7:
        return False
    return True


def gen_candi_oneway(mol_list, mol_idx, start, end, direction, max_candi_len=70):
    target = mol_list[mol_idx]
    t_cond = target[1]
    candi_idx = []
    tot_candi = 0
    for i in range(start, end, direction):
        s_cond = mol_list[i][1]
        # 分子量相差百分比不能大于20% 如果大于直接break（mol list已经排序）
        if abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) > 0.1:
            break
        if check_cond(s_cond, t_cond):
            # Debug only
            # print("Found candidate: %s\nwt: %.1f, SSSR: %d Natom: %d"
            #      %(str(mol_list[i][0]),
            #        float(s_cond[0]),
            #        int(s_cond[1]),
            #        int(s_cond[2])))
            if direction == -1:
                candi_idx.insert(0, i)
            else:
                candi_idx.append(i)
            tot_candi += 1
            if tot_candi >= max_candi_len:
                break
    return candi_idx


def gen_neg_candi_oneway(mol_list, mol_idx, start, end, direction, max_candi_len=70):
    target = mol_list[mol_idx]
    t_cond = target[1]
    candi_idx = []
    tot_candi = 0
    for i in range(start, end, direction):
        s_cond = mol_list[i][1]
        # 分子量相差百分比不能大于20% 如果大于直接break（mol list已经排序）
        if (abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) > 0.2) or (abs(t_cond[0] - s_cond[0]) > 150):
            break
        if not check_cond(s_cond, t_cond):
            # Debug only
            # print("Found candidate: %s\nwt: %.1f, SSSR: %d Natom: %d"
            #      %(str(mol_list[i][0]),
            #        float(s_cond[0]),
            #        int(s_cond[1]),
            #        int(s_cond[2])))
            if direction == -1:
                candi_idx.insert(0, i)
            else:
                candi_idx.append(i)
            tot_candi += 1
            if tot_candi >= max_candi_len:
                break
    return candi_idx

def gen_neg_candi_oneway_dis_wei(mol_list, mol_idx, start, end, direction, max_candi_len=70):
    target = mol_list[mol_idx]
    t_cond = target[1]
    candi_idx = []
    tot_candi = 0
    for i in range(start, end, direction):
        s_cond = mol_list[i][1]
        # 分子量相差百分比不能大于20% 如果大于直接break（mol list已经排序）
        if (abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) <= 0.2):
            continue
        if (abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) > 0.4):
            break
        # if (abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) > 0.2) or (abs(t_cond[0] - s_cond[0]) > 150):
        #     break
        if direction == -1:
            candi_idx.insert(0, i)
        else:
            candi_idx.append(i)
        tot_candi += 1
        if (tot_candi >= max_candi_len):
            break
    return candi_idx

def gen_neg_candi_oneway_sim_wei(mol_list, mol_idx, start, end, direction, max_candi_len=70):
    target = mol_list[mol_idx]
    t_cond = target[1]
    candi_idx = []
    tot_candi = 0
    for i in range(start, end, direction):
        s_cond = mol_list[i][1]
        # 分子量相差百分比不能大于20% 如果大于直接break（mol list已经排序）
        if (abs(float(t_cond[0]) - float(s_cond[0])) / float(t_cond[0]) > 0.2) or (abs(t_cond[0] - s_cond[0]) > 150):
            break
        if not check_cond(s_cond, t_cond):
            if direction == -1:
                candi_idx.insert(0, i)
            else:
                candi_idx.append(i)
            tot_candi += 1
            if tot_candi >= max_candi_len:
                break
    return candi_idx


import time
def compute_sim_based_on_wei_per_sec(infos, max_candi_len=70):
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    # print(zinc_id_list)
    # num_atoms_per_sample = 20

    idx_wei_items_sorted_list = np.load(os.path.join(processed_path, "idx_to_weis_list_wei_nc_natom_sorted.npy"))
    print("loaded")
    i_l, i_r = infos
    lenn = len(smiles_list)
    print(lenn, i_l, i_r)
    sim_idx_to_score_dict_dict = dict()
    stt = 0
    edt = 0
    cumsum_sims = 0
    for i in range(i_l, i_r):
        if i % 100 == 0:
            edt = time.time()
            print(i, edt - stt, cumsum_sims / 100.0)
            stt = time.time()
            cumsum_sims = 0
        now_idx, now_wei, now_nc, now_natom = int(idx_wei_items_sorted_list[i][0]), float(idx_wei_items_sorted_list[i][1][0]), \
                int(idx_wei_items_sorted_list[i][1][1]), int(idx_wei_items_sorted_list[i][1][2]),

        res = []

        res.extend(gen_candi_oneway(idx_wei_items_sorted_list, i, i - 1, -1, -1))
        res.extend(gen_candi_oneway(idx_wei_items_sorted_list, i, i + 1, lenn, 1))


        if len(res) > max_candi_len:
            l_gap = (len(res) - max_candi_len) // 2
            res = res[l_gap: l_gap + max_candi_len]

        sim_dict = dict()

        for j in res:
            assert j != i
            other_idx = int(idx_wei_items_sorted_list[j][0])
            sim_score = compute_similarity([smiles_list[now_idx], smiles_list[other_idx]])
            sim_dict[other_idx] = sim_score
        # assert len(sim_dict) == 20 # if the candidate is empty --- should be
        # print(len(sim_dict))
        cumsum_sims += len(sim_dict)
        sim_idx_to_score_dict_dict[now_idx] = sim_dict
    return sim_idx_to_score_dict_dict

def compute_unsim_based_on_wei_per_sec(infos, max_candi_len=70):
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    # print(zinc_id_list)
    # num_atoms_per_sample = 20

    idx_wei_items_sorted_list = np.load(os.path.join(processed_path, "idx_to_weis_list_wei_nc_natom_sorted.npy"))
    print("loaded")
    i_l, i_r = infos
    lenn = len(smiles_list)
    print(lenn, i_l, i_r)
    sim_idx_to_score_dict_dict = dict()
    stt = 0
    edt = 0
    cumsum_sims = 0
    for i in range(i_l, i_r):
        if i % 100 == 0:
            edt = time.time()
            print(i, edt - stt, cumsum_sims / 100.0)
            stt = time.time()
            cumsum_sims = 0
        now_idx, now_wei, now_nc, now_natom = int(idx_wei_items_sorted_list[i][0]), float(idx_wei_items_sorted_list[i][1][0]), \
                int(idx_wei_items_sorted_list[i][1][1]), int(idx_wei_items_sorted_list[i][1][2]),

        res = []

        # res.extend(gen_neg_candi_oneway(idx_wei_items_sorted_list, i, i - 1, -1, -1))
        # res.extend(gen_neg_candi_oneway(idx_wei_items_sorted_list, i, i + 1, lenn, 1))

        res.extend(gen_neg_candi_oneway_dis_wei(idx_wei_items_sorted_list, i, i - 1, -1, -1))
        res.extend(gen_neg_candi_oneway_dis_wei(idx_wei_items_sorted_list, i, i + 1, lenn, 1))

        # print(len(res))

        if len(res) > max_candi_len:
            l_gap = (len(res) - max_candi_len) // 2
            res = res[l_gap: l_gap + max_candi_len]

        sim_dict = dict()

        for j in res:
            assert j != i
            other_idx = int(idx_wei_items_sorted_list[j][0])
            sim_score = compute_similarity([smiles_list[now_idx], smiles_list[other_idx]])
            sim_dict[other_idx] = sim_score
        # assert len(sim_dict) == 20 # if the candidate is empty --- should be
        # print(len(sim_dict))
        cumsum_sims += len(sim_dict)
        sim_idx_to_score_dict_dict[now_idx] = sim_dict
    return sim_idx_to_score_dict_dict



def test_load():
    processed_path = "./dataset/zinc_standard_agent/processed"
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_0.npy"))
    print(sim_to_score_dict_list)
    print(len(sim_to_score_dict_list))
    print(len(sim_to_score_dict_list[0]))


def compute_all(n_cpu=20):
    all_len = 2000000
    len_per = all_len // n_cpu
    lis = [[i * len_per, (i + 1) * len_per] for i in range(n_cpu)]
    pool = multiprocessing.Pool(processes=n_cpu)
    for i in range(len(lis)):
        # msg = "hello %d" % (i)
        pool.apply_async(compute_similarity_dataset, (lis[i],))
    pool.close()
    pool.join()


"""
extract statistic information from dataset 
"""
def extract_mol_weis_from_dataset_all(num_cpu=20, dataset="zinc_standard_agent", DEBUGE=True):
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])

    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    assert lenn == 2000000
    len_per = lenn // num_cpu

    if DEBUGE:
        len_per = 10
        lenn = num_cpu * len_per

    lis = [[i * len_per, (i + 1) * len_per] for i in range(num_cpu)]
    pool = multiprocessing.Pool(processes=num_cpu)
    results = list()
    for i in range(len(lis)):
        results.append(pool.apply_async(compute_mol_wei, (smiles_list[lis[i][0]: lis[i][1]], )))
    pool.close()
    pool.join()
    weis = list()

    #### TODO: check the order?
    for res in results:
        weis += res.get()
        print(len(weis))
    assert len(weis) == lenn
    # weis = compute_mol_wei(smiles_list)
    idx_to_weis = {i: wei for i, wei in enumerate(weis)}
    print(len(idx_to_weis))
    np.save(os.path.join(processed_path, "idx_to_weis_dict_wei_nc_natom.npy"), idx_to_weis)
    idx_to_weis_sorted = sorted(idx_to_weis.items(), key=lambda item: item[1][0], reverse=False)
    np.save(os.path.join(processed_path, "idx_to_weis_list_wei_nc_natom_sorted.npy"), idx_to_weis_sorted)
    assert len(idx_to_weis_sorted) == len(idx_to_weis)
    print(idx_to_weis_sorted[0], idx_to_weis_sorted[-1])


def extract_mol_weis_from_dataset(dataset="zinc_standard_agent"):
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    # print(input_df)
    # smiles_list = list(input_df)
    zinc_id_list = list(input_df['zinc_id'])
    lenn = len(smiles_list)
    print("length of this dataset = ", len(smiles_list))
    print(smiles_list[0], smiles_list[1])
    weis = compute_mol_wei(smiles_list)
    idx_to_weis = {i: wei for i, wei in enumerate(weis)}
    print(len(idx_to_weis))
    np.save(os.path.join(processed_path, "idx_to_weis_dict.npy"), idx_to_weis)
    idx_to_weis_sorted = sorted(idx_to_weis.items(), key=lambda item: item[1][0], reverse=False)
    np.save(os.path.join(processed_path, "idx_to_weis_list_sorted.npy"), idx_to_weis_sorted)
    assert len(idx_to_weis_sorted) == len(idx_to_weis)
    print(idx_to_weis_sorted[0], idx_to_weis_sorted[-1])


def compute_mol_wei(smiles: List[str]):
    print(len(smiles))

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    weis = [[Descriptors.ExactMolWt(mol), len(Chem.GetSymmSSSR(mol)), len(mol.GetAtoms())] for mol in mols]

    print(type(weis))
    print(type(weis[0]), weis[0])
    return weis

def compute_similarity(batch_smiles: List[str]):
    """
        compute the similarity between molecules in a batch

        batch_smiles: a list of smiles str represent a batch of molecules, batch_size must be even

        return:
            a torch.tensor represents the similarity between the i th molecule and i+batch_size/2 th molecule
                shape : (batch_size/2, 1)
                dtype : torch.float32
    """

    molecules = [Chem.MolFromSmiles(smiles) for smiles in batch_smiles]
    fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]
    batch_size = len(batch_smiles)
    assert batch_size == 2
    mol_a, mol_b = fingerprints[0], fingerprints[1]
    sim_score = DataStructs.FingerprintSimilarity(mol_a, mol_b)
    return sim_score
    # assert batch_size % 2 == 0
    # pair_num = int(batch_size / 2)
    # sims = np.empty((pair_num, 1))
    # for i in range(0, pair_num):
    #     sims[i][0] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[i + pair_num])
    # return torch.tensor(sims, dtype=torch.float32)

def compute_similarity_1vn(base_smiles, batch_smiles: List[str]):
    """
        compute the similarity between molecules in a batch

        batch_smiles: a list of smiles str represent a batch of molecules, batch_size must be even

        return:
            a torch.tensor represents the similarity between the i th molecule and i+batch_size/2 th molecule
                shape : (batch_size/2, 1)
                dtype : torch.float32
    """

    molecules = [Chem.MolFromSmiles(smiles) for smiles in batch_smiles]
    fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]
    base_mol = Chem.MolFromSmiles(base_smiles)
    base_finger = Chem.RDKFingerprint(base_mol)
    res = list()
    for other_finger in fingerprints:
        res.append(DataStructs.FingerprintSimilarity(base_finger, other_finger))
    assert len(res) == len(batch_smiles)
    return res


def compute_similarity_batch(batch_smiles: List[str]):
    """
        compute the similarity between molecules in a batch

        batch_smiles: a list of smiles str represent a batch of molecules, batch_size must be even

        return:
            a torch.tensor represents the similarity between the i th molecule and i+batch_size/2 th molecule
                shape : (batch_size/2, 1)
                dtype : torch.float32
    """
    molecules = [Chem.MolFromSmiles(smiles) for smiles in batch_smiles]
    fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]
    batch_size = len(batch_smiles)
    assert batch_size % 2 == 0
    pair_num = int(batch_size / 2)
    sims = np.empty((pair_num, 1))
    for i in range(0, pair_num):
        sims[i][0] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[i + pair_num])
    return torch.tensor(sims, dtype=torch.float32)


def check_unsim_computed():
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    unsim_list = np.load(os.path.join(processed_path, "unsim_to_score_dict_list_wei_nc_natom.npy"))
    print(len(unsim_list), type(unsim_list))
    print(type(unsim_list[0]))
    per_sim_dict = unsim_list[0]
    print(per_sim_dict)
    # other_idx = list(per_sim_dict.keys())[0]
    # print(smiles_list[0])
    # print(smiles_list[other_idx])
    for j in range(len(unsim_list)):
        print(len(unsim_list[j]))


import math
def get_normalized_factor():
    # input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    # input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    # smiles_list = list(input_df['smiles'])

    lenn = 2000000
    # assert len(smiles_list) == lenn
    # self.calculated_sim_list = [torch.empty((0, ), dtype=torch.long) for i in range(lenn)]
    # self.calculated_sim_score = [torch.empty((0, ), dtype=torch.float64) for i in range(lenn)]
    #
    # len_per = lenn // 20
    # ls = [i * len_per for i in range(20)]
    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))
    # print(sim_to_score_dict_list)
    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])  # the pushed item is a node_idx to score dict
        # self.calculated_sim_list.append(torch.empty((0, ), dtype=torch.long))
        # self.calculated_sim_score.append(torch.empty((0, ), dtype=torch.float64))
        # self.pos_idxes += sim_to_score_dict_list
    assert len(pos_idxes) == lenn
    T = 4

    # TODO: we need to get a pre_computed result which is the sumation of the similarity between nodes (having edges.)
    # and assume it is the "self.idx_to_sim_sum" which is a dict.
    # we need to define a T --- the

    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    assert isinstance(idx_to_adjs, dict) and len(idx_to_adjs) == lenn
    idx_to_adj_list_dict = idx_to_adjs

    idx_to_normalized_factor_dict = dict()
    for now_idx in idx_to_adjs:
        cumsum_normalized_factor = 0.0
        for other_idx in idx_to_adj_list_dict[now_idx]:
            cumsum_normalized_factor += math.exp(pos_idxes[now_idx][other_idx] / T)
        idx_to_normalized_factor_dict[now_idx] = cumsum_normalized_factor
    np.save(os.path.join(processed_path, "idx_to_sim_sum.npy"), idx_to_normalized_factor_dict)
    print(len(idx_to_normalized_factor_dict))
    assert len(idx_to_normalized_factor_dict) == lenn

def check_normalized_factor():
    processed_path = "./dataset/zinc_standard_agent/processed"
    idx_to_normalized_factor = np.load(os.path.join(processed_path, "idx_to_sim_sum.npy")).item()
    print(len(idx_to_normalized_factor))
    for j in range(30):
        print(idx_to_normalized_factor[j])

# import dgl
def rwr_sample_neg_graphs(dgl_bg, idx, num_samples=5, exc_idx=None):
    # print(type(dgl_bg))
    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [idx],
        prob='p',
        restart_prob=0,
        length=num_samples + 1)
    # print(traces)
    subv = torch.unique(traces[0]).tolist()
    try:
        subv.remove(idx)
    except ValueError:
        pass
    return subv

#/home/lxyww7/grover/dmpnn-automl/data

def get_mol_sim_smiles():
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])

    pos_idxes = list()
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    smiles_to_sim_smiles_dict_dict = dict()
    smiles_to_nei_smiles_dict_dict = dict()
    for j in range(10):
        candi_idx_to_score_dict = pos_idxes[j]
        tot = 0
        t_smiles = smiles_list[j]
        smiles_to_sim_smiles_dict_dict[t_smiles] = dict()
        smiles_to_nei_smiles_dict_dict[t_smiles] = dict()
        for candi_idx in candi_idx_to_score_dict:
            if tot < 10:
                smiles_to_sim_smiles_dict_dict[t_smiles][smiles_list[candi_idx]] = candi_idx_to_score_dict[candi_idx]
                tot += 1
            else:
                break
        tot_nei = 0
        for candi_idx in candi_idx_to_score_dict:
            nei_candi = pos_idxes[candi_idx]
            for nei_nei in nei_candi:
                neinei_smile = smiles_list[nei_nei]
                if (neinei_smile not in smiles_to_sim_smiles_dict_dict[t_smiles]):
                    tot_nei += 1
                    smiles_to_nei_smiles_dict_dict[t_smiles][neinei_smile] = 1
                if (tot_nei >= 10):
                    break
            if tot_nei >= 10:
                break

    print(len(smiles_to_sim_smiles_dict_dict))
    print(len(smiles_to_nei_smiles_dict_dict))
    np.save(os.path.join(processed_path, "sampled_sim_mols.npy"), smiles_to_sim_smiles_dict_dict)
    np.save(os.path.join(processed_path, "sampled_sim_nei_mols.npy"), smiles_to_nei_smiles_dict_dict)

def plot_smiles():
    processed_path = "./dataset/zinc_standard_agent/processed"
    smiles_to_sim_smiles_dict_dict = np.load(os.path.join(processed_path, "sampled_sim_mols.npy")).item()
    smiles_to_nei_smiles_dict_dict = np.load(os.path.join(processed_path, "sampled_sim_nei_mols.npy")).item()
    idxx = 0
    for t_smile in smiles_to_sim_smiles_dict_dict:
        s_smiles = [s_smile for s_smile in smiles_to_sim_smiles_dict_dict[t_smile]]
        s_smiles = [t_smile] + s_smiles
        s_smiles = [Chem.MolFromSmiles(s_smile) for s_smile in s_smiles]
        img = Draw.MolsToGridImage(s_smiles, molsPerRow=6, subImgSize=(300, 300))
        # s_smile_list = smiles_to_sim_smiles_dict_dict[t_smile]
        img.save("figs/" + str(idxx) + ".png")

        nei_s_smiles = [s_smile for s_smile in smiles_to_nei_smiles_dict_dict[t_smile]]
        nei_s_smiles = [t_smile] + nei_s_smiles
        nei_s_smiles = [Chem.MolFromSmiles(s_smile) for s_smile in nei_s_smiles]
        img = Draw.MolsToGridImage(nei_s_smiles, molsPerRow=6, subImgSize=(300, 300))
        img.save("figs/" + str(idxx) + "_nei_nei.png")


        idxx += 1

def get_big_graph_edge_list_all(cpu_num=20, DEBUG=True):
    len = 2000000 if DEBUG == False else 400
    processed_path = "./dataset/zinc_standard_agent/processed"
    num_per_cpu = len // cpu_num
    if (num_per_cpu * cpu_num < len):
        cpu_num += 1
    res = list()
    pool = multiprocessing.Pool(cpu_num)
    for i in range(cpu_num):
        st, ed = i * num_per_cpu, min((i + 1) * num_per_cpu, len)
        res.append(pool.apply_async(construct_big_graph_edge_list,
                                    args=(st, ed,)))
    pool.close()
    pool.join()
    edge_list_with_weight = []
    # for r in res:
    #     tmp_list = r.get()
    #     print(tmp_list)
    #     edge_list_with_weight += tmp_list
    #     # print(len(edge_list_with_weight))
    #     print(len(edge_list_with_weight))
    # np.save(os.path.join(processed_path, "sim_edge_list.npy"), edge_list_with_weight)


def construct_big_graph_edge_list(st_node, ed_node):
    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    # assert len(self.pos_idxes) == lenn

    # TODO: we need to get a pre_computed result which is the sumation of the similarity between nodes (having edges.)
    # and assume it is the "self.idx_to_sim_sum" which is a dict.
    # we need to define a T --- the

    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    # assert isinstance(idx_to_adjs, dict) and len(idx_to_adjs) == lenn
    # idx to self adjacency nodes --- selected by similarity
    idx_to_adj_list_dict = idx_to_adjs
    print("loaded", st_node, ed_node)
    # idx_to_adj_list_dict
    edge_list_with_weight = []
    for inode in range(st_node, ed_node):
        if inode % 100 == 0:
            print(inode, len(edge_list_with_weight))
        edge_list_with_weight += [(inode, other_node, pos_idxes[inode][other_node]) \
                                  for other_node in idx_to_adj_list_dict[inode]]
        # edge_list_with_weight.extend()
    assert type(edge_list_with_weight) == list
    np.save(os.path.join(processed_path, "sim_edge_list_{:d}.npy".format(st_node)), edge_list_with_weight)
    # return edge_list_with_weight
    print("saved", st_node)

import networkx as nx
def test_saved_list():
    processed_path = "./dataset/zinc_standard_agent/processed"
    tmp = np.load(os.path.join(processed_path, "sim_edge_list_0.npy"))
    g = nx.Graph()
    g.add_weighted_edges_from(tmp)
    print(g.number_of_nodes(), g.number_of_edges())
    print(tmp)

def get_distant_nodes_list():
    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    # assert len(self.pos_idxes) == lenn

    # TODO: we need to get a pre_computed result which is the sumation of the similarity between nodes (having edges.)
    # and assume it is the "self.idx_to_sim_sum" which is a dict.
    # we need to define a T --- the

    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    # assert isinstance(idx_to_adjs, dict) and len(idx_to_adjs) == lenn
    # idx to self adjacency nodes --- selected by similarity
    idx_to_adj_list_dict = idx_to_adjs
    lenn = 2000000

    nxg = nx.Graph()
    nxg.add_nodes_from([inode for inode in range(0, len(idx_to_adj_list_dict))])
    num_cpu = 20
    num_per_cpu = lenn // num_cpu
    for ii in range(num_cpu):
        print(ii)
        st_node_idx = ii * num_per_cpu
        tmp_edge_list = np.load(os.path.join(processed_path, "sim_edge_list_{:d}.npy".format(st_node_idx)))
        print(len(tmp_edge_list))
        nxg.add_weighted_edges_from(tmp_edge_list)
    # print(self.nxg.number_of_edges(), self.nxg.number_of_nodes())

import pickle
def transfer_pickle_error():
    processed_path = "./dataset/zinc_standard_agent/processed"
    with open(os.path.join(processed_path, "big_graph.pkl"), "rb") as f:
        nxg = pickle.load(f)
        f.close()
    print("loaded")
    with open(os.path.join(processed_path, "big_graph.pkl"), "wb") as f:
        pickle.dump(nxg.nxg, f)
    print("ok")

def view_neg_samples_similarity():
    processed_path = "./dataset/zinc_standard_agent/processed"
    neg_sim_dict = np.load(os.path.join(processed_path, "unsim_to_score_dict_list_wei_nc_natom_dis_wei.npy"))
    print(len(neg_sim_dict))
    all_sim_scores = list()
    for st_node, aa in enumerate(neg_sim_dict):
        for ed_node in neg_sim_dict[st_node]:
            all_sim_scores.append(neg_sim_dict[st_node][ed_node])
    all_sim_scores = sorted(all_sim_scores, reverse=False)
    print(len(all_sim_scores))
    print(all_sim_scores[len(all_sim_scores) // 3 * 2], all_sim_scores[len(all_sim_scores) // 2])
    # cutoff = all_sim_scores[len(all_sim_scores) // 2]
    cutoff = 0.1
    idx_to_idx_sim_dict = dict()
    for st_node, idx_to_sim in enumerate(neg_sim_dict):
        idx_to_sim_list = list()

        for ed_node in idx_to_sim:
            score = idx_to_sim[ed_node]
            if score <= cutoff:
                idx_to_sim_list.append(ed_node)
        idx_to_idx_sim_dict[st_node] = idx_to_sim_list
        if st_node % 100 == 0:
            print(st_node, len(idx_to_sim_list))
    np.save(os.path.join(processed_path, "unsim_node_adj_list_cutoff_{}.npy".format(str(cutoff))), idx_to_idx_sim_dict)



def compare_distri_sim():
    processed_path = "./dataset/zinc_standard_agent/processed"
    neg_sim_dict = np.load(os.path.join(processed_path, "unsim_to_score_dict_list_wei_nc_natom_dis_wei.npy"))
    print(len(neg_sim_dict))
    target_node_idx = random.choice(range(len(neg_sim_dict)))
    print(target_node_idx)
    num_unsim_nodes = len(neg_sim_dict[target_node_idx])
    print("num_unsim_nodes", num_unsim_nodes)
    random_nodes = np.random.choice(a=len(neg_sim_dict), size=num_unsim_nodes, replace=False)
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    random_nodes_smiles = [smiles_list[node_idx] for node_idx in random_nodes]
    random_nodes_sim_scores = compute_similarity_1vn(smiles_list[target_node_idx], random_nodes_smiles)
    unsim_nodes_sim_scores = [neg_sim_dict[target_node_idx][node_idx] for node_idx in neg_sim_dict[target_node_idx]]
    random_nodes_sim_scores = sorted(random_nodes_sim_scores, reverse=False)
    unsim_nodes_sim_scores = sorted(unsim_nodes_sim_scores, reverse=False)
    print(random_nodes_sim_scores)
    print(unsim_nodes_sim_scores)
    plt.hist(random_nodes_sim_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.savefig("figs/random_nodes_sim_scores.png")
    plt.hist(unsim_nodes_sim_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.savefig("figs/unsim_nodes_sim_scores.png")


from n2v_graph import Graph
def sample_pos_based_on_sim(idx, pos_idxes, self_num_samples):
    idx_to_sim_score_dict = pos_idxes[idx]
    keys = list()
    values = list()
    for idxbb in idx_to_sim_score_dict:
        keys.append(idxbb)
        values.append(idx_to_sim_score_dict[idxbb])
    if len(keys) > 0:
        values = torch.tensor(values, dtype=torch.float64)
        values = torch.softmax(values, dim=0)
        num_samples = self_num_samples if self_num_samples <= values.size(0) else values.size(0)
        choose_idx = torch.multinomial(values, num_samples)

        if num_samples < self_num_samples:
            keys.append(idx)
            choose_idx = torch.cat([choose_idx, torch.full((self_num_samples - num_samples,), len(keys) - 1,
                                                           dtype=torch.long)], dim=0)
    else:
        choose_idx = [0 for j in range(self_num_samples)]
        keys = [idx for j in range(self_num_samples)]

    # sim_datas = list()
    sampled_pos_idx = [int(keys[int(choose_idx[i])]) for i in range(self_num_samples)]
    return sampled_pos_idx

def get_rhop_ego_nx_graph(idx_to_adj_list_dict, pos_idxes, idx, num_hops=3, max_nodes_num=100): # assume "T" is equal to 4 and self.idx_to_sim_sum[idx] = sum(exp(sim_score / T))
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
        for other_idx in idx_to_adj_list_dict[now_idx]:
            if other_idx not in idx_to_dis:
                new_idx += 1
                # map other_idx (old idx) to new_idx (new idx)
                old_to_new_idx[other_idx] = new_idx
                # set the distance of newly added node
                idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                edge_list_with_weight.append((new_now_idx, new_idx, pos_idxes[now_idx][other_idx]))
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

def sample_neg_based_on_possi(neg_p_q, idx, r_hop_nx_G, new_idx_to_old_idx, idx_to_dis, num_samples, totlen, idx_to_sim_score_dict):
    neg_n2v_G = Graph(r_hop_nx_G, False, neg_p_q[0], neg_p_q[1])
    neg_n2v_G.preprocess_transition_probs()
    neg_walk_nodes = neg_n2v_G.node2vec_walk(num_samples * 7, 0)
    neg_walk_unique_nodes = list(set(neg_walk_nodes))
    try:
        neg_walk_unique_nodes.remove(0)
    except ValueError:
        pass

    neg_walk_unique_nodes = [new_idx_to_old_idx[neg_idx] for neg_idx in neg_walk_unique_nodes]
    neg_walk_unique_nodes = sorted(neg_walk_unique_nodes, key=lambda neg_idx: idx_to_dis[neg_idx])
    neg_walk_unique_nodes = [neg_idx for neg_idx in neg_walk_unique_nodes if neg_idx not in idx_to_sim_score_dict]

    if len(neg_walk_unique_nodes) < num_samples:
        neg_walk_unique_nodes = neg_walk_unique_nodes + [(idx + _) % totlen \
                                                         for _ in range(1, num_samples - len(neg_walk_unique_nodes) + 1)]
    elif len(neg_walk_unique_nodes) > num_samples:
        neg_walk_unique_nodes = neg_walk_unique_nodes[-num_samples:]
    return neg_walk_unique_nodes

def sample_n2v_neg_pos_based_on_possi(neg_p_q, idx, r_hop_nx_G, new_idx_to_old_idx, idx_to_dis, num_samples,
                                      num_neg_samples, totlen, idx_to_sim_score_dict):
    neg_n2v_G = Graph(r_hop_nx_G, False, neg_p_q[0], neg_p_q[1])
    neg_n2v_G.preprocess_transition_probs()
    neg_walk_nodes = neg_n2v_G.node2vec_walk((num_samples + num_neg_samples) * 7, 0)
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
    neg_candi_idxes = neg_walk_unique_nodes[pos_offset + 1: ]
    if len(pos_candi_idxes) < num_samples:
        pos_candi_idxes = pos_candi_idxes + [idx for _ in range(num_samples - len(pos_candi_idxes))]
    elif len(pos_candi_idxes) > num_samples:
        pos_candi_idxes = pos_candi_idxes[: num_samples]


    if len(neg_candi_idxes) < num_neg_samples:
        neg_candi_idxes = neg_candi_idxes + [(idx + _) % totlen \
                                                         for _ in range(1, num_neg_samples - len(neg_candi_idxes) + 1)]
    elif len(neg_candi_idxes) > num_neg_samples:
        neg_candi_idxes = neg_candi_idxes[-num_neg_samples:]
    return pos_candi_idxes, neg_candi_idxes

def get_neg_pos_sim_distri_n2v(num_hops, neg_p_q, num_sim_samples, num_neg_samples):
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])

    lenn = 2000000
    assert len(smiles_list) == lenn

    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    assert len(pos_idxes) == lenn
    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    idx_to_adj_list_dict = idx_to_adjs

    random_nodes = np.random.choice(a=len(smiles_list), size=1000, replace=False)
    # print(random_nodes)
    div_scores = []

    n2v_ps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4]
    for i in range(len(n2v_ps)):
        div_scores.append([])
    for idx in random_nodes:
        sim_sampled_idx = sample_pos_based_on_sim(idx, pos_idxes, num_sim_samples)
        r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = get_rhop_ego_nx_graph(idx_to_adj_list_dict,
                                                                           pos_idxes,
                                                                           idx,
                                                                           num_hops=num_hops)
        for ip, p in enumerate(n2v_ps):
            q = 2.0 - p
            neg_sampled_idx = sample_neg_based_on_possi([p, q],
                                                        idx,
                                                        r_hop_nx_G,
                                                        new_idx_to_old_idx,
                                                        idx_to_dis,
                                                        num_neg_samples,
                                                        len(smiles_list),
                                                        pos_idxes[idx])
            sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                        pos_idx in sim_sampled_idx])
            neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                        neg_idx in neg_sampled_idx])
            div_score = (sum(neg_samples_sim) / num_neg_samples) / (sum(sim_samples_sim) / num_sim_samples + 1e-9)
            # div_scores.append(div_score)
            div_scores[ip].append(div_score)

    print("n2v sampling similarities:")
    for ip, p in enumerate(n2v_ps):
        print(p, sum(div_scores[ip]) / len(div_scores[ip]))
    # plt.hist(div_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.savefig("figs/neg_pos_div_scores_hop_{:d}_num_pos_samples_{:d}_neg_{:d}_p_{}_q_{}.png".format(num_hops,
    #                                                                                          num_sim_samples,
    #                                                                                          num_neg_samples,
    #                                                                                          str(neg_p_q[0]),
    #                                                                                          str(neg_p_q[1])))
    # plt.close()

def get_neg_pos_sim_distri_n2v_one_walk(num_hops, neg_p_q, num_sim_samples, num_neg_samples):
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])

    lenn = 2000000
    assert len(smiles_list) == lenn

    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    assert len(pos_idxes) == lenn
    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    idx_to_adj_list_dict = idx_to_adjs

    random_nodes = np.random.choice(a=len(smiles_list), size=1000, replace=False)
    # print(random_nodes)
    div_scores = []
    for idx in random_nodes:
        # sim_sampled_idx = sample_pos_based_on_sim(idx, pos_idxes, num_sim_samples)
        r_hop_nx_G, new_idx_to_old_idx, idx_to_dis = get_rhop_ego_nx_graph(idx_to_adj_list_dict,
                                                                           pos_idxes,
                                                                           idx,
                                                                           num_hops=num_hops)
        pos_sampled_idx, neg_sampled_idx = sample_n2v_neg_pos_based_on_possi(neg_p_q,
                                                                            idx,
                                                                            r_hop_nx_G,
                                                                            new_idx_to_old_idx,
                                                                            idx_to_dis,
                                                                            num_sim_samples,
                                                                            num_neg_samples,
                                                                            len(smiles_list),
                                                                            pos_idxes[idx])
        sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                    pos_idx in pos_sampled_idx])
        neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                    neg_idx in neg_sampled_idx])
        div_score = (sum(neg_samples_sim) / num_neg_samples) / (sum(sim_samples_sim) / num_sim_samples + 1e-9)
        div_scores.append(div_score)
    plt.hist(div_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.savefig("figs/n2v_one_walk_neg_pos_div_scores_hop_{:d}_num_pos_samples_{:d}_neg_{:d}_p_{}_q_{}.png".format(num_hops,
                                                                                             num_sim_samples,
                                                                                             num_neg_samples,
                                                                                             str(neg_p_q[0]),
                                                                                             str(neg_p_q[1])))
    plt.close()

def get_rhop_ego_graph(idx_to_adj_list_dict, pos_idxes,
                       idx_to_sim_sum, idx, num_hops=3, neg_sampling=False, pos_sampling=True): # assume "T" is equal to 4 and self.idx_to_sim_sum[idx] = sum(exp(sim_score / T))
    idx_to_dis = dict()
    que = [idx]
    idx_to_dis[idx] = 0
    edge_fr, edge_to = list(), list()
    edge_sample_prob = list()
    # pos_sample_prob = list()
    pos_edge_sample_prob = list()
    new_idx = 0

    old_to_new_idx = {idx: new_idx}
    T = 1
    while len(que) > 0:
        now_idx = que[-1]
        que.pop()
        new_now_idx = old_to_new_idx[now_idx]
        if idx_to_dis[now_idx] >= num_hops:
            break
        for other_idx in idx_to_adj_list_dict[now_idx]:
            if other_idx not in idx_to_dis:
                new_idx += 1
                old_to_new_idx[other_idx] = new_idx
                idx_to_dis[other_idx] = idx_to_dis[now_idx] + 1
                edge_fr.append(new_now_idx)
                edge_to.append(new_idx)
                edge_fr.append(new_idx)
                edge_to.append(new_now_idx)
                if neg_sampling:
                    ori_sim_score = pos_idxes[now_idx][other_idx]
                    div_sim_score = ori_sim_score / T
                    neg_sample_prob = math.exp(div_sim_score) / idx_to_sim_sum[now_idx]
                    edge_sample_prob.append(neg_sample_prob)
                    edge_sample_prob.append(neg_sample_prob)
                if pos_sampling:
                    # assert self.T == 1 # need sim score to be equal to 1
                    ori_sim_score = pos_idxes[now_idx][other_idx]
                    # div_sim_score = ori_sim_score / T
                    pos_sample_prob = math.exp(ori_sim_score) / idx_to_sim_sum[now_idx]
                    pos_edge_sample_prob.append(pos_sample_prob)
                    pos_edge_sample_prob.append(pos_sample_prob)
                que.insert(0, other_idx)
    edge_fr, edge_to = torch.tensor(edge_fr, dtype=torch.long), torch.tensor(edge_to, dtype=torch.long)

    dgl_g = dgl.DGLGraph()
    dgl_g.add_nodes(new_idx + 1)
    dgl_g.add_edges(edge_fr, edge_to)

    if neg_sampling:
        edge_sample_prob = torch.tensor(edge_sample_prob, dtype=torch.float64)
        dgl_g.edata["neg_sample_p"] = edge_sample_prob
    if pos_sampling:
        pos_edge_sample_prob = torch.tensor(pos_edge_sample_prob, dtype=torch.float64)
        dgl_g.edata["pos_sample_p"] = pos_edge_sample_prob
    new_idx_to_old_idx = {old_to_new_idx[i]: i for i in old_to_new_idx}
    return dgl_g, new_idx_to_old_idx, idx_to_dis


def rwr_sample_pos_graphs(restart_prob, dgl_bg, idx, num_samples=5):
    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [idx for __ in range(9)],
        restart_prob=restart_prob,
        length=num_samples)
    subv = torch.unique(traces).tolist()
    try:
        subv.remove(idx)
    except:
        pass
    try:
        subv.remove(-1)
    except:
        pass
    return subv

def get_neg_pos_sim_distri_rwr_hop(num_samples, k, num_hops, restart_prob=0.2):
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])

    lenn = 2000000
    assert len(smiles_list) == lenn

    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    assert len(pos_idxes) == lenn
    idx_to_adjs = np.load(os.path.join(processed_path, "node_idx_to_adj_idx_list_dict.npy")).item()
    idx_to_adj_list_dict = idx_to_adjs

    random_nodes = np.random.choice(a=len(smiles_list), size=1000, replace=False)
    # print(random_nodes)
    div_scores = []
    idx_to_sim_sum = np.load(os.path.join(processed_path, "idx_to_sim_sum_T_{:d}.npy".format(1))).item()

    rs_probs = [0.0, 0.2, 0.4, 0.6]
    # div_scores = {}
    for rsprob in rs_probs:
        div_scores.append([])

    for idx in random_nodes:
        if len(idx_to_adj_list_dict[idx]) > 0:
            dgl_bg, new_idx_to_old_idx, old_idx_to_dis = get_rhop_ego_graph(idx_to_adj_list_dict, pos_idxes,
                                                                            idx_to_sim_sum, idx,
                                                                            num_hops=num_hops,
                                                                            pos_sampling=True,
                                                                            neg_sampling=False)
            for ir, restart_prob in enumerate(rs_probs):
                sampled_idx = rwr_sample_pos_graphs(restart_prob, dgl_bg, 0, num_samples=(num_samples * (k + 1)) * 2)
                # convert smapled idx from new idx to old idx
                sampled_idx = [new_idx_to_old_idx[new_idx] for new_idx in sampled_idx]
                sorted_sampled_idx = sorted(sampled_idx, key=lambda i: old_idx_to_dis[i])
                pos_idx_offset = int(len(sorted_sampled_idx) / 8)
                pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                neg_sampled_idx = sorted_sampled_idx[pos_idx_offset:]
                num_pos_samples = num_samples
                num_neg_samples = k * num_samples
                # print("numpos", num_pos_samples, num_neg_samples) # only for debug
                if len(pos_sampled_idx) < num_pos_samples:
                    pos_sampled_idx = pos_sampled_idx + [idx for _ in range(num_pos_samples - len(pos_sampled_idx))]
                elif len(pos_sampled_idx) > num_pos_samples:
                    pos_sampled_idx = pos_sampled_idx[: num_pos_samples]

                if len(neg_sampled_idx) < num_neg_samples:
                    neg_sampled_idx = neg_sampled_idx + [(idx + _) % len(smiles_list) for _ in
                                                         range(1, num_neg_samples - len(neg_sampled_idx) + 1)]
                elif len(neg_sampled_idx) > num_neg_samples:
                    neg_sampled_idx = neg_sampled_idx[-num_neg_samples:]

                sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                            pos_idx in pos_sampled_idx])
                neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                            neg_idx in neg_sampled_idx])
                div_score = (sum(neg_samples_sim) / (num_samples * k)) / (sum(sim_samples_sim) / num_samples + 1e-9)
                div_scores[ir].append(div_score)
        else:
            # sampled_idx = [idx for _ in range(self.num_samples)]
            pos_sampled_idx = [idx for _ in range(num_samples)]
            neg_sampled_idx = [(idx + _) % len(smiles_list) for _ in range(1, num_samples * k + 1)]
        # pos_sampled_idx = sample_pos_based_on_sim(idx, pos_idxes, num_samples)
            sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                        pos_idx in pos_sampled_idx])
            neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                        neg_idx in neg_sampled_idx])
            div_score = (sum(neg_samples_sim) / (num_samples * k)) / (sum(sim_samples_sim) / num_samples + 1e-9)
            # div_scores.append(div_score)
            for ir, restart_prob in enumerate(rs_probs):
                div_scores[ir].append(div_score)

    # plt.hist(div_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.savefig("figs/rwr_v2_hop_neg_pos_div_scores_hop_{:d}_num_pos_samples_{:d}_neg_{:d}_rsprb_{}.png".format(num_hops,
    #                                                                                                   num_samples,
    #                                                                                                   num_samples * k,
    #                                                                                                   str(restart_prob)))
    # plt.close()
    for i, rsprob in enumerate(rs_probs):
        print(rsprob, sum(div_scores[i]) / len(div_scores[i]))


def rwr_sample_pos_graphs_path_dis(dgl_bg, idx, num_samples=5, num_path=7):
    traces, _ = dgl.sampling.random_walk(
        dgl_bg,
        [idx for __ in range(num_path)],
        # prob="pos_sample_p",
        # restart_prob=srestart_prob,
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
        for ic in range(len(subv) - 1, -1, -1):
            if subv[ic] != idx and subv[ic] != -1:
                candi_to_min_dis[subv[ic]] = ic
        for candi in candi_to_min_dis:
            if candi not in candi_to_count:
                candi_to_count[candi] = 1
                candi_to_dis_sum[candi] = candi_to_min_dis[candi]
            else:
                candi_to_count[candi] += 1
                candi_to_dis_sum[candi] += candi_to_min_dis[candi]
    candi_to_mean_dis = {candi: float(candi_to_dis_sum[candi]) / float(candi_to_count[candi]) \
                         for candi in candi_to_count}

    return candi_to_mean_dis, candi_to_count


def get_degree_array():
    processed_path = "./dataset/zinc_standard_agent/processed"
    with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "rb") as f:
        dgl_big_gra = pickle.load(f)

    degrees_exp = list()
    for i in range(dgl_big_gra.num_nodes()):
        if i % 10000 == 0:
            print(i)
        degrees_exp.append(dgl_big_gra.out_degree(i))
    np.save(os.path.join(processed_path, "big_graph_degree_array_list.npy"), degrees_exp)
    degrees_exp_nparray = np.array(degrees_exp, dtype=np.float)
    np.save(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"), degrees_exp_nparray)
    print("done")


def get_neg_pos_sim_distri_rwr_hop_big_gra():
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])

    lenn = 2000000
    assert len(smiles_list) == lenn

    pos_idxes = list()
    # for l in ls:  # not only one positive sample?
    processed_path = "./dataset/zinc_standard_agent/processed"
    # sim to score idx to sim {fri_idx: sim_score} dict
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))

    for j in range(len(sim_to_score_dict_list)):
        assert isinstance(sim_to_score_dict_list[j], dict)
        pos_idxes.append(sim_to_score_dict_list[j])
    assert len(pos_idxes) == lenn

    random_nodes = np.random.choice(a=len(smiles_list), size=1000, replace=False)
    # print(random_nodes)
    div_scores = []
    processed_path = "./dataset/zinc_standard_agent/processed"
    with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "rb") as f:
        dgl_big_gra = pickle.load(f)

    k = 24
    restart_prob = 0.0

    num_samples = 1

    all_rw_hops = [8, 16, 32, 64]

    hops_to_div_score_list = dict()
    for hop in all_rw_hops:
        hops_to_div_score_list[hop] = list()

    for hop in all_rw_hops:
        for idx in random_nodes:
            if dgl_big_gra.out_degree(idx) > 0:
                num_path = max(7, int(dgl_big_gra.out_degree(idx) * math.e / (math.e - 1) + 0.5))
                num_samples = 4
                num_path = 7
                # rw_hops = 16
                rw_hops = hop
                max_nodes_per_seed = max(rw_hops,
                                         int((dgl_big_gra.out_degree(idx) * math.e / (
                                                 math.e - 1) / (restart_prob + 0.2)) + 0.5)
                                         )
                num_samples = max_nodes_per_seed
                sampled_idx_to_mean_dis, sampled_idx_to_count = rwr_sample_pos_graphs_path_dis(dgl_big_gra,
                                                                                                idx,
                                                                                                num_samples,
                                                                                                num_path)

                sampled_nodes = list(sampled_idx_to_mean_dis.keys())
                sorted_sampled_idx = sorted(sampled_nodes, key=lambda idx: sampled_idx_to_mean_dis[idx])

                pos_idx_offset = int(len(sorted_sampled_idx) / (k + 1) + 1)

                num_pos_samples = num_samples
                num_neg_samples = k * num_samples

                # pos_idx_offset = min(pos_idx_offset, num_samples + 2)
                # neg_idx_offset = max(pos_idx_offset, len(sorted_sampled_idx) - num_neg_samples - 2)
                neg_idx_offset = pos_idx_offset
                # print("all sampled nodes idxes", len(sorted_sampled_idx), pos_idx_offset, neg_idx_offset)

                pos_sampled_idx = sorted_sampled_idx[: pos_idx_offset]
                # neg_sampled_idx = sorted_sampled_idx[pos_idx_offset: ]
                neg_sampled_idx = sorted_sampled_idx[neg_idx_offset:]

                idx_to_sim_score_dict = pos_idxes[idx]
                keys = list()
                values = list()
                for idxbb in idx_to_sim_score_dict:
                    keys.append(idxbb)
                    values.append(idx_to_sim_score_dict[idxbb])
                if len(keys) > 0:
                    values = torch.tensor(values, dtype=torch.float64)
                    values = torch.softmax(values, dim=0)
                    self_num_samples = num_samples if num_samples <= values.size(0) else values.size(0)
                    choose_idx = torch.multinomial(values, self_num_samples)

                    if self_num_samples < num_samples:
                        keys.append(idx)
                        choose_idx = torch.cat([choose_idx, torch.full((num_samples - self_num_samples,), len(keys) - 1,
                                                                       dtype=torch.long)], dim=0)
                else:
                    choose_idx = [0 for j in range(num_samples)]
                    keys = [idx for j in range(num_samples)]

                # sim_datas = list()
                # data.sim_x, data.sim_edge_index, data.sim_edge_attr = list(), list(), list()
                pos_sampled_idx = []
                for i in range(num_samples):
                    sample_idx = int(choose_idx[i])
                    real_idx = int(keys[sample_idx])
                    pos_sampled_idx.append(real_idx)

                # if len(pos_sampled_idx) == 0:
                #     pos_sampled_idx = [idx for _ in range(num_pos_samples)]
                # else:
                #     pos_sampled_prob = [1.0 / len(pos_sampled_idx) for __ in range(len(pos_sampled_idx))]
                #     pos_sampled_prob_tensor = torch.tensor(pos_sampled_prob, dtype=torch.float64) / sum(
                #         pos_sampled_prob)
                #     pos_sampled_idxs = torch.multinomial(pos_sampled_prob_tensor,
                #                                          replacement=True, num_samples=num_samples)
                #     pos_sampled_idx = [pos_sampled_idx[int(pos_sampled_idxs[ipos])] for ipos in
                #                        range(pos_sampled_idxs.size(0))]

                if len(neg_sampled_idx) == 0:
                    neg_sampled_idx = [(idx + _) % len(smiles_list) for _ in range(1, 1 + num_neg_samples)]
                else:
                    len_neg = len(neg_sampled_idx)

                    neg_sampled_prob = [1.0 / len_neg for __ in range(len_neg)]
                    neg_sampled_prob_tensor = torch.tensor(neg_sampled_prob, dtype=torch.float64) / sum(neg_sampled_prob)
                    neg_sampled_idxs = torch.multinomial(neg_sampled_prob_tensor,
                                                         replacement=True, num_samples=num_neg_samples)
                    neg_sampled_idx = [neg_sampled_idx[int(neg_sampled_idxs[ineg])] for ineg in
                                       range(neg_sampled_idxs.size(0))]

                sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                            pos_idx in pos_sampled_idx])
                neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                            neg_idx in neg_sampled_idx])
                div_score = (sum(neg_samples_sim) / (num_samples * k)) / (sum(sim_samples_sim) / num_samples + 1e-9)
                div_scores.append(div_score)
                hops_to_div_score_list[hop].append(div_score)

            else:
                # sampled_idx = [idx for _ in range(self.num_samples)]
                pos_sampled_idx = [idx for _ in range(num_samples)]
                neg_sampled_idx = [(idx + _) % len(smiles_list) for _ in range(1, num_samples * k + 1)]
                # pos_sampled_idx = sample_pos_based_on_sim(idx, pos_idxes, num_samples)
                sim_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[pos_idx] for \
                                                                            pos_idx in pos_sampled_idx])
                neg_samples_sim = compute_similarity_1vn(smiles_list[idx], [smiles_list[neg_idx] for \
                                                                            neg_idx in neg_sampled_idx])
                div_score = (sum(neg_samples_sim) / (num_samples * k)) / (sum(sim_samples_sim) / num_samples + 1e-9)
                # div_scores.append(div_score)
                # for ir, restart_prob in enumerate(rs_probs):
                div_scores.append(div_score)
                hops_to_div_score_list[hop].append(div_score)

    # plt.hist(div_scores, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.savefig("figs/rwr_v2_hop_neg_pos_div_scores_hop_{:d}_num_pos_samples_{:d}_neg_{:d}_rsprb_{}.png".format(num_hops,
    #                                                                                                   num_samples,
    #                                                                                                   num_samples * k,
    #                                                                                                   str(restart_prob)))
    # plt.close()
    # walk_length = 16: 0.8319077553012331 walk_length = 8: 0.833795093460429 walk_length = 4: 0.8302836421341463
    for hop in all_rw_hops:
        div_scores = hops_to_div_score_list[hop]
        print(hop, sum(div_scores) / len(div_scores))
    # print(sum(div_scores) / len(div_scores)) # walk_length = 16: 0.8319077553012331 walk_length = 8: 0.833795093460429

def plot_loss_curve():
    file_names = ["contrast_sim_based_multi_pos_v2_sample_stra_n2v_neg_pos_one_walk_1_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.1_p_0.4_q_1.6",
                  "contrast_sim_based_multi_pos_v2_sample_stra_n2v_neg_pos_one_walk_1_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.1_p_0.6_q_1.4",
                  "contrast_sim_based_multi_pos_v2_sample_stra_n2v_neg_pos_one_walk_1_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.1_p_1.0_q_1.0"]
    val = list()

    for i, fn in enumerate(file_names):
        val_per = list()
        cnt, sum = 0, 0.0
        with open(os.path.join("log_loss", fn), "r") as wf:
            for line in wf:
                los = float(line.strip())
                if cnt < 10:
                    sum += los
                    cnt += 1
                else:
                    val_per.append(sum / cnt)
                    cnt = 1
                    sum = los
                # val_per.append(los)
            if cnt > 0:
                val_per.append(sum / cnt)
                # cnt = 0
        val.append(val_per)
    col = ["r", "g", "b"]
    for i, val_per in enumerate(val):
        per_len = len(val_per)
        plt.plot(np.arange(per_len), val_per, col[i], label="ver {:d}".format(i))
    plt.savefig("figs/loss_curve.png")
    plt.close()

def draw_acc_sim_curve():
    # p = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.4]
    p = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4]
    for i, val in enumerate(p):
        p[i] = str(val)
    datasets = ["sider", "clintox", "bace", "hiv", "bbbp", "sim_div_scores"]
    # dataset_accs = {
    #     "sider": [0.6096, 0.6371, 0.6383, 0.6343, 0.6280, 0.6242, 0.6128],
    #     "clintox": [0.7616, 0.7846, 0.8165, 0.8229, 0.7983, 0.7853, 0.7898],
    #     "bace": [0.8191, 0.8109, 0.8133, 0.8249, 0.8066, 0.8455, 0.8152],
    #     "hiv": [0.7800, 0.7855, 0.7866, 0.7847, 0.7734, 0.7698, 0.7761],
    #     "bbbp": [0.7231, 0.7303, 0.7476, 0.7244, 0.7257, 0.7281, 0.7380]
    # }
    dataset_accs = {
        "sider": [0.6096, 0.6371, 0.6383, 0.6280, 0.6242, 0.6128],
        "clintox": [0.7616, 0.7846, 0.8165, 0.7983, 0.7853, 0.7898],
        "bace": [0.8191, 0.8109, 0.8133, 0.8066, 0.8455, 0.8152],
        "hiv": [0.7800, 0.7855, 0.7866, 0.7734, 0.7698, 0.7761],
        "bbbp": [0.7231, 0.7303, 0.7476, 0.7257, 0.7281, 0.7380],
        "sim_div_scores": [0.9789366032120397, 0.9789366032120397, 0.9985499507466545, 1.011962865645926,
                           1.0131846079432256, 1.0147047479004763]
    }

    dataset_stds = {
        "sider": [0.0011, 0.0015, 0.0019, 0.0026, 0.0020, 0.0016],
        "clintox": [0.0057, 0.0082, 0.0121, 0.0053, 0.0026, 0.0040],
        "bace": [0.0118, 0.0015, 0.0074, 0.0048, 0.0029, 0.0025],
        "hiv": [0.0012, 0.0071, 0.0027, 0.0032, 0.0077, 0.0039],
        "bbbp": [0.0010, 0.0020, 0.0026, 0.0010, 0.0049, 0.0040]
    }

    # plt.get_cachedir()
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(18, 3))
    colors = ["r", "g", "b", "brown", "y", "c"]
    plt.subplot(161)
    plt.plot(p, dataset_accs[datasets[0]], color=colors[0], linestyle="-", label=datasets[0])
    plt.plot(p, dataset_accs[datasets[0]], "r^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[0]][j] - dataset_stds[datasets[0]][j],
                   dataset_accs[datasets[0]][j] + dataset_stds[datasets[0]][j], color="r")
    plt.legend()
    plt.xlabel("p")
    plt.ylabel("auc")

    plt.subplot(162)
    plt.plot(p, dataset_accs[datasets[1]], color=colors[1], linestyle="-", label=datasets[1])
    plt.plot(p, dataset_accs[datasets[1]], "g^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[1]][j] - dataset_stds[datasets[1]][j],
                   dataset_accs[datasets[1]][j] + dataset_stds[datasets[1]][j], color="g")
    plt.legend()
    plt.xlabel("p")

    plt.subplot(163)
    plt.plot(p, dataset_accs[datasets[2]], color=colors[2], linestyle="-", label=datasets[2])
    plt.plot(p, dataset_accs[datasets[2]], "b^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[2]][j] - dataset_stds[datasets[2]][j],
                   dataset_accs[datasets[2]][j] + dataset_stds[datasets[2]][j], color="b")
    plt.legend()
    plt.xlabel("p")

    plt.subplot(164)
    plt.plot(p, dataset_accs[datasets[3]], color=colors[3], linestyle="-", label=datasets[3])
    plt.plot(p, dataset_accs[datasets[3]], "r^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[3]][j] - dataset_stds[datasets[3]][j],
                   dataset_accs[datasets[3]][j] + dataset_stds[datasets[3]][j], color="brown")
    plt.legend()
    plt.xlabel("p")

    plt.subplot(165)
    plt.plot(p, dataset_accs[datasets[4]], color=colors[4], linestyle="-", label=datasets[4])
    plt.plot(p, dataset_accs[datasets[4]], "y^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[4]][j] - dataset_stds[datasets[4]][j],
                   dataset_accs[datasets[4]][j] + dataset_stds[datasets[4]][j], color="y")
    # for i, ds in enumerate(datasets):
    #     vals = dataset_accs[ds]
    #     plt.plot(p, vals, colors[i], label=ds)
    plt.xlabel("p")
    # plt.ylabel("auc")
    plt.legend()

    plt.subplot(166)
    plt.plot(p, dataset_accs[datasets[5]], color=colors[5], linestyle="-", label=datasets[5])
    plt.plot(p, dataset_accs[datasets[5]], "c^")
    # for i, ds in enumerate(datasets):
    #     vals = dataset_accs[ds]
    #     plt.plot(p, vals, colors[i], label=ds)
    plt.xlabel("p")
    # plt.ylabel("auc")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/auc_curve_with_div_scores_with_std.pdf")
    plt.close()


def draw_acc_sim_curve_rs_prob():
    # p = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.4]
    p = [0.0, 0.2, 0.4, 0.6]
    # sims = [0.7382668661081925, 0.7411878098789468, 0.5739256767855078, 0.3768664959538144]
    for i, val in enumerate(p):
        p[i] = str(val)
    datasets = ["sider", "clintox", "bace", "hiv", "bbbp", "sim_div_scores"]
    dataset_accs = {
        "sider": [0.6221, 0.6236, 0.6270, 0.5281],
        "clintox": [0.8245, 0.8479, 0.7659, 0.6227],
        "bace": [0.8182, 0.8080, 0.8026, 0.8233],
        "hiv": [0.7800, 0.7991, 0.7880, 0.7791],
        "bbbp": [0.7260, 0.7386, 0.7299, 0.6976],
        "sim_div_scores": [0.7382668661081925, 0.7411878098789468, 0.5739256767855078, 0.3768664959538144],
    }
    dataset_stds = {
        "sider": [0.0075, 0.0082, 0.0016, 0.0033],
        "clintox": [0.0275, 0.0434, 0.0069, 0.0207],
        "bace": [0.0136, 0.0025, 0.0165, 0.0033],
        "hiv": [0.0070, 0.0061, 0.0062, 0.0042],
        "bbbp": [0.0090, 0.0071, 0.0088, 0.0203]
    }



    # plt.get_cachedir()
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(18, 3))
    colors = ["r", "g", "b", "brown", "y", "c"]

    plt.subplot(161)
    plt.plot(p, dataset_accs[datasets[0]], color=colors[0], linestyle="-", label=datasets[0])
    plt.plot(p, dataset_accs[datasets[0]], "r^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[0]][j] - dataset_stds[datasets[0]][j],
                   dataset_accs[datasets[0]][j] + dataset_stds[datasets[0]][j], color="r")

    plt.legend()
    plt.xlabel("restart prob")
    plt.ylabel("auc")

    plt.subplot(162)
    plt.plot(p, dataset_accs[datasets[1]], color=colors[1], linestyle="-", label=datasets[1])
    plt.plot(p, dataset_accs[datasets[1]], "g^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[1]][j] - dataset_stds[datasets[1]][j],
                   dataset_accs[datasets[1]][j] + dataset_stds[datasets[1]][j], color="g")
    plt.legend()
    plt.xlabel("restart prob")

    plt.subplot(163)
    plt.plot(p, dataset_accs[datasets[2]], color=colors[2], linestyle="-", label=datasets[2])
    plt.plot(p, dataset_accs[datasets[2]], "b^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[2]][j] - dataset_stds[datasets[2]][j],
                   dataset_accs[datasets[2]][j] + dataset_stds[datasets[2]][j], color="b")
    plt.legend()
    plt.xlabel("restart prob")

    plt.subplot(164)
    plt.plot(p, dataset_accs[datasets[3]], color=colors[3], linestyle="-", label=datasets[3])
    plt.plot(p, dataset_accs[datasets[3]], "r^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[3]][j] - dataset_stds[datasets[3]][j],
                   dataset_accs[datasets[3]][j] + dataset_stds[datasets[3]][j], color="brown")
    plt.legend()
    plt.xlabel("restart prob")

    plt.subplot(165)
    plt.plot(p, dataset_accs[datasets[4]], color=colors[4], linestyle="-", label=datasets[4])
    plt.plot(p, dataset_accs[datasets[4]], "y^")
    for j in range(len(p)):
        plt.vlines(p[j], dataset_accs[datasets[4]][j] - dataset_stds[datasets[4]][j],
                   dataset_accs[datasets[4]][j] + dataset_stds[datasets[4]][j], color="y")
    # for i, ds in enumerate(datasets):
    #     vals = dataset_accs[ds]
    #     plt.plot(p, vals, colors[i], label=ds)
    plt.legend()
    plt.xlabel("restart prob")

    plt.subplot(166)
    plt.plot(p, dataset_accs[datasets[5]], color=colors[5], linestyle="-", label=datasets[5])
    plt.plot(p, dataset_accs[datasets[5]], "c^")
    # for i, ds in enumerate(datasets):
    #     vals = dataset_accs[ds]
    #     plt.plot(p, vals, colors[i], label=ds)
    plt.legend()
    plt.xlabel("restart prob")


    plt.tight_layout()
    plt.savefig("figs/auc_curve_rs_prob_with_div_scores_with_std.pdf")
    plt.close()

def draw_acc_sim_curve_rs_prob_n2v_q(dataset="bbbp"):
    # p = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.4]
    p = [0.0, 0.2, 0.4, 0.6]
    q = [0.2, 0.4, 0.6, 0.8, 1.0]
    # sims = [0.7382668661081925, 0.7411878098789468, 0.5739256767855078, 0.3768664959538144]
    for i, val in enumerate(p):
        p[i] = str(val)

    for i, val in enumerate(q):
        q[i] = str(val)
    datasets = ["sider", "clintox", "bace", "hiv", "bbbp", "sim_div_scores"]
    dataset_accs = {
        "sider": [0.6221, 0.6236, 0.6270, 0.5281],
        "clintox": [0.8245, 0.8479, 0.7659, 0.6227],
        "bace": [0.8182, 0.8080, 0.8026, 0.8233],
        "hiv": [0.7800, 0.7991, 0.7880, 0.7791],
        "bbbp": [0.7060, 0.7186, 0.7099, 0.6776],
        "sim_div_scores": [0.7382668661081925, 0.7411878098789468, 0.5739256767855078, 0.3768664959538144],
    }
    dataset_stds = {
        "sider": [0.0075, 0.0082, 0.0016, 0.0033],
        "clintox": [0.0275, 0.0434, 0.0069, 0.0207],
        "bace": [0.0136, 0.0025, 0.0165, 0.0033],
        "hiv": [0.0070, 0.0061, 0.0062, 0.0042],
        "bbbp": [0.0090, 0.0071, 0.0088, 0.0203]
    }

    dataset_accs_q = {
        "sider": [0.6096, 0.6371, 0.6383, 0.6280, 0.6242],
        "clintox": [0.7616, 0.7846, 0.8165, 0.7983, 0.7853],
        "bace": [0.8191, 0.8109, 0.8133, 0.8066, 0.8355],
        "hiv": [0.7800, 0.7855, 0.7866, 0.7734, 0.7698],
        "bbbp": [0.7031, 0.7103, 0.7276, 0.7057, 0.7081],
        "sim_div_scores": [0.9789366032120397, 0.9789366032120397, 0.9985499507466545, 1.011962865645926,
                           1.0131846079432256]
    }

    dataset_stds_q = {
        "sider": [0.0011, 0.0015, 0.0019, 0.0026, 0.0020],
        "clintox": [0.0057, 0.0082, 0.0121, 0.0053, 0.0026],
        "bace": [0.0118, 0.0015, 0.0074, 0.0048, 0.0029],
        "hiv": [0.0012, 0.0071, 0.0027, 0.0032, 0.0077],
        "bbbp": [0.0010, 0.0020, 0.0026, 0.0010, 0.0049]
    }

    # plt.get_cachedir()
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(6, 3))
    colors = ["r", "g", "b", "brown", "y", "c"]
    plt.subplot(121)
    rsprob_vals = dataset_accs[dataset]
    rsprob_stds = dataset_stds[dataset]
    n2v_vals = dataset_accs_q[dataset]
    n2v_stds = dataset_stds_q[dataset]

    plt.plot(p, rsprob_vals, color="b", linestyle="-", label=dataset)
    plt.plot(p, rsprob_vals, "b^")
    for j in range(len(p)):
        plt.vlines(p[j], rsprob_vals[j] - rsprob_stds[j],
                   rsprob_vals[j] + rsprob_stds[j], color="b")

    plt.legend()
    plt.xlabel("restart probability")
    plt.ylabel("ROC-AUC")

    plt.subplot(122)
    plt.plot(q, n2v_vals, color="b", linestyle="-", label=dataset)
    plt.plot(q, n2v_vals, "b^")
    for j in range(len(q)):
        plt.vlines(q[j], n2v_vals[j] - n2v_stds[j],
                   n2v_vals[j] + n2v_stds[j], color="b")
    plt.legend()

    plt.xlabel("q")

    plt.tight_layout()
    plt.savefig("figs/auc_curve_rs_prob_n2v_q_with_std_{}.pdf".format(dataset))
    plt.close()

import pickle

def to_nx_graph(edge_list, node_idx_list, masked_nodes_list):
    idx_offset = min(node_idx_list)
    nxg = nx.Graph()
    node_idx_list = [item - idx_offset for item in node_idx_list]
    edge_list = [(item[0] - idx_offset, item[1] - idx_offset) for item in edge_list]
    masked_nodes_list = [item - idx_offset for item in masked_nodes_list]
    nxg.add_nodes_from(node_idx_list)
    nxg.add_edges_from(edge_list)
    tot_dis = []
    tot_dis = {node_idx: -1 for node_idx in node_idx_list}
    minn_disss = []
    for i_st in range(len(masked_nodes_list) - 1):
        minn_dis = tot_dis[masked_nodes_list[i_st]]
        for i_ed in range(i_st + 1, len(masked_nodes_list)):
            st, ed = masked_nodes_list[i_st], masked_nodes_list[i_ed]
            dis = nx.shortest_path_length(nxg, source=st, target=ed)
            if tot_dis[ed] == -1 or tot_dis[ed] > dis:
                tot_dis[ed] = dis
            if minn_dis == -1:
                minn_dis = dis
            elif minn_dis > dis:
                minn_dis = dis
        if (minn_dis >= 0):
            minn_disss.append(minn_dis)

    return minn_disss

def get_distance_distri(rf, epoch=1, batch_idx=1):
    mask_info = pickle.load(rf)
    edge_index, masked_nodes, batch, x = mask_info["edge_index"], mask_info["masked_nodes"], mask_info["batch"], \
                                            mask_info["x"]
    gra_idx_to_nodes_idx = dict()

    for i_node in range(x.size(0)):
        gra_idx = int(batch[i_node])
        if gra_idx not in gra_idx_to_nodes_idx:
            gra_idx_to_nodes_idx[gra_idx] = [i_node]
        else:
            gra_idx_to_nodes_idx[gra_idx].append(i_node)
    gra_idx_to_edge_list = dict()
    for i_edge in range(0, edge_index.size(1), 2):
        a, b = int(edge_index[0, i_edge]), int(edge_index[1, i_edge])
        gra_a, gra_b = int(batch[a]), int(batch[b])
        assert gra_a == gra_b
        if gra_a not in gra_idx_to_edge_list:
            gra_idx_to_edge_list[gra_a] = [(a, b), (b, a)]
        else:
            gra_idx_to_edge_list[gra_a].append((a, b))
            gra_idx_to_edge_list[gra_a].append((b, a))
    gra_idx_to_masked_nodes = dict()
    for i in range(masked_nodes.size(0)):
        node_idx = int(masked_nodes[i])
        gra_idx = int(batch[node_idx])
        if gra_idx not in gra_idx_to_masked_nodes:
            gra_idx_to_masked_nodes[gra_idx] = [node_idx]
        else:
            gra_idx_to_masked_nodes[gra_idx].append(node_idx)
    all_gra_dis = []
    all_gra_random_dis = []
    for gra_idx in gra_idx_to_nodes_idx:
        masked_nodes_list_now = [] if gra_idx not in gra_idx_to_masked_nodes else gra_idx_to_masked_nodes[gra_idx]
        edge_list_now = gra_idx_to_edge_list[gra_idx]
        node_list_now = gra_idx_to_nodes_idx[gra_idx]
        masked_nodes_list_random = np.random.choice(node_list_now, len(masked_nodes_list_now), replace=False)
        dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_now)
        random_dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_random)
        all_gra_dis += dis
        all_gra_random_dis += random_dis
    wf = open("masked_info_exp/all_gra_dis_epoch_{:d}_batch_{:d}".format(epoch, batch_idx), "wb")
    pickle.dump({"masker_dis": all_gra_dis, "random_dis": all_gra_random_dis}, wf)
    print(sum(all_gra_dis) / len(all_gra_dis))
    print("random", sum(all_gra_random_dis) / len(all_gra_random_dis))
    return all_gra_dis, all_gra_random_dis

def compute_degree_tensor():
    processed_path = "./dataset/zinc_standard_agent/processed"
    with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "rb") as f:
        dgl_big_gra = pickle.load(f)
    print("dgl big graph loaded!")
    degrees = []
    out_degs = dgl_big_gra.out_degrees(torch.arange(dgl_big_gra.num_nodes()))
    degrees = out_degs.tolist()
    # for i in range(dgl_big_gra.num_nodes()):
    #     if i % 10000 == 0:
    #         print(i)
    #     degrees.append(dgl_big_gra.out_degree(i))
    # degrees = torch.tensor(degrees, dtype=torch.float32)
    degrees = np.array(degrees, dtype=np.float)
    np.save(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy"), degrees)
    print("no_exp_degrees saved!")

def get_neg_sampling_possi():
    processed_path = "./dataset/zinc_standard_agent/processed"
    degrees = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array.npy"))
    # degrees = degrees / np.sum(degrees, axis=0, keepdims=True)
    # degrees = np.exp(degrees)
    # degrees = degrees / np.sum(degrees, axis=0, keepdims=True)
    degrees_no_exp = np.load(os.path.join(processed_path, "big_graph_degree_array_np_array_no_exp.npy"))
    degrees_no_exp = degrees_no_exp / np.sum(degrees_no_exp, axis=0, keepdims=True)
    print(np.max(degrees), np.max(degrees_no_exp))

    plt.hist(degrees, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # print(np.max(degrees))
    plt.tight_layout()
    plt.savefig("figs/normalized_degrees_no_norm")

def get_info_dgl_big_gra():
    processed_path = "./dataset/zinc_standard_agent/processed"
    with open(os.path.join(processed_path, "big_graph_dgl.pkl"), "rb") as f:
        nxg = pickle.load(f)
    print(nxg.edata.keys())
    # print(nxg.edges())
    print(nxg.num_edges())
    sim_to_score_dict_list = np.load(os.path.join(processed_path, "sim_to_score_dict_list_wei_nc_natom.npy"))
    sim_to_score_sum = dict()
    for i in range(len(sim_to_score_dict_list)):
        if i % 100000 == 0:
            print("for node", i)
        all_sim_scores = sim_to_score_dict_list[i]
        sim_scores_list = [all_sim_scores[other_idx] for other_idx in all_sim_scores]
        sim_to_score_sum[i] = sum(sim_scores_list)
    edge_weights = []
    fr, to = nxg.edges()
    print(fr.size(), to.size())
    for i in range(fr.size(0)):
        if i % 1000000 == 0:
            print(i)
        fridx, toidx = int(fr[i]), int(to[i])
        if toidx in sim_to_score_dict_list[fridx]:
            edge_weights.append(sim_to_score_dict_list[fridx][toidx] / sim_to_score_sum[fridx])
        elif fridx in sim_to_score_dict_list[toidx]:
            edge_weights.append(sim_to_score_dict_list[toidx][fridx] / sim_to_score_sum[toidx])
        else:
            print("what", fridx, toidx)
    edata = torch.tensor(edge_weights, dtype=torch.float32)
    nxg.edata["pos_sample_p"] = edata
    with open(os.path.join(processed_path, "big_graph_dgl_with_edge_p_normed.pkl"), "wb") as f:
        pickle.dump(nxg, f)

def get_neg_samples_idx():
    fn = "contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_1_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.0_p_0.7_q_0.6_rw_hops_3_dl_other_p_neg_sampled_idx_epoch_1_step_400"
    neg_sampled_idxs = dict()
    with open("log_loss/" + fn, "r") as rf:
        for line in rf:
            aa = line.strip().split(": ")
            idx, num = int(aa[0]), int(aa[1])
            neg_sampled_idxs[idx] = num
    values = list(neg_sampled_idxs.values())
    print(len(neg_sampled_idxs), max(values), min(values))
    plt.hist(values, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # print(np.max(degrees))
    plt.tight_layout()
    plt.savefig("figs/sampled_neg_num_distri")

import re
def get_best_aucs_from_files():
    root_file_path = "/apdcephfs/private_meowliu"
    sll_log_file_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/logs"
    dirs = os.listdir(root_file_path)
    pat = re.compile(r'(?:auc = )\d+\.?\d*')
    for i, subdir in enumerate(dirs):
        print(i, subdir)
        print(type(subdir))
        if str(subdir).startswith("DrugAI"):
            print("inin")
            subdirpath = os.path.join(root_file_path, subdir)
            dir_in_subdir = os.listdir(subdirpath)
            for j, subsubdir in enumerate(dir_in_subdir):
                log_file_path = os.path.join(os.path.join(subdirpath, subsubdir), "out.log")
                rf = open(log_file_path)
                patstr = rf.read()
                res = pat.findall(patstr)
                ans = 0.0
                print(res)
                for ress in res:
                    tmpans = ress.split(" ")[-1]
                    ans = max(ans, float(tmpans))
                with open(os.path.join(sll_log_file_path, "out.log"), "a") as wf:
                    wf.write("{} {} auc = {:.4f}\n".format(subdir, subsubdir, ans))
                    wf.close()
    print("all done")

from rdkit import Chem
from utils2 import mol_to_graph_data_obj_simple, sampled_subgraph_gcc, get_subgraph_data, graph_data_obj_to_mol_simple
from torch_geometric.data import Data
def check_mol_val(mol):
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    return True if m else False

def compute_sim_mol_pair(molecules):
    assert len(molecules) == 2
    fingerprints = [Chem.RDKFingerprint(mol) for mol in molecules]
    # for other_finger in fingerprints:
    res = DataStructs.FingerprintSimilarity(fingerprints[0], fingerprints[1])
    return res

def get_subgraph_drop_edge(data, droped_edge_idx):
    edge_index = list()
    edge_attr = list()
    if isinstance(droped_edge_idx, list):
        droped_edge_idx = {idx: i for i, idx in enumerate(droped_edge_idx)}
    for i in range(data.edge_index.size(1)):
        if i not in droped_edge_idx:
            edge_index.append(data.edge_index[:, i].view(-1, 1))
            edge_attr.append(data.edge_attr[i, :].view(1, -1))
    edge_index = torch.cat(edge_index, dim=1)
    edge_attr = torch.cat(edge_attr, dim=0)
    return edge_index, edge_attr

def get_dropedge_samples_validity():
    input_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    # processed_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    cutof = 1000
    smiles_list = list(input_df['smiles'])
    selected_idx = np.random.choice(range(len(smiles_list)), cutof, replace=False)
    drop_prob_to_mean_val = dict()
    drop_prob_to_sim_avg = dict()
    for drop_prob in [0.1, 0.15, 0.2, 0.25]:
        tot_cnt = 0
        cutof = 1000
        sim_res = []
        for i in range(cutof):
            smi = smiles_list[int(selected_idx[i])]
            mol = Chem.MolFromSmiles(smi)
            mol_data = mol_to_graph_data_obj_simple(mol)
            num_edges = mol_data.edge_attr.size(0) // 2
            drop_edges_num = int(drop_prob * num_edges + 1)
            droped_edges_idx = np.random.choice(range(num_edges), drop_edges_num, replace=False)
            droped_edges_idx_dict = dict()
            for jj in range(drop_edges_num):
                single_edge_idx = int(droped_edges_idx[jj])
                droped_edges_idx_dict[single_edge_idx * 2] = 1
                droped_edges_idx_dict[single_edge_idx * 2 + 1] = 1

            try:
                droped_edge_index, droped_edge_attr = get_subgraph_drop_edge(mol_data, droped_edges_idx_dict)
                # x, ei, ea = get_subgraph_data(mol_data, exist_nodes)
                sub_mol = graph_data_obj_to_mol_simple(mol_data.x, droped_edge_index, droped_edge_attr)
                if check_mol_val(sub_mol):
                    tot_cnt += 1

                droped_edges_idx = np.random.choice(range(num_edges), drop_edges_num, replace=False)
                droped_edges_idx_dict = dict()
                for jj in range(drop_edges_num):
                    single_edge_idx = int(droped_edges_idx[jj])
                    droped_edges_idx_dict[single_edge_idx * 2] = 1
                    droped_edges_idx_dict[single_edge_idx * 2 + 1] = 1
                droped_edge_index, droped_edge_attr = get_subgraph_drop_edge(mol_data, droped_edges_idx_dict)
                # x, ei, ea = get_subgraph_data(mol_data, exist_nodes)
                sub_mol_b = graph_data_obj_to_mol_simple(mol_data.x, droped_edge_index, droped_edge_attr)
                if check_mol_val(sub_mol_b):
                    tot_cnt += 1
                sim_res.append(compute_sim_mol_pair([sub_mol, sub_mol_b]))
            except:
                sim_res.append(0)
            if i == cutof - 1:
                break
        print(float(tot_cnt) / float(cutof * 2), sum(sim_res) / (len(sim_res) + 1e-9), "rsprob=%.1f" % drop_prob)
        drop_prob_to_mean_val[drop_prob] = float(tot_cnt) / float(cutof * 2)
        drop_prob_to_sim_avg[drop_prob] = sum(sim_res) / (len(sim_res) + 1e-9)
    return drop_prob_to_mean_val, drop_prob_to_sim_avg


def get_dropnode_samples_validity():
    input_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    # processed_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    cutof = 1000
    smiles_list = list(input_df['smiles'])
    selected_idx = np.random.choice(range(len(smiles_list)), cutof, replace=False)
    drop_prob_to_mean_val = dict()
    drop_prob_to_sim_avg = dict()
    for drop_prob in [0.1, 0.15, 0.2, 0.25]:
        tot_cnt = 0
        cutof = 1000
        sim_res = []
        for i in range(cutof):
            smi = smiles_list[int(selected_idx[i])]
            mol = Chem.MolFromSmiles(smi)
            mol_data = mol_to_graph_data_obj_simple(mol)
            num_nodes = mol_data.x.size(0)
            drop_nodes_num = int(drop_prob * num_nodes + 1)
            droped_nodes = np.random.choice(range(num_nodes), drop_nodes_num, replace=False)
            droped_nodes_dict = {int(droped_nodes[jj]): 1for jj in range(drop_nodes_num)}
            exist_nodes = [jj for jj in range(num_nodes) if jj not in droped_nodes_dict]
            try:
                x, ei, ea = get_subgraph_data(mol_data, exist_nodes)
                sub_mol = graph_data_obj_to_mol_simple(x, ei, ea)
                if check_mol_val(sub_mol):
                    tot_cnt += 1

                droped_nodes = np.random.choice(range(num_nodes), drop_nodes_num, replace=False)
                droped_nodes_dict = {int(droped_nodes[jj]): 1 for jj in range(drop_nodes_num)}
                exist_nodes = [jj for jj in range(num_nodes) if jj not in droped_nodes_dict]
                x_b, ei_b, ea_b = get_subgraph_data(mol_data, exist_nodes)
                sub_mol_b = graph_data_obj_to_mol_simple(x_b, ei_b, ea_b)
                if check_mol_val(sub_mol_b):
                    tot_cnt += 1
                sim_res.append(compute_sim_mol_pair([sub_mol, sub_mol_b]))
            except:
                sim_res.append(0)

            if i == cutof - 1:
                break
        print(float(tot_cnt) / float(cutof * 2), sum(sim_res) / (len(sim_res) + 1e-9), "rsprob=%.1f" % drop_prob)
        drop_prob_to_mean_val[drop_prob] = float(tot_cnt) / float(cutof * 2)
        drop_prob_to_sim_avg[drop_prob] = sum(sim_res) / (len(sim_res) + 1e-9)
    return drop_prob_to_mean_val, drop_prob_to_sim_avg


def get_gcc_sample_validity():

    input_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    # processed_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    step_dist = [1.0, 0.0, 0.0]
    length = 64
    # rsprob = 0.0
    smiles_list = list(input_df['smiles'])
    tot_cnt = 0
    sim_res = list()
    cutof = 1000
    selected_idx = np.random.choice(range(len(smiles_list)), cutof, replace=False)
    rsprob_to_mean_val = dict()
    rsprob_to_sim_avg = dict()
    for rsprob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tot_cnt = 0
        cutof = 1000
        sim_res = []
        for i in range(cutof):
            smi = smiles_list[int(selected_idx[i])]
        # for i, smi in enumerate(smiles_list):
            # if i % 100 == 0:
            #     print(i, tot_cnt)
            mol = Chem.MolFromSmiles(smi)
            mol_data = mol_to_graph_data_obj_simple(mol)
            node_a, node_b = \
                sampled_subgraph_gcc(mol_data, step_dist=step_dist, length=length, rsprob=rsprob)

            x_a, ei_a, ea_a = get_subgraph_data(mol_data, node_a)
            x_b, ei_b, ea_b = get_subgraph_data(mol_data, node_b)
            # data_a = Data(x=x_a, edge_index=ei_a, edge_attr=ea_a)
            # data_b = Data(x=x_b, edge_index=ei_b, edge_attr=ea_b)
            mol_a = graph_data_obj_to_mol_simple(x_a, ei_a, ea_a)
            mol_b = graph_data_obj_to_mol_simple(x_b, ei_b, ea_b)
            if check_mol_val(mol_a) and x_a.size(0) < mol_data.x.size(0):
                tot_cnt += 1
            if check_mol_val(mol_b) and x_b.size(0) < mol_data.x.size(0):
                tot_cnt += 1
            sim_res.append(compute_sim_mol_pair([mol_a, mol_b]))
            if i == cutof - 1:
                break
        print(float(tot_cnt) / float(cutof * 2), sum(sim_res) / (len(sim_res) + 1e-9), "rsprob=%.1f" % rsprob)
        rsprob_to_mean_val[rsprob] = float(tot_cnt) / float(cutof * 2)
        rsprob_to_sim_avg[rsprob] = sum(sim_res) / (len(sim_res) + 1e-9)
    # dump_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/contrast_info_exp"
    # with open(os.path.join(dump_path, "gcc_sample_len_{:d}_rsprob_{:.4f}.pkl".format(length, rsprob)), "wb") as f:
    #     pickle.dump(sim_res, f)
    # print("dumped!")
    return rsprob_to_mean_val, rsprob_to_sim_avg

def get_distance_distri_v2(rf, epoch=1, batch_idx=1, mask_times=5): # 0.2764(16, 0.0) 0.2774(64, 0.8) 0.2761(64, 0.0)
    mask_info = pickle.load(rf)
    edge_index, masked_nodes, batch = mask_info["edge_index"], mask_info["masked_nodes"], mask_info["batch"]
    gra_idx_to_nodes_idx = dict()

    for i_node in range(batch.size(0)):
        gra_idx = int(batch[i_node])
        if gra_idx not in gra_idx_to_nodes_idx:
            gra_idx_to_nodes_idx[gra_idx] = [i_node]
        else:
            gra_idx_to_nodes_idx[gra_idx].append(i_node)
    gra_idx_to_edge_list = dict()
    for i_edge in range(0, edge_index.size(1), 2):
        a, b = int(edge_index[0, i_edge]), int(edge_index[1, i_edge])
        gra_a, gra_b = int(batch[a]), int(batch[b])
        assert gra_a == gra_b
        if gra_a not in gra_idx_to_edge_list:
            gra_idx_to_edge_list[gra_a] = [(a, b), (b, a)]
        else:
            gra_idx_to_edge_list[gra_a].append((a, b))
            gra_idx_to_edge_list[gra_a].append((b, a))
    gra_idx_to_masked_nodes = dict()
    for i in range(masked_nodes.size(0)):
        node_idx = int(masked_nodes[i])
        gra_idx = int(batch[node_idx])
        if gra_idx not in gra_idx_to_masked_nodes:
            gra_idx_to_masked_nodes[gra_idx] = [node_idx]
        else:
            gra_idx_to_masked_nodes[gra_idx].append(node_idx)
    all_gra_dis = []
    all_gra_random_dis = []
    for gra_idx in gra_idx_to_nodes_idx:
        masked_nodes_list_now = [] if gra_idx not in gra_idx_to_masked_nodes else gra_idx_to_masked_nodes[gra_idx]
        edge_list_now = gra_idx_to_edge_list[gra_idx]
        node_list_now = gra_idx_to_nodes_idx[gra_idx]
        masked_nodes_list_random = np.random.choice(node_list_now, len(masked_nodes_list_now), replace=False)
        dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_now)
        random_dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_random)
        all_gra_dis += dis
        all_gra_random_dis += random_dis
    if len(gra_idx_to_masked_nodes) >= 100:
        wf = open("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/distance_epoch_{:d}_batch_{:d}_mask_times_{:d}".format(epoch, batch_idx, mask_times), "wb")
        pickle.dump({"masker_dis": all_gra_dis, "random_dis": all_gra_random_dis}, wf)
        print(sum(all_gra_dis) / (len(all_gra_dis) + 1e-9))
        print("random", sum(all_gra_random_dis) / (len(all_gra_random_dis) + 1e-9))
        print(len(gra_idx_to_nodes_idx), len(gra_idx_to_edge_list), len(gra_idx_to_masked_nodes), epoch, batch_idx)
        # the average node distances in one batch and one epoch --- and return them to the caller.
        return sum(all_gra_dis) / (len(all_gra_dis) + 1e-9), sum(all_gra_random_dis) / (len(all_gra_random_dis) + 1e-9)
    else:
        return None, None
    # return all_gra_dis, all_gra_random_dis

def get_distance_distri_v3(rf, epoch=1, batch_idx=1, mask_times=5): # 0.2764(16, 0.0) 0.2774(64, 0.8) 0.2761(64, 0.0)
    mask_info = pickle.load(rf)
    edge_index, masked_nodes, batch = mask_info["edge_index"], mask_info["masked_nodes"], mask_info["batch"]
    gra_idx_to_nodes_idx = dict()

    for i_node in range(batch.size(0)):
        gra_idx = int(batch[i_node])
        if gra_idx not in gra_idx_to_nodes_idx:
            gra_idx_to_nodes_idx[gra_idx] = [i_node]
        else:
            gra_idx_to_nodes_idx[gra_idx].append(i_node)
    gra_idx_to_edge_list = dict()
    for i_edge in range(0, edge_index.size(1), 2):
        a, b = int(edge_index[0, i_edge]), int(edge_index[1, i_edge])
        gra_a, gra_b = int(batch[a]), int(batch[b])
        assert gra_a == gra_b
        if gra_a not in gra_idx_to_edge_list:
            gra_idx_to_edge_list[gra_a] = [(a, b), (b, a)]
        else:
            gra_idx_to_edge_list[gra_a].append((a, b))
            gra_idx_to_edge_list[gra_a].append((b, a))
    gra_idx_to_masked_nodes = dict()
    for i in range(masked_nodes.size(0)):
        node_idx = int(masked_nodes[i])
        gra_idx = int(batch[node_idx])
        if gra_idx not in gra_idx_to_masked_nodes:
            gra_idx_to_masked_nodes[gra_idx] = [node_idx]
        else:
            gra_idx_to_masked_nodes[gra_idx].append(node_idx)
    all_gra_dis = []
    all_gra_random_dis = []
    for gra_idx in gra_idx_to_nodes_idx:
        masked_nodes_list_now = [] if gra_idx not in gra_idx_to_masked_nodes else gra_idx_to_masked_nodes[gra_idx]
        edge_list_now = gra_idx_to_edge_list[gra_idx]
        node_list_now = gra_idx_to_nodes_idx[gra_idx]
        masked_nodes_list_random = np.random.choice(node_list_now, len(masked_nodes_list_now), replace=False)
        dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_now)
        random_dis = to_nx_graph(edge_list_now, node_list_now, masked_nodes_list_random)
        all_gra_dis += dis
        all_gra_random_dis += random_dis

    return all_gra_dis, all_gra_random_dis
    # if len(gra_idx_to_masked_nodes) >= 100:
    #     wf = open("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/distance_epoch_{:d}_batch_{:d}_mask_times_{:d}".format(epoch, batch_idx, mask_times), "wb")
    #     pickle.dump({"masker_dis": all_gra_dis, "random_dis": all_gra_random_dis}, wf)
    #     print(sum(all_gra_dis) / (len(all_gra_dis) + 1e-9))
    #     print("random", sum(all_gra_random_dis) / (len(all_gra_random_dis) + 1e-9))
    #     print(len(gra_idx_to_nodes_idx), len(gra_idx_to_edge_list), len(gra_idx_to_masked_nodes), epoch, batch_idx)
    #     # the average node distances in one batch and one epoch --- and return them to the caller.
    #     return sum(all_gra_dis) / (len(all_gra_dis) + 1e-9), sum(all_gra_random_dis) / (len(all_gra_random_dis) + 1e-9)
    # else:
    #     return None, None

def draw_samples_distances(mask_times):
    mask_times = 3
    rf = open(
        "mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
            3), "rb")
    mask_info = pickle.load(rf)
    masker_dis, random_dis = mask_info["masker_dis"], mask_info["random_dis"]
    totlen = len(masker_dis)
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.plot(np.arange(totlen), masker_dis, "r", label="masker_dis {:d}".format(mask_times))
    plt.plot(np.arange(totlen), random_dis, "g", label="random_dis {:d}".format(mask_times))
    # plt.savefig("./figs/fig_distance_mask_{:d}.png".format(mask_times))
    plt.legend()
    plt.ylabel("Average Distances")
    plt.xlabel("Checkpoints")

    mask_times = 5
    rf = open(
        "mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
            5), "rb")
    mask_info = pickle.load(rf)
    masker_dis, random_dis = mask_info["masker_dis"], mask_info["random_dis"]
    totlen = len(masker_dis)
    plt.subplot(1, 4, 2)
    plt.plot(np.arange(totlen), masker_dis, "r", label="masker_dis {:d}".format(mask_times))
    plt.plot(np.arange(totlen), random_dis, "g", label="random_dis {:d}".format(mask_times))
    plt.legend()
    plt.xlabel("Checkpoints")

    mask_times = 6
    rf = open(
        "mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
            6), "rb")
    mask_info = pickle.load(rf)
    masker_dis, random_dis = mask_info["masker_dis"], mask_info["random_dis"]
    totlen = len(masker_dis)
    plt.subplot(1, 4, 3)
    plt.plot(np.arange(totlen), masker_dis, "r", label="masker_dis {:d}".format(mask_times))
    plt.plot(np.arange(totlen), random_dis, "g", label="random_dis {:d}".format(mask_times))
    plt.legend()
    plt.xlabel("Checkpoints")

    mask_times = 7
    rf = open(
        "mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
            7), "rb")
    mask_info = pickle.load(rf)
    masker_dis, random_dis = mask_info["masker_dis"], mask_info["random_dis"]
    totlen = len(masker_dis)
    plt.subplot(1, 4, 4)
    plt.plot(np.arange(totlen), masker_dis, "r", label="masker_dis {:d}".format(mask_times))
    plt.plot(np.arange(totlen), random_dis, "g", label="random_dis {:d}".format(mask_times))
    plt.legend()
    plt.xlabel("Checkpoints")

    plt.tight_layout()
    plt.savefig("./figs/masked_nodes_dis_masker_random.pdf")


# 83.92684911835453 (16) 77.77774058021008(32)

def plot_val_sim_sampled_by_gcc():
    # rsprob = range(0.0, 0.9, 0.1)
    rsprob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(len(rsprob))
    val = [0.2935, 0.2775, 0.2735, 0.2755, 0.2755, 0.274, 0.2805, 0.29, 0.2755]
    sim = [0.25618991274278263, 0.24523844727768437, 0.24309331629148268, 0.24251283136299465,
           0.21883947812389204, 0.22156481020371982, 0.2376175029184149, 0.24160102700933478,
           0.2506675383120859]
    plt.plot(rsprob, val, "r", label="validate possibility")
    plt.plot(rsprob, val, "r^")

    for x, y in zip(rsprob, val):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.plot(rsprob, sim, "g", label="average similarity")
    plt.plot(rsprob, sim, "g^")
    for x, y in zip(rsprob, sim):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.xlabel("restart probability")
    plt.ylabel("possibility / similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figs/sim_val_wrt_rsprob_gcc.pdf")

def plot_val_sim_sampled_by_gcc_with_std():
    # rsprob = range(0.0, 0.9, 0.1)
    rsprob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(len(rsprob))
    val = [0.2749, 0.2720, 0.2711, 0.2717, 0.2763, 0.2749, 0.2770, 0.2769, 0.2708]
    val_std = [0.0086, 0.0137, 0.0091, 0.0088, 0.0084, 0.0125, 0.0145, 0.0104, 0.0093]
    sim = [0.2425, 0.2398, 0.2402, 0.2430, 0.2439, 0.2434, 0.2426, 0.2436, 0.2405]
    sim_std = [0.0061, 0.0098, 0.0080, 0.0061, 0.0061, 0.0030, 0.0049, 0.0058, 0.0046]
    # sim = [0.25618991274278263, 0.24523844727768437, 0.24309331629148268, 0.24251283136299465,
    #        0.21883947812389204, 0.22156481020371982, 0.2376175029184149, 0.24160102700933478,
    #        0.2506675383120859]
    plt.plot(rsprob, val, "r", label="validate possibility")
    plt.plot(rsprob, val, "r^")
    for x, y, std in zip(rsprob, val, val_std):
        plt.vlines(x, y - std, y + std, color="r")
    for x, y in zip(rsprob, val):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.plot(rsprob, sim, "g", label="average similarity")
    plt.plot(rsprob, sim, "g^")
    for x, y, std in zip(rsprob, sim, sim_std):
        plt.vlines(x, y - std, y + std, color="g")
    for x, y in zip(rsprob, sim):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)

    plt.xlabel("restart probability")
    plt.ylabel("possibility / similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figs/sim_val_wrt_rsprob_gcc.pdf")


def plot_val_sim_sampled_by_drop_node_with_std():
    # rsprob = range(0.0, 0.9, 0.1)
    rsprob = [0.1, 0.15, 0.2, 0.25]
    print(len(rsprob))
    val = [0.02645, 0.0254, 0.02915, 0.03445]
    val_std = [0.0024743686063317246, 0.004042276586281548, 0.0056614927360193615, 0.0032361242250568798]
    sim = [0.400810924736014, 0.30295165131308704, 0.2424529787227993, 0.2118182713364382]
    sim_std = [0.006143885869046763, 0.004978911440940126, 0.0038776238090327576, 0.0038361991725474544]
    # sim = [0.25618991274278263, 0.24523844727768437, 0.24309331629148268, 0.24251283136299465,
    #        0.21883947812389204, 0.22156481020371982, 0.2376175029184149, 0.24160102700933478,
    #        0.2506675383120859]

    plt.plot(rsprob, val, "r", label="validate possibility")
    plt.plot(rsprob, val, "r^")
    for x, y, std in zip(rsprob, val, val_std):
        plt.vlines(x, y - std, y + std, color="r")
    for x, y in zip(rsprob, val):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.plot(rsprob, sim, "g", label="average similarity")
    plt.plot(rsprob, sim, "g^")
    for x, y, std in zip(rsprob, sim, sim_std):
        plt.vlines(x, y - std, y + std, color="g")
    for x, y in zip(rsprob, sim):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)

    plt.xlabel("drop node ratio")
    plt.ylabel("possibility / similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figs/sim_val_wrt_drnode_ratio_drop_node.pdf")

def plot_val_sim_sampled_by_drop_edge_with_std():
    # rsprob = range(0.0, 0.9, 0.1)
    rsprob = [0.1, 0.15, 0.2, 0.25]
    print(len(rsprob))
    val = [0.021799999999999996, 0.022399999999999996, 0.021099999999999997, 0.022399999999999996]
    val_std = [0.003241913015489465, 0.0036386810797320503, 0.0030479501308256335, 0.0029647934160747184]
    sim = [0.488979549612509, 0.38900640629908706, 0.316311676673045, 0.2735922921536392]
    sim_std = [0.0037130660761991656, 0.0042247000327538635, 0.0025152004093076533, 0.0020287909260074104]
    # sim = [0.25618991274278263, 0.24523844727768437, 0.24309331629148268, 0.24251283136299465,
    #        0.21883947812389204, 0.22156481020371982, 0.2376175029184149, 0.24160102700933478,
    #        0.2506675383120859]
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 2)
    plt.errorbar(rsprob, val, yerr=val_std, label="Legal Prob.")
    # plt.plot(rsprob, val, "r", label="val possi")
    # plt.plot(rsprob, val, "r^")
    # for x, y, std in zip(rsprob, val, val_std):
    #     plt.vlines(x, y - std, y + std, color="r")
    for x, y in zip(rsprob, val):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    # plt.plot(rsprob, sim, "g", label="avg sim")
    # plt.plot(rsprob, sim, "g^")
    # for x, y, std in zip(rsprob, sim, sim_std):
    #     plt.vlines(x, y - std, y + std, color="g")
    plt.errorbar(rsprob, sim, yerr=sim_std, label="Avg. Sim.")
    for x, y in zip(rsprob, sim):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)

    plt.xlabel("Drop Edge Ratio")
    plt.ylabel("Probability / Similarity")
    plt.legend()

    plt.subplot(1, 2, 1)
    rsprob = [0.1, 0.15, 0.2, 0.25]
    print(len(rsprob))
    val = [0.02645, 0.0254, 0.02915, 0.03445]
    val_std = [0.0024743686063317246, 0.004042276586281548, 0.0056614927360193615, 0.0032361242250568798]
    sim = [0.400810924736014, 0.30295165131308704, 0.2424529787227993, 0.2118182713364382]
    sim_std = [0.006143885869046763, 0.004978911440940126, 0.0038776238090327576, 0.0038361991725474544]
    # sim = [0.25618991274278263, 0.24523844727768437, 0.24309331629148268, 0.24251283136299465,
    #        0.21883947812389204, 0.22156481020371982, 0.2376175029184149, 0.24160102700933478,
    #        0.2506675383120859]
    plt.errorbar(rsprob, val, yerr=val_std, label="Legal Prob.")
    # plt.plot(rsprob, val, "r", label="val possi")
    # plt.plot(rsprob, val, "r^")
    # for x, y, std in zip(rsprob, val, val_std):
    #     plt.vlines(x, y - std, y + std, color="r")
    for x, y in zip(rsprob, val):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    # plt.plot(rsprob, sim, "g", label="avg sim")
    # plt.plot(rsprob, sim, "g^")
    # for x, y, std in zip(rsprob, sim, sim_std):
    #     plt.vlines(x, y - std, y + std, color="g")
    plt.errorbar(rsprob, sim, yerr=sim_std, label="Avg. Sim.")
    for x, y in zip(rsprob, sim):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)

    plt.xlabel("Drop Node Ratio")
    plt.ylabel("Probability / Similarity")
    plt.legend()

    # handles, labels = ax.get_legend_handles_labels()
    #
    # fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))

    plt.tight_layout()
    plt.savefig("./figs/sim_val_wrt_dredge_ratio_drop_edge_node_all_errbar.pdf")

# from utils2 import mol_to_dgl_data_obj_simple
import dgl
def get_dgl_graphs_from_smiles(DEBUG=True):
    input_path = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression='gzip', dtype='str')
    smiles_list = list(input_df['smiles'])
    dgl_gs = list()
    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            print(i)
        # if DEBUG and i >= 3000:
        #     break
        mol = Chem.MolFromSmiles(smi)
        dgl_gra = mol_to_dgl_data_obj_simple(mol)
        dgl_gs.append(dgl_gra)
    labels = torch.tensor(range(len(dgl_gs)), dtype=torch.long)
    dgl.data.utils.save_graphs(processed_dir + "/dgl_graphs.bin", dgl_gs,
                               {"graph_idx_labels": labels})
    print("all saved")

def get_avg_samples_distances():
    mask_times = [3, 5, 6, 7]
    dis = {}
    for i, mt in enumerate(mask_times):
        rf = open("mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
                mt), "rb")
        mask_info = pickle.load(rf)
        masker_dis, random_dis = mask_info["masker_dis"], mask_info["random_dis"]
        totlen = len(masker_dis)
        mea_masker_dis = sum(masker_dis) / totlen
        mea_random_dis = sum(random_dis) / totlen
        print(mt, mea_masker_dis, mea_random_dis)

def merge_soc_dgl_graphs():
    processed_dir = "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/data"
    idx_to_sim_scores = dict()
    for i in range(160000):
        pat = os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict_{:d}.npy".format(i))
        if os.path.isfile(pat):
            print("is file!", i)
            iss = np.load(pat, allow_pickle=True).item()
            for idx in iss:
                assert idx not in idx_to_sim_scores
                idx_to_sim_scores[idx] = iss[idx]
    np.save(os.path.join(processed_dir, "social_gra_idx_to_sim_score_dict.npy"), idx_to_sim_scores)
    print("saved!~")

def compute_molwt_ssr_ntom(dataset):
    # print(multiprocessing.current_process().name + " started!")
    # print(idx_range)
    st_time = time.time()
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    idx_to_mol_wt = dict()
    idx_to_mol_ssr = dict()
    idx_to_mol_natom = dict()
    st_time = time.time()
    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            print(i, time.time() - st_time)
        mol = Chem.MolFromSmiles(smi)
        ssr = len(Chem.GetSymmSSSR(mol))
        natom = len(mol.GetAtoms())
        weight = Descriptors.MolWt(mol)
        idx_to_mol_wt[i] = weight
        idx_to_mol_ssr[i] = ssr
        idx_to_mol_natom[i] = natom
    with open(os.path.join(processed_path, "mol_weights"), "w") as wf:
        for i in range(len(idx_to_mol_wt)):
            wf.write("%.4f\n" % idx_to_mol_wt[i])
        wf.close()
    print("wt writen")

    with open(os.path.join(processed_path, "mol_ssrs"), "w") as wf:
        for i in range(len(idx_to_mol_ssr)):
            wf.write("%d\n" % idx_to_mol_ssr[i])
        wf.close()
    print("ssr writen")

    with open(os.path.join(processed_path, "mol_natoms"), "w") as wf:
        for i in range(len(idx_to_mol_natom)):
            wf.write("%d\n" % idx_to_mol_natom[i])
        wf.close()
    print("natoms writen")
    print(time.time() - st_time)

def sort_and_save_idx():
    processed_path = "./dataset/zinc_standard_agent/processed"
    idx_to_wei = dict()
    with open(os.path.join(processed_path, "mol_weights"), "r") as rf:
        for i, line in enumerate(rf):
            wt = float(line.strip())
            idx_to_wei[i] = wt
    st_time = time.time()
    idx_wt_items_sorted = sorted(idx_to_wei.items(), key=lambda i: i[1])
    with open(os.path.join(processed_path, "mol_idx"), "w") as wf:
        for i, item in enumerate(idx_wt_items_sorted):
            wf.write("%d\n" % int(item[0]))
        wf.close()
    print("saved!")
    print(time.time() - st_time)
# 1187.8 11.1 (sort) 230 （get candi by c++) ~3 hrs 168 mins (compute fingerprint for each molecule) 218.3s
# (get candilist and idx_to_old_idx_dict  1257.6s  ~ 21 mins (compute similarity given fingerprint)

def get_fingerprint(dataset="zinc_standard_agent"):
    assert dataset == "zinc_standard_agent", "Others have not been implemented"
    input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
    processed_path = "./dataset/zinc_standard_agent/processed"
    input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
    smiles_list = list(input_df['smiles'])
    idx_to_fingerprint = dict()
    st_time = time.time()

    for i, smi in enumerate(smiles_list):
        if i % 1000 == 0:
            print(i, time.time() - st_time)
        mol = Chem.MolFromSmiles(smi)
        fig = Chem.RDKFingerprint(mol)
        idx_to_fingerprint[i] = fig
    wf = open(os.path.join(processed_path, "idx_to_fingerprint.pkl"), "wb")
    pickle.dump(idx_to_fingerprint, wf)
    print(time.time() - st_time)
    print("dumped!")

def get_idx_to_candi_list():
    processed_path = "./dataset/zinc_standard_agent/processed"
    idx_to_candi_list = dict()
    new_idx_to_ac_idx = dict()
    st_time = time.time()
    with open(os.path.join(processed_path, "mol_idx"), "r") as rf:
        for i, line in enumerate(rf):
            ac_idx = int(line.strip())
            new_idx_to_ac_idx[i] = ac_idx
        rf.close()
    with open(os.path.join(processed_path, "mol_candis"), "r") as rf:
        for i, line in enumerate(rf):
            cl = line.strip().split(" ")
            aci = new_idx_to_ac_idx[i]
            try:
                cl = [int(jj) for jj in cl]

                idx_to_candi_list[aci] = cl
            except:
                print(i, cl)
                idx_to_candi_list[aci] = []
        rf.close()
    print("load over")
    wf = open(os.path.join(processed_path, "new_idx_to_old_idx.pkl"), "wb")
    pickle.dump(new_idx_to_ac_idx, wf)
    wf = open(os.path.join(processed_path, "mol_idx_to_candi_list.pkl"), "wb")
    pickle.dump(idx_to_candi_list, wf)
    print("all dumped!")
    print(time.time() - st_time)


def get_similarity():
    processed_path = "./dataset/zinc_standard_agent/processed"
    rf = open(os.path.join(processed_path, "mol_idx_to_candi_list.pkl"), "rb")
    mol_idx_to_candi_list = pickle.load(rf)
    rf = open(os.path.join(processed_path, "idx_to_fingerprint.pkl"), "rb")
    mol_idx_to_fingerprint = pickle.load(rf)
    mol_idx_to_similarity = dict()
    st_time = time.time()
    totidx = 0
    for mol_idx in mol_idx_to_candi_list:
        if totidx % 1000 == 0:
            print(mol_idx, time.time() - st_time)
        totidx += 1
        candi_list = mol_idx_to_candi_list[mol_idx]
        sims = [DataStructs.FingerprintSimilarity(mol_idx_to_fingerprint[mol_idx], mol_idx_to_fingerprint[ci])
                for ci in candi_list]
        mol_idx_to_similarity[mol_idx] = sims
    wf = open(os.path.join(processed_path, "idx_to_sims_list.pkl"), "wb")
    pickle.dump(mol_idx_to_similarity, wf)


if __name__ == "__main__":
    # test_load()
    # TODO: draw the similarity distribution!
    # TODO: how to sample similar atoms with the similarities computed?
    ## TODO: 原子的种类的分布 也会影响是否是正例？ --- 但其实这些分子的原子分布已经很相近了。。
    ## TODO： 验证邻居的邻居不是我的邻居这个猜想 --- b不是特别明显。。。 --- 对于hop=2的邻居而言
    ## TODO：hop=3的邻居们？
    # compute_similarity_dataset()
    # compute_all()
    # extract_mol_weis_from_dataset()
    # extract_mol_weis_from_dataset_all()
    # compute_similarity_dataset_based_on_wei_all(num_cpu=20, DEBUGE=False)
    """
    format of the computed dict: {idx: {unsim_idx: sim_score}} 
    """
    """
    construction method --- need reconsideration? 
    """
    # compute_unsimilarity_dataset_based_on_wei_all(num_cpu=20, DEBUGE=False)
    # get_big_graph_edge_list_all(cpu_num=20, DEBUG=False)
    # transfer_pickle_error()
    # test_saved_list()
    # check_unsim_computed()
    # get_normalized_factor()
    # check_normalized_factor()
    # view_neg_samples_similarity()
    # compare_distri_sim()
    # view_neg_samples_similarity()
    # for p in range(17, 20, 1):
    #     get_neg_pos_sim_distri_n2v(3, [2.0 - (p / 10.0), p / 10.0], 1, 7)
    # for hop in range(2, 6):
        # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.2)

    #
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.4, 1.6], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.5, 1.5], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.6, 1.4], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.7, 1.3], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.8, 1.2], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[0.9, 1.1], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[1.0, 1.0], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[1.4, 0.6], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[1.5, 0.5], num_sim_samples=1, num_neg_samples=7)
    # get_neg_pos_sim_distri_n2v_one_walk(num_hops=2, neg_p_q=[1.6, 0.4], num_sim_samples=1, num_neg_samples=7)

    # plot_loss_curve()
    # draw_acc_sim_curve_rs_prob()
    # get_neg_pos_sim_distri_n2v(3, [1, 1], 1, 7)
    # get_neg_pos_sim_distri_rwr_hop(1, 7, 2)
    # draw_acc_sim_curve()
    # allgra_dis = []
    # allgra_random_dis = []
    # all_gra_dis = list()
    # all_gra_random_dis = list()
    # but all we have done is just choose what nodes to mask... without any other things...
    # so where is the rule behind dynamic masking??? --- is distances between nodes being masked matter?
    # or some other things --- to disturb the output distribution to the most extend ...
    # greedily choose what nodes to be masked..
    # or we can say that nodes chosen randomly cannot disturb the output distribution to the extend we expected
    # disturbances -- why not do the statistices over the disturbances directly??

    # ranges = [ii for ii in range(0, 100)] + [ii for ii in range(1000, 1100)] + [ii for ii in range(2000, 2100)]
    # ranges = ranges + [ii for ii in range(3000, 3100)] + [ii for ii in range(4000, 4100)] + [ii for ii in range(5000, 5100)]
    # ranges = ranges + [ii for ii in range(6000, 6100)] + [ii for ii in range(7000, 7100)]
    # masker_dis, random_mask_dis = list(), list()
    # masker_dis_per_check_point, random_mask_dis_per_check_point = list(), list()
    # for mask_times in [3, 5, 6, 7]:
    #     masker_dis, random_mask_dis = list(), list()
    #     masker_dis_per_check_point, random_mask_dis_per_check_point = list(), list()
    #     for epoch in range(1, 5):
    #         for batch in ranges:
    #             print(mask_times, epoch, batch)
    #             if batch % 100 == 0 and batch != 0:
    #                 print(epoch, batch)
    #                 masker_dis_per_check_point.append(sum(masker_dis) / len(masker_dis))
    #                 random_mask_dis_per_check_point.append(sum(random_mask_dis) / len(random_mask_dis))
    #                 masker_dis = list()
    #                 random_mask_dis = list()
    #             rf = open("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/all_gra_dis_epoch_{:d}_batch_{:d}_{:d}".format(epoch, batch, mask_times), "rb")
    #             a, b = get_distance_distri_v3(rf, epoch, batch, mask_times)
    #             masker_dis += a
    #             random_mask_dis += b
    #         if len(random_mask_dis) != 0:
    #             masker_dis_per_check_point.append(sum(masker_dis) / (len(masker_dis) + 1e-9))
    #             random_mask_dis_per_check_point.append(sum(random_mask_dis) / (len(random_mask_dis) + 1e-9))
    #             masker_dis = list()
    #             random_mask_dis = list()
    #     with open(
    #             "/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(
    #                     mask_times), "wb") as wf:
    #         pickle.dump({"masker_dis": masker_dis_per_check_point, "random_dis": random_mask_dis_per_check_point}, wf)
    #         print("all dumped!")

            # for batch in range(0, 7394, 100):
            #     rf = open("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/all_gra_dis_epoch_{:d}_batch_{:d}_{:d}".format(epoch, batch, mask_times), "rb")
            #     a, b = get_distance_distri_v2(rf, epoch, batch, 5)
            #     # allgra_dis += a
            #     #
            #     # allgra_random_dis += b
            #     if a is not None and b is not None:
            #         all_gra_dis.append(a)
            #         all_gra_random_dis.append(b)
        # print("tot dis", sum(allgra_dis) / len(allgra_dis))
        # print("tot random dis", sum(allgra_random_dis) / len(allgra_random_dis))
        # with open("/apdcephfs/private_meowliu/ft_local/gnn_pretraining/mask_info_exp/distances_for_each_sample_step_mask_times_{:d}".format(mask_times), "wb") as wf:
        #     pickle.dump({"masker_dis": all_gra_dis, "random_dis": all_gra_random_dis}, wf)
        #     print("all dumped!")
    # draw_samples_distances(3)
    # draw_acc_sim_curve_rs_prob()
    # draw_acc_sim_curve()
    # get_gcc_sample_validity()
    # rsprob = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # rsprob_to_mean_val = dict()
    # rsprob_to_sim_avg = dict()
    # for rsp in rsprob:
    #     rsprob_to_sim_avg[rsp] = list()
    #     rsprob_to_mean_val[rsp] = list()
    # #
    # for _ in range(10):
    #     val, sim = get_gcc_sample_validity()
    #     for rsp in val:
    #         rsprob_to_mean_val[rsp].append(val[rsp])
    #         rsprob_to_sim_avg[rsp].append(sim[rsp])
    # for rsp in rsprob:
    #     print(rsp, "val_mean: %.4f, val_std: %.4f, sim_mean: %.4f, sim_std: %.4f" % (np.mean(rsprob_to_mean_val[rsp]),
    #                                                                                  np.std(rsprob_to_mean_val[rsp]),
    #                                                                                  np.mean(rsprob_to_sim_avg[rsp]),
    #                                                                                  np.std(rsprob_to_sim_avg[rsp]))
    #           )

    # plot_val_sim_sampled_by_drop_edge_with_std()
    # compute_molwt_ssr_ntom("zinc_standard_agent")
    # sort_and_save_idx()
    # get_fingerprint()
    # get_idx_to_candi_list()
    get_similarity()
    # rsprob = [0.1, 0.15, 0.2, 0.25]
    # rsprob_to_mean_val = dict()
    # rsprob_to_sim_avg = dict()
    # for rsp in rsprob:
    #     rsprob_to_mean_val[rsp] = []
    #     rsprob_to_sim_avg[rsp] = []
    # for _ in range(10):
    #     val, sim = get_dropedge_samples_validity()
    #     for rsp in val:
    #         rsprob_to_mean_val[rsp].append(val[rsp])
    #         rsprob_to_sim_avg[rsp].append(sim[rsp])
    # mean_val, std_val, mean_sim, std_sim = [], [], [], []
    # for rsp in rsprob:
    #     print(rsp, "val_mean: %.4f, val_std: %.4f, sim_mean: %.4f, sim_std: %.4f" % (np.mean(rsprob_to_mean_val[rsp]),
    #                                                                                  np.std(rsprob_to_mean_val[rsp]),
    #                                                                                  np.mean(rsprob_to_sim_avg[rsp]),
    #                                                                                  np.std(rsprob_to_sim_avg[rsp]))
    #           )
    #     mean_val.append(np.mean(rsprob_to_mean_val[rsp]))
    #     std_val.append(np.std(rsprob_to_mean_val[rsp]))
    #     mean_sim.append(np.mean(rsprob_to_sim_avg[rsp]))
    #     std_sim.append(np.std(rsprob_to_sim_avg[rsp]))
    # print(mean_val)
    # print(std_val)
    # print(mean_sim)
    # print(std_sim)

    # plot_val_sim_sampled_by_drop_edge_with_std()
    # get_dgl_graphs_from_smiles(DEBUG=True)
    # draw_acc_sim_curve_rs_prob_n2v_q(dataset="sider")
    # plot_val_sim_sampled_by_drop_node_with_std()
    # get_avg_samples_distances()
    # merge_soc_dgl_graphs()
    # plot_val_sim_sampled_by_gcc_with_std()

    # processed_path = "./dataset/zinc_standard_agent/processed"
    # with open(os.path.join(processed_path, "big_graph_n2v_graph_p_{}_q_{}.pkl".format(str(0.7), str(0.6))),
    #         "rb") as f:
    #     big_graph_n2v_graph = pickle.load(f)
    #     print("loaded!!")

    # get_neg_pos_sim_distri_rwr_hop_big_gra()
    # get_degree_array()
    # hop = 3
    # draw_samples_distances(7)
    # compute_degree_tensor()
    # get_neg_sampling_possi()
    # get_info_dgl_big_gra()
    # get_neg_samples_idx()
    # get_best_aucs_from_files()
    # get_gcc_sample_validity()
    # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.1)
    # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.3)
    # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.4)
    # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.5)
    # get_neg_pos_sim_distri_rwr_hop(1, 7, hop, 0.6)
    # for hop in range(2, 6):
    #     get_neg_pos_sim_distri_n2v(hop, [0.4, 1.6], 1, 7)
    # get_mol_sim_smiles()
    # plot_smiles()
    # dgl_g = dgl.DGLGraph()
    # dgl_g.add_nodes(3 + 1)
    # dgl_g.add_edges(torch.tensor([0, 1, 1, 2, 3]), torch.tensor([1, 2, 3, 0, 0]))
    #
    #
    # # g1 = dgl.graph(([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]))
    # prob = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0])
    # dgl_g.edata['p'] = prob
    # res = rwr_sample_neg_graphs(dgl_g, 0)
    # print(res)

    # extract_mol_weis_from_dataset_all()
