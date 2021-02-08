# from rdkit import Chem, DataStructs
# from typing import List
# import numpy as np
# import torch
# import pandas as pd
# import random
# import os
# import multiprocessing
# from rdkit.Chem import Descriptors, PandasTools
# from os.path import join
#
# input_path = "./dataset/zinc_standard_agent/raw/zinc_combined_apr_8_2019.csv.gz"
# processed_path = "./dataset/zinc_standard_agent/processed"
# input_df = pd.read_csv(input_path, sep=',', compression="gzip", dtype='str')
# smiles_list = list(input_df['smiles'])
#
# ps = "./dataset/zinc_standard_agent/processed"
# sim_list = np.load(join(ps, "idx_to_weis_list_wei_nc_natom_sorted.npy"))
# print(int(sim_list[0][0]))
# sim_dict = np.load(os.path.join(ps, "sim_to_score_dict_list_wei_nc_natom.npy"))
# print(type(sim_dict))
# print(len(sim_dict))
# print(sim_dict[0])
#
# sidx, tidx = 995113, 43210
# sm, tm = smiles_list[sidx], smiles_list[tidx]
# print(sm, tm)
# smol, tmol = Chem.MolFromSmiles(sm), Chem.MolFromSmiles(tm)
# print([Descriptors.ExactMolWt(smol), len(Chem.GetSymmSSSR(smol)), len(smol.GetAtoms())])
# print([Descriptors.ExactMolWt(tmol), len(Chem.GetSymmSSSR(tmol)), len(tmol.GetAtoms())])


# from scipy.stats import norm
# import math
#
# print(norm.ppf(0.5))
# q = norm.cdf(math.sqrt(250) * math.sin(math.pi / 16))  #累计密度函数
#
# print(0.5 * (1-q))
# q = norm.ppf(1 - 1e-4)
# print(q)
# print(q * q * 2e-10)
# q = norm.cdf(math.sqrt(24))
# print(2 * (1 - q))
# print(q)
# print((q * q) * 2)
# print((1- q) / 2)

# class aa:
#     pass
#
# import torch
# # ori_a = torch.tensor([0.998, 0.766, 1.990, 4.98, 9.008, 0.996])
# # print(torch.argsort(ori_a, dim=0, descending=True))
# # args = aa()
# # args.loop_epoch = 20
# # T = 1
# # for i in range(1, 21):
# #     a = ori_a.clone()
# #     sort_dims = torch.argsort(a, dim=0, descending=True)
# #     args.epoch = i
# #     st_idx = int((float((args.epoch - 1) % args.loop_epoch) / float(args.loop_epoch)) * \
# #                                              a.size(0))
# #     block_idx = sort_dims[: st_idx]
# #     # then a after softmax is regarded as the probabilities
# #     a = torch.softmax(a / T, dim=0)
# #     a[block_idx] = 0.0
# #     masked_prob = a / torch.sum(a, dim=0)
# #     print(masked_prob)
# #     masked_nodes = torch.multinomial(masked_prob, 1, replacement=False)
# #     print(st_idx)
# #     print(block_idx)
# #     print(masked_nodes)
#
# import numpy as np
# #entropy = torch.tensor([0.000, 0.001, 0.002, 1.009, 1.009, 1.009, 1.009, 1.009, 0.778])
# #
# # T = 1
# # entropy = torch.softmax(entropy / T, dim=0)
# # entropy[torch.tensor([0], dtype=torch.long)] = 0.0
# # print(entropy)
# # permute_idxes = np.arange(entropy.size(0))
# #
# # np.random.shuffle(permute_idxes)
# # np.random.shuffle(permute_idxes)
# # print(permute_idxes)
# # permute_idxes = torch.from_numpy(permute_idxes).to(entropy.device)
# # permute_entropys = entropy[permute_idxes]
# # print(entropy)
# # print(permute_entropys)
# # sort_dims = torch.argsort(permute_entropys, dim=0, descending=True)  # sort entropy in descending order
# # print(sort_dims)
# # num_mask_nodes = 2
# # mask_idxes = sort_dims[:num_mask_nodes]
# # mask_idxes = permute_idxes[mask_idxes]
# # print(mask_idxes)
# import math
#
# ps = [0.49, 0.26, 0.12, 0.04, 0.04, 0.03, 0.02]
# h = 0.0
# for i, p in enumerate(ps):
#     h += (-1) * (p * math.log(p, 2))
#
# print(h)
#
# print(1.75 - 3./8. * math.log(3./16., 2))
#
# print(math.pow(126., 0.75))

# import re
# pat = re.compile(r'(?:auc = )\d+\.?\d*')
#
# patstr = '''
# time = 0.1111, auc = 0.8022
# time = 0.1111, auc = 0.8055
# time = 0.1111, auc = 0.8055
# '''
# rf = open("./other_datas/out.log")
# lines = rf.read()
# print(type(lines))
# res = pat.findall(lines)
#
# # res = pat.findall(patstr)
#
# print(res)
#
# ans = 0
#
# for ress in res:
#     tmpans = ress.split(" ")[-1]
#     ans = max(ans, float(tmpans))
#
# print(ans)

import torch
a = torch.tensor([0.9, 0.5, 0.6, 0.2, 0.1])
b = torch.softmax(a, dim=-1)
print(b)
print(a / torch.sum(a))
