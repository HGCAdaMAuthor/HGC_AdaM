import datetime
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_wei_distri_hist():
    processed_path = "./dataset/zinc_standard_agent/processed"
    idx_value_list_sorted = np.load(os.path.join(processed_path, "idx_to_weis_list_sorted.npy"))#.item()
    print(type(idx_value_list_sorted))
    print(idx_value_list_sorted[0])
    all_vals = [float(item[1]) for item in idx_value_list_sorted]
    print(len(all_vals), all_vals[0])
    probability_distribution(all_vals)

def probability_distribution(data, bins_interval=1, margin=1):
    bins = range(int(min(data)), int(max(data)) + bins_interval - 1, bins_interval)
    print(len(bins))
    # for i in range(0, len(bins)):
    #     print(bins[i])
    plt.xlim(int(min(data)) - margin, int(max(data)) + margin)
    plt.title("Probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    # 频率分布normed=True，频次分布normed=False
    prob,left,rectangle = plt.hist(x=data, bins=bins, normed=True, histtype='bar', color=['r'])
    for x, y in zip(left, prob):
        # 字体上边文字
        # 频率分布数据 normed=True
        plt.text(x + bins_interval / 2, y + 0.003, '%.2f' % y, ha='center', va='top')
        # 频次分布数据 normed=False
        # plt.text(x + bins_interval / 2, y + 0.25, '%.2f' % y, ha='center', va='top')
    # plt.show()
    plt.savefig("distri.png")

def plot_similarities_distri():
    ps = "./dataset/zinc_standard_agent/processed"
    sim_to_score_dict_list = np.load(os.path.join(ps, "sim_to_score_dict_list_wei_nc_natom.npy"))
    scores = list()
    print(type(sim_to_score_dict_list), len(sim_to_score_dict_list))
    sim_score_to_num_edges = dict()
    st_sim_score, ed_sim_score = 0.3, 0.5 
    sim_score_interval = 0.01
    num_scores = ((ed_sim_score - st_sim_score) // sim_score_interval) + 1
    candi_scores = [(st_sim_score + i * (sim_score_interval)) for i in range(int(num_scores))]
    discreted_sims = {i: 0 for i in range(101)}

    for i in range(len(sim_to_score_dict_list)):
        score_for_one_mol = sim_to_score_dict_list[i]
        # print(type(score_for_one_mol))
        for other_idx in score_for_one_mol:
            other_sim_score = score_for_one_mol[other_idx]
            discreted_other_sim_score = int(other_sim_score * 100)
            assert discreted_other_sim_score >= 0 and discreted_other_sim_score <= 100
            discreted_sims[discreted_other_sim_score] += 1

            # scores.append(score_for_one_mol[other_idx] * 100)
    # scores_sorted = sorted(scores, reverse=False)
    # probability_distribution(scores_sorted)
    print(discreted_sims)
    np.save("./plot_imgs/discreted_sims_dict.npy", discreted_sims)

def get_cumsum_nums():
    discreted_sims = np.load("./plot_imgs/discreted_sims_dict.npy").item()
    # print(type(discreted_sims))
    # print(type(discreted_sims.item()))
    cumsum = 0
    for j in range(100, -1, -1):
        cumsum += discreted_sims[j]
        discreted_sims[j] = cumsum
    np.save("./plot_imgs/discreted_sims_cumsum_dict.npy", discreted_sims)
    print(discreted_sims)

def get_graph_edges_by_thred():
    thred = 0.36
    ps = "./dataset/zinc_standard_agent/processed"
    sim_to_score_dict_list = np.load(os.path.join(ps, "sim_to_score_dict_list_wei_nc_natom.npy"))
    # scores = list()
    print(type(sim_to_score_dict_list), len(sim_to_score_dict_list))
    lenn = len(sim_to_score_dict_list)
    node_idx_to_adjs = dict()
    import time
    st_time = time.time()
    adj_num = dict()
    for i in range(lenn):
        if i % 1000 == 0:
            print(i, time.time() - st_time)
        per_sim_dict = sim_to_score_dict_list[i]
        adjs = list()
        for other_idx in per_sim_dict:
            other_sim_score = per_sim_dict[other_idx]
            if other_sim_score >= thred:
                adjs.append(other_idx)
        node_idx_to_adjs[i] = adjs
        # adj_num.append(len(adjs))
        an = len(adjs)
        if an not in adj_num:
            adj_num[an] = 1
        else:
            adj_num[an] += 1
    print(time.time() - st_time)
    assert len(node_idx_to_adjs) == len(sim_to_score_dict_list)
    np.save(os.path.join(ps, "node_idx_to_adj_idx_list_dict.npy"), node_idx_to_adjs)
    np.save(os.path.join(ps, "adj_num_dict.npy"), adj_num)
    print(adj_num)



if __name__ == '__main__':
    # plot_wei_distri_hist()
    # plot_similarities_distri()
    # get_cumsum_nums()
    get_graph_edges_by_thred()