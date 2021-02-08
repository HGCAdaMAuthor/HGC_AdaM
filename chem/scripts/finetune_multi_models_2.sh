#!/bin/sh
model=$1
gpu=$2
gnn=$3
for ver in 20 15 10 5 1
do
nowmol=${model}_${ver}.pth
for dataset in bbbp bbbp clintox clintox bace bace sider sider # hiv hiv
do
set=${dataset}
python finetune.py --input_model_file ${nowmol} --device ${gpu} --test_ratio 0.1 --train_ratio 0.8 --dataset ${set} --gnn_type ${gnn}
done
done

# negative sampling methods and positive sampling methods
# bash scripts/finetune_multi_models_2.sh temp/contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_02_p_07_q_06_rw_hops_16_num_path_7_dl_other_p_w2v_neg_p 0 gin
# bash scripts/finetune_model.sh temp/contrast_sim_pos_rwr_9_gnn_gin_1.pth 6 gin