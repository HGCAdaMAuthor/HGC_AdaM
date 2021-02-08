#!/bin/sh
model=$1
gpu=$2
gnn=$3
for dataset in bbbp bbbp clintox clintox bace bace sider sider hiv hiv
do
set=${dataset}
python finetune_other_fea.py --input_model_file ${model} --device ${gpu} --test_ratio 0.1 --train_ratio 0.8 --dataset ${set} --gnn_type ${gnn} --vocab_path ./dataset/zinc_standard_agent/processed/atom_vocab.pkl
done

# contrast_sim_based_vallina_2_3_20
# contrast_sim_based_vallina_3_1
# contrast_sim_based_vallina_4_3_15 rwr_sample
# contrast_sim_based_vallina_rwr_precomputed_1_5 rwr_precomputed
# temp/contrast_sim_based_vallina_rwr_precomputed_1_5.pth
# bash ./scripts/finetune_model.sh temp/contrast_sim_based_neg_rwr_neg_1_num_neg_samples_32_T_7_2.pth 4  ### todo
# python finetune.py --input_model_file temp/masker_exp/masker_entropy_based_gnn_graphsage_mask_times_v2_1_batch_size_256_30.pth --device 0 --test_ratio 0.1 --train_ratio 0.8 --dataset bbbp --gnn_type graphsage
# how to negative sample ---- still choose some similar but not such similar moleculars?
# negative sampling methods and positive sampling methods
# 写脚本做实验  想用automl调参。。。 --- restart prob  ### todo: check一下构造图的时候是否真的中间截断了？按照相似度截断的？？
#### todo：继续构造一个负样本的图 --- 或许我们可以用一个unsim graph来比较好的解决负采样的问题？？ --- 传递中的负样本
# bash scripts/finetune_other_fea.sh temp/masker_entropy_based_gnn_gin_mask_times_3_batch_size_256_reverse_False_other_fea_in_also_max_size_966_15.pth 6 gin
# bash scripts/finetune_model.sh temp/contrast_sim_pos_rwr_9_gnn_gin_1.pth 6 gin

#