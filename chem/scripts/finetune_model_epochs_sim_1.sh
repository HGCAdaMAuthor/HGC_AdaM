#!/bin/sh
gpu=$1
for epoch in 5 10 15
do
epochNum=${epoch}
python finetune.py --input_model_file temp/contrast_sim_based_vallina_3_1_${epochNum}.pth --device ${gpu} --test_ratio 0.1 --train_ratio 0.8 --dataset hiv
done

# contrast_sim_based_vallina_2_3_20
# contrast_sim_based_vallina_3_1
# contrast_sim_based_vallina_4_3_15 rwr_sample
# contrast_sim_based_vallina_rwr_precomputed_1_5 rwr_precomputed
# temp/contrast_sim_based_vallina_rwr_precomputed_1_5.pth
# bash ./scripts/finetune_model.sh temp/contrast_sim_based_vallina_rwr_with_precomputed_neg_1_5.pth 2  ### todo
# python finetune.py --input_model_file temp/contrast_sim_based_vallina_rwr_with_neg_1_num_neg_samples_32_17.pth --device 3 --test_ratio 0.1 --train_ratio 0.8 --dataset hiv
# how to negative sample ---- still choose some similar but not such similar moleculars?
# negative sampling methods and positive sampling methods
# 写脚本做实验 for rwr_neg; for uniform_neg --- 先看一下最大的epoch的结果 如果可以再看其他epochs的结果; batch_neg :不同epochs的结果？