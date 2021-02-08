#!/bin/sh
model=$1
gpu=$2
for dataset in bbbp clintox bace sider toxcast tox21 hiv
do
set=${dataset}
python finetune.py --input_model_file ${model} --device ${gpu} --test_ratio 0.1 --train_ratio 0.8 --dataset ${set}
done

# contrast_sim_based_vallina_2_3_20
# contrast_sim_based_vallina_3_1
# contrast_sim_based_vallina_4_3_15 rwr_sample
# contrast_sim_based_vallina_rwr_precomputed_1_5 rwr_precomputed
# temp/contrast_sim_based_vallina_rwr_precomputed_1_5.pth
# bash ./scripts/finetune_model_2.sh temp/contrast_sim_based_vallina_rwr_with_neg_1_9.pth 5  ### todo

# how to negative sample ---- still choose some similar but not such similar moleculars?
