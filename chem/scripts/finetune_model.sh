#!/bin/sh
model=$1
gpu=$2
gnn=$3
for dataset in  sider clintox bace bbbp  sider clintox bace bbbp hiv hiv
do
set=${dataset}
python finetune.py --input_model_file ${model} --device ${gpu} --test_ratio 0.1 --train_ratio 0.8 --dataset ${set} --gnn_type ${gnn}
done

python finetune.py --input_model_file temp/masker_entropy_based_mask_times_5_5.pth --device 2 --dataset toxcast --gnn_type gin --batch_size 64 --dropout_ratio 0.4 --lr 0.01

# contrast_sim_based_vallina_2_3_20
# contrast_sim_based_vallina_3_1
# contrast_sim_based_vallina_4_3_15 rwr_sample
# contrast_sim_based_vallina_rwr_precomputed_1_5 rwr_precomputed
# temp/contrast_sim_based_vallina_rwr_precomputed_1_5.pth
# bash ./scripts/finetune_model.sh temp/contrast_sim_based_neg_rwr_neg_1_num_neg_samples_32_T_7_2.pth 4  ### todo
# python finetune.py --input_model_file temp/masker_exp/masker_entropy_based_gnn_graphsage_mask_times_v2_1_batch_size_256_30.pth --device 0 --test_ratio 0.1 --train_ratio 0.8 --dataset bbbp --gnn_type graphsage
# how to negative sample ---- still choose some similar but not such similar moleculars?
# negative sampling methods and positive sampling methods
# bash scripts/finetune_model.sh temp/contrast_subgraph_15.pth 0 gin
# bash scripts/finetune_model.sh model_gin/supervised_infomax.pth 6 gin

#