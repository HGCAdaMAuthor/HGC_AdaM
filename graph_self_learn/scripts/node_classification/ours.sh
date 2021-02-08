#!/bin/bash
hidden_size=$1
ARGS=${@:2}

for dataset in $ARGS
do
    python cogdl/scripts/train.py --task unsupervised_node_classification --dataset $dataset --seed 0 --hidden-size $hidden_size --model from_numpy --emb-path "saved/$dataset.npy"
done


# python cogdl/scripts/train.py --task unsupervised_node_classification --dataset usa_airport --seed 0 --hidden-size 64 --model from_numpy --emb-path saved/Pretrain_moco_True_con_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/usa_airport.npy
