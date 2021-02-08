#!/bin/bash
gpu=$1
ARGS=${@:2}
# bash scripts/fint.sh 3 --moco --nce-k 256 # h-index 0.8180 usa_airport 0.7060  collab gin 0.72 con 0.736 imdb-multi con 0.513 gin 0.467
python train_graph_moco.py --exp Finetune --dataset imdb-multi --finetune --model-path saved --tb-path tensorboard --resume saved/Pretrain_moco_True_con_dgl_la_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_1.pth --gpu $gpu $ARGS