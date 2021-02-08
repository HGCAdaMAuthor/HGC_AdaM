#!/bin/bash
gpu=$1
ARGS=${@:2}
# bash scripts/fint.sh 2 --moco --nce-k 256 # h-index 0.8180 usa_airport 0.7060
#python train_graph_moco.py --exp Finetune --dataset imdb-binary --finetune --model-path saved --tb-path tensorboard --resume saved/Pretrain_moco_True_con_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_100.pth --gpu $gpu $ARGS
python train_graph_moco.py --exp Finetune --dataset imdb-multi --finetune --model-path saved --tb-path tensorboard --resume saved/p1/ckpt_epoch_33.pth --gpu 6 --moco --nce-k 256 $gpu $ARGS