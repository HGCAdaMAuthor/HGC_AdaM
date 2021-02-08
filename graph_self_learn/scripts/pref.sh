# #!/usr/bin/env bash

##!/usr/bin/env bash
##!/usr/bin/env bas
#!/bin/bash
gpu=$1
ARGS=${@:2}

python train_graph_moco.py --exp Finetune4 --model-path saved --tb-path tensorboard  --gpu 0 --moco --nce-k 256  --dataset usa_airport --finetune --resume ./saved/Pretrain_moco_True_usa_airport_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_100.pth