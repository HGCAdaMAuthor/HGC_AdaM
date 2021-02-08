python3 /apdcephfs/private_meowliu/ft_local/graph_self_learn/train_graph_moco.py \
--exp Finetune \
--model-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved \
--tb-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/tensorboard \
--gpu 0 \
--moco \
--nce-k 256 \
--rw_hops 16 \
--num_path 7 \
--dataset imdb-binary \
--env jizhi \
--batch-size 32 \
--finetune \
--resume /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/con_dgl/ckpt_epoch_100.pth
#  usa_airport imdb-multi collab
#  con_dgl/ckpt_epoch_100.pth con_dgl_la/ckpt_epoch_44.pth
#  /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/Pretrain_moco_True_con_dgl_other_sample_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_16_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_33.pth