#python3 /apdcephfs/private_meowliu/ft_local/graph_self_learn/train_graph_moco.py \
#--exp Pretrain \
#--model-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved \
#--tb-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/tensorboard \
#--gpu 0 \
#--moco \
#--nce-k 256 \
#--rw-hops 16 \
#--num_path 7 \
#--dataset con_dgl_resam_l \
#--env jizhi \
#--batch-size 32 \
#--num-workers 1
#--resume /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/Pretrain_moco_True_con_dgl_other_sample_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_16_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_32.pth
#
#python3 /apdcephfs/private_meowliu/ft_local/graph_self_learn/test_graph_moco.py \
#--gpu 0 \
#--dataset sigmod \
#--load-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/con_dgl/ckpt_epoch_100.pth

python /apdcephfs/private_meowliu/ft_local/graph_self_learn/sim_sea.py \
--model from_numpy_align \
--dataset sigmod_icde \
--seed 0 \
--hidden-size 64 \
--model from_numpy_align \
--emb-path-1 "/apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/con_dgl/sigmod.npy" \
--emb-path-2 "/apdcephfs/private_meowliu/ft_local/graph_self_learn/saved/con_dgl/icde.npy"