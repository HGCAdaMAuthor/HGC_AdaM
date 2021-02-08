gpu=$1
load_path=$2
ARGS=${@:3}

for dataset in $ARGS
do
    python test_graph_moco.py --gpu $gpu --dataset $dataset --load-path $load_path
done
# python test_graph_moco.py --gpu 0 --dataset kdd --load-path ./saved/con_dgl/ckpt_epoch_100.pth  kdd icdm sigir cikm sigmod icde
# python test_graph_moco.py --gpu 3 --dataset usa_airport --load-path saved/Pretrain_moco_True_con_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_256_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_0_pos_32_momentum_0.999/ckpt_epoch_100.pth