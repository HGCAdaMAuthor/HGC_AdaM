python /apdcephfs/private_meowliu/ft_local/graph_self_learn/train_graph_moco.py \
--exp Pretrain \
--model-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/saved \
--tb-path /apdcephfs/private_meowliu/ft_local/graph_self_learn/tensorboard \
--gpu 0 \
--dataset con_dgl_mol_data \
--num-workers 1