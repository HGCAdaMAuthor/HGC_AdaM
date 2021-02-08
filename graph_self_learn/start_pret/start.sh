python /apdcephfs/share_1142145/meowliu/graph_self_learn/train_graph_moco.py \
--exp Pretrain \
--model-path /apdcephfs/share_1142145/meowliu/graph_self_learn/saved \
--tb-path /apdcephfs/share_1142145/meowliu/graph_self_learn/tensorboard \
--gpu 0 \
--dataset con_dgl_mol_data \
--num-workers 2 \
--batch-size 256