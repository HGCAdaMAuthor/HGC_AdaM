#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch.py --device 0 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 --T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph --rw_hops 16 --num_path 7 --num_com_negs 255 --gnn_type gcn --env jizhi # --val_dataset bbbp # --data_para
#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_masker_nll_with_eval.py --device 0 --epochs 100 --mask_times 5 --batch_size 256 --gnn_type gin --env jizhi --val_dataset bbbp # --val_dataset bbbp # --data_para
#python3 \
#/apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_masker_nll_masked_node_exp.py \
#--device 0 \
#--epochs 100 \
#--mask_times 7 \
#--batch_size 256 \
#--gnn_type gin

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch.py \
#--device 0 --epochs 100 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
#--T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph \
#--rw_hops 2 --num_path 5 --num_com_negs 255 --env jizhi --gnn_type gin

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_mask_nll.py \
--device 0 --epochs 100 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
--T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph \
--rw_hops 2 --num_path 5 --num_com_negs 255 --env jizhi --gnn_type graphsage --mask_times 5 --mask_rate 0.15
# 16 5; 8 10; 4 20; 2 40

# contrast_gcc 25.22 2_5_rw 41.92
#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_sim_exp.py \
#--device 0 --epochs 100 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
#--T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph \
#--rw_hops 2 --num_path 32 --num_com_negs 255 --env jizhi --gnn_type gin

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_sim_exp.py \
#--device 0 --epochs 100 --num_neg_samples 7 --neg_sample_stra biased_rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
#--T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.2 --construct_big_graph \
#--rw_hops 16 --num_path 5 --num_com_negs 255 --env jizhi --gnn_type gin

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_masker_nll.py \
#--device 0 --epochs 100 \
#--env jizhi --gnn_type gin --mask_times 1 --mask_rate 0.15

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_subgraph.py \
#--device 0 --epochs 100 \
#--gnn_type gin
# 3 -- 68.62 4 -- 72.66 5 -- 75.53 6 -- 78.78 7 -- 81.12 1 -- 56.12

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_soc_datasets.py \
#--device 0 --epochs 100 --num_samples 3 \
#--env jizhi --gnn_type gin --batch_size 32

# "kdd", "icdm", "sigir", "cikm", "sigmod", "icde"

#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/generate.py \
#--device 0 --epochs 100 --dataset icde \
#--env jizhi --gnn_type gin --batch_size 32 \
#--input_model_file /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pt_saved_model/contrast_sim_based_vallina_other_sim_other_dataset_3_gnn_gin_55.pth