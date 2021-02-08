#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch.py --device 0 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 --T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph --rw_hops 16 --num_path 7 --num_com_negs 49 --env jizhi # --val_dataset bbbp # --data_para
#### for sider
#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_mask_nll.py \
#--device 0 --epochs 100 --mask_times 5 --batch_size 256 --gnn_type gin --env jizhi \
#--neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
#--construct_big_graph --rw_hops 4 --num_path 7 --num_com_negs 256 --lr 0.0080
# --val_dataset bbbp # --data_para

#### for bace
#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_mask_nll.py \
#--device 0 --epochs 100 --mask_times 5 --batch_size 256 --gnn_type gin --env jizhi \
#--neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
#--construct_big_graph --rw_hops 12 --num_path 11 --num_com_negs 256 --lr 0.0094

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_mask_nll.py \
--device 0 --epochs 100 --mask_times 5 --batch_size 256 --gnn_type gin --env jizhi \
--neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
--construct_big_graph --rw_hops 4 --num_path 7 --num_com_negs 256 --lr 0.0026

# "temp/contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.6_q_1.4_rw_hops_16_num_path_7_dl_other_p_w2v_neg_p_mask_times_5_mask_rate_{:.4f}".format(
#        args.neg_sample_stra,
#        args.num_samples,
#        args.num_neg_samples,
#        args.T,
#        args.select_other_node_stra,
#        args.select_other_node_hop,
#        str(args.restart_prob),
#        args.p,
#        args.q,
#        args.rw_hops,
#        args.num_path,
#        args.mask_times,
#        args.mask_rate)