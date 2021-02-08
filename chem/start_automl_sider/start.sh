
log_dir=/apdcephfs/private_meowliu/$TASK_FLAG/$METRICS_TRIAL_NAME
mkdir -p $log_dir

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch_with_eval.py \
--device 0 --epochs 20 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 \
--T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph \
--rw_hops 16 --num_path 7 --num_com_negs 255 --env jizhi --val_dataset sider --gnn_type gcn \
> $log_dir/out.log 2>&1

#contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_4_num_path_7_dl_other_p_w2v_neg_p_no_automl