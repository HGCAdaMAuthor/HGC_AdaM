#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch.py --device 0 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 --T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph --rw_hops 16 --num_path 7 --num_com_negs 49 --env jizhi # --val_dataset bbbp # --data_para
#log_dir=/apdcephfs/private_meowliu/$TASK_FLAG/$METRICS_TRIAL_NAME
#mkdir -p $log_dir

for dataset in hiv hiv hiv # bace bace bace clintox clintox clintox bbbp bbbp bbbp sider sider sider hiv hiv hiv tox21 tox21 tox21 toxcast toxcast toxcast
do

set=${dataset}

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/finetune_automl.py \
--device 0 --epochs 100 --batch_size 128 --gnn_type gin --env jizhi --dataset ${set} \
--input_model_file /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pt_saved_model/contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_2_num_path_5_dl_other_p_w2v_neg_p_no_automl_6.pth \
--lr 0.001 --dropout_ratio 0.6 --no_automl --no_val --lr_scale 0.7
# --val_dataset bbbp # --data_para

done
# --input_model_file /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pt_saved_model/supervised_contextpred.pth \
#
# imdb-binary imdb-binary imdb-binary imdb-multi imdb-multi imdb-multi rdt-5k rdt-5k rdt-5k rdt-b rdt-b rdt-b
# collab collab collab 0.4597 0.763 0.8175 0.6011
# bace: 2_40 0.7245(0.0368) 4_20 0.7968(0.0029) 8_10 0.7504(0.0120) 16_5 0.7287(0.0248)
# clintox: 2_40 0.5678(0.0195) 4_20 0.7655(0.0248) 8_10 0.7128(0.0384) 16_5 0.6174(0.0814)
# sider: 2_40 0.6042(0.0180) 4_20 0.5939(0.0064) 8_10 0.5803(0.0106) 16_5 0.5935(0.0135)
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_2_num_path_5_dl_other_p_w2v_neg_p_no_automl_6.pth
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_2_num_path_5_dl_other_p_w2v_neg_p_mask_times_5_mask_rate_0.1500_gnn_gin_6.pth
# contrast_sim_based_vallina_other_dataset_3_gnn_gin_90.pth
# model_pretrained_on_other_soc_datasets: contrast_sim_based_vallina_shuffle_true_soc_set_soft_other_sim_other_dataset_3_gnn_gin_35.pth
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_2_num_path_40_dl_other_p_w2v_neg_p_no_automl_gnn_gin_sim_exp_40.pth
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_2_num_path_5_dl_other_p_w2v_neg_p_mask_times_5_mask_rate_0.1500_gnn_gin_10.pth
# --input_model_file /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pt_saved_model/supervised.pth \
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_1_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.7_q_0.6_rw_hops_8_num_path_10_dl_other_p_w2v_neg_p_mask_times_5_mask_rate_0.1500_gnn_gin_sim_exp.pth
# contrast_sim_based_multi_pos_v3_sample_stra_rwr_hop_pos_neg_on_big_graph_bat_5_num_neg_samples_7_T_4_sel_other_stra_last_one_hop_5_rstprob_0.2_p_0.6_q_1.4_rw_hops_16_num_path_7_dl_other_p_w2v_neg_p_mask_times_5_mask_rate_0.1500_5.pth