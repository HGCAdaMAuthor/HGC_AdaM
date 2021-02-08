#python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_contrast_sim_based_with_neg_multi_pos_batch.py --device 0 --num_neg_samples 7 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples 5 --T 1 --select_other_node_stra last_one --restart_prob 0.2 --num_hops 2 --p 0.7 --q 0.6 --construct_big_graph --rw_hops 16 --num_path 7 --num_com_negs 49 --env jizhi # --val_dataset bbbp # --data_para
log_dir=/apdcephfs/private_meowliu/$TASK_FLAG/$METRICS_TRIAL_NAME
mkdir -p $log_dir

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/finetune_automl.py \
--device 0 --epochs 100 --batch_size 32 --gnn_type gin --env jizhi --dataset toxcast \
--input_model_file /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pt_saved_model/contrast_subgraph_20.pth  \
--no_val \
> $log_dir/out.log 2>&1
# --val_dataset bbbp # --data_para