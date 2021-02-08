log_dir=/apdcephfs/private_meowliu/$TASK_FLAG/$METRICS_TRIAL_NAME
mkdir -p $log_dir

python3 /apdcephfs/private_meowliu/ft_local/gnn_pretraining/pretrain_masker_nll_with_eval.py \
--device 0 --epochs 100 --mask_times 5 --batch_size 256 --gnn_type gin --env jizhi --val_dataset hiv \
> $log_dir/out.log 2>&1