#! /usr/bin/env bash

log_dir=/apdcephfs/private_meowliu/ft_local/mnist_automl/logs/$TASK_FLAG/$METRICS_TRIAL_NAME
mkdir -p $log_dir

python3.6 -u fully_connected_feed.py --train_dir /apdcephfs/private_meowliu/ft_local/mnist_automl/mnist_dataset > $log_dir/out.log 2>&1
