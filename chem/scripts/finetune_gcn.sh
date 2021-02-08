##!/usr/bin/env bash

#for train_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
##do
##trat=${train_ratio}
##python finetune.py --input_model_file/contextpred.pth --device 1 --dataset bbbp --train_ratio ${trat} --output_file contextpred
##done  temp/masked_strategies_variances_25.pth
# masked_strategies_variances_2_55  dataset in bbbp bace clintox tox21 sider
#

python finetune.py --input_model_file temp/masker_entropy_based_gnn_gcn_mask_times_v2_5_10.pth --device 5 --test_ratio 0.1 --train_ratio 0.8 --gnn_type gcn --dataset bace
sleep 1
python finetune.py --input_model_file temp/masker_entropy_based_gnn_gcn_mask_times_v2_5_10.pth --device 5 --test_ratio 0.1 --train_ratio 0.8 --gnn_type gcn --dataset bace
sleep 1
python finetune.py --input_model_file temp/masker_entropy_based_gnn_gcn_mask_times_v2_5_10.pth --device 5 --test_ratio 0.1 --train_ratio 0.8 --gnn_type gcn --dataset bace
