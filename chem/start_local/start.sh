#mkdir -p $log_dir
sett=$1
device=$2
mf=$3
lr=$4
drr=$5
lrs=$6
bs=$7
gnnt=$8


for dataset in hiv hiv hiv # bace bace bace clintox clintox clintox bbbp bbbp bbbp sider sider sider hiv hiv hiv tox21 tox21 tox21 toxcast toxcast toxcast
do

set=${sett}
dev=${device}
model_file=${mf}
learning_rate=${lr}
dr_rate=${drr}
lr_scale=${lrs}
batch_size=${bs}
gnn_ty=${gnnt}

python3 ./finetune_automl.py \
--device ${dev} \
--epochs 100 \
--batch_size ${batch_size} \
--gnn_type ${gnn_ty} \
--dataset ${set} \
--input_model_file ./pt_saved_model/${model_file} \
--lr ${learning_rate} \
--dropout_ratio ${dr_rate} \
--no_automl \
--no_val \
--lr_scale ${lr_scale}

done