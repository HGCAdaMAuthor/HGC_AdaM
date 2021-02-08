sett=$1
device=$2

set=${sett}
dev=${device}

python3 ./train_graph_moco.py \
--exp Finetune \
--model-path ./saved \
--tb-path ./tensorboard \
--gpu ${dev} \
--moco \
--nce-k 256 \
--rw_hops 16 \
--num_path 7 \
--dataset ${set} \
--batch-size 32 \
--finetune \
--resume ./saved/con_dgl/ckpt_epoch_100.pth