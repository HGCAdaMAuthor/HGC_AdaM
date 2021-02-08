### Hierarchical Graph Contrastive Pretraining with AdaptiveMasking

#### Datasets 

##### Pre-training datasets

For molecular graph pre-training dataset (`MolD`), please refer to [snap-pretrain](https://github.com/snap-stanford/pretrain-gnns) and proceed following their description for `chem data`. 

For `SSAD`, data will be obtained in pre-training process. Moreover, download the processed data  from [SSAD](https://drive.google.com/file/d/1HcARX612-71cU14xNhKb-jGoqenz7AI0/view?usp=sharing), and put `subgraphs_rwr_3.bin` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder, put `idx_to_candi_idx_list_dict_rwr_3.npy` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder, put `idx_to_sim_score_dict_3.npy` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder. 

For `BSGD`, download from [BSGD](https://drive.google.com/file/d/16Y3YfZYMVLl3MQMH8mmLCC9tUPMwjRzp/view?usp=sharing), unzip the file and put them under `./chem/dataset` folder. 

For `SGCD`, download from [SGCD](https://drive.google.com/file/d/1N7mE4BvdewAn7hX18oOerxrXM9tsRyS0/view?usp=sharing), unzip the file and put them under `./chem/dataset/ALL_GRA` folder. 

##### Fine-tuning datasets

For molecular graph fine-tuning datasets, please refer to [snap-pretrain](https://github.com/snap-stanford/pretrain-gnns) and proceed following their description for `chem data`. 

For social network node classification datasets, please download `downstream.tar.gz`	from  [GCC-node](https://drive.google.com/file/d/12kmPV3XjVufxbIVNx5BQr-CFM9SmaFvM/view) and put them under `./chem/dataset` folder after unzip the file. 

For social network graph classification datasets, we have save the `soc_graph.zip` file in the root path. Please unzip it and put the data under `./chem/dataset` folder. 

#### Pre-training 

For `SSAD` pre-training, refer to the following commands: 

```bash
cd ./graph_self_learn
bash scripts/pret.sh <gpu> --moco --nce-k 256 --dataset con_dgl
```

For `SGCD` pre-training, refer to the following commands: 

```bash
cd ./chem
python3 ./pretrain_contrast_sim_based_soc_datasets.py \
--device 0 --epochs 100 --num_samples 3 \
--gnn_type gin --batch_size 32 --dataset other_soc
```

For `BSGD` pre-training, refer to the following commands: 

```bash
cd ./chem
python3 ./pretrain_contrast_sim_based_soc_datasets.py \
--device 0 --epochs 100 --num_samples 3 \
--gnn_type gin --batch_size 32 --dataset cls_soc
```

For `MolD` pre-training, refer to the following commands: 

```bash
cd ./chem
# for AdaM 
python3 pretrain_masker_nll.py --device <gpu> --epochs 100 --mask_times <Mask times> --batch_size 256 --gnn_type <gnn_type> 
# for HGC (FO)
python3 pretrain_contrast_sim_based.py --device <gpu> --epochs 20 --gnn_type <gnn_type>
# for HGC (HO)
python3 pretrain_contrast_sim_based_with_neg_multi_pos_batch.py --device <gpu> --epochs 20 --neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat --num_samples <PS> --construct_big_graph --rw_hops <Walk length> --num_path <Walks> --num_com_negs <NS> 
# for HGC_AdaM (FO)
python3 pretrain_masker_nll_con_sim.py --device <gpu> --epochs 100 --mask_times <Mask times> --batch_size 256 --gnn_type <gnn_type> 
# for HGC_AdaM (HO)
python3 pretrain_contrast_sim_based_with_neg_multi_pos_batch_mask_nll.py \
--device <gpu> --epochs 20 \
--neg_sample_stra rwr_hop_pos_neg_on_big_graph_bat \
--num_samples <PS> \
--construct_big_graph \
--rw_hops <Walk length> --num_path <Walks> --num_com_negs <NS> \
--gnn_type <gnn_type> --mask_times <Mask times> \
--mask_rate 0.15
```

Please refer hyper-parameters to those listed in Appendix for each methods using different backbones. 

#### Fine-tuning 

##### Molecular graph datasets 

We save our pre-trained model on the molecular dataset in `./chem/pt_saved_model` folder.

For `gnn` in `[gin, gcn, graphsage]`, the correspondence between  `Model Name` listed in `param_{gnn}.md ` and `model_file_pth` in `./chem/pt_saved_model` folder is as follows: 

| Model Name   | model_file_pth          |
| ------------ | ----------------------- |
| AM           | `masking_am_{gnn}.pth`  |
| CON_HGC      | `con_hgc_{gnn}.pth`     |
| CON_HGC_RW   | `con_rw_{gnn}.pth`      |
| CON_HGCAM    | `con_mask_{gnn}.pth`    |
| CON_HGCAM_RW | `con_rw_mask_{gnn}.pth` |

To run the fine-tuning process for reproducing, please refer hyper-parameters for the corresponding model name listed in each `param_{gnn}.md` when using the saved model in `./chem/pt_saved_model`. 

The correspondence other pre-training strategies in the paper and `model_file_pth` in `./chem/pt_saved_model` folder is as follows: 

| Strategy     | model_file_pth     |
| ------------ | ------------------ |
| C_Subgraph   | `con_subgraph.pth` |
| GraphCL      | `graphcl_80.pth`   |
| Infomax      | `infomax.pth`      |
| Edge_Pred    | `edgepred.pth`     |
| Attr_Mask    | `masking.pth`      |
| Context_Pred | `contextpred.pth`  |

For hyper-parameters, we use setting provided by the authors for all the models listed above except `C_Subgraph` (since it is only a contrastive learning strategy and has not been tested before), whose hyper-parameters can be found in the file `./chem/param_gin.md`. 

Commands can refer to the follows: 

```bash
cd chem/
bash ./start_local/start.sh <dataset> <device> <model_file_pth> <lr> <dropout_ratio> <lr_scale> <batch_size> <gnn_type> 
```

##### Social network graph datasets --- Node classification 

To test `HGC (SSAD)` model in the paper on node classification datasets, please refer to the following commands: 

```bash
cd ./graph_self_learn
bash ./start_local/start.sh <dataset> <device>
```

where `dataset` in `[usa_airport, h-index]`. 

##### Social network graph datasets --- Graph classification 

The correspondance between dataset names listed **in the paper** and those should be used **in commands** is as follows: 

| In paper | In commands   |
| -------- | ------------- |
| IMDB-B   | `imdb-binary` |
| IMDB-M   | `imdb-multi`  |
| RDT-B    | `rdt-b`       |
| RDT-M    | `rdt-5k`      |

For models trained on the **molecular graph** pre-training dataset and **social classification datasets** (`SGCD` and `BSGD`), the correspondance between strategies listed in the paper and the `model_file_pth` in `./chem/pt_saved_model` is as follows: 

| Strategy       | model_file_pth               |
| -------------- | ---------------------------- |
| Context_Pred   | `contextpred.pth`            |
| S_Context_Pred | `supervised_contextpred.pth` |
| HGC (CSim)     | `con_hgc_gin.pth`            |
| AdaM           | `masking_am_gin.pth`         |
| HGCAdaM        | `con_mask_gin.pth`           |
| HGC (SGCD)     | `con_hgc_sgcd.pth`           |
| HGC (BSGD)     | `con_hgc_bsgd.pth`           |

Commands can refer to the follows: 

```bash
cd chem/
bash ./start_local/start.sh <dataset> <device> <model_file_pth> 0.001 0.5 1.0 32 gin 
```