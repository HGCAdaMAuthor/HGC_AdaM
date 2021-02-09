### Hierarchical Graph Contrastive Pretraining with AdaptiveMasking

#### Requirements 

##### Pre-training stage

```
pytorch 1.4.0 
torch-geometric 1.4.3 
rdkit 2020.03.2
scikit-learn 0.22.1
python 3.6.10
torch-scatter 2.0.3
torch-sparse 0.5.1
dgl 0.5.2
```

##### Fine-tuning stage

```
pytorch 1.4.0
dgl 0.5.3 
scikit-learn 0.22.1
torch-geometric 1.6.1
torch-scatter 2.0.5
torch-cluster 1.5.7
torch-sparse 0.6.7
```



#### Time consumption comparison

##### Pre-training stage

We compare the time consumption between `AdaM` strategy using different masking times and `HGC` strategy with basic pre-training strategyes (`RdM` and `C_Subgraph`. The results are summarized as follows. Experiments are completed on a single P40 GPU, which is different from the pre-training hardward configuration used in our pre-training stage. 

|                          | RdM  | k=3   | k=5   | k=7   |
| ------------------------ | ---- | ----- | ----- | ----- |
| Time (min. / 7813 steps) | 0    | +12.5 | +19.4 | +25.0 |

|                          | C_Subgraph | HGC (HO) |
| ------------------------ | ---------- | -------- |
| Time (min. / 7813 steps) | 0          | +16.7    |

##### Pre-processing stage

Actually, the progressive time complexity for the pre-processing process is O((k + log(N)N), where k is the maximum number of candidates, N is the number of graph instances in the pre-training dataset. We then present some experimental values of the time consumption in each step of our pre-processing to demonstrate that although it do take noticeable amount of time, time spent on such process is tolerable if implemented carefully and also worthy concerning the benefit it brings. For each step in the molecular pre-training dataset (`MolD`, containing `2000000` graph instances) pre-processing, specific values of the spent time are listed as follows: (1) Calculate the molecular weight, numbers of rings and atoms in each molecule using the Python package RDKit (~1187.8 s ~ 20 mins). (2) Sort molecules based on the molecular weight using the build-in function \verb|sorted| in Python (~ 11.1 s). (3) Get candidates with the maximum number of candidates set to 70 based on the molecular weight, numbers of rings and atoms which have been stored in step (1) implemented with C++ (~230.0 s ~ 3.8 mins). (4) Calculated fingerprints for molecules using the Python package RDKit (~ 168 mins ~ 3 hrs). (5) Calculate fingerprint similarity scores for each molecule and its candidates using the Python package RDKit (~ 1257.6 s ~ 21 mins). The whole pre-processing process for `MolD` can be completed in less than 4 hours with no parallel calculation or complex optimization. 
As for social network graph pre-training datasets where graph instances are RWR induced subgraphs (i.e., `SSAD`) or small social graphs (i.e., `SGCD` and `BSGD`), the difference between whose pre-processing and the one for `MolD` is how to compute similarity scores between each graph instance and its candidates. We use Weisfeiler-Lehman Graph Kernel with 3 iterations implemented in the Python package GraKel to get normalized similarity scores. Since the progressive time complexity for fitting a graph kernel using the target graph and then use it to transform candidate graph instances is also O(kN), the time complexity for the whole pre-processing is then O((k + log(N)N), the same with pre-processing for `MolD`. However, we observe that such fit-and-transform process really takes time in practice. To speed up the process, we process the calculation in parallel on 20 CPUs, which takes less than 1 hour for`SGCD` which containing `156754` graph instances to complete. 

#### Datasets 

##### Pre-training datasets

For molecular graph pre-training dataset (`MolD`), please refer to [snap-pretrain](https://github.com/snap-stanford/pretrain-gnns) and proceed following their description for `chem data`. 

For `AC_PU_NF`, data will be obtained in pre-training process. Moreover, download the processed data  from [AC_PU_NF](https://drive.google.com/file/d/1HcARX612-71cU14xNhKb-jGoqenz7AI0/view?usp=sharing), and put `subgraphs_rwr_3.bin` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder, put `idx_to_candi_idx_list_dict_rwr_3.npy` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder, put `idx_to_sim_score_dict_3.npy` under `./graph_self_learn/data_bin/dgl/subgraphs2` folder. 

For `Soc_S_NF`, download from [Soc_S_NF](https://drive.google.com/file/d/16Y3YfZYMVLl3MQMH8mmLCC9tUPMwjRzp/view?usp=sharing), unzip the file and put them under `./chem/dataset` folder. 

For `Soc_L_NF`, download from [Soc_L_NF](https://drive.google.com/file/d/1N7mE4BvdewAn7hX18oOerxrXM9tsRyS0/view?usp=sharing), unzip the file and put them under `./chem/dataset/ALL_GRA` folder. 

For `MolD`, download from [MolD](https://drive.google.com/file/d/1iKygf_0FnMXTx6daZuv4W1lSMSaN8Rc5/view?usp=sharing), unzip the file and put them under `./chem/dataset/zinc_standard_agent/processed` folder. 

##### Fine-tuning datasets

For molecular graph fine-tuning datasets, please refer to [snap-pretrain](https://github.com/snap-stanford/pretrain-gnns) and proceed following their description for `chem data`. 

For social network node classification datasets, please download `downstream.tar.gz`	from  [GCC-node](https://drive.google.com/file/d/12kmPV3XjVufxbIVNx5BQr-CFM9SmaFvM/view) and put them under `./chem/dataset` folder after unzip the file. 

For social network graph classification datasets, we have save the `soc_graph.zip` file in the root path. Please unzip it and put the data under `./chem/dataset` folder. 

#### Pre-training 

For `AC_PU_NF` pre-training, refer to the following commands: 

```bash
cd ./graph_self_learn
bash scripts/pret.sh <gpu> --moco --nce-k 256 --dataset con_dgl
```

For `Soc_L_NF` pre-training, refer to the following commands: 

```bash
cd ./chem
python3 ./pretrain_contrast_sim_based_soc_datasets.py \
--device 0 --epochs 100 --num_samples 3 \
--gnn_type gin --batch_size 32 --dataset other_soc
```

For `Soc_S_NF` pre-training, refer to the following commands: 

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

To test `HGC (AC_PU_NF)` model in the paper on node classification datasets, please refer to the following commands: 

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

For models trained on the **molecular graph** pre-training dataset and **social classification datasets** (`Soc_L_NF` and `Soc_S_NF`), the correspondance between strategies listed in the paper and the `model_file_pth` in `./chem/pt_saved_model` is as follows: 

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