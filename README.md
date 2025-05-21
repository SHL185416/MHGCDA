# Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification (MHGCDA)

![model](https://gitee.com/l18541900/picgo/raw/master/img/202403261027391.png)

This repository provides the Pytorch code for the work "Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification" published in Expert Systems with Applications, 2025.

#### Environment requirement

All experiments were conducted on a system equipped with dual RTX 3080 GPUs (20GB each), a 12-core Intel Xeon Platinum 8352V CPU @ 2.10GHz, and 48GB of RAM.

The code has been tested running under the required packages as follows:

- torch==1.13.0+cu116
- dgl==0.6.1
- numpy==1.24.4
- scipy==1.8.1
- scikit-learn==1.1.1

#### Dataset folder

The folder structure required

- input
  - citation
    - `citation1_acmv9.mat`
    - `citation1_citationv1.mat`
    - `citation1_dblpv7.mat`
    - `citation2_acmv8.mat`
    - `citation2_citationv1.mat`
    - `citation2_dblpv4.mat`

##### How to run

```shell
python main.py --PP_wei=1 --Clf_wei=1 --NN_wei=1 --P_wei=1 --attn_drop=0.7 --batch_size=6000 --data_key='citation' --epochs=100 --gpu=0 --in_drop=0.3 --l2_w=0.01 --lr_ini=0.01 --num_heads=16 --num_hidden=16 --num_layers=2 --num_out_heads=3 --target='citation1_citationv1' --tau_n_rate=0.2 --tau_p_rate=0.8
```

For more details of this multi-source domain adaptation approach, please refer to the following work:

@article{LIN2025127900,
title = {Hierarchical graph contrastive domain adaptation for multi-source cross-network node classification},
journal = {Expert Systems with Applications},
volume = {284},
pages = {127900},
year = {2025},
issn = {0957-4174},
author = {Chuanyun Lin and Xi Zhou and Xiao Shen}
}

If you have any questions regarding the code, please contact email [cylin@hainanu.edu.cn](mailto:cylin@hainanu.edu.cn).
