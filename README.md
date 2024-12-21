# Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification (MHGCDA)

![model](https://gitee.com/l18541900/picgo/raw/master/img/202403261027391.png)

This repository provides the Pytorch code for the work "Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification" published in XXX, 202X.



Our work investigates a more realistic CNNC problem called Multi-Source Cross-Network Node Classification (MSCNNC), aiming to accurately classify nodes in a target network by leveraging the complementary knowledge from multiple source networks. To address the MSCNNC problem, we propose a novel Multi-source Hierarchical Graph Contrastive Domain Adaptation (MHGCDA) model. Firstly, MHGCDA designs a transferability weight learning module to measure the fitness between each source network and the target network based on information entropy, thus controlling the influence of each source network knowledge on the target network. Secondly, MHGCDA conducts hierarchical graph contrastive domain adaptation to alleviate intra-class domain divergence while expanding inter-class domain discrepancy at both node and prototype levels. Lastly, MHGCDA employs a novel pseudo-labeling strategy to assign positive pseudo-labels to target nodes highly likely to belong to a specific class and negative pseudo-labels to those highly unlikely to belong to a specific class. By taking such target nodes with positive or negative pseudo-labels to iteratively re-train the model in a self-training manner, more accurate pseudo-labels can be obtained to assist in class-aware domain alignment between source and target networks.



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

@article{XXX,
title = {Hierarchical Graph Contrastive Domain Adaptation for Multi-source Cross-network Node Classification},
journal = {},
volume = {},
pages = {},
year = {202X},
url = {https://www.},
author = {..}
}

If you have any questions regarding the code, please contact email [cylin@hainanu.edu.cn](mailto:cylin@hainanu.edu.cn).
