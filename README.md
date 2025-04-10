# Mowst: Mixture of Weak and Strong Experts on Graphs

[Hanqing Zeng](https://hanqingzeng.com)\* (zengh@meta.com), [Hanjia Lyu](https://brucelyu17.github.io/)\* (hlyu5@ur.rochester.edu), [Diyi Hu](https://sites.google.com/a/usc.edu/diyi_hu/), [Yinglong Xia](https://sites.google.com/site/yinglongxia/home), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

*: equal contribution

[Paper](https://openreview.net/forum?id=wYvuY60SdD)

### Example commands

1. run vanilla GCN on flickr, one run
```
python main.py --dataset flickr --method baseline --model2 GCN
```

2. run Mowst*-GCN on ogbn-products, one run, and the input features for the gating module only contain dispersion
```
python main.py --dataset product --method mowst_star --model2 GCN --original_data false
```

3. run Mowst-Sage on pokec, one run, and the input features for the gating module contain dispersion and the node self-features
```
python main.py --dataset pokec --method mowst --model2 Sage --original_data true
```

4. run Mowst-Sage on penn94 (grid search, 10 runs), and the input features for the gating module contain dispersion and the node self-features
```
python main.py --dataset penn94 --method mowst --model2 Sage --original_data true --setting ten
```

### To Reproduce Results
#### Mowst(*)-GCN
dataset=Flickr
```
python main.py --dataset flickr --method mowst --submethod pretrain_model2 --subloss separate --infer_method multi --original_data false --model2 GCN --model1_hidden_dim 128 --model2_hidden_dim 128 --model1_num_layers 3 --model2_num_layers 3 --lr 0.1 --lr_gate 0.01 --weight_decay 0.0005 --dropout 0.3 --dropout_gate 0.5
```
dataset=ogbn-products
```
python main.py --dataset product --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --original_data true --model2 GCN --model1_hidden_dim 256 --model2_hidden_dim 256 --model1_num_layers 3 --model2_num_layers 3 --lr 0.01 --lr_gate 0.01 --weight_decay 0 --dropout 0.5 --dropout_gate 0.5
```
dataset=ogbn-arxiv
```
python main.py --dataset arxiv --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --crit crossentropy --original_data true --model2 GCN --model1_hidden_dim 256 --model2_hidden_dim 256 --model1_num_layers 3 --model2_num_layers 3 --lr 0.001 --lr_gate 0.001 --weight_decay 0 --dropout 0.2 --dropout_gate 0.5
```
dataset=Penn94
```
python main.py --dataset penn94 --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --original_data false --model2 GCN --model1_hidden_dim 64 --model2_hidden_dim 64 --model1_num_layers 2 --model2_num_layers 2 --lr 0.1 --lr_gate 0.01 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.5
```
dataset=pokec
```
python main.py --dataset pokec --method mowst_star --submethod pretrain_model1 --subloss joint --infer_method joint --no_cached --crit crossentropy --original_data false --model2 GCN --model1_hidden_dim 64 --model2_hidden_dim 64 --model1_num_layers 2 --model2_num_layers 2 --lr 0.1 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.2 --dropout_gate 0.3
```
dataset=twitch-gamer
```
python main.py --dataset twitch-gamer --method mowst --submethod none --subloss separate --infer_method multi --no_cached --crit nllloss --original_data false --model2 GCN --model1_hidden_dim 64 --model2_hidden_dim 64 --model1_num_layers 2 --model2_num_layers 2 --lr 0.1 --lr_gate 0.1 --weight_decay 0.001 --dropout 0.2 --dropout_gate 0.1
```
#### Mowst(*)-GIN
dataset=Flickr
```
python main.py --dataset flickr --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --model2 GIN --crit crossentropy --original_data false --model1_hidden_dim 256 --model2_hidden_dim 256 --model1_num_layers 3 --model2_num_layers 3 --early_signal val_acc --lr 0.001 --lr_gate 0.001 --weight_decay 0.0005 --dropout 0.1 --dropout_gate 0.5
```
dataset=ogbn-arxiv
```
python main.py --dataset arxiv --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --crit crossentropy --model2 GIN --original_data false --model1_hidden_dim 256 --model2_hidden_dim 256 --model1_num_layers 3 --model2_num_layers 3 --early_signal val_acc --lr 0.001 --lr_gate 0.001 --weight_decay 0 --dropout 0.1 --dropout_gate 0.5
```
dataset=Penn94
```
python main.py --dataset penn94 --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --model2 GIN --original_data false --model1_hidden_dim 12 --model2_hidden_dim 12 --model1_num_layers 2 --model2_num_layers 2 --early_signal val_acc --lr 0.001 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.5 --dropout_gate 0.1
```
dataset=pokec
```
python main.py --dataset pokec --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --model2 GIN --original_data false -- model1_hidden_dim 4 --model2_hidden_dim 4 --model1_num_layers 2 --model2_num_layers 2 --early_signal val_acc --lr 0.01 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.1
```
dataset=twitch-gamer
```
python main.py --dataset twitch-gamer --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --model2 GIN --original_data false --model1_hidden_dim 12 --model2_hidden_dim 12 --model1_num_layers 2 --model2_num_layers 2 --early_signal val_acc --lr 0.01 --lr_gate 0.01 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.5
```
#### Mowst(*)-GIN-Skip
dataset=pokec
```
python main.py --dataset pokec --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --model2 gin_mlp_res --original_data false --model1_hidden_dim 64 --model2_hidden_dim 64 --model1_num_layers 2 --model2_num_layers 2 --early_signal val_loss --lr 0.01 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.5
```
dataset=twitch-gamer
```
python main.py --dataset twitch-gamer --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --model2 gin_mlp_res --original_data false --model1_hidden_dim 64 --model2_hidden_dim 64 --model1_num_layers 2 --model2_num_layers 2 --early_signal val_loss --lr 0.01 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.1
```
#### Mowst(*)-SAGE
dataset=Flickr
```
python main.py --dataset flickr --method mowst_star --submethod pretrain_both --subloss joint --infer_method joint --original_data true --model2 Sage --crit crossentropy model1_hidden_dim 12 --model2_hidden_dim 12 --model1_num_layers 3 --model2_num_layers 3 --lr 0.01 --lr_gate 0.01 --weight_decay 0.0005 --dropout 0.5 --dropout_gate 0.5
```
dataset=ogbn-products
```
python main.py --dataset product --method mowst_star --submethod pretrain_model1 --subloss joint --infer_method joint --crit crossentropy --original_data true --model2 Sage --model1_hidden_dim 256 --model2_hidden_dim 256 --model1_num_layers 3 --model2_num_layers 3 --early_signal val_loss --lr 0.01 --lr_gate 0.01 --weight_decay 0 --dropout 0.1 --dropout_gate 0.5
```
dataset=ogbn-arxiv
```
python main.py --dataset arxiv --method mowst --submethod pretrain_model2 --subloss separate --infer_method multi --crit nllloss --original_data false --model2 Sage --model1_hidden_dim 512 --model2_hidden_dim 512 --model1_num_layers 4 --model2_num_layers 4 --early_signal val_acc --lr 0.01 --lr_gate 0.01 --weight_decay 0 --dropout 0.3 --dropout_gate 0.5
```
dataset=Penn94
```
python main.py --dataset penn94 --method mowst_star --submethod pretrain_model2 --subloss joint --infer_method joint --no_cached --crit crossentropy --original_data true --model2 Sage --model1_hidden_dim 32 --model2_hidden_dim 32 --model1_num_layers 2 --model2_num_layers 2 --lr 0.01 --lr_gate 0.001 --weight_decay 0.001 --dropout 0.5 --dropout_gate 0.5
```
dataset=pokec
```
python main.py --dataset pokec --method mowst --submethod pretrain_model2 --subloss separate --infer_method multi --no_cached --crit nllloss --original_data true --model2 Sage --model1_hidden_dim 12 --model2_hidden_dim 12 --model1_num_layers 2 --model2_num_layers 2 --lr 0.1 --lr_gate 0.1 --weight_decay 0.001 --dropout 0.1 --dropout_gate 0.1
```
dataset=twitch-gamer
```
python main.py --dataset twitch-gamer --method mowst --submethod pretrain_model2 --subloss separate --infer_method multi --no_cached --crit nllloss --original_data false --model2 Sage --model1_hidden_dim 32 --model2_hidden_dim 32 --model1_num_layers 2 --model2_num_layers 2 --lr 0.01 --lr_gate 0.01 --weight_decay 0.001 --dropout 0.3 --dropout_gate 0.1
```

## License
Mowst is MIT licensed, as found in the LICENSE file.

## Citation
```
@inproceedings{mowstgnn-iclr24,
title={Mixture of Weak and Strong Experts on Graphs},
author={Hanqing Zeng and Hanjia Lyu and Diyi Hu and Yinglong Xia and Jiebo Luo},
booktitle={International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=wYvuY60SdD}
}
```
