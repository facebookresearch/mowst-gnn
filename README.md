# Mowst: Mixture of Weak and Strong Experts on Graphs

[Hanqing Zeng](https://sites.google.com/a/usc.edu/zengh/home)\*, [Hanjia Lyu](https://brucelyu17.github.io/)\*, [Diyi Hu](https://sites.google.com/a/usc.edu/diyi_hu/), [Yinglong Xia](https://sites.google.com/site/yinglongxia/home), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/)

Contact: Hanqing Zeng (zengh@meta.com), Hanjia Lyu (hlyu5@ur.rochester.edu)

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
