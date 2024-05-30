import os
import wandb
from model import (MLP, 
                   GCN, 
                   GAT, 
                   Sage, 
                   GIN,
                   MLPLearn)
import torch
import numpy as np
import random
import torch_geometric.transforms as T
from hetero_data_utils import load_hetero_data


# initialize the experiment (e.g., creating directories to save results)
def initialize_experiment(args):
    # to create the path to store the wandb results
    if not os.path.exists(args.wandb_save_path):
        os.makedirs(args.wandb_save_path)

    # to create the path to save models
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    # to create the path to store experimental results
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)

    # to use the same network structure as reported in paper
    if args.preset_struc is True:
        args = get_preset_struc(args)

    # to initialize wandb
    if args.log:
        wandb.init(project='mowst', config=args, dir=args.wandb_save_path)

    if args.method == "mowst":
        args.subloss = "separate"
        args.infer_method = "multi"
    elif args.method == "mowst_star":
        args.subloss = "joint"
        args.infer_method = "joint"

    # verify the combination of subloss and infer_method
    if args.subloss == 'joint':
        if args.infer_method != 'joint':
            raise ValueError(
                'Invalid Args Combination. If subloss is joint then infer_method has to be joint as well.')

    if args.dataset in ["penn94", "twitch-gamer", "pokec"]:
        args.setting = 'hetero_' + args.setting

    return args


# use the same network structure as reported in paper
def get_preset_struc(args):
    dataset = args.dataset
    if dataset == 'flickr':
        args.model1_hidden_dim = 256
        args.model2_hidden_dim = 256
        args.model1_num_layers = 3
        args.model2_num_layers = 3
        if args.model2 == "GAT":  # this is the best struture with which GAT achieves the optimal result on flickr
            args.model1_num_layers = 2
            args.model2_num_layers = 2
            args.model1_hidden_dim = 12
            args.model2_hidden_dim = 12
    # this is the structure as reported in the OGB paper
    elif dataset in ['arxiv', "product"]:
        args.model1_hidden_dim = 256
        args.model2_hidden_dim = 256
        args.model1_num_layers = 3
        args.model2_num_layers = 3
        args.weight_decay = 0
        # this is the best structure with which GAT achieves the optimal result on arxiv by searching from: num_layers == 2, hidden_dim \in {4, 8, 12, 32} (same grid as in LINKX)
        if dataset == "arxiv" and args.model2 == "GAT":
            args.model1_num_layers = 2
            args.model2_num_layers = 2
            args.model1_hidden_dim = 32
            args.model2_hidden_dim = 32
        # this is the same structure as reported in the OGB paper
        elif dataset == "product" and args.model2 == "GCN":
            args.no_batch_norm = True
            args.no_cached = True
        elif dataset == "product" and args.model2 == "GAT":  # OOM
            args.model1_num_layers = 2
            args.model2_num_layers = 2
        # this is the same structure as reported in the OGB paper
        elif dataset == "product" and args.model2 == "Sage":
            args.no_batch_norm = True
    elif dataset in ['penn94', 'twitch-gamer', 'pokec']:
        args.no_cached = True  # same as reported in the LINKX paper
        args.model1_hidden_dim = 64
        args.model2_hidden_dim = 64
        args.weight_decay = 0.001  # same setting as reported in the LINKX paper
        if dataset == 'penn94':
            # this is the best structure with which GraphSage achieves the optimal result on penn94 by searching from: num_layers == 2, hidden_dim \in {4, 8, 12, 32} (same grid as in LINKX)
            if args.model2 == "Sage":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32
        elif dataset == 'pokec':
            # this is the best structure with which GAT and GraphSage achieve the optimal result on pokec by searching from: num_layers == 2, hidden_dim \in {4, 8, 12} (same grid as in LINKX)
            if args.model2 in ["GAT", "Sage"]:
                args.model1_hidden_dim = 12
                args.model2_hidden_dim = 12
        elif dataset == 'twitch-gamer':
            # this is the best structure with which GAT achieves the optimal result on arxiv by searching from: num_layers == 2, hidden_dim \in {4, 8, 12, 32} (same grid as in LINKX)
            if args.model2 == "GAT":
                args.model1_hidden_dim = 8
                args.model2_hidden_dim = 8
            # this is the best structure with which GraphSage achieves the optimal result on arxiv by searching from: num_layers == 2, hidden_dim \in {4, 8, 12, 32} (same grid as in LINKX)
            elif args.model2 == "Sage":
                args.model1_hidden_dim = 32
                args.model2_hidden_dim = 32

    return args

# load data for experiment,


def load_data(args):
    if args.dataset == 'flickr':
        from torch_geometric.datasets.flickr import Flickr
        dataset = Flickr(root='/tmp/Flickr', transform=T.ToSparseTensor())
        data = dataset[0]
        data.y = data.y.view(-1, 1)
        data.mlp_x = data.x
        split_idx = mask2idx(data)
    elif args.dataset == 'arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)        
        data = T.ToSparseTensor()(data)
        data.mlp_x = data.x
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = process_ogb(data, train_idx, valid_idx, test_idx)
    elif args.dataset == 'product':
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected
        
        dataset = PygNodePropPredDataset(name='ogbn-products',
                                             transform=T.ToSparseTensor())
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data.mlp_x = data.x
        data = process_ogb(data, train_idx, valid_idx, test_idx)
    elif args.dataset in ['penn94', 'twitch-gamer', 'pokec']:
        dataset, data, split_idx = load_hetero_data(args)
        data.mlp_x = data.x
    else:
        raise ValueError('Invalid dataset name')
    return dataset, data, split_idx


def process_ogb(data, train_idx, valid_idx, test_idx):
    n = data.num_nodes
    train_mask = create_mask(n, train_idx)
    val_mask = create_mask(n, valid_idx)
    test_mask = create_mask(n, test_idx)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def create_mask(n, pos):
    res = torch.zeros(n, dtype=torch.bool)
    res[pos] = True
    return res


def mask2idx(data):
    split_idx = {'train': (data.train_mask == True).nonzero(as_tuple=True)[0],
                 'valid': (data.val_mask == True).nonzero(as_tuple=True)[0],
                 'test': (data.test_mask == True).nonzero(as_tuple=True)[0]}
    return split_idx


def get_model_hyperparameters(args, data, dataset):
    input_dim = data.num_features
    if not args.setting in ['hetero_exp', 'hetero_ten']:
        output_dim = dataset.num_classes
    else:
        output_dim = data.num_classes

    if args.method == 'baseline':
        hidden_dim = args.model2_hidden_dim
        num_layers = args.model2_num_layers
        return input_dim, hidden_dim, output_dim, num_layers
    elif args.method in ['mowst_star', 'mowst']:
        hidden_dim2 = args.model2_hidden_dim
        num_layers2 = args.model2_num_layers
        hidden_dim1 = args.model1_hidden_dim
        num_layers1 = args.model1_num_layers
        return [input_dim, hidden_dim1, output_dim, num_layers1], [input_dim, hidden_dim2, output_dim, num_layers2]


def load_model(model_name, input_dim, hidden_dim, output_dim, num_layers, dropout, args):
    if model_name == 'GCN':
        return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'GAT':
        return GAT(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'Sage':
        return Sage(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'GIN':
        return GIN(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'MLP':
        return MLP(input_dim, hidden_dim, output_dim, num_layers, dropout, args)
    elif model_name == 'MLPLearn':
        return MLPLearn(input_dim, hidden_dim, output_dim, num_layers, dropout, args)


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(args, model, label):
    s = args.model_save_path
    s += args.setting + '_'
    s += args.method + '_'
    s += args.submethod + '_'
    s += args.subloss + '_'
    s += args.infer_method + '_'
    s += args.dataset + '_'
    s += args.model1 + "_"
    s += str(args.model1_hidden_dim) + "_"
    s += str(args.model1_num_layers) + "_"
    s += args.model2 + "_"
    s += str(args.model2_hidden_dim) + "_"
    s += str(args.model2_num_layers) + "_"
    s += str(args.heads) + "_"
    s += str(args.lr) + '_'
    s += str(args.dropout) + '_'
    s += str(args.lr_gate) + '_'
    s += str(args.dropout_gate) + '_'
    s += str(args.weight_decay) + '_'
    s += str(args.no_batch_norm) + '_'
    s += str(args.no_cached) + '_'
    s += str(args.no_add_self_loops) + '_'
    s += str(args.no_normalize) + '_'
    s += args.original_data + '_'
    s += args.early_signal + '_'
    s += label + '.pt'
    torch.save(model.state_dict(), s)
    return s


def save_moe_model(args, model, model_id):
    s = args.model_save_path
    s += args.setting + '_'
    s += args.method + '_'
    s += args.submethod + '_'
    s += args.subloss + '_'
    s += args.infer_method + '_'
    s += args.dataset + '_'
    s += args.model1 + "_"
    s += str(args.model1_hidden_dim) + "_"
    s += str(args.model1_num_layers) + "_"
    s += args.model2 + "_"
    s += str(args.model2_hidden_dim) + "_"
    s += str(args.model2_num_layers) + "_"
    s += str(args.heads) + "_"
    s += str(args.lr) + '_'
    s += str(args.dropout) + '_'
    s += str(args.lr_gate) + '_'
    s += str(args.dropout_gate) + '_'
    s += str(args.weight_decay) + '_'
    s += str(args.no_batch_norm) + '_'
    s += str(args.no_cached) + '_'
    s += str(args.no_add_self_loops) + '_'
    s += str(args.no_normalize) + '_'
    s += args.original_data + '_'
    s += args.early_signal + '_'
    s += f'_{model_id}.pt'
    torch.save(model.state_dict(), s)
    return s


def append_results(RES, res):
    train_list, valid_list, test_list, weight_list, val_loss_list = RES
    train_score, valid_score, test_score, weight_score, val_loss = res
    train_list.append(train_score)
    valid_list.append(valid_score)
    test_list.append(test_score)
    weight_list.append(weight_score)
    val_loss_list.append(-val_loss)


def create_result_dict(RES, args):
    train_list, valid_list, test_list, weight_list, val_loss_list = RES
    res = {}
    res["train score (avg)"] = np.mean(train_list)
    res["train score (std)"] = np.std(train_list)
    res["valid score (avg)"] = np.mean(valid_list)
    res["valid score (std)"] = np.std(valid_list)
    res["test score (avg)"] = np.mean(test_list)
    res["test score (std)"] = np.std(test_list)
    res["weight (avg)"] = np.mean(weight_list)
    res["weight (std)"] = np.std(weight_list)
    res["valloss (avg)"] = np.mean(val_loss_list)
    res["valloss (std)"] = np.std(val_loss_list)
    res["train score (all)"] = train_list
    res["train score (max)"] = np.max(train_list)
    res["train score (min)"] = np.min(train_list)
    res["valid score (all)"] = train_list
    res["valid score (max)"] = np.max(valid_list)
    res["valid score (min)"] = np.min(valid_list)
    res["test score (all)"] = train_list
    res["test score (max)"] = np.max(test_list)
    res["test score (min)"] = np.min(test_list)
    res["valloss (all)"] = val_loss_list
    res["vallosse (max)"] = np.max(val_loss_list)
    res["valloss (min)"] = np.min(val_loss_list)
    res["weight (all)"] = weight_list

    res["dataset"] = args.dataset
    res["setting"] = args.setting
    res["method"] = args.method
    res["submethod"] = args.submethod
    res["subloss"] = args.subloss
    res["infer_method"] = args.infer_method

    res["model1"] = args.model1
    res["model2"] = args.model2
    res["model1_hidden_dim"] = args.model1_hidden_dim
    res["model2_hidden_dim"] = args.model2_hidden_dim
    res["model1_num_layers"] = args.model1_num_layers
    res["model2_num_layers"] = args.model2_num_layers
    res["original_data"] = args.original_data
    res["hyper_hidden"] = args.hyper_hidden
    res["hyper_num_layers"] = args.hyper_num_layers

    res["no_cached"] = args.no_cached
    res["no_add_self_loops"] = args.no_add_self_loops
    res["no_normalize"] = args.no_normalize
    res["no_batch_norm"] = args.no_batch_norm

    res["lr"] = args.lr
    res["lr_gate"] = args.lr_gate
    res["weight_decay"] = args.weight_decay
    res["dropout"] = args.dropout
    res["dropout_gate"] = args.dropout_gate
    res["crit"] = args.crit
    res["adam"] = args.adam
    res["activation"] = args.activation
    res["train_prop"] = args.train_prop
    res["valid_prop"] = args.valid_prop
    res["preset_struc"] = args.preset_struc
    res["seed"] = args.seed
    res["epoch"] = args.epoch
    res["big_epoch"] = args.big_epoch
    res["patience"] = args.patience
    res["big_patience"] = args.big_patience
    res["m_times"] = args.m_times
    res["early_signal"] = args.early_signal
    res["cpu"] = args.cpu
    res["gpu"] = args.gpu
    res["print_freq"] = args.print_freq
    res["run"] = args.run
    res["log"] = args.log
    res["rand_split"] = args.rand_split
    res["split_run_idx"] = args.split_run_idx
    

    res["heads"] = args.heads  
    res["model_save_path"] = args.model_save_path
    res["wandb_save_path"] = args.wandb_save_path
    res["result_save_path"] = args.result_save_path

    return res