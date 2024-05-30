from utils import (initialize_experiment, 
                   load_data,
                   append_results,
                   create_result_dict)
import torch
from train_test_engine import get_engine
import os
import numpy as np
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mixture of Weak and Strong Experts (Mowst)")
    parser.add_argument("--dataset", type=str, default="flickr",
                        choices=["flickr", "product", "arxiv", "penn94", "pokec", "twitch-gamer"])
        # homophily: flickr, product, arxiv
        # heterophily: penn94, pokec, twitch-gamer
    parser.add_argument("--setting", type=str, default="exp", choices=["exp", "ten"], help='decide experiment settings')
        # exp: single run
        # ten: ten run for homophily graphs, five run for heterophily graphs; including grid searh
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "mowst_star", "mowst"], help="choose methods")
        # baseline: MLP, GCN, GAT, GIN, GraphSage, use --model2 to specify the exact baselien
        # mowst_star: Mowst-*
        # mowst: Mowst
    parser.add_argument('--submethod', type=str, default='none', choices=["none", "pretrain_model1", "pretrain_model2", "pretrain_both"])
        # choose the pretraining strategy
    parser.add_argument("--subloss", type=str, default="separate", choices=["joint", "separate"])
        # choose how to compute the loss
        # joint for Mowst-*
        # separate for Mowst
        # note that we can also choose the seperate loss for Mowst-*
    parser.add_argument("--infer_method", type=str, default="joint", choices=["joint", "multi"])
        # choose how to make the inference
        # joint for Mowst-*
        # multi for Mowst
    parser.add_argument('--model1', type=str, default='MLP', choices=["MLP", "GCN", "Sage"],
                        help="choose the model arch for model 1")
        # for the comparison between the weak-strong and strong-strong settings, we can set model 1 the same as model 2
    parser.add_argument('--model2', type=str, default='GCN', choices=["MLP","GCN","GAT","GIN"
                         "Sage"], 
                         help="choose the model arch for model 2")
        # We call it model 2 because it is considered the "strong" expert of Mowst.
        # To run experiments for other baselines, we can specify it using this argument.
    parser.add_argument('--model1_hidden_dim', type=int, default=64, help="set the hidden dim for model 1")
    parser.add_argument('--model2_hidden_dim', type=int, default=64, help="set the hidden dim for model 2")
    parser.add_argument('--model1_num_layers', type=int, default=2, help="set the number of layers for model 1")
    parser.add_argument('--model2_num_layers', type=int, default=2, help="set the number of layers for model 2")
    parser.add_argument('--original_data', type=str, choices=["true","false","hypermlp","hypergcn","none"], default="none")
        # choose the way to incorporate the node self features
        #   true: we concatenate the node self features to the dispersion to form the input for the gating module
        #   false: we do NOT use any node self features
        #   hypermlp: use the hypernet arch to compute the parameters for the gating module. The input to hypernet is node self features
    parser.add_argument("--hyper_hidden", default=64, type=int, help="set the hidden dim for the hypernet")
    parser.add_argument("--hyper_num_layers", default=2, type=int, help="set the number of layers for the hypernet")
    parser.add_argument('--no_cached', default=False, action='store_true')
        # specifying this argument will set cached to False in the GCN layer
    parser.add_argument('--no_add_self_loops', default=False, action='store_true')
        # specifying this argument will set add_self_loops to False in the GCN layer
    parser.add_argument('--no_normalize', default=False, action='store_true')
        # specifying this argument will set normalize to False in the GCN layer
    parser.add_argument('--no_batch_norm', default=False, action='store_true')
        # specifying this argument will disable batch norm

    # # # # # # training settings
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="set the learning rate for model 1 and model 2 during pretraining")
    parser.add_argument('--lr_gate', type=float, default=0.001,
                        help="set the learning rate for model 1, model 2, and the gating module for training Mowst and Mowst-*")
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="set the dropout ratio for model 1 and model 2 during pretraining")
    parser.add_argument('--dropout_gate', type=float, default=0.5,
                        help="set the dropout ratio for model 1, model 2, and the gating module for training Mowst and Mowst-*")
    parser.add_argument('--crit', type=str, default='nllloss', choices=['nllloss', 'bceloss', 'crossentropy'],
                        help="choose the loss function")
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--activation', type=str, default='relu', help="activation function")
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--preset_struc', default=False, action='store_true')
        # specify this argument to use the same network structure (i.e., number of layers, hidden dim) as reported in paper
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epoch', type=int, default=20000)
        # number of epochs for (1) baselines, and (2) the training turn of weak and strong expert, respectively
    parser.add_argument('--big_epoch', type=int, default=20000)
        # number of epochs of the training session which is composed of a weak expert training turn and a strong expert training turn
    parser.add_argument('--patience', type=int, default=100)
        # window size for the early stop for (1) baselines, and (2) the training turn of weak and strong expert, respectively
    parser.add_argument('--big_patience', type=int, default=10)
        # window size for the early stop for the training session which is composed of a weak expert training turn and a strong expert training turn
    parser.add_argument('--m_times', type=int, default=50)
        # in Algorithm 1, we randomly generate a number q \in [0,1]. To ensure the robustness of this process, we run this process m times.
    parser.add_argument("--early_signal", type=str, default="val_loss", choices=["val_loss", "val_acc"])
        # For the early stop mechanism, we can choose different metrics to decide whether or not to stop, 
        # either the validation loss or validation accuracy
    parser.add_argument('--cpu', default=False, action="store_true")
        # specifying this argument will use cpu for computation
    parser.add_argument('--gpu', default=0, type=int)
        # By default, we use gpu for computation. This argument can allow us to choose cuda id
    parser.add_argument('--print_freq', type=int, default=100)
        # The frequency the results is printed 
    parser.add_argument('--run', type=int, default=10)
        # number of runs, combined to use when setting == exp
    parser.add_argument('--log', default=False, action='store_true')
        # specifying this argument will log result into wandb
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
        # specifying this argument will use random splits for the experiments on heterophily graphs
    parser.add_argument('--split_run_idx', type=int, default=0)
        # choose the split id for the experiments on heterophily graphs
    
    # # # # # # setting hyperparameters for GAT
    parser.add_argument('--heads', type=int, default=1, help="set the number of heads") 
        # number of heads for GAT

    parser.add_argument('--model_save_path', type=str, default='saved_model/')
    parser.add_argument('--wandb_save_path', type=str, default='wandb/')
    parser.add_argument("--result_save_path", type=str, default="saved_result/")
    args = parser.parse_args()

    # initialize the experiment
    args = initialize_experiment(args)


    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                                    and not args.cpu else "cpu")
    print('Using {}'.format(device))


    if args.setting == 'exp' or args.setting == 'hetero_exp':
        print('Experiment')
        dataset, data, split_idx = load_data(args)
        engine = get_engine(args, device, data, dataset, split_idx)
        res = engine.run_model(args.seed)
        train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
        RES = train_list, valid_list, test_list, weight_list, val_loss_list
        append_results(RES, res)
        result_dict = create_result_dict(RES, args)
        print(result_dict)

    elif args.setting == 'ten':
        dataset, data, split_idx = load_data(args)
        print('Ten run, ', args.method)
        if args.method == 'mowst_star' or args.method == 'mowst':
            lrs_large = [0.1, 0.01, 0.001]
            dps_large = [0.1, 0.2, 0.3, 0.4, 0.5]
            lrs = [0.1, 0.01, 0.001]
            dps = [0.1, 0.2, 0.3, 0.4, 0.5]
            for lr_large in lrs_large:
                for dp_large in dps_large:
                    for lr_gate in lrs:
                        for dropout_gate in dps:
                            args.lr = lr_large
                            args.dropout = dp_large
                            args.lr_gate = lr_gate
                            args.dropout_gate = dropout_gate
                            print(args.dataset, args.setting, args.submethod, args.lr, args.dropout, args.lr_gate,
                                  args.dropout_gate)
                            engine = get_engine(args, device, data, dataset, split_idx)
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                res = engine.run_model(args.seed + run_idx)

                                append_results(RES, res)
                            result_dict = create_result_dict(RES, args)

                            with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')


        elif args.method == 'baseline':
            if args.model2 == "GIN":
                lrs = [0.1, 0.01, 0.001]
                dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
                for lr in lrs:
                    for dp in dropouts:
                        args.lr = lr
                        args.dropout = dp

                        args.run = 10
                        train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                        RES = train_list, valid_list, test_list, weight_list, val_loss_list
                        for run_idx in range(args.run):
                            args.split_run_idx = run_idx
                            dataset, data, split_idx = load_data(args)
                            engine = get_engine(args, device, data, dataset, split_idx)
                            res = engine.run_model(args.seed)

                            append_results(RES, res)

                        result_dict = create_result_dict(RES, args)


                        with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json",
                                  "a+") as f:
                            json.dump(result_dict, f)
                            f.write('\n')
            else:
                
                lrs = [0.001, 0.01, 0.1]
                model2_hidden_dims = [4, 8, 12, 32]
                dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
                for lr in lrs:
                    for dp in dropouts:
                        for m2hd in model2_hidden_dims:
    
                            args.lr = lr
                            args.dropout = dp
                            args.model2_hidden_dim = m2hd
                            args.model1_hidden_dim = m2hd

                            args.run = 10
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                args.split_run_idx = run_idx
                                dataset, data, split_idx = load_data(args)
                                engine = get_engine(args, device, data, dataset, split_idx)
                                res = engine.run_model(args.seed)

                                append_results(RES, res)

                            result_dict = create_result_dict(RES, args)

                            with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')
                
    elif args.setting == 'hetero_ten':
        print('Hetero Ten Run, ', args.method)
        if args.method == 'mowst_star' or args.method == 'mowst':
            
            lrs_large = [0.1, 0.01, 0.001]
            dps_large = [0.1, 0.2, 0.3, 0.4, 0.5]
            lrs = [0.1, 0.01, 0.001]
            dps = [0.1, 0.2, 0.3, 0.4, 0.5]
            for lr_large in lrs_large:
                for dp_large in dps_large:
                    for lr_gate in lrs:
                        for dropout_gate in dps:
                            args.lr = lr_large
                            args.dropout = dp_large
                            args.lr_gate = lr_gate
                            args.dropout_gate = dropout_gate
                            args.run = 5
                            train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                            RES = train_list, valid_list, test_list, weight_list, val_loss_list
                            for run_idx in range(args.run):
                                args.split_run_idx = run_idx
                                dataset, data, split_idx = load_data(args)
                                engine = get_engine(args, device, data, dataset, split_idx)
                                res = engine.run_model(args.seed)

                                append_results(RES, res)
                            result_dict = create_result_dict(RES, args)

                            with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                                json.dump(result_dict, f)
                                f.write('\n')


        elif args.method == 'baseline':
            if args.model2 == "GIN":
                lrs = [0.001, 0.01, 0.1]
                if args.dataset == "pokec":
                    model2_hidden_dims = [4, 8, 12]
                else:
                    model2_hidden_dims = [4, 8, 12, 32]
                for lr in lrs:
                    for m2hd in model2_hidden_dims:
                        
                        args.lr = lr
                        args.model2_hidden_dim = m2hd

                        args.run = 5
                        train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                        RES = train_list, valid_list, test_list, weight_list, val_loss_list
                        for run_idx in range(args.run):
                            args.split_run_idx = run_idx
                            dataset, data, split_idx = load_data(args)
                            engine = get_engine(args, device, data, dataset, split_idx)
                            res = engine.run_model(args.seed)

                            append_results(RES, res)

                        result_dict = create_result_dict(RES, args)

                        with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                            json.dump(result_dict, f)
                            f.write('\n')

            else:
                lrs = [0.001, 0.01, 0.1]
                if args.dataset == "pokec":
                    model2_hidden_dims = [4, 8, 12]
                else:
                    model2_hidden_dims = [4, 8, 12, 32]
                    
                for lr in lrs:
                    for m2hd in model2_hidden_dims:
                        
                        args.lr = lr
                        args.model2_hidden_dim = m2hd
                        

                        args.run = 5
                        train_list, valid_list, test_list, weight_list, val_loss_list = [], [], [], [], []
                        RES = train_list, valid_list, test_list, weight_list, val_loss_list
                        for run_idx in range(args.run):
                            args.split_run_idx = run_idx
                            dataset, data, split_idx = load_data(args)
                            engine = get_engine(args, device, data, dataset, split_idx)
                            res = engine.run_model(args.seed)

                            append_results(RES, res)

                        result_dict = create_result_dict(RES, args)

                        with open(args.result_save_path + f"res_{args.dataset}_{args.method}_{args.model2}.json", "a+") as f:
                            json.dump(result_dict, f)
                            f.write('\n')