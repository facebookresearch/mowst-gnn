# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from utils import (get_model_hyperparameters, 
                   load_model,
                   set_random_seed,
                   save_model,
                   save_moe_model)

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import wandb
from torch.distributions import Categorical
import pandas as pd
from model import GateMLP


def get_engine(args, device, data, dataset, split_idx):
    if args.method == 'baseline':
        engine = BaselineEngine(args, device, data, dataset, split_idx)
    elif args.method == 'mowst_star':
        engine = MowstStarEngine(args, device, data, dataset, split_idx)
    elif args.method == 'mowst':
        engine = MowstEngine(args, device, data, dataset, split_idx)

    return engine


class Evaluator(object):
    def __init__(self, name):
        self.name = name
        if self.name in ['flickr', 'arxiv', "product", 'penn94', 'pokec', 'twitch-gamer']:
            self.eval_metric = 'acc'
        
    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'acc':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
        return y_true, y_pred

    def eval(self, input_dict):
        if self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
    

    def _eval_acc(self, y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(correct.float().sum().item() / len(correct))

        return sum(acc_list) / len(acc_list)



def train_vanilla(model, data, crit, optimizer, args):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = crit(F.log_softmax(out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_vanilla(model, data, split_idx, evaluator, args):
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_acc, valid_acc, test_acc, 0


@torch.no_grad()
def val_loss_vanilla(model, data, crit, args):
    model.eval()
    out = model(data)
    loss = crit(F.log_softmax(out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    return loss.item()


def vanilla_train_test_wrapper(args, model, data, crit, optimizer, split_idx, evaluator, additional=None):
    check = 0
    best_score = 0
    best_loss = float('-inf')

    for i in range(args.epoch):
        loss = train_vanilla(model, data, crit, optimizer, args)
        val_loss = val_loss_vanilla(model, data, crit, args)
        val_loss = - val_loss
        result = test_vanilla(model, data, split_idx, evaluator, args)

        if i % args.print_freq == 0:
            print(f'{i} epochs trained, loss {loss:.4f}')

        if args.early_signal == "val_loss":
            if val_loss > best_loss:
                check = 0
                best_loss = val_loss
                best_score = result[1]
                if additional:
                    saved_model = save_model(args, model, f'only_model{additional}')
                else:
                    saved_model = save_model(args, model, 'only_model')
            else:
                check += 1
                if check > args.patience:
                    print(f"{i} epochs trained, best val loss {-best_loss:.4f}")
                    break
        elif args.early_signal == "val_acc":
            if result[1] > best_score:
                check = 0
                best_loss = val_loss
                best_score = result[1]
                if additional:
                    saved_model = save_model(args, model, f'only_model{additional}')
                else:
                    saved_model = save_model(args, model, 'only_model')
            else:
                check += 1
                if check > args.patience:
                    print(f"{i} epochs trained, best val acc {best_loss:.4f}")
                    break
        else:
            raise ValueError('Invalid early_signal. Choose between val_loss and val_acc')

    if (args.early_signal == "val_loss" and best_loss > float('-inf')) or (args.early_signal == "val_acc" and best_score > 0):
        model.load_state_dict(torch.load(saved_model))
    result = test_vanilla(model, data, split_idx, evaluator, args)
    train_acc, val_acc, test_acc, _ = result
    print(f'Final results: Train {train_acc * 100:.2f} Val {val_acc * 100:.2f} Test {test_acc * 100:.2f}')
    return train_acc, val_acc, test_acc, _, best_loss


@torch.no_grad()
def compute_confidence(logit, method):
    n_classes = logit.shape[1]
    logit = nn.Softmax(1)(logit)
    if method == "variance":
        variance = torch.var(logit, dim=1, unbiased=False)
        zero_tensor = torch.zeros(n_classes)
        zero_tensor[0] = 1
        max_variance = torch.var(zero_tensor, unbiased=False)

        res = variance / max_variance
    elif method == "entropy":
        res = 1 - Categorical(probs=logit).entropy() / np.log(n_classes)

    return res.view(-1, 1)


def train_mowst1(model1, model2, data, crit, optimizer1, args):
    model1.train()
    model2.eval()
    optimizer1.zero_grad()
    if args.confidence == 'variance':
        model1_x = model1(data)[data.train_mask]
        model1_conf = compute_confidence(args, model1_x)
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)
        model1_x = model1_x[data.train_mask]
        model1_conf = model1_conf[data.train_mask]
    model1_conf = (model1_conf ** args.alpha).view(-1)
    with torch.no_grad():
        model2_x = model2(data)[data.train_mask]
    if crit.__class__.__name__ == 'BCEWithLogitsLoss':
        loss_model1 = crit(F.softmax(model1_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
        loss_model2 = crit(F.softmax(model2_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss_model1 = crit(F.log_softmax(model1_x, dim=1), data.y.squeeze(1)[data.train_mask])
        loss_model2 = crit(F.log_softmax(model2_x, dim=1), data.y.squeeze(1)[data.train_mask])

    loss = model1_conf * loss_model1 + (1 - model1_conf) * loss_model2
    loss.mean().backward()
    optimizer1.step()
    return loss.mean().item()


def train_mowst2(model1, model2, data, crit, optimizer2, args):
    model1.eval()
    model2.train()
    optimizer2.zero_grad()
    model2_x = model2(data)[data.train_mask]

    with torch.no_grad():
        if args.confidence == 'variance':
            model1_x = model1(data)[data.train_mask]
            model1_conf = compute_confidence(args, model1_x)
        elif args.confidence == 'learnable':
            model1_x, model1_conf = model1(data)
            model1_x = model1_x[data.train_mask]
            model1_conf = model1_conf[data.train_mask]

    model1_conf = (model1_conf ** args.alpha).view(-1)

    if crit.__class__.__name__ == 'BCEWithLogitsLoss':
        loss_model1 = crit(F.softmax(model1_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
        loss_model2 = crit(F.softmax(model2_x, dim=-1)[:, 1].view(-1),
                           data.y.squeeze(1)[data.train_mask].to(torch.float))
    else:
        loss_model1 = crit(F.log_softmax(model1_x, dim=1), data.y.squeeze(1)[data.train_mask])
        loss_model2 = crit(F.log_softmax(model2_x, dim=1), data.y.squeeze(1)[data.train_mask])

    loss = model1_conf * loss_model1 + (1 - model1_conf) * loss_model2
    loss.mean().backward()
    optimizer2.step()
    return loss.mean().item()


@torch.no_grad()
def test_mowst(model1, model2, data, split_idx, evaluator, args, device, check_g_dist):
    model1.eval()
    model2.eval()
    if args.confidence == 'variance':
        model1_x = model1(data)
        model1_conf = compute_confidence(args, model1_x)
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)

    model1_conf = (model1_conf ** args.alpha).view(-1)
    model1_out = nn.Softmax(1)(model1_x)
    model2_out = nn.Softmax(1)(model2(data))

    tmp_train_acc_list = []
    tmp_val_acc_list = []
    tmp_test_acc_list = []

    for t in range(args.m_times):
        m = torch.rand(model1_conf.shape).to(device)
        gate = (m < model1_conf).int().view(-1, 1)
        model1_pred = model1_out.argmax(dim=-1, keepdim=True)
        model2_pred = model2_out.argmax(dim=-1, keepdim=True)
        y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
        y_pred = y_pred.view(-1, 1)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })
        tmp_train_acc_list.append(train_acc)
        tmp_val_acc_list.append(valid_acc)
        tmp_test_acc_list.append(test_acc)

    train_acc = np.mean(tmp_train_acc_list)
    valid_acc = np.mean(tmp_val_acc_list)
    test_acc = np.mean(tmp_test_acc_list)

    if check_g_dist:
        gate = gate.view(-1)
        model1_percent = gate.sum().item() / len(gate) * 100
        print(f'Model1 / Model2: {model1_percent:.2f} / {100 - model1_percent:.2f}')

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def get_cur_confidence(args, model1, data):
    model1.eval()
    if args.confidence == 'variance':
        model1_x = model1(data)[data.train_mask]
        model1_conf = compute_confidence(args, model1_x) ** args.alpha
    elif args.confidence == 'learnable':
        model1_x, model1_conf = model1(data)
        model1_conf = (model1_conf[data.train_mask] ** args.alpha).view(-1)
    return model1_conf


class BaselineEngine(object):
    def __init__(self, args, device, data, dataset, split_idx):
        self.args = args
        self.device = device
        self.data = data.to(self.device)
        self.dataset = dataset
        input_dim, hidden_dim, output_dim, num_layers = get_model_hyperparameters(args, data, dataset)
        model_name = args.model2
        dropout = args.dropout
        self.model = load_model(model_name, input_dim, hidden_dim, output_dim, num_layers, dropout, args).to(
            self.device)
        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)

        self.crit = nn.NLLLoss()
        if self.args.adam:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model.reset_parameters()
        result = vanilla_train_test_wrapper(
            self.args, self.model, self.data, 
            self.crit, self.optimizer, 
            self.split_idx, self.evaluator)
        return result



class MowstStarEngine(object):
    def __init__(self, args, device, data, dataset, split_idx):
        self.args = args
        self.device = device
        self.data = data.to(self.device)
        model1_hyper, model2_hyper = get_model_hyperparameters(args, data, dataset)
        input_dim, hidden_dim1, output_dim, num_layers1 = model1_hyper
        input_dim, hidden_dim2, output_dim, num_layers2 = model2_hyper
        dropout = args.dropout

        self.model1 = load_model(args.model1, input_dim, hidden_dim1, output_dim, num_layers1, dropout, args).to(
            self.device)
        self.model2 = load_model(args.model2, input_dim, hidden_dim2, output_dim, num_layers2, dropout, args).to(
            self.device)

        
        
        if args.original_data == "true":
            self.gate_model = GateMLP(input_dim + 2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.original_data == "false":
            self.gate_model = GateMLP(2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.original_data == "hypermlp":
            self.para1_model = GateMLP(in_channels=input_dim,
                                        hidden_channels=args.hyper_hidden,
                                        out_channels=2 * args.model1_hidden_dim,
                                        num_layers=args.hyper_num_layers,
                                        dropout=0.5, args=args).to(self.device)
            self.parabias1_model = GateMLP(in_channels=input_dim,
                                            hidden_channels=args.hyper_hidden,
                                            out_channels=args.model1_hidden_dim,
                                            num_layers=args.hyper_num_layers,
                                            dropout=0.5, args=args).to(self.device)
            self.para2_model = GateMLP(in_channels=input_dim,
                                        hidden_channels=args.hyper_hidden,
                                        out_channels=args.model1_hidden_dim,
                                        num_layers=args.hyper_num_layers,
                                        dropout=0.5, args=args).to(self.device)
            self.parabias2_model = GateMLP(in_channels=input_dim,
                                            hidden_channels=args.hyper_hidden,
                                            out_channels=1,
                                            num_layers=args.hyper_num_layers,
                                            dropout=0.5, args=args).to(self.device)
            self.gate_model = [self.para1_model, self.parabias1_model, self.para2_model, self.parabias2_model]
        elif args.original_data == "hypergcn":
            pass
        

        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)
        if self.args.subloss == 'separate':
            self.args.crit = 'nllloss'
            self.crit = nn.NLLLoss(reduction='none')
            self.crit_pretrain = nn.NLLLoss()
        elif self.args.subloss == 'joint':
            self.args.crit = 'crossentropy'
            self.crit = nn.CrossEntropyLoss()
            self.crit_pretrain = nn.CrossEntropyLoss()
        if self.args.adam:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            if self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.Adam(
                self.model1.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.Adam(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            elif self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.AdamW(
                self.model1.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.AdamW(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model1.reset_parameters()
        self.model2.reset_parameters()
        if self.args.original_data != "hypermlp":
            self.gate_model.reset_parameters()
        else:
            self.para1_model.reset_parameters()
            self.parabias1_model.reset_parameters()
            self.para2_model.reset_parameters()
            self.parabias2_model.reset_parameters()

        if self.args.submethod in ['pretrain_model1', 'pretrain_both']:
            if self.args.model1 != self.args.model2:
                _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                               self.optimizer1, self.split_idx, self.evaluator)
            else:
                _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                               self.optimizer1, self.split_idx, self.evaluator, 1)
        if self.args.submethod in ['pretrain_model2', 'pretrain_both']:
            if self.args.model1 != self.args.model2:
                _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                               self.optimizer2, self.split_idx, self.evaluator)
            else:
                _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                               self.optimizer2, self.split_idx, self.evaluator, 2)

        self.model1.dropout = self.args.dropout_gate
        self.model2.dropout = self.args.dropout_gate

        if self.args.original_data != "hypermlp":
            self.gate_model.dropout = self.args.dropout_gate

        result = simple_gate_train_test_wrapper(self.args, self.model1, self.model2, self.gate_model,
                                                self.data, self.crit, self.optimizer, self.split_idx,
                                                self.evaluator, self.device)

        return result


class MowstEngine(object):
    def __init__(self, args, device, data, dataset, split_idx):

        self.args = args
        self.device = device
        self.data = data.to(self.device)
        model1_hyper, model2_hyper = get_model_hyperparameters(args, data, dataset)
        input_dim, hidden_dim1, output_dim, num_layers1 = model1_hyper
        input_dim, hidden_dim2, output_dim, num_layers2 = model2_hyper
        dropout = args.dropout

        self.model1 = load_model(args.model1, input_dim, hidden_dim1, output_dim, num_layers1, dropout, args).to(
            self.device)
        self.model2 = load_model(args.model2, input_dim, hidden_dim2, output_dim, num_layers2, dropout, args).to(
            self.device)

        if args.original_data == "true":
            self.gate_model = GateMLP(input_dim + 2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.original_data == "false":
            self.gate_model = GateMLP(2, hidden_dim1, 1, num_layers1, dropout, args).to(self.device)
        elif args.original_data == "hypermlp":
            self.para1_model = GateMLP(in_channels=input_dim,
                                        hidden_channels=args.hyper_hidden,
                                        out_channels=2 * args.model1_hidden_dim,
                                        num_layers=args.hyper_num_layers,
                                        dropout=0.5, args=args).to(self.device)
            self.parabias1_model = GateMLP(in_channels=input_dim,
                                            hidden_channels=args.hyper_hidden,
                                            out_channels=args.model1_hidden_dim,
                                            num_layers=args.hyper_num_layers,
                                            dropout=0.5, args=args).to(self.device)
            self.para2_model = GateMLP(in_channels=input_dim,
                                        hidden_channels=args.hyper_hidden,
                                        out_channels=args.model1_hidden_dim,
                                        num_layers=args.hyper_num_layers,
                                        dropout=0.5, args=args).to(self.device)
            self.parabias2_model = GateMLP(in_channels=input_dim,
                                            hidden_channels=args.hyper_hidden,
                                            out_channels=1,
                                            num_layers=args.hyper_num_layers,
                                            dropout=0.5, args=args).to(self.device)
            self.gate_model = [self.para1_model, self.parabias1_model, self.para2_model, self.parabias2_model]
        
        self.split_idx = split_idx
        self.evaluator = Evaluator(args.dataset)

    def initialize(self, seed):
        set_random_seed(seed)
        if self.args.crit == 'nllloss':
            self.crit = nn.NLLLoss(reduction='none')
            self.crit_pretrain = nn.NLLLoss()
        else:
            raise ValueError('Invalid Crit. In In Turn setting, only NLLLoss can be used')

        if self.args.adam:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            if self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.Adam(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.Adam(
                list(self.model1.parameters()) + list(self.gate_model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.Adam(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            if self.args.original_data != "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.gate_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            elif self.args.original_data == "hypermlp":
                self.optimizer = torch.optim.AdamW(
                    list(self.model1.parameters()) + list(self.model2.parameters()) + list(
                        self.para1_model.parameters()) + list(self.parabias1_model.parameters()) + list(
                        self.para2_model.parameters()) + list(self.parabias2_model.parameters()),
                    lr=self.args.lr_gate,
                    weight_decay=self.args.weight_decay)
            self.optimizer1 = torch.optim.AdamW(
                list(self.model1.parameters()) + list(self.gate_model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
            self.optimizer2 = torch.optim.AdamW(
                self.model2.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)

    def run_model(self, seed):
        self.initialize(seed)
        self.model1.reset_parameters()
        self.model2.reset_parameters()
        if self.args.original_data != "hypermlp":
            self.gate_model.reset_parameters()
        else:
            self.para1_model.reset_parameters()
            self.parabias1_model.reset_parameters()
            self.para2_model.reset_parameters()
            self.parabias2_model.reset_parameters()

        if self.args.submethod in ['pretrain_model1', 'pretrain_both']:
            _ = vanilla_train_test_wrapper(self.args, self.model1, self.data, self.crit_pretrain,
                                           self.optimizer1, self.split_idx, self.evaluator)
        if self.args.submethod in ['pretrain_model2', 'pretrain_both']:
            _ = vanilla_train_test_wrapper(self.args, self.model2, self.data, self.crit_pretrain,
                                           self.optimizer2, self.split_idx, self.evaluator)
            print('Model pretraining done')
        self.model1.dropout = self.args.dropout_gate
        self.model2.dropout = self.args.dropout_gate
        if self.args.original_data != "hypermlp":
            self.gate_model.dropout = self.args.dropout_gate
        result = mowst_train_test_wrapper_simple_gate(self.args, self.model1, self.model2, self.gate_model, self.data,
                                                      self.crit,
                                                      self.optimizer1, self.optimizer2,
                                                      self.split_idx, self.evaluator, self.device)

        return result


def train_simple_gate(model1, model2, gate_model, data, crit, optimizer, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()

    optimizer.zero_grad()

    model1_out = model1(data)
    model2_out = model2(data)
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)


    if args.subloss == 'joint':
        out = model1_out * gating + model2_out * (1 - gating)
        loss = crit(out[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    elif args.subloss == 'separate':
        if crit.__class__.__name__ == 'NLLLoss':
            loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
            loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        elif crit.__class__.__name__ == 'CrossEntropyLoss':
            loss1 = crit(model1_out[data.train_mask], data.y.squeeze(1)[data.train_mask])
            loss2 = crit(model2_out[data.train_mask], data.y.squeeze(1)[data.train_mask])
        gating = gating[data.train_mask].view(-1)
        loss = loss1 * gating + loss2 * (1 - gating)
        loss.mean().backward()
        optimizer.step()
        return loss.mean().item()


def eval_for_simplicity(evaluator, data, model1_out, model2_out, gating, split_idx, args):
    out = model1_out * gating + model2_out * (1 - gating)

    y_pred = out.argmax(dim=-1, keepdim=True)
    model1_weight = gating.mean().item()

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_acc, valid_acc, test_acc, model1_weight


def eval_for_multi(args, gating, device, model1_out, model2_out, evaluator, data, split_idx):
    tmp_train_acc_list = []
    tmp_val_acc_list = []
    tmp_test_acc_list = []
    model1_weight_list = []
    for t in range(args.m_times):
        m = torch.rand(gating.shape).to(device)
        gate = (m < gating).int().view(-1, 1)

        model1_pred = model1_out.argmax(dim=-1, keepdim=True)
        model2_pred = model2_out.argmax(dim=-1, keepdim=True)
        y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
        y_pred = y_pred.view(-1, 1)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })

        tmp_train_acc_list.append(train_acc)
        tmp_val_acc_list.append(valid_acc)
        tmp_test_acc_list.append(test_acc)
        gate = gate.view(-1)
        model1_weight = gate.sum().item() / len(gate)
        model1_weight_list.append(model1_weight)

    train_acc = np.mean(tmp_train_acc_list)
    valid_acc = np.mean(tmp_val_acc_list)
    test_acc = np.mean(tmp_test_acc_list)
    model1_weight = np.mean(model1_weight_list)
    return train_acc, valid_acc, test_acc, model1_weight


@torch.no_grad()
def test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = F.softmax(model1(data), dim=1)
    model2_out = F.softmax(model2(data), dim=1)
    

    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)

    if args.infer_method == 'joint':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_simplicity(evaluator, data, model1_out, model2_out,
                                                                            gating, split_idx, args)
        return train_acc, valid_acc, test_acc, model1_weight
    elif args.infer_method == 'multi':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
                                                                       evaluator, data, split_idx)
        return train_acc, valid_acc, test_acc, model1_weight


def simple_gate_train_test_wrapper(args, model1, model2, gate_model, data, crit, optimizer, split_idx, evaluator,
                                   device):
    check = 0
    best_score = 0
    best_loss = float('-inf')

    for i in range(args.epoch):
        loss = train_simple_gate(model1, model2, gate_model, data, crit, optimizer, args)
        result = test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        val_loss, gate_weight, l1, l2, g = cal_val_loss_simple_gate(model1, model2, gate_model, data, crit, args)
        val_loss = - val_loss

        model1_result = test_vanilla(model1, data, split_idx, evaluator, args)
        model2_result = test_vanilla(model2, data, split_idx, evaluator, args)

        if args.log:
            wandb.log({'Train Acc': result[0],
                       'Val Acc': result[1],
                       'Test Acc': result[2],
                       'Train Loss': loss,
                       'Val Loss': -val_loss,
                       'Gate Weight': gate_weight,
                       "Gate Weight Dis": g,
                       'Model1 Val Acc': model1_result[1],
                       'Model2 Val Acc': model2_result[2],
                       'L1': l1,
                       'L2': l2})

        if i % args.print_freq == 0:
            print(f'{i} epochs trained, loss {loss:.4f}')

        
        if val_loss > best_loss:
            check = 0
            best_score = result[1]
            best_loss = val_loss
            saved_model1 = save_model(args, model1, 'model1')
            saved_model2 = save_model(args, model2, 'model2')
            if args.original_data != "hypermlp":
                saved_gate_model = save_model(args, gate_model, 'gate_model')
            else:
                para1_model, parabias1_model, para2_model, parabias2_model = gate_model
                saved_para1_model = save_model(args, para1_model, 'para1_model')
                save_parabias1_model = save_model(args, parabias1_model, 'parabias1_model')
                saved_para2_model = save_model(args, para2_model, 'para2_model')
                save_parabias2_model = save_model(args, parabias2_model, 'parabias2_model')

    model1.load_state_dict(torch.load(saved_model1))
    model2.load_state_dict(torch.load(saved_model2))
    if args.original_data != "hypermlp":
        gate_model.load_state_dict(torch.load(saved_gate_model))
    else:
        para1_model.load_state_dict(torch.load(saved_para1_model))
        parabias1_model.load_state_dict(torch.load(save_parabias1_model))
        para2_model.load_state_dict(torch.load(saved_para2_model))
        parabias2_model.load_state_dict(torch.load(save_parabias2_model))
        gate_model = [para1_model, parabias1_model, para2_model, parabias2_model]

    result = test_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
    train_acc, val_acc, test_acc, model1_weight = result
    print(f'Final results: Train {train_acc * 100:.2f} Val {val_acc * 100:.2f} Test {test_acc * 100:.2f} Model1 Weight {model1_weight * 100:.2f}')

    return train_acc, val_acc, test_acc, model1_weight, best_loss


@torch.no_grad()
def cal_val_loss_simple_gate(model1, model2, gate_model, data, crit, args):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()

    model1_out = model1(data)
    model2_out = model2(data)
    
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)


    loss1 = crit(F.log_softmax(model1_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    loss2 = crit(F.log_softmax(model2_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    if args.subloss == 'joint':
        out = model1_out * gating + model2_out * (1 - gating)
        loss = crit(F.log_softmax(out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])

    elif args.subloss == 'separate':
        loss1 = loss1.mean()
        loss2 = loss2.mean()
        loss = (loss1 * gating[data.val_mask] + loss2 * (1 - gating[data.val_mask])).mean()

    return loss.item(), gating.mean().item(), loss1.item(), loss2.item(), gating.cpu().numpy()


def mowst_train_test_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer1, optimizer2,
                                         split_idx, evaluator, device):
    big_epoch_check = 0
    big_best_score = 0
    
    n = len(data.y)
    df = pd.DataFrame({"id":list(range(n))})


    for j in range(args.big_epoch):
        print(f'------ Big epoch {j} Model 1 ------')
        model1_turn_val_acc = mowst_train_test_model1_turn_wrapper_simple_gate(args, model1, model2, gate_model, data,
                                                                               crit, optimizer1,
                                                                               split_idx, evaluator, device,
                                                                               big_best_score)
    
        print(f'------ Big epoch {j} Model 2 ------')
        mowst_train_test_model2_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer2,
                                                         split_idx, evaluator, device,
                                                         model1_turn_val_acc)


        result = test_mowst_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        
        log_conf = get_cur_confidence_simple_gate(args, gate_model, model1, data)
        if args.log:
            wandb.log({'Train Acc': result[0],
                       'Val Acc': result[1],
                       'Test Acc': result[2],
                       'Confidence': log_conf.mean().item(),
                       "Confidence Distribution": log_conf.cpu().numpy()
                       })

        
            
        if result[1] > big_best_score:
            big_epoch_check = 0
            big_best_score = result[1]
            
            saved_model1_big = save_moe_model(args, model1, '1_big')
            saved_model2_big = save_moe_model(args, model2, '2_big')
        else:
            big_epoch_check += 1
            if big_epoch_check > args.big_patience:
                print(f"{j} big epochs trained, best val acc {big_best_score:.4f}")
                break
        

    model1.load_state_dict(torch.load(saved_model1_big))
    model2.load_state_dict(torch.load(saved_model2_big))
    result = test_mowst_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
    train_acc, val_acc, test_acc, model1_weight, _ = result
    print(f'Final results: Train {train_acc * 100:.2f} Val {val_acc * 100:.2f} Test {test_acc * 100:.2f} Model1 Weight {model1_weight * 100:.2f}')
    return result


@torch.no_grad()
def generate_embedding(args, model1, model2, gate_model, data):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    _, model2_emb = model2(data, mode=True)
    
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)
    

    return model2_emb.cpu().numpy(), gating.view(-1).cpu().numpy()






def mowst_train_test_model1_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer1,
                                                     split_idx, evaluator, device,
                                                     big_best_score):
    saved_model1_previous_big_turn = save_moe_model(args, model1, '1_big')
    saved_model2_previous_big_turn = save_moe_model(args, model2, '2_big')

    check = 0
    best_score = 0
    for i in range(args.epoch):

        loss, l1, l2 = train_mowst1_simple_gate(model1, model2, gate_model, data, crit, optimizer1, args)
        result = test_mowst_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        
        if args.log:
            wandb.log({'L1 train loss': l1.mean().item(),
                       "L1 train loss Distribution": l1.detach().cpu().numpy(),
                       'L2 train loss': l2.mean().item(),
                       "L2 train loss Distribution": l2.detach().cpu().numpy()
                       })

        if i % args.print_freq == 0:
            print(f'{i} epochs trained, loss {loss:.4f}')

        
        if result[1] > best_score:
            check = 0
            best_score = result[1]
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print(f"{i} epochs trained, best val loss {best_score:.4f}")
                break

    if best_score > big_best_score:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))
    else:
        model1.load_state_dict(torch.load(saved_model1_previous_big_turn))
        model2.load_state_dict(torch.load(saved_model2_previous_big_turn))

    saved_model1 = save_moe_model(args, model1, 1)
    saved_model2 = save_moe_model(args, model2, 2)
    return max(best_score, big_best_score)


def mowst_train_test_model2_turn_wrapper_simple_gate(args, model1, model2, gate_model, data, crit, optimizer2,
                                                     split_idx, evaluator, device,
                                                     model1_turn_val_acc):
    saved_model1_previous_small_turn = save_moe_model(args, model1, 1)
    saved_model2_previous_small_turn = save_moe_model(args, model2, 2)

    check = 0
    best_score = 0
    for i in range(args.epoch):

        loss, l1, l2 = train_mowst2_simple_gate(model1, model2, gate_model, data, crit, optimizer2, args)
        result = test_mowst_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device)
        
        if args.log:
            wandb.log({'L1 train loss': l1.mean().item(),
                       "L1 train loss Distribution": l1.detach().cpu().numpy(),
                       'L2 train loss': l2.mean().item(),
                       "L2 train loss Distribution": l2.detach().cpu().numpy()
                       })

        if i % args.print_freq == 0:
            print(f'{i} epochs trained, loss {loss:.4f}')

        if result[1] > best_score:
            check = 0
            best_score = result[1]
            saved_model1 = save_moe_model(args, model1, '1_inner')
            saved_model2 = save_moe_model(args, model2, '2_inner')
        else:
            check += 1
            if check > args.patience:
                print(f"{i} epochs trained, best val acc {best_score:.4f}")
                break
    if best_score > model1_turn_val_acc:
        model1.load_state_dict(torch.load(saved_model1))
        model2.load_state_dict(torch.load(saved_model2))
    else:
        model1.load_state_dict(torch.load(saved_model1_previous_small_turn))
        model2.load_state_dict(torch.load(saved_model2_previous_small_turn))

    saved_model1 = save_moe_model(args, model1, 1)
    saved_model2 = save_moe_model(args, model2, 2)
    

@torch.no_grad()
def get_denoise_info(model1, model2, gate_model, data, crit, args, evaluator, split_idx, device):
    model1.eval()
    model2.eval()
    # get loss info

    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()

    model1_out = model1(data)
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)
    
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1), data.y.squeeze(1))
        loss2 = crit(F.log_softmax(model2_out, dim=1), data.y.squeeze(1))
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating.view(-1)
    _ = loss1 * gating + loss2 * (1 - gating)

    model1_out = F.softmax(model1_out, dim=1)
    model2_out = F.softmax(model2_out, dim=1)

    m = torch.rand(gating.shape).to(device)
    gate = (m < gating).int().view(-1, 1)

    model1_pred = model1_out.argmax(dim=-1, keepdim=True)
    model2_pred = model2_out.argmax(dim=-1, keepdim=True)
    y_pred = model1_pred.view(-1) * gate.view(-1) + model2_pred.view(-1) * (1 - gate.view(-1))
    y_pred = y_pred.view(-1, 1)
    correct = (y_pred == data.y).view(-1).float()

    return loss1.cpu().numpy(), loss2.cpu().numpy(), gating.cpu().numpy(), correct.cpu().numpy()



def train_mowst1_simple_gate(model1, model2, gate_model, data, crit, optimizer1, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()
    optimizer1.zero_grad()
    model1_out = model1(data)

    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)
    
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.train_mask].view(-1)
    loss = loss1 * gating + loss2 * (1 - gating)
    loss.mean().backward()
    optimizer1.step()
    return loss.mean().item(), loss1, loss2


def train_mowst2_simple_gate(model1, model2, gate_model, data, crit, optimizer2, args):
    model1.train()
    model2.train()
    if args.original_data != "hypermlp":
        gate_model.train()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.train()
        parabias1_model.train()
        para2_model.train()
        parabias2_model.train()
    optimizer2.zero_grad()
    model2_out = model2(data)

    with torch.no_grad():
        model1_out = model1(data)
        
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
        

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.train_mask].view(-1)

    loss = loss1 * gating + loss2 * (1 - gating)
    loss.mean().backward()
    optimizer2.step()
    return loss.mean().item(), loss1, loss2


@torch.no_grad()
def loss_mowst_simple_gate(model1, model2, gate_model, data, crit, args):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)
    
    with torch.no_grad():
        model2_out = model2(data)

    if crit.__class__.__name__ == 'NLLLoss':
        loss1 = crit(F.log_softmax(model1_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
        loss2 = crit(F.log_softmax(model2_out, dim=1)[data.val_mask], data.y.squeeze(1)[data.val_mask])
    else:
        raise ValueError('Invalid Crit found during training')
    gating = gating[data.val_mask].view(-1)
    loss = loss1 * gating + loss2 * (1 - gating)

    return loss.mean().item(), loss, loss1, loss2


@torch.no_grad()
def test_mowst_simple_gate(model1, model2, gate_model, data, split_idx, evaluator, args, device):
    model1.eval()
    model2.eval()
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1_out = model1(data)
    model2_out = model2(data)
    
    
    var_conf = compute_confidence(model1_out, "variance")
    ent_conf = compute_confidence(model1_out, "entropy")
    dispersion = torch.cat((var_conf, ent_conf), dim=1)
    if args.original_data == "true":
        gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
        gating = nn.Sigmoid()(gate_model(gate_input))
    elif args.original_data == "false":
        gating = nn.Sigmoid()(gate_model(dispersion))
    elif args.original_data == "hypermlp":
        node_feature = data.mlp_x
        para1 = para1_model(node_feature)
        parabias1 = parabias1_model(node_feature)
        para2 = para2_model(node_feature)
        parabias2 = parabias2_model(node_feature)

        para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
        dispersion = dispersion[:, np.newaxis, :]
        para2 = para2[:, :, np.newaxis]

        gating = torch.matmul(dispersion, para1)
        gating += parabias1[:, np.newaxis, :]
        gating = torch.matmul(gating, para2)
        gating += parabias2[:, np.newaxis, :]
        gating = nn.Sigmoid()(gating).view(-1, 1)
    

    model1_out = F.softmax(model1_out, dim=1)
    model2_out = F.softmax(model2_out, dim=1)

    if args.infer_method == 'simple':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_simplicity(evaluator, data, model1_out, model2_out,
                                                                            gating, split_idx, args)
        return train_acc, valid_acc, test_acc, model1_weight, float(0)
    elif args.infer_method == 'multi':
        train_acc, valid_acc, test_acc, model1_weight = eval_for_multi(args, gating, device, model1_out, model2_out,
                                                                       evaluator, data, split_idx)
        return train_acc, valid_acc, test_acc, model1_weight, float(0)

    


@torch.no_grad()
def get_cur_confidence_simple_gate(args, gate_model, model1, data):
    if args.original_data != "hypermlp":
        gate_model.eval()
    else:
        para1_model, parabias1_model, para2_model, parabias2_model = gate_model
        para1_model.eval()
        parabias1_model.eval()
        para2_model.eval()
        parabias2_model.eval()
    model1.eval()
    with torch.no_grad():
        
        model1_out = model1(data)
        var_conf = compute_confidence(model1_out, "variance")
        ent_conf = compute_confidence(model1_out, "entropy")
        dispersion = torch.cat((var_conf, ent_conf), dim=1)
        if args.original_data == "true":
            gate_input = torch.cat((dispersion, data.mlp_x), dim=1)
            gating = nn.Sigmoid()(gate_model(gate_input))
        elif args.original_data == "false":
            gating = nn.Sigmoid()(gate_model(dispersion))
        elif args.original_data == "hypermlp":
            node_feature = data.mlp_x
            para1 = para1_model(node_feature)
            parabias1 = parabias1_model(node_feature)
            para2 = para2_model(node_feature)
            parabias2 = parabias2_model(node_feature)

            para1 = para1.reshape(-1, 2, args.model1_hidden_dim)
            dispersion = dispersion[:, np.newaxis, :]
            para2 = para2[:, :, np.newaxis]

            gating = torch.matmul(dispersion, para1)
            gating += parabias1[:, np.newaxis, :]
            gating = torch.matmul(gating, para2)
            gating += parabias2[:, np.newaxis, :]
            gating = nn.Sigmoid()(gating).view(-1, 1)
    
    return gating.view(-1)