import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(GCN, self).__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=~args.no_cached, add_self_loops=~args.no_add_self_loops,
                    normalize=~args.no_normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=~args.no_cached,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=~args.no_cached, add_self_loops=~args.no_add_self_loops,
                    normalize=~args.no_normalize))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, mode=None):
        x, adj_t = data.x, data.adj_t
        if not mode:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, adj_t)
                if self.args.no_batch_norm is False:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_final = self.convs[-1](x, adj_t)
            return x_final
        else:
            for i, conv in enumerate(self.convs[:-2]):
                x = conv(x, adj_t)
                if self.args.no_batch_norm is False:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-2](x, adj_t)


            x_last2final = F.relu(x)
            x_last2final = F.dropout(x_last2final, p=self.dropout, training=self.training)
            x_final = self.convs[-1](x_last2final, adj_t)
            return x_final, x


class Sage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final





class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(MLP(in_channels, hidden_channels, hidden_channels, num_layers, dropout, args, gin=True)))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, num_layers, dropout, args, gin=True)))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GINConv(MLP(hidden_channels, out_channels, hidden_channels, num_layers, dropout, args, gin=True)))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        print(out_channels)
        self.args = args
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=args.heads, cached=~args.no_cached,
                                  add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * args.heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, heads=args.heads, cached=~args.no_cached,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * args.heads))
        if args.dataset == "pubmed":
            self.convs.append(
                GATConv(hidden_channels * args.heads, out_channels, heads=1, cached=~args.no_cached, concat=False,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        else:
            self.convs.append(
                GATConv(hidden_channels * args.heads, out_channels, heads=1, cached=~args.no_cached, concat=False,
                        add_self_loops=~args.no_add_self_loops, normalize=~args.no_normalize))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.convs[-1](x, adj_t)
        return x_final


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args, gin=False):
        super(MLP, self).__init__()
        self.args = args
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.gin = gin
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        if self.gin:
            x = data
        else:
            x = data.mlp_x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        return x_final


class GateMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super().__init__()
        self.args = args
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm is False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        return x_final


class MLPLearn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, args):
        super(MLPLearn, self).__init__()
        self.args = args

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.conf_layer1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conf_layer2 = torch.nn.Linear(hidden_channels, 1)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.mlp_x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.args.no_batch_norm == False:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lins[-1](x)
        conf = self.conf_layer1(x)
        conf = F.relu(conf)
        conf = F.dropout(conf, p=self.dropout, training=self.training)
        conf = torch.nn.Sigmoid()(self.conf_layer2(conf))
        return x_final, conf
