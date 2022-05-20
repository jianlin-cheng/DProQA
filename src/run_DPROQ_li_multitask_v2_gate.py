"""
@ Description: Evaluate DPROQ multitask
"""

import json
import dgl
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
import torchmetrics
import pytorch_lightning as pl
from src.graph_transformer_edge_layer import GraphTransformerLayer

# get father path
father_path = Path(__file__).resolve().parents[1]

# load config file
config_file = f'{father_path}/configs/Lab_saw_mulitase_gate_af2_decoy_knn10_seed2222.json'
with open(config_file) as f:
    print(f'Loading config file {config_file}')
    config = json.load(f)

# readout parameters from config
_version = config['version']
_optimizer = config['opt']
_n_gpu = config['n_gpu']
_out_dir = config['out_dir']
_seed = config['seed']
_epochs = config['epochs']
_batch_size = config['batch_size']
_init_lr = config['init_lr']
_lr_reduce_factor = config['lr_reduce_factor']
_early_stop_patience = config['early_stop_patience']
_weight_decay = config['weight_decay']
_graph_n_layer = config['graph_n_layer']
_hidden_dim = config['hidden_dim']
_residual = config['residual']
_readout = config['readout']
_graph_n_heads = config['graph_n_heads']
_graph_transformer_dropout = config['graph_transformer_dropout']
_readout_dropout = config['readout_dropout']
_layer_norm = config['layer_norm']
_batch_norm = config['batch_norm']
_node_input_dim = config['node_input_dim']
_edge_input_dim = config['edge_input_dim']
_accumulate_grad_batches = config['accumulate_grad_batches']
_mse_weight = config['mse_weight']
_dataset = config['dataset']
_criterion = config['criterion']


class ResNetEmbedding(nn.Module):
    """Feature Learning Module"""
    def __init__(self, node_input_dim: int, edge_input_dim: int, out_dim: int):
        super(ResNetEmbedding, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # net work module
        self.node_embedding = nn.Linear(node_input_dim, out_dim)
        self.bn_node = nn.BatchNorm1d(num_features=out_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, out_dim)
        self.bn_edge = nn.BatchNorm1d(num_features=out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, node_feature, edge_feature):
        node_feature_embedded = self.node_embedding(node_feature)
        node_feature_embedded = self.relu(self.bn_node(node_feature_embedded))
        edge_feature_embedded = self.edge_embedding(edge_feature)
        edge_feature_embedded = self.relu(self.bn_edge(edge_feature_embedded))

        return node_feature_embedded, edge_feature_embedded


class MLPReadoutClassV2(nn.Module):
    """Read-out Module"""
    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5, L=2):
        super(MLPReadoutClassV2, self).__init__()
        self.L = L
        self.list_FC_layer = nn.Sequential()
        for i in range(L):
            self.list_FC_layer.add_module(f'Linear {i}', nn.Linear(input_dim // 2 ** i, input_dim // 2 ** (i + 1), bias=True))
            self.list_FC_layer.add_module(f'BN {i}', nn.BatchNorm1d(input_dim // 2 ** (i + 1)))
            self.list_FC_layer.add_module(f'relu {i}', nn.LeakyReLU())
            self.list_FC_layer.add_module(f'dp {i}', nn.Dropout(p=dp_rate))
        self.last_layer_classification = nn.Linear(input_dim // 2 ** L, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_2 = self.last_layer_classification(x)  # class label
        y_1 = torch.sigmoid(self.last_layer(y_2))  # dockq_score
        return y_1, y_2


class DPROQLi(pl.LightningModule):
    """DProQ model"""
    def __init__(self):
        super().__init__()
        self.node_input_dim = _node_input_dim
        self.edge_input_dim = _edge_input_dim
        self.num_heads = _graph_n_heads
        self.graph_n_layer = _graph_n_layer
        self.readout = _readout
        self.dp_rate = _graph_transformer_dropout
        self.layer_norm = _layer_norm
        self.batch_norm = _batch_norm
        self.residual = _residual
        self.init_lr = _init_lr
        self.weight_decay = _weight_decay
        self.hidden_dim = _hidden_dim
        self.opt = _optimizer
        if _criterion == 'mse':
            print('USE MSE')
            self.criterion = torchmetrics.MeanSquaredError()
        elif _criterion == 'mae':
            print('USE MAE')
            self.criterion = torchmetrics.MeanAbsoluteError()
        else:
            print('DEFAULT IS MSE')
            self.criterion = torchmetrics.MeanSquaredError()
        self.criterion_acc = torchmetrics.Accuracy()

        pl.utilities.seed.seed_everything(_seed)

        # model components
        self.mse_weight = _mse_weight
        self.ce_weight = 1 - self.mse_weight

        self.resnet_embedding = ResNetEmbedding(self.node_input_dim,
                                                self.edge_input_dim,
                                                self.hidden_dim)
        self.graph_transformer_layer = nn.ModuleList(
            [GraphTransformerLayer(in_dim=self.hidden_dim,
                                   out_dim=self.hidden_dim,
                                   num_heads=self.num_heads,
                                   dropout=self.dp_rate,
                                   layer_norm=self.layer_norm,
                                   batch_norm=self.batch_norm,
                                   residual=self.residual,
                                   use_bias=True
                                   ) for _ in range(self.graph_n_layer)]
        )
        self.MLP_layer = MLPReadoutClassV2(input_dim=self.hidden_dim, output_dim=1, dp_rate=_readout_dropout)

    def forward(self, g, node_feature, edge_feature):
        # node feature dim and edge feature dim
        node_feature_embedded, edge_feature_embedded = self.resnet_embedding(node_feature, edge_feature)
        for layer in self.graph_transformer_layer:
            h, e = layer(g, node_feature_embedded, edge_feature_embedded)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        y_1 = self.MLP_layer(hg)
        return y_1

    def configure_optimizers(self):
        if self.opt == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.init_lr,
                                   weight_decay=self.weight_decay)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW(self.parameters(),
                                    lr=self.init_lr,
                                    weight_decay=self.weight_decay,
                                    amsgrad=True)
        else:
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.init_lr,
                                  weight_decay=self.weight_decay,
                                  momentum=0.9,
                                  nesterov=True)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                             step_size=_lr_reduce_factor,
                                                             gamma=0.5),
                'monitor': 'train_loss'
            }
        }
