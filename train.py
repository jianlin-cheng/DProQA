"""
@ Description: Training script for DProQA
"""

import os
from datetime import datetime
import argparse
import json

import dgl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

# customer modules
from src.run_DPROQ_li_multitask_v2_gate import ResNetEmbedding, MLPReadoutClassV2, GraphTransformerLayer
from src.data_module import DProQAData, collate

# create ArgumentParser
parser = argparse.ArgumentParser(
    usage='python train.py --config config.json',
)

parser.add_argument('--config', type=str, help="Please give a config.json file with training/model/data/param details")
args = parser.parse_args()
with open(args.config) as f:
    print(f'Loading {args.config} for training.')
    config = json.load(f)

# readout parameters from config
_version = config['version']
_optimizer = config['opt']
_n_gpu = config['n_gpu']
_wb_out_dir = config['wb_out_dir']
_ckpt_out_dir = config['ckpt_out_dir']
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

_wb_out_dir = os.path.abspath(_wb_out_dir)
_ckpt_out_dir = os.path.abspath(_ckpt_out_dir)

now = datetime.now()
_CURRENT_TIME = now.strftime("%H_%M_%S")


class DProQ(pl.LightningModule):
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
                                   ) for _ in range(self.graph_n_layer)])

        self.MLP_layer = MLPReadoutClassV2(input_dim=self.hidden_dim, output_dim=1, dp_rate=_readout_dropout)

    def forward(self, g, node_feature, edge_feature):
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
            print('USING ADAM')
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

        if self.current_epoch < 20:
            return {
                'optimizer': optimizer,
                "lr_scheduler": {
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                 step_size=_lr_reduce_factor,
                                                                 gamma=0.5),
                    'monitor': 'val_loss'
                }
            }
        else:
            print('START to USE SGD OPTIMIZER!')
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.init_lr * 0.1,
                                  weight_decay=self.weight_decay,
                                  momentum=0.9,
                                  nesterov=True)
            return {
                'optimizer': optimizer,
                "lr_scheduler": {
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                 step_size=_lr_reduce_factor,
                                                                 gamma=0.5),
                    'monitor': 'val_loss'
                }
            }

    def training_step(self, train_batch, batch_idx):
        batch_graphs, batch_targets, batch_targets_class = train_batch
        batch_x = batch_graphs.ndata['f'].to(torch.float)
        batch_e = batch_graphs.edata['f'].to(torch.float)
        batch_scores, batch_class = self.forward(batch_graphs, batch_x, batch_e)

        # mse    
        train_mse = self.criterion(batch_scores, batch_targets)
        self.log('train_mse', train_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        train_mse *= self.mse_weight

        # ce
        train_ce = F.cross_entropy(batch_class, batch_targets_class.squeeze(1))
        self.log('train_ce', train_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        train_ce *= self.ce_weight

        # acc
        train_acc = self.criterion_acc(F.softmax(batch_class, dim=1), batch_targets_class.squeeze(1))

        # train loss
        train_loss = train_mse + train_ce

        # log
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        batch_graphs, batch_targets, batch_targets_class = val_batch
        batch_x = batch_graphs.ndata['f'].to(torch.float)
        batch_e = batch_graphs.edata['f'].to(torch.float)
        batch_scores, batch_class = self.forward(batch_graphs, batch_x, batch_e)

        # mse
        val_mse = self.criterion(batch_scores, batch_targets)
        self.log('val_mse', val_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        val_mse *= self.mse_weight

        # ce
        val_ce = F.cross_entropy(batch_class, batch_targets_class.squeeze(1))
        self.log('val_ce', val_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        val_ce *= self.ce_weight

        # acc
        val_acc = self.criterion_acc(F.softmax(batch_class, dim=1), batch_targets_class.squeeze(1))

        # val loss
        val_loss = val_mse + val_ce

        # log
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)

        return val_loss

    def test_step(self, test_batch, batch_idx):
        batch_graphs, batch_targets, batch_targets_class = test_batch
        batch_x = batch_graphs.ndata['f'].to(torch.float)
        batch_e = batch_graphs.edata['f'].to(torch.float)
        batch_scores, batch_class = self.forward(batch_graphs, batch_x, batch_e)

        # mse
        test_mse = self.criterion(batch_scores, batch_targets)
        self.log('test_mse', test_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        test_mse *= self.mse_weight

        # ce
        test_ce = F.cross_entropy(batch_scores, batch_targets_class.squeeze(1))
        self.log('test_ce', test_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        test_ce *= self.ce_weight

        # acc
        test_acc = self.criterion_acc(F.softmax(batch_class, dim=1), batch_targets_class.squeeze(1))

        # loss
        test_loss = test_mse + test_ce
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)

        return test_loss


# data
trainset = DProQAData(data_path=_dataset, mode='train')
valset = DProQAData(data_path=_dataset, mode='val')
testset = DProQAData(data_path=_dataset, mode='test')

train_loader = DataLoader(trainset.data,
                          batch_size=_batch_size,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=collate,
                          shuffle=True)

val_loader = DataLoader(valset.data,
                        batch_size=_batch_size,
                        num_workers=16,
                        pin_memory=True,
                        collate_fn=collate)


# init the model
model = DProQ()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)

# logger
wandb_logger = WandbLogger(project="DProQ",
                           name=_version,
                           id=_CURRENT_TIME,
                           offline=False,
                           log_model=False,
                           save_dir=_wb_out_dir)

# training
# define callbacks
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    patience=_early_stop_patience,
                                    verbose=True,
                                    mode="min")

lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(_ckpt_out_dir, _version),
    filename='{epoch}_{val:.5f}',
    monitor='val_loss',
    save_top_k=3,
    mode='min'
)

saw = StochasticWeightAveraging(swa_epoch_start=0.7,
                                swa_lrs=_init_lr * 0.1,
                                annealing_epochs=10,
                                annealing_strategy='cos')

# defile a trainer
trainer = pl.Trainer(
    amp_backend="native",
    gpus=_n_gpu,
    num_nodes=1,
    strategy='ddp',
    max_epochs=_epochs,
    logger=wandb_logger,
    callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, saw],
    sync_batchnorm=True,
    accumulate_grad_batches=_accumulate_grad_batches)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)
