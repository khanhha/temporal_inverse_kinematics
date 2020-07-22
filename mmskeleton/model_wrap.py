import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
import pytorch_lightning as pl
from mmskeleton.datasets import AmassDataset
from pathlib import Path
from mmskeleton.models import ST_GCN_18


def _load_amass_path_list(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        lines = [Path(ll.replace('\n', '')) for ll in lines]
    return lines


class IKModelWrapper(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.graph_cfg = dict(layout=self.hparams.graph_layout,
                              strategy='uniform',
                              max_hop=self.hparams.max_hop,
                              dilation=self.hparams.dilation)

        self.ik_model = ST_GCN_18(in_channels=self.hparams.kps_channel, graph_cfg=self.graph_cfg,
                                  n_out_joints=self.hparams.n_out_joints, n_out_channels=self.hparams.n_out_channels)

    def forward(self, keypoints_3d):
        preds = self.ik_model(keypoints_3d)
        return preds

    def training_step(self, batch, batch_idx):
        poses_3d = batch["keypoints_3d"].cuda()
        preds = self.forward(poses_3d)
        return preds

    def validation_step(self, batch):
        poses_3d = batch["keypoints_3d"].cuda()
        preds = self.forward(poses_3d)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        train_paths = _load_amass_path_list(Path(self.hparams.data_dir) / 'train.csv')[:2]
        smpl_x_dir = Path(self.hparams.amass) / 'smplx'
        cache_dir = Path(self.hparams.data_dir) / 'train_epoch_data'
        ds = AmassDataset(smplx_dir=smpl_x_dir, amass_paths=train_paths, window_size=self.hparams.win_size,
                          keypoint_format=self.hparams.keypoint_format,
                          cache_dir=cache_dir, reset_cache=True)
        return DataLoader(ds, batch_size=self.hparams.bs, shuffle=True)

    def val_dataloader(self):
        val_paths = _load_amass_path_list(Path(self.hparams.data_dir) / 'valid.csv')[:2]
        smpl_x_dir = Path(self.hparams.amass) / 'smplx'
        cache_dir = Path(self.hparams.data_dir) / 'valid_epoch_data'
        ds = AmassDataset(smplx_dir=smpl_x_dir, amass_paths=val_paths, window_size=self.hparams.win_size,
                          keypoint_format=self.hparams.keypoint_format, cache_dir=cache_dir, reset_cache=True)
        return DataLoader(ds, batch_size=self.hparams.bs, shuffle=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--win_size', default=9, type=int, help='batch size in terms of predicted frames')
        parser.add_argument('--bs', default=256, type=int, help='batch size in terms of predicted frames')
        parser.add_argument('--kps_channel', default=3, type=int, help='keypoint channel')
        parser.add_argument('--graph_layout', default='coco', type=str,
                            choices=['coco', 'openpose'], help='skeleton graph layout')
        parser.add_argument('--max_hop', default=2, type=int, help='graph max hop')
        parser.add_argument('--dilation', default=1, type=int, help='conv dilation')
        parser.add_argument('--keypoint_format', default='coco', type=str, help='input model keypoint format')
        parser.add_argument('--n_out_joints', default=24, type=int, help='number of output joints')
        parser.add_argument('--n_out_channels', default=3, type=int, help='number of output channels')
        return parser


def add_args(parser):
    parser.add_argument('--amass', type=str, default='/media/F/datasets/amass', help='amass dir')
    parser.add_argument('--data_dir', type=str, default='/media/F/datasets/amass/ik_model', help='data dir')


def run_train():
    parser = argparse.ArgumentParser()
    add_args(parser)
    parser = IKModelWrapper.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    ckp_dir = f'{hparams.data_dir}' + '/model_ckps/checkpoint_{epoch}-{val_loss:.2f}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=ckp_dir,
        save_top_k=30,
        verbose=True)

    print('start new training with new model weights')
    model = IKModelWrapper(hparams)
    trainer = pl.Trainer(gpus=1, fast_dev_run=False, num_sanity_val_steps=0, benchmark=False,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    run_train()
