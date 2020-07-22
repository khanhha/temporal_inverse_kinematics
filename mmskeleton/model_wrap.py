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


def calc_mean_losses(outputs, loss_keys):
    loss_means = {key: 0.0 for key in loss_keys}
    loss_cnts = {key: 0 for key in loss_keys}
    for output in outputs:
        for key in loss_keys:
            if key in output:
                loss_means[key] += output[key]
                loss_cnts[key] += 1

    for key in loss_keys:
        if loss_cnts[key] > 0:
            loss_means[key] /= loss_cnts[key]
    return {key: loss for key, loss in loss_means.items() if loss_cnts[key] > 0}


class IKLoss(nn.Module):
    def __init__(self, device):
        super(IKLoss, self).__init__()
        self.device = device
        self.criterion_pose = nn.MSELoss().to(self.device)

    def forward(self, outputs, data_3d):
        gt_poses = outputs["poses"]
        pred_poses = data_3d["poses"]
        pred_poses = pred_poses.view(pred_poses.shape[0], pred_poses.shape[1], -1, 3)
        pose_loss = self.criterion_pose(gt_poses, pred_poses)
        return pose_loss


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

        self.criterion = IKLoss('cuda')

    def forward(self, keypoints_3d):
        preds = self.ik_model(keypoints_3d)
        return {"poses": preds}

    def training_step(self, batch, batch_idx):
        poses_3d = batch["keypoints_3d"].cuda()
        preds = self.forward(poses_3d)
        pose_mse = self.criterion(preds, {"keypoints_3d": poses_3d, "poses": batch["poses"]})

        losses = {"pose_mse": pose_mse}
        loss_log = {f'train/{loss_name}': loss for loss_name, loss in losses.items()}

        return {"loss": pose_mse, 'log': loss_log}

    def validation_step(self, batch, batch_idx):
        poses_3d = batch["keypoints_3d"].cuda()
        preds = self.forward(poses_3d)
        lss = self.criterion(preds, {"keypoints_3d": poses_3d, "poses": batch["poses"]})
        return {"val_loss": lss, "pose_mse": lss}

    def validation_epoch_end(self, outputs):
        losses = calc_mean_losses(outputs, ['pose_mse', 'val_loss'])
        val_log = {
            'val_loss': losses['val_loss'],
            'log': {f'val/{loss_name}': loss for loss_name, loss in losses.items()}
        }
        return val_log

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        train_paths = _load_amass_path_list(Path(self.hparams.data_dir) / 'train.csv')[:100]
        smpl_x_dir = Path(self.hparams.amass) / 'smplx'
        cache_dir = Path(self.hparams.data_dir) / 'train_epoch_data'
        ds = AmassDataset(smplx_dir=smpl_x_dir, amass_paths=train_paths, window_size=self.hparams.win_size,
                          keypoint_format=self.hparams.keypoint_format,
                          cache_dir=cache_dir, reset_cache=self.hparams.regen_data)
        return DataLoader(ds, batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.n_workers)

    def val_dataloader(self):
        val_paths = _load_amass_path_list(Path(self.hparams.data_dir) / 'valid.csv')[:50]
        smpl_x_dir = Path(self.hparams.amass) / 'smplx'
        cache_dir = Path(self.hparams.data_dir) / 'valid_epoch_data'
        ds = AmassDataset(smplx_dir=smpl_x_dir, amass_paths=val_paths, window_size=self.hparams.win_size,
                          keypoint_format=self.hparams.keypoint_format,
                          cache_dir=cache_dir, reset_cache=self.hparams.regen_data)
        return [DataLoader(ds, batch_size=self.hparams.bs, shuffle=False, num_workers=self.hparams.n_workers)]

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
        parser.add_argument('--n_out_joints', default=22, type=int, help='number of output joints')
        parser.add_argument('--n_out_channels', default=3, type=int, help='number of output channels')
        return parser


def add_args(parser):
    parser.add_argument('--amass', type=str, default='/media/F/datasets/amass', help='amass dir')
    parser.add_argument('--data_dir', type=str, default='/media/F/datasets/amass/ik_model', help='data dir')
    parser.add_argument('--n_workers', type=int, default=8, help='data dir')
    parser.add_argument('--regen_data', action='store_true', help='data dir')


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
