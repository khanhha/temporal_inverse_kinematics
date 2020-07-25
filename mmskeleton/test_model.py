import torch
from pathlib import Path
from model_wrap import IKModelWrapper
from mmskeleton.datasets import AmassDataset
from common.smpl_util import load_smplx_models
import trimesh
import numpy as np


def _load_amass_path_list(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        lines = [Path(ll.replace('\n', '')) for ll in lines]
    return lines


if __name__ == "__main__":
    ckpt_path = Path('/media/F/datasets/amass/ik_model/model_ckps/checkpoint_epoch=31-val_loss=0.09.ckpt')
    amass_dir = Path('/media/F/datasets/amass/')
    data_dir = Path('/media/F/datasets/amass/ik_model')
    cache_dir = Path('/media/F/datasets/amass/tmp_debug')
    model = IKModelWrapper.load_from_checkpoint(str(ckpt_path))
    val_paths = _load_amass_path_list(Path(data_dir) / 'valid.csv')[2:3]
    smpl_x_dir = Path(amass_dir) / 'smplx'
    smplx_models = load_smplx_models(smpl_x_dir, 'cpu', 9)

    ds = AmassDataset(smplx_models=smplx_models, amass_paths=val_paths, smplx_gender=None,
                      window_size=model.hparams.win_size, keypoint_format='coco', cache_dir=cache_dir,
                      reset_cache=True, device='cpu', add_gaussian_noise=False)

    n = len(ds)
    in_data = ds[50]
    kps_3d = in_data['keypoints_3d']
    poses = in_data["poses"]
    betas = in_data["betas"]
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(kps_3d).unsqueeze(0))
        pred_poses = preds["poses"].squeeze(0)
        poses = torch.from_numpy(poses)
        betas = torch.from_numpy(betas)
        pred_poses = pred_poses.view(9, -1)
        smplx_model = smplx_models["male"]
        # prd_body = smplx_model(betas=None, global_orient=pred_poses[:, :3], body_pose=pred_poses[:, 3:66])
        # gt_body = smplx_model(betas=None, global_orient=poses[:, :3], body_pose=poses[:, 3:66])
        prd_body = smplx_model(betas=None, global_orient=None, body_pose=pred_poses[:, 3:66])
        gt_body = smplx_model(betas=None, global_orient=None, body_pose=poses[:, 3:66])

        t_idx = 4
        prd_mesh = trimesh.Trimesh(vertices=prd_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
                                   vertex_colors=np.tile((150, 150, 150), (10475, 1)))
        gt_mesh = trimesh.Trimesh(vertices=gt_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
                                  vertex_colors=np.tile((0, 90, 170), (10475, 1)))

        scn = trimesh.Scene()
        scn.add_geometry(prd_mesh)
        scn.add_geometry(gt_mesh)
        scn.show()
