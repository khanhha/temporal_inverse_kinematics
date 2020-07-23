import torch
from pathlib import Path
from mmskeleton.model_wrap import IKModelWrapper
from mmskeleton.datasets import AmassDataset
from smplx import create as smplx_create
import trimesh
import numpy as np


def _load_amass_path_list(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        lines = [Path(ll.replace('\n', '')) for ll in lines]
    return lines


def load_smplx_models(smplx_dir, device, batch_size):
    male_path = f'{smplx_dir}/SMPLX_MALE.npz'
    female_path = f'{smplx_dir}/SMPLX_FEMALE.npz'
    # use_pca: for hand pose parameter. smplx model is kind of a bit different from human_pose_prior. not sure why
    male_smplx = smplx_create(model_path=male_path, model_type='smplx', gender='male', use_pca=False,
                              use_face_contour=True, batch_size=batch_size).to(device)
    female_smplx = smplx_create(model_path=female_path, model_type='smplx', gender='female', use_pca=False,
                                use_face_contour=True, batch_size=batch_size).to(device)
    return {'male_smplx': male_smplx, 'female_smplx': female_smplx}


if __name__ == "__main__":
    ckpt_path = Path('/media/F/datasets/amass/ik_model/model_ckps/checkpoint_epoch=14-val_loss=0.10.ckpt')
    amass_dir = Path('/media/F/datasets/amass/')
    data_dir = Path('/media/F/datasets/amass/ik_model')
    cache_dir = Path('/media/F/datasets/amass/tmp_debug')
    model = IKModelWrapper.load_from_checkpoint(str(ckpt_path))
    val_paths = _load_amass_path_list(Path(data_dir) / 'train.csv')[:1]
    smpl_x_dir = Path(amass_dir) / 'smplx'
    smplx_models = load_smplx_models(smpl_x_dir, 'cpu', 9)

    ds = AmassDataset(smplx_dir=smpl_x_dir, amass_paths=val_paths,
                      window_size=model.hparams.win_size, keypoint_format='coco', cache_dir=cache_dir,
                      reset_cache=True)

    n = len(ds)
    in_data = ds[0]
    kps_3d = in_data['keypoints_3d']
    poses = in_data["poses"]
    betas = in_data["betas"]
    preds = model(torch.from_numpy(kps_3d).unsqueeze(0))
    pred_poses = preds["poses"].squeeze(0)
    poses = torch.from_numpy(poses)
    betas = torch.from_numpy(betas)
    pred_poses = pred_poses.view(9, -1)
    smplx_model = smplx_models["male_smplx"]
    # prd_body = smplx_model(betas=None, global_orient=pred_poses[:, :3], body_pose=pred_poses[:, 3:66])
    # gt_body = smplx_model(betas=None, global_orient=poses[:, :3], body_pose=poses[:, 3:66])
    prd_body = smplx_model(betas=None, global_orient=None, body_pose=pred_poses[:, 3:66])
    gt_body = smplx_model(betas=None, global_orient=None, body_pose=poses[:, 3:66])

    t_idx = 5
    prd_mesh = trimesh.Trimesh(vertices=prd_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
                               vertex_colors=np.tile((150, 150, 150), (10475, 1)))
    gt_mesh = trimesh.Trimesh(vertices=gt_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
                              vertex_colors=np.tile((0, 90, 170), (10475, 1)))

    scn = trimesh.Scene()
    scn.add_geometry(prd_mesh)
    scn.add_geometry(gt_mesh)
    scn.show()
