from common.mesh_viewer import MeshViewer
from pathlib import Path
import numpy as np
from typing import Optional
from common.smpl_util import load_smplx_models
from common.kornia_geometry_conversion import angle_axis_to_quaternion, quaternion_to_angle_axis
import torch
import trimesh
from scipy.spatial import transform
from imageio import get_writer
from tqdm import tqdm


def my_run_smpl_inference(data, smplx_models, device, gender: Optional[str]):
    if gender is None:
        gender = str(data["gender"])
    smplx_model = smplx_models["male"] if 'male' in gender else smplx_models["female"]
    batch_size = smplx_model.batch_size
    frm_poses = data["poses"].astype(np.float32)
    frm_trans = data["trans"].astype(np.float32)
    n_poses = frm_poses.shape[0]
    n_batch = (n_poses // batch_size) + 1
    bodies = []
    for i in range(n_batch):
        s = i * batch_size
        e = (i + 1) * batch_size
        if s >= n_poses:
            break
        poses = frm_poses[s:e, :]
        trans = frm_trans[s:e, :]
        org_bsize = poses.shape[0]
        pad = 0
        # print(f'n batch = {n_batch}. batch from {s} to {e}. cur_batch_size = {org_bsize}')
        if org_bsize < batch_size:
            # padding because smplx_model require fixed batch size
            pad = batch_size - org_bsize
            poses = np.concatenate([poses, np.zeros((pad, poses.shape[1]), dtype=np.float32)], axis=0)
            trans = np.concatenate([trans, np.zeros((pad, trans.shape[1]), dtype=np.float32)], axis=0)

        poses = torch.from_numpy(poses).to(device)
        trans = torch.from_numpy(trans).to(device)
        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
        left_pose_hand = poses[:, 66:66 + 45]
        right_pose_hand = poses[:, 66 + 45:66 + 90]

        # print(root_orient.shape, pose_body.shape, left_pose_hand.shape, right_pose_hand.shape, trans.shape)
        body = smplx_model(global_orient=root_orient, body_pose=pose_body,
                           left_hand_pose=left_pose_hand, right_hand_pose=right_pose_hand,
                           transl=None)
        bodies.append(body)

    return bodies


copy2cpu = lambda tensor: tensor.detach().cpu().numpy()

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}


def run_main():
    amass_dir = Path('/media/F/datasets/amass/')
    smpl_x_dir = Path(amass_dir) / 'smplx'
    device = 'cuda'
    smplx_models = load_smplx_models(smpl_x_dir, device, 9)

    apaths = [ap for ap in (amass_dir / 'motion_data').rglob("*.npz") if
              ap.stem.endswith('_poses')]
    for ap in apaths:
        if '10_02' in ap.stem:
            apath = ap
            break

    data = np.load(str(apath))
    data = {key: data[key] for key in data.keys()}

    poses = data["poses"]
    aug_angle = np.pi  # augmentation angle
    aug_axis = [0.0, 0.0, 1.0]  # augmentation axis
    org_rots = transform.Rotation.from_rotvec(poses[:, :3])
    aug_rot = transform.Rotation.from_rotvec(np.array(aug_axis) * aug_angle)
    new_rots = aug_rot * org_rots
    new_aa = new_rots.as_rotvec()
    data["poses"][:, :3] = new_aa

    bodies = my_run_smpl_inference(data, smplx_models, device, gender=None)

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 5.0])
    mv.update_camera_pose(camera_pose)

    images = []
    for bd_idx, bd in tqdm(enumerate(bodies)):
        v = bd.vertices.detach().cpu()
        mesh = trimesh.Trimesh(vertices=v[0].numpy(),
                               faces=smplx_models["male"].faces,
                               vertex_colors=np.tile((0, 90, 170), (10475, 1)))

        mv.set_static_meshes([mesh])
        images.append(mv.render())

    out_viz = f'/media/F/datasets/amass/debug_viz/{apath.stem}.mp4'
    vwriter = get_writer(out_viz)
    for img in images:
        vwriter.append_data(img)
    vwriter.close()


if __name__ == "__main__":
    run_main()
