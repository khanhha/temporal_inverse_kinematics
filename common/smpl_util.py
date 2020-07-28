import numpy as np
import torch
from smplx import create as smplx_create
from typing import Optional
from tqdm import tqdm


def load_smplx_models(smplx_dir, device, batch_size):
    male_path = f'{smplx_dir}/SMPLX_MALE.npz'
    female_path = f'{smplx_dir}/SMPLX_FEMALE.npz'
    neutral_path = f'{smplx_dir}/SMPLX_NEUTRAL.npz'
    # use_pca: for hand pose parameter. smplx model is kind of a bit different from human_pose_prior. not sure why
    male_smplx = smplx_create(model_path=male_path, model_type='smplx', gender='male', use_pca=False,
                              use_face_contour=True, batch_size=batch_size).to(device)
    female_smplx = smplx_create(model_path=female_path, model_type='smplx', gender='female', use_pca=False,
                                use_face_contour=True, batch_size=batch_size).to(device)
    neutral_smplx = smplx_create(model_path=neutral_path, model_type='smplx', gender='neutral', use_pca=False,
                                 use_face_contour=True, batch_size=batch_size).to(device)
    return {'male': male_smplx, 'female': female_smplx, 'neutral': neutral_smplx}


def run_smpl_inference(data, smplx_models, device,
                       apply_trans=True,
                       apply_root_rot=True,
                       apply_shape=True,
                       return_mesh=False):
    smplx_model = smplx_models[str(data["gender"])]
    batch_size = smplx_model.batch_size
    frm_poses = data["poses"].astype(np.float32)
    n_poses = frm_poses.shape[0]

    frm_trans = data["trans"].astype(np.float32) if apply_trans else None
    beta = data["betas"][:10][np.newaxis, :] if apply_shape else None
    frm_betas = np.tile(beta, (n_poses, 1)).astype(np.float32) if apply_shape else None
    frm_joints = []
    n_batch = (n_poses // batch_size) + 1
    meshes = []
    for i in tqdm(range(n_batch), desc=f'run_smpl_inference'):
        s = i * batch_size
        e = (i + 1) * batch_size
        if s >= n_poses:
            break
        poses = frm_poses[s:e, :]
        trans = frm_trans[s:e, :] if apply_trans else None
        betas = frm_betas[s:e, :] if apply_shape else None
        org_bsize = poses.shape[0]
        pad = 0
        # print(f'n batch = {n_batch}. batch from {s} to {e}. cur_batch_size = {org_bsize}')
        if org_bsize < batch_size:
            # padding because smplx_model require fixed batch size
            pad = batch_size - org_bsize
            poses = np.concatenate([poses, np.zeros((pad, poses.shape[1]), dtype=np.float32)], axis=0)
            trans = np.concatenate([trans, np.zeros((pad, trans.shape[1]), dtype=np.float32)],
                                   axis=0) if apply_trans else None
            betas = np.concatenate([betas, np.zeros((pad, betas.shape[1]), dtype=np.float32)],
                                   axis=0) if apply_shape else None

        poses = torch.from_numpy(poses).to(device)
        trans = torch.from_numpy(trans).to(device) if apply_trans else None
        betas = torch.from_numpy(betas).to(device) if apply_shape else None
        root_orient = poses[:, :3] if apply_root_rot else None
        pose_body = poses[:, 3:66]
        left_pose_hand = poses[:, 66:66 + 45]
        right_pose_hand = poses[:, 66 + 45:66 + 90]

        # print(root_orient.shape, pose_body.shape, left_pose_hand.shape, right_pose_hand.shape, trans.shape)
        body = smplx_model(global_orient=root_orient, body_pose=pose_body, betas=betas,
                           left_hand_pose=left_pose_hand, right_hand_pose=right_pose_hand,
                           transl=trans)
        joints = body.joints.detach().cpu().numpy()
        if pad > 0:
            joints = joints[:org_bsize]
        frm_joints.append(joints)
        if return_mesh:
            meshes.append(body.vertices.detach().cpu().numpy())

    frm_joints = np.concatenate(frm_joints, axis=0)
    if return_mesh:
        meshes = np.concatenate(meshes, axis=0)
        return frm_joints, meshes
    else:
        return frm_joints
