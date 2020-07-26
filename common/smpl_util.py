import numpy as np
import torch
from smplx import create as smplx_create
from typing import Optional


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


def run_smpl_inference(data, smplx_models, device, gender: Optional[str], apply_trans=True, apply_root_rot=True):
    if gender is None:
        gender = str(data["gender"])
    smplx_model = smplx_models["male"] if 'male' in gender else smplx_models["female"]
    batch_size = smplx_model.batch_size
    frm_poses = data["poses"].astype(np.float32)
    frm_trans = data["trans"].astype(np.float32)
    n_poses = frm_poses.shape[0]
    frm_joints = []
    n_batch = (n_poses // batch_size) + 1
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
        trans = torch.from_numpy(trans).to(device) if apply_trans else None
        root_orient = poses[:, :3] if apply_root_rot else None
        pose_body = poses[:, 3:66]
        left_pose_hand = poses[:, 66:66 + 45]
        right_pose_hand = poses[:, 66 + 45:66 + 90]

        # print(root_orient.shape, pose_body.shape, left_pose_hand.shape, right_pose_hand.shape, trans.shape)
        body = smplx_model(global_orient=root_orient, body_pose=pose_body,
                           left_hand_pose=left_pose_hand, right_hand_pose=right_pose_hand,
                           transl=trans)
        joints = body.joints.detach().cpu().numpy()
        if pad > 0:
            joints = joints[:org_bsize]
        frm_joints.append(joints)

    frm_joints = np.concatenate(frm_joints, axis=0)

    return frm_joints
