from typing import List
import numpy as np


def generate_smplx_to_coco_mappings(smplx_kps_names: List[str]):
    mappings = 17 * [0]
    mappings[0] = smplx_kps_names.index('nose')
    mappings[1] = smplx_kps_names.index('left_eye')
    mappings[2] = smplx_kps_names.index('right_eye')
    mappings[3] = smplx_kps_names.index('left_ear')
    mappings[4] = smplx_kps_names.index('right_ear')
    mappings[5] = smplx_kps_names.index('left_shoulder')
    mappings[6] = smplx_kps_names.index('right_shoulder')
    mappings[7] = smplx_kps_names.index('left_elbow')
    mappings[8] = smplx_kps_names.index('right_elbow')
    mappings[9] = smplx_kps_names.index('left_wrist')
    mappings[10] = smplx_kps_names.index('right_wrist')
    mappings[11] = smplx_kps_names.index('left_hip')
    mappings[12] = smplx_kps_names.index('right_hip')
    mappings[13] = smplx_kps_names.index('left_knee')
    mappings[14] = smplx_kps_names.index('right_knee')
    mappings[15] = smplx_kps_names.index('left_ankle')
    mappings[16] = smplx_kps_names.index('right_ankle')
    return mappings


def generate_moveai3d_to_coco_mappings(mvai_3d_joint_names):
    mappings = 17 * [0]
    mappings[0] = -1
    mappings[1] = -1
    mappings[2] = -1
    mappings[3] = mvai_3d_joint_names.index('L_Ear')
    mappings[4] = mvai_3d_joint_names.index('R_Ear')
    mappings[5] = mvai_3d_joint_names.index('L_Shoulder')
    mappings[6] = mvai_3d_joint_names.index('R_Shoulder')
    mappings[7] = mvai_3d_joint_names.index('L_Elbow')
    mappings[8] = mvai_3d_joint_names.index('R_Elbow')
    mappings[9] = mvai_3d_joint_names.index('L_Wrist')
    mappings[10] = mvai_3d_joint_names.index('R_Wrist')
    mappings[11] = mvai_3d_joint_names.index('L_Hip')
    mappings[12] = mvai_3d_joint_names.index('R_Hip')
    mappings[13] = mvai_3d_joint_names.index('L_Knee')
    mappings[14] = mvai_3d_joint_names.index('R_Knee')
    mappings[15] = mvai_3d_joint_names.index('L_Ankle')
    mappings[16] = mvai_3d_joint_names.index('R_Ankle')
    return mappings


def convert_seq_keypoints(in_seq_kps, mappings, do_copy=False):
    """
    :param in_seq_kps: BxJxC
    :param mappings: kps mapping from smplx to target format.
    :param do_copy:
    """
    n_kps = len(mappings)
    out_kps = np.zeros((in_seq_kps.shape[0], n_kps, in_seq_kps.shape[2]), dtype=np.float32)
    for target_idx, smplx_idx in enumerate(mappings):
        if smplx_idx >= 0:
            out_kps[:, target_idx, :] = in_seq_kps[:, smplx_idx, :].copy() if do_copy else in_seq_kps[:, smplx_idx, :]
    return out_kps
