from pathlib import Path
from typing import List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from smplx import create as smplx_create
from tqdm import tqdm
import os
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES
from common.draw_util import draw_3d_pose
from common.smpl_util import run_smpl_inference


def sample_window(arr, idx, h_win_size):
    """
    :param arr: NxJx...
    :param idx:
    :param h_win_size:
    :return:
    """
    pad_left, pad_right = 0, 0
    pads = [[0, 0] for _ in range(len(arr.shape))]
    if h_win_size > idx > arr.shape[0] - h_win_size:
        raise ValueError(f'h_win_size > idx > arr.shape[0] - h_win_size: '
                         f'{h_win_size} > {idx} > {arr.shape[0]} - {h_win_size}')
    elif idx < h_win_size:
        pad_left = h_win_size - idx
        pads[0][0] = pad_left
        arr = np.pad(arr, pads, 'edge')

    elif idx > arr.shape[0] - h_win_size - 1:
        pad_right = idx - (arr.shape[0] - h_win_size) + 1
        pads[0][1] = pad_right
        arr = np.pad(arr, pads, 'edge')

    win = arr[idx + pad_left - h_win_size:idx + pad_left + h_win_size + 1]
    # assert win.shape[0] == 2*h_win_size + 1, f'unexpected shape: {win.shape}. idx = {idx}. pad_right = {pad_right}'
    return win


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


def convert_smplx(smplx_kps, mappings, do_copy=False):
    """
    :param smplx_kps: BxJxC
    :param mappings: kps mapping from smplx to target format.
    :param do_copy:
    """
    n_kps = len(mappings)
    out_kps = np.zeros((smplx_kps.shape[0], n_kps, smplx_kps.shape[2]), dtype=np.float32)
    for target_idx, smplx_idx in enumerate(mappings):
        out_kps[:, target_idx, :] = smplx_kps[:, smplx_idx, :]
    return out_kps


class AmassDataset(Dataset):
    def __init__(self, smplx_models, smplx_gender: Optional[str], amass_paths: List, window_size: int, keypoint_format: str,
                 cache_dir: Path, reset_cache: bool, device='cuda'):
        self.smplx_gender = smplx_gender
        self.origin_amass_paths = amass_paths
        self.device = device
        self.smplx_models = smplx_models
        self.half_win_size = window_size // 2
        if keypoint_format == 'coco':
            self.target_kps_mapping = generate_smplx_to_coco_mappings(SMPLX_JOINT_NAMES)
        else:
            raise ValueError('unsupported keypoint format')

        self.data_dir = cache_dir
        if reset_cache:
            os.makedirs(str(self.data_dir), exist_ok=True)
            self.data_paths = self.generate_data()
        else:
            self.data_paths = self.list_data_paths(amass_paths)
        self.data_anims = []
        for dpath in self.data_paths:
            d = np.load(str(dpath), allow_pickle=True)
            d = {k: d[k].item() if d[k].dtype == object else d[k] for k, v in d.items()}
            self.data_anims.append(d)

        self.index_mappings = self.generate_index_file_mapping()

    def __len__(self):
        return len(self.index_mappings)

    def __getitem__(self, idx):
        data_idx, offset = self.index_mappings[idx]
        local_idx = idx - offset
        # data = np.load(str(self.data_paths[data_idx]), allow_pickle=True)
        data = self.data_anims[data_idx]

        keypoints_3d = sample_window(data["keypoints_3d"], local_idx, self.half_win_size)
        keypoints_3d = convert_smplx(keypoints_3d, self.target_kps_mapping, False)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        # ax.set_zlim(-2, 2)
        # draw_3d_pose(ax, keypoints_3d[0, :, :], 'coco')
        # plt.show(block=True)
        # fig.clear()
        # plt.clf()
        poses = sample_window(data["poses"], local_idx, self.half_win_size)
        betas = data["betas"]
        return {"keypoints_3d": keypoints_3d.astype(np.float32),
                "poses": poses[:, :66].astype(np.float32),
                "betas": betas.astype(np.float32)}

    def count_samples(self):
        apaths = sorted([apath for apath in self.data_dir.rglob('*.npz')])
        n_samples = 0
        for apath in apaths:
            data = np.load(str(apath))
            data = {key: data[key] for key in data.keys()}
            kps = data["keypoints_3d"]
            n_samples += kps.shape[0]
        return n_samples

    def generate_index_file_mapping(self):
        n_samples = 0
        for apath in self.data_paths:
            data = np.load(str(apath))
            data = {key: data[key] for key in data.keys()}
            kps = data["keypoints_3d"]
            n_samples += kps.shape[0]

        mappings = n_samples * [None]
        current_offset = 0
        for path_idx, apath in enumerate(self.data_paths):
            data = np.load(str(apath))
            kps = data["keypoints_3d"]
            new_offset = current_offset + kps.shape[0]
            for i in range(current_offset, new_offset):
                mappings[i] = (path_idx, current_offset)
            current_offset = new_offset
        return mappings

    def list_data_paths(self, amass_paths):
        hash_paths = {apath.stem for apath in amass_paths}
        return sorted([apath for apath in self.data_dir.rglob('*.npz') if apath.stem in hash_paths])

    def generate_data(self):
        data_paths = []
        for apath in tqdm(self.origin_amass_paths, 'regenerate epoch data'):
            data = np.load(str(apath))
            data = {key: data[key] for key in data.keys()}
            keypoints = run_smpl_inference(data, self.smplx_models, self.device, self.smplx_gender)
            data["keypoints_3d"] = keypoints
            dpath = self.data_dir / apath.name
            np.savez_compressed(str(dpath), **data)
            data_paths.append(dpath)
        return data_paths


def run_test():
    smplx_dir = Path('/media/F/datasets/amass/smplx')
    amss_dir = Path('/media/F/datasets/amass/motion_data/')
    post_process_dir = Path('/media/F/datasets/amass/motion_data/test_data')
    os.makedirs(post_process_dir, exist_ok=True)
    amss_paths = [apath for apath in amss_dir.rglob('*.npz') if apath.stem.endswith('_poses')]
    amss_paths = amss_paths[:2]
    ds = AmassDataset(smplx_dir=smplx_dir, amass_paths=amss_paths, window_size=7, keypoint_format='coco',
                      cache_dir=post_process_dir, reset_cache=True)
    print(ds[500])
    dl = DataLoader(ds, batch_size=16)
    for batch in dl:
        for k, v in batch.items():
            print(k, v.shape)


if __name__ == "__main__":
    run_test()
