from pathlib import Path
from typing import List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from smplx import create as smplx_create
from tqdm import tqdm
import os
from scipy.spatial import transform
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
        out_kps[:, target_idx, :] = smplx_kps[:, smplx_idx, :].copy() if do_copy else smplx_kps[:, smplx_idx, :]
    return out_kps


def coco_kps_sigma():
    # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L498
    predef_signal_arr = np.array(
        [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], dtype=np.float32) * 0.1
    return predef_signal_arr


def _aug_3d_keypoints(anim_poses_3d, kps_sigmas):
    """
    :param anim_poses_3d: NxJx3
    """
    n_poses, n_kps = anim_poses_3d.shape[:2]

    poses_bmin = np.min(anim_poses_3d, axis=1)
    poses_bmax = np.max(anim_poses_3d, axis=1)
    sizes = poses_bmax - poses_bmin
    mean_size = np.mean(sizes, axis=0)
    # scale sigmal down a bit => original sigma from coco amin_ds is too big
    px_kps_sigmas = np.array([s * kps_sigmas * 0.003 for s in mean_size]).T
    # each keypoint has its own noise distribution
    gaus_mean = (0.0, 0.0, 0.0)
    for j, sigma in enumerate(px_kps_sigmas):
        cov = ((sigma[0], 0.0, 0.0), (0.0, sigma[1], 0.0), (0.0, 0.0, sigma[2]))
        kps_noise = np.random.multivariate_normal(gaus_mean, cov, n_poses)
        anim_poses_3d[:, j] = anim_poses_3d[:, j] + kps_noise.astype(np.float32)

    return anim_poses_3d


class AmassDataset(Dataset):
    def __init__(self, smplx_models, smplx_gender: Optional[str], amass_paths: List, window_size: int,
                 keypoint_format: str, cache_dir: Path, reset_cache: bool, device='cuda', add_gaussian_noise=True):
        self.smplx_gender = smplx_gender
        self.origin_amass_paths = amass_paths
        self.device = device
        self.smplx_models = smplx_models
        self.half_win_size = window_size // 2
        self.relative_pose = True
        self.add_gaussian_noise = True
        self.kps_noise_sigmas = coco_kps_sigma()
        self.aug_root_orientation = True
        if keypoint_format == 'coco':
            self.target_kps_mapping = generate_smplx_to_coco_mappings(SMPLX_JOINT_NAMES)
        else:
            raise ValueError('unsupported keypoint format')

        self.data_dir = cache_dir
        os.makedirs(str(self.data_dir), exist_ok=True)
        self.data_paths = []
        self.data_anims = []
        self.index_mappings = []
        self.prepare_epoch_training_data(0)

    def prepare_epoch_training_data(self, epoch_idx):
        for apath in self.data_dir.glob('*.npz'):
            os.remove(str(apath))
        self.data_paths = self.regenerate_data(epoch_idx)
        self.data_anims = []
        for dpath in self.data_paths:
            d = np.load(str(dpath), allow_pickle=True)
            d = {k: d[k].item() if d[k].dtype == object else d[k] for k, v in d.items()}
            self.data_anims.append(d)
        self.index_mappings = self.generate_index_file_mapping()

    def __len__(self):
        return len(self.index_mappings)

    def on_epoch_end(self, epoch_idx):
        """
        should be called when epoch ended for data augmentation
        """
        self.prepare_epoch_training_data(epoch_idx)

    def __getitem__(self, idx):
        data_idx, offset = self.index_mappings[idx]
        local_idx = idx - offset
        # data = np.load(str(self.data_paths[data_idx]), allow_pickle=True)
        data = self.data_anims[data_idx]

        keypoints_3d = sample_window(data["keypoints_3d"], local_idx, self.half_win_size)
        keypoints_3d = convert_smplx(keypoints_3d, self.target_kps_mapping, True)
        if self.relative_pose:
            roots = 0.5 * (keypoints_3d[:, 11, :] + keypoints_3d[:, 12, :])
            keypoints_3d = keypoints_3d - roots[:, np.newaxis, :]

        if self.add_gaussian_noise:
            keypoints_3d = _aug_3d_keypoints(keypoints_3d, self.kps_noise_sigmas)

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
        apaths = sorted([apath for apath in self.data_dir.glob('*.npz')])
        n_samples = 0
        for apath in apaths:
            data = np.load(str(apath))
            data = {key: data[key] for key in data.keys()}
            kps = data["keypoints_3d"]
            n_samples += kps.shape[0]
        return n_samples

    def generate_index_file_mapping(self):
        n_samples = self.count_samples()

        mappings = n_samples * [None]
        current_offset = 0
        for path_idx, apath in enumerate(self.data_paths):
            data = np.load(str(apath))
            kps = data["keypoints_3d"]
            new_offset = current_offset + kps.shape[0]
            for i in range(current_offset, new_offset):
                mappings[i] = (path_idx, current_offset)
            current_offset = new_offset
        assert all([m[1] >= 0 for m in mappings]), 'invalid mapping index'
        return mappings

    def regenerate_data(self, random_seed):
        data_paths = []
        rand_stt = np.random.RandomState(seed=random_seed)
        for apath in tqdm(self.origin_amass_paths, 'regenerate epoch data'):
            data = np.load(str(apath))
            data = {key: data[key] for key in data.keys()}
            if self.aug_root_orientation:
                aug_angle = np.pi * rand_stt.rand()
                org_rots = transform.Rotation.from_rotvec(data["poses"][:, :3])
                # randomly rotate around z axis
                aug_rot = transform.Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * aug_angle)
                new_rots = aug_rot * org_rots
                data["poses"][:, :3] = new_rots.as_rotvec()

                # we don't care about global translation
            keypoints = run_smpl_inference(data, self.smplx_models, self.device, self.smplx_gender,
                                           apply_trans=False,
                                           apply_root_rot=True)
            data["keypoints_3d"] = keypoints
            dpath = self.data_dir / apath.name
            np.savez_compressed(str(dpath), **data)
            data_paths.append(dpath)
        return data_paths


def run_test():
    from common.smpl_util import load_smplx_models
    smplx_dir = Path('/media/F/datasets/amass/smplx')
    amss_dir = Path('/media/F/datasets/amass/motion_data/')
    post_process_dir = Path('/media/F/datasets/amass/motion_data/test_data')
    os.makedirs(post_process_dir, exist_ok=True)
    amss_paths = [apath for apath in amss_dir.rglob('*.npz') if apath.stem.endswith('_poses')]
    amss_paths = amss_paths[:1]
    device = 'cpu'
    smplx_models = load_smplx_models(smplx_dir, device, 9)
    ds = AmassDataset(smplx_models=smplx_models, smplx_gender='neutral', amass_paths=amss_paths, window_size=9,
                      keypoint_format='coco',
                      cache_dir=post_process_dir, reset_cache=True, device=device)
    print(ds[500])
    dl = DataLoader(ds, batch_size=16)
    for batch in dl:
        for k, v in batch.items():
            print(k, v.shape)


if __name__ == "__main__":
    run_test()
