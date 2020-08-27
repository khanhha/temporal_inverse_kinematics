from pathlib import Path
from typing import List, Optional
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from smplx import create as smplx_create
from tqdm import tqdm
import os
from scipy.spatial import transform
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES
from common.draw_util import draw_3d_pose
from common.smpl_util import run_smpl_inference
from common.keypoints_util import generate_smplx_to_coco_mappings


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
    def __init__(self, smplx_models, amass_paths: List, window_size: int,
                 keypoint_format: str, device='cuda', add_gaussian_noise=True, shape_db_path: Path = None,
                 aug_shape=True, aug_root_orientation=True):
        self.origin_amass_paths = amass_paths
        self.device = device
        self.smplx_models = smplx_models
        self.half_win_size = window_size // 2
        self.relative_pose = True
        self.add_gaussian_noise = add_gaussian_noise
        self.kps_noise_sigmas = coco_kps_sigma()
        self.aug_root_orientation = aug_root_orientation
        self.aug_shape = aug_shape
        self.smplx_shape_db = np.load(str(shape_db_path), allow_pickle=True)[
            "shapes"] if shape_db_path is not None else None
        if keypoint_format == 'coco':
            self.target_kps_mapping = generate_smplx_to_coco_mappings(SMPLX_JOINT_NAMES)
        else:
            raise ValueError('unsupported keypoint format')

        self.data_paths = []
        self.data_anims = []
        self.index_mappings = []
        self.prepare_epoch_training_data(0)

    def prepare_epoch_training_data(self, epoch_idx):
        self.data_anims = self.regenerate_data(epoch_idx)
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
                "poses": poses[-1, :66].astype(np.float32),
                "betas": betas.astype(np.float32)}

    def count_samples(self):
        n_samples = 0
        for data in self.data_anims:
            kps = data["keypoints_3d"]
            n_samples += kps.shape[0]
        return n_samples

    def generate_index_file_mapping(self):
        n_samples = self.count_samples()

        mappings = n_samples * [None]
        current_offset = 0
        for data_idx, data in enumerate(self.data_anims):
            new_offset = current_offset + data["poses"].shape[0]
            for i in range(current_offset, new_offset):
                mappings[i] = (data_idx, current_offset)
            current_offset = new_offset
        assert all([m[1] >= 0 for m in mappings]), 'invalid mapping index'
        return mappings

    def regenerate_data(self, random_seed):
        data_s = []
        rand_stt = np.random.RandomState(seed=random_seed)
        shape_rand_stt = np.random.RandomState(seed=random_seed)
        for apath in tqdm(self.origin_amass_paths, 'regenerate epoch data'):
            data = np.load(str(apath))
            data = {k: data[k].item() if data[k].dtype == object else data[k] for k, v in data.items()}

            if self.aug_root_orientation:
                aug_angle = 2.0 * np.pi * rand_stt.rand()
                org_rots = transform.Rotation.from_rotvec(data["poses"][:, :3])
                # randomly rotate around z axis
                aug_rot = transform.Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * aug_angle)
                new_rots = aug_rot * org_rots
                data["poses"][:, :3] = new_rots.as_rotvec()

            if self.aug_shape and self.smplx_shape_db is not None:
                # sample a random shape from the shape database
                shape_idx = int(shape_rand_stt.randint(0, len(self.smplx_shape_db), 1))
                beta_gender = self.smplx_shape_db[shape_idx]
                beta, gender = beta_gender[0], str(beta_gender[1])
                # for avoiding b'female'. not sure why.
                if 'female' in gender:
                    gender = 'female'
                elif 'male' in gender:
                    gender = 'male'
                else:
                    gender = 'neutral'
                # apply some variations to the sampled shape
                aug_beta = beta + 0.4 * np.random.rand() * beta
                # replace the shape and gender from the motion
                data["betas"] = aug_beta.astype(np.float32)
                data["gender"] = gender

            data["betas"] = data["betas"].astype(np.float32)

            # we don't care about global translation
            keypoints = run_smpl_inference(data, self.smplx_models, self.device,
                                           apply_trans=False,
                                           apply_root_rot=True)
            data["keypoints_3d"] = keypoints
            data_s.append(data)
        return data_s


class InferenceDataset(Dataset):
    def __init__(self, input_3d_poses: np.ndarray, win_size: int, relative_pose=True):
        self.poses_3d = input_3d_poses
        self.half_win_size = win_size // 2
        self.relative_pose = relative_pose

    def __len__(self):
        return self.poses_3d.shape[0]

    def __getitem__(self, idx):
        win_3d_poses = sample_window(self.poses_3d, idx, self.half_win_size)
        if self.relative_pose:
            # root point of COCO format
            roots = 0.5 * (win_3d_poses[:, 11, :] + win_3d_poses[:, 12, :])
            win_3d_poses = win_3d_poses - roots[:, np.newaxis, :]
        return win_3d_poses, idx


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
    ds = AmassDataset(smplx_models=smplx_models, amass_paths=amss_paths, window_size=9,
                      keypoint_format='coco', device=device)
    print(ds[500])
    dl = DataLoader(ds, batch_size=16)
    for batch in dl:
        for k, v in batch.items():
            print(k, v.shape)


if __name__ == "__main__":
    run_test()
