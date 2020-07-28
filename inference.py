import torch
from torch.utils.data import DataLoader
from pathlib import Path
from model_wrap import IKModelWrapper
from mmskeleton.datasets import AmassDataset
from common.smpl_util import load_smplx_models, run_smpl_inference
from common.mesh_viewer import MeshViewer
from common.sphere import points_to_spheres
from common.colors_def import colors
from common.keypoints_util import (generate_smplx_to_coco_mappings, convert_seq_keypoints,
                                   generate_moveai3d_to_coco_mappings)
from smplx.joint_names import JOINT_NAMES
from mmskeleton.datasets.data_amass import InferenceDataset
import trimesh
from tqdm import tqdm
import numpy as np
from imageio import get_writer


def _load_amass_path_list(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        lines = [Path(ll.replace('\n', '')) for ll in lines]
    return lines


def apply_mesh_tranfsormations_(meshes, transf):
    """
    apply inplace translations to meshes
    :param meshes: list of trimesh meshes
    :param transf:
    :return:
    """
    for i in range(len(meshes)):
        meshes[i] = meshes[i].apply_transform(transf)


def run_inference(model, seq_3d_kps):
    """
    :param model:
    :param seq_3d_kps: NxJx3
    """
    ds = InferenceDataset(seq_3d_kps, win_size=model.hparams.win_size, relative_pose=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    ret_batch_idxs = []
    ret_batch_preds = []
    with tqdm(total=len(dl), desc="run_inference") as bar:
        for dl_idx, batch in enumerate(dl):
            bar.update()
            in_poses = batch[0].to(model.device)
            batch_preds = model(in_poses)
            batch_preds = batch_preds["poses"].detach().cpu().numpy()
            ret_batch_idxs.append(batch[1].numpy())
            ret_batch_preds.append(batch_preds)

    seq_len = len(ds)
    win_prds = [[] for _ in range(seq_len)]
    h_w_size = model.hparams.win_size // 2
    for b_idxs, b_preds in zip(ret_batch_idxs, ret_batch_preds):
        for idx, preds in zip(b_idxs, b_preds):
            for offset in range(-h_w_size, h_w_size + 1):
                frm_idx = offset + idx
                if (0 <= frm_idx) and (frm_idx < seq_len):
                    win_prds[frm_idx].append(preds[offset + h_w_size])

    avg_preds = [np.mean(np.array(win), axis=0) for win in win_prds]
    return np.array(avg_preds)


def amass_data_to_3d_keypoints(smplx_models, amass_path, out_kps_format):
    data = np.load(str(amass_path), allow_pickle=True)
    seq_kps = run_smpl_inference(data, smplx_models=smplx_models, device='cuda',
                                 apply_trans=True, apply_root_rot=True, apply_shape=True)

    if out_kps_format == 'coco':
        kps_map = generate_smplx_to_coco_mappings(JOINT_NAMES)
    else:
        raise ValueError()
    return convert_seq_keypoints(seq_kps, kps_map)


def render_seq_poses_meshes(seq_3d_kps, seq_vertices, faces, out_vid_path, imw=1600, imh=1800, step=1, fps=24):
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    vwriter = get_writer(out_vid_path, fps=fps)
    n_samples = len(seq_3d_kps)
    for idx in tqdm(range(0, n_samples, step), desc='rendering'):
        in_kps = seq_3d_kps[idx]
        pred_vertices = seq_vertices[idx]
        joints_mesh = points_to_spheres(in_kps, vc=colors['red'])
        apply_mesh_tranfsormations_(joints_mesh,
                                    trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

        mv.set_static_meshes(joints_mesh)
        img0 = mv.render()

        prd_mesh = trimesh.Trimesh(vertices=pred_vertices, faces=faces,
                                   vertex_colors=np.tile((150, 150, 150), (10475, 1)))
        apply_mesh_tranfsormations_([prd_mesh], trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

        mv.set_static_meshes([prd_mesh])
        img1 = mv.render()

        img = np.concatenate([img0, img1], axis=1)
        vwriter.append_data(img)

    vwriter.close()


def run_test():
    amass_dir = Path('/media/F/datasets/amass/')
    smpl_x_dir = Path(amass_dir) / 'smplx'
    smplx_models = load_smplx_models(smpl_x_dir, 'cuda', 9)

    seq_kps_3d_path = Path(
        '/media/F/moveai/data/MOX_projects/MOX-009-LOC/contemporary_03/run_2020-07-08-14-17-21-807325_/'
        'poses_3d/cam01_track00.npz')

    data = np.load(seq_kps_3d_path, allow_pickle=True)
    seq_3d_kps_mvai = data["joints_3d"]
    joint_3d_names = data["joint_3d_names"].tolist()
    kps_mappings = generate_moveai3d_to_coco_mappings(joint_3d_names)
    seq_3d_kps = convert_seq_keypoints(seq_3d_kps_mvai, kps_mappings)
    seq_3d_kps[:, 0] = 0.5 * (seq_3d_kps_mvai[:, -1] + seq_3d_kps_mvai[:, -2])
    seq_3d_kps[:, 1] = seq_3d_kps_mvai[:, -2]  # left eye = left ear
    seq_3d_kps[:, 2] = seq_3d_kps_mvai[:, -1]  # right eye = right ear

    y = seq_3d_kps[:, :, 1].copy()
    z = seq_3d_kps[:, :, 2].copy()
    seq_3d_kps[:, :, 1] = z
    seq_3d_kps[:, :, 2] = -y

    ckpt_path = Path('/media/F/datasets/amass/ik_model/model_ckps/checkpoint_epoch=98-val_loss=0.02.ckpt')
    model = IKModelWrapper.load_from_checkpoint(str(ckpt_path))
    model.eval()

    seq_smpl_poses = run_inference(model, seq_3d_kps)
    sample_poses = np.zeros((seq_smpl_poses.shape[0], 156), dtype=np.float32)
    sample_poses[:, :66] = seq_smpl_poses[:, :66]
    data = {"poses": sample_poses, "gender": "male"}
    pred_seq_3d_kps, pred_seq_meshes = run_smpl_inference(data, smplx_models, 'cuda',
                                                          apply_trans=False,
                                                          apply_shape=False, return_mesh=True)

    faces = smplx_models["male"].faces
    o_viz_path = f'/media/F/thesis/motion_capture/data/debug/{seq_kps_3d_path.stem}.mp4'
    render_seq_poses_meshes(seq_3d_kps, pred_seq_meshes, faces, o_viz_path)


def run_main():
    ckpt_path = Path('/media/F/datasets/amass/ik_model/model_ckps/checkpoint_epoch=98-val_loss=0.02.ckpt')
    amass_dir = Path('/media/F/datasets/amass/')
    data_dir = Path('/media/F/datasets/amass/ik_model')
    apaths = _load_amass_path_list(Path(data_dir) / 'valid.csv')
    apath = apaths[50]
    # for tmp_path in apaths:
    #     if '64_20' in tmp_path.stem:
    #         apath = tmp_path
    #         break

    model = IKModelWrapper.load_from_checkpoint(str(ckpt_path))
    model.eval()

    smpl_x_dir = Path(amass_dir) / 'smplx'
    smplx_models = load_smplx_models(smpl_x_dir, 'cuda', 9)

    seq_3d_kps = amass_data_to_3d_keypoints(smplx_models, apath, out_kps_format='coco')
    seq_smpl_poses = run_inference(model, seq_3d_kps)

    data = np.load(apath)
    sample_data = {k: data[k] for k in data.keys()}
    org_poses = sample_data["poses"]
    org_poses[:, :66] = seq_smpl_poses
    sample_data["poses"] = org_poses
    pred_seq_3d_kps, pred_seq_meshes = run_smpl_inference(sample_data, smplx_models, 'cuda', return_mesh=True)

    imw, imh = 1600, 1800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    faces = smplx_models["male"].faces
    o_viz_path = f'/media/F/thesis/motion_capture/data/debug/{apath.stem}.mkv'
    vwriter = get_writer(o_viz_path, fps=24)
    n_samples = len(pred_seq_3d_kps)
    step = 10
    for idx in tqdm(range(0, n_samples, step), desc='rendering'):
        in_kps = seq_3d_kps[idx]
        pred_vertices = pred_seq_meshes[idx]
        joints_mesh = points_to_spheres(in_kps, vc=colors['red'])
        apply_mesh_tranfsormations_(joints_mesh,
                                    trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

        mv.set_static_meshes(joints_mesh)
        img0 = mv.render()

        prd_mesh = trimesh.Trimesh(vertices=pred_vertices, faces=faces,
                                   vertex_colors=np.tile((150, 150, 150), (10475, 1)))
        apply_mesh_tranfsormations_([prd_mesh], trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

        mv.set_static_meshes([prd_mesh])
        img1 = mv.render()

        img = np.concatenate([img0, img1], axis=1)
        vwriter.append_data(img)

    vwriter.close()


if __name__ == "__main__":
    # run_main()
    run_test()
