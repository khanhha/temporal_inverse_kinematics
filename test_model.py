import torch
from torch.utils.data import DataLoader
from pathlib import Path
from model_wrap import IKModelWrapper
from mmskeleton.datasets import AmassDataset
from common.smpl_util import load_smplx_models, run_smpl_inference
from common.mesh_viewer import MeshViewer
from common.sphere import points_to_spheres
from common.colors_def import colors
from common.keypoints_util import generate_smplx_to_coco_mappings, convert_seq_keypoints
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
    for dl_idx, batch in enumerate(dl):
        in_poses = batch[0].to(model.device)
        batch_preds = model(in_poses)
        batch_preds = batch_preds["poses"].detach().cpu().numpy()
        ret_batch_idxs.append(batch[1].numpy())
        ret_batch_preds.append(batch_preds)

    seq_len = len(ds)
    win_prds = seq_len * [[]]
    h_w_size = model.hparams.win_size // 2
    for b_idxs, b_preds in zip(ret_batch_idxs, ret_batch_preds):
        for idx, preds in zip(b_idxs, b_preds):
            for offset in range(-h_w_size, h_w_size+1):
                frm_idx = offset + idx
                if (0 < frm_idx) and (frm_idx < seq_len):
                    win_prds[frm_idx].append(preds[offset + h_w_size])

    win_prds = np.array(win_prds)
    win_prds = np.mean(win_prds, axis=1)
    return win_prds


def amass_data_to_3d_keypoints(smplx_models, amass_path, out_kps_format):
    data = np.load(str(amass_path), allow_pickle=True)
    seq_kps = run_smpl_inference(data, smplx_models=smplx_models, device='cuda',
                                 apply_trans=True, apply_root_rot=True, apply_shape=True)

    if out_kps_format == 'coco':
        kps_map = generate_smplx_to_coco_mappings(JOINT_NAMES)
    else:
        raise ValueError()
    return convert_seq_keypoints(seq_kps, kps_map)


def run_main():
    ckpt_path = Path('/media/F/datasets/amass/ik_model/model_ckps/checkpoint_epoch=98-val_loss=0.02.ckpt')
    amass_dir = Path('/media/F/datasets/amass/')
    data_dir = Path('/media/F/datasets/amass/ik_model')
    apaths = _load_amass_path_list(Path(data_dir) / 'valid.csv')
    apath = apaths[20]
    # for tmp_path in apaths:
    #     if '64_20' in tmp_path.stem:
    #         apath = tmp_path
    #         break

    model = IKModelWrapper.load_from_checkpoint(str(ckpt_path))
    model.eval()

    smpl_x_dir = Path(amass_dir) / 'smplx'
    smplx_models = load_smplx_models(smpl_x_dir, 'cuda', 9)

    seq_3d_kps = amass_data_to_3d_keypoints(smplx_models, apath, out_kps_format='coco')
    run_inference(model, seq_3d_kps)

    imw, imh = 1600, 1800
    step = 10
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    ds = AmassDataset(smplx_models=smplx_models, amass_paths=[apath],
                      window_size=model.hparams.win_size, keypoint_format='coco', device='cpu',
                      add_gaussian_noise=False)
    T = len(ds)
    t_idx = 4

    o_viz_path = f'/media/F/thesis/motion_capture/data/debug/{apath.stem}.mkv'
    vwriter = get_writer(o_viz_path, fps=24)
    with torch.no_grad():
        for idx in tqdm(range(len(ds))):
            in_data = ds[idx]
            kps_3d = in_data['keypoints_3d']
            preds = model(torch.from_numpy(kps_3d).unsqueeze(0))
            pred_poses = preds["poses"].squeeze(0)
            pred_poses = pred_poses.view(9, -1)
            smplx_model = smplx_models["male"]

            joints_mesh = points_to_spheres(kps_3d[t_idx], vc=colors['red'])
            apply_mesh_tranfsormations_(joints_mesh,
                                        trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

            mv.set_static_meshes(joints_mesh)
            img0 = mv.render()

            # betas = in_data["betas"]
            # betas = torch.from_numpy(betas)
            prd_body = smplx_model(betas=None, global_orient=pred_poses[:, :3], body_pose=pred_poses[:, 3:66])
            prd_mesh = trimesh.Trimesh(vertices=prd_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
                                       vertex_colors=np.tile((150, 150, 150), (10475, 1)))
            apply_mesh_tranfsormations_([prd_mesh], trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))

            mv.set_static_meshes([prd_mesh])
            img1 = mv.render()

            img = np.concatenate([img0, img1], axis=1)
            vwriter.append_data(img)

            # poses = in_data["poses"]
            # poses = torch.from_numpy(poses)
            # gt_body = smplx_model(betas=betas, global_orient=gt, body_pose=poses[:, 3:66])
            # gt_mesh = trimesh.Trimesh(vertices=gt_body.vertices[t_idx].detach().numpy(), faces=smplx_model.faces,
            #                           vertex_colors=np.tile((0, 90, 170), (10475, 1)))
            # scn.add_geometry(gt_mesh)

    vwriter.close()


if __name__ == "__main__":
    run_main()
