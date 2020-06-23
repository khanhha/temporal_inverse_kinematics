import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_pose_2d_gt(data_path, cam_idx):
    data = np.load(data_path, allow_pickle=True)
    cameras = [cam for cam in data["camera"]]
    cam = cameras[cam_idx]
    poses_3d = data["animation_joints"]
    n_poses, n_joints = poses_3d.shape[:2]
    cam_p = np.array(cam["K"]).reshape((3, 3)) @ np.array(cam["RT"]).reshape((3, 4))
    poses_3d = np.concatenate([poses_3d, np.ones((poses_3d.shape[0], poses_3d.shape[1], 1))], axis=2)
    poses_2d = cam_p @ poses_3d.reshape((-1, 4)).T
    poses_2d = poses_2d[:2] / poses_2d[2]
    return poses_2d.T.reshape((n_poses, n_joints, 2))


def run_main():
    video_dir = "/media/F/projects/moveai/codes/run_data/amass/syn/render/"
    data_dir = "/media/F/projects/moveai/codes/run_data/amass/syn/data/"
    opn_path = '/media/F/projects/moveai/codes/libs/openpose'
    model_path = '/media/F/projects/moveai/codes/motion_extraction/data/models'
    out_dir = '/media/F/projects/moveai/codes/run_data/amass/test'
    os.makedirs(out_dir, exist_ok=True)

    for vpath in Path(video_dir).rglob("*.mp4"):
        print(vpath)
        if '0017_SpeedVault001_poses_cam-1' not in vpath.stem:
            continue

        hint_idx = vpath.stem.index('_cam-')
        anim_name = vpath.stem[:hint_idx]
        cam_name = vpath.stem[hint_idx:]
        cam_idx = int(cam_name.split('-')[1])
        dpath = Path(f'{data_dir}/{anim_name}.npz')
        if not dpath.exists():
            print(f'no data file found for video file {anim_name}')
            continue

        frm_poses_gts = load_pose_2d_gt(dpath, cam_idx)
        vis = True
        if vis:
            vid = cv2.VideoCapture(str(vpath))
            for frm_id in range(len(frm_poses_gts)):
                ok, frm = vid.read()
                pose_gt = frm_poses_gts[frm_id, :, :]
                plt.imshow(frm[:, :, ::-1])
                plt.scatter(pose_gt[:, 0], pose_gt[:, 1], marker='o', c='blue')
                plt.show(block=True)
                if frm_id > 5:
                    break


if __name__ == "__main__":
    run_main()
