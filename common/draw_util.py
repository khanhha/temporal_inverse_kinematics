import cv2
import matplotlib.pyplot as plt
from common.pose_def import get_pose_bones_index


def draw_3d_pose(axis, p_keypoints, p_type):
    """
    :param axis:
    :param p_keypoints: Jx3
    :param p_type: 'coco', 'openpose', 'smplx'
    """
    bones_idx = get_pose_bones_index(p_type)
    axis.plot(p_keypoints[:, 0], p_keypoints[:, 1], p_keypoints[:, 2], '+')
    for i, j in bones_idx:
        axis.plot([p_keypoints[j, 0], p_keypoints[i, 0]],
                  [p_keypoints[j, 1], p_keypoints[i, 1]],
                  [p_keypoints[j, 2], p_keypoints[i, 2]])
    return
