import sys
import os
import random
import argparse
import csv
import bpy
import numpy as np
import logging
from os import getenv
from os import remove
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty, FloatVectorProperty
from bpy.types import PropertyGroup, Panel, Scene, AddonPreferences, Operator, Material, WindowManager, Object, World, \
    Image, Node, Texture
from pathlib import Path
import subprocess
import tempfile
import torch
import time
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from smplx.body_models import create as smplx_create
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from math import radians
from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam

sys.path.append("/media/F/projects/moveai/codes/motion_extraction/addon_bld/")
from blender_utils import (get_calibration_matrix_K_from_blender, get_3x4_RT_matrix_from_blender,
                       get_3x4_P_matrix_from_blender, look_at)

sorted_parts = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']
# order
part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand',
              'bone_23': 'R_Hand'}
part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}

SMPLX_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]


def calc_bb(frm_joints):
    frm_joints = frm_joints.reshape(-1, 3)
    bmin = np.min(frm_joints, axis=0)
    bmax = np.max(frm_joints, axis=0)
    return bmin, bmax


def init_scene(scene, params, gender='female'):
    # load fbx model
    bpy.ops.import_scene.fbx(
        filepath=join(params['smpl_data_folder'], 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0]),
        axis_forward='Y', axis_up='Z', global_scale=100)


# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return mat_rots, bshapes


# reset the joint positions of the character according to its new shape
def reset_joint_positions(orig_trans, shape, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
    # since the regression is sparse, only the relevant vertex
    #     elements (joint_reg) and their indices (reg_ivs) are loaded
    reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
    # zero the pose and trans to obtain joint positions in zero pose
    apply_trans_pose_shape(orig_trans, np.zeros(72), shape, ob, arm_ob, obname, scene, cam_ob)

    # obtain a mesh after applying modifiers
    bpy.ops.wm.memory_statistics()
    # me holds the vertices after applying the shape blendshapes
    me = ob.to_mesh(scene, True, 'PREVIEW')

    # fill the regressor vertices matrix
    for iiv, iv in enumerate(reg_ivs):
        reg_vs[iiv] = me.vertices[iv].co
    bpy.data.meshes.remove(me)

    # regress joint positions in rest pose
    joint_xyz = joint_reg.dot(reg_vs)
    # adapt joint positions in rest pose
    arm_ob.hide = False
    bpy.ops.object.mode_set(mode='EDIT')
    arm_ob.hide = True
    for ibone in range(24):
        bb = arm_ob.data.edit_bones[obname + '_' + part_match['bone_%02d' % ibone]]
        bboffset = bb.tail - bb.head
        bb.head = joint_xyz[ibone]
        bb.tail = bb.head + bboffset
    bpy.ops.object.mode_set(mode='OBJECT')

    return shape


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, cam_ob, frame=None):
    # transform pose into rotation matrices (for pose) and pose blendshapes
    mrots, bsh = rodrigues2bshapes(pose)

    # set the location of the first bone to the translation parameter
    arm_ob.pose.bones[obname + '_Pelvis'].location = trans
    if frame is not None:
        arm_ob.pose.bones[obname + '_root'].keyframe_insert('location', frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[obname + '_' + part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)


def create_smpl_mesh(faces, verts, name, col_name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, [], faces)
    return obj


def shift_animation_onto_ground(frm_meshes, frm_joints):
    z_min = np.min(frm_meshes[:, :, 2].flatten())
    logging.info(f'\tmin z ground level = {z_min}')
    frm_meshes[:, :, 2] -= z_min
    frm_joints[:, :, 2] -= z_min
    return frm_meshes, frm_joints


def run_smpl_inference(data, smplx_models, device):
    gender = str(data["gender"])
    smplx_model = smplx_models["male_smplx"] if 'male' in gender else smplx_models["female_smplx"]
    batch_size = smplx_model.batch_size
    frm_poses = data["poses"].astype(np.float32)
    frm_trans = data["trans"].astype(np.float32)
    n_poses = frm_poses.shape[0]
    frm_meshes = []
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
        trans = torch.from_numpy(trans).to(device)
        root_orient = poses[:, :3]
        pose_body = poses[:, 3:66]
        left_pose_hand = poses[:, 66:66 + 45]
        right_pose_hand = poses[:, 66 + 45:66 + 90]

        # print(root_orient.shape, pose_body.shape, left_pose_hand.shape, right_pose_hand.shape, trans.shape)
        body = smplx_model(global_orient=root_orient, body_pose=pose_body,
                           left_hand_pose=left_pose_hand, right_hand_pose=right_pose_hand,
                           transl=trans)
        meshes = c2c(body.vertices)
        joints = c2c(body.joints)
        if pad > 0:
            meshes = meshes[:org_bsize]
            joints = joints[:org_bsize]
        frm_meshes.append(meshes)
        frm_joints.append(joints)

    frm_meshes = np.concatenate(frm_meshes, axis=0)
    frm_joints = np.concatenate(frm_joints, axis=0)

    frm_meshes, frm_joints = shift_animation_onto_ground(frm_meshes, frm_joints)

    return frm_meshes, frm_joints


def project_points(cam_obj, world_points):
    cam_p = np.array(get_3x4_P_matrix_from_blender(cam_obj)[0])
    world_points = np.concatenate([world_points, np.ones((world_points.shape[0], 1))], axis=1)
    point_projs = cam_p @ world_points.T
    point_projs = point_projs[:2] / point_projs[2]
    return point_projs.T


def find_optimal_dst_focal_length(cam_obj, world_points, min_dst_search, dst_step, cam_look_at_pos, to_camera_dir,
                                  focal_len,
                                  n_max_tries=100):
    scn = bpy.context.scene
    w = scn.render.resolution_x
    h = scn.render.resolution_y
    bpy.data.cameras[cam_obj.name].lens = focal_len

    for i in range(n_max_tries):
        dst = min_dst_search + dst_step * i

        cam_loc = cam_look_at_pos + to_camera_dir * dst
        cam_obj.location = Vector(cam_loc.tolist())

        look_at(cam_obj, Vector(cam_look_at_pos.tolist()))

        bpy.context.view_layer.update()

        projs = project_points(cam_obj, world_points)
        co_tl = np.min(projs, axis=0)
        co_br = np.max(projs, axis=0)
        tl_inside = co_tl[0] > 0 and co_tl[1] > 0
        br_inside = co_br[0] < w and co_br[1] < h
        # print(f'tl_px={co_tl}, br_px={co_br}')
        if tl_inside and br_inside:
            return dst, i

    return None, None


def bb_to_points(bmin, bmax):
    dim = bmax - bmin
    points = []
    for i in [True, False]:
        for j in [True, False]:
            for k in [True, False]:
                d = [dim[0] if i else 0,
                     dim[1] if j else 0,
                     dim[2] if k else 0]
                p = bmin + np.array(d)
                points.append(p)
    return np.array(points)


def create_empty(collection, name, empty_type, display_size):
    empty = bpy.data.objects.new(name, None)
    collection.objects.link(empty)
    empty.empty_display_size = display_size
    empty.empty_display_type = empty_type
    return empty


def add_empties_if_not_exist(points, collection):
    for idx, p in enumerate(points):
        name = f'empty_{idx}'
        if name not in bpy.data.objects:
            ob = create_empty(collection, f'empty_{idx}', 'CUBE', 0.05)
        else:
            ob = bpy.data.objects[name]
        ob.location = Vector(p)


def generate_random_cameras(cam_obj, frm_joints, ncam, head_idx, lfoot_idx, rfoot_idx, focal_length):
    head = frm_joints[:, head_idx, :]
    foot = 0.5 * (frm_joints[:, lfoot_idx, :] + frm_joints[:, rfoot_idx, :])
    h = np.mean(np.linalg.norm(head - foot, axis=1).flatten())
    bmin, bmax = calc_bb(frm_joints)
    mid_point = 0.5 * (bmin + bmax)
    bb_pnts = bb_to_points(bmin, bmax)

    debug = False
    if debug:
        c = bpy.data.collections["bb_debug"]
        add_empties_if_not_exist(bb_pnts.tolist(), c)

    radius_noise = 0.5*h
    scn = bpy.context.scene

    rand_cams = []
    for i in range(ncam):
        angle = np.random.rand() * np.pi * 2
        to_camera_dir = np.array([np.cos(angle), np.sin(angle), 0.0])

        # randomize the look-at point
        cam_look_at = mid_point.copy()
        cam_look_at[2] = cam_look_at[2] + np.random.rand() * 0.2 * h

        # give the maximum focal legnth, look_at point,
        # find the minimum distance to the camera that the whole projected animation is still inside the input image.
        dst_step = 0.1
        min_dst, n_tried = find_optimal_dst_focal_length(cam_obj, bb_pnts,
                                                         min_dst_search=h, dst_step=dst_step,
                                                         cam_look_at_pos=cam_look_at, to_camera_dir=to_camera_dir,
                                                         focal_len=focal_length,
                                                         n_max_tries=300)
        if min_dst is None:
            min_dst = 8 * h
            logging.error('cannot find minimum distance which makes the whole animation in screen. resort to :', min_dst)
        else:
            # add some distance margin
            min_dst += 0.1*h

        logging.info(f'\tcammera {i}. disance to camera: {min_dst}. n_tried = {n_tried}', )

        # randomly increase the camera distance
        cam_dst = min_dst + np.random.rand() * radius_noise

        # randomly increase the foca length
        # rand_focal_length = focal_len_range[1]
        bpy.data.cameras[cam_obj.name].lens = focal_length

        cam_loc = cam_look_at + to_camera_dir * cam_dst
        cam_obj.location = Vector(cam_loc.tolist())

        look_at(cam_obj, Vector(cam_look_at.tolist()))

        bpy.context.view_layer.update()

        P, K, RT = get_3x4_P_matrix_from_blender(cam_obj)
        # Make Matrix objects JSON serializable
        K = np.array(K).flatten().tolist()
        RT = np.array(RT).flatten().tolist()
        P = np.array(P[0]).flatten().tolist()

        w = scn.render.resolution_x
        h = scn.render.resolution_y
        rand_cams.append({"K": K, "RT": RT, "P": P,
                          "res_w": w, "res_h": h,
                          "cam_dst": cam_dst,
                          "angle": angle,
                          "f": focal_length})

    return rand_cams


def get_bld_cam_data(cam_ob):
    scn = bpy.context.scene
    camd = bpy.data.cameras[cam_ob.name]
    data = {"sensor_fit": camd.sensor_fit,
            "sensor_width": camd.sensor_width,
            "sensor_height": camd.sensor_height,
            "resolution_x": scn.render.resolution_x,
            "resolution_y": scn.render.resolution_y,
            "pixel_aspect_x": scn.render.pixel_aspect_x,
            "pixel_aspect_y": scn.render.pixel_aspect_y,
            "lens": camd.lens,
            "shift_x": camd.shift_x,
            "shift_y": camd.shift_y,
            "matrix_world": np.array(cam_ob.matrix_world),
            "location": np.array(cam_ob.location),
            "rotation_euler": np.array(cam_ob.rotation_euler)}

    return data


def set_bld_cam_data(cam_ob, cam_data):
    scn = bpy.context.scene
    camd = bpy.data.cameras[cam_ob.name]
    camd.sensor_fit = cam_data["sensor_fit"]
    camd.sensor_width = cam_data["sensor_width"]
    camd.sensor_height = cam_data["sensor_height"]
    scn.render.resolution_x = cam_data["resolution_x"]
    scn.render.resolution_y = cam_data["resolution_y"]
    scn.render.pixel_aspect_x = cam_data["pixel_aspect_x"]
    scn.render.pixel_aspect_y = cam_data["pixel_aspect_y"]
    camd.lens = cam_data["lens"]
    camd.shift_x = cam_data["shift_x"]
    camd.shift_y = cam_data["shift_y"]
    cam_ob.matrix_world = Matrix(cam_data["matrix_world"].tolist())

    bpy.context.view_layer.update()


def imgs_to_video(img_dir, vid_path, fps):
    # cmd_ffmpeg = f"ffmpeg -r 60 -i {img_dir}/%04d.png -c:v h264 {vid_path}"
    # os.system(cmd_ffmpeg)
    # log_message("Generating RGB video (%s)" % cmd_ffmpeg)
    subprocess.run(['ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'panic',
                    '-framerate', f'{fps}',
                    '-i', f'{img_dir}/%5d.png',
                    '-c:v', 'h264',
                    f'{vid_path}',
                    '-y'])
    # if Path(vid_path).exists():
    #     print(f'converte dimages to file {vid_path}')


def sample_random_texture_dir(root_dir):
    text_dirs = [path for path in Path(root_dir).glob("*") if path.is_dir()]
    return text_dirs[np.random.randint(0, len(text_dirs))]


def sample_random_texture_file(root_dir):
    f_paths = [path for path in Path(root_dir).glob("*.*") if path.is_file()]
    return f_paths[np.random.randint(0, len(f_paths))]


def replace_node_texture(node, new_img_path):
    old_img = node.image
    node.image = bpy.data.images.load(str(new_img_path))
    if old_img:
        bpy.data.images.remove(old_img)


def replace_ground_plane_textures(texture_dir, ob):
    mat = ob.data.materials[0]
    nodes = mat.node_tree.nodes
    color_node = [node for node in nodes if node.label == 'basecolor_tex'][0]
    roughness_node = [node for node in nodes if node.label == 'roughness_tex'][0]
    normal_node = [node for node in nodes if node.label == 'normal_tex'][0]
    height_node = [node for node in nodes if node.label == 'height_tex'][0]
    img_paths = [path for path in Path(texture_dir).glob('*.*')]
    for img_path in img_paths:
        if 'basecolor' in img_path.stem:
            replace_node_texture(color_node, img_path)
        elif 'roughness' in img_path.stem:
            replace_node_texture(roughness_node, img_path)
        elif 'normal' in img_path.stem:
            replace_node_texture(normal_node, img_path)
        elif "height" in img_path.stem:
            replace_node_texture(height_node, img_path)


def replace_env_texture(text_file):
    nodes = bpy.context.scene.world.node_tree.nodes
    env_tex_node = [n for n in nodes if n.label == "env_tex"][0]
    replace_node_texture(env_tex_node, text_file)


def replace_avatar_cloth_texture(ob, text_file):
    mat = ob.data.materials[0]
    nodes = mat.node_tree.nodes
    color_node = [node for node in nodes if node.label == 'basecolor_tex'][0]
    replace_node_texture(color_node, text_file)


def render_animation_cam(ob, frm_meshes, frm_joints, out_vid_dir):
    n_frms = frm_meshes.shape[0]
    frm_cnt = 0
    for frm_idx in range(n_frms):
        vertices = frm_meshes[frm_idx, :]
        s_t = time.time()
        ob.data.vertices.foreach_set("co", vertices.flatten())

        load_mesh_time = time.time() - s_t
        s_t = time.time()

        bpy.context.view_layer.update()
        vid_out_path = f'{out_vid_dir}/{str(frm_cnt).zfill(5)}'
        bpy.data.scenes["Scene"].render.filepath = vid_out_path
        bpy.data.scenes["Scene"].render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=False, write_still=True)
        frm_cnt += 1

        render_time = time.time() - s_t
        logging.info(f' load mesh time: {load_mesh_time}. render time: {render_time}')


def disable_blender_log(logfile='./blender_render.log'):
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)
    return old


def enable_blender_log(log_data):
    # disable output redirection
    os.close(1)
    os.dup(log_data)
    os.close(log_data)


def purgeOldImage():
    for wrl in bpy.data.worlds:
        if not wrl.users:
            if wrl.hdri_prop_world.world_hdri_maker is True:
                bpy.data.worlds.remove(wrl)

    for mat in bpy.data.materials:
        if not mat.users:
            if mat.hdri_prop_mat.mat_id_name == 'SHADOW_CATCHER':
                bpy.data.materials.remove(mat)

    for img in bpy.data.images:
        if not img.users:
            if img.hdri_prop_image.image_tag:
                bpy.data.images.remove(img)
            elif img.hdri_prop_image.image_tag_normal:
                bpy.data.images.remove(img)
            elif img.hdri_prop_image.image_tag_displace:
                bpy.data.images.remove(img)


def GlobalPathHdri():
    # preferences = bpy.context.preferences
    # addon_prefs = preferences.addons[__name__].preferences
    # GlobalPathHdri.HdriLib = bpy.path.abspath(addon_prefs.hdri_maker_library)
    # GlobalPathHdri.HdriUser = bpy.path.abspath(addon_prefs.hdri_maker_user_library)
    GlobalPathHdri.tools_lib = os.path.join("/media/F/projects/moveai/codes/run_data/blender/hdri_maker_2_83_v2_0_82/",
                                            "HDRi_tools_lib")


def subOperatorSky(realContext, type_save, immagine, nomePreview):
    scn = realContext
    trovata_immagine = []

    # create a new world
    percorso = GlobalPathHdri.tools_lib + os.sep + 'Files' + os.sep + 'Background Node v2.blend'

    with bpy.data.libraries.load(percorso, link=False) as (data_from, data_to):
        data_to.worlds = [w for w in data_from.worlds]

    for wrld in data_to.worlds:
        world = wrld.copy()

    new_world = world
    scn.world = new_world
    new_world.use_nodes = True
    new_world.hdri_prop_world.world_user = True
    new_world.hdri_prop_world.world_hdri_maker = True

    enviroment = new_world.node_tree.nodes.get('HDRi Maker Background')

    new_world.hdri_prop_world.world_id_base_name = 'HDRi Maker'
    if type_save == 'IMPORT':
        enviroment.image = immagine
        new_world.hdri_prop_world.world_id_name = enviroment.image.name[:-4]

    enviroment.image.hdri_prop_image.image_tag = True
    trovata_immagine.append(enviroment.image)

    purgeOldImage()


def load_dome(context):
    envImage = None
    for n in context.scene.world.node_tree.nodes:
        if n.name == 'HDRi Maker Background':
            envImage = n.image

    dome = None

    for o in context.scene.objects:
        if o.name == 'HDRi_Maker_Dome':
            if o.hdri_prop_obj.object_id_name == 'HDRi_Maker_Dome':
                dome = o

    if dome is None:
        for o in bpy.data.objects:
            if o.hdri_prop_obj.object_id_name == 'HDRi_Maker_Dome':
                if o.name == 'HDRi_Maker_Dome':
                    dome = o
                    context.scene.collection.objects.link(dome)

    if envImage is not None:
        if dome is None:

            percorso = GlobalPathHdri.tools_lib + os.sep + 'Files' + os.sep + 'hdri_dome.blend'

            with bpy.data.libraries.load(percorso, link=False) as (data_from, data_to):
                data_to.objects = [o for o in data_from.objects]

            for o in data_to.objects:
                if o.name == 'HDRi_Maker_Dome':
                    o.hdri_prop_obj.object_id_name = 'HDRi_Maker_Dome'
                    dome = o.copy()
                    bpy.data.objects.remove(o)
                    context.scene.collection.objects.link(dome)
                    dome.name = 'HDRi_Maker_Dome'
                    dome.location = (0, 0, 0)
                    dome.hide_select = True
                    dome.lock_location[:] = (True, True, True)

        mat = dome.material_slots[0].material
        mat.name = 'HDRi_Maker_Sky_Dome'
        mat.hdri_prop_mat.mat_id_name = 'HDRi_Maker_Sky_Dome'
        mat.diffuse_color = (1, 1, 1, 0)

        for n in dome.material_slots[0].material.node_tree.nodes:
            if n.name == 'HDRI_ENV':
                n.image = envImage
            if n.name == 'Texture Coordinate':
                n.object = dome

        for m in bpy.data.materials:
            if m.hdri_prop_mat.mat_id_name == 'Hdri_On_Ground':
                for n in m.node_tree.nodes:
                    if n.name == 'HDRI_ENV':
                        n.image = envImage
                    if n.name == 'Texture Coordinate':
                        n.object = dome

                # un po pasticciato , ma recupera il ground material se presente in bpy.data e lo sostituisce per
                # non avere dupplicati .001 etc
                if m.name == 'Hdri_On_Ground':
                    for s in dome.material_slots:
                        if 'Hdri_On_Ground' in s.material.name:
                            s.material = m


def updatebackground(self, context):
    scn = bpy.context.scene
    set = scn.hdri_prop_scn

    if scn.world:
        for n in scn.world.node_tree.nodes:
            if n.name == 'World Rotation':
                n.inputs[2].default_value[0] = radians(set.rot_world_x)
                n.inputs[2].default_value[1] = radians(set.rot_world_y)
                n.inputs[2].default_value[2] = radians(set.rot_world)
                n.inputs[1].default_value[2] = -set.menu_bottom

            if n.name == 'Background light':
                n.inputs[1].default_value = set.emission_force

            if n.name == 'Hdri hue_sat':
                n.inputs[1].default_value = set.hue_saturation

            if n.name == 'HDRI_COLORIZE':
                n.outputs[0].default_value = set.colorize

            if n.name == 'HDRI_COLORIZE_MIX':
                n.inputs[0].default_value = set.colorize_mix

            if n.name == 'BLURRY_Value':
                n.outputs[0].default_value = set.blurry_value / 2

                if scn.hdri_prop_scn.show_dome:
                    n.outputs[0].default_value = 0

            if n.name == 'HDRI_Maker_Exposure':
                n.inputs[1].default_value = set.exposure_hdri

        for o in context.scene.objects:
            if o.type == 'LIGHT':
                if o.data.type == 'SUN':
                    if o.hdri_prop_obj.object_id_name == 'HDRI_MAKER_SUN':
                        o.rotation_euler.z = radians((-set.sunRot_z) - set.rot_world)
                        o.rotation_euler.x = radians(set.sunRot_x)
                        o.data.energy = set.sun_light
                        o.data.color = set.sun_color

            if o.hdri_prop_obj.object_id_name == 'HDRi_Maker_Dome':

                o.hide_render = False if scn.hdri_prop_scn.show_dome else True
                o.hide_viewport = False if scn.hdri_prop_scn.show_dome else True
                o.hide_set(state=False if scn.hdri_prop_scn.show_dome else True)

                siz = scn.hdri_prop_scn.dome_size

                try:
                    for a in bpy.context.screen.areas:
                        if a.type == 'VIEW_3D':
                            for s in a.spaces:
                                if s.type == 'VIEW_3D':
                                    if s.clip_end < siz:
                                        s.clip_end = siz + 10
                except:
                    pass
                ##Conversione della dimensione della cupola basato sul sistema metrico e sulla unità della scala del progetto

                siz = (siz / 50) / scn.unit_settings.scale_length if scn.unit_settings.system == 'METRIC' else ((
                                                                                                                        siz / 3.28084) / 50) / scn.unit_settings.scale_length
                o.scale = (siz, siz, siz)

                for mod in o.modifiers:
                    if mod.name == 'HDRI_SMOOTH':
                        mod.iterations = set.wrap_smooth

            # distanza nebbia in eevee se il box supera la soglia:
            def fog_view_end(value):
                if value > scn.eevee.volumetric_end:
                    scn.eevee.volumetric_end = value

            if o.hdri_prop_obj.object_id_name == 'hdri_fog_box':
                if set.show_dome:
                    siz = scn.hdri_prop_scn.dome_size
                    siz += 20
                    o.dimensions = (siz, siz, siz / 2.08298755186722)
                    fog_view_end(siz)
                else:
                    siz = set.fog_box_size
                    siz = (siz / 50) / scn.unit_settings.scale_length if scn.unit_settings.system == 'METRIC' else ((
                                                                                                                            siz / 3.28084) / 50) / scn.unit_settings.scale_length

                    o.scale = (siz, siz, siz)
                    fog_view_end(o.dimensions[0])

        for m in bpy.data.materials:
            if m.hdri_prop_mat.mat_id_name == 'HDRi_Maker_Sky_Dome' or m.hdri_prop_mat.mat_id_name == 'Hdri_On_Ground':
                for n in m.node_tree.nodes:
                    if n.name == 'World Rotation':
                        n.inputs[1].default_value[0] = scn.hdri_prop_scn.domeMap_x
                        n.inputs[1].default_value[1] = scn.hdri_prop_scn.domeMap_y
                        n.inputs[1].default_value[2] = scn.hdri_prop_scn.domeMap_z

                        n.inputs[2].default_value[0] = radians(-set.rot_world_x)
                        n.inputs[2].default_value[1] = radians(-set.rot_world_y)
                        n.inputs[2].default_value[2] = radians(-set.rot_world)

                    if n.name == 'Emi_Map':
                        n.inputs[1].default_value = scn.hdri_prop_scn.dome_top_light

                    if n.name == 'Background light':
                        n.inputs[1].default_value = set.emission_force

                    if n.name == 'Hdri hue_sat':
                        n.inputs[1].default_value = set.hue_saturation

                    if n.name == 'HDRI_COLORIZE':
                        n.outputs[0].default_value = set.colorize

                    if n.name == 'HDRI_COLORIZE_MIX':
                        n.inputs[0].default_value = set.colorize_mix

                    if n.name == 'BLURRY_Value':
                        n.outputs[0].default_value = set.blurry_value / 2

                    if n.name == 'HDRI_Maker_Exposure':
                        n.inputs[1].default_value = set.exposure_hdri

            if m.hdri_prop_mat.mat_id_name == 'HDRI_MAKER_FOG':
                for n in m.node_tree.nodes:
                    if n.name == 'fog_level':
                        n.inputs[1].default_value[0] = 0
                        n.inputs[1].default_value[1] = 0
                        n.inputs[1].default_value[2] = 1 - set.fog_level if set.fog_flip else set.fog_level

                        n.inputs[2].default_value[0] = 0
                        n.inputs[2].default_value[1] = radians(-90) if set.fog_flip else radians(90)
                        n.inputs[2].default_value[2] = 0

                    if n.name == 'fog':
                        n.inputs[2].default_value = set.fog_density
                        n.inputs[6].default_value = set.fog_emission

                    if n.name == 'fog_mapping':
                        n.inputs[2].default_value[2] = radians(set.fog_direction)

                    if n.name == 'fog_ramp':
                        n.color_ramp.elements[0].color = (0, 0, 0, 1)
                        n.color_ramp.elements[1].color = (1, 1, 1, 1)

                        n.color_ramp.elements[0].position = set.fog_patches * 0.25
                        n.color_ramp.elements[1].position = 1 - (set.fog_patches * 0.60)

                    if n.name == 'fog_noise':
                        n.inputs[2].default_value = 1 + (set.fog_patches_size * 10)


def gen_single_anim_cams(cam_ob, anim_name, data, frm_joints, fps,
                         n_cams, out_data_dir):
    joint_map = {name: idx for idx, name in enumerate(SMPLX_JOINT_NAMES)}
    head_idx = joint_map["head"]
    lfoot_idx = joint_map["left_foot"]
    rfoot_idx = joint_map["right_foot"]

    mocap_framerate = data["mocap_framerate"]
    cams = []
    bld_cams = []
    focal_len_range = [40, 60]
    rand_focal_length = random.randint(focal_len_range[0], focal_len_range[1])
    for cam_idx in range(n_cams):
        cam_data = generate_random_cameras(cam_ob, frm_joints, 1,
                                           head_idx, lfoot_idx, rfoot_idx, focal_length=rand_focal_length)[0]
        bld_cams.append(get_bld_cam_data(cam_ob))
        cams.append(cam_data)

    out_data_path = f'{out_data_dir}/{anim_name}.npz'
    logging.info(f'\toutput data file: {out_data_path}')
    np.savez_compressed(out_data_path,
                        camera=cams,
                        bld_cameras=bld_cams,
                        animation_joints=frm_joints,
                        poses=data["poses"],
                        trans=data["trans"],
                        gender=str(data["gender"]),
                        betas=data["betas"],
                        mocap_framerate=mocap_framerate,
                        framerate=fps)


def create_smpl_template_object():
    # faces = c2c(bm.f)
    # smpl_dict = np.load(bm_path, encoding='latin1')
    # v_tpl = smpl_dict["v_template"]
    # ob = create_smpl_mesh(faces.tolist(), v_tpl.tolist(), "smpl_mesh", "Collection")
    pass


def load_smplx_models(smplx_dir, device, batch_size):
    male_path = f'{smplx_dir}/SMPLX_MALE.npz'
    female_path = f'{smplx_dir}/SMPLX_FEMALE.npz'
    # use_pca: for hand pose parameter. smplx model is kind of a bit different from human_pose_prior. not sure why
    male_smplx = smplx_create(model_path=male_path, model_type='smplx', gender='male', use_pca=False,
                              use_face_contour=True, batch_size=batch_size).to(device)
    female_smplx = smplx_create(model_path=female_path, model_type='smplx', gender='female', use_pca=False,
                                use_face_contour=True, batch_size=batch_size).to(device)
    return {'male_smplx': male_smplx, 'female_smplx': female_smplx}


def get_animation_poses(obj, bone_names, s_frm_idx, e_frm_idx, root_bone_name):
    frame_joints = []
    bones = [obj.pose.bones[bname] for bname in bone_names]
    scn = bpy.context.scene
    for fidx in range(s_frm_idx, e_frm_idx):
        scn.frame_set(fidx)
        pose_joints = [obj.matrix_world @ b.matrix @ b.location if root_bone_name not in b.name
                       else obj.matrix_world @ b.head
                       for b in bones]
        # if fidx >= e_frm_idx-2:
        #     print(pose_joints)
        frame_joints.append(pose_joints)

    return np.array(frame_joints)


def animation_frames(obj):
    fcurve = obj.animation_data.action.fcurves[0]
    return len(fcurve.keyframe_points)


def ensure_naming_conventions(mxm_obj):
    """
    each mixamo charater has different bone name prefix like mixamorig1:, mixamorig2:..
    this function makes sure that all mixamo bone names always start with "mixamorig:"
    :param mxm_obj:
    """
    prefix = 'mixamorig:'
    for b in mxm_obj.pose.bones:
        if ':' in b.name:
            b.name = prefix + b.name[b.name.index(':') + 1:]
        else:
            b.name = prefix + b.name


def reduce_smpl_data_fps(data, frame_stride):
    data["poses"] = data["poses"].astype(np.float32)[::frame_stride]
    data["trans"] = data["trans"].astype(np.float32)[::frame_stride]
    data["dmpls"] = data["dmpls"].astype(np.float32)[::frame_stride]
    return data


def cut_off_smpl_data(data, max_frame):
    n_frms = data["poses"].shape[0]
    if n_frms > max_frame > 0:
        data["poses"] = data["poses"][:max_frame]
        data["trans"] = data["trans"][:max_frame]
        data["dmpls"] = data["dmpls"][:max_frame]
    return data


def load_hdri_env(epath, context):
    immagine = bpy.data.images.load(str(epath))
    subOperatorSky(context.scene, 'IMPORT', immagine, 'None')
    load_dome(context)
    updatebackground(None, context)


def clear_mems():
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    for g in bpy.data.node_groups:
        if g.users == 0:
            bpy.data.node_groups.remove(g)

    for o in bpy.data.objects:
        if o.users == 0:
            bpy.data.objects.remove(o)

    for w in bpy.data.worlds:
        if w.users == 0:
            bpy.data.worlds.remove(w)


def collect_all_shape_params(amass_dir):
    all_paths = [path for path in Path(amass_dir).rglob("*.npz")]
    subject_marks = set()
    subject_paths = []
    for path in all_paths:
        if path.parent not in subject_marks:
            subject_paths.append(path)
            subject_marks.add(path.parent)
    logging.info(f'amass: total subject = {len(subject_paths)}')
    shapes = []
    for path in subject_paths:
        data = np.load(str(path))
        betas = data["betas"]
        gender = data["gender"]
        shapes.append((betas, gender))
    return shapes


def render_single_anim_cam(cam_ob, ob, anim_name, data, frm_meshes, frm_joints,
                           cameras, env_tex_paths, cloth_tex_paths, out_vid_dir):
    total_render_time = 0

    env_file = env_tex_paths[np.random.randint(0, len(env_tex_paths))]
    load_hdri_env(env_file, bpy.context)

    cloth_file = cloth_tex_paths[np.random.randint(0, len(cloth_tex_paths))]
    replace_avatar_cloth_texture(ob, cloth_file)

    for cam_idx, cam_data in enumerate(cameras):
        t = time.time()

        set_bld_cam_data(cam_ob, cam_data)

        # env_file = env_tex_paths[np.random.randint(0, len(env_tex_paths))]
        # load_hdri_env(env_file, bpy.context)
        #
        # cloth_file = cloth_tex_paths[np.random.randint(0, len(cloth_tex_paths))]
        # replace_avatar_cloth_texture(ob, cloth_file)

        video_name = f'{anim_name}_cam-{cam_idx}.mp4'
        out_video_path = f'{out_vid_dir}/{video_name}'
        fps = data["framerate"]
        with tempfile.TemporaryDirectory() as tmp_img_dir:
            # tmp_img_dir = f'/media/F/projects/moveai/codes/run_data/amass/debug/'
            logging.info(f'\trendering {video_name}')
            log_file = disable_blender_log()
            render_animation_cam(ob, frm_meshes, frm_joints,
                                 out_vid_dir=tmp_img_dir)
            enable_blender_log(log_file)
            # use ffmpeg to convert rendered images to video
            imgs_to_video(tmp_img_dir, out_video_path, fps)
            render_time = time.time() - t
            logging.info(f'\trender time = {render_time / 60} minutes.  output video file: {out_video_path}')
            total_render_time += render_time

    logging.info(f'\ttotal render time = {total_render_time / 60} minutes.')


def render_multi_anims_cams_videos(smpl_bld_ob, bld_cam_ob, texture_dir, smplx_dir, amass_dir,
                                   amass_files, out_vid_dir, device):
    smplx_models = load_smplx_models(smplx_dir, device, 64)
    data_paths = amass_files

    cloth_tex_paths = [path for path in Path(f'{texture_dir}/cloth').rglob('*.*')]
    env_tex_paths = [path for path in Path(f'{texture_dir}/env').rglob('*.*')]

    for data_path in data_paths:
        t = time.time()
        anim_name = data_path.stem
        logging.info(f'animation: {anim_name}')

        data = np.load(str(data_path), allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        framrate = int(data["framerate"])
        cameras = data["bld_cameras"]
        frm_meshes, frm_joints = run_smpl_inference(data, smplx_models, device)
        mesh_time = time.time() - t

        logging.info(f'\tmesh generation time: {mesh_time}. '
              f'data shape: {frm_meshes.shape}. {frm_joints.shape} '
              f'n_frames = {frm_joints.shape[0]}. '
              f'fps = {framrate}')

        rel_data_path = data_path.relative_to(amass_dir)
        rel_data_dir = rel_data_path.parent
        anim_out_video_dir = out_vid_dir / rel_data_dir
        os.makedirs(anim_out_video_dir, exist_ok=True)

        render_single_anim_cam(cam_ob=bld_cam_ob, ob=smpl_bld_ob,
                               anim_name=anim_name,
                               data=data, frm_meshes=frm_meshes, frm_joints=frm_joints,
                               cameras=cameras,
                               cloth_tex_paths=cloth_tex_paths,
                               env_tex_paths=env_tex_paths,
                               out_vid_dir=anim_out_video_dir)
        clear_mems()


def sample_random_shape(shape_db):
    n_shapes = len(shape_db)
    return shape_db[int(np.random.randint(0, n_shapes, 1))]


def gen_multi_anims_cams(bld_cam_ob, smplx_dir, amass_dir, shape_db,
                         amass_files, out_data_dir, device, fps, max_anim_frame, n_cams):
    smplx_models = load_smplx_models(smplx_dir, device, 64)
    data_paths = amass_files

    for data_path in data_paths:
        t = time.time()
        anim_name = data_path.stem
        logging.info(f'animation: {anim_name}')

        data = np.load(str(data_path))
        data = {key: data[key] for key in data.keys()}

        if shape_db is not None:
            shape = sample_random_shape(shape_db)
            data["gender"] = shape[0]
            data["betas"] = shape[1]

        n_org_frames = len(data["poses"])
        mocap_framrate = int(data["mocap_framerate"])
        frame_stride = max(int(mocap_framrate // fps), 1)

        data = reduce_smpl_data_fps(data, frame_stride)

        data = cut_off_smpl_data(data, max_anim_frame)

        frm_meshes, frm_joints = run_smpl_inference(data, smplx_models, device)
        mesh_time = time.time() - t

        logging.info(f'\tmesh generation time: {mesh_time}. '
              f'data shape: {frm_meshes.shape}. {frm_joints.shape} '
              f'n_org_frames = {n_org_frames}. '
              f'n_process_frames = {frm_joints.shape[0]}. '
              f'mocap_fps = {mocap_framrate}. fps = {fps}. frm_stride = {frame_stride}')

        rel_data_path = data_path.relative_to(amass_dir)
        rel_data_dir = rel_data_path.parent
        anim_out_data_dir = out_data_dir / rel_data_dir
        os.makedirs(anim_out_data_dir, exist_ok=True)

        gen_single_anim_cams(cam_ob=bld_cam_ob,
                             anim_name=anim_name,
                             data=data, frm_joints=frm_joints,
                             n_cams=n_cams,
                             fps=fps,
                             out_data_dir=anim_out_data_dir)
        clear_mems()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synth amin_ds images.')
    parser.add_argument('--smplx', type=str, help='smplx directory')
    parser.add_argument('--amass', type=str, help='amass motion data dir')
    parser.add_argument('--amass_csv', type=str, default='', help='csv file: a list of amass relative file paths')
    parser.add_argument('--texture_dir', type=str, default='', help='texture directory /env, /floor, /cloth')
    parser.add_argument('--shape_file', type=str, default='', help='npz file contain human shape params')
    parser.add_argument('--out_dir', type=str, help='ouput dir. output/render. output/data')
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help='cuda or cpu', )
    parser.add_argument('--n_cams', type=int, default=5, help='n camera per animation', )
    parser.add_argument('--fps', type=int, default=30, help='render every stride..', )
    parser.add_argument('--gen_data', action='store_true', help='gen multiview data', )
    parser.add_argument('--gen_video', action='store_true', help='gen video from multiview data', )
    parser.add_argument('--max_anim_frame', type=int, default=1000,
                        help='max frame number that no animation should exist. -1 for unlimited')
    parser.add_argument('--log', type=str, default="./my_log.txt")

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args


def load_file_list_from_csv(amass_csv_path, root_dir):
    all_paths = [path for path in Path(root_dir).rglob('*.npz')]

    if amass_csv_path:
        with open(amass_csv_path, 'rt') as file:
            file_names = {name[0] for name in list(csv.reader(file, delimiter=','))}
        my_paths = [path for path in all_paths if path.stem in file_names]
        return my_paths
    else:
        return all_paths


def run_from_args():
    args = parse_args()
    bld_cam_ob = bpy.data.objects["Camera"]
    smpl_bld_ob = bpy.data.objects["smpl_v1"]
    device = args.device

    logging.basicConfig(filename=args.log, level=logging.DEBUG)
    logging.info('\narguments: args \n')

    # shapes = collect_all_shape_params(args.amass)
    # out_path = f'/media/F/projects/moveai/codes/run_data/amass/smplx_shapes.npz'
    # np.savez_compressed(out_path, shapes=shapes)
    if args.gen_data:
        smplx_shapes = np.load(args.shape_file, allow_pickle=True)["shapes"] if args.shape_file else None
        if smplx_shapes is not None:
            logging.info(f'load {len(smplx_shapes)} smplx shape params from the file {len(smplx_shapes)}')
        else:
            logging.info('no usage of randam shape. using default shapes from the amass amin_ds')

        amass_dir = args.amass
        logging.info('\n\ngenerating multi-view animation data')
        amass_files = load_file_list_from_csv(args.amass_csv, amass_dir)
        logging.info(f' {len(amass_files)} amass npz files will be processed\n')

        out_data_dir = Path(f'{args.out_dir}/data/')
        gen_multi_anims_cams(bld_cam_ob=bld_cam_ob,
                             smplx_dir=args.smplx, amass_dir=args.amass, amass_files=amass_files, shape_db=smplx_shapes,
                             out_data_dir=out_data_dir, n_cams=args.n_cams,
                             device=device, fps=args.fps, max_anim_frame=args.max_anim_frame)

    if args.gen_video:
        data_dir = f'{args.out_dir}/data'
        amass_files = load_file_list_from_csv(args.amass_csv, data_dir)
        logging.info('\n\nsynthesizing multi-view videos')
        out_vid_dir = Path(f'{args.out_dir}/render')
        render_multi_anims_cams_videos(smpl_bld_ob=smpl_bld_ob, bld_cam_ob=bld_cam_ob,
                                       smplx_dir=args.smplx, amass_dir=data_dir, amass_files=amass_files,
                                       texture_dir=args.texture_dir,
                                       out_vid_dir=out_vid_dir,
                                       device=device)


class AmassRandCamera(bpy.types.Operator):
    bl_idname = "wm.mxm_rand_camera"
    bl_label = "mxm_rand_camera"
    bl_description = "randomize camera"
    bl_options = {'REGISTER'}

    def import_hdri(self, context):
        edir = '/media/F/projects/moveai/codes/run_data/blender/hdri_maker_database/HDMK_LIB_VOL_16/HDRI_MAKER_LIB/08k_Library'
        env_paths = [path for path in Path(edir).glob("*.hdr")]
        epath = env_paths[int(np.random.randint(0, len(env_paths), 1))]
        immagine = bpy.data.images.load(str(epath))
        subOperatorSky(context.scene, 'IMPORT', immagine, 'None')
        load_dome(context)
        updatebackground(self, context)

    def execute(self, context):
        self.import_hdri(context)

        # mxm_obj = bpy.data.objects["mixamo"]
        # mxm_obj.select_set(True)
        # bpy.context.view_layer.objects.active = mxm_obj
        # ensure_naming_conventions(mxm_obj)
        #
        # cam_ob = bpy.data.objects["Camera"]
        # mixamo = False
        # if mixamo:
        #     bpy.ops.object.mode_set(mode='POSE')
        #     bnames = [b.name for b in mxm_obj.pose.bones]
        #     head_idx = bnames.index("mixamorig:HeadTop_End")
        #     lfoot_idx = bnames.index("mixamorig:LeftFoot")
        #     rfoot_idx = bnames.index("mixamorig:RightFoot")
        #     n_frms = animation_frames(mxm_obj)
        #     root_bone_name = "Hips"
        #     anim_joints = get_animation_poses(mxm_obj, bnames, 0, n_frms, root_bone_name)
        #     cam_data = generate_random_cameras(cam_ob, anim_joints, 1,
        #                                        head_idx, lfoot_idx, rfoot_idx)[0]
        # else:
        #     bm_path = '/media/F/projects/moveai/codes/run_data/amass/data/smpl/models_smplx_v1_0/' \
        #               'models/smplx/SMPLX_MALE.npz'
        #     data_dir = "/media/F/projects/moveai/codes/run_data/amass/data/ACCAD/ACCAD/"
        #     joint_map = {name: idx for idx, name in enumerate(SMPLX_JOINT_NAMES)}
        #     head_idx = joint_map["head"]
        #     lfoot_idx = joint_map["left_ankle"]
        #     rfoot_idx = joint_map["right_ankle"]
        #     data_paths = sorted([path for path in Path(data_dir).rglob("*.npz")])
        #     data_path = data_paths[100]
        #     data, frm_meshes, frm_joints = run_smpl_inference(data_path, bm_path, 'cuda')
        #     tmp_joints = frm_joints.copy()
        #     tmp_joints[:, head_idx, :] += np.array([0, 0.3, 0.0]).reshape(1, 3)
        #     tmp_joints[:, lfoot_idx, :] += np.array([0, -0.3, 0.0]).reshape(1, 3)
        #     tmp_joints[:, rfoot_idx, :] += np.array([0, -0.3, 0.0]).reshape(1, 3)
        #     test = tmp_joints.reshape((-1, 3))
        #     print(tmp_joints.shape, head_idx, lfoot_idx, rfoot_idx, np.min(test, axis=0), np.max(test, axis=0))
        #     cam_data = generate_random_cameras(cam_ob, tmp_joints, 1,
        #                                        head_idx, lfoot_idx, rfoot_idx)[0]
        #
        # bpy.context.view_layer.update()
        # bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


class VIEW3D_PT_Amass(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "Mixamo retarget"
    bl_category = "MixamoRetarget"

    def draw(self, context):
        scene = context.scene

        layout = self.layout
        layout.use_property_split = True

        col = layout.column()
        col.operator("wm.mxm_rand_camera", text="rand_camera", icon="LIGHT")


my_classes = (
    VIEW3D_PT_Amass,
    AmassRandCamera
)


def register():
    from bpy.utils import register_class
    for cls in my_classes:
        register_class(cls)


def unregister():
    from bpy.utils import unregister_class
    for cls in my_classes:
        unregister_class(cls)


if __name__ == '__main__':
    GlobalPathHdri()
    register()
    run_from_args()
