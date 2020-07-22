from enum import Enum


class KpsType(Enum):
    """official name for different type of joints"""
    Nose = 0,
    L_Eye = 1,
    R_Eye = 2,
    L_Ear = 3,
    R_Ear = 4,
    Head_Top = 5,
    Head_Bottom = 6,  # upper_neck
    Head = 7,
    Neck = 8,
    L_Shoulder = 9,
    R_Shoulder = 10,
    L_Elbow = 11,
    R_Elbow = 12,
    L_Wrist = 13,
    R_Wrist = 14,
    L_Hip = 15,
    R_Hip = 16,
    Mid_Hip = 17,
    L_Knee = 18,
    R_Knee = 19,
    L_Ankle = 20,
    R_Ankle = 21,
    Pelvis = 22,
    Spine = 23,
    L_BaseBigToe = 24,
    R_BaseBigToe = 25,
    L_BigToe = 26
    R_BigToe = 27,
    L_SmallToe = 28
    R_SmallToe = 29,
    L_Hand = 30,
    R_Hand = 31
    L_Heel = 32,
    R_Heel = 33,
    Chest = 34,
    # for hand key-point annotation, check it here.
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format
    L_HandRoot = 35,
    L_Thumb1CMC = 36,
    L_Thumb2Knuckles = 37,
    L_Thumb3IP = 38,
    L_Thumb4FingerTip = 39,
    L_Index1Knuckles = 40,
    L_Index2PIP = 41,
    L_Index3DIP = 42,
    L_Index4FingerTip = 43,
    L_Middle1Knuckles = 44,
    L_Middle2PIP = 45,
    L_Middle3DIP = 46,
    L_Middle4FingerTip = 47,
    L_Ring1Knuckles = 48,
    L_Ring2PIP = 49,
    L_Ring3DIP = 50,
    L_Ring4FingerTip = 51,
    L_Pinky1Knuckles = 52,
    L_Pinky2PIP = 53,
    L_Pinky3DIP = 54,
    L_Pinky4FingerTip = 55,

    R_HandRoot = 56,
    R_Thumb1CMC = 57,
    R_Thumb2Knuckles = 58,
    R_Thumb3IP = 59,
    R_Thumb4FingerTip = 60,
    R_Index1Knuckles = 61,
    R_Index2PIP = 62,
    R_Index3DIP = 63,
    R_Index4FingerTip = 64,
    R_Middle1Knuckles = 65,
    R_Middle2PIP = 66,
    R_Middle3DIP = 67,
    R_Middle4FingerTip = 68,
    R_Ring1Knuckles = 69,
    R_Ring2PIP = 70,
    R_Ring3DIP = 71,
    R_Ring4FingerTip = 72,
    R_Pinky1Knuckles = 73,
    R_Pinky2PIP = 74,
    R_Pinky3DIP = 75,
    R_Pinky4FingerTip = 76


def get_pose_kps_names(p_type):
    if p_type == 'coco':
        return []
    else:
        raise ValueError("get_pose_kps_names")


def get_pose_bones(p_type):
    if p_type == 'coco':
        return _COCO_Bone
    else:
        raise ValueError('get_pose_bones_index')


def get_pose_bones_index(p_type):
    if p_type == 'coco':
        return _COCO_Bone_Index
    else:
        raise ValueError('get_pose_bones_index')


_COCO = [KpsType.Nose,
         KpsType.L_Eye,
         KpsType.R_Eye,

         KpsType.L_Ear,
         KpsType.R_Ear,

         KpsType.L_Shoulder,
         KpsType.R_Shoulder,

         KpsType.L_Elbow,
         KpsType.R_Elbow,

         KpsType.L_Wrist,
         KpsType.R_Wrist,

         KpsType.L_Hip,
         KpsType.R_Hip,

         KpsType.L_Knee,
         KpsType.R_Knee,

         KpsType.L_Ankle,
         KpsType.R_Ankle
         ]

_COCO_Index = {jtype: jidx for jidx, jtype in enumerate(_COCO)}

_COCO_Bone = [(KpsType.Nose, KpsType.L_Eye), (KpsType.L_Eye, KpsType.L_Ear),
              (KpsType.Nose, KpsType.R_Eye), (KpsType.R_Eye, KpsType.R_Ear),
              (KpsType.L_Shoulder, KpsType.R_Shoulder),
              (KpsType.L_Shoulder, KpsType.L_Elbow), (KpsType.L_Elbow, KpsType.L_Wrist),
              (KpsType.R_Shoulder, KpsType.R_Elbow), (KpsType.R_Elbow, KpsType.R_Wrist),
              (KpsType.L_Shoulder, KpsType.L_Hip), (KpsType.L_Hip, KpsType.L_Knee), (KpsType.L_Knee, KpsType.L_Ankle),
              (KpsType.R_Shoulder, KpsType.R_Hip), (KpsType.R_Hip, KpsType.R_Knee), (KpsType.R_Knee, KpsType.R_Ankle)]

_COCO_Bone_Index = [(_COCO_Index[j0], _COCO_Index[j1]) for (j0, j1) in _COCO_Bone]
