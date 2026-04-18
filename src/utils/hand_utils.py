'''
@Project ：SPNet 
@File    ：hand_utils.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2025/11/27 21:28 
'''

import numpy as np
import torch

# ---- Kinematics: MANO/MPII mapping ----
class MANOHandJoints:
    n_keypoints = 21
    n_joints = 21
    center = 4 # hand center at M0
    root = 0 # wrist
    labels = [
        'W', 'I0', 'I1', 'I2', 'M0', 'M1', 'M2', 'L0', 'L1', 'L2',
        'R0', 'R1', 'R2', 'T0', 'T1', 'T2', 'I3', 'M3', 'L3', 'R3', 'T3'
    ] # 0 → wrist, 1-3 → index0-2, 4-6 → middle0-2, 7-9 → little0-2, 10-12 → ring0-2,
    # 13-15 → thumb0-2, 16 → index3, 17 → middle3, 18 → little3, 19 → ring3, 20 → thumb3
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}  # fingertip vertex indices in MANO mesh
    parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15] # parent indices for each joint
    # 0 → None, 1 → 0, 2 → 1, 3 → 2, 4 → 0, 5 → 4, 6 → 5, 7 → 0, 8 → 7, 9 → 8,
    # 10 → 0, 11 → 10, 12 → 11, 13 → 0, 14 → 13, 15 → 14, 16 → 3, 17 → 6, 18 → 9, 19 → 12, 20 → 15
    end_points = [0, 16, 17, 18, 19, 20]

    # [18]    [19]    [17]    [16]
    #   |       |       |       |
    #  [9]    [12]     [6]     [3]    [20]
    #   |       |       |       |      |
    #  [8]    [11]     [5]     [2]    [15]
    #    |      |       |       |      |
    #   [7]    [10]     [4]     [1]   [14]
    #      \     \     |     /      /
    #        \     \   |   /      /
    #          \     \ | /    [13]
    #                  [0]

class MPIIHandJoints:
    n_keypoints = 21
    n_joints = 21
    center = 9
    root = 0
    labels = [
        'W', 'T0', 'T1', 'T2', 'T3', 'I0', 'I1', 'I2', 'I3',
        'M0', 'M1', 'M2', 'M3', 'R0', 'R1', 'R2', 'R3', 'L0', 'L1', 'L2', 'L3'
    ]
    parents = [None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    # 0 → None, 1 → 0, 2 → 1, 3 → 2, 4 → 3, 5 → 0, 6 → 5, 7 → 6, 8 → 7,
    # 9 → 0, 10 → 9, 11 → 10, 12 → 11, 13 → 0, 14 → 13, 15 → 14, 16 → 15, 17 → 0, 18 → 17, 19 → 18, 20 → 19
    colors = [
        (0, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
        (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),
        (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    ]

    # [20]    [16]    [12]    [8]
    #   |       |       |      |
    #  [19]    [15]    [11]    [7]    [4]
    #   |       |       |      |      |
    #  [18]    [14]    [10]    [6]    [3]
    #   |       |       |      |      |
    #   [17]    [13]    [9]     [5]   [2]
    #      \      \     |     /     /
    #        \      \   |   /    /
    #          \      \ | /   [1]
    #                  [0]



def mano_to_mpii(mano):
    """Convert MANO joint format to MPII format (numpy version)"""
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])])
    mpii = np.stack(mpii, 0)
    return mpii

def mpii_to_mano(mpii):
    """Convert MPII joint format to MANO format (numpy version)"""
    mano = []
    for j in range(MANOHandJoints.n_joints):
        mano.append(mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])])
    mano = np.stack(mano, 0)
    return mano


def mano_to_mpii_torch(mano_joints):
    """
    Convert MANO joint format to MPII format (PyTorch version, CUDA Graph compatible)

    Args:
        mano_joints: Tensor, shape [..., 21, 3] - MANO format joints

    Returns:
        mpii_joints: Tensor, shape [..., 21, 3] - MPII format joints
    """
    # Build index mapping from MANO to MPII
    # MANO labels: ['W', 'I0', 'I1', 'I2', 'M0', 'M1', 'M2', 'L0', 'L1', 'L2',
    #               'R0', 'R1', 'R2', 'T0', 'T1', 'T2', 'I3', 'M3', 'L3', 'R3', 'T3']
    # MPII labels: ['W', 'T0', 'T1', 'T2', 'T3', 'I0', 'I1', 'I2', 'I3',
    #               'M0', 'M1', 'M2', 'M3', 'R0', 'R1', 'R2', 'R3', 'L0', 'L1', 'L2', 'L3']

    # Mapping: MPII index -> MANO index
    mano_to_mpii_indices = [
        0,   # W -> W
        13,  # T0 -> T0
        14,  # T1 -> T1
        15,  # T2 -> T2
        20,  # T3 -> T3
        1,   # I0 -> I0
        2,   # I1 -> I1
        3,   # I2 -> I2
        16,  # I3 -> I3
        4,   # M0 -> M0
        5,   # M1 -> M1
        6,   # M2 -> M2
        17,  # M3 -> M3
        10,  # R0 -> R0
        11,  # R1 -> R1
        12,  # R2 -> R2
        19,  # R3 -> R3
        7,   # L0 -> L0
        8,   # L1 -> L1
        9,   # L2 -> L2
        18,  # L3 -> L3
    ]

    # Use advanced indexing to reorder (CUDA Graph compatible)
    mpii_joints = mano_joints[..., mano_to_mpii_indices, :]

    return mpii_joints