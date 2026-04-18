"""Utility helpers and loss functions extracted from the MobRecon project."""

import pickle
from os import path as osp

import openmesh as om
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.utils import mesh_sampling
import numpy as np


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals



def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)


def save_mesh(fp, x, f):
    om.write_mesh(fp, om.TriMesh(x, f))


def save_obj(v, f, file_name='output.obj'):
    with open(file_name, 'w') as obj_file:
        for vert in v:
            obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in f:
            obj_file.write(
                f"f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1}\n"
            )


def spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation):
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
            'up_transform': U,
        }
        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print(f"Transform matrices are saved in '{transform_fp}'")
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx])
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [to_sparse(down_transform) for down_transform in tmp['down_transform']]
    up_transform_list = [to_sparse(up_transform) for up_transform in tmp['up_transform']]

    return spiral_indices_list, down_transform_list, up_transform_list, tmp


def l1_loss(pred, gt, is_valid=None, drop_nan=False):
    if drop_nan:
        pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
        pred = torch.where(torch.isinf(pred), torch.full_like(pred, 0), pred)
        gt = torch.where(torch.isnan(gt), torch.full_like(gt, 0), gt)
        gt = torch.where(torch.isinf(gt), torch.full_like(gt, 0), gt)
    loss = F.l1_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        return loss.sum() / pos_num
    return loss.mean()


def bce_loss(pred, gt, is_valid=None):
    loss = F.binary_cross_entropy(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        return loss.sum() / pos_num
    return loss.mean()


def bce_wlog_loss(pred, gt, is_valid=None):
    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        return loss.sum() / pos_num
    return loss.mean()


def normal_loss(pred, gt, face, is_valid=None):
    v1_out = pred[:, face[:, 1], :] - pred[:, face[:, 0], :]
    v1_out = F.normalize(v1_out, p=2, dim=2)
    v2_out = pred[:, face[:, 2], :] - pred[:, face[:, 0], :]
    v2_out = F.normalize(v2_out, p=2, dim=2)
    v3_out = pred[:, face[:, 2], :] - pred[:, face[:, 1], :]
    v3_out = F.normalize(v3_out, p=2, dim=2)

    v1_gt = gt[:, face[:, 1], :] - gt[:, face[:, 0], :]
    v1_gt = F.normalize(v1_gt, p=2, dim=2)
    v2_gt = gt[:, face[:, 2], :] - gt[:, face[:, 0], :]
    v2_gt = F.normalize(v2_gt, p=2, dim=2)
    normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
    normal_gt = F.normalize(normal_gt, p=2, dim=2)

    cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
    cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
    cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
    loss = torch.cat((cos1, cos2, cos3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()


def edge_length_loss(pred, gt, face, is_valid=None):
    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    diff1 = torch.abs(d1_out - d1_gt)
    diff2 = torch.abs(d2_out - d2_gt)
    diff3 = torch.abs(d3_out - d3_gt)
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()


def contrastive_loss_3d(verts, aug_param):
    # Vectorized implementation to avoid dynamic tensor creation (required for CUDA Graph)
    batch_size = aug_param.size()[0]
    rot_angle = aug_param[:, 4] - aug_param[:, 0]
    ang_rad = torch.deg2rad(rot_angle)

    # Create rotation matrices in a vectorized way
    cos_vals = torch.cos(ang_rad)
    sin_vals = torch.sin(ang_rad)
    zeros = torch.zeros_like(cos_vals)
    ones = torch.ones_like(cos_vals)

    # Build rotation matrix [B, 3, 3] without loops
    rot_mat = torch.stack([
        torch.stack([cos_vals, sin_vals, zeros], dim=1),
        torch.stack([-sin_vals, cos_vals, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)

    verts_rot = torch.bmm(rot_mat.to(verts.device), verts[..., :3].permute(0, 2, 1)).permute(0, 2, 1)
    return F.l1_loss(verts_rot, verts[..., 3:], reduction='mean')


def revtrans_points(uv_point, trans):
    uv1 = torch.cat((uv_point, torch.ones_like(uv_point[:, :, :1])), 2)
    uv_crop = torch.bmm(trans, uv1.transpose(2, 1)).transpose(2, 1)[:, :, :2]
    return uv_crop


def contrastive_loss_2d(uv_pred, uv_trans, size):
    uv_pred_pre = uv_pred[:, :, :2]
    uv_pred_lat = uv_pred[:, :, 2:]

    uv_trans_pre = uv_trans[:, :, :3]
    uv_trans_lat = uv_trans[:, :, 3:]

    uv_pred_pre_rev = revtrans_points(uv_pred_pre * size, uv_trans_pre) / size
    uv_pred_lat_rev = revtrans_points(uv_pred_lat * size, uv_trans_lat) / size

    return F.l1_loss(uv_pred_pre_rev, uv_pred_lat_rev, reduction='mean')


def align_uv(t, uv, xyz, K):
    """对齐2D关键点和3D关键点的投影"""
    xyz_t = xyz + t
    proj = np.matmul(K, xyz_t.T).T
    uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz_t[:, 2:]
    return np.sum(np.abs(proj - uvz))


def registration(vertex, uv, j_regressor, K, size, uv_conf=None):
    """
    自适应2D-1D配准：通过2D关键点对齐mesh到图像空间
    :param vertex: 3D mesh顶点 (778, 3)
    :param uv: 2D关键点 (21, 2) 像素坐标
    :param j_regressor: 顶点到关节的回归矩阵 (16, 778)
    :param K: 相机内参 (3, 3)
    :param size: 图像尺寸
    :param uv_conf: 2D关键点置信度 (21, 1)，可选
    :return: 对齐后的顶点
    """
    from scipy.optimize import minimize
    from src.utils.hand_utils import mano_to_mpii

    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.3, 2))

    # 将顶点转换为关节点 (16个关节)
    vertex2xyz_base = np.matmul(j_regressor, vertex)

    # 添加5个指尖顶点 (MANO order: index, middle, little, ring, thumb)
    fingertip_indices = [333, 444, 672, 555, 744]
    fingertips = vertex[fingertip_indices]
    vertex2xyz_base = np.concatenate([vertex2xyz_base, fingertips], axis=0)  # (21, 3)

    # 转换为MPII格式
    vertex2xyz_base = mano_to_mpii(vertex2xyz_base)

    # 保持像素坐标，不归一化（与K矩阵保持一致）
    uv_base = uv

    # 初始化置信度
    if uv_conf is None:
        uv_conf = np.ones([uv_base.shape[0], 1])

    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        return vertex

    # 迭代优化，剔除outlier
    loss = np.array([5.0])
    attempt = 5

    while loss.mean() > 1 and attempt > 0:
        attempt -= 1

        # 从原始数据中选择
        uv = uv_base[uv_select.squeeze()]
        vertex2xyz = vertex2xyz_base[uv_select.squeeze()]

        sol = minimize(align_uv, t, method='SLSQP', bounds=bounds,
                       args=(uv, vertex2xyz, K))
        t = sol.x

        # 计算投影误差
        xyz = vertex2xyz + t
        proj = np.matmul(K, xyz.T).T
        uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
        loss = np.abs((proj - uvz).sum(axis=1))

        # 更新选择mask（基于当前选中的点）
        loss_mask = loss < loss.mean() + loss.std()
        if loss_mask.sum() < 7:
            print("break")
            break

        # 将局部mask映射回全局
        new_select = np.zeros(uv_select.shape[0], dtype=bool)
        new_select[uv_select.squeeze()] = loss_mask
        uv_select = new_select[:, np.newaxis]

    return vertex + t


__all__ = [
    'read_mesh',
    'save_mesh',
    'save_obj',
    'spiral_tramsform',
    'l1_loss',
    'bce_loss',
    'bce_wlog_loss',
    'normal_loss',
    'edge_length_loss',
    'contrastive_loss_3d',
    'contrastive_loss_2d',
    'revtrans_points',
    'align_uv',
    'registration',
]
