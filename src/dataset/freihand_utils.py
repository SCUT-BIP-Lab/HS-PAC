

import sys
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
from termcolor import cprint
import vctoolkit as vc
import json
import time
import skimage.io as io
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om
import random
import math

def load_db_annotation(base_path, set_name):
    if set_name in ['training', 'train']:
        t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % 'training')
        mano_path = os.path.join(base_path, '%s_mano.json' % 'training')
        xyz_path = os.path.join(base_path, '%s_xyz.json' % 'training')
        K_list = json_load(k_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        return zip(K_list, mano_list, xyz_list)
    elif set_name in ['evaluation', 'eval', 'val', 'test']:
        t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % 'evaluation')
        scale_path = os.path.join(base_path, '%s_scale.json' % 'evaluation')
        mano_path = os.path.join(base_path, '%s_mano.json' % 'evaluation')
        xyz_path = os.path.join(base_path, '%s_xyz.json' % 'evaluation')
        K_list = json_load(k_path)
        scale_list = json_load(scale_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)
        assert len(K_list) == len(scale_list), 'Size mismatch.'
        return zip(K_list, scale_list, mano_list, xyz_list)
    else:
        raise Exception('set_name error: ' + set_name)

def read_mesh(idx, base_path):
    path = os.path.join(base_path, 'training', 'mesh', '%08d.ply' % idx)
    _assert_exist(path)
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)

def read_mesh_eval(idx, base_path):
    path = os.path.join(base_path, 'evaluation', 'mesh', '%08d.ply' % idx)
    _assert_exist(path)
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs
    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'
    img_rgb_path = os.path.join(base_path, set_name, 'rgb', '%08d.jpg' % sample_version.map_id(idx, version))
    if not os.path.exists(img_rgb_path):
        img_rgb_path = os.path.join(base_path, set_name, 'rgb2', '%08d.jpg' % idx)
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)

def read_img_abs(idx, base_path, set_name):
    img_rgb_path = os.path.join(base_path, set_name, 'rgb', '%08d.jpg' % idx)
    if not os.path.exists(img_rgb_path):
        img_rgb_path = os.path.join(base_path, set_name, 'rgb2', '%08d.jpg' % idx)
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)

def read_mask_woclip(idx, base_path, set_name):
    mask_path = os.path.join(base_path, set_name, 'mask', '%08d.jpg' % idx)
    _assert_exist(mask_path)
    return io.imread(mask_path)[:, :, 0]

def read_color_mask(idx, base_path, set_name):
    mask_path = os.path.join(base_path, set_name, 'colormap', '%08d.png' % idx)
    _assert_exist(mask_path)
    mask = io.imread(mask_path)
    # 如果有3个通道，转为灰度
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # 灰度大于10的像素设为255
    mask = np.where(mask > 10, 255, 0).astype(np.uint8)
    return mask

def projectPoints(xyz, K):
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)
    return x

def inv_base_tranmsform(x, mean=0.5, std=0.5):
    x = x.transpose(1, 2, 0)
    image = (x * std + mean) * 255
    return image.astype(np.uint8)

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

class Augmentation(object):
    def __init__(self, size=224):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
        ])
    def __call__(self, img):
        return self.augment(img)

def augmentation(img, bbox, data_split, exclude_flip=False, input_img_shape=(256, 256), mask=None, base_scale=1.1, scale_factor=0.25, rot_factor=60, shift_wh=None, gaussian_std=1, color_aug=False):
    if data_split == 'train':
        scale, rot, shift, color_scale, do_flip = get_aug_config(exclude_flip, base_scale=base_scale, scale_factor=scale_factor, rot_factor=rot_factor, gaussian_std=gaussian_std)
    else:
        scale, rot, shift, color_scale, do_flip = base_scale, 0.0, [0, 0], np.array([1, 1, 1]), False
    img, trans, inv_trans, mask, shift_xy = generate_patch_image(img, bbox, scale, rot, shift, do_flip, input_img_shape, shift_wh=shift_wh, mask=mask)
    if color_aug:
        img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, np.array([rot, scale, *shift_xy]), do_flip, input_img_shape[0]/(bbox[3]*scale), mask

def augmentation_2d(img, joint_img, princpt, trans, do_flip):
    joint_img = joint_img.copy()
    joint_num = len(joint_img)
    original_img_shape = img.shape
    if do_flip:
        joint_img[:, 0] = original_img_shape[1] - joint_img[:, 0] - 1
        princpt[0] = original_img_shape[1] - princpt[0] - 1
    for i in range(joint_num):
        joint_img[i,:2] = trans_point2d(joint_img[i,:2], trans)
    princpt = trans_point2d(princpt, trans)
    return joint_img, princpt

class MPIIHandJoints:
    n_keypoints = 21
    n_joints = 21
    center = 9
    root = 0
    labels = [
        'W', 'T0', 'T1', 'T2', 'T3', 'I0', 'I1', 'I2', 'I3',
        'M0', 'M1', 'M2', 'M3', 'R0', 'R1', 'R2', 'R3',
        'L0', 'L1', 'L2', 'L3',
    ]
    parents = [
        None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11,
        0, 13, 14, 15, 0, 17, 18, 19
    ]
    colors = [
        (0, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
        (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),
        (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    ]

def contrastive_loss_3d(verts, aug_param):
    rot_mat = torch.empty(aug_param.size()[0], 3, 3)
    rot_angle = aug_param[:, 4] - aug_param[:, 0]
    ang_rad = torch.deg2rad(rot_angle)
    for i in range(aug_param.size()[0]):
        rot_mat[i] = torch.tensor([[torch.cos(ang_rad[i]), torch.sin(ang_rad[i]), 0],
                                   [-torch.sin(ang_rad[i]), torch.cos(ang_rad[i]), 0],
                                   [0, 0, 1]])
    verts_rot = torch.bmm(rot_mat.to(verts.device), verts[..., :3].permute(0, 2, 1)).permute(0, 2, 1)
    return torch.nn.functional.l1_loss(verts_rot, verts[..., 3:], reduction='mean')

def contrastive_loss_2d(uv_pred, uv_trans, size):
    uv_pred_pre = uv_pred[:, :, :2]
    uv_pred_lat = uv_pred[:, :, 2:]
    uv_trans_pre = uv_trans[:, :, :3]
    uv_trans_lat = uv_trans[:, :, 3:]
    uv_pred_pre_rev = revtrans_points(uv_pred_pre * size, uv_trans_pre) / size
    uv_pred_lat_rev = revtrans_points(uv_pred_lat * size, uv_trans_lat) / size
    return torch.nn.functional.l1_loss(uv_pred_pre_rev, uv_pred_lat_rev, reduction='mean')

# Helper functions from the original context that are used by the main functions
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

class sample_version:
    gs = 'gs'
    hom = 'hom'
    sample = 'sample'
    auto = 'auto'
    db_size = 32560
    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]
    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg
    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size * cls.valid_options().index(version)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(0,1):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return im

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, image):
        if random.randint(0,1):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image

class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current
    def __call__(self, image):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, image):
        if random.randint(0,1):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image

class RandomHue(object):
    def __init__(self, delta=18.0):
        self.delta = delta
    def __call__(self, image):
        if random.randint(0,1):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image

class RandomBrightness(object):
    def __init__(self, delta=32):
        self.delta = delta
    def __call__(self, image):
        if random.randint(0,1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image

def get_aug_config(exclude_flip, base_scale=1.1, scale_factor=0.25, rot_factor=60, color_factor=0.2, gaussian_std=1):
    scale = get_m1to1_gaussian_rand(gaussian_std) * scale_factor + base_scale
    rot = get_m1to1_gaussian_rand(gaussian_std) * rot_factor if random.random() <= 0.8 else 0
    shift = [get_m1to1_gaussian_rand(gaussian_std), get_m1to1_gaussian_rand(gaussian_std)]
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5
    return scale, rot, shift, color_scale, do_flip

def get_m1to1_gaussian_rand(scale):
    r = 2
    while r < -1 or r > 1:
        r = np.random.normal(scale=scale)
    return r

def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape, shift_wh=None, mask=None):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
        if mask is not None:
            mask = mask[:, ::-1]
    trans, shift_xy = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, return_shift=True)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    if mask is not None:
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = (mask > 150).astype(np.uint8)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, inv=True)
    return img_patch, trans, inv_trans, mask, shift_xy

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift, shift_wh=None, inv=False, return_shift=False):
    src_w = src_width * scale
    src_h = src_height * scale
    if shift_wh is not None:
        shift_lim = (max((src_w - shift_wh[0]) / 2, 0), max((src_h - shift_wh[1]) / 2, 0))
        x_shift = shift[0] * shift_lim[0]
        y_shift = shift[1] * shift_lim[1]
    else:
        x_shift = y_shift = 0
    src_center = np.array([c_x + x_shift, c_y + y_shift], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    trans = trans.astype(np.float32)
    if return_shift:
        return trans, [x_shift/src_w, y_shift/src_h]
    return trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def revtrans_points(uv_point, trans):
    uv1 = torch.cat((uv_point, torch.ones_like(uv_point[:, :, :1])), 2)
    uv_crop = torch.bmm(trans, uv1.transpose(2, 1)).transpose(2, 1)[:, :, :2]
    return uv_crop

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register(self, name=None):
        def decorator(cls):
            if name is None:
                self._module_dict[cls.__name__] = cls
            else:
                self._module_dict[name] = cls
            return cls
        return decorator

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not in the {self._name} registry")
        return self._module_dict[name]

DATA_REGISTRY = Registry('DATA')
