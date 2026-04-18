# -*- coding: utf-8 -*-
"""
hand_mesh_renderer.py — 独立的手部 Mesh 渲染 / 可视化工具模块

该模块从 WiLoR-mini 项目中提取，可独立于 WiLoR pipeline 使用。
只要你的算法能输出以下信息，即可调用本模块进行可视化：
  - vertices:       手部 mesh 顶点 (N, 3)
  - faces:          mesh 面片索引 (F, 3)，如 MANO 模型的 faces
  - cam_t:          相机位移 (3,)
  - focal_length:   缩放后的焦距 (标量)
  - image:          原始图像 (H, W, 3) RGB uint8
  - is_right:       是否为右手 (bool/int)

依赖:，
    pip install numpy opencv-python trimesh pyrender torch

用法示例:
    from hand_mesh_renderer import HandMeshRenderer

    renderer = HandMeshRenderer(faces)  # faces: MANO faces (F, 3)

    # 渲染单只手的 mesh 叠加图
    vis_image = renderer.render_on_image(
        image, vertices, cam_t, focal_length, is_right=1
    )

    # 渲染多只手
    hands = [
        {"vertices": verts1, "cam_t": cam_t1, "focal_length": fl1, "is_right": 1},
        {"vertices": verts2, "cam_t": cam_t2, "focal_length": fl2, "is_right": 0},
    ]
    vis_image = renderer.render_multi_hands_on_image(image, hands)

    # 绘制 2D 关键点
    vis_image = renderer.draw_keypoints_2d(vis_image, keypoints_2d)

    # 导出 .obj mesh
    renderer.export_mesh(vertices, cam_t, "output.obj", is_right=1)

    # 保存最终图像
    import cv2
    cv2.imwrite("result.jpg", vis_image)
"""

import numpy as np
import cv2

import trimesh
import pyrender
import torch


# ============================================================
#  内部辅助函数（光照位姿计算）
# ============================================================

def _make_4x4_pose(R, t):
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def _make_translation(t):
    return _make_4x4_pose(torch.eye(3), t)


def _rotx(theta):
    return torch.tensor([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ], dtype=torch.float32)


def _roty(theta):
    return torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ], dtype=torch.float32)


def _rotz(theta):
    return torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=torch.float32)


def _make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx, Ry, Rz = _rotx(rx), _roty(ry), _rotz(rz)
    ops = {"xyz": Rz @ Ry @ Rx, "xzy": Ry @ Rz @ Rx,
           "yxz": Rz @ Rx @ Ry, "yzx": Rx @ Rz @ Ry,
           "zyx": Rx @ Ry @ Rz, "zxy": Ry @ Rx @ Rz}
    return _make_4x4_pose(ops[order], torch.zeros(3))


def _get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = _make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = _make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def _create_raymond_lights():
    """创建三盏 Raymond 方向光，提供均匀柔和的照明。"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


# ============================================================
#  MANO 额外面片（使 mesh 闭合 / watertight）
# ============================================================

MANO_WATERTIGHT_FACES = np.array([
    [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
    [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
    [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
    [120, 108, 78], [78, 108, 79]
])


# ============================================================
#  默认颜色
# ============================================================

COLOR_LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
COLOR_RIGHT_HAND = COLOR_LIGHT_PURPLE[::-1]   # 左右手用不同颜色便于区分


# ============================================================
#  HandMeshRenderer 类
# ============================================================

class HandMeshRenderer:
    """
    手部 Mesh 渲染器。

    初始化时传入 MANO faces（或其他 mesh 的面片索引），
    之后可反复调用渲染方法，无需重复初始化。

    Parameters
    ----------
    faces : np.ndarray, shape (F, 3)
        mesh 面片索引（如 mano.faces）。
    watertight : bool
        是否自动追加额外面片使 MANO mesh 闭合，默认 True。
    mesh_color : tuple
        默认 mesh 颜色 (R, G, B)，范围 [0, 1]。
    """

    def __init__(self, faces, watertight=True, mesh_color=COLOR_LIGHT_PURPLE):
        faces = np.array(faces).copy()
        if watertight:
            faces = np.concatenate([faces, MANO_WATERTIGHT_FACES], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]  # 左手翻转面法线
        self.mesh_color = mesh_color

    # ---- Trimesh 构建 ----

    def vertices_to_trimesh(self, vertices, camera_translation,
                            mesh_color=None, rot_axis=(1, 0, 0),
                            rot_angle=0, is_right=1):
        """
        将顶点 + 相机位移转换为 trimesh.Trimesh 对象。

        Parameters
        ----------
        vertices : np.ndarray, shape (V, 3)
        camera_translation : np.ndarray, shape (3,)
        mesh_color : tuple or None
            (R, G, B)，None 则使用默认颜色。
        is_right : int/bool
            1 = 右手，0 = 左手。

        Returns
        -------
        trimesh.Trimesh
        """
        if mesh_color is None:
            mesh_color = self.mesh_color
        vertex_colors = np.array([(*mesh_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation,
                                   self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation,
                                   self.faces_left.copy(), vertex_colors=vertex_colors)
        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    # ---- 导出 .obj ----

    def export_mesh(self, vertices, camera_translation, save_path,
                    mesh_color=None, is_right=1):
        """
        将手部 mesh 导出为 .obj 文件。

        Parameters
        ----------
        vertices : np.ndarray, shape (V, 3)
        camera_translation : np.ndarray, shape (3,)
        save_path : str
            输出文件路径，如 "hand.obj"
        is_right : int/bool
        """
        tmesh = self.vertices_to_trimesh(vertices, camera_translation,
                                         mesh_color=mesh_color, is_right=is_right)
        tmesh.export(save_path)

    # ---- Pyrender 离屏渲染 ----

    def render_rgba(self, vertices, cam_t=None, rot_axis=(1, 0, 0),
                    rot_angle=0, camera_z=3, mesh_color=None,
                    scene_bg_color=(0, 0, 0), render_res=(256, 256),
                    focal_length=None, is_right=1):
        """
        离屏渲染 mesh，返回 RGBA float 图像 [0, 1]。

        Parameters
        ----------
        vertices : np.ndarray, shape (V, 3)
        cam_t : np.ndarray or None, shape (3,)
        render_res : tuple (W, H)
        focal_length : float
        is_right : int/bool

        Returns
        -------
        np.ndarray, shape (H, W, 4), dtype float32, [0, 1]
        """
        if mesh_color is None:
            mesh_color = self.mesh_color

        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0)

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])

        # 左右手使用不同颜色
        if is_right:
            mesh_color = mesh_color[::-1]

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]),
                                        mesh_color, rot_axis, rot_angle,
                                        is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1],
                                           zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        # 添加光照
        self._add_point_lighting(scene, camera_node)
        self._add_lighting(scene, camera_node)
        for node in _create_raymond_lights():
            scene.add_node(node)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()
        return color

    # ---- 高层 API：单手渲染到图像 ----

    def render_on_image(self, image, vertices, cam_t, focal_length,
                        is_right=1, mesh_color=None):
        """
        将单只手的 mesh 渲染并叠加到原始图像上。

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), dtype uint8, RGB 格式
        vertices : np.ndarray, shape (V, 3)
        cam_t : np.ndarray, shape (3,)
        focal_length : float
        is_right : int/bool
        mesh_color : tuple or None

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8, BGR 格式（可直接 cv2.imwrite）
        """
        render_image = image.copy().astype(np.float32)[:, :, ::-1] / 255.0

        cam_view = self.render_rgba(
            vertices, cam_t=cam_t,
            render_res=[image.shape[1], image.shape[0]],
            focal_length=focal_length,
            is_right=is_right,
            mesh_color=mesh_color or self.mesh_color,
            scene_bg_color=(1, 1, 1),
        )
        # Alpha blending
        render_image = (render_image[:, :, :3] * (1 - cam_view[:, :, 3:])
                        + cam_view[:, :, :3] * cam_view[:, :, 3:])
        return (255 * render_image).astype(np.uint8)

    # ---- 高层 API：多手渲染到图像 ----

    def render_multi_hands_on_image(self, image, hands, mesh_color=None):
        """
        将多只手的 mesh 渲染并叠加到原始图像上。

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), dtype uint8, RGB 格式
        hands : list of dict
            每个 dict 包含:
              - "vertices":     np.ndarray (V, 3)
              - "cam_t":        np.ndarray (3,)
              - "focal_length": float
              - "is_right":     int/bool
        mesh_color : tuple or None

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8, BGR 格式
        """
        if mesh_color is None:
            mesh_color = self.mesh_color

        render_image = image.copy().astype(np.float32)[:, :, ::-1] / 255.0

        for hand in hands:
            cam_view = self.render_rgba(
                hand["vertices"],
                cam_t=hand["cam_t"],
                render_res=[image.shape[1], image.shape[0]],
                focal_length=hand["focal_length"],
                is_right=hand["is_right"],
                mesh_color=mesh_color,
                scene_bg_color=(1, 1, 1),
            )
            render_image = (render_image[:, :, :3] * (1 - cam_view[:, :, 3:])
                            + cam_view[:, :, :3] * cam_view[:, :, 3:])

        return (255 * render_image).astype(np.uint8)

    # ---- 2D 关键点绘制 ----

    @staticmethod
    def draw_keypoints_2d(image_bgr, keypoints_2d, radius=3, color=(0, 0, 255)):
        """
        在图像上绘制 2D 关键点。

        Parameters
        ----------
        image_bgr : np.ndarray, shape (H, W, 3), dtype uint8, BGR 格式
            （即 render_on_image / render_multi_hands_on_image 的输出）
        keypoints_2d : np.ndarray, shape (K, 2)
            关键点坐标 (x, y)。
        radius : int
        color : tuple (B, G, R)

        Returns
        -------
        np.ndarray (原地修改并返回)
        """
        for j in range(keypoints_2d.shape[0]):
            x, y = keypoints_2d[j]
            cv2.circle(image_bgr, (int(x), int(y)), radius, color, -1)
        return image_bgr

    # ---- 内部光照方法 ----

    @staticmethod
    def _add_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = _get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    @staticmethod
    def _add_point_lighting(scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = _get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
