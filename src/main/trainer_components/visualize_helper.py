import cv2
import torch
from src.utils.draw3d import draw_2d_skeleton, draw_3d_skeleton, draw_3d_mesh, draw_mesh
import numpy as np
from typing import Optional
import math



def perspective(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix

    Args:
        points (tensot): [Bx3xN] tensor of 3D points
        calibrations (tensor): [Bx4x4] Tensor of projection matrix

    Returns:
        tensor: [Bx3xN] Tensor of uvz coordinates in the image plane
    """
    if points.shape[1] == 2:
        points = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    z = points[:, 2:3].clone()
    points[:, :3] = points[:, :3] / z
    points1 = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    points_img = torch.bmm(calibrations, points1)
    points_img = torch.cat([points_img[:, :2], z], 1)

    return points_img

def normalization(img):
    img_channel = img.shape[0]
    img_f = torch.flatten(img, 1)
    min_, _ = torch.min(img_f, dim=-1)
    max_, _ = torch.max(img_f, dim=-1)
    min_ex = min_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    max_ex = max_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    img_n = (img - min_ex) / (max_ex - min_ex)
    return img_n


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # restore coordinates x and y
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals # BxJx2 , BxJx1

def get_final_preds(batch_heatmaps):
    # get predictions from heatmaps
    coords, maxvals = get_max_preds(batch_heatmaps) # BxJx2 , BxJx1
    # print(coords.shape)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing and fine-tuning
    # a simple implementation of Dark Pose algorithm
    # move coords to the direction of the gradient
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # print(preds.shape)

    kp_point = np.abs(preds)
    return kp_point, maxvals


# a static class for visualization during training and evaluation
class VisualizationHelper:
    """
    Helper class for visualization during training and evaluation
    """

    HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
        [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
        [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]

    @staticmethod
    def visualize_keypoints(rgb_image: torch.Tensor,
                            pred_heatmap: torch.Tensor,
                            target_heatmap: torch.Tensor,
                            heatmap_size: int = 56,
                            image_size: int = 224) -> torch.Tensor:
        """
        Visualize predicted and target keypoints on RGB image

        Args:
            rgb_image: RGB image tensor [3, H, W]
            pred_heatmap: Predicted heatmap [21, H, W]
            target_heatmap: Target heatmap [21, H, W]
            heatmap_size: Size of heatmap
            image_size: Size of image

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Get coordinates
        pred_heatmap_np = pred_heatmap.unsqueeze(0).cpu().detach().numpy()
        target_heatmap_np = target_heatmap.unsqueeze(0).cpu().detach().numpy()

        pred_coords, _ = get_final_preds(pred_heatmap_np)
        target_coords, _ = get_final_preds(target_heatmap_np)

        # Scale to image size
        scale = image_size / heatmap_size
        pred_coords = pred_coords * scale
        target_coords = target_coords * scale

        # Convert image to numpy
        img = np.ascontiguousarray(rgb_image.cpu().numpy().transpose(1, 2, 0))

        # Draw keypoints
        for i in range(21):
            # Predicted (yellow)
            cv2.circle(img, (int(pred_coords[0][i][0]), int(pred_coords[0][i][1])),
                       2, (1, 1, 0), -1)
            # Target (cyan)
            cv2.circle(img, (int(target_coords[0][i][0]), int(target_coords[0][i][1])),
                       2, (0, 1, 1), -1)

        # Draw connections
        for connection in VisualizationHelper.HAND_CONNECTIONS:
            p1_idx, p2_idx = connection

            # Predicted connections (yellow)
            pred_p1 = (int(pred_coords[0][p1_idx][0]), int(pred_coords[0][p1_idx][1]))
            pred_p2 = (int(pred_coords[0][p2_idx][0]), int(pred_coords[0][p2_idx][1]))
            cv2.line(img, pred_p1, pred_p2, (1, 1, 0), 1)

            # Target connections (cyan)
            target_p1 = (int(target_coords[0][p1_idx][0]), int(target_coords[0][p1_idx][1]))
            target_p2 = (int(target_coords[0][p2_idx][0]), int(target_coords[0][p2_idx][1]))
            cv2.line(img, target_p1, target_p2, (0, 1, 1), 1)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        img_tensor = normalization(img_tensor)

        return img_tensor

    @staticmethod
    def visualize_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Visualize segmentation mask

        Args:
            mask: Mask tensor [1, H, W] or [3, H, W]

        Returns:
            Visualization tensor [3, H, W]
        """
        if mask.shape[0] == 1:
            mask = mask.repeat(3, 1, 1)
        return normalization(mask)

    @staticmethod
    def visualize_heatmap_sum(heatmap: torch.Tensor) -> torch.Tensor:
        """
        Visualize sum of all keypoint heatmaps

        Args:
            heatmap: Heatmap tensor [21, H, W]

        Returns:
            Visualization tensor [3, H, W]
        """
        heatmap_sum = heatmap.sum(dim=0, keepdim=True)
        heatmap_sum_3 = heatmap_sum.repeat(3, 1, 1)
        return normalization(heatmap_sum_3)

    @staticmethod
    def apply_mask_to_image(image: torch.Tensor, mask: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
        """
        Apply mask to image

        Args:
            image: Image tensor [3, H, W]
            mask: Mask tensor [3, H, W]
            threshold: Threshold for masking

        Returns:
            Masked image tensor [3, H, W]
        """
        masked = image * mask
        masked[masked > threshold] = 0
        return masked

    @staticmethod
    def visualize_joint_img(rgb_image: torch.Tensor,
                            pred_joint_img: torch.Tensor,
                            gt_joint_img: torch.Tensor,
                            image_size: int) -> torch.Tensor:
        """
        Visualize 2D joint coordinates (joint_img) on RGB image

        Args:
            rgb_image: RGB image tensor [3, H, W], normalized to [0, 1]
            pred_joint_img: Predicted 2D joint coordinates [21, 2], normalized to [0, 1]
            gt_joint_img: Ground truth 2D joint coordinates [21, 2], normalized to [0, 1]
            image_size: Size of the image (H or W, assuming square)

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Convert image to numpy
        img = np.ascontiguousarray(rgb_image.cpu().numpy().transpose(1, 2, 0))

        # Scale coordinates to image size
        pred_coords = pred_joint_img.cpu().detach().numpy() * image_size
        gt_coords = gt_joint_img.cpu().detach().numpy() * image_size

        # Draw connections first (so they appear behind the points)
        for connection in VisualizationHelper.HAND_CONNECTIONS:
            p1_idx, p2_idx = connection

            # GT connections (orange: RGB in 0-1 range)
            gt_p1 = (int(gt_coords[p1_idx, 0]), int(gt_coords[p1_idx, 1]))
            gt_p2 = (int(gt_coords[p2_idx, 0]), int(gt_coords[p2_idx, 1]))
            cv2.line(img, gt_p1, gt_p2, (1.0, 0.5, 0.0), 1)  # Orange

            # Predicted connections (green)
            pred_p1 = (int(pred_coords[p1_idx, 0]), int(pred_coords[p1_idx, 1]))
            pred_p2 = (int(pred_coords[p2_idx, 0]), int(pred_coords[p2_idx, 1]))
            cv2.line(img, pred_p1, pred_p2, (0.0, 1.0, 0.0), 1)  # Green

        # Draw keypoints on top
        for i in range(21):
            # GT (orange)
            cv2.circle(img, (int(gt_coords[i, 0]), int(gt_coords[i, 1])),
                       3, (1.0, 0.5, 0.0), -1)
            # Predicted (green)
            cv2.circle(img, (int(pred_coords[i, 0]), int(pred_coords[i, 1])),
                       3, (0.0, 1.0, 0.0), -1)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        img_tensor = normalization(img_tensor)

        return img_tensor

    @staticmethod
    def visualize_joint_cam(rgb_image: torch.Tensor,
                            pred_joint_cam: torch.Tensor,
                            gt_joint_cam: torch.Tensor,
                            pred_root: torch.Tensor,
                            gt_root: torch.Tensor,
                            calib: torch.Tensor,
                            image_size: int) -> torch.Tensor:
        """
        Visualize 3D joint coordinates (joint_cam) projected to 2D on RGB image

        Args:
            rgb_image: RGB image tensor [3, H, W], normalized to [0, 1]
            pred_joint_cam: Predicted 3D joint coordinates [21, 3], relative coordinates
            gt_joint_cam: Ground truth 3D joint coordinates [21, 3], relative coordinates
            pred_root: Predicted root coordinate [3], for recovering absolute coordinates
            gt_root: Ground truth root coordinate [3], for recovering absolute coordinates
            calib: Camera calibration matrix [4, 4]
            image_size: Size of the image

        Returns:
            Visualization image tensor [3, H, W]
        """

        # Convert image to numpy
        img = np.ascontiguousarray(rgb_image.cpu().numpy().transpose(1, 2, 0))

        # Recover absolute 3D coordinates: joint_cam * 0.2 + root
        pred_joint_3d = pred_joint_cam * 0.2 + pred_root.unsqueeze(0)  # [21, 3]
        gt_joint_3d = gt_joint_cam * 0.2 + gt_root.unsqueeze(0)  # [21, 3]

        # Project to 2D using perspective projection
        # perspective expects [B, 3, N] format
        pred_joint_3d_t = pred_joint_3d.unsqueeze(0).permute(0, 2, 1)  # [1, 3, 21]
        gt_joint_3d_t = gt_joint_3d.unsqueeze(0).permute(0, 2, 1)  # [1, 3, 21]
        calib_batch = calib.unsqueeze(0)  # [1, 4, 4]
        # print("gt_joint_3d_t:",gt_joint_3d_t.shape,"gt_joint_3d_t",calib_batch.shape)
        pred_proj = perspective(pred_joint_3d_t, calib_batch)[0].cpu().detach().numpy().T  # [21, 3]
        gt_proj = perspective(gt_joint_3d_t, calib_batch)[0].cpu().detach().numpy().T  # [21, 3]
        # print("gt_proj",gt_proj.shape)
        # Draw connections first
        for connection in VisualizationHelper.HAND_CONNECTIONS:
            p1_idx, p2_idx = connection

            # GT connections (orange)
            gt_p1 = (int(gt_proj[p1_idx, 0]), int(gt_proj[p1_idx, 1]))
            gt_p2 = (int(gt_proj[p2_idx, 0]), int(gt_proj[p2_idx, 1]))
            cv2.line(img, gt_p1, gt_p2, (1.0, 0.5, 0.0), 1)  # Orange

            # Predicted connections (green)
            pred_p1 = (int(pred_proj[p1_idx, 0]), int(pred_proj[p1_idx, 1]))
            pred_p2 = (int(pred_proj[p2_idx, 0]), int(pred_proj[p2_idx, 1]))
            cv2.line(img, pred_p1, pred_p2, (0.0, 1.0, 0.0), 1)  # Green

        # Draw keypoints on top
        for i in range(21):
            # GT (orange)
            cv2.circle(img, (int(gt_proj[i, 0]), int(gt_proj[i, 1])),
                       3, (1.0, 0.5, 0.0), -1)
            # Predicted (green)
            cv2.circle(img, (int(pred_proj[i, 0]), int(pred_proj[i, 1])),
                       3, (0.0, 1.0, 0.0), -1)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        img_tensor = normalization(img_tensor)

        return img_tensor

    @staticmethod
    def visualize_verts(rgb_image: torch.Tensor,
                        verts: torch.Tensor,
                        root: torch.Tensor,
                        calib: torch.Tensor,
                        alpha: float = 0.3,
                        faces: Optional[torch.Tensor] = None,
                        colorize_parts: bool = True) -> torch.Tensor:
        """
        Visualize mesh vertices projected to 2D on RGB image with transparency

        Args:
            rgb_image: RGB image tensor [3, H, W], normalized to [0, 1]
            verts: Mesh vertices [778, 3], relative coordinates
            root: Root coordinate [3], for recovering absolute coordinates
            calib: Camera calibration matrix [4, 4]
            alpha: Transparency for vertex points (default: 0.3)
            faces: Mesh faces [F, 3] (optional). If provided, draws wireframe.
            colorize_parts: If True, draw different hand parts with different colors

        Returns:
            Visualization image tensor [3, H, W]
        """

        # Convert image to numpy and create a copy for blending
        img = np.ascontiguousarray(rgb_image.cpu().numpy().transpose(1, 2, 0)).copy()
        overlay = img.copy()

        # Recover absolute 3D coordinates: verts * 0.2 + root
        verts_3d = verts * 0.2 + root.unsqueeze(0)  # [778, 3]

        # Project to 2D using perspective projection
        verts_3d_t = verts_3d.unsqueeze(0).permute(0, 2, 1)  # [1, 3, 778]
        calib_batch = calib.unsqueeze(0)  # [1, 4, 4]

        verts_proj = perspective(verts_3d_t, calib_batch)[0].cpu().detach().numpy().T  # [778, 3]

        # Build vertex-to-part mapping dictionary (FINGER_DICT)
        # This maps each vertex index to its corresponding hand part
        FINGER_DICT = {}

        # Palm vertices (complex non-contiguous ranges)
        palm_index = [i for i in range(46)] + [i for i in range(50, 56)] + [i for i in range(60, 86)] + \
                     [i for i in range(88, 133)] + [i for i in range(137, 155)] + [i for i in range(157, 164)] + \
                     [i for i in range(168, 174)] + [i for i in range(178, 189)] + [i for i in range(190, 194)] + \
                     [i for i in range(196, 212)] + [i for i in range(214, 221)] + [i for i in range(227, 237)] + \
                     [i for i in range(239, 245)] + [i for i in range(246, 261)] + [i for i in range(262, 272)] + \
                     [i for i in range(274, 280)] + [i for i in range(284, 294)] + [i for i in range(769, 778)]

        # Index finger vertices (complex non-contiguous ranges)
        index_index = [i for i in range(46, 50)] + [i for i in range(56, 60)] + [i for i in range(86, 88)] + \
                      [i for i in range(133, 137)] + [i for i in range(155, 157)] + [i for i in range(164, 168)] + \
                      [i for i in range(174, 178)] + [i for i in range(189, 190)] + [i for i in range(194, 196)] + \
                      [i for i in range(212, 214)] + [i for i in range(221, 227)] + [i for i in range(237, 239)] + \
                      [i for i in range(245, 246)] + [i for i in range(261, 262)] + [i for i in range(272, 274)] + \
                      [i for i in range(280, 284)] + [i for i in range(294, 356)]

        # Populate FINGER_DICT
        for index in palm_index:
            FINGER_DICT[index] = 'palm'
        for index in index_index:
            FINGER_DICT[index] = 'index'
        for index in range(697, 769):
            FINGER_DICT[index] = 'thumb'
        for index in range(356, 468):
            FINGER_DICT[index] = 'middle'
        for index in range(468, 580):
            FINGER_DICT[index] = 'ring'
        for index in range(580, 697):
            FINGER_DICT[index] = 'pinky'

        # Define distinct colors for each part (RGB in 0-1 range)
        part_colors = {
            'palm': (1.0, 0.8, 0.6),    # Light peach - 手掌
            'thumb': (1.0, 0.0, 0.0),   # Red - 拇指
            'index': (0.0, 1.0, 0.0),   # Green - 食指
            'middle': (0.0, 0.0, 1.0),  # Blue - 中指
            'ring': (1.0, 1.0, 0.0),    # Yellow - 无名指
            'pinky': (1.0, 0.0, 1.0),   # Magenta - 小指
        }


        # Draw wireframe if faces are provided
        if faces is not None:
            faces_np = faces.cpu().numpy()
            verts_proj_2d = verts_proj[:, :2]

            if colorize_parts:
                # Draw faces with different colors based on which part they belong to
                for face_idx, face in enumerate(faces_np):
                    # Determine which part this face belongs to based on its vertices
                    # Use the first vertex of the face to determine the part
                    v0_idx = face[0]

                    # Find which part this vertex belongs to using FINGER_DICT
                    part_name = FINGER_DICT.get(v0_idx, None)
                    if part_name is not None:
                        color = part_colors[part_name]
                    else:
                        color = (0.0, 0.5, 1.0)  # Default blue for unmapped vertices

                    # Get the three vertices of this face
                    pts = verts_proj_2d[face].astype(np.int32)
                    pts = pts.reshape((-1, 1, 2))  # Reshape for cv2.polylines
                    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)
            else:
                # Draw all faces with single color
                pts = verts_proj_2d[faces_np].astype(np.int32)
                cv2.polylines(overlay, list(pts), isClosed=True, color=(0.0, 0.5, 1.0), thickness=1)
        else:
            # Draw vertices as small circles
            if colorize_parts:
                # Draw each vertex with color based on FINGER_DICT
                for i in range(verts_proj.shape[0]):
                    part_name = FINGER_DICT.get(i, None)
                    if part_name is not None:
                        color = part_colors[part_name]
                    else:
                        color = (0.0, 0.5, 1.0)  # Default blue for unmapped vertices

                    x, y = int(verts_proj[i, 0]), int(verts_proj[i, 1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(overlay, (x, y), 1, color, -1)
            else:
                # Draw all vertices with single color
                for i in range(verts_proj.shape[0]):
                    x, y = int(verts_proj[i, 0]), int(verts_proj[i, 1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        cv2.circle(overlay, (x, y), 1, (0.0, 0.5, 1.0), -1)

        # Blend overlay with original image using alpha transparency
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Convert back to tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        img_tensor = normalization(img_tensor)

        return img_tensor


    # ==================== Matplotlib-based Visualization Methods ====================
    # These methods use draw3d.py for high-quality visualization (mainly for evaluation)

    @staticmethod
    def _tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor image to numpy array for draw3d functions

        Args:
            tensor: Image tensor [3, H, W], normalized to [0, 1]

        Returns:
            numpy array [H, W, 3], uint8, RGB format, range [0, 255]
        """
        # tensor is [3, H, W] in [0, 1] range
        tensor = tensor.cpu().detach()

        # Normalize to [0, 1] if needed (e.g. if input is [-1, 1])
        if tensor.min() < 0 or tensor.max() > 1:
            t_min = tensor.min()
            t_max = tensor.max()
            if t_max > t_min:
                tensor = (tensor - t_min) / (t_max - t_min)
            else:
                tensor = torch.zeros_like(tensor)

        img = tensor.numpy().transpose(1, 2, 0)  # [H, W, 3]
        img = (img * 255).astype(np.uint8)
        return img

    @staticmethod
    def _numpy_to_tensor_image(img: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array image to torch tensor

        Args:
            img: numpy array [H, W, 3] or [H, W, 4], uint8, range [0, 255]

        Returns:
            tensor [3, H, W], normalized to [0, 1]
        """
        # Handle RGBA images (from matplotlib)
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Drop alpha channel

        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        return tensor

    @staticmethod
    def _extract_cam_param(calib: torch.Tensor) -> np.ndarray:
        """
        Extract 3x3 camera intrinsic matrix from 4x4 calibration matrix

        Args:
            calib: Camera calibration matrix [4, 4]

        Returns:
            Camera intrinsic matrix [3, 3], numpy array
        """
        return calib[:3, :3].cpu().numpy()

    @staticmethod
    def _recover_absolute_coords(relative_coords: torch.Tensor,
                                 root: torch.Tensor,
                                 scale: float = 0.2) -> np.ndarray:
        """
        Recover absolute 3D coordinates from relative coordinates

        Args:
            relative_coords: Relative coordinates [N, 3]
            root: Root coordinate [3]
            scale: Scale factor (default: 0.2)

        Returns:
            Absolute coordinates [N, 3], numpy array
        """
        absolute = relative_coords * scale + root.unsqueeze(0)
        return absolute.cpu().numpy()

    @staticmethod
    def visualize_2d_skeleton_matplotlib(rgb_image: torch.Tensor,
                                         pose_uv: torch.Tensor,
                                         image_size: int = 224) -> torch.Tensor:
        """
        Visualize 2D skeleton using matplotlib (colorful version from draw3d)

        Args:
            rgb_image: RGB image tensor [3, H, W], normalized to [0, 1]
            pose_uv: 2D joint coordinates [21, 2], normalized to [0, 1]
            image_size: Size of the image

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Convert image to numpy
        img_np = VisualizationHelper._tensor_to_numpy_image(rgb_image)

        # Scale coordinates to image size
        pose_uv_scaled = (pose_uv.cpu().detach().numpy() * image_size).astype(np.int32)

        # Draw skeleton using draw3d
        result_img = draw_2d_skeleton(img_np, pose_uv_scaled)

        # Convert back to tensor
        return VisualizationHelper._numpy_to_tensor_image(result_img)

    @staticmethod
    def visualize_3d_skeleton_matplotlib(pose_cam: torch.Tensor,
                                         root: torch.Tensor,
                                         image_size: tuple = (224, 224)) -> torch.Tensor:
        """
        Visualize 3D skeleton using matplotlib

        Args:
            pose_cam: 3D joint coordinates [21, 3], relative coordinates
            root: Root coordinate [3]
            image_size: Image size (H, W)

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Recover absolute coordinates
        pose_cam_abs = VisualizationHelper._recover_absolute_coords(pose_cam, root)

        # Draw 3D skeleton using draw3d
        result_img = draw_3d_skeleton(pose_cam_abs, image_size)

        # Convert back to tensor
        return VisualizationHelper._numpy_to_tensor_image(result_img)

    @staticmethod
    def visualize_3d_mesh_matplotlib(mesh_verts: torch.Tensor,
                                     root: torch.Tensor,
                                     faces: torch.Tensor,
                                     image_size: tuple = (224, 224)) -> torch.Tensor:
        """
        Visualize 3D mesh using matplotlib

        Args:
            mesh_verts: Mesh vertices [V, 3], relative coordinates
            root: Root coordinate [3]
            faces: Mesh faces [F, 3]
            image_size: Image size (H, W)

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Recover absolute coordinates
        mesh_verts_abs = VisualizationHelper._recover_absolute_coords(mesh_verts, root)

        # Convert faces to numpy
        faces_np = faces.cpu().numpy()

        # Draw 3D mesh using draw3d
        result_img = draw_3d_mesh(mesh_verts_abs, image_size, faces_np)

        # Convert back to tensor
        return VisualizationHelper._numpy_to_tensor_image(result_img)

    @staticmethod
    def visualize_2d_mesh_matplotlib(rgb_image: torch.Tensor,
                                     mesh_verts: torch.Tensor,
                                     root: torch.Tensor,
                                     calib: torch.Tensor,
                                     faces: torch.Tensor) -> torch.Tensor:
        """
        Visualize 2D mesh projection using matplotlib

        Args:
            rgb_image: RGB image tensor [3, H, W], normalized to [0, 1]
            mesh_verts: Mesh vertices [V, 3], relative coordinates
            root: Root coordinate [3]
            calib: Camera calibration matrix [4, 4]
            faces: Mesh faces [F, 3]

        Returns:
            Visualization image tensor [3, H, W]
        """
        # Convert image to numpy
        img_np = VisualizationHelper._tensor_to_numpy_image(rgb_image)

        # Recover absolute coordinates
        mesh_verts_abs = VisualizationHelper._recover_absolute_coords(mesh_verts, root)

        # Extract camera intrinsic matrix
        cam_param = VisualizationHelper._extract_cam_param(calib)

        # Convert faces to numpy
        faces_np = faces.cpu().numpy()

        # Draw 2D mesh projection using draw3d
        result_img = draw_mesh(img_np, cam_param, mesh_verts_abs, faces_np)

        # Convert back to tensor
        return VisualizationHelper._numpy_to_tensor_image(result_img)

