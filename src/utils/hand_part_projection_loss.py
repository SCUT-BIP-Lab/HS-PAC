"""
Hand Part Projection Loss - Pure PyTorch Implementation

This module provides functions to compute hand part projection loss
using only PyTorch operations (no cv2 or numpy).

The key improvement is using mesh faces to render filled surface masks
instead of just projecting vertices as points.
"""

import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image



def _get_hand_part_indices():
    """
    Get vertex indices for each hand part

    Returns:
        dict: Dictionary mapping part names to vertex indices
    """
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

    # Thumb vertices
    thumb_index = list(range(697, 769))

    # Middle finger vertices
    middle_index = list(range(356, 468))

    # Ring finger vertices
    ring_index = list(range(468, 580))

    # Pinky finger vertices
    pinky_index = list(range(580, 697))

    return {
        'palm': palm_index,
        'thumb': thumb_index,
        'index': index_index,
        'middle': middle_index,
        'ring': ring_index,
        'pinky': pinky_index
    }


# Global cache for vertex-to-part mapping (CUDA Graph compatible)
_VERTEX_TO_PART_CACHE = {}
_FACE_TO_PART_CACHE = {}

def _get_vertex_to_part_mapping_cached(device):
    """
    Get cached vertex-to-part mapping tensor (CUDA Graph compatible)

    This function caches the mapping to avoid creating new tensors during CUDA Graph capture.

    Args:
        device: torch device

    Returns:
        tuple: (vertex_to_part_id tensor [778], part_names list)
    """
    device_key = str(device)

    if device_key not in _VERTEX_TO_PART_CACHE:
        # Create mapping on CPU first, then move to device
        part_indices = _get_hand_part_indices()
        part_names = list(part_indices.keys())

        # Create vertex-to-part mapping on CPU
        vertex_to_part_id_cpu = torch.zeros(778, dtype=torch.long)
        for part_id, part_name in enumerate(part_names):
            vertex_indices = torch.tensor(part_indices[part_name], dtype=torch.long)
            vertex_to_part_id_cpu[vertex_indices] = part_id

        # Move to target device and cache
        vertex_to_part_id = vertex_to_part_id_cpu.to(device)
        _VERTEX_TO_PART_CACHE[device_key] = (vertex_to_part_id, part_names)

    return _VERTEX_TO_PART_CACHE[device_key]


def _get_face_to_part_masks_cached(faces, device):
    """
    Get cached face-to-part masks (CUDA Graph compatible)

    Pre-computes boolean masks for each part to avoid boolean indexing during graph capture.

    Args:
        faces: Face tensor [F, 3]
        device: torch device

    Returns:
        tuple: (part_face_indices_list, part_names)
            - part_face_indices_list: List of tensors, each containing face indices for a part
            - part_names: List of part names
    """
    cache_key = f"{device}_{faces.shape[0]}"

    if cache_key not in _FACE_TO_PART_CACHE:
        # Get vertex-to-part mapping
        vertex_to_part_id, part_names = _get_vertex_to_part_mapping_cached(device)

        # Map faces to parts based on first vertex
        face_part_ids = vertex_to_part_id[faces[:, 0]]  # [F]

        # Pre-compute face indices for each part (on CPU to avoid graph capture issues)
        part_face_indices_list = []
        for part_id in range(len(part_names)):
            # Find indices where face belongs to this part
            part_mask = (face_part_ids == part_id)
            part_face_indices = torch.nonzero(part_mask, as_tuple=True)[0]  # [F_part]
            part_face_indices_list.append(part_face_indices)

        _FACE_TO_PART_CACHE[cache_key] = (part_face_indices_list, part_names)

    return _FACE_TO_PART_CACHE[cache_key]

def perspective(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix

    Args:
        points (tensot): [Bx3xN] tensor of 3D points
        calibrations (tensor): [Bx4x4] Tensor of projection matrix

    Returns:
        tensor: [Bx3xN] Tensor of uvz coordinates in the image plane
    """
    B, C, N = points.shape
    device = points.device
    dtype = points.dtype

    if C == 2:
        # Create ones tensor with same dtype and device (CUDA Graph compatible)
        ones = torch.ones(B, 1, N, dtype=dtype, device=device)
        points = torch.cat([points, ones], dim=1)

    z = points[:, 2:3].clone()
    # Avoid inplace operation: create normalized points directly
    points_normalized = points / z

    # Create homogeneous coordinates (CUDA Graph compatible)
    ones = torch.ones(B, 1, N, dtype=dtype, device=device)
    points1 = torch.cat([points_normalized, ones], dim=1)

    points_img = torch.bmm(calibrations, points1)
    points_img = torch.cat([points_img[:, :2], z], dim=1)

    return points_img


def _load_mano_faces(mano_path: str = None):
    """
    Load MANO mesh faces

    Args:
        mano_path: Path to MANO template directory. If None, uses default path.

    Returns:
        torch.Tensor: Faces array [F, 3] where F is number of faces
    """
    if mano_path is None:
        # Default path based on project structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        mano_path = os.path.join(project_root, 'template')

    faces_path = os.path.join(mano_path, 'right_faces.npy')

    import numpy as np
    faces = np.load(faces_path)
    return torch.from_numpy(faces).long()


def _rasterize_triangle(v0, v1, v2, H, W, device):
    """
    Rasterize a single triangle using barycentric coordinates

    Args:
        v0, v1, v2: Triangle vertices [2] (x, y coordinates)
        H, W: Image dimensions
        device: torch device

    Returns:
        torch.Tensor: Binary mask [H, W] with triangle filled
    """
    # Get bounding box
    x_min = max(0, int(torch.min(torch.stack([v0[0], v1[0], v2[0]])).item()))
    x_max = min(W-1, int(torch.max(torch.stack([v0[0], v1[0], v2[0]])).item()))
    y_min = max(0, int(torch.min(torch.stack([v0[1], v1[1], v2[1]])).item()))
    y_max = min(H-1, int(torch.max(torch.stack([v0[1], v1[1], v2[1]])).item()))

    if x_max < x_min or y_max < y_min:
        return None

    # Create grid of points in bounding box
    y_coords, x_coords = torch.meshgrid(
        torch.arange(y_min, y_max + 1, device=device),
        torch.arange(x_min, x_max + 1, device=device),
        indexing='ij'
    )

    # Compute barycentric coordinates
    # Using the formula: P = w0*v0 + w1*v1 + w2*v2, where w0+w1+w2=1
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])

    if abs(denom) < 1e-10:  # Degenerate triangle
        return None

    w0 = ((v1[1] - v2[1]) * (x_coords - v2[0]) + (v2[0] - v1[0]) * (y_coords - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (x_coords - v2[0]) + (v0[0] - v2[0]) * (y_coords - v2[1])) / denom
    w2 = 1 - w0 - w1

    # Point is inside triangle if all barycentric coordinates are >= 0
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    return (x_coords[inside], y_coords[inside])



def _rasterize_triangles_batch_optimized(verts_2d: torch.Tensor,
                                          faces: torch.Tensor,
                                          part_face_indices: torch.Tensor,
                                          H: int, W: int,
                                          soft_rasterization: bool = True,
                                          sigma: float = 1.0) -> torch.Tensor:
    """
    Optimized batch triangle rasterization using vectorized operations (CUDA Graph compatible)

    Supports soft rasterization for differentiable rendering.

    Args:
        verts_2d: Projected 2D vertices [B, 778, 2]
        faces: Mesh faces [F, 3]
        part_face_indices: Integer indices of faces belonging to this part [F_part]
        H, W: Image dimensions
        soft_rasterization: If True, use soft (differentiable) rasterization. If False, use hard thresholding.
        sigma: Temperature parameter for soft rasterization. Smaller values make it sharper (closer to hard).
               Typical range: 0.1 (very sharp) to 10.0 (very soft). Default: 1.0

    Returns:
        torch.Tensor: Masks [B, H, W] (binary if soft_rasterization=False, continuous [0,1] if True)
    """
    B = verts_2d.shape[0]
    device = verts_2d.device

    # Filter faces for this part using integer indexing (CUDA Graph compatible)
    if part_face_indices.shape[0] == 0:
        return torch.zeros(B, H, W, device=device, dtype=torch.float32)

    part_faces = faces[part_face_indices]  # [F_part, 3]

    # Get triangle vertices: [B, F_part, 3, 2]
    v0 = verts_2d[:, part_faces[:, 0], :]  # [B, F_part, 2]
    v1 = verts_2d[:, part_faces[:, 1], :]  # [B, F_part, 2]
    v2 = verts_2d[:, part_faces[:, 2], :]  # [B, F_part, 2]

    # Compute bounding boxes for all triangles: [B, F_part, 4]
    x_min = torch.min(torch.min(v0[:, :, 0], v1[:, :, 0]), v2[:, :, 0]).clamp(0, W-1).long()
    x_max = torch.max(torch.max(v0[:, :, 0], v1[:, :, 0]), v2[:, :, 0]).clamp(0, W-1).long()
    y_min = torch.min(torch.min(v0[:, :, 1], v1[:, :, 1]), v2[:, :, 1]).clamp(0, H-1).long()
    y_max = torch.max(torch.max(v0[:, :, 1], v1[:, :, 1]), v2[:, :, 1]).clamp(0, H-1).long()

    # Create pixel grid [H, W, 2]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]

    # Process in chunks to avoid OOM
    chunk_size = 200
    num_faces = part_faces.shape[0]

    # Collect all chunk masks first, then aggregate (avoid inplace operations)
    batch_masks = []

    for b in range(B):
        chunk_masks_list = []

        for start_idx in range(0, num_faces, chunk_size):
            end_idx = min(start_idx + chunk_size, num_faces)

            # Get triangle vertices for this chunk
            v0_chunk = v0[b, start_idx:end_idx]  # [chunk, 2]
            v1_chunk = v1[b, start_idx:end_idx]
            v2_chunk = v2[b, start_idx:end_idx]

            # Compute barycentric coordinates for all pixels and triangles
            # [H, W, 1, 2] - [1, 1, chunk, 2] -> [H, W, chunk, 2]
            denom = ((v1_chunk[:, 1] - v2_chunk[:, 1]) * (v0_chunk[:, 0] - v2_chunk[:, 0]) +
                     (v2_chunk[:, 0] - v1_chunk[:, 0]) * (v0_chunk[:, 1] - v2_chunk[:, 1]))  # [chunk]

            # Expand pixel coords: [H, W, 2] -> [H, W, 1, 2]
            px = pixel_coords[:, :, None, 0]  # [H, W, 1]
            py = pixel_coords[:, :, None, 1]  # [H, W, 1]

            # Compute barycentric coordinates (handle degenerate triangles with safe division)
            # Add small epsilon to avoid division by zero, degenerate triangles will have invalid barycentric coords
            safe_denom = denom + 1e-10
            w0 = ((v1_chunk[:, 1] - v2_chunk[:, 1]) * (px - v2_chunk[:, 0]) +
                  (v2_chunk[:, 0] - v1_chunk[:, 0]) * (py - v2_chunk[:, 1])) / safe_denom  # [H, W, chunk]
            w1 = ((v2_chunk[:, 1] - v0_chunk[:, 1]) * (px - v2_chunk[:, 0]) +
                  (v0_chunk[:, 0] - v2_chunk[:, 0]) * (py - v2_chunk[:, 1])) / safe_denom
            w2 = 1 - w0 - w1

            if soft_rasterization:
                # Soft rasterization: use sigmoid to create smooth, differentiable boundaries
                # The minimum barycentric coordinate determines how "inside" a pixel is
                # If all coords >= 0, pixel is inside; we use min to capture this
                min_barycentric = torch.min(torch.min(w0, w1), w2)  # [H, W, chunk]

                # Apply sigmoid: sigmoid(sigma * min_barycentric)
                # - When min_barycentric > 0 (inside): sigmoid -> 1
                # - When min_barycentric < 0 (outside): sigmoid -> 0
                # - sigma controls sharpness: larger sigma = sharper transition
                inside_soft = torch.sigmoid(sigma * min_barycentric)  # [H, W, chunk]

                # Aggregate across triangles: max pooling (any triangle covers the pixel)
                # Use soft max approximation for differentiability
                chunk_mask = inside_soft.max(dim=-1)[0]  # [H, W]
                chunk_masks_list.append(chunk_mask)
            else:
                # Hard rasterization: original non-differentiable version
                inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)  # [H, W, chunk]
                chunk_mask = inside.any(dim=-1).float()
                chunk_masks_list.append(chunk_mask)

        # Aggregate all chunks for this batch element (non-inplace)
        if len(chunk_masks_list) > 0:
            stacked_masks = torch.stack(chunk_masks_list, dim=0)  # [num_chunks, H, W]
            batch_mask = stacked_masks.max(dim=0)[0]  # [H, W]
        else:
            batch_mask = torch.zeros(H, W, device=device, dtype=torch.float32)

        batch_masks.append(batch_mask)

    # Stack all batch masks
    masks = torch.stack(batch_masks, dim=0)  # [B, H, W]

    return masks


def compute_hand_part_projection_loss_with_faces_batch(pred_verts: torch.Tensor,
                                                        gt_verts: torch.Tensor,
                                                        pred_root: torch.Tensor,
                                                        gt_root: torch.Tensor,
                                                        calib: torch.Tensor,
                                                        faces: torch.Tensor = None,
                                                        image_size: tuple = (224, 224),
                                                        loss_type: str = 'dice',
                                                        mano_path: str = None,
                                                        visualize_first: bool = False,
                                                        save_dir: str = './hand_part_masks_vis',
                                                        original_img: torch.Tensor = None,
                                                        soft_rasterization: bool = True,
                                                        sigma: float = 10.0) -> torch.Tensor:
    """
    Compute projection loss for each hand part using mesh surface rendering (optimized batch version)

    Args:
        pred_verts: Predicted mesh vertices [B, 778, 3], relative coordinates
        gt_verts: Ground truth mesh vertices [B, 778, 3], relative coordinates
        pred_root: Predicted root coordinate [B, 3]
        gt_root: Ground truth root coordinate [B, 3]
        calib: Camera calibration matrix [B, 4, 4]
        faces: Mesh faces [F, 3]. If None, will load from MANO template
        image_size: Image size (H, W)
        loss_type: Type of loss ('dice', 'iou', 'bce', 'mse')
        mano_path: Path to MANO template directory
        soft_rasterization: If True, use differentiable soft rasterization. Default: True
        sigma: Temperature parameter for soft rasterization (larger = sharper). Default: 1.0

    Returns:
        torch.Tensor: Average loss across batch (fully differentiable if soft_rasterization=True)
    """
    B = pred_verts.shape[0]
    H, W = image_size
    device = pred_verts.device

    # Load faces if not provided
    if faces is None:
        faces = _load_mano_faces(mano_path)
    faces = faces.to(device)

    # ============ Batch projection ============
    # Recover absolute 3D coordinates: verts * 0.2 + root
    pred_verts_3d = pred_verts * 0.2 + pred_root.unsqueeze(1)  # [B, 778, 3]
    gt_verts_3d = gt_verts * 0.2 + gt_root.unsqueeze(1)  # [B, 778, 3]

    # Project to 2D using perspective projection (batch)
    pred_verts_3d_t = pred_verts_3d.permute(0, 2, 1)  # [B, 3, 778]
    gt_verts_3d_t = gt_verts_3d.permute(0, 2, 1)  # [B, 3, 778]

    pred_verts_proj = perspective(pred_verts_3d_t, calib)  # [B, 3, 778]
    gt_verts_proj = perspective(gt_verts_3d_t, calib)  # [B, 3, 778]

    pred_verts_2d = pred_verts_proj[:, :2, :].permute(0, 2, 1)  # [B, 778, 2]
    gt_verts_2d = gt_verts_proj[:, :2, :].permute(0, 2, 1)  # [B, 778, 2]

    # ============ Get hand part indices (CUDA Graph compatible) ============
    # Use cached mapping to avoid creating new tensors during graph capture
    vertex_to_part_id, part_names = _get_vertex_to_part_mapping_cached(device)

    # Map faces to parts based on first vertex
    face_part_ids = vertex_to_part_id[faces[:, 0]]  # [F]

    # ============ Visualize first sample (optional) ============
    if visualize_first and B > 0:
        # 只可视化第一个样本的GT
        try:
            visualize_and_save_hand_part_masks(
                verts=gt_verts[0],  # [778, 3]
                root=gt_root[0],    # [3]
                calib=calib[0],     # [4, 4]
                faces=faces,
                image_size=image_size,
                save_dir=save_dir,
                prefix='gt_sample',
                mano_path=mano_path,
                original_img=original_img[0] if original_img is not None else None
            )
            print(f"[Visualization] Saved GT hand part masks to {save_dir}/")
        except Exception as e:
            print(f"[Warning] Failed to visualize hand part masks: {e}")

    # ============ Get cached face indices for each part (CUDA Graph compatible) ============
    # Pre-compute face indices to avoid boolean indexing during graph capture
    part_face_indices_list, _ = _get_face_to_part_masks_cached(faces, device)

    # ============ Compute loss for each part ============
    total_loss = 0.0
    num_parts = len(part_names)

    for part_id, part_name in enumerate(part_names):
        # Get pre-computed face indices for this part (CUDA Graph compatible)
        part_face_indices = part_face_indices_list[part_id]

        # Rasterize triangles for this part (batch) with soft rasterization
        pred_masks = _rasterize_triangles_batch_optimized(
            pred_verts_2d, faces, part_face_indices, H, W,
            soft_rasterization=soft_rasterization, sigma=sigma
        )  # [B, H, W]
        gt_masks = _rasterize_triangles_batch_optimized(
            gt_verts_2d, faces, part_face_indices, H, W,
            soft_rasterization=soft_rasterization, sigma=sigma
        )  # [B, H, W]

        # Compute loss
        if loss_type == 'dice':
            intersection = (pred_masks * gt_masks).sum(dim=[1, 2])  # [B]
            union = pred_masks.sum(dim=[1, 2]) + gt_masks.sum(dim=[1, 2])  # [B]
            loss = 1.0 - (2.0 * intersection + 1e-7) / (union + 1e-7)  # [B]
            loss = loss.mean()  # Average over batch

        elif loss_type == 'iou':
            intersection = (pred_masks * gt_masks).sum(dim=[1, 2])
            union = pred_masks.sum(dim=[1, 2]) + gt_masks.sum(dim=[1, 2]) - intersection
            loss = 1.0 - (intersection + 1e-7) / (union + 1e-7)
            loss = loss.mean()

        elif loss_type == 'bce':
            loss = F.binary_cross_entropy(pred_masks, gt_masks, reduction='mean')

        elif loss_type == 'mse':
            loss = F.mse_loss(pred_masks, gt_masks, reduction='mean')

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        total_loss += loss

    # Average loss across all parts
    return total_loss / num_parts


def generate_hand_part_masks_with_faces(verts: torch.Tensor,
                                         root: torch.Tensor,
                                         calib: torch.Tensor,
                                         faces: torch.Tensor = None,
                                         image_size: tuple = (224, 224),
                                         mano_path: str = None) -> dict:
    """
    Generate hand part masks using mesh surface rendering

    Args:
        verts: Mesh vertices [778, 3], relative coordinates
        root: Root coordinate [3], for recovering absolute coordinates
        calib: Camera calibration matrix [4, 4]
        faces: Mesh faces [F, 3]. If None, will load from MANO template
        image_size: Image size (H, W)
        mano_path: Path to MANO template directory

    Returns:
        dict: Dictionary mapping part names to masks [H, W]
    """
    H, W = image_size
    device = verts.device

    # Load faces if not provided
    if faces is None:
        faces = _load_mano_faces(mano_path)
    faces = faces.to(device)

    # Recover absolute 3D coordinates: verts * 0.2 + root
    verts_3d = verts * 0.2 + root.unsqueeze(0)  # [778, 3]

    # Project to 2D using perspective projection
    verts_3d_t = verts_3d.unsqueeze(0).permute(0, 2, 1)  # [1, 3, 778]
    calib_batch = calib.unsqueeze(0)  # [1, 4, 4]

    verts_proj = perspective(verts_3d_t, calib_batch)  # [1, 3, 778]
    verts_2d = verts_proj[:, :2, :].permute(0, 2, 1)  # [1, 778, 2]

    # Get cached face indices for each part
    part_face_indices_list, part_names = _get_face_to_part_masks_cached(faces, device)

    # Generate masks for each part
    part_masks = {}
    for part_id, part_name in enumerate(part_names):
        part_face_indices = part_face_indices_list[part_id]

        # Rasterize triangles for this part
        mask = _rasterize_triangles_batch_optimized(
            verts_2d, faces, part_face_indices, H, W,
            soft_rasterization=False, sigma=1.0
        )  # [1, H, W]

        part_masks[part_name] = mask[0]  # [H, W]

    return part_masks


def visualize_and_save_hand_part_masks(verts: torch.Tensor,
                                        root: torch.Tensor,
                                        calib: torch.Tensor,
                                        faces: torch.Tensor = None,
                                        image_size: tuple = (224, 224),
                                        save_dir: str = './hand_part_masks',
                                        prefix: str = 'sample',
                                        mano_path: str = None,
                                        original_img: torch.Tensor = None):
    """
    Visualize and save hand part masks as images

    Args:
        verts: Mesh vertices [778, 3], relative coordinates
        root: Root coordinate [3], for recovering absolute coordinates
        calib: Camera calibration matrix [4, 4]
        faces: Mesh faces [F, 3]. If None, will load from MANO template
        image_size: Image size (H, W)
        save_dir: Directory to save visualization images
        prefix: Prefix for saved image filenames
        mano_path: Path to MANO template directory
        original_img: Optional original image [3, H, W] to overlay masks on

    Returns:
        dict: Dictionary of part masks
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate masks
    part_masks = generate_hand_part_masks_with_faces(
        verts, root, calib, faces, image_size, mano_path
    )

    # Define colors for each part (RGB)
    part_colors = {
        'palm': (255, 200, 200),      # Light red
        'thumb': (255, 100, 100),     # Red
        'index': (100, 255, 100),     # Green
        'middle': (100, 100, 255),    # Blue
        'ring': (255, 255, 100),      # Yellow
        'pinky': (255, 100, 255),     # Magenta
    }

    H, W = image_size

    # Save individual part masks
    for part_name, mask in part_masks.items():
        # Convert to numpy and scale to 0-255
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

        # Create colored mask
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color = part_colors.get(part_name, (255, 255, 255))
        for c in range(3):
            colored_mask[:, :, c] = (mask_np > 0) * color[c]

        # Save individual part mask
        img_part = Image.fromarray(colored_mask)
        img_part.save(os.path.join(save_dir, f'{prefix}_{part_name}.png'))

    # Create combined visualization with all parts
    combined_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for part_name, mask in part_masks.items():
        mask_np = mask.cpu().numpy()
        color = part_colors.get(part_name, (255, 255, 255))
        for c in range(3):
            # Use additive blending for overlapping regions
            combined_mask[:, :, c] = np.maximum(
                combined_mask[:, :, c],
                (mask_np * color[c]).astype(np.uint8)
            )

    # Save combined mask
    img_combined = Image.fromarray(combined_mask)
    img_combined.save(os.path.join(save_dir, f'{prefix}_combined.png'))

    # If original image is provided, create overlay
    if original_img is not None:
        # Convert original image to numpy [H, W, 3]
        if isinstance(original_img, torch.Tensor):
            if original_img.shape[0] == 3:  # [3, H, W]
                original_img = original_img.permute(1, 2, 0)  # [H, W, 3]
            original_np = (original_img.cpu().numpy() * 255).astype(np.uint8)
        else:
            original_np = original_img

        # Blend original image with masks (50% transparency)
        overlay = (original_np * 0.5 + combined_mask * 0.5).astype(np.uint8)
        img_overlay = Image.fromarray(overlay)
        img_overlay.save(os.path.join(save_dir, f'{prefix}_overlay.png'))

    print(f"Saved hand part masks to {save_dir}/")
    print(f"  - Individual parts: {', '.join(part_masks.keys())}")
    print(f"  - Combined visualization: {prefix}_combined.png")
    if original_img is not None:
        print(f"  - Overlay with original image: {prefix}_overlay.png")

    return part_masks




