"""
Hand Part Relative Position Loss with Neighbor Fingertip Constraints

This module computes loss for hand part vertices relative to their reference nodes
and adjacent fingertips to better constrain hand pose.

Each finger part computes relative distances to multiple fingertips:
- Pinky: relative to pinky tip + ring tip
- Ring: relative to pinky tip + ring tip + middle tip
- Middle: relative to ring tip + middle tip + index tip
- Index: relative to middle tip + index tip + thumb tip
- Thumb: relative to index tip + thumb tip
- Palm: skipped (already relative to wrist)
"""

import torch
import torch.nn.functional as F



def get_hand_part_vertex_indices():
    """
    Get vertex indices for each hand part (same as in MultiScaleHMR.py)

    NOTE: This function creates new tensors each time it's called.
    For CUDA Graph compatibility, use the registered buffers in the model instead.

    Returns:
        dict: Dictionary mapping part names to vertex indices tensors
    """
    # Palm vertices (complex non-contiguous ranges)
    palm_index = [i for i in range(46)] + [i for i in range(50, 56)] + [i for i in range(60, 86)] + \
                 [i for i in range(88, 133)] + [i for i in range(137, 155)] + [i for i in range(157, 164)] + \
                 [i for i in range(168, 174)] + [i for i in range(178, 189)] + [i for i in range(190, 194)] + \
                 [i for i in range(196, 212)] + [i for i in range(214, 221)] + [i for i in range(227, 237)] + \
                 [i for i in range(239, 245)] + [i for i in range(246, 261)] + [i for i in range(262, 272)] + \
                 [i for i in range(274, 280)] + [i for i in range(284, 294)] + [i for i in range(769, 778)]

    # Index finger vertices
    index_index = [i for i in range(46, 50)] + [i for i in range(56, 60)] + [i for i in range(86, 88)] + \
                  [i for i in range(133, 137)] + [i for i in range(155, 157)] + [i for i in range(164, 168)] + \
                  [i for i in range(174, 178)] + [i for i in range(189, 190)] + [i for i in range(194, 196)] + \
                  [i for i in range(212, 214)] + [i for i in range(221, 227)] + [i for i in range(237, 239)] + \
                  [i for i in range(245, 246)] + [i for i in range(261, 262)] + [i for i in range(272, 274)] + \
                  [i for i in range(280, 284)] + [i for i in range(294, 356)]

    return {
        'palm': torch.tensor(palm_index, dtype=torch.long),
        'index': torch.tensor(index_index, dtype=torch.long),
        'middle': torch.tensor(list(range(356, 468)), dtype=torch.long),
        'ring': torch.tensor(list(range(468, 580)), dtype=torch.long),
        'pinky': torch.tensor(list(range(580, 697)), dtype=torch.long),
        'thumb': torch.tensor(list(range(697, 769)), dtype=torch.long),
    }


def get_fingertip_vertex_indices():
    """
    Get fingertip vertex indices from MANO mesh (from hand_utils.py)

    Returns:
        dict: Dictionary mapping finger names to fingertip vertex indices
    """
    # From MANOHandJoints.mesh_mapping in hand_utils.py
    return {
        'index': 333,   # Index fingertip
        'middle': 444,  # Middle fingertip
        'pinky': 672,   # Pinky fingertip (Little finger)
        'ring': 555,    # Ring fingertip
        'thumb': 744,   # Thumb fingertip
    }


def get_neighbor_fingertips():
    """
    Get neighbor fingertip relationships for each finger part.
    Each finger should compute relative distances to its own fingertip and adjacent fingertips.

    Returns:
        dict: Dictionary mapping part names to list of neighbor fingertip names
    """
    return {
        'pinky': ['pinky', 'ring'],                    # Pinky: self + ring
        'ring': ['pinky', 'ring', 'middle'],           # Ring: pinky + self + middle
        'middle': ['ring', 'middle', 'index'],         # Middle: ring + self + index
        'index': ['middle', 'index', 'thumb'],         # Index: middle + self + thumb
        'thumb': ['index', 'thumb'],                   # Thumb: index + self
        'palm': []                                     # Palm: no fingertip references
    }

def compute_hand_part_relative_loss(verts_pred: torch.Tensor,
                                    verts_gt: torch.Tensor,
                                    loss_type: str = 'l1',
                                    part_indices_buffers: dict = None,
                                    fingertip_indices_dict: dict = None) -> torch.Tensor:
    """
    Compute loss for hand part vertices relative to their reference nodes.
    CUDA Graph compatible version - uses pre-registered buffers.

    For each finger: vertices are relative to their fingertip
    For palm: vertices are already relative to wrist (centered at zero)

    Args:
        verts_pred: Predicted vertices [B, 778, 3] or [B, 778, 6] (dual-view)
        verts_gt: Ground truth vertices [B, 778, 3] or [B, 778, 6] (dual-view)
        loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
        part_indices_buffers: Pre-registered buffers for part indices (CUDA Graph compatible)
        fingertip_indices_dict: Dictionary of fingertip indices

    Returns:
        loss: Scalar tensor representing the relative position loss
    """
    # If buffers not provided, create them (not CUDA Graph compatible)
    if part_indices_buffers is None:
        part_indices_buffers = get_hand_part_vertex_indices()
        # Move to device
        device = verts_pred.device
        for key in part_indices_buffers:
            part_indices_buffers[key] = part_indices_buffers[key].to(device)

    if fingertip_indices_dict is None:
        fingertip_indices_dict = get_fingertip_vertex_indices()

    # Handle dual-view mode
    is_dual_view = verts_pred.shape[-1] == 6

    if is_dual_view:
        # Split into two views
        verts_pred_v1 = verts_pred[..., :3]  # [B, 778, 3]
        verts_pred_v2 = verts_pred[..., 3:]  # [B, 778, 3]
        verts_gt_v1 = verts_gt[..., :3]      # [B, 778, 3]
        verts_gt_v2 = verts_gt[..., 3:]      # [B, 778, 3]

        # Compute loss for both views
        loss_v1 = _compute_single_view_relative_loss(
            verts_pred_v1, verts_gt_v1, part_indices_buffers, fingertip_indices_dict, loss_type
        )
        loss_v2 = _compute_single_view_relative_loss(
            verts_pred_v2, verts_gt_v2, part_indices_buffers, fingertip_indices_dict, loss_type
        )

        # Average loss across views
        loss = (loss_v1 + loss_v2) / 2.0
    else:
        # Single view mode
        loss = _compute_single_view_relative_loss(
            verts_pred, verts_gt, part_indices_buffers, fingertip_indices_dict, loss_type
        )

    return loss


def _compute_single_view_relative_loss(verts_pred: torch.Tensor,
                                       verts_gt: torch.Tensor,
                                       part_indices: dict,
                                       fingertip_indices: dict,
                                       loss_type: str) -> torch.Tensor:
    """
    Compute relative position loss for a single view with neighbor fingertip constraints.

    Each finger part computes relative distances to its own fingertip and adjacent fingertips:
    - Pinky: relative to pinky tip + ring tip
    - Ring: relative to pinky tip + ring tip + middle tip
    - Middle: relative to ring tip + middle tip + index tip
    - Index: relative to middle tip + index tip + thumb tip
    - Thumb: relative to index tip + thumb tip

    Args:
        verts_pred: Predicted vertices [B, 778, 3]
        verts_gt: Ground truth vertices [B, 778, 3]
        part_indices: Dictionary of part vertex indices
        fingertip_indices: Dictionary of fingertip vertex indices
        loss_type: Type of loss ('l1', 'l2', 'smooth_l1')

    Returns:
        loss: Scalar tensor
    """
    total_loss = 0.0
    neighbor_fingertips = get_neighbor_fingertips()

    # Process each hand part
    for part_name, vert_indices in part_indices.items():
        if part_name == 'palm':
            # Palm is already relative to wrist (centered at zero)
            # Skip palm for now
            continue

        # Get neighbor fingertip names for this part
        neighbor_names = neighbor_fingertips[part_name]

        if len(neighbor_names) == 0:
            # No neighbors, skip
            continue

        # Get finger part vertices
        finger_pred = verts_pred[:, vert_indices, :]  # [B, N_finger, 3]
        finger_gt = verts_gt[:, vert_indices, :]      # [B, N_finger, 3]

        # Compute loss relative to each neighbor fingertip
        part_loss = 0.0
        for neighbor_name in neighbor_names:
            # Get neighbor fingertip index
            neighbor_tip_idx = fingertip_indices[neighbor_name]

            # Get neighbor fingertip positions
            neighbor_tip_pred = verts_pred[:, neighbor_tip_idx:neighbor_tip_idx+1, :]  # [B, 1, 3]
            neighbor_tip_gt = verts_gt[:, neighbor_tip_idx:neighbor_tip_idx+1, :]      # [B, 1, 3]

            # Compute relative positions (subtract neighbor fingertip)
            finger_pred_rel = finger_pred - neighbor_tip_pred  # [B, N_finger, 3]
            finger_gt_rel = finger_gt - neighbor_tip_gt        # [B, N_finger, 3]

            # Compute loss on relative positions
            neighbor_loss = _compute_loss(finger_pred_rel, finger_gt_rel, loss_type)
            part_loss += neighbor_loss

        # Average loss across all neighbor fingertips for this part
        part_loss = part_loss / len(neighbor_names)
        total_loss += part_loss

    return total_loss


def _compute_loss(pred: torch.Tensor, gt: torch.Tensor, loss_type: str) -> torch.Tensor:
    """
    Compute loss between predicted and ground truth tensors

    Args:
        pred: Predicted tensor [B, N, 3]
        gt: Ground truth tensor [B, N, 3]
        loss_type: Type of loss ('l1', 'l2', 'smooth_l1')

    Returns:
        loss: Scalar tensor
    """
    if loss_type == 'l1':
        loss = F.l1_loss(pred, gt)
    elif loss_type == 'l2':
        loss = F.mse_loss(pred, gt)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred, gt)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Hand Part Relative Position Loss Example")
    print("=" * 60)

    # Test single view
    print("\n[Test 1] Single view mode")
    verts_pred = torch.randn(4, 778, 3)  # Batch size 4
    verts_gt = torch.randn(4, 778, 3)

    loss = compute_hand_part_relative_loss(verts_pred, verts_gt, loss_type='l1')
    print(f"Single view loss: {loss.item():.6f}")

    # Test dual view
    print("\n[Test 2] Dual view mode")
    verts_pred_dual = torch.randn(4, 778, 6)  # Batch size 4, dual view
    verts_gt_dual = torch.randn(4, 778, 6)

    loss_dual = compute_hand_part_relative_loss(verts_pred_dual, verts_gt_dual, loss_type='l1')
    print(f"Dual view loss: {loss_dual.item():.6f}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)