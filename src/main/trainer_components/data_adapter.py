"""
Data Adapter: Handles different types of data inputs and prepares them for model consumption

使用统一的数据键管理（src/utils/data_keys.py）来确保整个框架的一致性
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.data_keys import DataKeys, PredKeys

####################################################################################
# decouple data preparation from model training loop
# allows easy extension for new data types and formats by a new DataAdapter subclass
#####################################################################################

class DataAdapter(ABC):
    """
    Base class for data adapters that transform raw data batches into model inputs
    """

    def __init__(self, device: torch.device, conf: Dict[str, Any]):
        self.device = device
        self.conf = conf

    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """Return list of required keys in data batch"""
        pass

    @abstractmethod
    def get_optional_keys(self) -> List[str]:
        """Return list of optional keys in data batch"""
        pass

    @abstractmethod
    def prepare_batch(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Transform raw data batch into model-ready tensors

        Args:
            data: Raw data batch from dataloader

        Returns:
            Dictionary of prepared tensors ready for model input
        """
        pass

    @abstractmethod
    def get_ground_truth(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract ground truth labels from data batch

        Args:
            data: Raw data batch from dataloader

        Returns:
            Dictionary of ground truth tensors
        """
        pass

    def validate_batch(self, data: Dict[str, Any]) -> bool:
        """Check if batch contains all required keys"""
        required = self.get_required_keys()
        return all(key in data for key in required)


class HandPoseDataAdapter(DataAdapter):
    """
    Adapter for hand pose estimation data (2D keypoints + optional mask)
    使用DataKeys统一管理数据键名
    """

    def get_required_keys(self) -> List[str]:
        return [DataKeys.RGB, DataKeys.KPT_HEATMAP]

    def get_optional_keys(self) -> List[str]:
        return [DataKeys.MASK]

    def prepare_batch(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare hand pose data for model input"""
        prepared = {}

        # RGB image - 支持多种可能的键名
        rgb_key = DataKeys.RGB if DataKeys.RGB in data else 'RGB'
        if rgb_key in data:
            rgb = data[rgb_key].to(self.device)
            prepared[DataKeys.RGB] = rgb.view(-1, 3, 224, 224).contiguous()

        # Mask (optional)
        if DataKeys.MASK in data:
            mask = data[DataKeys.MASK].to(self.device)
            prepared[DataKeys.MASK] = mask.view(-1, 3, 224, 224).contiguous()[:, 0, :, :]
        else:
            prepared[DataKeys.MASK] = None

        # Keypoint heatmap
        if DataKeys.KPT_HEATMAP in data:
            kpt_heatmap = data[DataKeys.KPT_HEATMAP].to(self.device)
            prepared[DataKeys.KPT_HEATMAP] = kpt_heatmap.view(-1, 21, 56, 56).contiguous()

        return prepared

    def get_ground_truth(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract ground truth for hand pose"""
        gt = {}

        if DataKeys.KPT_HEATMAP in data:
            gt[DataKeys.KPT_HEATMAP] = data[DataKeys.KPT_HEATMAP].to(self.device)

        if DataKeys.MASK in data:
            gt[DataKeys.MASK] = data[DataKeys.MASK].to(self.device)

        return gt


class Hand3DDataAdapter(DataAdapter):
    """
    Adapter for 3D hand reconstruction data (2D + 3D keypoints + mesh vertices)
    使用DataKeys统一管理数据键名
    """

    def get_required_keys(self) -> List[str]:
        return [DataKeys.RGB, DataKeys.KPT_HEATMAP]

    def get_optional_keys(self) -> List[str]:
        return [
            DataKeys.MASK,
            DataKeys.JOINT_CAM,
            DataKeys.VERTS,
            DataKeys.MANO_POSE,
            DataKeys.MANO_SHAPE,
            DataKeys.JOINT_IMG,
            DataKeys.ROOT,
            DataKeys.CALIB,
        ]

    def prepare_batch(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare 3D hand data for model input and loss calculation"""
        prepared = {}

        # RGB image - 支持多种可能的键名
        rgb_key = DataKeys.RGB if DataKeys.RGB in data else 'RGB'
        if rgb_key in data:
            rgb = data[rgb_key].to(self.device)
            prepared[DataKeys.RGB] = rgb
            prepared[DataKeys.IMG] = rgb  # 为损失函数提供img键

        # Mask (optional)
        if DataKeys.MASK in data:
            mask = data[DataKeys.MASK].to(self.device)
            prepared[DataKeys.MASK] = mask
        else:
            prepared[DataKeys.MASK] = None

        # Keypoint heatmap
        if DataKeys.KPT_HEATMAP in data:
            kpt_heatmap = data[DataKeys.KPT_HEATMAP].to(self.device)
            prepared[DataKeys.KPT_HEATMAP] = kpt_heatmap.view(-1, 21, 56, 56).contiguous()

        # ========== 损失计算所需的额外数据 ==========

        # 2D关键点坐标 (joint_img_gt)
        if DataKeys.JOINT_IMG in data:
            joint_img = data[DataKeys.JOINT_IMG].to(self.device)
            prepared[DataKeys.JOINT_IMG] = joint_img

        # Mesh顶点 (verts_gt)
        if DataKeys.VERTS in data:
            verts = data[DataKeys.VERTS].to(self.device)
            prepared[DataKeys.VERTS] = verts

        # 数据增强参数 (aug_param)
        if DataKeys.AUG_PARAM in data:
            aug_param = data[DataKeys.AUG_PARAM]
            # aug_param可能不需要转到GPU，取决于具体实现
            if isinstance(aug_param, torch.Tensor):
                prepared[DataKeys.AUG_PARAM] = aug_param.to(self.device)
            else:
                prepared[DataKeys.AUG_PARAM] = aug_param

        # Bounding box到图像的变换矩阵 (bb2img_trans)
        if DataKeys.BB2IMG_TRANS in data:
            bb2img_trans = data[DataKeys.BB2IMG_TRANS].to(self.device)
            prepared[DataKeys.BB2IMG_TRANS] = bb2img_trans

        # 3D关键点 (joint_cam)
        if DataKeys.JOINT_CAM in data:
            joint_cam = data[DataKeys.JOINT_CAM].to(self.device)
            prepared[DataKeys.JOINT_CAM] = joint_cam

        # Root joint position
        if DataKeys.ROOT in data:
            root = data[DataKeys.ROOT].to(self.device)
            prepared[DataKeys.ROOT] = root

        # MANO参数
        if DataKeys.MANO_POSE in data:
            mano_pose = data[DataKeys.MANO_POSE].to(self.device)
            prepared[DataKeys.MANO_POSE] = mano_pose

        if DataKeys.MANO_SHAPE in data:
            mano_shape = data[DataKeys.MANO_SHAPE].to(self.device)
            prepared[DataKeys.MANO_SHAPE] = mano_shape

        # 相机标定
        if DataKeys.CALIB in data:
            calib = data[DataKeys.CALIB].to(self.device)
            prepared[DataKeys.CALIB] = calib

        return prepared

    def get_ground_truth(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract ground truth for 3D hand reconstruction"""
        gt = {}

        # 2D keypoints heatmap
        if DataKeys.KPT_HEATMAP in data:
            gt[DataKeys.KPT_HEATMAP] = data[DataKeys.KPT_HEATMAP].to(self.device)

        # 2D keypoints coordinates
        if DataKeys.JOINT_IMG in data:
            gt[DataKeys.JOINT_IMG] = data[DataKeys.JOINT_IMG].to(self.device)

        # Mask
        if DataKeys.MASK in data:
            gt[DataKeys.MASK] = data[DataKeys.MASK].to(self.device)

        # 3D keypoints (camera coordinates)
        if DataKeys.JOINT_CAM in data:
            gt[DataKeys.JOINT_CAM] = data[DataKeys.JOINT_CAM].to(self.device)

        # Root joint position
        if DataKeys.ROOT in data:
            gt[DataKeys.ROOT] = data[DataKeys.ROOT].to(self.device)

        # Mesh vertices
        if DataKeys.VERTS in data:
            gt[DataKeys.VERTS] = data[DataKeys.VERTS].to(self.device)

        # MANO pose parameters
        if DataKeys.MANO_POSE in data:
            gt[DataKeys.MANO_POSE] = data[DataKeys.MANO_POSE].to(self.device)

        # MANO shape parameters
        if DataKeys.MANO_SHAPE in data:
            gt[DataKeys.MANO_SHAPE] = data[DataKeys.MANO_SHAPE].to(self.device)

        # Camera calibration
        if DataKeys.CALIB in data:
            gt[DataKeys.CALIB] = data[DataKeys.CALIB].to(self.device)

        return gt


class DataAdapterRegistry:
    """
    Registry for managing different data adapters
    """

    _adapters = {
        'hand_pose_2d': HandPoseDataAdapter,
        'hand_3d': Hand3DDataAdapter,
    }

    @classmethod
    def register(cls, name: str, adapter_class: type):
        """Register a new adapter"""
        cls._adapters[name] = adapter_class
    @classmethod
    def get_adapter(cls, name: str, device: torch.device, conf: Dict[str, Any]) -> DataAdapter:
        """Get an adapter instance by name"""
        if name not in cls._adapters:
            raise ValueError(f"Unknown adapter: {name}. Available: {list(cls._adapters.keys())}")
        return cls._adapters[name](device, conf)

    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapters"""
        return list(cls._adapters.keys())
