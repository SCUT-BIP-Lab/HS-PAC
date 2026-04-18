"""
Evaluator: Pluggable evaluation system for different metrics
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.utils.hand_eval_utils import (
    EvalUtil,
    rigid_align,
    mano_to_mpii,
    get_final_preds,
    accuracy,
    calc_dists,
    AUCMeter,
    AverageMeter,
    MPJPEMeter,
    calculate_iou,
    VertexMeter,
    rigid_align_vertices,
)
from src.utils.data_keys import DataKeys, PredKeys, MetricKeys
# from .fscore_utils import calculate_fscore


def to_numpy(tensor):
    """Convert tensor to numpy, detaching gradients if present"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


class BaseEvaluator(ABC):
    """
    Base class for metric evaluators
    """

    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset all internal metrics"""
        pass

    @abstractmethod
    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """
        Update metrics with a batch of predictions and ground truth

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
        """
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics

        Returns:
            Dictionary of metric names and values
        """
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Return list of metric names this evaluator computes"""
        pass

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        """Check if this evaluator can be applied to the given data"""
        return True

class Keypoint2DEvaluator(BaseEvaluator):
    """
    Evaluator for 2D keypoint detection (PCK, AUC, accuracy)
    """

    def reset(self):
        self.acc_meter = AverageMeter()
        self.acc_10_meter = AverageMeter()
        self.mpjpe_meter = MPJPEMeter()
        self.evaluator_2d = EvalUtil(num_kp=21)
        self.joint_img_errors = []

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update 2D keypoint metrics"""
        if PredKeys.SKELETON not in predictions or DataKeys.KPT_HEATMAP not in ground_truth:
            return

        kpt_pred_heatmaps = to_numpy(predictions[PredKeys.SKELETON])
        kpt_target_heatmaps = to_numpy(ground_truth[DataKeys.KPT_HEATMAP])

        # PCK@0.5
        _, avg_acc, cnt, _ = accuracy(kpt_pred_heatmaps, kpt_target_heatmaps)
        self.acc_meter.update(avg_acc, cnt)

        # Get coordinates
        pred_coords, _ = get_final_preds(kpt_pred_heatmaps)
        target_coords, _ = get_final_preds(kpt_target_heatmaps)

        # Scale to image size
        pred_coords = pred_coords * 224 / 56
        target_coords = target_coords * 224 / 56

        # Calculate distances
        distances = np.linalg.norm(pred_coords - target_coords, axis=2)

        # Feed to evaluator for AUC
        for sample_idx in range(pred_coords.shape[0]):
            self.evaluator_2d.feed(target_coords[sample_idx], pred_coords[sample_idx])

        self.joint_img_errors.append(distances)

        # 10 pixel accuracy
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if distances[i][j] < 10:
                    self.acc_10_meter.update(1, 1)
                else:
                    self.acc_10_meter.update(0, 1)

    def compute(self) -> Dict[str, float]:
        """Compute final 2D keypoint metrics"""
        metrics = {}

        # PCK@0.5
        metrics[MetricKeys.PCK_05] = self.acc_meter.avg

        # 10px accuracy
        metrics[MetricKeys.ACC_10PX] = self.acc_10_meter.avg

        # UVE (mean error)
        if self.joint_img_errors:
            joint_img_errors_np = np.concatenate(self.joint_img_errors, axis=0)
            metrics[MetricKeys.UVE] = joint_img_errors_np.mean()
        else:
            metrics[MetricKeys.UVE] = 0.0

        # AUC
        _, _, _, auc_2d, pck_curve_2d, pck_thresholds = self.evaluator_2d.get_measures(0, 30, 20)
        metrics[MetricKeys.AUC_2D] = auc_2d

        # PCK curve samples
        pck_sample_indices = np.linspace(0, len(pck_thresholds) - 1, 5, dtype=int)
        pck_curve_summary = [
            (float(pck_thresholds[idx]), float(pck_curve_2d[idx])) for idx in pck_sample_indices
        ]
        metrics[MetricKeys.PCK_CURVE] = pck_curve_summary

        return metrics

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.PCK_05, MetricKeys.ACC_10PX, MetricKeys.UVE, MetricKeys.AUC_2D, MetricKeys.PCK_CURVE]

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        return PredKeys.SKELETON in predictions and DataKeys.KPT_HEATMAP in ground_truth


class Keypoint3DEvaluator(BaseEvaluator):
    """
    Evaluator for 3D keypoint estimation (MPJPE, PA-MPJPE, AUC)
    """

    def reset(self):
        self.evaluator_3d_rel = EvalUtil(num_kp=21)
        self.evaluator_3d_pa = EvalUtil(num_kp=21)
        self.joint_cam_errors = []
        self.pa_joint_cam_errors = []
        self.j_reg = None

        # Load MANO joint regressor if available
        if 'mano_path' in self.conf and self.conf['mano_path']:
            try:
                import os
                self.j_reg = np.load(os.path.join(self.conf['mano_path'], 'j_reg.npy'))
            except:
                self.j_reg = None

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update 3D keypoint metrics"""
        if DataKeys.JOINT_CAM not in ground_truth:
            return

        batch_size = ground_truth[DataKeys.JOINT_CAM].size(0)

        for sample_idx in range(batch_size):
            # Get predicted 3D keypoints
            joint_cam_pred = None

            # Option 1: From vertex output
            if PredKeys.VERTS in predictions and self.j_reg is not None:
                verts_pred = to_numpy(predictions[PredKeys.VERTS][sample_idx]) * 0.2#去归一化，dataloader中归一化到0.2了
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred)) * 1000.0

            # Option 2: Direct 3D keypoint output
            if joint_cam_pred is None and PredKeys.JOINT_CAM in predictions:
                joint_cam_pred = to_numpy(predictions[PredKeys.JOINT_CAM][sample_idx]) * 0.2 * 1000.0

            if joint_cam_pred is not None:
                joint_cam_gt = to_numpy(ground_truth[DataKeys.JOINT_CAM][sample_idx]) * 0.2 * 1000.0

                # Rigid alignment
                joint_cam_align = rigid_align(joint_cam_pred, joint_cam_gt)

                # Feed to evaluators
                self.evaluator_3d_rel.feed(joint_cam_gt, joint_cam_pred)
                self.evaluator_3d_pa.feed(joint_cam_gt, joint_cam_align)

                # Calculate errors
                joint_cam_error = np.sqrt(np.sum((joint_cam_pred - joint_cam_gt) ** 2, axis=1))
                pa_joint_cam_error = np.sqrt(np.sum((joint_cam_gt - joint_cam_align) ** 2, axis=1))

                self.joint_cam_errors.extend(joint_cam_error)
                self.pa_joint_cam_errors.extend(pa_joint_cam_error)

    def compute(self) -> Dict[str, float]:
        """Compute final 3D keypoint metrics"""
        metrics = {}

        if not self.joint_cam_errors:
            return metrics

        # MPJPE and PA-MPJPE
        metrics[MetricKeys.MPJPE] = np.mean(self.joint_cam_errors)
        metrics[MetricKeys.PA_MPJPE] = np.mean(self.pa_joint_cam_errors)

        # AUC
        _, _, _, auc_3d_rel, _, _ = self.evaluator_3d_rel.get_measures(0, 50, 20)
        _, _, _, auc_3d_pa, _, _ = self.evaluator_3d_pa.get_measures(0, 50, 20)

        metrics[MetricKeys.AUC_3D_REL] = auc_3d_rel
        metrics[MetricKeys.AUC_3D_PA] = auc_3d_pa

        return metrics

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.MPJPE, MetricKeys.PA_MPJPE, MetricKeys.AUC_3D_REL, MetricKeys.AUC_3D_PA]

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        has_gt = DataKeys.JOINT_CAM in ground_truth
        has_pred = PredKeys.JOINT_CAM in predictions or (PredKeys.VERTS in predictions and self.j_reg is not None)
        return has_gt and has_pred


class VertexEvaluator(BaseEvaluator):
    """
    Evaluator for mesh vertex reconstruction (MPVPE, PA-MPVPE, F-scores)
    """

    def reset(self):
        self.vertex_meter = VertexMeter()

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update vertex metrics"""
        if PredKeys.VERTS not in predictions or DataKeys.VERTS not in ground_truth:
            return

        batch_size = predictions[PredKeys.VERTS].size(0)

        for sample_idx in range(batch_size):
            verts_pred = to_numpy(predictions[PredKeys.VERTS][sample_idx]) * 0.2
            verts_gt = to_numpy(ground_truth[DataKeys.VERTS][sample_idx]) * 0.2

            # Convert to mm
            verts_pred_mm = verts_pred * 1000.0
            verts_gt_mm = verts_gt * 1000.0

            # MPVPE
            verts_error = np.sqrt(np.sum((verts_pred_mm - verts_gt_mm) ** 2, axis=1))

            # PA-MPVPE
            verts_aligned_mm = rigid_align_vertices(verts_pred_mm, verts_gt_mm)
            pa_verts_error = np.sqrt(np.sum((verts_aligned_mm - verts_gt_mm) ** 2, axis=1))

            # F-scores using official calculation (th in meters)
            f_score_5mm, _, _ = calculate_fscore(verts_gt, verts_pred, th=0.005)
            f_score_15mm, _, _ = calculate_fscore(verts_gt, verts_pred, th=0.015)

            # PA F-scores
            verts_aligned = verts_aligned_mm / 1000.0
            pa_f_score_5mm, _, _ = calculate_fscore(verts_gt, verts_aligned, th=0.005)
            pa_f_score_15mm, _, _ = calculate_fscore(verts_gt, verts_aligned, th=0.015)

            self.vertex_meter.update(verts_error, pa_verts_error, f_score_5mm, f_score_15mm, pa_f_score_5mm, pa_f_score_15mm)

    def compute(self) -> Dict[str, float]:
        """Compute final vertex metrics"""
        vertex_metrics = self.vertex_meter.get_metrics()
        if not vertex_metrics:
            return {}

        return {
            MetricKeys.MPVPE: float(vertex_metrics['mpvpe']),
            MetricKeys.PA_MPVPE: float(vertex_metrics['pa_mpvpe']),
            MetricKeys.F_SCORE_5MM: vertex_metrics['f_score_5mm'],
            MetricKeys.F_SCORE_15MM: vertex_metrics['f_score_15mm'],
            MetricKeys.PA_F_SCORE_5MM: vertex_metrics.get('pa_f_score_5mm', 0.0),
            MetricKeys.PA_F_SCORE_15MM: vertex_metrics.get('pa_f_score_15mm', 0.0)
        }

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.MPVPE, MetricKeys.PA_MPVPE, MetricKeys.F_SCORE_5MM, MetricKeys.F_SCORE_15MM, MetricKeys.PA_F_SCORE_5MM, MetricKeys.PA_F_SCORE_15MM]

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        return PredKeys.VERTS in predictions and DataKeys.VERTS in ground_truth


class SegmentationEvaluator(BaseEvaluator):
    """
    Evaluator for segmentation masks (IoU)
    """

    def reset(self):
        self.iou_meter = AverageMeter()

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update segmentation metrics"""
        if PredKeys.SEGMENT not in predictions or DataKeys.MASK not in ground_truth:
            return

        mask_pred = predictions[PredKeys.SEGMENT]
        mask_target = ground_truth[DataKeys.MASK].to(mask_pred.device)

        batch_iou = calculate_iou(mask_pred, mask_target)
        self.iou_meter.update(batch_iou, n=mask_pred.size(0))

    def compute(self) -> Dict[str, float]:
        """Compute final segmentation metrics"""
        return {MetricKeys.IOU: self.iou_meter.avg}

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.IOU]

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        return PredKeys.SEGMENT in predictions and DataKeys.MASK in ground_truth


class JointImgEvaluator(BaseEvaluator):
    """
    Evaluator for direct 2D keypoint coordinates (joint_img)
    直接输出关键点坐标的评估器，评估指标和流程与热图评估器相同
    """

    def reset(self):
        self.pck_5pix = AverageMeter()
        self.pck_10pix = AverageMeter()
        self.evaluator_2d = EvalUtil(num_kp=21)
        self.joint_img_errors = []

        # 从配置中获取缩放参数，默认为1（不缩放）
        self.scale_factor = self.conf.get('joint_img_scale_factor', 1.0)

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update 2D keypoint metrics for direct coordinate predictions"""
        if PredKeys.JOINT_IMG not in predictions or DataKeys.JOINT_IMG not in ground_truth:
            return

        # 获取预测和ground truth坐标
        # 预测: [B, 21, 2] 或 [B, 21, 3]（如果包含置信度）
        pred_coords = to_numpy(predictions[PredKeys.JOINT_IMG])
        target_coords = to_numpy(ground_truth[DataKeys.JOINT_IMG])

        # 处理对比学习情况（坐标拼接）: [B, 21, 4] -> [2B, 21, 2]
        if pred_coords.shape[-1] == 4:
            pred_coords = np.concatenate((pred_coords[:, :, :2], pred_coords[:, :, 2:]), axis=0)
            target_coords = np.concatenate((target_coords[:, :, :2], target_coords[:, :, 2:]), axis=0)

        # 如果是3维（包含置信度），只取前两维
        if pred_coords.shape[-1] == 3:
            pred_coords = pred_coords[:, :, :2]
        if target_coords.shape[-1] == 3:
            target_coords = target_coords[:, :, :2]

        # 缩放到图像尺度
        imgsz=224  # use gt img size
        pred_coords = pred_coords * imgsz
        target_coords = target_coords * imgsz

        # 计算距离
        distances = np.linalg.norm(pred_coords - target_coords, axis=2)  # [B, 21]

        # Feed to evaluator for AUC
        for sample_idx in range(pred_coords.shape[0]):
            self.evaluator_2d.feed(target_coords[sample_idx], pred_coords[sample_idx])

        self.joint_img_errors.append(distances)

        # 10 and pixel accuracy
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if distances[i][j] < 10:
                    self.pck_10pix.update(1, 1)
                else:
                    self.pck_10pix.update(0, 1)

                if distances[i][j] < 5:
                    self.pck_5pix.update(1, 1)
                else:
                    self.pck_5pix.update(0, 1)


    def compute(self) -> Dict[str, float]:
        """Compute final 2D keypoint metrics"""
        metrics = {}

        # PCK@0.5
        metrics[MetricKeys.PCK_5PX_IMG] = self.pck_5pix.avg

        # 10px accuracy
        metrics[MetricKeys.PCK_10PX_IMG] = self.pck_10pix.avg

        # UVE (mean error)
        if self.joint_img_errors:
            joint_img_errors_np = np.concatenate(self.joint_img_errors, axis=0)
            metrics[MetricKeys.UVE_IMG] = joint_img_errors_np.mean()
        else:
            metrics[MetricKeys.UVE_IMG] = 0.0

        # AUC
        _, _, _, auc_2d, pck_curve_2d, pck_thresholds = self.evaluator_2d.get_measures(0, 30, 20)
        metrics[MetricKeys.AUC_2D_IMG] = auc_2d

        # PCK curve samples
        pck_sample_indices = np.linspace(0, len(pck_thresholds) - 1, 10, dtype=int)
        pck_curve_summary = [
            (float(pck_thresholds[idx]), float(pck_curve_2d[idx])) for idx in pck_sample_indices
        ]
        metrics[MetricKeys.PCK_CURVE_IMG] = pck_curve_summary

        return metrics

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.PCK_5PX_IMG, MetricKeys.PCK_10PX_IMG, MetricKeys.UVE_IMG, MetricKeys.AUC_2D_IMG, MetricKeys.PCK_CURVE_IMG]

    def is_applicable(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> bool:
        return PredKeys.JOINT_IMG in predictions and DataKeys.JOINT_IMG in ground_truth


class LossEvaluator(BaseEvaluator):
    """
    Evaluator for tracking loss values
    """

    def reset(self):
        self.loss_meter = AverageMeter()

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """Update loss metrics"""
        if PredKeys.LOSS not in predictions:
            return

        loss = predictions[PredKeys.LOSS].mean()
        batch_size = ground_truth.get(DataKeys.KPT_HEATMAP, ground_truth.get(DataKeys.JOINT_IMG)).size(0)
        self.loss_meter.update(loss.item(), n=batch_size)

    def compute(self) -> Dict[str, float]:
        """Compute final loss metrics"""
        return {MetricKeys.LOSS: self.loss_meter.avg}

    def get_metric_names(self) -> List[str]:
        return [MetricKeys.LOSS]


class EvaluatorRegistry:
    """
    Registry for managing different evaluators
    """

    _evaluators = {
        'keypoint_2d': Keypoint2DEvaluator,
        'keypoint_3d': Keypoint3DEvaluator,
        'joint_img': JointImgEvaluator,
        'vertex': VertexEvaluator,
        # 'segmentation': SegmentationEvaluator,
        'loss': LossEvaluator,
    }

    @classmethod
    def register(cls, name: str, evaluator_class: type):
        """Register a new evaluator"""
        cls._evaluators[name] = evaluator_class

    @classmethod
    def get_evaluator(cls, name: str, conf: Dict[str, Any]) -> BaseEvaluator:
        """Get an evaluator instance by name"""
        if name not in cls._evaluators:
            raise ValueError(f"Unknown evaluator: {name}. Available: {list(cls._evaluators.keys())}")
        return cls._evaluators[name](conf)

    @classmethod
    def list_evaluators(cls) -> List[str]:
        """List all registered evaluators"""
        return list(cls._evaluators.keys())

    @classmethod
    def get_all_evaluators(cls, conf: Dict[str, Any]) -> List[BaseEvaluator]:
        """Get instances of all registered evaluators"""
        return [evaluator_class(conf) for evaluator_class in cls._evaluators.values()]


import open3d as o3d
import numpy as np


def verts2pcd(verts, color=None):
    """将顶点数组转换为点云"""
    # Ensure data is in correct format for Open3D
    if not isinstance(verts, np.ndarray):
        verts = np.array(verts)

    # Handle contrastive learning case where verts might be (N, 6)
    if verts.ndim == 2 and verts.shape[1] > 3:
        verts = verts[:, :3]

    # Ensure float64 and contiguous
    verts = np.ascontiguousarray(verts, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        elif color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        elif color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    """
    计算 F-score

    Args:
        gt: ground truth vertices (N, 3)
        pr: predicted vertices (M, 3)
        th: threshold for distance (默认 0.01, 即 1cm)

    Returns:
        fscore, precision, recall
    """
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)

    # 计算点云间距离
    d1 = gt.compute_point_cloud_distance(pr)  # 每个gt点到最近pred点的距离
    d2 = pr.compute_point_cloud_distance(gt)  # 每个pred点到最近gt点的距离

    if len(d1) and len(d2):
        # Recall: 有多少预测点接近真实点
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        # Precision: 有多少真实点被预测点匹配
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall