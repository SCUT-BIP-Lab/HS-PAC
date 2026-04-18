"""
Evaluation Pipeline: Configuration-driven evaluation system
"""

import torch
import numpy as np
import cv2
import numbers
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from .evaluator import BaseEvaluator, EvaluatorRegistry
from .data_adapter import DataAdapter

from src.utils.data_keys import MetricKeys


class MetricHistoryTracker:
    """
    Tracks historical best values for metrics (in-memory only)
    """

    def __init__(self, lower_is_better_keywords: Optional[List[str]] = None):
        """
        Args:
            lower_is_better_keywords: List of keywords for metrics where lower is better
        """
        self.best_metrics = {}  # {metric_name: {'value': float, 'epoch': int}}
        self.lower_is_better_keywords = lower_is_better_keywords or ['loss', 'mpjpe', 'mpvpe', 'uve']

    def update(self, metrics: Dict[str, float], epoch: int = 0) -> Dict[str, bool]:
        """
        Update history with new metrics

        Args:
            metrics: Current metrics
            epoch: Current epoch number

        Returns:
            Dictionary indicating which metrics are new bests
        """
        is_best = {}

        for metric_name, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            # Determine if higher or lower is better
            is_higher_better = self._is_higher_better(metric_name)

            # Check if this is a new best
            if metric_name not in self.best_metrics:
                # First time seeing this metric
                self.best_metrics[metric_name] = {
                    'value': value,
                    'epoch': epoch
                }
                is_best[metric_name] = True
            else:
                old_best = self.best_metrics[metric_name]['value']
                if is_higher_better:
                    if value > old_best:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'epoch': epoch
                        }
                        is_best[metric_name] = True
                    else:
                        is_best[metric_name] = False
                else:
                    if value < old_best:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'epoch': epoch
                        }
                        is_best[metric_name] = True
                    else:
                        is_best[metric_name] = False

        return is_best

    def _is_higher_better(self, metric_name: str) -> bool:
        """
        Determine if higher values are better for a metric

        Args:
            metric_name: Name of the metric

        Returns:
            True if higher is better, False if lower is better
        """
        # Check if any of the lower-is-better keywords are in the metric name
        for keyword in self.lower_is_better_keywords:
            if keyword in metric_name.lower():
                return False

        # Default: higher is better (for accuracy, IoU, PCK, AUC, F-score, etc.)
        return True

    def get_best(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get best value for a metric

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with 'value' and 'epoch', or None if not found
        """
        return self.best_metrics.get(metric_name)

    def get_all_best(self) -> Dict[str, Dict[str, Any]]:
        """Get all best metrics"""
        return self.best_metrics.copy()


class EvaluationPipeline:
    """
    Configuration-driven evaluation pipeline that automatically applies
    appropriate evaluators based on data and model outputs
    """

    def __init__(self,
                 conf: Dict[str, Any],
                 data_adapter: DataAdapter,
                 evaluators: Optional[List[str]] = None):
        """
        Args:
            conf: Configuration dictionary
            data_adapter: Data adapter for preparing batches
            evaluators: List of evaluator names to use. If None, uses all available evaluators.
        """
        self.conf = conf
        self.data_adapter = data_adapter

        # Initialize evaluators
        if evaluators is None:
            # Use all available evaluators
            self.evaluators = EvaluatorRegistry.get_all_evaluators(conf)
        else:
            # Use specified evaluators
            self.evaluators = [EvaluatorRegistry.get_evaluator(name, conf) for name in evaluators]

        self.active_evaluators = []

        # 历史最佳记录 {metric_name: best_value}
        # 对于越小越好的指标（如loss、mpjpe），记录最小值
        # 对于越大越好的指标（如auc、iou），记录最大值
        self.best_metrics = {}

    def reset(self):
        """Reset all evaluators"""
        for evaluator in self.evaluators:
            evaluator.reset()
        self.active_evaluators = []

    def update(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        """
        Update metrics with a batch of predictions and ground truth

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
        """
        # On first update, determine which evaluators are applicable
        if not self.active_evaluators:
            self.active_evaluators = [
                evaluator for evaluator in self.evaluators
                if evaluator.is_applicable(predictions, ground_truth)
            ]

            if self.active_evaluators:
                print(f"✅ 激活的评估器: {[type(e).__name__ for e in self.active_evaluators]}")

        # Update all active evaluators
        for evaluator in self.active_evaluators:
            evaluator.update(predictions, ground_truth)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from all active evaluators

        Returns:
            Dictionary of all computed metrics
        """
        all_metrics = {}

        for evaluator in self.active_evaluators:
            metrics = evaluator.compute()
            all_metrics.update(metrics)

        # 更新历史最佳记录
        self._update_best_metrics(all_metrics)

        return all_metrics

    def get_active_metric_names(self) -> List[str]:
        """Get names of all metrics from active evaluators"""
        names = []
        for evaluator in self.active_evaluators:
            names.extend(evaluator.get_metric_names())
        return names

    def _is_better(self, metric_name: str, current_value: float, best_value: float) -> bool:
        """
        判断当前值是否优于历史最佳值

        Args:
            metric_name: 指标名称
            current_value: 当前值
            best_value: 历史最佳值

        Returns:
            是否更好
        """
        # 越小越好的指标（使用MetricKeys中定义的精确键名）
        lower_is_better = {
            MetricKeys.LOSS,
            MetricKeys.MPJPE,
            MetricKeys.PA_MPJPE,
            MetricKeys.MPVPE,
            MetricKeys.PA_MPVPE,
            MetricKeys.UVE,
            MetricKeys.UVE_IMG,
        }

        # 越大越好的指标（使用MetricKeys中定义的精确键名）
        higher_is_better = {
            MetricKeys.PCK_05,
            MetricKeys.PCK_5PX_IMG,
            MetricKeys.ACC_10PX,
            MetricKeys.ACC_10PX_IMG,
            MetricKeys.AUC_2D,
            MetricKeys.AUC_2D_IMG,
            MetricKeys.AUC_3D_REL,
            MetricKeys.AUC_3D_PA,
            MetricKeys.F_SCORE_5MM,
            MetricKeys.F_SCORE_15MM,
            MetricKeys.PA_F_SCORE_5MM,
            MetricKeys.PA_F_SCORE_15MM,
            MetricKeys.IOU,
            MetricKeys.MIOU,
        }

        # 精确匹配指标名称
        if metric_name in lower_is_better:
            return current_value <= best_value
        elif metric_name in higher_is_better:
            return current_value >= best_value

        # 默认：越小越好
        return current_value <= best_value

    def _update_best_metrics(self, metrics: Dict[str, float]):
        """
        更新历史最佳指标记录

        Args:
            metrics: 当前评估指标字典
        """
        for key, value in metrics.items():
            # 跳过非数值类型（如pck_curve）
            if not isinstance(value, numbers.Number):
                continue

            if key not in self.best_metrics:
                # 首次记录
                self.best_metrics[key] = value
            else:
                # 比较并更新
                if self._is_better(key, value, self.best_metrics[key]):
                    self.best_metrics[key] = value

    def is_best_model(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        判断当前模型是否为历史最佳模型

        优先级规则（从高到低）：
        1. MPJPE (3D关键点误差) - 最重要的3D指标
        2. PA_MPJPE (Procrustes对齐后的MPJPE)
        3. MPVPE (3D顶点误差) - Mesh质量
        4. PA_MPVPE (Procrustes对齐后的MPVPE)
        5. AUC_3D_REL (3D关键点的AUC - 相对坐标)
        6. AUC_3D_PA (3D关键点的AUC - 对齐后)
        7. F_SCORE_5MM (F-score @ 5mm)
        8. AUC_2D (2D关键点的AUC)
        9. AUC_2D_IMG (2D关键点的AUC - 直接坐标版本)
        10. PCK_05 (PCK@0.5)
        11. PCK_05_IMG (PCK@0.5 - 直接坐标版本)
        12. IoU (分割质量)
        13. Loss (总体损失)

        Args:
            metrics: 当前评估指标字典

        Returns:
            (is_best, reason): 是否为最佳模型，以及原因说明
        """
        # 按优先级定义关键指标（使用MetricKeys中定义的精确键名）
        priority_metrics = [
            (MetricKeys.PA_MPJPE, 'PA-MPJPE'),  # Procrustes对齐MPJPE
            (MetricKeys.PA_MPVPE, 'PA-MPVPE'),  # Procrustes对齐MPVPE
            (MetricKeys.MPJPE, 'MPJPE'),  # 3D关键点误差
            (MetricKeys.MPVPE, 'MPVPE'),  # 3D顶点误差
            (MetricKeys.AUC_3D_REL, 'AUC_3D_REL'),  # 3D AUC (相对坐标)
            (MetricKeys.AUC_3D_PA, 'AUC_3D_PA'),  # 3D AUC (对齐后)
            (MetricKeys.F_SCORE_5MM, 'F-Score@5mm'),  # F-score @ 5mm
            (MetricKeys.F_SCORE_15MM, 'F-Score@15mm'),  # F-score @ 15mm
            (MetricKeys.PA_F_SCORE_5MM, 'PA-F-Score@5mm'),  # PA F-score @ 5mm
            (MetricKeys.PA_F_SCORE_15MM, 'PA-F-Score@15mm'),  # PA F-score @ 15mm
            (MetricKeys.AUC_2D, 'AUC_2D'),  # 2D AUC
            (MetricKeys.AUC_2D_IMG, 'AUC_2D_IMG'),  # 2D AUC (直接坐标)
            (MetricKeys.PCK_05, 'PCK@0.5'),  # PCK@0.5
            (MetricKeys.PCK_5PX_IMG, 'PCK@0.5_IMG'),  # PCK@0.5 (直接坐标)
            (MetricKeys.ACC_10PX, 'ACC@10px'),  # 10像素准确率
            (MetricKeys.ACC_10PX_IMG, 'ACC@10px_IMG'),  # 10像素准确率 (直接坐标)
            (MetricKeys.IOU, 'IoU'),  # 分割IoU
            (MetricKeys.MIOU, 'mIoU'),  # 平均IoU
            (MetricKeys.LOSS, 'Loss'),  # 总体损失
        ]

        # 按优先级检查每个指标（精确匹配）
        for metric_key, metric_display_name in priority_metrics:
            # 精确匹配指标名称
            if metric_key not in metrics:
                continue

            current_value = metrics[metric_key]

            # 跳过非数值类型
            if not isinstance(current_value, numbers.Number):
                continue

            # 如果这是第一次评估，认为是最佳
            if metric_key not in self.best_metrics:
                return True, f"{metric_display_name}: {current_value:.4f} (首次评估)"

            # 检查是否优于历史最佳
            best_value = self.best_metrics[metric_key]
            if self._is_better(metric_key, current_value, best_value):
                return True, f"{metric_display_name}: {current_value:.4f} (历史最佳: {best_value:.4f})"
            else:
                return False, f"{metric_display_name}未改进: {current_value:.4f} (历史最佳: {best_value:.4f})"

        # 如果所有关键指标都不是最佳，返回False
        return False, "无关键指标改进"

    def format_results(self, metrics: Dict[str, float]) -> str:
        """
        Format evaluation results as a readable string

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Formatted string
        """
        lines = [
            "\n" + "=" * 60,
            "📊 评估结果",
            "=" * 60
        ]

        # Group metrics by category (基于MetricKeys常量分类)
        loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
        seg_metrics = {k: v for k, v in metrics.items() if 'iou' in k.lower()}
        # 2D关键点指标：pck, acc, uve, auc_2d
        kpt_2d_metrics = {k: v for k, v in metrics.items() if
                          any(x in k.lower() for x in ['pck', 'acc', 'uve', 'auc_2d'])}
        # 3D关键点指标：mpjpe, auc_3d, pa_mpjpe
        kpt_3d_metrics = {k: v for k, v in metrics.items() if any(x in k.lower() for x in ['mpjpe', 'auc_3d'])}
        # 顶点指标：mpvpe, f_score, pa_mpvpe
        vertex_metrics = {k: v for k, v in metrics.items() if any(x in k.lower() for x in ['mpvpe', 'f_score'])}

        # Loss
        if loss_metrics:
            lines.append("\n【损失】")
            for key, value in loss_metrics.items():
                best_str = f"\t\t(史上最佳: {self.best_metrics[key]:.6f})" if key in self.best_metrics else ""
                lines.append(f"  {key}: {value:.6f}{best_str}")

        # Segmentation
        if seg_metrics:
            lines.append("\n【分割任务】")
            for key, value in seg_metrics.items():
                best_str = f"\t\t(史上最佳: {self.best_metrics[key]:.6f})" if key in self.best_metrics else ""
                lines.append(f"  {key}: {value:.6f}{best_str}")

        # 2D Keypoints
        if kpt_2d_metrics:
            lines.append("\n【2D关键点】")
            for key, value in kpt_2d_metrics.items():
                if key == 'pck_curve':
                    curve_str = ', '.join([f"{t:.1f}:{v:.3f}" for t, v in value])
                    lines.append(f"  PCK曲线: [{curve_str}]")
                else:
                    if isinstance(value, list):
                        lines.append(f"  {key}: {value}")
                    else:
                        best_str = f"\t\t(史上最佳: {self.best_metrics[key]:.6f})" if key in self.best_metrics else ""
                        lines.append(f"  {key}: {value:.6f}{best_str}")

        # 3D Keypoints
        if kpt_3d_metrics:
            lines.append("\n【3D关键点】")
            for key, value in kpt_3d_metrics.items():
                unit = " mm" if 'mpjpe' in key else ""
                if isinstance(value, list):
                    lines.append(f"  {key}: {value}")
                else:
                    best_str = f"\t\t(史上最佳: {self.best_metrics[key]:.6f}{unit})" if key in self.best_metrics else ""
                    lines.append(f"  {key}: {value:.6f}{unit}{best_str}")

        # Vertices
        if vertex_metrics:
            lines.append("\n【Mesh顶点】")
            for key, value in vertex_metrics.items():
                unit = " mm" if 'mpvpe' in key else ""
                best_str = f"\t\t(史上最佳: {self.best_metrics[key]:.6f}{unit})" if key in self.best_metrics else ""
                lines.append(f"  {key}: {value:.6f}{unit}{best_str}")

        lines.append("=" * 60 + "\n")

        return "\n".join(lines)
