'''
@Project ：SPNet 
@File    ：trainer.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2025/11/25 16:39
'''


import json
import os
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from termcolor import cprint
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from src.utils.util import set_random_seed

# from src.dataset.dataset_freihand import FreiHAND  # Original dataset - kept for reference
from src.dataset.dataset_freihand_v2 import FreiHANDV2 as FreiHAND  # New dataset with improved augmentation pipeline
from src.dataset.dataset_ho3d import HO3DDataset

from src.utils.model_utils import make_model,load_pretrained_params

from src.utils.hand_utils import mano_to_mpii

from trainer_components import DataAdapterRegistry,CUDAGraphManager,EvaluationPipeline,VisualizationHelper

from src.utils.data_keys import DataKeys, PredKeys, MetricKeys

from src.model.module.heatmap_manager import HeatmapVisualizer


class Trainer:
    """
    重构的Trainer类 - 更灵活、更易扩展

    核心特性:
    - 模块化的数据处理
    - 可插拔的评估系统
    - 配置驱动的训练流程
    - 支持CUDA Graph加速
    """

    def __init__(self, conf: Dict[str, Any]):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]
        self.logger = conf["logger"]
        self.tb_writer = conf["tb_writer"]
        self.verbose = conf["verbose"]
        self.local_rank = conf["local_rank"]
        self.distributed = conf["distributed"]


        # Training parameters
        self.mode = conf["mode"]
        self.epochs = conf["num_epochs"]
        self.eval_internal = conf["eval_internal"]
        self.model_save_dir = conf["model_save_dir"]
        self.save_model = conf["save_model"]
        # set_random_seed(conf.get("seed", 42))

        # Tracking variables
        self.global_step = 0
        self.eval_step = 0
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.works_seed = 0

        # ============ 1. 初始化数据适配器 ============
        self._init_data_adapter()

        # ============ 2. 初始化数据集和数据加载器 ============
        self._init_dataloaders()

        # ============ 3. 初始化模型 ============
        self.cuda_graph_manager = None
        self._init_model()

        # ============ 4. 初始化优化器 ============
        if self.mode == "train":
            self._init_optimizer()

        # ============ 5. 分布式设置 ============
        if self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            self.logger.info(f"Multi GPU mode: {torch.cuda.device_count()} GPUs")

        # ============ 6. 模型加载 ============
        if 'load_from' in conf:
            self._load_checkpoint(conf['load_from'])

        # ============ 7. 初始化CUDA Graph管理器 ============
        # CUDA Graph is incompatible with DDP due to gradient synchronization hooks
        self.use_cuda_graph = conf.get("use_cuda_graph", True)

        if self.distributed:
            self.use_cuda_graph = False
            print("Warning : CUDA Graph is disabled in DistributedDataParallel mode.")
            print("for step backward has interaction between gpus")

        if self.use_cuda_graph:
            self.cuda_graph_manager = CUDAGraphManager(self.model, self.device, conf)
        else:
            self.cuda_graph_manager = None

        # ============ 8. 初始化评估流程 ============
        self._init_evaluation_pipeline()

        # ============ 9. 初始化可视化助手 ============
        self.vis_helper = VisualizationHelper()
        self.heatmap_visualizer = HeatmapVisualizer()

        # Load J_regressor if available (for visualization when model only outputs verts)
        # J_regressor can convert verts to joints
        if 'mano_path' in self.conf and self.conf['mano_path']:
            try:
                j_reg_path = os.path.join(self.conf['mano_path'], 'j_reg.npy')
                if os.path.exists(j_reg_path):
                    self.j_reg = np.load(j_reg_path)
                    self.j_reg = torch.from_numpy(self.j_reg).float().to(self.device)
                else:
                    self.j_reg = None
            except:
                self.j_reg = None
        else:
            self.j_reg = None

        print("\n" + "="*60)
        print("✅ Trainer初始化完成")
        print(f"   模式: {self.mode}")
        print(f"   数据适配器: {self.adapter_type}")
        print(f"   CUDA Graph: {'启用' if self.use_cuda_graph else '禁用'}")
        print(f"   评估器: {len(self.eval_pipeline.evaluators)} 个")
        print("="*60 + "\n")

    def _init_data_adapter(self):
        """初始化数据适配器"""
        # 根据配置选择合适的数据适配器
        task_name = self.conf.get('task_name', '')

        if 'FreiHand' in task_name or '3d' in task_name.lower():
            self.adapter_type = 'hand_3d'
        else:
            self.adapter_type = 'hand_pose_2d'

        # 允许配置文件覆盖
        self.adapter_type = self.conf.get('data_adapter', self.adapter_type)

        self.data_adapter = DataAdapterRegistry.get_adapter(
            self.adapter_type,
            self.device,
            self.conf
        )

        print(f"✅ 数据适配器: {self.adapter_type}")

    def _init_dataloaders(self):
        """初始化数据加载器"""
        print('#' * 40)
        print('Building Dataset ...')

        if self.mode == 'train':
            # 训练数据集
            if 'FreiHand' in self.conf.get('task_name', '') or \
               'FreiHand' in self.conf.get('dataset_root', '') or \
               'Mobrecon' in self.conf.get('task_name', ''):
                print("Using FreiHAND dataset")
                train_dataset = FreiHAND(self.conf, phase='train')

            elif 'HO3D' in self.conf.get('task_name', '') or \
                    'HO3D' in self.conf.get('dataset_root', ''):
                print("Using HO3D dataset")
                train_dataset = HO3DDataset(self.conf, phase='train')

            else:
                raise NotImplementedError

            # 训练数据加载器
            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, shuffle=True
                )
                self.train_dataloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=int(self.conf["batchsize_train"]),
                    shuffle=False,
                    num_workers=int(self.conf["train_dataloader_workers"]),
                    pin_memory=False,
                    sampler=train_sampler,
                    drop_last=True
                )
            else:
                self.train_dataloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=int(self.conf["batchsize_train"]),
                    shuffle=True,
                    num_workers=int(self.conf["train_dataloader_workers"]),
                    pin_memory=False,
                    drop_last=True
                )

            if self.local_rank==0:
                # 验证数据集
                print('Building evaluation dataset for validation during training...')
                if 'FreiHand' in self.conf.get('task_name', '') or \
                   'FreiHand' in self.conf.get('dataset_root', '') or \
                   'Mobrecon' in self.conf.get('task_name', ''):
                    print("Using FreiHAND dataset")
                    eval_dataset = FreiHAND(self.conf, phase='eval')

                elif "HO3DV2" in self.conf.get('task_name', '') or \
                        "HO3D" in self.conf.get('task_name', ''):
                    print("Using HO3D dataset")
                    eval_dataset = HO3DDataset(self.conf, phase='eval')


                else:
                    raise NotImplementedError
                self.eval_dataloader = DataLoader(
                    dataset=eval_dataset,
                    batch_size=int(self.conf["batchsize_test"]),
                    shuffle=False,
                    num_workers=int(self.conf["eval_dataloader_workers"]),
                    pin_memory=True,
                    drop_last=True
                )

        elif self.mode in ('evaluate', 'inference', 'gradcam'):
            if self.local_rank == 0:
                # 验证数据集
                print('Building evaluation dataset for validation during training...')
                if 'FreiHand' in self.conf.get('task_name', '') or \
                   'FreiHand' in self.conf.get('dataset_root', '') or \
                   'Mobrecon' in self.conf.get('task_name', ''):
                    print("Using FreiHAND dataset")
                    eval_dataset = FreiHAND(self.conf, phase='eval')

                elif "HO3DV2" in self.conf.get('task_name', '') or \
                        "HO3D" in self.conf.get('task_name', ''):
                    print("Using HO3D dataset")
                    eval_dataset = HO3DDataset(self.conf, phase='eval')

                else:
                    raise NotImplementedError
                self.eval_dataloader = DataLoader(
                    dataset=eval_dataset,
                    batch_size=int(self.conf["batchsize_test"]),
                    shuffle=False,
                    num_workers=int(self.conf["eval_dataloader_workers"]),
                    pin_memory=True,
                    drop_last=True
                )

        elif self.mode == 'analysis':
            pass

        else:  # inference
            raise NotImplementedError


    def _init_model(self):
        """初始化模型"""
        print('#' * 40)
        print(f"Model is {self.conf['model_path']}.{self.conf['model_name']}")
        print(f"selfdevice is {self.device}")
        self.model = make_model(self.conf).to(self.device)

        # 模型分析（如果配置中启用）
        if self.conf.get("model_analysis", False):
            self._model_analysis()


    def _init_optimizer(self):
        """初始化优化器"""
        optimizer_type = self.conf['optimizer']
        learning_rate = self.conf["learning_rate"].split(",")

        print(f"Optimizer is {optimizer_type}")
        self.logger.info(f'optimizer is {optimizer_type}')

        if optimizer_type == 'adam':
            if len(learning_rate) > 1:
                # 双学习率设置（预训练参数 + 新参数）
                self._init_dual_lr_optimizer(learning_rate)
            else:
                # 单一学习率
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=float(learning_rate[0]),
                    weight_decay=0.01,
                    capturable=True,
                    foreach=True
                )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=float(learning_rate[0]),
                momentum=0.9,
                weight_decay=0.0000001
            )
        else:
            raise ValueError(f"{optimizer_type} is not supported yet!")

    def _init_dual_lr_optimizer(self, learning_rate: List[str]):
        """初始化双学习率优化器（预训练参数使用较小学习率）"""
        assert "pretrained_lr_step" in self.conf, "pretrained_lr_step is needed!"

        # 分离预训练参数和新参数
        pretrained_named_params = filter(
            lambda named_parameters: (
                    named_parameters[0].split(".")[0] in ["?", "?"]
                    and named_parameters[0].split(".")[-2] not in ["fc"]
            ),
            self.model.named_parameters()
        )

        pretrained_id_name_params = list(map(lambda p: (id(p[1]), p[0], p[1]), pretrained_named_params))
        if pretrained_id_name_params:
            pretrained_id, pretrained_name, pretrained_params = list(zip(*pretrained_id_name_params))
        else:
            pretrained_id, pretrained_name, pretrained_params = [], [], []

        scratch_named_params = filter(
            lambda p: id(p[1]) not in pretrained_id,
            self.model.named_parameters()
        )
        scratch_named_params = list(zip(*scratch_named_params))
        if scratch_named_params:
            scratch_name, scratch_params = scratch_named_params
        else:
            scratch_name = scratch_params = []

        # 打印参数信息
        if self.local_rank == 0:
            print("#" * 40)
            print(f"Pretrained parameters: {len(pretrained_name)}, LR: {learning_rate[0]}")
            for param_name in pretrained_name:
                self.logger.info(param_name)
            print("-" * 70)
            print(f"New parameters: {len(scratch_name)}, LR: {learning_rate[1]}")
            for param_name in scratch_name:
                self.logger.info(param_name)

        # 创建优化器
        self.optimizer = torch.optim.Adam([
            {"params": filter(lambda p: p.requires_grad, pretrained_params), "lr": float(learning_rate[0])},
            {"params": filter(lambda p: p.requires_grad, scratch_params), "lr": float(learning_rate[1])},
        ], lr=float(learning_rate[0]), capturable=True, foreach=True)

    def _load_checkpoint(self, ckpt_path: str):
        """加载检查点"""
        print('#' * 40)
        print(f"Loading pretrained parameters at '{ckpt_path}'")

        if self.device.type == "cpu":
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.current_epoch = checkpoint.get('epoch', 0)
        print(f"{self.current_epoch} epoch model params is loading...")

        model_state_load = checkpoint.get("model", checkpoint)
        model_state = self.model.state_dict()
        state_cur = load_pretrained_params(model_state_load, model_state)
        self.model.load_state_dict(state_cur)

        if self.mode == "train" and "optimizer" in checkpoint:
            print("Loading optimizer parameters...")
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print('Reload successfully')
        print('#' * 40)

    def _init_evaluation_pipeline(self):
        """初始化评估流程"""
        # 可以通过配置指定使用哪些评估器
        evaluator_names = self.conf.get('evaluators', None)

        self.train_pipeline = EvaluationPipeline(
            self.conf,
            self.data_adapter,
            evaluators=evaluator_names
        )

        self.eval_pipeline = EvaluationPipeline(
            self.conf,
            self.data_adapter,
            evaluators=evaluator_names
        )

    def _reparameterize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """重参数化模型用于推理加速（仅重参数化backbone）"""
        if hasattr(model, 'backbone'):
            try:
                from timm.utils.model import reparameterize_model
                model.backbone = reparameterize_model(model.backbone, inplace=True)
                print("  ✅ Backbone重参数化成功")
            except Exception as e:
                print(f"  ⚠️  Backbone重参数化失败: {type(e).__name__}")
        return model

    def _model_analysis(self):
        """
        模型分析：计算参数量、GFLOPs和推理速度

        分析内容：
        1. 总参数量和可训练参数量
        2. 模型的GFLOPs（需要输入尺寸）
        3. 推理速度（FPS）
        """
        print("\n" + "="*70)
        print("📊 模型分析")
        print("="*70)

        # ============ 重参数化模型 ============
        enable_reparam = self.conf.get('model_analysis_reparam', True)
        analysis_model = self.model

        if enable_reparam:
            print(f"\n{'重参数化':^70}")
            print("-"*70)
            print("  尝试重参数化模型...")
            analysis_model = self._reparameterize_model(self.model)


        # # ============ 1. 参数量统计 ============
        # total_params = sum(p.numel() for p in analysis_model.parameters())
        # trainable_params = sum(p.numel() for p in analysis_model.parameters() if p.requires_grad)
        # non_trainable_params = total_params - trainable_params
        #
        # print(f"\n{'参数统计':^70}")
        # print("-"*70)
        # print(f"  总参数量:        {total_params:>15,} ({total_params/1e6:.2f}M)")
        # print(f"  可训练参数:      {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
        # print(f"  不可训练参数:    {non_trainable_params:>15,} ({non_trainable_params/1e6:.2f}M)")
        #
        # # ============ 2. GFLOPs计算 ============
        input_size = int(self.conf.get('input_size', 128))
        batch_size = 1
        #
        # dummy_input = {
        #     DataKeys.IMG: torch.randn(batch_size, 3, input_size, input_size).to(self.device),
        # }
        #
        # # 尝试使用thop库计算FLOPs
        # try:
        #     from thop import profile, clever_format
        #
        #     analysis_model.eval()
        #
        #     # 计算FLOPs
        #     with torch.no_grad():
        #         flops, params = profile(analysis_model, inputs=(dummy_input, False), verbose=False)
        #         flops, params = clever_format([flops, params], "%.3f")
        #
        #     print(f"\n{'计算复杂度 (使用thop)':^70}")
        #     print("-"*70)
        #     print(f"  FLOPs:           {flops:>20}")
        #     print(f"  参数量:          {params:>20}")
        #
        # except ImportError:
        #     print(f"\n{'计算复杂度':^70}")
        #     print("-"*70)
        #     print("  ⚠️  未安装thop库，无法计算FLOPs")
        #     print("  提示: pip install thop")

        # ============ 3. 推理速度测试 ============
        try:
            print(f"\n{'推理速度测试':^70}")
            print("-"*70)

            test_input = {
                DataKeys.IMG: torch.randn(batch_size, 3, input_size, input_size).to(self.device),
            }

            analysis_model.eval()

            # Warmup
            print(f"  预热中... (50次)")
            with torch.no_grad():
                for _ in range(50):
                    _ = analysis_model(test_input, is_train=False)

            # 测试推理速度
            num_iterations = 500
            print(f"  测试中... ({num_iterations}次迭代)")

            # 计时
            latencies = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    t0 = time.perf_counter()
                    _ = analysis_model(test_input, is_train=False)
                    latencies.append((time.perf_counter() - t0) * 1000)

            end_time = time.time()

            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            mean_ms = sum(latencies) / n
            median_ms = latencies_sorted[n // 2]
            p95_ms = latencies_sorted[int(n * 0.95)]
            p99_ms = latencies_sorted[int(n * 0.99)]
            min_ms = latencies_sorted[0]
            max_ms = latencies_sorted[-1]
            throughput = batch_size / (mean_ms / 1000)  # images/sec

            print(f"\n  批次大小:        {batch_size:>20}")
            print(f'  输入大小:        {input_size:>20}')
            print(f"  平均延迟:        {min_ms:>17.2f} ms")
            print(f"  吞吐量:          {throughput:>17.2f} FPS")
            # print(f"  单样本延迟:      {latency/batch_size:>17.2f} ms")
            print(f"  重参数化:        {'启用' if enable_reparam and analysis_model is not self.model else '禁用':>20}")


            # print(analysis_model)

        except Exception as e:
            print(f"\n  ⚠️  推理速度测试失败: {e}")


    def _model_forward(self, prepared_data: Dict[str, torch.Tensor], is_train: bool = True) -> Dict[str, torch.Tensor]:
        """
        模型前向传播的统一接口

        Args:
            prepared_data: 准备好的数据字典(处理过了)
            is_train: 是否训练模式

        Returns:
            模型输出字典
        """
        return self.model(
            prepared_data,  # 解包在forward中完成
            is_train=is_train
        )

    def train(self):
        """训练主循环"""
        print("Begin training ...")
        self.logger.info("Begin training ...")

        print(f"\n{'=' * 60}")
        print(f"🔧 训练模式: {'CUDA Graph' if self.use_cuda_graph else '常规模式'}")
        print(f"{'=' * 60}\n")

        # 从配置加载初始模型（如果指定）
        if 'init_from' in self.conf:
            self._load_checkpoint(self.conf['init_from'])

        for epoch in range(self.current_epoch, int(self.epochs)):
            # val_metrics = self.evaluate(epoch)

            # set_random_seed(epoch)
            # self.evaluate(epoch)#用于测试评估方法，平时应当注释掉
            self.works_seed = epoch

            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            # 更新学习率
            self._update_learning_rate(epoch)

            # 打印epoch信息
            if self.local_rank == 0:
                lr_info = self._get_learning_rate_info()
                log_str = f'{"#" * 30} Epoch: {epoch} ({lr_info}) {"#" * 30}'
                print('\n', log_str)
                self.logger.info(log_str)

            # 训练一个epoch
            epoch_metrics = self._train_epoch(epoch)

            # 打印epoch统计
            if self.local_rank == 0:
                self._log_epoch_metrics(epoch, epoch_metrics, 'train')

            # 保存最佳训练模型
            if epoch_metrics[MetricKeys.LOSS] < getattr(self, 'best_train_loss', float('inf')):
                self.best_train_loss = epoch_metrics[MetricKeys.LOSS]
                self._save_checkpoint(epoch, 'best_train.pth')

            # 保存最新模型（用于断点续训）
            if self.local_rank == 0:
                self._save_checkpoint(epoch, 'latest.pth', include_optimizer=True)
                print(f"Saved latest model at epoch {epoch}")

            # 定期评估
            if (((epoch + 1) % int(self.eval_internal) == 0) or epoch == 0) and self.local_rank == 0:
                val_metrics = self.evaluate(epoch)

                # 保存最佳验证模型
                primary_metric = val_metrics.get('auc_2d', val_metrics.get('auc', 0))
                if primary_metric > self.best_val_metric:
                    self.best_val_metric = primary_metric
                    print(f"✅ 发现新的最佳模型，主要指标: {primary_metric:.4f} (epoch {epoch + 1})")
                    self._save_checkpoint(epoch, 'best_val_model.pth', include_optimizer=True)

                self.model.train()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            Epoch级别的指标字典
        """
        self.model.train()
        self.train_pipeline.reset()

        batch_loss = 0.0
        steps = 0

        for batch_idx, data in enumerate(tqdm(self.train_dataloader,
                                              desc=f'Epoch {epoch + 1}',
                                              dynamic_ncols=True)):
            torch.cuda.synchronize()

            # ============ CUDA Graph初始化（首次） ============
            if self.use_cuda_graph and not self.cuda_graph_manager.is_train_ready() and \
               batch_idx == 0 and epoch == self.current_epoch:

                # 准备样本数据
                sample_prepared = self.data_adapter.prepare_batch(data)

                # 定义前向函数
                def forward_fn(static_inputs):
                    return self._model_forward(static_inputs, is_train=True)

                # 初始化CUDA Graph
                self.cuda_graph_manager.init_train_graph(
                    sample_prepared,
                    self.optimizer,
                    forward_fn,
                    warmup_steps=3
                )

            # ============ 准备数据 ============
            prepared_data = self.data_adapter.prepare_batch(data)
            ground_truth = self.data_adapter.get_ground_truth(data)

            # ============ 前向+反向传播 ============
            if self.use_cuda_graph and self.cuda_graph_manager.is_train_ready():
                # CUDA Graph模式
                outputs = self.cuda_graph_manager.replay_train_graph(prepared_data)
            else:
                # 常规模式
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self._model_forward(prepared_data, is_train=True)
                loss = outputs[PredKeys.LOSS].mean()
                loss.backward()
                self.optimizer.step()

            # ============ 记录指标 ============
            loss_value = outputs[PredKeys.LOSS].mean().item()
            batch_loss += loss_value
            steps += 1
            if hasattr(self.model, "epoch"):
                self.model.epoch = epoch
            # 更新评估流程
            self.train_pipeline.update(outputs, ground_truth)

            # ============ TensorBoard日志 ============
            if self.local_rank == 0:
                self._log_batch_metrics(outputs, 'train')

            # ============ 可视化 ============
            if self.global_step % 100 == 0 and self.local_rank == 0:
                self._visualize_batch(data, outputs, 'train')

            self.global_step += 1

        # 计算epoch级别的指标
        epoch_metrics = self.train_pipeline.compute()
        epoch_metrics[MetricKeys.LOSS] = batch_loss / steps

        return epoch_metrics

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        在验证集上评估模型

        Args:
            epoch: 当前epoch编号

        Returns:
            评估指标字典
        """
        print("正在评估模型性能...")
        self.logger.info("正在评估模型性能...")

        self.model.eval()
        self.eval_pipeline.reset()

        # ============ CUDA Graph初始化（首次） ============
        if self.use_cuda_graph and not self.cuda_graph_manager.is_eval_ready():
            first_batch = next(iter(self.eval_dataloader))
            sample_prepared = self.data_adapter.prepare_batch(first_batch)

            def forward_fn(static_inputs):
                return self._model_forward(static_inputs, is_train=False)

            self.cuda_graph_manager.init_eval_graph(
                sample_prepared,
                forward_fn,
                warmup_steps=3
            )

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.eval_dataloader,
                                                  desc='Evaluating',
                                                  dynamic_ncols=True)):
                # 准备数据
                prepared_data = self.data_adapter.prepare_batch(data)
                ground_truth = self.data_adapter.get_ground_truth(data)

                # 前向传播
                if self.use_cuda_graph and self.cuda_graph_manager.is_eval_ready():
                    outputs = self.cuda_graph_manager.replay_eval_graph(prepared_data)
                else:
                    outputs = self._model_forward(prepared_data, is_train=False)

                # 更新评估指标
                self.eval_pipeline.update(outputs, ground_truth)

                # 可视化
                if self.eval_step % 20 == 0 and self.local_rank == 0:
                    self._visualize_batch(data, outputs, 'evaluate')

                self.eval_step += 1

        # 计算最终指标
        metrics = self.eval_pipeline.compute()

        # 格式化并打印结果
        result_str = self.eval_pipeline.format_results(metrics)
        print(result_str)
        self.logger.info(result_str)

        # 记录到TensorBoard
        if self.local_rank == 0:
            self._log_epoch_metrics(epoch, metrics, 'evaluate')

            # 判断是否为历史最佳模型并保存
            is_best, reason = self.eval_pipeline.is_best_model(metrics)
            if is_best:
                print(f"✅ 发现新的最佳评估模型！{reason} (epoch {epoch+1})")
                self.logger.info(f"✅ 发现新的最佳评估模型！{reason} (epoch {epoch+1})")
                self._save_checkpoint(epoch, 'best_eval.pth', include_optimizer=False)
            else:
                print("当前模型未超过历史最佳模型。")
                self.logger.info("当前模型未超过历史最佳模型。")

        return metrics

    def _update_learning_rate(self, epoch: int):
        """更新学习率"""
        if 'pretrained_lr_step' in self.conf:
            # 双学习率模式
            if (epoch + 1) == int(self.conf.get('new_lr_step', 999999)):
                self.optimizer.param_groups[1]['lr'] *= 0.1
            if (epoch + 1) % int(self.conf['pretrained_lr_step']) == 0:
                lr_gamma = float(self.conf.get('lr_gamma', 0.1))
                self.optimizer.param_groups[0]['lr'] *= lr_gamma
                self.optimizer.param_groups[1]['lr'] *= lr_gamma
        else:
            # 单一学习率模式
            if (epoch + 1) == int(self.conf.get('new_lr_step', 999999)):
                lr_gamma = float(self.conf.get('lr_gamma', 0.1))
                self.optimizer.param_groups[0]['lr'] *= lr_gamma

    def _get_learning_rate_info(self) -> str:
        """获取当前学习率信息"""
        if len(self.optimizer.param_groups) > 1:
            pretrain_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self.optimizer.param_groups[1]['lr']
            return f"pretrain_lr: {pretrain_lr:.6f}, new_lr: {new_lr:.6f}"
        else:
            lr = self.optimizer.param_groups[0]['lr']
            return f"lr: {lr:.6f}"

    def _log_batch_metrics(self, outputs: Dict[str, torch.Tensor], phase: str):
        """记录batch级别的指标到TensorBoard"""
        log_lst = [f'{phase.capitalize()} => step: {self.global_step:>5}']

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                scalar_value = value.mean().item()
                self.tb_writer.add_scalar(f"{phase}/{key}", scalar_value, self.global_step)
                log_lst.append(f'{key}: {scalar_value:.5f}')

        log_str = ',   '.join(log_lst)
        self.logger.info(log_str)

    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], phase: str):
        """记录epoch级别的指标到TensorBoard"""
        for key, value in metrics.items():
            # print("tensornoard",key,type(value))
            if key != 'pck_curve' and isinstance(value, (int, float)):
                # print("tensornoard", key,type(value))
                self.tb_writer.add_scalar(f"{phase}_epoch/{key}", value, epoch)

    def _visualize_batch(self, data: Dict[str, Any], outputs: Dict[str, torch.Tensor], phase: str):
        """可视化一个batch的结果"""
        step = self.global_step if phase == 'train' else self.eval_step

        # 获取图像尺寸
        image_size = data[DataKeys.RGB].shape[-1]  # 假设图像是正方形

        # 准备RGB图像
        # 如果是对比学习(stacked inputs)，RGB可能包含多个图像(channel > 3)
        # 我们只可视化第一张图像
        rgb_tensor = data[DataKeys.RGB][0]
        if rgb_tensor.shape[0] > 3:
            rgb_tensor = rgb_tensor[:3, :, :]

        # 可视化mask
        if DataKeys.MASK in data and PredKeys.SEGMENT in outputs:
            mask_pred = self.vis_helper.visualize_mask(outputs[PredKeys.SEGMENT][0, :, :, :])  # [B,1,H,W] -> [1,H,W]

            # 处理mask GT：可能是单视图[B,H,W]或双视图[B,2,H,W]
            mask_gt = data[DataKeys.MASK]
            if mask_gt.ndim == 3:
                # 单视图模式：[B,H,W] -> 需要扩展通道维度
                mask_target = self.vis_helper.visualize_mask(mask_gt[0:1, :, :])  # [1,H,W]
            elif mask_gt.shape[1] == 2:
                # 双视图模式：[B,2,H,W] -> 取第一个视图
                mask_target = self.vis_helper.visualize_mask(mask_gt[0, 0:1, :, :])  # [1,H,W]
            else:
                # 单视图模式（带通道维度）：[B,1,H,W]
                mask_target = self.vis_helper.visualize_mask(mask_gt[0, :, :, :])  # [1,H,W]

            self.tb_writer.add_image(f"{phase}/mask_pred", mask_pred, global_step=step)
            self.tb_writer.add_image(f"{phase}/mask_label", mask_target, global_step=step)

        # 可视化关键点
        if PredKeys.SKELETON in outputs and DataKeys.KPT_HEATMAP in data:
            kpt_vis = self.vis_helper.visualize_keypoints(
                rgb_tensor,
                outputs[PredKeys.SKELETON][0, :, :, :],
                data[DataKeys.KPT_HEATMAP][0, :, :, :]
            )
            self.tb_writer.add_image(f"{phase}/keypoints", kpt_vis, global_step=step)

            # 可视化热图总和
            heatmap_sum = self.vis_helper.visualize_heatmap_sum(data[DataKeys.KPT_HEATMAP][0, :, :, :])
            self.tb_writer.add_image(f"{phase}/heatmap_sum", heatmap_sum, global_step=step)

            # 可视化masked关键点
            if DataKeys.MASK in data and PredKeys.SEGMENT in outputs:
                mask_pred = self.vis_helper.visualize_mask(outputs[PredKeys.SEGMENT][0:1, :, :])
                kpt_masked = self.vis_helper.apply_mask_to_image(kpt_vis, mask_pred)
                self.tb_writer.add_image(f"{phase}/keypoints_masked", kpt_masked, global_step=step)

        # 可视化joint_img（2D关键点坐标）
        if DataKeys.JOINT_IMG in data and PredKeys.JOINT_IMG in outputs:
            try:
                # 处理预测和GT的维度，如果是对比学习(4维)，只取前2维
                pred_joints = outputs[PredKeys.JOINT_IMG][0]
                if pred_joints.shape[-1] > 2:
                    pred_joints = pred_joints[:, :2]

                gt_joints = data[DataKeys.JOINT_IMG][0]
                if gt_joints.shape[-1] > 2:
                    gt_joints = gt_joints[:, :2]

                joint_img_vis = self.vis_helper.visualize_joint_img(
                    rgb_tensor,
                    pred_joints,
                    gt_joints,
                    image_size
                )
                self.tb_writer.add_image(f"{phase}/joint_img", joint_img_vis, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize joint_img: {e}")

        # 可视化joint_cam（3D关键点投影到2D）
        # Relax condition: allow missing PredKeys.JOINT_CAM if we have VERTS and j_reg
        has_pred_joints = PredKeys.JOINT_CAM in outputs and outputs[PredKeys.JOINT_CAM] is not None
        has_pred_verts = PredKeys.VERTS in outputs and self.j_reg is not None and outputs[PredKeys.VERTS] is not None

        if (DataKeys.JOINT_CAM in data and (has_pred_joints or has_pred_verts) and
                DataKeys.ROOT in data and DataKeys.CALIB in data):
            # try:
                # 处理预测的root（如果模型输出了root，否则使用GT的root）
                pred_root = outputs.get(PredKeys.TRANS, data[DataKeys.ROOT][0, :3])
                if pred_root.dim() > 1:
                    pred_root = pred_root[0]

                gt_root = data[DataKeys.ROOT][0, :3]

                # 处理3D关键点维度
                if has_pred_joints:
                    pred_joints_3d = outputs[PredKeys.JOINT_CAM][0]
                else:
                    # Regress from verts
                    pred_verts = outputs[PredKeys.VERTS][0]  # [778, 3]
                    # j_reg: [21, 778], verts: [778, 3] -> [21, 3]
                    pred_joints_mano = torch.matmul(self.j_reg, pred_verts)
                    # Convert to MPII format (requires numpy)
                    pred_joints_mano_np = pred_joints_mano.detach().cpu().numpy()
                    pred_joints_mpii_np = mano_to_mpii(pred_joints_mano_np)
                    pred_joints_3d = torch.from_numpy(pred_joints_mpii_np).to(pred_verts.device)

                if pred_joints_3d.shape[-1] > 3:
                    pred_joints_3d = pred_joints_3d[:, :3]

                gt_joints_3d = data[DataKeys.JOINT_CAM][0]
                if gt_joints_3d.shape[-1] > 3:
                    gt_joints_3d = gt_joints_3d[:, :3]

                # print(data[DataKeys.CALIB][0, :, :].cpu())

                # 确保所有Tensor都在CPU上，避免设备不一致错误
                joint_cam_vis = self.vis_helper.visualize_joint_cam(
                    rgb_tensor.cpu(),
                    pred_joints_3d.detach().cpu(),
                    gt_joints_3d.cpu(),
                    pred_root.detach().cpu(),
                    gt_root.cpu(),
                    data[DataKeys.CALIB][0, :, :].cpu(),
                    image_size
                )
                vis_3d_skeleton = self.vis_helper.visualize_3d_skeleton_matplotlib(
                    pose_cam=pred_joints_3d.detach().cpu(),
                    root=pred_root.detach().cpu(),
                    image_size=(image_size, image_size)
                )
                self.tb_writer.add_image(f"{phase}/3d_skeleton_matplotlib", vis_3d_skeleton, global_step=step)
                self.tb_writer.add_image(f"{phase}/joint_cam", joint_cam_vis, global_step=step)
            # except Exception as e:
            #     print(f"Warning: Failed to visualize joint_cam: {e}")

        # 可视化verts（mesh顶点，分别显示GT和预测）
        if DataKeys.CALIB in data and DataKeys.ROOT in data:
            # Get faces from model if available
            faces = None
            if hasattr(self.model, 'module'):
                if hasattr(self.model.module, 'face'):
                    faces = self.model.module.face
            elif hasattr(self.model, 'face'):
                faces = self.model.face

            # 可视化GT verts
            if DataKeys.VERTS in data:
                try:
                    gt_root = data[DataKeys.ROOT][0]
                    if gt_root.shape[0] > 3:
                        gt_root = gt_root[:3]

                    gt_verts = data[DataKeys.VERTS][0]
                    if gt_verts.shape[-1] > 3:
                        gt_verts = gt_verts[:, :3]

                    verts_gt_vis = self.vis_helper.visualize_verts(
                        rgb_tensor.cpu(),
                        gt_verts.cpu(),  # [778, 3]
                        gt_root.cpu(),  # [3]
                        data[DataKeys.CALIB][0, :, :].cpu(),  # [4, 4]
                        alpha=0.7,
                        faces=faces
                    )
                    self.tb_writer.add_image(f"{phase}/verts_gt", verts_gt_vis, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to visualize verts_gt: {e}")

            # 可视化预测的verts
            if PredKeys.VERTS in outputs:
                if outputs[PredKeys.VERTS] is not None:
                    try:
                        # 处理预测的root（如果模型输出了root，否则使用GT的root）
                        if PredKeys.TRANS in outputs:
                            pred_root = outputs[PredKeys.TRANS][0]
                        else:
                            pred_root = data[DataKeys.ROOT][0]

                        if pred_root.dim() > 1:
                            pred_root = pred_root[0]

                        if pred_root.shape[0] > 3:
                            pred_root = pred_root[:3]

                        pred_verts = outputs[PredKeys.VERTS][0]
                        if pred_verts.shape[-1] > 3:
                            pred_verts = pred_verts[:, :3]

                        verts_pred_vis = self.vis_helper.visualize_verts(
                            rgb_tensor.cpu(),
                            pred_verts.detach().cpu(),   # [778, 3]
                            pred_root.detach().cpu(),    # [3]
                            data[DataKeys.CALIB][0, :, :].cpu(),      # [4, 4]
                            alpha=0.8,
                            faces=faces
                        )
                        vis_3d_mesh = self.vis_helper.visualize_3d_mesh_matplotlib(
                            mesh_verts=pred_verts.detach().cpu(),
                            root=pred_root.detach().cpu(),
                            faces=faces,
                            image_size=(image_size, image_size)
                        )
                        vis_2d_mesh = self.vis_helper.visualize_2d_mesh_matplotlib(
                            rgb_image=rgb_tensor.cpu(),
                            mesh_verts=pred_verts.detach().cpu(),
                            root=pred_root.detach().cpu(),
                            calib=data[DataKeys.CALIB][0].cpu(),
                            faces=faces
                        )
                        self.tb_writer.add_image(f"{phase}/2d_mesh_matplotlib", vis_2d_mesh, global_step=step)
                        self.tb_writer.add_image(f"{phase}/3d_mesh_matplotlib", vis_3d_mesh, global_step=step)
                        self.tb_writer.add_image(f"{phase}/verts_pred", verts_pred_vis, global_step=step)
                    except Exception as e:
                        print(f"Warning: Failed to visualize verts_pred: {e}")

        # ==================== Matplotlib-based Visualization (Evaluation Only) ====================
        # 使用matplotlib生成高质量可视化（仅在评估阶段）
        if phase != 'train':
            # 获取faces（如果模型有的话）
            faces = None
            if hasattr(self.model, 'module'):
                if hasattr(self.model.module, 'face'):
                    faces = self.model.module.face
            elif hasattr(self.model, 'face'):
                faces = self.model.face

            # 1. 2D骨架可视化（彩色版本）
            if DataKeys.JOINT_IMG in data and PredKeys.JOINT_IMG in outputs:
                try:
                    pred_joints = outputs[PredKeys.JOINT_IMG][0]
                    if pred_joints.shape[-1] > 2:
                        pred_joints = pred_joints[:, :2]

                    vis_2d_skeleton_plt = self.vis_helper.visualize_2d_skeleton_matplotlib(
                        rgb_image=rgb_tensor.cpu(),
                        pose_uv=pred_joints.detach().cpu(),
                        image_size=image_size
                    )
                    self.tb_writer.add_image(f"{phase}/2d_skeleton_matplotlib", vis_2d_skeleton_plt, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to visualize 2d_skeleton_matplotlib: {e}")

            # 2. 3D骨架可视化
            if PredKeys.JOINT_CAM in outputs and PredKeys.ROOT in data:
                try:
                    # 处理预测的root（如果模型输出了root，否则使用GT的root）
                    if PredKeys.TRANS in outputs:
                        pred_root = outputs[PredKeys.TRANS][0]
                    else:
                        pred_root = data[DataKeys.ROOT][0]
                    if pred_root.dim() > 1:
                        pred_root = pred_root[0]
                    if pred_root.shape[0] > 3:
                        pred_root = pred_root[:3]

                    pred_joint_cam = outputs[PredKeys.JOINT_CAM][0]
                    if pred_joint_cam.shape[-1] > 3:
                        pred_joint_cam = pred_joint_cam[:, :3]

                    vis_3d_skeleton = self.vis_helper.visualize_3d_skeleton_matplotlib(
                        pose_cam=pred_joint_cam.detach().cpu(),
                        root=pred_root.detach().cpu(),
                        image_size=(image_size, image_size)
                    )
                    self.tb_writer.add_image(f"{phase}/3d_skeleton_matplotlib", vis_3d_skeleton, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to visualize 3d_skeleton_matplotlib: {e}")

            # 3. 3D mesh可视化
            if PredKeys.VERTS in outputs and PredKeys.TRANS in outputs and faces is not None:
                try:
                    pred_root = outputs[PredKeys.TRANS][0]
                    if pred_root.dim() > 1:
                        pred_root = pred_root[0]
                    if pred_root.shape[0] > 3:
                        pred_root = pred_root[:3]

                    pred_verts = outputs[PredKeys.VERTS][0]
                    if pred_verts.shape[-1] > 3:
                        pred_verts = pred_verts[:, :3]

                    vis_3d_mesh = self.vis_helper.visualize_3d_mesh_matplotlib(
                        mesh_verts=pred_verts.detach().cpu(),
                        root=pred_root.detach().cpu(),
                        faces=faces,
                        image_size=(image_size, image_size)
                    )
                    self.tb_writer.add_image(f"{phase}/3d_mesh_matplotlib", vis_3d_mesh, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to visualize 3d_mesh_matplotlib: {e}")

            # 4. 2D mesh投影可视化
            if PredKeys.VERTS in outputs and PredKeys.TRANS in outputs and DataKeys.CALIB in data and faces is not None:
                try:
                    pred_root = outputs[PredKeys.TRANS][0]
                    if pred_root.dim() > 1:
                        pred_root = pred_root[0]
                    if pred_root.shape[0] > 3:
                        pred_root = pred_root[:3]

                    pred_verts = outputs[PredKeys.VERTS][0]
                    if pred_verts.shape[-1] > 3:
                        pred_verts = pred_verts[:, :3]

                    vis_2d_mesh = self.vis_helper.visualize_2d_mesh_matplotlib(
                        rgb_image=rgb_tensor.cpu(),
                        mesh_verts=pred_verts.detach().cpu(),
                        root=pred_root.detach().cpu(),
                        calib=data[DataKeys.CALIB][0].cpu(),
                        faces=faces
                    )
                    self.tb_writer.add_image(f"{phase}/2d_mesh_matplotlib", vis_2d_mesh, global_step=step)
                except Exception as e:
                    print(f"Warning: Failed to visualize 2d_mesh_matplotlib: {e}")

        # ==================== 多尺度热图可视化 ====================
        # 可视化21个关键点热图、5个手指热图、整手热图
        # 可视化关键点热图（GT + Pred）
        if DataKeys.KPT_HEATMAP_FULL in outputs:
            try:
                gt_kpt_heatmap = outputs[DataKeys.KPT_HEATMAP_FULL][0].cpu().numpy()  # [21, 56, 56]
                gt_kpt_vis = self.heatmap_visualizer.visualize_kpt_heatmaps_summary(gt_kpt_heatmap)
                gt_kpt_tensor = torch.from_numpy(gt_kpt_vis).unsqueeze(0)  # [1, H, W]
                self.tb_writer.add_image(f"{phase}/heatmap_gt_kpt", gt_kpt_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize GT kpt heatmap: {e}")

        if PredKeys.KPT_HEATMAP_PRED in outputs:
            try:
                pred_kpt_heatmap = outputs[PredKeys.KPT_HEATMAP_PRED][0].detach().cpu().numpy()  # [21, 56, 56]
                pred_kpt_vis = self.heatmap_visualizer.visualize_kpt_heatmaps_summary(pred_kpt_heatmap)
                pred_kpt_tensor = torch.from_numpy(pred_kpt_vis).unsqueeze(0)  # [1, H, W]
                self.tb_writer.add_image(f"{phase}/heatmap_pred_kpt", pred_kpt_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize pred kpt heatmap: {e}")

        # 可视化手指热图（GT + Pred）
        if DataKeys.FINGER_HEATMAP in outputs:
            try:
                gt_finger_heatmap = outputs[DataKeys.FINGER_HEATMAP][0].cpu().numpy()  # [5, H, W]
                gt_finger_vis_list = self.heatmap_visualizer.visualize_finger_heatmaps(gt_finger_heatmap)
                for i, gt_finger_vis in enumerate(gt_finger_vis_list):
                    gt_finger_tensor = torch.from_numpy(gt_finger_vis).unsqueeze(0)  # [1, H, W]
                    self.tb_writer.add_image(f"{phase}/heatmap_gt_finger_{i}", gt_finger_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize GT finger heatmap: {e}")

        if PredKeys.FINGER_HEATMAP_PRED in outputs:
            try:
                pred_finger_heatmap = outputs[PredKeys.FINGER_HEATMAP_PRED][0].detach().cpu().numpy()  # [C, H, W]
                pred_finger_vis_list = self.heatmap_visualizer.visualize_finger_heatmaps(pred_finger_heatmap)
                for i, pred_finger_vis in enumerate(pred_finger_vis_list):
                    pred_finger_tensor = torch.from_numpy(pred_finger_vis).unsqueeze(0)  # [1, H, W]
                    self.tb_writer.add_image(f"{phase}/heatmap_pred_finger_{i}", pred_finger_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize pred finger heatmap: {e}")

        # 可视化整手热图（GT + Pred）
        if DataKeys.HAND_HEATMAP in outputs:
            try:
                gt_hand_heatmap = outputs[DataKeys.HAND_HEATMAP][0].cpu().numpy()  # [1, H, W]
                gt_hand_vis = self.heatmap_visualizer.visualize_hand_heatmap(gt_hand_heatmap[0])
                gt_hand_tensor = torch.from_numpy(gt_hand_vis).unsqueeze(0)  # [1, H, W]
                self.tb_writer.add_image(f"{phase}/heatmap_gt_hand", gt_hand_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize GT hand heatmap: {e}")

        if PredKeys.HAND_HEATMAP_PRED in outputs:
            try:
                pred_hand_heatmap = outputs[PredKeys.HAND_HEATMAP_PRED][0].detach().cpu().numpy()  # [1, H, W]
                pred_hand_vis = self.heatmap_visualizer.visualize_hand_heatmap(pred_hand_heatmap[0])
                pred_hand_tensor = torch.from_numpy(pred_hand_vis).unsqueeze(0)  # [1, H, W]
                self.tb_writer.add_image(f"{phase}/heatmap_pred_hand", pred_hand_tensor, global_step=step)
            except Exception as e:
                print(f"Warning: Failed to visualize pred hand heatmap: {e}")

    def _save_checkpoint(self, epoch: int, filename: str, include_optimizer: bool = False):
        """保存检查点"""
        if not os.path.isdir(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            "epoch": epoch + 1,
            "model": model_state_dict,
            "global_step": self.global_step,
        }

        if include_optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["best_val_metric"] = self.best_val_metric
            checkpoint["best_train_loss"] = getattr(self, 'best_train_loss', float('inf'))

        save_path = os.path.join(self.model_save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f'Model saved to {save_path}')



    def inference(self, data_save_dir, img_folder=None):
        """
        推理模式：对测试集或自定义图片进行推理并可视化手部mesh

        Args:
            data_save_dir: 保存结果的目录
            img_folder: 可选，自定义图片文件夹路径。如果提供则推理该文件夹下的图片，否则推理数据集
        """
        print("开始推理...")
        self.logger.info("开始推理...")

        # 导入必要的库
        import glob
        from src.utils.hand_mesh_renderer import HandMeshRenderer

        # 创建保存目录
        os.makedirs(data_save_dir, exist_ok=True)

        # 获取faces
        faces = self._get_faces()
        if faces is None:
            print("警告: 无法获取faces，将跳过mesh渲染")
            return

        # 初始化渲染器
        renderer = HandMeshRenderer(faces, watertight=True)

        self.model.eval()

        # 判断是使用自定义图片还是测试集
        if img_folder is not None:
            print(f"使用自定义图片文件夹进行推理: {img_folder}")
            self._inference_on_images(renderer, data_save_dir, img_folder)
        elif hasattr(self, 'eval_dataloader'):
            print("使用测试集进行推理...")
            self._inference_on_dataset(renderer, data_save_dir)
        else:
            print("错误: 未提供图片文件夹且未初始化测试集")
            return

    def _get_faces(self):
        """获取MANO faces"""
        # 从模型获取
        faces = None
        if hasattr(self.model, 'module'):
            if hasattr(self.model.module, 'face'):
                faces = self.model.module.face
        elif hasattr(self.model, 'face'):
            faces = self.model.face

        if faces is not None:
            if isinstance(faces, torch.Tensor):
                return faces.detach().cpu().numpy()
            return faces

        # 从MANO layer获取
        try:
            from manotorch.manolayer import ManoLayer
            mano_layer = ManoLayer(
                rot_mode="axisang",
                use_pca=False,
                mano_assets_root="/home/wxl/code/SPNet/mano_v1_2/",
                center_idx=None,
                flat_hand_mean=True,
            )
            return np.array(mano_layer.th_faces).astype(np.long)
        except:
            return None

    def _inference_on_dataset(self, renderer, save_dir):
        """在测试集上推理"""
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.eval_dataloader, desc='推理中')):
                prepared_data = self.data_adapter.prepare_batch(data)
                outputs = self._model_forward(prepared_data, is_train=False)

                # 渲染每张图片
                batch_size = data[DataKeys.RGB].shape[0]
                for i in range(batch_size):
                    self._render_and_save_dataset(
                        data, outputs, i, renderer, save_dir,
                        f"batch{batch_idx:04d}_img{i:02d}.jpg"
                    )

    def _inference_on_images(self, renderer, save_dir, img_folder):
        """在自定义图片上推理"""
        # 获取图片路径
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not img_paths:
            print(f"未找到图片: {img_folder}")
            return

        print(f"找到 {len(img_paths)} 张图片")

        # 默认相机内参 (假设图片尺寸为224x224)
        img_size = int(self.conf.get('input_size', 640))

        with torch.no_grad():
            for img_path in tqdm(img_paths, desc='推理中'):
                # 读取并预处理图片
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (img_size, img_size))

                # 转换为tensor并归一化 (mean=0.5, std=0.5)
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor - 0.5) / 0.5
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # 构造输入数据
                data = {
                    DataKeys.RGB: img_tensor,
                }

                # 推理
                prepared_data = self.data_adapter.prepare_batch(data)
                outputs = self._model_forward(prepared_data, is_train=False)

                # 渲染并保存
                filename = os.path.basename(img_path)
                self._render_and_save_custom(
                    data, outputs, 0, renderer, save_dir,
                    f"result_{filename}", original_img=img_rgb
                )


    def _render_and_save_dataset(self, data, outputs, idx, renderer, save_dir, filename):
        """数据集推理：使用数据集提供的root和calib渲染mesh（分别保存GT和预测）"""
        import pyrender
        import trimesh

        # 获取预测的verts
        if PredKeys.VERTS not in outputs or outputs[PredKeys.VERTS] is None:
            return

        pred_verts = outputs[PredKeys.VERTS][idx].detach().cpu().numpy()
        pred_verts = pred_verts * 0.2

        # 获取原始图片
        img_tensor = data[DataKeys.RGB][idx].cpu()
        if img_tensor.shape[0] > 3:
            img_tensor = img_tensor[:3]
        img_rgb = ((img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        img_h, img_w = img_rgb.shape[:2]

        try:
            # 获取faces
            from manotorch.manolayer import ManoLayer
            mano_layer = ManoLayer(
                rot_mode="axisang",
                use_pca=False,
                mano_assets_root="/home/wxl/code/SPNet/mano_v1_2/",
                center_idx=None,
                flat_hand_mean=True,
            )
            faces = mano_layer.th_faces.cpu().numpy()

            # 使用数据集提供的calib和root
            calib = data[DataKeys.CALIB][idx].cpu().numpy()
            if calib.shape[0] == 4:
                K = calib[:3, :3]
            else:
                K = calib

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # 渲染预测的mesh
            if PredKeys.TRANS in outputs:
                pred_root = outputs[PredKeys.TRANS][idx].detach().cpu().numpy()
            else:
                pred_root = data[DataKeys.ROOT][idx].cpu().numpy()

            if pred_root.shape[0] > 3:
                pred_root = pred_root[:3]

            pred_verts_abs = pred_verts + pred_root
            img_pred = self._render_mesh(img_rgb, pred_verts_abs, faces, fx, fy, cx, cy,
                                        img_w, img_h, color=(0.9, 0.85, 0.67, 1.0))  # 淡蓝色

            # 保存预测结果
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            pred_path = os.path.join(save_dir, f"{base_name}_pred{ext}")
            cv2.imwrite(pred_path, img_pred)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_input{ext}"), img_rgb[:, :, ::-1])  # 保存输入图像（BGR格式）

            # 保存预测mesh的PLY文件
            pred_mesh_ply = trimesh.Trimesh(pred_verts_abs, faces)
            pred_mesh_ply.export(os.path.join(save_dir, f"{base_name}_pred.ply"))

            # 渲染GT的mesh（如果存在）
            if DataKeys.VERTS in data and DataKeys.ROOT in data:
                gt_verts = data[DataKeys.VERTS][idx].cpu().numpy()
                gt_verts = gt_verts * 0.2
                gt_root = data[DataKeys.ROOT][idx].cpu().numpy()

                if gt_root.shape[0] > 3:
                    gt_root = gt_root[:3]

                gt_verts_abs = gt_verts + gt_root
                img_gt = self._render_mesh(img_rgb, gt_verts_abs, faces, fx, fy, cx, cy,
                                          img_w, img_h, color=(0.9, 0.85, 0.67, 1.0))  # 淡蓝色

                # 保存GT结果
                gt_path = os.path.join(save_dir, f"{base_name}_gt{ext}")
                cv2.imwrite(gt_path, img_gt)

                # 保存GT mesh的PLY文件
                gt_mesh_ply = trimesh.Trimesh(gt_verts_abs, faces)
                gt_mesh_ply.export(os.path.join(save_dir, f"{base_name}_gt.ply"))

        except Exception as e:
            print(f"数据集渲染失败: {e}")
            import traceback
            traceback.print_exc()

    def _render_mesh(self, img_rgb, verts_abs, faces, fx, fy, cx, cy, img_w, img_h, color=(0.8, 0.3, 0.3, 1.0)):
        """渲染单个mesh到图像上"""
        import os
        # 设置pyrender使用OSMesa后端（无显示环境）
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        import pyrender
        import trimesh

        # 创建mesh
        verts_render = verts_abs.copy()

        vertex_colors = np.array([color] * len(verts_render))
        mesh = trimesh.Trimesh(verts_render, faces, vertex_colors=vertex_colors)

        # 旋转180度（pyrender约定）
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # 创建pyrender场景
        mesh_pr = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh_pr, 'mesh')

        # 使用数据集提供的完整相机内参
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
        camera_pose = np.eye(4)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        # 添加光照
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=camera_pose)

        # 渲染
        r = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        rendered_color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        rendered_color = rendered_color.astype(np.float32) / 255.0
        r.delete()

        # Alpha混合
        img_rgb_float = img_rgb.astype(np.float32) / 255.0
        img_rgb_float = img_rgb_float[:, :, ::-1]  # RGB to BGR
        img_with_mesh = (img_rgb_float[:, :, :3] * (1 - rendered_color[:, :, 3:]) +
                       rendered_color[:, :, :3] * rendered_color[:, :, 3:])
        img_with_mesh = (255 * img_with_mesh).astype(np.uint8)

        return img_with_mesh

    def _render_and_save_custom(self, data, outputs, idx, renderer, save_dir, filename, original_img=None):
        """自定义图片推理：使用2D-1D配准渲染mesh"""
        from src.utils.mobrecon_utils import registration

        # 获取预测的verts
        if PredKeys.VERTS not in outputs or outputs[PredKeys.VERTS] is None:
            return

        pred_verts = outputs[PredKeys.VERTS][idx].detach().cpu().numpy()
        pred_verts = pred_verts * 0.2

        # 获取原始图片
        img_rgb = original_img.copy()
        img_h, img_w = original_img.shape[:2]

        try:
            # 获取faces
            from manotorch.manolayer import ManoLayer
            mano_layer = ManoLayer(
                rot_mode="axisang",
                use_pca=False,
                mano_assets_root="/home/wxl/code/SPNet/mano_v1_2/",
                center_idx=None,
                flat_hand_mean=True,
            )
            faces = mano_layer.th_faces.cpu().numpy()

            # 构建相机内参
            focal_length = img_w * 1.2
            fx = fy = focal_length
            cx, cy = img_w / 2, img_h / 2

            # 2D-1D配准
            if PredKeys.JOINT_IMG in outputs and outputs[PredKeys.JOINT_IMG] is not None:
                pred_2d = outputs[PredKeys.JOINT_IMG][idx].detach().cpu().numpy()
                pred_2d = pred_2d * img_w

                K = np.array([[focal_length, 0, img_w / 2],
                              [0, focal_length, img_h / 2],
                              [0, 0, 1]], dtype=np.float32)

                j_regressor = mano_layer.th_J_regressor.cpu().numpy()
                pred_verts = registration(pred_verts, pred_2d, j_regressor, K, img_w)

            # 渲染mesh（registration已经将顶点对齐到图像空间，所以顶点就是绝对坐标）
            img_with_mesh = self._render_mesh(img_rgb, pred_verts, faces, fx, fy, cx, cy,
                                             img_w, img_h, color=(0.7, 0.2, 0.2, 1.0))

            # 保存结果
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img_with_mesh)

        except Exception as e:
            print(f"自定义图片渲染失败: {e}")
            import traceback
            traceback.print_exc()

    def visualize_grad_cam(self, save_dir: str, target_layer_name: str = 'backbone', max_samples: int = 50):
        """
        多尺度 Grad-CAM：分别观察3个热图监督分支对 backbone 各尺度特征图的影响。

        3个尺度对应关系：
          scale0: backbone d1 [152@28x28] <- kpt_heatmap (21关节热图)
          scale1: backbone d2 [304@14x14] <- kpt_heatmap_f (5指热图)
          scale2: backbone e3 [608@7x7]  <- kpt_heatmap_h (手部热图)

        保存文件命名: scale{0,1,2}_{batch}_{i}.jpg
        """
        import torch.nn.functional as F

        os.makedirs(save_dir, exist_ok=True)

        # ---- 找到 backbone 模块 ----
        backbone = None
        for name, module in self.model.named_modules():
            if name == 'backbone':
                backbone = module
                break
        if backbone is None:
            raise ValueError("未找到 'backbone' 模块")

        # ---- 注册 hook，捕获3个尺度的特征图 ----
        # backbone.forward 返回 [d1, d2, e3]
        scale_acts = [None, None, None]  # 每个尺度的激活，需要 retain_grad

        def fwd_hook(module, input, output):
            # output: list [d1, d2, e3]，各自 retain_grad 以便后续 backward
            for k in range(3):
                scale_acts[k] = output[k]
                output[k].retain_grad()

        fwd_handle = backbone.register_forward_hook(fwd_hook)

        self.model.eval()
        count = 0

        # 3个尺度对应的 score key 和文件前缀
        score_keys = [
            PredKeys.KPT_HEATMAP_PRED,    # scale0: 21关节热图
            # PredKeys.FINGER_HEATMAP_PRED,  # scale1: 5指热图
            # PredKeys.HAND_HEATMAP_PRED,    # scale2: 手部热图
        ]
        scale_names = ['scale0_kpt']#, 'scale1_finger']#, 'scale2_hand']

        for batch_idx, data in enumerate(tqdm(self.eval_dataloader, desc='Grad-CAM')):
            if count >= max_samples:
                break

            prepared_data = self.data_adapter.prepare_batch(data)

            # 原图准备（用于叠加）
            img_key = DataKeys.RGB if DataKeys.RGB in data else DataKeys.IMG
            batch_size = data[img_key].shape[0]

            # 对3个尺度分别做一次 forward + backward
            cams = []  # 每个尺度的 cam [B, 1, H, W]
            for scale_idx, score_key in enumerate(score_keys):
                self.model.zero_grad()
                outputs = self._model_forward(prepared_data, is_train=False)

                heatmap_pred = outputs.get(score_key)[:,9:10,:,:]  # 取第一个通道作为score（假设是手部热图的第一个通道）
                if heatmap_pred is None:
                    cams.append(None)
                    continue

                score = heatmap_pred.mean()
                score.backward()

                act = scale_acts[scale_idx]   # [B, C, H, W]
                grad = act.grad               # [B, C, H, W]
                if grad is None:
                    cams.append(None)
                    continue

                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [B, 1, H, W]
                cams.append(cam.detach())

            # ---- 保存每张图的3个尺度 Grad-CAM ----
            for i in range(batch_size):
                if count >= max_samples:
                    break

                img_t = data[img_key][i].cpu()
                if img_t.shape[0] > 3:
                    img_t = img_t[:3]
                img_rgb = ((img_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                img_bgr = img_rgb[:, :, ::-1].copy()
                H, W = img_bgr.shape[:2]

                for scale_idx, (cam, name) in enumerate(zip(cams, scale_names)):
                    if cam is None:
                        continue
                    c = cam[i, 0].cpu().numpy()
                    c = c - c.min()
                    if c.max() > 0:
                        c = c / c.max()
                    c_resized = cv2.resize(c, (W, H))
                    heatmap_vis = cv2.applyColorMap((c_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    alpha = (0.35 + 0.5 * c_resized)[:, :, np.newaxis]  # 低激活透明，高激活不透明
                    overlay = (img_bgr * (1 - alpha) + heatmap_vis * alpha).astype(np.uint8)
                    save_path = os.path.join(save_dir, f"{batch_idx:04d}_{i:02d}_{name}.jpg")
                    cv2.imwrite(save_path, overlay)

                count += 1

        fwd_handle.remove()
        print(f"Grad-CAM 完成，共保存 {count} 张图（每张3个尺度）到 {save_dir}")









