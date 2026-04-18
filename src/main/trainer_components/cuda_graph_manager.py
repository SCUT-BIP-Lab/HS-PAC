"""
CUDA Graph Manager: Handles CUDA Graph initialization and execution in a data-agnostic way
"""

from src.utils.data_keys import LossKeys
import torch
from typing import Dict, Any, Optional, Callable


class CUDAGraphManager:
    """
    Manages CUDA Graph creation, warmup, and replay for both training and evaluation
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, conf: Dict[str, Any]):
        self.model = model
        self.device = device
        self.conf = conf

        # Training graph
        self.train_graph = None
        self.train_stream = None
        self.train_static_inputs = {}
        self.train_static_outputs = {}
        self.train_graph_ready = False

        # Evaluation graph
        self.eval_graph = None
        self.eval_stream = None
        self.eval_static_inputs = {}
        self.eval_static_outputs = {}
        self.eval_graph_ready = False

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_train_graph(self,
                         sample_data: Dict[str, torch.Tensor],
                         optimizer: torch.optim.Optimizer,
                         forward_fn: Callable,
                         warmup_steps: int = 5):
        """
        Initialize CUDA Graph for training

        Args:
            sample_data: Sample batch of prepared data tensors
            optimizer: The optimizer instance
            forward_fn: Function that takes static inputs and returns model outputs
            warmup_steps: Number of warmup iterations
        """
        print("\n" + "="*60)
        print("🚀 初始化训练 CUDA Graph...")
        print("="*60)

        # Create static input buffers
        for key, value in sample_data.items():
            if value is not None:
                self.train_static_inputs[key] = value.clone().contiguous()
            else:
                self.train_static_inputs[key] = None

        # Create independent CUDA stream
        self.train_stream = torch.cuda.Stream()

        # Warmup phase
        print(f"📊 预热阶段：执行{warmup_steps}次前向+反向...")
        self.model.train()

        with torch.cuda.stream(self.train_stream):
            # Copy sample data to static buffers
            for key, value in sample_data.items():
                if value is not None and self.train_static_inputs[key] is not None:
                    self.train_static_inputs[key].copy_(value)

            # Warmup iterations
            for i in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                outputs = forward_fn(self.train_static_inputs)

                # Compute loss
                loss = outputs[LossKeys.LOSS].mean()
                loss.backward()
                optimizer.step()

                # Initialize output buffers on first iteration
                if i == 0:
                    self._init_output_buffers(outputs, self.train_static_outputs)

                print(f"  预热 {i+1}/{warmup_steps} 完成，loss={loss.item():.4f}")

        # Wait for warmup to complete
        self.train_stream.synchronize()
        torch.cuda.synchronize()

        # Capture computation graph
        print("📸 捕获训练 CUDA Graph...")
        self.train_graph = torch.cuda.CUDAGraph()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.stream(self.train_stream):
            with torch.cuda.graph(self.train_graph):
                # Forward pass
                outputs = forward_fn(self.train_static_inputs)

                # Save outputs to static buffers
                loss_mean = outputs['loss'].mean()
                self._copy_outputs_to_buffers(outputs, self.train_static_outputs)

                # Backward pass
                loss_mean.backward()

                # Optimizer step
                optimizer.step()

        # Synchronize
        self.train_stream.synchronize()
        torch.cuda.synchronize()

        print("✅ 训练 CUDA Graph 初始化完成！")
        print("="*60 + "\n")

        self.train_graph_ready = True

    def init_eval_graph(self,
                        sample_data: Dict[str, torch.Tensor],
                        forward_fn: Callable,
                        warmup_steps: int = 3):
        """
        Initialize CUDA Graph for evaluation

        Args:
            sample_data: Sample batch of prepared data tensors
            forward_fn: Function that takes static inputs and returns model outputs
            warmup_steps: Number of warmup iterations
        """
        print("\n" + "="*60)
        print("🔍 初始化评估 CUDA Graph...")
        print("="*60)

        # Create static input buffers
        for key, value in sample_data.items():
            if value is not None:
                self.eval_static_inputs[key] = value.clone().contiguous()
            else:
                self.eval_static_inputs[key] = None

        # Warmup phase in default stream first
        print(f"📊 评估模式预热：执行{warmup_steps}次前向...")
        self.model.eval()

        # 保存原始 CuDNN 设置
        original_benchmark = torch.backends.cudnn.benchmark
        original_deterministic = torch.backends.cudnn.deterministic

        # 为 CUDA Graph 配置 CuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Create independent evaluation stream BEFORE warmup
        self.eval_stream = torch.cuda.Stream()

        # Warmup in the SAME stream that will be used for capture
        with torch.no_grad():
            with torch.cuda.stream(self.eval_stream):
                for i in range(warmup_steps):
                    outputs = forward_fn(self.eval_static_inputs)

                    # Initialize output buffers on first iteration
                    if i == 0:
                        self._init_output_buffers(outputs, self.eval_static_outputs)

                    print(f"  评估预热 {i+1}/{warmup_steps} 完成")

        # Synchronize before capture
        self.eval_stream.synchronize()
        torch.cuda.synchronize()

        # Capture evaluation graph in the same stream
        print("📸 捕获评估 CUDA Graph...")
        self.eval_graph = torch.cuda.CUDAGraph()

        with torch.no_grad():
            with torch.cuda.stream(self.eval_stream):
                with torch.cuda.graph(self.eval_graph):
                    outputs = forward_fn(self.eval_static_inputs)

                    # Save outputs to static buffers
                    self._copy_outputs_to_buffers(outputs, self.eval_static_outputs)

        # Synchronize
        self.eval_stream.synchronize()
        torch.cuda.synchronize()

        # 恢复原始 CuDNN 设置
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        print("✅ 评估 CUDA Graph 初始化完成！")
        print("="*60 + "\n")

        self.eval_graph_ready = True

    def replay_train_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Replay training graph with new data

        Args:
            data: New batch of prepared data tensors

        Returns:
            Dictionary of output tensors
        """
        if not self.train_graph_ready:
            raise RuntimeError("Training CUDA Graph not initialized. Call init_train_graph first.")

        # Copy new data to static buffers
        for key, value in data.items():
            if value is not None and key in self.train_static_inputs and self.train_static_inputs[key] is not None:
                self.train_static_inputs[key].copy_(value)

        # Replay graph
        with torch.cuda.stream(self.train_stream):
            self.train_graph.replay()

        # Synchronize
        self.train_stream.synchronize()
        torch.cuda.synchronize()

        # Clone outputs
        outputs = {}
        for key, value in self.train_static_outputs.items():
            if value is not None:
                outputs[key] = value.clone()

        return outputs

    def replay_eval_graph(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Replay evaluation graph with new data

        Args:
            data: New batch of prepared data tensors

        Returns:
            Dictionary of output tensors
        """
        if not self.eval_graph_ready:
            raise RuntimeError("Evaluation CUDA Graph not initialized. Call init_eval_graph first.")

        # Copy new data to static buffers
        for key, value in data.items():
            if value is not None and key in self.eval_static_inputs and self.eval_static_inputs[key] is not None:
                self.eval_static_inputs[key].copy_(value)

        # Replay graph
        with torch.cuda.stream(self.eval_stream):
            self.eval_graph.replay()

        # Synchronize
        self.eval_stream.synchronize()
        torch.cuda.synchronize()

        # Clone outputs
        outputs = {}
        for key, value in self.eval_static_outputs.items():
            if value is not None:
                outputs[key] = value.clone()

        return outputs

    def _init_output_buffers(self, outputs: Dict[str, torch.Tensor], buffer_dict: Dict[str, torch.Tensor]):
        """Initialize output buffers based on first forward pass"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                buffer_dict[key] = torch.zeros_like(value)
            else:
                buffer_dict[key] = None

    def _copy_outputs_to_buffers(self, outputs: Dict[str, torch.Tensor], buffer_dict: Dict[str, torch.Tensor]):
        """Copy outputs to static buffers"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and key in buffer_dict and buffer_dict[key] is not None:
                buffer_dict[key].copy_(value)

    def is_train_ready(self) -> bool:
        """Check if training graph is ready"""
        return self.train_graph_ready

    def is_eval_ready(self) -> bool:
        """Check if evaluation graph is ready"""
        return self.eval_graph_ready

    def reset_train_graph(self):
        """Reset training graph (useful for changing model architecture)"""
        self.train_graph = None
        self.train_stream = None
        self.train_static_inputs = {}
        self.train_static_outputs = {}
        self.train_graph_ready = False

    def reset_eval_graph(self):
        """Reset evaluation graph"""
        self.eval_graph = None
        self.eval_stream = None
        self.eval_static_inputs = {}
        self.eval_static_outputs = {}
        self.eval_graph_ready = False
