# Refactored Trainer Architecture

## 概述

这是一个完全重构的训练器架构，旨在提供更好的模块化、可扩展性和灵活性。主要改进包括：

1. **数据适配器系统** - 灵活处理不同类型的输入数据
2. **可插拔评估器** - 自动检测并应用合适的评估指标
3. **CUDA Graph管理器** - 数据无关的图优化
4. **配置驱动的评估流程** - 易于扩展新的评估指标

## 架构组件

### 1. Data Adapter (数据适配器)

**文件**: `data_adapter.py`

数据适配器负责将原始数据批次转换为模型可以使用的格式。
如此可以在不重构原始数据dataset的情况下支持新增数据类型
优化了数据准备逻辑的复用。

#### 核心类

- `DataAdapter` - 抽象基类
- `HandPoseDataAdapter` - 2D手部姿态估计数据适配器
- `Hand3DDataAdapter` - 3D手部重建数据适配器
- `DataAdapterRegistry` - 适配器注册表

#### 使用示例

```python
from src.main.trainer_components import DataAdapterRegistry

# 获取适配器
adapter = DataAdapterRegistry.get_adapter('hand_3d', device, conf)

# 准备数据批次
prepared_data = adapter.prepare_batch(raw_data)

# 提取ground truth
ground_truth = adapter.get_ground_truth(raw_data)
```

#### 添加自定义适配器

```python
from src.main.trainer_components.data_adapter import DataAdapter, DataAdapterRegistry

class MyCustomAdapter(DataAdapter):
    def get_required_keys(self):
        return ['image', 'label']

    def get_optional_keys(self):
        return ['metadata']

    def prepare_batch(self, data):
        prepared = {}
        prepared['image'] = data['image'].to(self.device)
        # ... 处理其他数据
        return prepared

    def get_ground_truth(self, data):
        return {'label': data['label'].to(self.device)}

# 注册适配器
DataAdapterRegistry.register('my_custom', MyCustomAdapter)
```

### 2. Evaluator (评估器)

**文件**: `evaluator.py`

评估器是可插拔的模块，用于计算不同的评估指标。

#### 核心类

- `BaseEvaluator` - 抽象基类
- `Keypoint2DEvaluator` - 2D关键点评估器 (PCK, AUC, accuracy)
- `Keypoint3DEvaluator` - 3D关键点评估器 (MPJPE, PA-MPJPE)
- `VertexEvaluator` - Mesh顶点评估器 (MPVPE, F-scores)
- `SegmentationEvaluator` - 分割评估器 (IoU)
- `LossEvaluator` - 损失评估器
- `EvaluatorRegistry` - 评估器注册表

#### 使用示例

```python
from src.main.trainer_components import EvaluatorRegistry

# 获取单个评估器
evaluator = EvaluatorRegistry.get_evaluator('keypoint_2d', conf)

# 重置
evaluator.reset()

# 更新指标
for batch in dataloader:
    predictions = model(batch)
    ground_truth = extract_gt(batch)
    evaluator.update(predictions, ground_truth)

# 计算最终指标
metrics = evaluator.compute()
print(metrics)  # {'pck_0.5': 0.95, 'auc_2d': 0.92, ...}
```

#### 添加自定义评估器

```python
from src.main.trainer_components.evaluator import BaseEvaluator, EvaluatorRegistry

class MyCustomEvaluator(BaseEvaluator):
    def reset(self):
        self.total = 0
        self.correct = 0

    def update(self, predictions, ground_truth):
        # 更新内部状态
        self.total += predictions['output'].size(0)
        self.correct += (predictions['output'].argmax(1) == ground_truth['label']).sum().item()

    def compute(self):
        return {'accuracy': self.correct / self.total if self.total > 0 else 0}

    def get_metric_names(self):
        return ['accuracy']

    def is_applicable(self, predictions, ground_truth):
        return 'output' in predictions and 'label' in ground_truth

# 注册评估器
EvaluatorRegistry.register('my_custom', MyCustomEvaluator)
```

### 3. CUDA Graph Manager (CUDA图管理器)

**文件**: `cuda_graph_manager.py`

CUDA Graph管理器提供数据无关的CUDA Graph初始化和重放功能。

#### 核心功能

- 训练图初始化和重放
- 评估图初始化和重放
- 自动预热和图捕获
- 静态缓冲区管理

#### 使用示例

```python
from src.main.trainer_components import CUDAGraphManager

# 创建管理器
cuda_graph_mgr = CUDAGraphManager(model, device, conf)

# 初始化训练图
sample_data = next(iter(train_loader))
prepared_data = adapter.prepare_batch(sample_data)

def forward_fn(static_inputs):
    return model(static_inputs['rgb'], static_inputs['mask'], ...)

cuda_graph_mgr.init_train_graph(
    prepared_data,
    optimizer,
    forward_fn,
    warmup_steps=5
)

# 重放训练图
for batch in train_loader:
    prepared = adapter.prepare_batch(batch)
    outputs = cuda_graph_mgr.replay_train_graph(prepared)
```

### 4. Evaluation Pipeline (评估流程)

**文件**: `evaluation_pipeline.py`

评估流程提供配置驱动的评估系统，自动应用合适的评估器。

#### 核心类

- `EvaluationPipeline` - 评估流程管理器
- `VisualizationHelper` - 可视化辅助工具

#### 使用示例

```python
from src.main.trainer_components.evaluation_pipeline import EvaluationPipeline

# 创建评估流程
eval_pipeline = EvaluationPipeline(
    conf,
    data_adapter,
    evaluators=['keypoint_2d', 'keypoint_3d', 'segmentation']  # 可选，None表示使用所有
)

# 评估循环
eval_pipeline.reset()
for batch in eval_loader:
    predictions = model(batch)
    ground_truth = adapter.get_ground_truth(batch)
    eval_pipeline.update(predictions, ground_truth)

# 计算并格式化结果
metrics = eval_pipeline.compute()
result_str = eval_pipeline.format_results(metrics)
print(result_str)
```

## 使用重构的Trainer

### 基本使用

```python
from src.main.trainer_refactored import RefactoredTrainer

# 创建配置
conf = {
    'mode': 'train',
    'device': torch.device('cuda'),
    'model_path': 'src.model.SPNet_liteUp',
    'model_name': 'SPNet',
    'data_adapter': 'hand_3d',  # 可选，自动检测
    'evaluators': None,  # None表示使用所有评估器
    'use_cuda_graph': True,
    # ... 其他配置
}

# 创建trainer
trainer = RefactoredTrainer(conf)

# 训练
if conf['mode'] == 'train':
    trainer.train()
# 评估
elif conf['mode'] == 'evaluate':
    metrics = trainer.evaluate(epoch=0)
```

### 配置选项

#### 数据适配器配置

```python
conf = {
    # 自动选择适配器（基于task_name）
    'task_name': 'Mobrecon_freihand',  # 包含'Mobrecon'或'3d'会使用hand_3d

    # 或手动指定
    'data_adapter': 'hand_3d',  # 'hand_pose_2d' 或 'hand_3d'
}
```

#### 评估器配置

```python
conf = {
    # 使用所有评估器（默认）
    'evaluators': None,

    # 或指定特定评估器
    'evaluators': ['keypoint_2d', 'keypoint_3d', 'vertex'],
}
```

#### CUDA Graph配置

```python
conf = {
    # 启用CUDA Graph（默认）
    'use_cuda_graph': True,

    # 禁用CUDA Graph
    'use_cuda_graph': False,
}
```

## 扩展指南

### 添加新的数据类型支持

1. 创建新的DataAdapter子类
2. 实现必需的方法
3. 注册到DataAdapterRegistry

```python
class VideoDataAdapter(DataAdapter):
    def get_required_keys(self):
        return ['video_frames', 'labels']

    def prepare_batch(self, data):
        # 处理视频数据
        pass

    def get_ground_truth(self, data):
        # 提取标签
        pass

DataAdapterRegistry.register('video', VideoDataAdapter)
```

### 添加新的评估指标

1. 创建新的BaseEvaluator子类
2. 实现评估逻辑
3. 注册到EvaluatorRegistry

```python
class ObjectDetectionEvaluator(BaseEvaluator):
    def reset(self):
        self.predictions = []
        self.ground_truths = []

    def update(self, predictions, ground_truth):
        self.predictions.append(predictions['boxes'])
        self.ground_truths.append(ground_truth['boxes'])

    def compute(self):
        # 计算mAP等指标
        return {'mAP': calculate_map(self.predictions, self.ground_truths)}

    def get_metric_names(self):
        return ['mAP', 'mAP_50', 'mAP_75']

    def is_applicable(self, predictions, ground_truth):
        return 'boxes' in predictions and 'boxes' in ground_truth

EvaluatorRegistry.register('object_detection', ObjectDetectionEvaluator)
```

## 与原始Trainer的对比

### 原始Trainer的问题

1. **硬编码的数据处理** - 数据准备逻辑分散在训练和评估代码中
2. **耦合的评估逻辑** - 所有评估指标混在一起，难以维护
3. **重复的CUDA Graph代码** - 训练和评估的图初始化代码重复
4. **难以扩展** - 添加新的数据类型或评估指标需要修改核心代码

### 重构后的优势

1. **模块化** - 每个组件职责清晰，易于理解和维护
2. **可扩展** - 通过注册表模式轻松添加新功能
3. **可配置** - 通过配置文件控制行为，无需修改代码
4. **可重用** - 组件可以在不同项目中重用
5. **易测试** - 每个组件可以独立测试

## 迁移指南

### 从原始Trainer迁移

1. **更新导入**:
```python
# 旧代码
from src.main.trainer_traverse_cudagraph_v4 import Trainer

# 新代码
from src.main.trainer_refactored import RefactoredTrainer as Trainer
```

2. **配置保持不变** - 大部分配置项兼容

3. **可选：指定适配器和评估器**:
```python
conf['data_adapter'] = 'hand_3d'
conf['evaluators'] = ['keypoint_2d', 'keypoint_3d', 'vertex']
```

### 向后兼容

重构的trainer提供了向后兼容的别名：

```python
# 这两种方式等价
from src.main.trainer_refactored import RefactoredTrainer
from src.main.trainer_refactored import Trainer  # 别名
```

## 性能考虑

### CUDA Graph

- **启用时机**: 适用于固定输入形状的场景
- **预热开销**: 首次初始化需要额外时间（训练5次，评估3次）
- **加速效果**: 通常可获得10-30%的加速

### 内存使用

- 每个评估器维护自己的状态
- CUDA Graph需要额外的静态缓冲区
- 建议在GPU内存充足时使用

## 故障排除

### 常见问题

1. **"Unknown adapter" 错误**
   - 检查配置中的`data_adapter`值
   - 使用`DataAdapterRegistry.list_adapters()`查看可用适配器

2. **"Unknown evaluator" 错误**
   - 检查配置中的`evaluators`列表
   - 使用`EvaluatorRegistry.list_evaluators()`查看可用评估器

3. **CUDA Graph初始化失败**
   - 确保输入形状固定
   - 检查模型是否支持CUDA Graph（避免动态控制流）
   - 尝试禁用CUDA Graph: `conf['use_cuda_graph'] = False`

4. **评估器没有激活**
   - 检查模型输出和ground truth是否包含必需的键
   - 使用`evaluator.is_applicable()`测试

## 示例

完整的使用示例请参考：
- [训练示例](../examples/train_example.py)
- [评估示例](../examples/eval_example.py)
- [自定义组件示例](../examples/custom_components.py)

## 贡献

欢迎贡献新的适配器和评估器！请遵循以下步骤：

1. 创建新的组件类
2. 添加单元测试
3. 更新文档
4. 提交Pull Request

## 许可证

与主项目相同
