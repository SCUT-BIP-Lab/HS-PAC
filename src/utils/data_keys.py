"""
数据键常量定义 - 统一管理所有字典键名！！！

这个模块集中定义了整个框架中使用的所有字典键名，包括：
- 模型输入数据的键
- 模型输出/预测的键
- Ground Truth数据的键
- 评估指标的键
- 损失函数的键

使用常量类的好处：
1. 集中管理，便于查看所有可用的键
2. IDE自动补全，减少拼写错误
3. 重构时只需修改一处
4. 便于文档化和维护
"""


class DataKeys:
    """数据加载器输出的数据字典键（输入到模型的数据）"""

    # ========== 图像数据 ==========
    IMG = 'rgb'                     # RGB图像 [B, 3, H, W]
    RGB = 'rgb'#rgb                     # RGB图像（另一种命名） [B, 3, H, W]

    # ========== 2D关键点相关 ==========
    JOINT_IMG = 'joint_img'         # 2D关键点坐标 [B, 21, 2] 或 [B, 21, 3]（含置信度）
    KPT_HEATMAP = 'kpt_heatmap'     # 2D关键点热图 [B, 21, H, W]

    # ========== 多尺度热图相关 ==========
    KPT_HEATMAP_FULL = 'kpt_heatmap_full'     # 21个关键点热图 [B, 21, H, W]
    FINGER_HEATMAP = 'finger_heatmap'         # 5个手指热图 [B, 5, H, W]
    HAND_HEATMAP = 'hand_heatmap'             # 整手热图 [B, 1, H, W]

    # ========== 3D关键点相关 ==========
    JOINT_CAM = 'joint_cam'         # 3D关键点坐标（相机坐标系） [B, 21, 3]
    ROOT = 'root'                   # 根节点坐标 [B, 3]

    # ========== Mesh相关 ==========
    VERTS = 'verts'                 # Mesh顶点坐标 [B, 778, 3]
    FACE = 'face'                   # Mesh面片索引 [F, 3]

    # ========== 分割掩码 ==========
    MASK = 'mask'                   # 分割掩码 [B, 1, H, W] 或 [B, H, W]
    MASK_GT = 'mask'             # 分割掩码Ground Truth（mobrecon命名）

    # ========== 相机参数 ==========
    CALIB = 'calib'                 # 相机标定矩阵 [B, 4, 4]

    # ========== 数据增强参数 ==========
    AUG_PARAM = 'aug_param'         # 数据增强参数（用于对比学习）
    BB2IMG_TRANS = 'bb2img_trans'   # Bounding box到图像的变换矩阵

    # ========== MANO参数（如果使用MANO模型） ==========
    MANO_POSE = 'mano_pose'         # MANO姿态参数
    MANO_SHAPE = 'mano_shape'       # MANO形状参数


class PredKeys:
    """模型预测输出的字典键"""

    # ========== 2D关键点预测 ==========
    SKELETON = 'skeleton'           # 2D关键点热图预测 [B, 21, H, W]
    JOINT_IMG = 'joint_img'         # 2D关键点坐标预测 [B, 21, 2]

    # ========== 多尺度热图预测 ==========
    KPT_HEATMAP_PRED = 'kpt_heatmap_pred'         # 21个关键点热图预测 [B, 21, H, W]
    FINGER_HEATMAP_PRED = 'finger_heatmap_pred'   # 5个手指热图预测 [B, 5, H, W]
    HAND_HEATMAP_PRED = 'hand_heatmap_pred'       # 整手热图预测 [B, 1, H, W]

    # ========== 3D关键点预测 ==========
    JOINT_CAM = 'joint_cam'         # 3D关键点坐标预测 [B, 21, 3]

    # ========== Mesh预测 ==========
    VERTS = 'verts'                 # Mesh顶点预测 [B, 778, 3]
    COARSE_VERTS = 'coarse_verts'   # 粗略Mesh顶点预测 [B, 778, 3]
    RESIDUAL_VERTS = 'residual_verts'  # 残差Mesh顶点预测 [B, 778, 3]

    # ========== 分割预测 ==========
    SEGMENT = 'segment'             # 分割掩码预测 [B, 1, H, W]
    MASK = 'mask'                   # 分割掩码预测（mobrecon命名）

    # ========== 其他预测 ==========
    TRANS = 'trans'                 # 平移参数预测
    ALPHA = 'alpha'                 # Alpha参数预测

    # ========== 损失值 ==========
    LOSS = 'loss'                   # 总损失


class LossKeys:
    """损失函数相关的字典键"""

    # ========== 总损失 ==========
    LOSS = 'loss'                   # 总损失

    # ========== 2D损失 ==========
    JOINT_IMG_LOSS = 'joint_img_loss'       # 2D关键点损失
    SKELETON_LOSS = 'skeleton_loss'         # 2D关键点热图损失

    # ========== 多尺度热图损失 ==========
    KPT_HEATMAP_LOSS = 'kpt_heatmap_loss'         # 21个关键点热图损失
    FINGER_HEATMAP_LOSS = 'finger_heatmap_loss'   # 5个手指热图损失
    HAND_HEATMAP_LOSS = 'hand_heatmap_loss'       # 整手热图损失

    # ========== 3D损失 ==========
    JOINT_CAM_LOSS = 'joint_cam_loss'       # 3D关键点损失
    VERTS_LOSS = 'verts_loss'               # Mesh顶点损失
    COARSE_VERTS_LOSS = 'coarse_verts_loss'   # 粗略Mesh顶点损失
    RESIDUAL_VERTS_LOSS = 'residual_verts_loss'  # 残差Mesh顶点损失

    # ========== Mesh几何损失 ==========
    NORMAL_LOSS = 'normal_loss'             # 法向量损失
    EDGE_LOSS = 'edge_loss'                 # 边长损失
    HAND_PART_LOSS = 'hand_part_loss'       # 手部各部分相对参考节点损失

    # ========== 分割损失 ==========
    MASK_LOSS = 'mask_loss'                 # 分割掩码损失
    SEGMENT_LOSS = 'segment_loss'           # 分割损失

    # ========== 对比学习损失 ==========
    CON3D_LOSS = 'con3d_loss'               # 3D对比学习损失
    CON2D_LOSS = 'con2d_loss'               # 2D对比学习损失

    # ========== 部位投影mask损失 ==========
    PART_MASK_LOSS = 'part_mask_loss'       # 手部部位投影mask损失

    # ========== 其他损失 ==========
    L1_LOSS = 'l1_loss'                     # L1损失


class MetricKeys:
    """评估指标的字典键"""

    # ========== 2D关键点指标 ==========
    PCK_05 = 'pck_0.5'                      # PCK@0.5
    PCK_5PX_IMG = 'pck@5_img'              # PCK@0.5（直接坐标版本）

    PCK_10PX_IMG='pck@10_img'
    ACC_10PX = 'acc_10px'                   # 10像素准确率
    ACC_10PX_IMG = 'acc_10px_img'           # 10像素准确率（直接坐标版本）
    UVE = 'uve'                             # UV误差（2D关键点平均误差）
    UVE_IMG = 'uve_img'                     # UV误差（直接坐标版本）
    AUC_2D = 'auc_2d'                       # 2D AUC
    AUC_2D_IMG = 'auc_2d_img'               # 2D AUC（直接坐标版本）
    PCK_CURVE = 'pck_curve'                 # PCK曲线
    PCK_CURVE_IMG = 'pck_curve_img'         # PCK曲线（直接坐标版本）

    # ========== 3D关键点指标 ==========
    MPJPE = 'mpjpe'                         # Mean Per Joint Position Error
    PA_MPJPE = 'pa_mpjpe'                   # Procrustes Aligned MPJPE
    AUC_3D_REL = 'auc_3d_rel'               # 3D AUC（相对坐标）
    AUC_3D_PA = 'auc_3d_pa'                 # 3D AUC（对齐后）

    # ========== Mesh顶点指标 ==========
    MPVPE = 'mpvpe'                         # Mean Per Vertex Position Error
    PA_MPVPE = 'pa_mpvpe'                   # Procrustes Aligned MPVPE
    F_SCORE_5MM = 'f_score_5mm'             # F-score @ 5mm
    F_SCORE_15MM = 'f_score_15mm'           # F-score @ 15mm
    PA_F_SCORE_5MM = 'pa_f_score_5mm'       # PA F-score @ 5mm
    PA_F_SCORE_15MM = 'pa_f_score_15mm'     # PA F-score @ 15mm

    # ========== 分割指标 ==========
    IOU = 'iou'                             # Intersection over Union
    MIOU = 'miou'                           # Mean IoU

    # ========== 损失指标 ==========
    LOSS = 'loss'                           # 损失值


class ConfigKeys:
    """配置文件中的键"""

    # ========== 模型配置 ==========
    MODEL_PATH = 'model_path'
    MODEL_NAME = 'model_name'
    MANO_PATH = 'mano_path'

    # ========== 训练配置 ==========
    MODE = 'mode'
    DEVICE = 'device'
    NUM_EPOCHS = 'num_epochs'
    LEARNING_RATE = 'learning_rate'
    OPTIMIZER = 'optimizer'
    BATCHSIZE_TRAIN = 'batchsize_train'
    BATCHSIZE_TEST = 'batchsize_test'

    # ========== 数据配置 ==========
    TASK_NAME = 'task_name'
    DATASET_TYPE = 'dataset_type'
    DATA_ADAPTER = 'data_adapter'

    # ========== 评估配置 ==========
    EVALUATORS = 'evaluators'
    EVAL_INTERNAL = 'eval_internal'

    # ========== CUDA Graph配置 ==========
    USE_CUDA_GRAPH = 'use_cuda_graph'

    # ========== 其他配置 ==========
    JOINT_IMG_SCALE_FACTOR = 'joint_img_scale_factor'


# ========== 便捷访问别名 ==========
# 为了向后兼容和便捷使用，提供一些常用的别名

# 输入数据键（包含GT和输入图像）
class InputKeys:
    """输入数据的所有键（包括图像、GT等）"""
    # 继承DataKeys的所有属性
    pass

# 将DataKeys的所有属性复制到InputKeys
for attr_name in dir(DataKeys):
    if not attr_name.startswith('_'):
        setattr(InputKeys, attr_name, getattr(DataKeys, attr_name))


# ========== 辅助函数 ==========

def get_all_data_keys():
    """获取所有数据键的列表"""
    return [getattr(DataKeys, attr) for attr in dir(DataKeys) if not attr.startswith('_')]


def get_all_pred_keys():
    """获取所有预测键的列表"""
    return [getattr(PredKeys, attr) for attr in dir(PredKeys) if not attr.startswith('_')]


def get_all_metric_keys():
    """获取所有指标键的列表"""
    return [getattr(MetricKeys, attr) for attr in dir(MetricKeys) if not attr.startswith('_')]


def get_all_loss_keys():
    """获取所有损失键的列表"""
    return [getattr(LossKeys, attr) for attr in dir(LossKeys) if not attr.startswith('_')]


def print_all_keys():
    """打印所有定义的键（用于调试和文档）"""
    print("=" * 60)
    print("数据键常量定义")
    print("=" * 60)

    print("\n【DataKeys - 输入数据键】")
    for attr in dir(DataKeys):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = '{getattr(DataKeys, attr)}'")

    print("\n【PredKeys - 预测输出键】")
    for attr in dir(PredKeys):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = '{getattr(PredKeys, attr)}'")

    print("\n【LossKeys - 损失函数键】")
    for attr in dir(LossKeys):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = '{getattr(LossKeys, attr)}'")

    print("\n【MetricKeys - 评估指标键】")
    for attr in dir(MetricKeys):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = '{getattr(MetricKeys, attr)}'")

    print("\n【ConfigKeys - 配置键】")
    for attr in dir(ConfigKeys):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = '{getattr(ConfigKeys, attr)}'")

    print("=" * 60)


if __name__ == '__main__':
    # 测试：打印所有键
    print_all_keys()