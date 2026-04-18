'''
@Project ：SPNet
@File    ：test.py
@IDE     ：PyCharm
@Author  ：WXL
@Date    ：2026/3/23 12:04
'''

import torch
import timm
import time
import gc
from typing import List, Dict

def get_gpu_memory():
    """获取当前GPU显存使用情况(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_model(model_name: str, batch_sizes: List[int], img_size: int = 224, num_warmup: int = 10, num_iter: int = 50):
    """测试模型在不同批量大小下的显存和推理速度"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}","input size:", img_size)
    print(f"{'='*80}")

    results = []

    # 清理显存并创建模型
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # 尝试使用img_size参数创建模型
        try:
            model = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        except TypeError:
            # 如果不支持img_size参数，则不使用
            model = timm.create_model(model_name, pretrained=False, num_classes=0)

        model = model.cuda().eval()
        model_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"Model memory: {model_memory:.2f}MB")
    except Exception as e:
        print(f"Failed to create model: {e}")
        return results

    for batch_size in batch_sizes:
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # 创建输入
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).cuda()

            # 预热
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy_input)

            torch.cuda.synchronize()

            # 记录显存
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

            # 测试推理速度
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_iter):
                    _ = model(dummy_input)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iter * 1000  # ms
            fps = batch_size / (avg_time / 1000)

            result = {
                'batch_size': batch_size,
                'memory_allocated_mb': f"{memory_allocated:.2f}",
                'memory_peak_mb': f"{memory_peak:.2f}",
                'avg_time_ms': f"{avg_time:.2f}",
                'fps': f"{fps:.2f}"
            }
            results.append(result)

            print(f"Batch: {batch_size:3d} | Memory: {memory_allocated:7.2f}MB (Peak: {memory_peak:7.2f}MB) | "
                  f"Time: {avg_time:6.2f}ms | FPS: {fps:6.2f}")

            # 清理输入
            del dummy_input

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch: {batch_size:3d} | OOM (Out of Memory)")
                results.append({'batch_size': batch_size, 'status': 'OOM'})
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise e

    # 清理模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    # 指定使用的GPU
    gpu_id = 0  # 修改这里来选择不同的GPU
    torch.cuda.set_device(gpu_id)

    # DINOv2和DINOv3模型列表
    models = [
        # DINOv2系列
        # 'vit_small_patch14_dinov2.lvd142m',
        # 'vit_base_patch14_dinov2.lvd142m',
        # 'vit_large_patch14_dinov2.lvd142m',
        # 'vit_giant_patch14_dinov2.lvd142m',
        # DINOv3系列
        'vit_small_plus_patch16_dinov3.lvd1689m',
        'vit_base_patch16_dinov3.lvd1689m',
        'vit_large_patch16_dinov3.lvd1689m',
        'vit_huge_plus_patch16_dinov3.lvd1689m',
        "fastvit_ma36"
    ]

    # 测试的批量大小
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # 图像尺寸
    img_size = 224

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024:.2f}MB")

    all_results = {}

    for model_name in models:
        results = test_model(model_name, batch_sizes, img_size)
        all_results[model_name] = results

    # # 打印汇总
    # print(f"\n{'='*80}")
    # print("Summary")
    # print(f"{'='*80}")
    # for model_name, results in all_results.items():
    #     print(f"\n{model_name}:")
    #     for r in results:
    #         print(f"  {r}")

