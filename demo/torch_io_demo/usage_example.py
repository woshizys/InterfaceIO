#!/usr/bin/env python3
"""
TensorIO使用示例
展示如何在实际项目中集成和使用自定义IO后端
"""

import os
import torch
import time
from pathlib import Path

# 导入自定义模块
from custom_dataset import CustomIODataset, PerformanceDataLoader, create_sample_dataset
from io_backend import get_tensor_io_backend

def example_1_basic_usage():
    """
    示例1: 基本使用方法
    """
    print("示例1: 基本使用方法")
    print("-" * 40)

    # 创建测试数据
    data_dir = "/tmp/tensor_io_example1"
    create_sample_dataset(data_dir, num_samples=20, sample_shape=(3, 32, 32))

    # 创建数据集
    dataset = CustomIODataset(
        data_dir=data_dir,
        dtype=torch.float32,
        io_backend='normal',  # 使用普通IO
        transform=None
    )

    print(f"数据集大小: {len(dataset)}")

    # 访问单个样本
    data, target = dataset[0]
    print(f"样本形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")


def example_2_dataloader_usage():
    """
    示例2: 使用DataLoader进行批量加载
    """
    print("\n示例2: DataLoader批量加载")
    print("-" * 40)

    # 使用之前创建的数据
    data_dir = "/tmp/tensor_io_example1"

    dataset = CustomIODataset(
        data_dir=data_dir,
        dtype=torch.float32,
        io_backend='io_uring',  # 使用io_uring
    )

    # 创建DataLoader
    dataloader = PerformanceDataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # 单线程，专注于IO性能
        enable_batch_loading=True
    )

    # 测量加载时间
    total_time = 0
    num_batches = 3

    for i, (batch_data, batch_targets) in enumerate(dataloader):
        if i >= num_batches:
            break

        start_time = time.time()
        # 模拟数据处理
        _ = batch_data.mean()
        end_time = time.time()

        batch_time = (end_time - start_time) * 1000
        total_time += batch_time

        print(f"批次 {i}: 形状={batch_data.shape}, 处理时间={batch_time:.2f}ms")

    avg_time = total_time / num_batches
    print(f"平均批次处理时间: {avg_time:.2f}ms")


def example_3_backend_comparison():
    """
    示例3: 不同后端性能对比
    """
    print("\n示例3: IO后端性能对比")
    print("-" * 40)

    # 创建较大的测试数据集
    data_dir = "/tmp/tensor_io_comparison"
    sample_shape = (3, 128, 128)  # 更大的图像
    num_samples = 50

    if not os.path.exists(data_dir):
        create_sample_dataset(data_dir, num_samples, sample_shape)

    backends = ['normal', 'io_uring']
    results = {}

    for backend_name in backends:
        print(f"\n测试 {backend_name.upper()} 后端:")

        try:
            dataset = CustomIODataset(
                data_dir=data_dir,
                dtype=torch.float32,
                io_backend=backend_name
            )

            dataloader = PerformanceDataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=0
            )

            # 测量完整的数据加载时间
            start_time = time.time()
            total_samples = 0

            for batch_data, _ in dataloader:
                total_samples += batch_data.size(0)
                # 确保数据被访问
                _ = batch_data.sum()

            end_time = time.time()

            total_time = end_time - start_time
            samples_per_sec = total_samples / total_time
            mb_per_sec = (total_samples * sample_shape[0] * sample_shape[1] * sample_shape[2] * 4) / (total_time * 1024 * 1024)

            results[backend_name] = {
                'total_time': total_time,
                'samples_per_sec': samples_per_sec,
                'mb_per_sec': mb_per_sec
            }

            print(f"  总时间: {total_time:.3f}s")
            print(f"  样本/秒: {samples_per_sec:.1f}")
            print(f"  吞吐量: {mb_per_sec:.1f} MB/s")

        except Exception as e:
            print(f"  错误: {e}")

    # 计算性能提升
    if 'normal' in results and 'io_uring' in results:
        normal_fps = results['normal']['samples_per_sec']
        uring_fps = results['io_uring']['samples_per_sec']
        speedup = uring_fps / normal_fps if normal_fps > 0 else 0

        print(f"\nio_uring性能提升: {speedup:.2f}x")


def example_4_custom_transforms():
    """
    示例4: 自定义数据变换
    """
    print("\n示例4: 自定义数据变换")
    print("-" * 40)

    import torchvision.transforms as transforms

    # 定义变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    data_dir = "/tmp/tensor_io_example1"

    dataset = CustomIODataset(
        data_dir=data_dir,
        dtype=torch.float32,
        io_backend='normal',
        transform=transform
    )

    # 获取变换后的样本
    original_data, _ = dataset[0]
    print(f"变换后数据形状: {original_data.shape}")
    print(f"变换后数据范围: [{original_data.min():.3f}, {original_data.max():.3f}]")


def example_5_cpp_backend_direct():
    """
    示例5: 直接使用C++后端
    """
    print("\n示例5: 直接使用C++后端接口")
    print("-" * 40)

    try:
        # 获取TensorIO后端
        backend = get_tensor_io_backend('io_uring')

        # 创建测试tensor
        test_tensor = torch.randn(100, 200, dtype=torch.float32)
        test_file = "/tmp/direct_backend_test.bin"

        print(f"原始tensor形状: {test_tensor.shape}")

        # 保存tensor
        success = backend.save_tensor_to_file(test_tensor, test_file)
        print(f"保存结果: {'成功' if success else '失败'}")

        # 加载tensor
        loaded_tensor = backend.load_tensor_from_file(test_file, [100, 200], torch.float32)
        print(f"加载tensor形状: {loaded_tensor.shape}")

        # 验证数据一致性
        data_equal = torch.allclose(test_tensor, loaded_tensor, atol=1e-6)
        print(f"数据一致性: {'通过' if data_equal else '失败'}")

        # 性能测试
        print("\n运行性能基准测试...")
        backend.benchmark_performance("/tmp/backend_benchmark.bin", 1024*1024, 5)

        # 清理
        if os.path.exists(test_file):
            os.remove(test_file)

    except Exception as e:
        print(f"C++后端不可用: {e}")
        print("使用Python fallback实现")


def example_6_memory_mapping():
    """
    示例6: 内存映射和大文件处理
    """
    print("\n示例6: 大文件和内存效率")
    print("-" * 40)

    # 创建大数据集
    large_data_dir = "/tmp/tensor_io_large"
    large_shape = (3, 256, 256)  # 较大的图像
    num_samples = 10  # 较少样本但更大

    if not os.path.exists(large_data_dir):
        print("创建大数据集...")
        create_sample_dataset(large_data_dir, num_samples, large_shape)

    dataset = CustomIODataset(
        data_dir=large_data_dir,
        dtype=torch.float32,
        io_backend='io_uring'
    )

    # 测量内存使用（简化版）
    import psutil
    process = psutil.Process()

    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # 加载几个大样本
    loaded_data = []
    for i in range(min(5, len(dataset))):
        data, _ = dataset[i]
        loaded_data.append(data)

    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before

    print(f"加载前内存: {memory_before:.1f} MB")
    print(f"加载后内存: {memory_after:.1f} MB")
    print(f"内存增量: {memory_used:.1f} MB")

    total_data_size = sum(data.numel() * 4 for data in loaded_data) / 1024 / 1024  # MB
    print(f"理论数据大小: {total_data_size:.1f} MB")
    print(f"内存效率: {(total_data_size / memory_used * 100) if memory_used > 0 else 0:.1f}%")


def cleanup_examples():
    """
    清理示例产生的临时文件
    """
    print("\n清理临时文件...")

    temp_dirs = [
        "/tmp/tensor_io_example1",
        "/tmp/tensor_io_comparison",
        "/tmp/tensor_io_large"
    ]

    temp_files = [
        "/tmp/direct_backend_test.bin",
        "/tmp/backend_benchmark.bin"
    ]

    import shutil

    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  删除目录: {dir_path}")

    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  删除文件: {file_path}")


def main():
    """
    运行所有示例
    """
    print("TensorIO使用示例集合")
    print("=" * 50)

    try:
        example_1_basic_usage()
        example_2_dataloader_usage()
        example_3_backend_comparison()

        # 可选示例（需要额外依赖）
        try:
            example_4_custom_transforms()
        except ImportError as e:
            print(f"\n示例4跳过: {e}")

        example_5_cpp_backend_direct()
        example_6_memory_mapping()

        print(f"\n{'='*50}")
        print("所有示例运行完成!")

        # 询问是否清理临时文件
        try:
            response = input("\n是否清理临时文件? (y/N): ").strip().lower()
            if response == 'y':
                cleanup_examples()
        except (EOFError, KeyboardInterrupt):
            print("\n跳过文件清理")

    except KeyboardInterrupt:
        print("\n\n示例运行被用户中断")
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
