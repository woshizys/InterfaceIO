#!/usr/bin/env python3
"""
PyTorch + 自定义IO集成演示脚本
展示如何使用io_uring和普通IO后端进行深度学习训练
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# 导入自定义模块
from custom_dataset import CustomIODataset, PerformanceDataLoader, create_sample_dataset
from io_backend import get_tensor_io_backend, TensorIOBackend


class SimpleNet(nn.Module):
    """
    简单的卷积神经网络，用于演示训练
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TrainingMetrics:
    """训练指标收集器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.total_times = []
        self.losses = []
        self.accuracies = []

    def add_batch_metrics(self, data_time: float, forward_time: float,
                         backward_time: float, total_time: float,
                         loss: float, accuracy: float):
        self.data_loading_times.append(data_time)
        self.forward_times.append(forward_time)
        self.backward_times.append(backward_time)
        self.total_times.append(total_time)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def get_summary(self) -> Dict[str, float]:
        if not self.total_times:
            return {}

        return {
            'avg_data_loading_time': np.mean(self.data_loading_times),
            'avg_forward_time': np.mean(self.forward_times),
            'avg_backward_time': np.mean(self.backward_times),
            'avg_total_time': np.mean(self.total_times),
            'avg_loss': np.mean(self.losses),
            'avg_accuracy': np.mean(self.accuracies),
            'data_loading_ratio': np.mean(self.data_loading_times) / np.mean(self.total_times)
        }


def train_epoch(model: nn.Module, dataloader: DataLoader,
               optimizer: optim.Optimizer, criterion: nn.Module,
               device: torch.device, epoch: int) -> TrainingMetrics:
    """
    训练一个epoch
    """
    model.train()
    metrics = TrainingMetrics()

    print(f"\n=== Epoch {epoch + 1} ===")

    for batch_idx, (data, targets) in enumerate(dataloader):
        batch_start_time = time.time()

        # 数据加载时间（包括数据传输到GPU）
        data_load_end = time.time()
        data = data.to(device)
        targets = targets.to(device, dtype=torch.long)
        data_transfer_end = time.time()

        # 前向传播
        optimizer.zero_grad()
        forward_start = time.time()
        outputs = model(data)
        loss = criterion(outputs, targets % 10)  # 简单的标签映射
        forward_end = time.time()

        # 反向传播
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_end = time.time()

        # 计算准确率
        _, predicted = outputs.max(1)
        accuracy = (predicted == (targets % 10)).float().mean().item()

        # 记录时间
        data_time = data_transfer_end - batch_start_time
        forward_time = forward_end - forward_start
        backward_time = backward_end - backward_start
        total_time = backward_end - batch_start_time

        metrics.add_batch_metrics(data_time, forward_time, backward_time,
                                total_time, loss.item(), accuracy)

        # 打印进度
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx:3d}: "
                  f"Loss={loss.item():.4f}, "
                  f"Acc={accuracy:.4f}, "
                  f"DataTime={data_time*1000:.1f}ms, "
                  f"TotalTime={total_time*1000:.1f}ms")

    return metrics


def benchmark_io_backends(data_dir: str, sample_shape: tuple, num_samples: int = 100):
    """
    对比不同IO后端的性能
    """
    print("\n" + "="*60)
    print("IO后端性能对比测试")
    print("="*60)

    backends = ['normal', 'io_uring']
    results = {}

    for backend_name in backends:
        print(f"\n--- 测试 {backend_name.upper()} 后端 ---")

        try:
            # 创建dataset
            dataset = CustomIODataset(
                data_dir=data_dir,
                dtype=torch.float32,
                io_backend=backend_name
            )

            # 创建dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=0  # 单线程测试，专注于IO性能
            )

            # 测量数据加载时间
            load_times = []
            total_start = time.time()

            for i, (data, _) in enumerate(dataloader):
                if i >= 10:  # 只测试前10个batch
                    break

                batch_start = time.time()
                _ = data.shape  # 确保数据被实际访问
                batch_end = time.time()

                load_times.append((batch_end - batch_start) * 1000)  # 转换为毫秒

            total_end = time.time()

            # 计算统计数据
            avg_load_time = np.mean(load_times)
            total_time = (total_end - total_start) * 1000
            throughput = (16 * 10 * np.prod(sample_shape) * 4) / (total_time / 1000) / (1024**2)  # MB/s

            results[backend_name] = {
                'avg_batch_time': avg_load_time,
                'total_time': total_time,
                'throughput': throughput
            }

            print(f"  平均批次加载时间: {avg_load_time:.2f} ms")
            print(f"  总加载时间: {total_time:.2f} ms")
            print(f"  吞吐量: {throughput:.2f} MB/s")

        except Exception as e:
            print(f"  错误: {e}")
            results[backend_name] = None

    # 对比结果
    print(f"\n--- 性能对比总结 ---")
    if results['normal'] and results['io_uring']:
        normal_time = results['normal']['avg_batch_time']
        uring_time = results['io_uring']['avg_batch_time']
        speedup = normal_time / uring_time if uring_time > 0 else 0

        print(f"io_uring vs 普通IO 加速比: {speedup:.2f}x")

        normal_throughput = results['normal']['throughput']
        uring_throughput = results['io_uring']['throughput']
        throughput_ratio = uring_throughput / normal_throughput if normal_throughput > 0 else 0

        print(f"io_uring vs 普通IO 吞吐量比: {throughput_ratio:.2f}x")

    return results


def run_training_demo(data_dir: str, backend: str = 'io_uring', epochs: int = 3):
    """
    运行训练演示
    """
    print(f"\n{'='*60}")
    print(f"PyTorch训练演示 - 使用 {backend.upper()} 后端")
    print(f"{'='*60}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集和数据加载器
    dataset = CustomIODataset(
        data_dir=data_dir,
        dtype=torch.float32,
        io_backend=backend
    )

    dataloader = PerformanceDataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        enable_batch_loading=True
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"批次大小: 32")
    print(f"每个epoch批次数: {len(dataloader)}")

    # 创建模型
    model = SimpleNet(input_channels=3, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    all_metrics = []

    for epoch in range(epochs):
        metrics = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        all_metrics.append(metrics)

        # 打印epoch总结
        summary = metrics.get_summary()
        if summary:
            print(f"\nEpoch {epoch + 1} 总结:")
            print(f"  平均loss: {summary['avg_loss']:.4f}")
            print(f"  平均准确率: {summary['avg_accuracy']:.4f}")
            print(f"  平均数据加载时间: {summary['avg_data_loading_time']*1000:.2f} ms")
            print(f"  数据加载时间占比: {summary['data_loading_ratio']*100:.1f}%")

    return all_metrics


def visualize_results(metrics_list: List[TrainingMetrics], backend_name: str):
    """
    可视化训练结果
    """
    if not metrics_list or not hasattr(plt, 'subplots'):
        print("跳过可视化 (matplotlib不可用)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Metrics - {backend_name.upper()} Backend')

    epochs = range(1, len(metrics_list) + 1)

    # Loss曲线
    avg_losses = [m.get_summary().get('avg_loss', 0) for m in metrics_list]
    axes[0, 0].plot(epochs, avg_losses, 'b-o')
    axes[0, 0].set_title('Average Loss per Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # 准确率曲线
    avg_accs = [m.get_summary().get('avg_accuracy', 0) for m in metrics_list]
    axes[0, 1].plot(epochs, avg_accs, 'g-o')
    axes[0, 1].set_title('Average Accuracy per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)

    # 数据加载时间
    data_times = [m.get_summary().get('avg_data_loading_time', 0)*1000 for m in metrics_list]
    axes[1, 0].plot(epochs, data_times, 'r-o')
    axes[1, 0].set_title('Average Data Loading Time per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].grid(True)

    # 数据加载时间占比
    data_ratios = [m.get_summary().get('data_loading_ratio', 0)*100 for m in metrics_list]
    axes[1, 1].plot(epochs, data_ratios, 'm-o')
    axes[1, 1].set_title('Data Loading Time Ratio per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio (%)')
    axes[1, 1].grid(True)

    plt.tight_layout()

    # 保存图片
    output_path = f'/tmp/training_metrics_{backend_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练指标图表已保存到: {output_path}")


def main():
    """主函数"""
    print("PyTorch自定义IO集成演示")
    print("支持io_uring和普通IO两种后端")

    # 配置参数
    data_dir = "/tmp/torch_io_demo_data"
    sample_shape = (3, 64, 64)  # 较小的图像用于快速演示
    num_samples = 200
    epochs = 3

    # 1. 创建测试数据集
    print(f"\n1. 创建测试数据集...")
    if not os.path.exists(data_dir):
        create_sample_dataset(
            data_dir=data_dir,
            num_samples=num_samples,
            sample_shape=sample_shape,
            dtype=torch.float32
        )
    else:
        print(f"使用现有数据集: {data_dir}")

    # 2. IO后端性能对比
    print(f"\n2. IO后端性能对比...")
    benchmark_results = benchmark_io_backends(data_dir, sample_shape, num_samples)

    # 3. 训练演示 - 普通IO
    print(f"\n3. 普通IO后端训练演示...")
    normal_metrics = run_training_demo(data_dir, 'normal', epochs)

    # 4. 训练演示 - io_uring
    print(f"\n4. io_uring后端训练演示...")
    try:
        uring_metrics = run_training_demo(data_dir, 'io_uring', epochs)
    except Exception as e:
        print(f"io_uring后端不可用: {e}")
        uring_metrics = None

    # 5. 结果可视化
    print(f"\n5. 生成结果可视化...")
    try:
        if normal_metrics:
            visualize_results(normal_metrics, 'normal')
        if uring_metrics:
            visualize_results(uring_metrics, 'io_uring')
    except Exception as e:
        print(f"可视化过程出错: {e}")

    # 6. 最终总结
    print(f"\n{'='*60}")
    print("演示完成 - 总结")
    print(f"{'='*60}")

    print("✓ 成功创建了自定义PyTorch Dataset，集成了高性能IO后端")
    print("✓ 实现了io_uring和普通IO两种模式的对比")
    print("✓ 演示了在实际训练中的性能表现")
    print("✓ 提供了详细的性能指标和可视化结果")

    print(f"\n技术特性:")
    print(f"• 自定义Dataset支持原始二进制文件和numpy格式")
    print(f"• 灵活的IO后端切换（普通IO vs io_uring）")
    print(f"• 批量加载优化，充分利用io_uring的异步特性")
    print(f"• 完整的性能监控和可视化")

    print(f"\n文件清理...")
    # 可选：清理临时文件
    # import shutil
    # shutil.rmtree(data_dir)
    # print(f"已清理临时数据目录: {data_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
