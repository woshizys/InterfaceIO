#!/usr/bin/env python3
"""
自定义PyTorch Dataset，集成高性能IO后端
支持io_uring和普通IO两种模式，提供灵活的数据加载方案
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json


class CustomIODataset(Dataset):
    """
    基于自定义IO后端的PyTorch Dataset
    支持从原始二进制文件或numpy格式加载tensor数据
    """

    def __init__(self,
                 data_dir: str,
                 file_list: Optional[List[str]] = None,
                 shape_info: Optional[Dict[str, List[int]]] = None,
                 dtype: torch.dtype = torch.float32,
                 io_backend: str = 'io_uring',
                 transform=None,
                 target_transform=None):
        """
        初始化数据集

        Args:
            data_dir: 数据文件目录
            file_list: 文件名列表，如果为None则自动扫描目录
            shape_info: 每个文件的tensor形状信息字典
            dtype: tensor数据类型
            io_backend: IO后端类型 ('io_uring' 或 'normal')
            transform: 数据变换函数
            target_transform: 标签变换函数
        """
        self.data_dir = data_dir
        self.dtype = dtype
        self.io_backend = io_backend
        self.transform = transform
        self.target_transform = target_transform

        # 如果没有提供文件列表，自动扫描目录
        if file_list is None:
            self.file_list = self._scan_data_files()
        else:
            self.file_list = file_list

        # 如果没有提供形状信息，尝试从同名json文件加载
        if shape_info is None:
            self.shape_info = self._load_shape_info()
        else:
            self.shape_info = shape_info

        # 验证数据
        self._validate_dataset()

        print(f"CustomIODataset initialized:")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Number of files: {len(self.file_list)}")
        print(f"  - IO backend: {io_backend}")
        print(f"  - Data type: {dtype}")

    def _scan_data_files(self) -> List[str]:
        """自动扫描数据目录中的数据文件"""
        data_files = []
        supported_extensions = ['.bin', '.dat', '.npy', '.tensor']

        for filename in os.listdir(self.data_dir):
            if any(filename.endswith(ext) for ext in supported_extensions):
                data_files.append(filename)

        data_files.sort()  # 确保顺序一致
        return data_files

    def _load_shape_info(self) -> Dict[str, List[int]]:
        """加载形状信息，优先从shape_info.json加载"""
        shape_info = {}
        shape_file = os.path.join(self.data_dir, 'shape_info.json')

        if os.path.exists(shape_file):
            with open(shape_file, 'r') as f:
                shape_info = json.load(f)
            print(f"Loaded shape info from {shape_file}")
        else:
            # 如果没有shape_info.json，尝试从numpy文件推断
            for filename in self.file_list:
                if filename.endswith('.npy'):
                    filepath = os.path.join(self.data_dir, filename)
                    arr = np.load(filepath, mmap_mode='r')  # 只读取头部信息
                    shape_info[filename] = list(arr.shape)

            if shape_info:
                print("Inferred shape info from numpy files")
            else:
                print("Warning: No shape information available. Using default shape inference.")

        return shape_info

    def _validate_dataset(self):
        """验证数据集的完整性"""
        missing_shapes = []
        for filename in self.file_list:
            if filename not in self.shape_info:
                missing_shapes.append(filename)

        if missing_shapes:
            print(f"Warning: Missing shape info for files: {missing_shapes}")
            print("These files will use dynamic shape inference during loading.")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        获取单个数据样本

        Args:
            idx: 样本索引

        Returns:
            (data_tensor, target): 数据tensor和对应的标签
        """
        filename = self.file_list[idx]
        filepath = os.path.join(self.data_dir, filename)

        # 加载数据
        if filename.endswith('.npy'):
            data = self._load_numpy_file(filepath)
        else:
            data = self._load_raw_file(filepath, filename)

        # 应用变换
        if self.transform:
            data = self.transform(data)

        # 生成或获取标签（这里简化为使用文件索引作为标签）
        target = idx  # 实际应用中可能从文件名或单独的标签文件获取

        if self.target_transform:
            target = self.target_transform(target)

        return data, target

    def _load_numpy_file(self, filepath: str) -> torch.Tensor:
        """从numpy文件加载数据"""
        arr = np.load(filepath)
        return torch.from_numpy(arr).to(self.dtype)

    def _load_raw_file(self, filepath: str, filename: str) -> torch.Tensor:
        """
        从原始二进制文件加载数据
        这里展示了如何集成C++的TensorIO实现
        """
        # 获取形状信息
        if filename in self.shape_info:
            shape = self.shape_info[filename]
        else:
            # 动态推断形状（这里需要额外的逻辑）
            file_size = os.path.getsize(filepath)
            element_size = self._get_element_size()
            total_elements = file_size // element_size
            shape = [total_elements]  # 简化为一维数组
            print(f"Inferred shape for {filename}: {shape}")

        # 使用普通Python文件读取（作为示例）
        # 在实际应用中，这里会调用C++的TensorIO
        with open(filepath, 'rb') as f:
            data = f.read()

        # 转换为numpy数组，再转为tensor
        np_array = np.frombuffer(data, dtype=self._torch_to_numpy_dtype())
        np_array = np_array.reshape(shape)

        return torch.from_numpy(np_array).to(self.dtype)

    def _get_element_size(self) -> int:
        """获取数据类型的字节大小"""
        dtype_sizes = {
            torch.float32: 4,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int8: 1,
            torch.uint8: 1,
            torch.int16: 2,
            torch.bool: 1
        }
        return dtype_sizes.get(self.dtype, 4)

    def _torch_to_numpy_dtype(self):
        """将PyTorch数据类型转换为numpy数据类型"""
        dtype_mapping = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.int16: np.int16,
            torch.bool: np.bool_
        }
        return dtype_mapping.get(self.dtype, np.float32)

    def batch_load(self, indices: List[int]) -> List[torch.Tensor]:
        """
        批量加载多个样本
        利用io_uring的批量操作优势
        """
        filenames = [self.file_list[i] for i in indices]
        filepaths = [os.path.join(self.data_dir, f) for f in filenames]

        # 这里可以调用C++端的batch_load_tensors方法
        # 目前使用简化版本
        tensors = []
        for i, filepath in enumerate(filepaths):
            filename = filenames[i]
            if filename.endswith('.npy'):
                tensor = self._load_numpy_file(filepath)
            else:
                tensor = self._load_raw_file(filepath, filename)
            tensors.append(tensor)

        return tensors


class PerformanceDataLoader(DataLoader):
    """
    高性能数据加载器
    优化了与自定义IO后端的集成
    """

    def __init__(self, dataset: CustomIODataset,
                 batch_size: int = 1,
                 enable_batch_loading: bool = True,
                 **kwargs):
        """
        初始化高性能数据加载器

        Args:
            dataset: CustomIODataset实例
            batch_size: 批次大小
            enable_batch_loading: 是否启用批量加载优化
            **kwargs: 其他DataLoader参数
        """
        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.enable_batch_loading = enable_batch_loading

    def __iter__(self):
        if self.enable_batch_loading and hasattr(self.dataset, 'batch_load'):
            return self._batch_optimized_iter()
        else:
            return super().__iter__()

    def _batch_optimized_iter(self):
        """使用批量加载优化的迭代器"""
        sampler_iter = iter(self.sampler)

        while True:
            try:
                # 收集一个batch的索引
                indices = []
                for _ in range(self.batch_size):
                    indices.append(next(sampler_iter))

                # 批量加载数据
                tensors = self.dataset.batch_load(indices)

                # 转换为batch格式
                batch_data = torch.stack(tensors)
                batch_targets = torch.tensor(indices, dtype=torch.long)

                yield batch_data, batch_targets

            except StopIteration:
                break


# 工具函数
def create_sample_dataset(data_dir: str, num_samples: int = 100,
                         sample_shape: Tuple[int, ...] = (224, 224, 3),
                         dtype: torch.dtype = torch.float32):
    """
    创建示例数据集用于测试

    Args:
        data_dir: 数据目录
        num_samples: 样本数量
        sample_shape: 每个样本的形状
        dtype: 数据类型
    """
    os.makedirs(data_dir, exist_ok=True)

    shape_info = {}

    # 生成随机数据文件
    for i in range(num_samples):
        # 生成随机tensor
        tensor = torch.randn(sample_shape, dtype=dtype)

        # 保存为二进制文件
        filename = f"sample_{i:04d}.bin"
        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(tensor.numpy().tobytes())

        # 记录形状信息
        shape_info[filename] = list(sample_shape)

    # 保存形状信息
    shape_file = os.path.join(data_dir, 'shape_info.json')
    with open(shape_file, 'w') as f:
        json.dump(shape_info, f, indent=2)

    print(f"Created sample dataset with {num_samples} samples in {data_dir}")
    print(f"Sample shape: {sample_shape}")
    print(f"Data type: {dtype}")


if __name__ == "__main__":
    # 测试代码
    print("Testing CustomIODataset...")

    # 创建示例数据集
    test_data_dir = "/tmp/torch_io_test_data"
    create_sample_dataset(test_data_dir, num_samples=10, sample_shape=(3, 32, 32))

    # 测试dataset
    dataset = CustomIODataset(
        data_dir=test_data_dir,
        dtype=torch.float32,
        io_backend='normal'  # 使用普通IO进行测试
    )

    print(f"\nDataset length: {len(dataset)}")

    # 测试单个样本加载
    data, target = dataset[0]
    print(f"Sample shape: {data.shape}")
    print(f"Sample dtype: {data.dtype}")
    print(f"Target: {target}")

    # 测试DataLoader
    dataloader = PerformanceDataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        enable_batch_loading=False  # 暂时禁用批量优化
    )

    print(f"\nTesting DataLoader...")
    for i, (batch_data, batch_targets) in enumerate(dataloader):
        print(f"Batch {i}: data shape = {batch_data.shape}, targets = {batch_targets}")
        if i >= 2:  # 只测试前3个batch
            break

    print("Test completed!")
