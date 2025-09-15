#!/usr/bin/env python3
"""
Python端IO后端接口
提供PyTorch与C++ TensorIO的桥接
"""

import ctypes
import os
from typing import List, Tuple, Optional, Union
import torch
import numpy as np


class TensorIOBackend:
    """
    TensorIO后端的Python接口
    封装C++实现，提供Python友好的API
    """

    def __init__(self, backend_type: str = 'io_uring', library_path: Optional[str] = None):
        """
        初始化TensorIO后端

        Args:
            backend_type: 后端类型 ('io_uring' 或 'normal')
            library_path: C++动态库路径
        """
        self.backend_type = backend_type
        self.library_path = library_path or self._find_library()
        self._lib = None
        self._tensor_io = None

        self._load_library()
        self._initialize_backend()

    def _find_library(self) -> str:
        """
        自动查找编译好的C++动态库
        """
        # 可能的库路径
        possible_paths = [
            # 相对于当前脚本的路径
            os.path.join(os.path.dirname(__file__), "libtensor_io.so"),
            os.path.join(os.path.dirname(__file__), "tensor_io.so"),
            # 构建目录
            os.path.join(os.path.dirname(__file__), "../../../build/libtensor_io.so"),
            # 系统路径
            "/usr/local/lib/libtensor_io.so",
            "/usr/lib/libtensor_io.so"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果找不到库，返回默认名称
        print("Warning: TensorIO library not found, using mock implementation")
        return None

    def _load_library(self):
        """
        加载C++动态库
        """
        if self.library_path is None or not os.path.exists(self.library_path):
            print(f"TensorIO library not available, using Python fallback implementation")
            self._lib = None
            return

        try:
            self._lib = ctypes.CDLL(self.library_path)
            self._setup_function_signatures()
            print(f"Successfully loaded TensorIO library: {self.library_path}")
        except Exception as e:
            print(f"Failed to load TensorIO library: {e}")
            self._lib = None

    def _setup_function_signatures(self):
        """
        设置C++函数的签名
        """
        if self._lib is None:
            return

        # 这里需要根据实际的C++接口定义函数签名
        # 示例签名（需要根据实际实现调整）
        try:
            # 创建TensorIO实例
            self._lib.create_tensor_io.argtypes = [ctypes.c_int]  # backend type
            self._lib.create_tensor_io.restype = ctypes.c_void_p

            # 初始化
            self._lib.tensor_io_initialize.argtypes = [ctypes.c_void_p]
            self._lib.tensor_io_initialize.restype = ctypes.c_bool

            # 删除实例
            self._lib.delete_tensor_io.argtypes = [ctypes.c_void_p]
            self._lib.delete_tensor_io.restype = None

            print("Function signatures set up successfully")
        except AttributeError as e:
            print(f"Some C++ functions not available: {e}")

    def _initialize_backend(self):
        """
        初始化后端实例
        """
        if self._lib is not None:
            try:
                backend_code = 1 if self.backend_type == 'io_uring' else 0
                self._tensor_io = self._lib.create_tensor_io(backend_code)

                if self._tensor_io and self._lib.tensor_io_initialize(self._tensor_io):
                    print(f"TensorIO backend '{self.backend_type}' initialized successfully")
                else:
                    print(f"Failed to initialize TensorIO backend")
                    self._tensor_io = None
            except Exception as e:
                print(f"Error initializing backend: {e}")
                self._tensor_io = None

    def __del__(self):
        """
        清理资源
        """
        if self._lib and self._tensor_io:
            try:
                self._lib.delete_tensor_io(self._tensor_io)
            except:
                pass

    def load_tensor_from_file(self, filename: str, shape: List[int],
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        从文件加载tensor

        Args:
            filename: 文件路径
            shape: tensor形状
            dtype: 数据类型

        Returns:
            加载的tensor
        """
        if self._tensor_io is not None:
            return self._load_tensor_cpp(filename, shape, dtype)
        else:
            return self._load_tensor_python(filename, shape, dtype)

    def save_tensor_to_file(self, tensor: torch.Tensor, filename: str) -> bool:
        """
        保存tensor到文件

        Args:
            tensor: 要保存的tensor
            filename: 文件路径

        Returns:
            是否保存成功
        """
        if self._tensor_io is not None:
            return self._save_tensor_cpp(tensor, filename)
        else:
            return self._save_tensor_python(tensor, filename)

    def batch_load_tensors(self, filenames: List[str], shapes: List[List[int]],
                          dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
        """
        批量加载多个tensor

        Args:
            filenames: 文件路径列表
            shapes: 对应的形状列表
            dtype: 数据类型

        Returns:
            tensor列表
        """
        if self._tensor_io is not None and len(filenames) > 1:
            return self._batch_load_cpp(filenames, shapes, dtype)
        else:
            # 回退到逐个加载
            tensors = []
            for filename, shape in zip(filenames, shapes):
                tensor = self.load_tensor_from_file(filename, shape, dtype)
                tensors.append(tensor)
            return tensors

    def benchmark_performance(self, test_file: str, file_size: int, iterations: int = 10):
        """
        性能基准测试

        Args:
            test_file: 测试文件路径
            file_size: 文件大小
            iterations: 测试迭代次数
        """
        if self._tensor_io is not None:
            print(f"Running C++ benchmark with {self.backend_type} backend...")
            # 调用C++端的benchmark方法
        else:
            print("Running Python fallback benchmark...")
            self._benchmark_python(test_file, file_size, iterations)

    # C++实现版本（需要实际的C++接口）
    def _load_tensor_cpp(self, filename: str, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """C++版本的tensor加载"""
        # 这里需要调用实际的C++函数
        # 目前使用Python fallback
        return self._load_tensor_python(filename, shape, dtype)

    def _save_tensor_cpp(self, tensor: torch.Tensor, filename: str) -> bool:
        """C++版本的tensor保存"""
        # 这里需要调用实际的C++函数
        return self._save_tensor_python(tensor, filename)

    def _batch_load_cpp(self, filenames: List[str], shapes: List[List[int]],
                       dtype: torch.dtype) -> List[torch.Tensor]:
        """C++版本的批量加载"""
        # 这里需要调用实际的C++函数
        tensors = []
        for filename, shape in zip(filenames, shapes):
            tensor = self._load_tensor_python(filename, shape, dtype)
            tensors.append(tensor)
        return tensors

    # Python fallback实现
    def _load_tensor_python(self, filename: str, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """Python版本的tensor加载（作为fallback）"""
        try:
            with open(filename, 'rb') as f:
                data = f.read()

            # 转换为numpy数组
            np_dtype = self._torch_to_numpy_dtype(dtype)
            np_array = np.frombuffer(data, dtype=np_dtype)
            np_array = np_array.reshape(shape)

            # 转换为PyTorch tensor
            tensor = torch.from_numpy(np_array.copy()).to(dtype)
            return tensor

        except Exception as e:
            print(f"Error loading tensor from {filename}: {e}")
            return torch.empty(0)

    def _save_tensor_python(self, tensor: torch.Tensor, filename: str) -> bool:
        """Python版本的tensor保存"""
        try:
            # 确保tensor是连续的
            contiguous_tensor = tensor.contiguous()

            # 转换为numpy并保存
            np_array = contiguous_tensor.numpy()

            with open(filename, 'wb') as f:
                f.write(np_array.tobytes())

            return True

        except Exception as e:
            print(f"Error saving tensor to {filename}: {e}")
            return False

    def _benchmark_python(self, test_file: str, file_size: int, iterations: int):
        """Python版本的性能测试"""
        import time

        # 创建测试数据
        test_data = np.random.random(file_size // 4).astype(np.float32)  # 假设float32

        # 写入测试文件
        with open(test_file, 'wb') as f:
            f.write(test_data.tobytes())

        # 测试读取性能
        shape = [file_size // 4]

        start_time = time.time()
        for _ in range(iterations):
            tensor = self._load_tensor_python(test_file, shape, torch.float32)
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        throughput = (file_size / 1024 / 1024) / avg_time  # MB/s

        print(f"Python backend benchmark results:")
        print(f"  Average load time: {avg_time * 1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} MB/s")

        # 清理测试文件
        os.remove(test_file)

    def _torch_to_numpy_dtype(self, dtype: torch.dtype):
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
        return dtype_mapping.get(dtype, np.float32)


# 单例模式的全局后端实例
_global_backend = None

def get_tensor_io_backend(backend_type: str = 'io_uring') -> TensorIOBackend:
    """
    获取全局TensorIO后端实例

    Args:
        backend_type: 后端类型

    Returns:
        TensorIOBackend实例
    """
    global _global_backend

    if _global_backend is None or _global_backend.backend_type != backend_type:
        _global_backend = TensorIOBackend(backend_type)

    return _global_backend


if __name__ == "__main__":
    # 测试代码
    print("Testing TensorIOBackend...")

    # 测试普通IO后端
    backend = TensorIOBackend('normal')

    # 创建测试tensor
    test_tensor = torch.randn(100, 200, dtype=torch.float32)
    test_file = "/tmp/test_tensor.bin"

    # 测试保存
    print("Testing tensor save...")
    success = backend.save_tensor_to_file(test_tensor, test_file)
    print(f"Save result: {success}")

    # 测试加载
    print("Testing tensor load...")
    loaded_tensor = backend.load_tensor_from_file(test_file, [100, 200], torch.float32)
    print(f"Loaded tensor shape: {loaded_tensor.shape}")
    print(f"Tensors equal: {torch.allclose(test_tensor, loaded_tensor)}")

    # 测试性能
    print("Testing performance...")
    backend.benchmark_performance("/tmp/perf_test.bin", 1024*1024, 5)  # 1MB file

    print("Test completed!")
