#!/usr/bin/env python3
"""
PyTorch TensorIO Python扩展构建脚本
使用setuptools和pybind11构建Python绑定
"""

import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
from setuptools import setup, Extension
import pybind11

# 获取PyTorch路径
try:
    import torch
    torch_include_path = torch.utils.cpp_extension.include_paths()
    torch_library_path = torch.utils.cpp_extension.library_paths()
    torch_libraries = torch.utils.cpp_extension.libraries()
    print(f"PyTorch include paths: {torch_include_path}")
    print(f"PyTorch library paths: {torch_library_path}")
except ImportError:
    print("Warning: PyTorch not found. Make sure PyTorch is installed.")
    torch_include_path = []
    torch_library_path = []
    torch_libraries = []

# 检查liburing
def check_liburing():
    try:
        result = subprocess.run(['pkg-config', '--exists', 'liburing'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            # 获取liburing编译标志
            cflags_result = subprocess.run(['pkg-config', '--cflags', 'liburing'],
                                         capture_output=True, text=True)
            libs_result = subprocess.run(['pkg-config', '--libs', 'liburing'],
                                       capture_output=True, text=True)

            cflags = cflags_result.stdout.strip().split() if cflags_result.returncode == 0 else []
            libs = libs_result.stdout.strip().split() if libs_result.returncode == 0 else []

            return True, cflags, libs
        else:
            return False, [], []
    except FileNotFoundError:
        print("Warning: pkg-config not found, trying to find liburing manually...")
        # 尝试手动查找liburing
        if os.path.exists('/usr/include/liburing.h') or os.path.exists('/usr/local/include/liburing.h'):
            return True, [], ['-luring']
        return False, [], []

# 检查liburing可用性
liburing_available, liburing_cflags, liburing_libs = check_liburing()
if not liburing_available:
    print("Warning: liburing not found. io_uring backend may not work.")

# 构建源文件列表
current_dir = Path(__file__).parent
parent_io_dir = current_dir.parent / "io_uring_demo"

source_files = [
    str(current_dir / "tensor_io_python.cpp"),
    str(current_dir / "tensor_io.cpp"),
    str(parent_io_dir / "io_uring_io.cpp"),
    str(parent_io_dir / "normal_io.cpp"),
]

# 验证源文件存在
missing_files = [f for f in source_files if not os.path.exists(f)]
if missing_files:
    print(f"Error: Missing source files: {missing_files}")
    print("Please make sure you're running this from the correct directory and parent IO files exist.")
    sys.exit(1)

# 包含目录
include_dirs = [
    str(current_dir),
    str(parent_io_dir),
    pybind11.get_include(),
] + torch_include_path

# 库目录
library_dirs = torch_library_path

# 链接库
libraries = ['pthread'] + torch_libraries
if liburing_available:
    libraries.append('uring')

# 编译标志
compile_args = [
    '-std=c++17',
    '-O3',
    '-Wall',
    '-Wextra',
    '-fPIC',
    '-DWITH_PYTHON',
] + liburing_cflags

# 链接标志
link_args = liburing_libs

# 如果没有liburing，添加编译宏
if not liburing_available:
    compile_args.append('-DNO_LIBURING')

# 创建扩展模块
ext_modules = [
    Pybind11Extension(
        name="tensor_io_py",
        sources=source_files,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        cxx_std=17,
    ),
]

# 设置编译参数
for ext in ext_modules:
    ext.extra_compile_args = compile_args
    ext.extra_link_args = link_args

# 自定义构建类
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # 打印构建信息
        print("\n" + "="*60)
        print("Building TensorIO Python Extension")
        print("="*60)
        print(f"PyTorch version: {torch.__version__ if 'torch' in globals() else 'Not found'}")
        print(f"liburing available: {liburing_available}")
        print(f"Include directories: {include_dirs}")
        print(f"Libraries: {libraries}")
        print(f"Compile args: {compile_args}")
        print("="*60 + "\n")

        super().build_extensions()

# 设置包信息
setup(
    name="tensor_io",
    version="0.1.0",
    author="TensorIO Development Team",
    author_email="dev@tensorio.com",
    description="High-performance IO backend for PyTorch with io_uring support",
    long_description="""
    TensorIO provides high-performance IO backends for PyTorch, including:
    - io_uring async IO support for Linux
    - Optimized tensor loading and saving
    - Batch operations for improved throughput
    - Seamless PyTorch integration

    This package allows you to load and save PyTorch tensors using high-performance
    IO backends, significantly improving data loading performance for machine learning workloads.
    """,
    long_description_content_type="text/plain",
    url="https://github.com/your-org/tensor-io",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "jupyter"],
        "test": ["pytest", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    keywords="pytorch, io, uring, machine-learning, deep-learning, high-performance",
    project_urls={
        "Bug Reports": "https://github.com/your-org/tensor-io/issues",
        "Source": "https://github.com/your-org/tensor-io",
        "Documentation": "https://tensor-io.readthedocs.io",
    },
)

# 构建后的说明
print("\n" + "="*60)
print("Build Instructions")
print("="*60)
print("To build and install:")
print("  python setup.py build_ext --inplace")
print("  python setup.py install")
print()
print("To build in-place for development:")
print("  python setup.py develop")
print()
print("To create distribution packages:")
print("  python setup.py sdist bdist_wheel")
print("="*60)
