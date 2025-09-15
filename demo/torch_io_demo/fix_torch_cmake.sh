#!/bin/bash

# 修复PyTorch CMake配置问题的脚本
# 作者: AI助手
# 日期: 2025-09-15

echo "=== PyTorch CMake配置修复脚本 ==="
echo

# 检查当前PyTorch版本
echo "1. 检查当前PyTorch安装..."
python3.10 -c "import torch; print('当前PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 如果是CUDA版本但CUDA不可用，安装CPU版本
echo
echo "2. 检查是否需要安装CPU版本的PyTorch..."
CUDA_AVAILABLE=$(python3.10 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
TORCH_VERSION=$(python3.10 -c "import torch; print(torch.__version__)" 2>/dev/null)

if [[ $TORCH_VERSION == *"cu"* ]] && [[ $CUDA_AVAILABLE == "False" ]]; then
    echo "检测到CUDA版本的PyTorch但CUDA不可用，正在安装CPU版本..."
    python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
    echo "CPU版本PyTorch安装完成"
else
    echo "PyTorch配置正常"
fi

# 安装pybind11（如果需要）
echo
echo "3. 检查并安装pybind11..."
python3.10 -c "import pybind11" 2>/dev/null || {
    echo "正在安装pybind11..."
    python3.10 -m pip install pybind11
}

# 获取CMake路径
echo
echo "4. 获取CMake配置路径..."
TORCH_CMAKE_PATH=$(python3.10 -c "import torch; print(torch.utils.cmake_prefix_path)")
PYBIND11_CMAKE_PATH=$(python3.10 -c "import pybind11; print(pybind11.get_cmake_dir())")

echo "Torch CMake路径: $TORCH_CMAKE_PATH"
echo "pybind11 CMake路径: $PYBIND11_CMAKE_PATH"

# 创建构建目录并配置
echo
echo "5. 配置CMake..."
mkdir -p build
cd build
rm -rf *

# 运行CMake配置
cmake \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH;$PYBIND11_CMAKE_PATH" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
    ..

if [ $? -eq 0 ]; then
    echo
    echo "✓ CMake配置成功！"
    echo
    echo "现在可以运行以下命令来编译项目："
    echo "  cd build"
    echo "  make -j4"
else
    echo
    echo "✗ CMake配置失败，请检查错误信息"
    exit 1
fi
