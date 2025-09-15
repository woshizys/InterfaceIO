# PyTorch 自定义IO集成演示

这个演示展示了如何将自定义IO接口（包括io_uring）集成到PyTorch中，实现高性能的数据加载。

## 功能特性

1. **自定义Dataset**: 基于自定义IO接口的PyTorch Dataset实现
2. **高性能数据加载**: 利用io_uring的异步IO能力
3. **灵活的IO后端**: 支持普通IO和io_uring两种模式
4. **Tensor直接加载**: 直接从文件系统加载数据到PyTorch tensors

## 文件结构

```
torch_io_demo/
├── README.md              # 说明文档
├── custom_dataset.py      # 自定义Dataset实现
├── tensor_io.cpp          # C++端Tensor IO实现
├── tensor_io.h            # C++头文件
├── io_backend.py          # Python IO后端接口
├── demo_train.py          # 演示训练脚本
├── setup.py               # Python扩展构建脚本
└── CMakeLists.txt         # C++编译配置
```

## 使用方法

1. 编译C++扩展
2. 运行演示脚本
3. 观察不同IO模式的性能对比

## 技术实现

- 使用PyBind11绑定C++和Python
- 集成父项目的io_uring和normal_io实现
- 提供PyTorch兼容的Dataset和DataLoader接口
