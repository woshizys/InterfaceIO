#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include "../io_uring_demo/io_uring_io.h"
#include "../io_uring_demo/normal_io.h"

/**
 * PyTorch Tensor IO 封装类
 * 提供直接从自定义IO后端加载数据到PyTorch tensor的功能
 */
class TensorIO {
public:
    enum class IOBackend {
        NORMAL,      // 使用普通文件IO
        IO_URING     // 使用io_uring异步IO
    };

    TensorIO(IOBackend backend = IOBackend::IO_URING);
    ~TensorIO();

    /**
     * 初始化IO后端
     */
    bool initialize();

    /**
     * 从文件加载raw数据到PyTorch tensor
     * @param filename 文件路径
     * @param shape tensor的形状
     * @param dtype 数据类型
     * @return PyTorch tensor
     */
    torch::Tensor load_tensor_from_file(const std::string& filename,
                                      const std::vector<int64_t>& shape,
                                      torch::ScalarType dtype = torch::kFloat32);

    /**
     * 将PyTorch tensor保存到文件
     * @param tensor 要保存的tensor
     * @param filename 文件路径
     * @return 是否成功
     */
    bool save_tensor_to_file(const torch::Tensor& tensor, const std::string& filename);

    /**
     * 批量加载多个文件到tensor列表
     * @param filenames 文件路径列表
     * @param shapes 每个tensor的形状列表
     * @param dtype 数据类型
     * @return tensor列表
     */
    std::vector<torch::Tensor> batch_load_tensors(const std::vector<std::string>& filenames,
                                                 const std::vector<std::vector<int64_t>>& shapes,
                                                 torch::ScalarType dtype = torch::kFloat32);

    /**
     * 获取当前使用的IO后端类型
     */
    IOBackend get_backend() const { return backend_; }

    /**
     * 基准测试：对比不同IO后端的性能
     */
    void benchmark_io_performance(const std::string& test_file, size_t file_size, int iterations = 10);

private:
    IOBackend backend_;
    std::unique_ptr<IoUringIO> io_uring_backend_;
    std::unique_ptr<NormalIO> normal_io_backend_;

    // 根据ScalarType获取元素大小
    size_t get_element_size(torch::ScalarType dtype);

    // 计算tensor的总字节数
    size_t calculate_tensor_bytes(const std::vector<int64_t>& shape, torch::ScalarType dtype);
};
