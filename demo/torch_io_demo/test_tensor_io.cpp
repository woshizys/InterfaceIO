#include "tensor_io.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

/**
 * TensorIO C++测试程序
 * 验证功能正确性和性能
 */

void test_basic_functionality() {
    std::cout << "\n=== 基础功能测试 ===" << std::endl;

    // 测试两种后端
    std::vector<std::pair<TensorIO::IOBackend, std::string>> backends = {
        {TensorIO::IOBackend::NORMAL, "Normal IO"},
        {TensorIO::IOBackend::IO_URING, "io_uring"}
    };

    for (const auto& [backend, name] : backends) {
        std::cout << "\n--- 测试 " << name << " 后端 ---" << std::endl;

        TensorIO tensor_io(backend);

        if (!tensor_io.initialize()) {
            std::cout << "❌ " << name << " 后端初始化失败" << std::endl;
            continue;
        }

        std::cout << "✓ " << name << " 后端初始化成功" << std::endl;

        // 创建测试tensor
        std::vector<int64_t> shape = {100, 200};
        torch::Tensor original_tensor = torch::randn(shape, torch::kFloat32);

        std::string test_file = "/tmp/test_tensor_" + std::to_string(static_cast<int>(backend)) + ".bin";

        // 测试保存
        bool save_success = tensor_io.save_tensor_to_file(original_tensor, test_file);
        if (!save_success) {
            std::cout << "❌ " << name << " tensor保存失败" << std::endl;
            continue;
        }
        std::cout << "✓ " << name << " tensor保存成功" << std::endl;

        // 测试加载
        torch::Tensor loaded_tensor = tensor_io.load_tensor_from_file(test_file, shape, torch::kFloat32);
        if (loaded_tensor.numel() == 0) {
            std::cout << "❌ " << name << " tensor加载失败" << std::endl;
            continue;
        }
        std::cout << "✓ " << name << " tensor加载成功" << std::endl;

        // 验证数据一致性
        bool data_equal = torch::allclose(original_tensor, loaded_tensor, 1e-6);
        if (!data_equal) {
            std::cout << "❌ " << name << " 数据一致性验证失败" << std::endl;
            continue;
        }
        std::cout << "✓ " << name << " 数据一致性验证通过" << std::endl;

        // 清理测试文件
        std::remove(test_file.c_str());
    }
}

void test_batch_operations() {
    std::cout << "\n=== 批量操作测试 ===" << std::endl;

    TensorIO tensor_io(TensorIO::IOBackend::IO_URING);

    if (!tensor_io.initialize()) {
        std::cout << "❌ io_uring后端初始化失败，跳过批量操作测试" << std::endl;
        return;
    }

    // 创建多个测试tensor
    int num_tensors = 5;
    std::vector<torch::Tensor> original_tensors;
    std::vector<std::string> filenames;
    std::vector<std::vector<int64_t>> shapes;

    for (int i = 0; i < num_tensors; ++i) {
        std::vector<int64_t> shape = {50 + i * 10, 100 + i * 20};  // 不同形状
        torch::Tensor tensor = torch::randn(shape, torch::kFloat32);

        original_tensors.push_back(tensor);
        shapes.push_back(shape);

        std::string filename = "/tmp/batch_test_" + std::to_string(i) + ".bin";
        filenames.push_back(filename);

        // 保存每个tensor
        tensor_io.save_tensor_to_file(tensor, filename);
    }

    // 批量加载
    std::vector<torch::Tensor> loaded_tensors = tensor_io.batch_load_tensors(filenames, shapes);

    if (loaded_tensors.size() != original_tensors.size()) {
        std::cout << "❌ 批量加载tensor数量不匹配" << std::endl;
        return;
    }

    // 验证每个tensor
    bool all_equal = true;
    for (size_t i = 0; i < loaded_tensors.size(); ++i) {
        if (!torch::allclose(original_tensors[i], loaded_tensors[i], 1e-6)) {
            std::cout << "❌ 第" << i << "个tensor数据不匹配" << std::endl;
            all_equal = false;
        }
    }

    if (all_equal) {
        std::cout << "✓ 批量操作测试通过，所有" << num_tensors << "个tensor数据一致" << std::endl;
    }

    // 清理测试文件
    for (const auto& filename : filenames) {
        std::remove(filename.c_str());
    }
}

void test_performance_comparison() {
    std::cout << "\n=== 性能对比测试 ===" << std::endl;

    // 测试参数
    std::vector<size_t> test_sizes = {1024*1024, 4*1024*1024, 16*1024*1024};  // 1MB, 4MB, 16MB
    int iterations = 10;

    std::cout << std::left << std::setw(12) << "文件大小"
              << std::setw(15) << "Normal IO(ms)"
              << std::setw(15) << "io_uring(ms)"
              << std::setw(15) << "加速比" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (size_t size : test_sizes) {
        // Normal IO测试
        TensorIO normal_io(TensorIO::IOBackend::NORMAL);
        normal_io.initialize();

        std::string normal_file = "/tmp/perf_normal.bin";

        auto start = std::chrono::high_resolution_clock::now();
        normal_io.benchmark_io_performance(normal_file, size, iterations);
        auto end = std::chrono::high_resolution_clock::now();

        double normal_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        // io_uring测试
        TensorIO uring_io(TensorIO::IOBackend::IO_URING);
        double uring_time = 0.0;

        if (uring_io.initialize()) {
            std::string uring_file = "/tmp/perf_uring.bin";

            start = std::chrono::high_resolution_clock::now();
            uring_io.benchmark_io_performance(uring_file, size, iterations);
            end = std::chrono::high_resolution_clock::now();

            uring_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        }

        // 计算加速比
        double speedup = (uring_time > 0) ? normal_time / uring_time : 0.0;

        // 格式化输出
        std::string size_str;
        if (size >= 1024*1024) {
            size_str = std::to_string(size / (1024*1024)) + "MB";
        } else if (size >= 1024) {
            size_str = std::to_string(size / 1024) + "KB";
        } else {
            size_str = std::to_string(size) + "B";
        }

        std::cout << std::left << std::setw(12) << size_str
                  << std::setw(15) << std::fixed << std::setprecision(2) << normal_time
                  << std::setw(15) << (uring_time > 0 ? std::to_string(uring_time) : "N/A")
                  << std::setw(15) << (speedup > 0 ? std::to_string(speedup) + "x" : "N/A")
                  << std::endl;
    }
}

void test_different_dtypes() {
    std::cout << "\n=== 不同数据类型测试 ===" << std::endl;

    TensorIO tensor_io(TensorIO::IOBackend::NORMAL);  // 使用稳定的normal IO

    if (!tensor_io.initialize()) {
        std::cout << "❌ 后端初始化失败" << std::endl;
        return;
    }

    // 测试不同数据类型
    std::vector<std::pair<torch::ScalarType, std::string>> dtypes = {
        {torch::kFloat32, "float32"},
        {torch::kFloat64, "float64"},
        {torch::kInt32, "int32"},
        {torch::kInt64, "int64"},
        {torch::kInt8, "int8"}
    };

    std::vector<int64_t> shape = {10, 20};

    for (const auto& [dtype, name] : dtypes) {
        std::cout << "测试数据类型: " << name << std::endl;

        // 创建tensor
        torch::Tensor original_tensor;
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            original_tensor = torch::randn(shape, dtype);
        } else {
            original_tensor = torch::randint(0, 100, shape, dtype);
        }

        std::string test_file = "/tmp/dtype_test_" + name + ".bin";

        // 保存和加载
        bool save_success = tensor_io.save_tensor_to_file(original_tensor, test_file);
        torch::Tensor loaded_tensor = tensor_io.load_tensor_from_file(test_file, shape, dtype);

        // 验证
        bool data_equal = torch::allclose(original_tensor, loaded_tensor, 1e-6);

        std::cout << "  保存: " << (save_success ? "✓" : "❌")
                  << ", 加载: " << (loaded_tensor.numel() > 0 ? "✓" : "❌")
                  << ", 数据一致: " << (data_equal ? "✓" : "❌") << std::endl;

        // 清理
        std::remove(test_file.c_str());
    }
}

int main() {
    std::cout << "TensorIO C++测试程序" << std::endl;
    std::cout << "===================" << std::endl;

    try {
        test_basic_functionality();
        test_batch_operations();
        test_different_dtypes();
        test_performance_comparison();

        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "所有测试完成!" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
