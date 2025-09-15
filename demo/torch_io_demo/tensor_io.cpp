#include "tensor_io.h"
#include <iostream>
#include <chrono>
#include <fstream>

TensorIO::TensorIO(IOBackend backend) : backend_(backend) {
    if (backend_ == IOBackend::IO_URING) {
        io_uring_backend_ = std::make_unique<IoUringIO>();
    } else {
        normal_io_backend_ = std::make_unique<NormalIO>();
    }
}

TensorIO::~TensorIO() = default;

bool TensorIO::initialize() {
    if (backend_ == IOBackend::IO_URING) {
        if (!io_uring_backend_->initialize()) {
            std::cerr << "Failed to initialize io_uring backend" << std::endl;
            return false;
        }
    } else {
        if (!normal_io_backend_->initialize()) {
            std::cerr << "Failed to initialize normal IO backend" << std::endl;
            return false;
        }
    }

    std::cout << "TensorIO initialized with "
              << (backend_ == IOBackend::IO_URING ? "io_uring" : "normal IO")
              << " backend" << std::endl;
    return true;
}

torch::Tensor TensorIO::load_tensor_from_file(const std::string& filename,
                                            const std::vector<int64_t>& shape,
                                            torch::ScalarType dtype) {
    // 计算需要读取的数据大小
    size_t total_bytes = calculate_tensor_bytes(shape, dtype);

    // 创建tensor
    torch::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));

    // 获取tensor的数据指针
    void* data_ptr = tensor.data_ptr();

    // 根据后端类型读取数据
    ssize_t bytes_read = 0;
    if (backend_ == IOBackend::IO_URING) {
        bytes_read = io_uring_backend_->read_file(filename,
                                                static_cast<char*>(data_ptr),
                                                total_bytes);
    } else {
        bytes_read = normal_io_backend_->read_file(filename,
                                                 static_cast<char*>(data_ptr),
                                                 total_bytes);
    }

    if (bytes_read != static_cast<ssize_t>(total_bytes)) {
        std::cerr << "Failed to read complete tensor data from " << filename
                  << ". Expected " << total_bytes << " bytes, got " << bytes_read << std::endl;
        return torch::empty(0); // 返回空tensor表示失败
    }

    return tensor;
}

bool TensorIO::save_tensor_to_file(const torch::Tensor& tensor, const std::string& filename) {
    // 确保tensor是连续的
    torch::Tensor contiguous_tensor = tensor.contiguous();

    // 获取数据指针和大小
    const void* data_ptr = contiguous_tensor.data_ptr();
    size_t total_bytes = contiguous_tensor.numel() * contiguous_tensor.element_size();

    // 根据后端类型写入数据
    ssize_t bytes_written = 0;
    if (backend_ == IOBackend::IO_URING) {
        bytes_written = io_uring_backend_->write_file(filename,
                                                    static_cast<const char*>(data_ptr),
                                                    total_bytes);
    } else {
        bytes_written = normal_io_backend_->write_file(filename,
                                                     static_cast<const char*>(data_ptr),
                                                     total_bytes);
    }

    if (bytes_written != static_cast<ssize_t>(total_bytes)) {
        std::cerr << "Failed to write complete tensor data to " << filename
                  << ". Expected " << total_bytes << " bytes, wrote " << bytes_written << std::endl;
        return false;
    }

    return true;
}

std::vector<torch::Tensor> TensorIO::batch_load_tensors(const std::vector<std::string>& filenames,
                                                       const std::vector<std::vector<int64_t>>& shapes,
                                                       torch::ScalarType dtype) {
    std::vector<torch::Tensor> tensors;

    if (filenames.size() != shapes.size()) {
        std::cerr << "Mismatch between filenames count and shapes count" << std::endl;
        return tensors;
    }

    // 如果使用io_uring后端，可以利用批量操作的优势
    if (backend_ == IOBackend::IO_URING && filenames.size() > 1) {
        // 预先创建所有tensors
        std::vector<size_t> tensor_bytes;
        for (const auto& shape : shapes) {
            tensors.push_back(torch::empty(shape, torch::TensorOptions().dtype(dtype)));
            tensor_bytes.push_back(calculate_tensor_bytes(shape, dtype));
        }

        // 使用批量读取（这里简化实现，实际可以进一步优化）
        for (size_t i = 0; i < filenames.size(); ++i) {
            ssize_t bytes_read = io_uring_backend_->read_file(filenames[i],
                                                            static_cast<char*>(tensors[i].data_ptr()),
                                                            tensor_bytes[i]);
            if (bytes_read != static_cast<ssize_t>(tensor_bytes[i])) {
                std::cerr << "Failed to read tensor data from " << filenames[i] << std::endl;
                tensors.clear();
                return tensors;
            }
        }
    } else {
        // 普通模式或单文件，逐个加载
        for (size_t i = 0; i < filenames.size(); ++i) {
            torch::Tensor tensor = load_tensor_from_file(filenames[i], shapes[i], dtype);
            if (tensor.numel() == 0) {
                std::cerr << "Failed to load tensor from " << filenames[i] << std::endl;
                tensors.clear();
                return tensors;
            }
            tensors.push_back(tensor);
        }
    }

    return tensors;
}

void TensorIO::benchmark_io_performance(const std::string& test_file, size_t file_size, int iterations) {
    std::cout << "\n=== TensorIO 性能基准测试 ===" << std::endl;
    std::cout << "测试文件: " << test_file << std::endl;
    std::cout << "文件大小: " << file_size << " bytes" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;

    // 创建测试数据
    std::vector<char> test_data(file_size, 42); // 用42填充测试数据

    // 先写入测试文件
    std::ofstream ofs(test_file, std::ios::binary);
    ofs.write(test_data.data(), file_size);
    ofs.close();

    // 创建用于测试的tensor形状（假设float32数据）
    std::vector<int64_t> shape = {static_cast<int64_t>(file_size / sizeof(float))};

    // 测试读取性能
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        torch::Tensor tensor = load_tensor_from_file(test_file, shape, torch::kFloat32);
        if (tensor.numel() == 0) {
            std::cerr << "Benchmark failed: could not load tensor" << std::endl;
            return;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time = duration.count() / 1000.0 / iterations; // 转换为毫秒
    double throughput = (file_size / 1024.0 / 1024.0) / (avg_time / 1000.0); // MB/s

    std::cout << "平均读取时间: " << avg_time << " ms" << std::endl;
    std::cout << "吞吐量: " << throughput << " MB/s" << std::endl;

    // 清理测试文件
    std::remove(test_file.c_str());
}

size_t TensorIO::get_element_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return sizeof(float);
        case torch::kFloat64: return sizeof(double);
        case torch::kInt32: return sizeof(int32_t);
        case torch::kInt64: return sizeof(int64_t);
        case torch::kInt8: return sizeof(int8_t);
        case torch::kUInt8: return sizeof(uint8_t);
        case torch::kInt16: return sizeof(int16_t);
        case torch::kBool: return sizeof(bool);
        default:
            std::cerr << "Unsupported dtype" << std::endl;
            return 0;
    }
}

size_t TensorIO::calculate_tensor_bytes(const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    size_t total_elements = 1;
    for (int64_t dim : shape) {
        total_elements *= dim;
    }
    return total_elements * get_element_size(dtype);
}
