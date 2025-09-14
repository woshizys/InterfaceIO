#include "io_uring_io.h"
#include "normal_io.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unistd.h>
#include <vector>

// 生成测试数据
std::vector<char> generate_test_data(size_t size) {
  std::vector<char> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(32, 126); // 可打印ASCII字符

  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<char>(dis(gen));
  }

  return data;
}

// 格式化文件大小显示
std::string format_size(size_t bytes) {
  const char *units[] = {"B", "KB", "MB", "GB"};
  double size = bytes;
  int unit = 0;

  while (size >= 1024 && unit < 3) {
    size /= 1024;
    unit++;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
  return oss.str();
}

// 测试读写功能正确性
bool test_correctness(IoUringIO &io_uring, NormalIO &normal_io) {
  std::cout << "\n=== 功能正确性测试 ===" << std::endl;

  const std::string test_data = "Hello, io_uring and normal IO test!";
  const std::string filename = "test_correctness.txt";

  // 测试io_uring写入
  std::cout << "测试io_uring写入..." << std::endl;
  ssize_t result =
      io_uring.write_file(filename, test_data.c_str(), test_data.size());
  if (result != static_cast<ssize_t>(test_data.size())) {
    std::cerr << "io_uring写入失败" << std::endl;
    return false;
  }

  // 测试普通IO读取
  std::cout << "测试普通IO读取..." << std::endl;
  std::vector<char> read_buffer(test_data.size() + 1, 0);
  result = normal_io.read_file(filename, read_buffer.data(), test_data.size());
  if (result != static_cast<ssize_t>(test_data.size())) {
    std::cerr << "普通IO读取失败" << std::endl;
    unlink(filename.c_str());
    return false;
  }

  // 验证数据一致性
  if (std::string(read_buffer.data(), test_data.size()) != test_data) {
    std::cerr << "数据一致性验证失败" << std::endl;
    unlink(filename.c_str());
    return false;
  }

  // 测试普通IO写入
  std::cout << "测试普通IO写入..." << std::endl;
  const std::string test_data2 = "Normal IO write test data!";
  result =
      normal_io.write_file(filename, test_data2.c_str(), test_data2.size());
  if (result != static_cast<ssize_t>(test_data2.size())) {
    std::cerr << "普通IO写入失败" << std::endl;
    unlink(filename.c_str());
    return false;
  }

  // 测试io_uring读取
  std::cout << "测试io_uring读取..." << std::endl;
  std::vector<char> read_buffer2(test_data2.size() + 1, 0);
  result = io_uring.read_file(filename, read_buffer2.data(), test_data2.size());
  if (result != static_cast<ssize_t>(test_data2.size())) {
    std::cerr << "io_uring读取失败" << std::endl;
    unlink(filename.c_str());
    return false;
  }

  // 验证数据一致性
  if (std::string(read_buffer2.data(), test_data2.size()) != test_data2) {
    std::cerr << "数据一致性验证失败" << std::endl;
    unlink(filename.c_str());
    return false;
  }

  // 清理
  unlink(filename.c_str());

  std::cout << "✓ 功能正确性测试通过!" << std::endl;
  return true;
}

// 性能对比测试
void performance_test(IoUringIO &io_uring, NormalIO &normal_io) {
  std::cout << "\n=== 性能对比测试 ===" << std::endl;

  // 测试不同大小的数据
  std::vector<size_t> test_sizes = {1024, 4096, 16384, 65536,
                                    262144}; // 1KB到256KB
  const int iterations = 100;

  std::cout << std::left << std::setw(12) << "数据大小" << std::setw(15)
            << "io_uring写(ms)" << std::setw(15) << "普通IO写(ms)"
            << std::setw(15) << "io_uring读(ms)" << std::setw(15)
            << "普通IO读(ms)" << std::setw(12) << "写入加速比" << std::setw(12)
            << "读取加速比" << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  for (size_t size : test_sizes) {
    // 生成测试数据
    auto test_data = generate_test_data(size);
    std::vector<char> read_buffer(size);

    std::string write_filename = "perf_write_test.dat";
    std::string read_filename = "perf_read_test.dat";

    // 创建读测试文件
    normal_io.write_file(read_filename, test_data.data(), size);

    // 测试写性能
    double io_uring_write_time = io_uring.benchmark_write(
        write_filename, test_data.data(), size, iterations);
    double normal_write_time = normal_io.benchmark_write(
        write_filename, test_data.data(), size, iterations);

    // 测试读性能
    double io_uring_read_time = io_uring.benchmark_read(
        read_filename, read_buffer.data(), size, iterations);
    double normal_read_time = normal_io.benchmark_read(
        read_filename, read_buffer.data(), size, iterations);

    // 计算加速比
    double write_speedup = (io_uring_write_time > 0 && normal_write_time > 0)
                               ? normal_write_time / io_uring_write_time
                               : 0.0;
    double read_speedup = (io_uring_read_time > 0 && normal_read_time > 0)
                              ? normal_read_time / io_uring_read_time
                              : 0.0;

    // 显示结果
    std::cout << std::left << std::setw(12) << format_size(size)
              << std::setw(15) << std::fixed << std::setprecision(2)
              << io_uring_write_time << std::setw(15) << normal_write_time
              << std::setw(15) << io_uring_read_time << std::setw(15)
              << normal_read_time << std::setw(12) << std::setprecision(2)
              << write_speedup << "x" << std::setw(12) << read_speedup << "x"
              << std::endl;

    // 清理测试文件
    unlink(write_filename.c_str());
    unlink(read_filename.c_str());
  }
}

// 测试C++流接口性能
void stream_comparison_test(NormalIO &normal_io) {
  std::cout << "\n=== 普通IO: 文件描述符 vs C++流 对比 ===" << std::endl;

  const size_t test_size = 65536; // 64KB
  const int iterations = 100;

  auto test_data = generate_test_data(test_size);
  std::vector<char> read_buffer(test_size);

  std::string filename = "stream_test.dat";

  // 创建读测试文件
  normal_io.write_file(filename, test_data.data(), test_size);

  // 测试文件描述符方式
  double fd_write_time = normal_io.benchmark_write(
      filename + "_fd", test_data.data(), test_size, iterations, false);
  double fd_read_time = normal_io.benchmark_read(filename, read_buffer.data(),
                                                 test_size, iterations, false);

  // 测试C++流方式
  double stream_write_time = normal_io.benchmark_write(
      filename + "_stream", test_data.data(), test_size, iterations, true);
  double stream_read_time = normal_io.benchmark_read(
      filename, read_buffer.data(), test_size, iterations, true);

  std::cout << "文件描述符写入时间: " << std::fixed << std::setprecision(2)
            << fd_write_time << " ms" << std::endl;
  std::cout << "C++流写入时间:     " << stream_write_time << " ms" << std::endl;
  std::cout << "文件描述符读取时间: " << fd_read_time << " ms" << std::endl;
  std::cout << "C++流读取时间:     " << stream_read_time << " ms" << std::endl;

  if (stream_write_time > 0) {
    std::cout << "写入性能比 (fd/stream): " << std::setprecision(2)
              << fd_write_time / stream_write_time << std::endl;
  }
  if (stream_read_time > 0) {
    std::cout << "读取性能比 (fd/stream): " << std::setprecision(2)
              << fd_read_time / stream_read_time << std::endl;
  }

  // 清理
  unlink(filename.c_str());
}

int main() {
  std::cout << "=== IO接口对比测试程序 ===" << std::endl;

  // 初始化IO接口
  IoUringIO io_uring;
  NormalIO normal_io;

  std::cout << "\n初始化IO接口..." << std::endl;

  if (!io_uring.initialize()) {
    std::cerr << "io_uring初始化失败，程序退出" << std::endl;
    return 1;
  }

  if (!normal_io.initialize()) {
    std::cerr << "普通IO初始化失败，程序退出" << std::endl;
    return 1;
  }

  // 功能正确性测试
  if (!test_correctness(io_uring, normal_io)) {
    std::cerr << "功能正确性测试失败，程序退出" << std::endl;
    return 1;
  }

  // 性能对比测试
  performance_test(io_uring, normal_io);

  // C++流对比测试
  stream_comparison_test(normal_io);

  std::cout << "\n=== 测试完成 ===" << std::endl;
  std::cout << "\n说明:" << std::endl;
  std::cout << "- io_uring是Linux内核的高性能异步IO接口" << std::endl;
  std::cout << "- 对于小文件，普通IO可能更快(系统调用开销较小)" << std::endl;
  std::cout << "- 对于大文件或高并发场景，io_uring通常表现更好" << std::endl;
  std::cout << "- 文件描述符通常比C++流更快，因为少了一层抽象" << std::endl;

  return 0;
}
