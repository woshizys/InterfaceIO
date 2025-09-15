#pragma once
#include <memory>
#include <string>
#include <vector>

class IoUringIO {
public:
  IoUringIO();
  ~IoUringIO();

  // 初始化io_uring
  bool initialize(int queue_depth = 256);

  // 写文件
  ssize_t write_file(const std::string &filename, const char *data,
                     size_t size);

  // 读文件
  ssize_t read_file(const std::string &filename, char *buffer, size_t size);

  // 批量写入文件（io_uring的优势场景）
  std::vector<ssize_t> batch_write_files(const std::vector<std::string>& filenames,
                                        const char* data, size_t size);

  // 批量读取文件
  std::vector<ssize_t> batch_read_files(const std::vector<std::string>& filenames,
                                       char* buffer, size_t size);

  // 写性能测试
  double benchmark_write(const std::string &filename, const char *data,
                         size_t size, int iterations);

  // 读性能测试
  double benchmark_read(const std::string &filename, char *buffer, size_t size,
                        int iterations);

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
