#pragma once
#include <string>

class NormalIO {
public:
  NormalIO();
  ~NormalIO();

  // 初始化（普通IO不需要特殊初始化）
  bool initialize();

  // 使用文件描述符写文件
  ssize_t write_file(const std::string &filename, const char *data,
                     size_t size);

  // 使用C++流写文件
  ssize_t write_file_stream(const std::string &filename, const char *data,
                            size_t size);

  // 使用文件描述符读文件
  ssize_t read_file(const std::string &filename, char *buffer, size_t size);

  // 使用C++流读文件
  ssize_t read_file_stream(const std::string &filename, char *buffer,
                           size_t size);

  // 写性能测试
  double benchmark_write(const std::string &filename, const char *data,
                         size_t size, int iterations, bool use_stream = false);

  // 读性能测试
  double benchmark_read(const std::string &filename, char *buffer, size_t size,
                        int iterations, bool use_stream = false);
};
