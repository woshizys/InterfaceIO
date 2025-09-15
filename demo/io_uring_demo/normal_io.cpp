#include "normal_io.h"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

NormalIO::NormalIO() = default;

NormalIO::~NormalIO() = default;

bool NormalIO::initialize() {
  return true;
}

ssize_t NormalIO::write_file(const std::string &filename, const char *data,
                             size_t size) {
  // 使用标准的文件描述符写入
  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "打开文件失败: " << filename << ", " << strerror(errno)
              << std::endl;
    return -1;
  }

  ssize_t bytes_written = write(fd, data, size);
  if (bytes_written < 0) {
    std::cerr << "写文件失败: " << strerror(errno) << std::endl;
  }

  close(fd);
  return bytes_written;
}

ssize_t NormalIO::write_file_stream(const std::string &filename,
                                    const char *data, size_t size) {
  // 使用C++流写入
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "打开文件失败: " << filename << std::endl;
    return -1;
  }

  file.write(data, size);
  if (file.fail()) {
    std::cerr << "写文件失败" << std::endl;
    return -1;
  }

  file.close();
  return size;
}

ssize_t NormalIO::read_file(const std::string &filename, char *buffer,
                            size_t size) {
  // 使用标准的文件描述符读取
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "打开文件失败: " << filename << ", " << strerror(errno)
              << std::endl;
    return -1;
  }

  ssize_t bytes_read = read(fd, buffer, size);
  if (bytes_read < 0) {
    std::cerr << "读文件失败: " << strerror(errno) << std::endl;
  }

  close(fd);
  return bytes_read;
}

ssize_t NormalIO::read_file_stream(const std::string &filename, char *buffer,
                                   size_t size) {
  // 使用C++流读取
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "打开文件失败: " << filename << std::endl;
    return -1;
  }

  file.read(buffer, size);
  ssize_t bytes_read = file.gcount();

  file.close();
  return bytes_read;
}

double NormalIO::benchmark_write(const std::string &filename, const char *data,
                                 size_t size, int iterations, bool use_stream) {
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    std::string test_filename = filename + "_" + std::to_string(i);
    ssize_t result;

    if (use_stream) {
      result = write_file_stream(test_filename, data, size);
    } else {
      result = write_file(test_filename, data, size);
    }

    if (result < 0) {
      std::cerr << "普通IO写测试失败，迭代: " << i << std::endl;
      return -1.0;
    }
    // 删除测试文件
    unlink(test_filename.c_str());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / 1000.0; // 返回毫秒
}

double NormalIO::benchmark_read(const std::string &filename, char *buffer,
                                size_t size, int iterations, bool use_stream) {
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    ssize_t result;

    if (use_stream) {
      result = read_file_stream(filename, buffer, size);
    } else {
      result = read_file(filename, buffer, size);
    }

    if (result < 0) {
      std::cerr << "普通IO读测试失败，迭代: " << i << std::endl;
      return -1.0;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / 1000.0; // 返回毫秒
}
