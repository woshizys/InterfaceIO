#include "io_uring_io.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <liburing.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

class IoUringIO::Impl {
public:
  struct io_uring ring;
  bool initialized = false;

  ~Impl() {
    if (initialized) {
      io_uring_queue_exit(&ring);
    }
  }
};

IoUringIO::IoUringIO() : pImpl(std::make_unique<Impl>()) {}

IoUringIO::~IoUringIO() = default;

bool IoUringIO::initialize(int queue_depth) {
  int ret = io_uring_queue_init(queue_depth, &pImpl->ring, 0);
  if (ret < 0) {
    std::cerr << "io_uring初始化失败: " << strerror(-ret) << std::endl;
    return false;
  }

  pImpl->initialized = true;
  return true;
}

ssize_t IoUringIO::write_file(const std::string &filename, const char *data,
                              size_t size) {
  if (!pImpl->initialized) {
    std::cerr << "io_uring未初始化" << std::endl;
    return -1;
  }

  // 打开文件
  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "打开文件失败: " << filename << ", " << strerror(errno)
              << std::endl;
    return -1;
  }

  // 获取提交队列条目
  struct io_uring_sqe *sqe = io_uring_get_sqe(&pImpl->ring);
  if (!sqe) {
    std::cerr << "获取SQE失败" << std::endl;
    close(fd);
    return -1;
  }

  // 准备写操作
  io_uring_prep_write(sqe, fd, data, size, 0);

  // 提交操作
  int ret = io_uring_submit(&pImpl->ring);
  if (ret < 0) {
    std::cerr << "提交io_uring操作失败: " << strerror(-ret) << std::endl;
    close(fd);
    return -1;
  }

  // 等待完成
  struct io_uring_cqe *cqe;
  ret = io_uring_wait_cqe(&pImpl->ring, &cqe);
  if (ret < 0) {
    std::cerr << "等待io_uring完成失败: " << strerror(-ret) << std::endl;
    close(fd);
    return -1;
  }

  ssize_t bytes_written = cqe->res;
  if (bytes_written < 0) {
    std::cerr << "io_uring写操作失败: " << strerror(-bytes_written)
              << std::endl;
  }

  // 标记CQE已处理
  io_uring_cqe_seen(&pImpl->ring, cqe);
  close(fd);

  return bytes_written;
}

ssize_t IoUringIO::read_file(const std::string &filename, char *buffer,
                             size_t size) {
  if (!pImpl->initialized) {
    std::cerr << "io_uring未初始化" << std::endl;
    return -1;
  }

  // 打开文件
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "打开文件失败: " << filename << ", " << strerror(errno)
              << std::endl;
    return -1;
  }

  // 获取提交队列条目
  struct io_uring_sqe *sqe = io_uring_get_sqe(&pImpl->ring);
  if (!sqe) {
    std::cerr << "获取SQE失败" << std::endl;
    close(fd);
    return -1;
  }

  // 准备读操作
  io_uring_prep_read(sqe, fd, buffer, size, 0);

  // 提交操作
  int ret = io_uring_submit(&pImpl->ring);
  if (ret < 0) {
    std::cerr << "提交io_uring操作失败: " << strerror(-ret) << std::endl;
    close(fd);
    return -1;
  }

  // 等待完成
  struct io_uring_cqe *cqe;
  ret = io_uring_wait_cqe(&pImpl->ring, &cqe);
  if (ret < 0) {
    std::cerr << "等待io_uring完成失败: " << strerror(-ret) << std::endl;
    close(fd);
    return -1;
  }

  ssize_t bytes_read = cqe->res;
  if (bytes_read < 0) {
    std::cerr << "io_uring读操作失败: " << strerror(-bytes_read) << std::endl;
  }

  // 标记CQE已处理
  io_uring_cqe_seen(&pImpl->ring, cqe);
  close(fd);

  return bytes_read;
}

double IoUringIO::benchmark_write(const std::string &filename, const char *data,
                                  size_t size, int iterations) {
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    std::string test_filename = filename + "_" + std::to_string(i);
    ssize_t result = write_file(test_filename, data, size);
    if (result < 0) {
      std::cerr << "io_uring写测试失败，迭代: " << i << std::endl;
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

double IoUringIO::benchmark_read(const std::string &filename, char *buffer,
                                 size_t size, int iterations) {
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    ssize_t result = read_file(filename, buffer, size);
    if (result < 0) {
      std::cerr << "io_uring读测试失败，迭代: " << i << std::endl;
      return -1.0;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / 1000.0; // 返回毫秒
}

// 批量写入文件 - io_uring的真正优势！
std::vector<ssize_t>
IoUringIO::batch_write_files(const std::vector<std::string> &filenames,
                             const char *data, size_t size) {
  std::vector<ssize_t> results(filenames.size(), -1);

  if (!pImpl->initialized) {
    std::cerr << "io_uring未初始化" << std::endl;
    return results;
  }

  if (filenames.empty()) {
    return results;
  }

  // 打开所有文件
  std::vector<int> fds(filenames.size());
  for (size_t i = 0; i < filenames.size(); i++) {
    fds[i] = open(filenames[i].c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fds[i] < 0) {
      std::cerr << "打开文件失败: " << filenames[i] << std::endl;
      // 清理已打开的文件描述符
      for (size_t j = 0; j < i; j++) {
        if (fds[j] >= 0)
          close(fds[j]);
      }
      return results;
    }
  }

  // 使用稳定并发模型：保持 in-flight 的请求数量，批量提交与收割CQE
  const unsigned MAX_INFLIGHT = 128; // 稳态并发上限，可根据队列深度调整
  const unsigned MAX_PEEK = 64; // 单次批量收割CQE数量

  size_t total = filenames.size();
  size_t next_to_submit = 0;
  size_t completed = 0;
  unsigned inflight = 0;

  while (completed < total) {
    // 尽量填满SQ，提交更多请求
    while (inflight < MAX_INFLIGHT && next_to_submit < total) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&pImpl->ring);
      if (!sqe) {
        // SQ已满，先提交让内核消费
        int ret_flush = io_uring_submit(&pImpl->ring);
        if (ret_flush < 0) {
          std::cerr << "提交失败: " << strerror(-ret_flush) << std::endl;
          break;
        }
        continue;
      }

      io_uring_prep_write(sqe, fds[next_to_submit], data, size, 0);
      io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(next_to_submit));
      next_to_submit++;
      inflight++;
    }

    // 将已准备的SQE提交到内核
    int ret_submit = io_uring_submit(&pImpl->ring);
    if (ret_submit < 0) {
      std::cerr << "提交失败: " << strerror(-ret_submit) << std::endl;
    }

    // 批量收割已完成的CQE
    struct io_uring_cqe *cqes[MAX_PEEK];
    int got = io_uring_peek_batch_cqe(&pImpl->ring, cqes, MAX_PEEK);
    if (got == 0) {
      // 没有完成，等待至少1个完成，避免忙轮询
      int ret_wait = io_uring_submit_and_wait(&pImpl->ring, 1);
      if (ret_wait < 0) {
        std::cerr << "等待完成失败: " << strerror(-ret_wait) << std::endl;
      }
      got = io_uring_peek_batch_cqe(&pImpl->ring, cqes, MAX_PEEK);
    }

    for (int i = 0; i < got; i++) {
      struct io_uring_cqe *cqe = cqes[i];
      size_t file_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
      if (file_index < results.size()) {
        results[file_index] = cqe->res;
      }
      io_uring_cqe_seen(&pImpl->ring, cqe);
    }
    completed += got;
    if (got > 0 && inflight >= static_cast<unsigned>(got)) {
      inflight -= static_cast<unsigned>(got);
    }
  }

  // 关闭所有文件描述符
  for (int fd : fds) {
    if (fd >= 0)
      close(fd);
  }

  return results;
}

// 批量读取文件
std::vector<ssize_t>
IoUringIO::batch_read_files(const std::vector<std::string> &filenames,
                            char *buffer, size_t size) {
  std::vector<ssize_t> results(filenames.size(), -1);

  if (!pImpl->initialized) {
    std::cerr << "io_uring未初始化" << std::endl;
    return results;
  }

  if (filenames.empty()) {
    return results;
  }

  // 打开所有文件
  std::vector<int> fds(filenames.size());
  std::vector<std::vector<char>> buffers(filenames.size());

  for (size_t i = 0; i < filenames.size(); i++) {
    fds[i] = open(filenames[i].c_str(), O_RDONLY);
    if (fds[i] < 0) {
      std::cerr << "打开文件失败: " << filenames[i] << std::endl;
      // 清理已打开的文件描述符
      for (size_t j = 0; j < i; j++) {
        if (fds[j] >= 0)
          close(fds[j]);
      }
      return results;
    }
    buffers[i].resize(size);
  }

  // 稳态并发读：持续填充SQ并批量收割CQE
  const unsigned MAX_INFLIGHT = 128;
  const unsigned MAX_PEEK = 64;

  size_t total = filenames.size();
  size_t next_to_submit = 0;
  size_t completed = 0;
  unsigned inflight = 0;

  while (completed < total) {
    while (inflight < MAX_INFLIGHT && next_to_submit < total) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&pImpl->ring);
      if (!sqe) {
        int ret_flush = io_uring_submit(&pImpl->ring);
        if (ret_flush < 0) {
          std::cerr << "提交失败: " << strerror(-ret_flush) << std::endl;
          break;
        }
        continue;
      }

      io_uring_prep_read(sqe, fds[next_to_submit],
                         buffers[next_to_submit].data(), size, 0);
      io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(next_to_submit));
      next_to_submit++;
      inflight++;
    }

    int ret_submit = io_uring_submit(&pImpl->ring);
    if (ret_submit < 0) {
      std::cerr << "提交失败: " << strerror(-ret_submit) << std::endl;
    }

    struct io_uring_cqe *cqes[MAX_PEEK];
    int got = io_uring_peek_batch_cqe(&pImpl->ring, cqes, MAX_PEEK);
    if (got == 0) {
      int ret_wait = io_uring_submit_and_wait(&pImpl->ring, 1);
      if (ret_wait < 0) {
        std::cerr << "等待完成失败: " << strerror(-ret_wait) << std::endl;
      }
      got = io_uring_peek_batch_cqe(&pImpl->ring, cqes, MAX_PEEK);
    }

    for (int i = 0; i < got; i++) {
      struct io_uring_cqe *cqe = cqes[i];
      size_t file_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
      if (file_index < results.size() && cqe->res > 0) {
        results[file_index] = cqe->res;
        if (buffer && cqe->res > 0) {
          memcpy(buffer, buffers[file_index].data(),
                 std::min(static_cast<size_t>(cqe->res), size));
        }
      }
      io_uring_cqe_seen(&pImpl->ring, cqe);
    }
    completed += got;
    if (got > 0 && inflight >= static_cast<unsigned>(got)) {
      inflight -= static_cast<unsigned>(got);
    }
  }

  // 关闭所有文件描述符
  for (int fd : fds) {
    if (fd >= 0)
      close(fd);
  }

  return results;
}
