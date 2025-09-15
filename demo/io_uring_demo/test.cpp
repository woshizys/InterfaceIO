#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <future>
#include <atomic>
#include <iomanip>
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>
#include "io_uring_io.h"
#include "normal_io.h"

// 更可靠的高并发测试
class ReliableConcurrentTest {
private:
    std::vector<char> generate_test_data(size_t size) {
        std::vector<char> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(65, 90); // A-Z

        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<char>(dis(gen));
        }
        return data;
    }

    // 系统预热 - 避免冷启动影响
    void warmup_system() {
        std::cout << "系统预热中..." << std::endl;
        auto test_data = generate_test_data(4096);

        // 预热io_uring
        IoUringIO io_uring;
        if (io_uring.initialize(64)) {
            for (int i = 0; i < 10; i++) {
                std::string filename = "warmup_uring_" + std::to_string(i);
                io_uring.write_file(filename, test_data.data(), 4096);
                unlink(filename.c_str());
            }
        }

        // 预热普通IO
        NormalIO normal_io;
        normal_io.initialize();
        for (int i = 0; i < 10; i++) {
            std::string filename = "warmup_normal_" + std::to_string(i);
            normal_io.write_file(filename, test_data.data(), 4096);
            unlink(filename.c_str());
        }

        // 清理内存缓存 (尽力而为)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    struct TestResult {
        double duration_ms;
        int success_count;
        int total_count;

        double get_throughput() const {
            if (duration_ms > 0) {
                return (success_count * 1000.0) / duration_ms; // ops/sec
            }
            return 0.0;
        }
    };

    // 正确的io_uring批量测试 - 共享一个io_uring实例，充分利用批量优势
    TestResult single_test_io_uring(int thread_count, int files_per_thread, size_t file_size, const std::vector<char>& test_data) {
        TestResult result = {0.0, 0, thread_count * files_per_thread};

        auto start_time = std::chrono::high_resolution_clock::now();

        // 使用单个io_uring实例进行批量处理
        IoUringIO io_uring;
        if (!io_uring.initialize(thread_count * files_per_thread + 64)) {
            return result;
        }

        // 准备所有文件操作
        std::vector<std::string> filenames;
        for (int t = 0; t < thread_count; t++) {
            for (int f = 0; f < files_per_thread; f++) {
                filenames.push_back("test_uring_t" + std::to_string(t) + "_f" + std::to_string(f) + ".dat");
            }
        }

        // 使用真正的批量写入 - 这才是io_uring的正确用法！
        auto batch_results = io_uring.batch_write_files(filenames, test_data.data(), file_size);

        int success_count = 0;
        for (ssize_t result_size : batch_results) {
            if (result_size > 0) {
                success_count++;
            }
        }

        result.success_count = success_count;

        // 清理文件
        for (const auto& filename : filenames) {
            unlink(filename.c_str());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.duration_ms = duration.count() / 1000.0;

        return result;
    }

    TestResult single_test_normal_io(int thread_count, int files_per_thread, size_t file_size, const std::vector<char>& test_data) {
        TestResult result = {0.0, 0, thread_count * files_per_thread};

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::future<int>> futures;
        for (int t = 0; t < thread_count; t++) {
            futures.push_back(std::async(std::launch::async, [=, &test_data]() -> int {
                NormalIO normal_io;
                normal_io.initialize();

                int success_count = 0;
                for (int f = 0; f < files_per_thread; f++) {
                    std::string filename = "test_normal_t" + std::to_string(t) + "_f" + std::to_string(f) + ".dat";
                    if (normal_io.write_file(filename, test_data.data(), file_size) > 0) {
                        success_count++;
                    }
                    unlink(filename.c_str());
                }
                return success_count;
            }));
        }

        for (auto& future : futures) {
            result.success_count += future.get();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.duration_ms = duration.count() / 1000.0;

        return result;
    }

public:
    // 多次采样的可靠测试
    void reliable_concurrent_test(int thread_count, int files_per_thread, size_t file_size, int test_rounds = 5) {
        std::cout << "\n=== 可靠性多线程测试 ===" << std::endl;
        std::cout << "线程数: " << thread_count
                  << ", 每线程文件数: " << files_per_thread
                  << ", 文件大小: " << file_size / 1024 << "KB"
                  << ", 测试轮数: " << test_rounds << std::endl;

        warmup_system();

        auto test_data = generate_test_data(file_size);

        std::vector<double> uring_times, normal_times;
        std::vector<double> uring_throughputs, normal_throughputs;

        // 多轮测试
        for (int round = 0; round < test_rounds; round++) {
            std::cout << "第 " << (round + 1) << "/" << test_rounds << " 轮测试..." << std::endl;

            // 在每轮之间稍作休息，让系统状态稳定
            if (round > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }

            // io_uring测试
            auto uring_result = single_test_io_uring(thread_count, files_per_thread, file_size, test_data);
            uring_times.push_back(uring_result.duration_ms);
            uring_throughputs.push_back(uring_result.get_throughput());

            // 中间休息
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // 普通IO测试
            auto normal_result = single_test_normal_io(thread_count, files_per_thread, file_size, test_data);
            normal_times.push_back(normal_result.duration_ms);
            normal_throughputs.push_back(normal_result.get_throughput());

            std::cout << "  io_uring: " << std::fixed << std::setprecision(1)
                      << uring_result.duration_ms << "ms, "
                      << uring_result.get_throughput() << " ops/sec" << std::endl;
            std::cout << "  普通IO:   " << normal_result.duration_ms << "ms, "
                      << normal_result.get_throughput() << " ops/sec" << std::endl;
        }

        // 计算统计结果
        auto calculate_stats = [](const std::vector<double>& values) {
            if (values.empty()) return std::make_tuple(0.0, 0.0, 0.0);

            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            double mean = sum / values.size();

            std::vector<double> sorted_values = values;
            std::sort(sorted_values.begin(), sorted_values.end());
            double median = sorted_values[sorted_values.size() / 2];

            double variance = 0.0;
            for (double val : values) {
                variance += (val - mean) * (val - mean);
            }
            double stddev = std::sqrt(variance / values.size());

            return std::make_tuple(mean, median, stddev);
        };

        auto [uring_mean_time, uring_median_time, uring_stddev_time] = calculate_stats(uring_times);
        auto [normal_mean_time, normal_median_time, normal_stddev_time] = calculate_stats(normal_times);
        auto [uring_mean_throughput, uring_median_throughput, uring_stddev_throughput] = calculate_stats(uring_throughputs);
        auto [normal_mean_throughput, normal_median_throughput, normal_stddev_throughput] = calculate_stats(normal_throughputs);

        // 显示统计结果
        std::cout << "\n=== 统计结果 ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "io_uring时间: 平均=" << uring_mean_time << "ms, 中位数=" << uring_median_time
                  << "ms, 标准差=" << uring_stddev_time << "ms" << std::endl;
        std::cout << "普通IO时间:   平均=" << normal_mean_time << "ms, 中位数=" << normal_median_time
                  << "ms, 标准差=" << normal_stddev_time << "ms" << std::endl;

        std::cout << std::setprecision(1);
        std::cout << "io_uring吞吐量: 平均=" << uring_mean_throughput << " ops/sec, 中位数=" << uring_median_throughput << " ops/sec" << std::endl;
        std::cout << "普通IO吞吐量:   平均=" << normal_mean_throughput << " ops/sec, 中位数=" << normal_median_throughput << " ops/sec" << std::endl;

        // 性能比较
        if (normal_mean_time > 0 && uring_mean_time > 0) {
            double time_speedup = normal_mean_time / uring_mean_time;
            double throughput_speedup = uring_mean_throughput / normal_mean_throughput;

            std::cout << std::setprecision(2);
            std::cout << "平均性能提升: " << time_speedup << "x (基于时间), "
                      << throughput_speedup << "x (基于吞吐量)" << std::endl;

            // 结果一致性检查
            double time_cv = uring_stddev_time / uring_mean_time * 100;
            double normal_time_cv = normal_stddev_time / normal_mean_time * 100;

            std::cout << "结果稳定性: io_uring CV=" << std::setprecision(1) << time_cv
                      << "%, 普通IO CV=" << normal_time_cv << "%" << std::endl;

            if (time_cv > 20 || normal_time_cv > 20) {
                std::cout << "⚠️  警告: 结果变异较大，建议增加测试轮数或检查系统负载" << std::endl;
            }
        }
    }

    // 扫描不同并发级别并输出JSON格式数据
    void concurrency_sweep_test() {
        std::cout << "\n=== 并发级别扫描测试 ===" << std::endl;

        std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16, 20, 24};
        const int files_per_thread = 20;
        const size_t file_size = 16384; // 16KB

        // 输出JSON格式的数据供Python脚本处理
        std::cout << "\n--- BENCHMARK_DATA_START ---" << std::endl;
        std::cout << "{\"benchmark_results\": [" << std::endl;

        bool first = true;
        for (int threads : thread_counts) {
            if (threads > static_cast<int>(std::thread::hardware_concurrency()) * 2) {
                continue; // 跳过过高的并发级别
            }

            auto test_data = generate_test_data(file_size);

            // 进行5轮测试获取详细数据
            std::vector<double> uring_throughputs, normal_throughputs;

            for (int round = 0; round < 5; round++) {
                auto uring_result = single_test_io_uring(threads, files_per_thread, file_size, test_data);
                auto normal_result = single_test_normal_io(threads, files_per_thread, file_size, test_data);

                uring_throughputs.push_back(uring_result.get_throughput());
                normal_throughputs.push_back(normal_result.get_throughput());

                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            double avg_uring = std::accumulate(uring_throughputs.begin(), uring_throughputs.end(), 0.0) / 5;
            double avg_normal = std::accumulate(normal_throughputs.begin(), normal_throughputs.end(), 0.0) / 5;

            double speedup = (avg_normal > 0) ? avg_uring / avg_normal : 0.0;

            if (!first) std::cout << "," << std::endl;
            std::cout << "  {\"threads\": " << threads << ",\n";
            std::cout << "   \"io_uring_samples\": [";
            for (size_t i = 0; i < uring_throughputs.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(1) << uring_throughputs[i];
            }
            std::cout << "],\n";
            std::cout << "   \"normal_io_samples\": [";
            for (size_t i = 0; i < normal_throughputs.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(1) << normal_throughputs[i];
            }
            std::cout << "],\n";
            std::cout << "   \"io_uring_avg\": " << avg_uring << ",\n";
            std::cout << "   \"normal_io_avg\": " << avg_normal << ",\n";
            std::cout << "   \"speedup\": " << std::setprecision(3) << speedup << "}";
            first = false;
        }

        std::cout << std::endl << "]}" << std::endl;
        std::cout << "--- BENCHMARK_DATA_END ---" << std::endl;
    }
};

int main() {
    std::cout << "可靠性高并发IO测试程序" << std::endl;
    std::cout << "========================" << std::endl;

    ReliableConcurrentTest test;

    // 详细测试几个关键并发级别
    test.reliable_concurrent_test(8, 20, 16384, 5);  // 8线程，5轮测试
    test.reliable_concurrent_test(16, 20, 16384, 5); // 16线程，5轮测试

    // 并发级别扫描
    test.concurrency_sweep_test();

    return 0;
}
