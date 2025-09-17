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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
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

    // 共享io_uring + 批量处理的正确实现
    TestResult single_test_io_uring_shared_batch(int thread_count, int files_per_thread, size_t file_size, const std::vector<char>& test_data) {
        TestResult result = {0.0, 0, thread_count * files_per_thread};

        auto start_time = std::chrono::high_resolution_clock::now();

        // 共享的io_uring实例
        IoUringIO shared_io_uring;
        if (!shared_io_uring.initialize(thread_count * files_per_thread + 64)) {
            return result;
        }

        // 任务队列和同步机制
        struct IOTask {
            std::string filename;
            std::shared_ptr<std::promise<ssize_t>> promise;
        };

        std::queue<IOTask> task_queue;
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::atomic<bool> stop_flag{false};
        std::atomic<int> completed_tasks{0};
        std::vector<std::string> all_filenames; // 收集所有文件名，用于计时结束后清理
        std::mutex filename_mutex;

        // 批量处理线程
        std::thread batch_processor([&]() {
            const int BATCH_SIZE = 8; // 批量大小

            while (!stop_flag || !task_queue.empty()) {
                std::vector<IOTask> batch;

                // 收集一批任务
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cv.wait_for(lock, std::chrono::milliseconds(1),
                               [&]() { return !task_queue.empty() || stop_flag; });

                    while (!task_queue.empty() && batch.size() < BATCH_SIZE) {
                        batch.push_back(std::move(task_queue.front()));
                        task_queue.pop();
                    }
                }

                if (batch.empty()) continue;

                // 批量处理这些任务
                std::vector<std::string> filenames;
                for (auto& task : batch) {
                    filenames.push_back(task.filename);
                }

                // 使用共享io_uring进行批量写入
                auto batch_results = shared_io_uring.batch_write_files(filenames, test_data.data(), file_size);

                // 设置结果
                for (size_t i = 0; i < batch.size() && i < batch_results.size(); i++) {
                    batch[i].promise->set_value(batch_results[i]);
                    completed_tasks.fetch_add(1);
                }

                // 收集文件名，稍后在计时结束后清理
                {
                    std::lock_guard<std::mutex> lock(filename_mutex);
                    all_filenames.insert(all_filenames.end(), filenames.begin(), filenames.end());
                }
            }
        });

        // 多线程提交任务
        std::vector<std::future<void>> submitters;
        for (int t = 0; t < thread_count; t++) {
            submitters.push_back(std::async(std::launch::async, [=, &task_queue, &queue_mutex, &cv]() {
                for (int f = 0; f < files_per_thread; f++) {
                    std::string filename = "test_shared_batch_t" + std::to_string(t) + "_f" + std::to_string(f) + ".dat";
                    auto promise = std::make_shared<std::promise<ssize_t>>();

                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        task_queue.push({std::move(filename), promise});
                    }
                    cv.notify_one();
                }
            }));
        }

        // 等待所有任务提交完成
        for (auto& submitter : submitters) {
            submitter.wait();
        }

        // 等待所有任务处理完成
        while (completed_tasks.load() < thread_count * files_per_thread) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        stop_flag = true;
        cv.notify_all();
        batch_processor.join();

        result.success_count = completed_tasks.load();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.duration_ms = duration.count() / 1000.0;

        // 在计时结束后清理文件
        for (const auto& filename : all_filenames) {
            unlink(filename.c_str());
        }

        return result;
    }

    // 普通IO多线程测试
    TestResult single_test_normal_io(int thread_count, int files_per_thread, size_t file_size, const std::vector<char>& test_data) {
        TestResult result = {0.0, 0, thread_count * files_per_thread};

        std::vector<std::string> all_filenames; // 收集所有文件名，用于计时结束后清理
        std::mutex filename_mutex;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::future<int>> futures;
        for (int t = 0; t < thread_count; t++) {
            futures.push_back(std::async(std::launch::async, [=, &test_data, &all_filenames, &filename_mutex]() -> int {
                NormalIO normal_io;
                normal_io.initialize();

                int success_count = 0;
                std::vector<std::string> local_filenames; // 本线程的文件名

                for (int f = 0; f < files_per_thread; f++) {
                    std::string filename = "test_normal_t" + std::to_string(t) + "_f" + std::to_string(f) + ".dat";
                    if (normal_io.write_file(filename, test_data.data(), file_size) > 0) {
                        success_count++;
                    }
                    local_filenames.push_back(filename); // 收集文件名，不立即删除
                }

                // 将本线程的文件名添加到全局列表
                {
                    std::lock_guard<std::mutex> lock(filename_mutex);
                    all_filenames.insert(all_filenames.end(), local_filenames.begin(), local_filenames.end());
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

        // 在计时结束后清理文件
        for (const auto& filename : all_filenames) {
            unlink(filename.c_str());
        }

        return result;
    }

public:
    // 多次采样的可靠测试 - 对比共享批量处理 vs 普通IO
    void reliable_concurrent_test(int thread_count, int files_per_thread, size_t file_size, int test_rounds = 5) {
        std::cout << "\n=== 可靠性多线程测试（共享io_uring批量处理 vs 普通IO） ===" << std::endl;
        std::cout << "线程数: " << thread_count
                  << ", 每线程文件数: " << files_per_thread
                  << ", 文件大小: " << file_size / 1024 << "KB"
                  << ", 测试轮数: " << test_rounds << std::endl;

        warmup_system();

        auto test_data = generate_test_data(file_size);

        std::vector<double> uring_shared_times, normal_times;
        std::vector<double> uring_shared_throughputs, normal_throughputs;

        // 多轮测试
        for (int round = 0; round < test_rounds; round++) {
            std::cout << "第 " << (round + 1) << "/" << test_rounds << " 轮测试..." << std::endl;

            // 在每轮之间稍作休息，让系统状态稳定
            if (round > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }

            // 共享io_uring批量处理测试
            auto uring_shared_result = single_test_io_uring_shared_batch(thread_count, files_per_thread, file_size, test_data);
            uring_shared_times.push_back(uring_shared_result.duration_ms);
            uring_shared_throughputs.push_back(uring_shared_result.get_throughput());

            // 中间休息
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // 普通IO测试
            auto normal_result = single_test_normal_io(thread_count, files_per_thread, file_size, test_data);
            normal_times.push_back(normal_result.duration_ms);
            normal_throughputs.push_back(normal_result.get_throughput());

            std::cout << "  共享io_uring批量: " << std::fixed << std::setprecision(1)
                      << uring_shared_result.duration_ms << "ms, "
                      << uring_shared_result.get_throughput() << " ops/sec" << std::endl;
            std::cout << "  普通IO多线程:     " << normal_result.duration_ms << "ms, "
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

        auto [uring_shared_mean_time, uring_shared_median_time, uring_shared_stddev_time] = calculate_stats(uring_shared_times);
        auto [normal_mean_time, normal_median_time, normal_stddev_time] = calculate_stats(normal_times);
        auto [uring_shared_mean_throughput, uring_shared_median_throughput, uring_shared_stddev_throughput] = calculate_stats(uring_shared_throughputs);
        auto [normal_mean_throughput, normal_median_throughput, normal_stddev_throughput] = calculate_stats(normal_throughputs);

        // 显示统计结果
        std::cout << "\n=== 统计结果 ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "共享io_uring批量时间: 平均=" << uring_shared_mean_time << "ms, 中位数=" << uring_shared_median_time
                  << "ms, 标准差=" << uring_shared_stddev_time << "ms" << std::endl;
        std::cout << "普通IO时间:   平均=" << normal_mean_time << "ms, 中位数=" << normal_median_time
                  << "ms, 标准差=" << normal_stddev_time << "ms" << std::endl;

        std::cout << std::setprecision(1);
        std::cout << "共享io_uring批量吞吐量: 平均=" << uring_shared_mean_throughput << " ops/sec, 中位数=" << uring_shared_median_throughput << " ops/sec" << std::endl;
        std::cout << "普通IO多线程吞吐量:     平均=" << normal_mean_throughput << " ops/sec, 中位数=" << normal_median_throughput << " ops/sec" << std::endl;

        // 性能比较
        if (normal_mean_time > 0 && uring_shared_mean_time > 0) {
            double time_speedup = normal_mean_time / uring_shared_mean_time;
            double throughput_speedup = uring_shared_mean_throughput / normal_mean_throughput;

            std::cout << std::setprecision(2);
            std::cout << "平均性能提升: " << time_speedup << "x (基于时间), "
                      << throughput_speedup << "x (基于吞吐量)" << std::endl;

            // 结果一致性检查
            double shared_time_cv = uring_shared_stddev_time / uring_shared_mean_time * 100;
            double normal_time_cv = normal_stddev_time / normal_mean_time * 100;

            std::cout << "\n=== 结果稳定性 ===" << std::endl;
            std::cout << "共享io_uring批量 CV=" << std::setprecision(1) << shared_time_cv << "%" << std::endl;
            std::cout << "普通IO多线程 CV=" << normal_time_cv << "%" << std::endl;

            if (shared_time_cv > 20 || normal_time_cv > 20) {
                std::cout << "⚠️  警告: 结果变异较大，建议增加测试轮数或检查系统负载" << std::endl;
            }
        }
    }

    // 扫描不同并发级别并输出JSON格式数据
    void concurrency_sweep_test() {
        std::cout << "\n=== 并发级别扫描测试 ===" << std::endl;

        std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64};
        const int files_per_thread = 30;
        const size_t file_size = 16 * 1024;

        // 输出JSON格式的数据供Python脚本处理
        std::cout << "\n--- BENCHMARK_DATA_START ---" << std::endl;
        std::cout << "{\"benchmark_results\": [" << std::endl;

        bool first = true;
        for (int threads : thread_counts) {
            // 移除硬件并发限制，允许测试更高的线程数
            std::cout << "测试 " << threads << " 线程..." << std::endl;

            auto test_data = generate_test_data(file_size);

            // 进行5轮测试获取详细数据
            std::vector<double> uring_throughputs, normal_throughputs;

            for (int round = 0; round < 5; round++) {
                auto uring_result = single_test_io_uring_shared_batch(threads, files_per_thread, file_size, test_data);
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

    // 并发级别扫描
    test.concurrency_sweep_test();

    return 0;
}
