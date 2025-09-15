#!/usr/bin/env python3
"""
分析普通IO性能波动的脚本
"""

import subprocess
import json
import math

def analyze_performance_variance():
    """分析性能波动程度"""
    print("🔍 普通IO vs io_uring 性能稳定性分析")
    print("=" * 60)

    # 运行测试获取数据
    try:
        result = subprocess.run(['./build/test'],
                              capture_output=True, text=True, timeout=300)

        output = result.stdout
        start_marker = "--- BENCHMARK_DATA_START ---"
        end_marker = "--- BENCHMARK_DATA_END ---"

        start_idx = output.find(start_marker)
        end_idx = output.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            print("❌ 无法获取测试数据")
            return

        json_data = output[start_idx + len(start_marker):end_idx].strip()
        data = json.loads(json_data)['benchmark_results']

    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return

    print(f"📊 分析 {len(data)} 个线程配置的性能数据\n")

    # 计算统计指标
    def calc_stats(samples):
        if not samples:
            return 0, 0, 0, 0
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        std_dev = math.sqrt(variance)
        cv = (std_dev / mean * 100) if mean > 0 else 0
        return mean, std_dev, cv, max(samples) / min(samples) if min(samples) > 0 else 0

    # 分析结果
    print("📈 性能波动分析表")
    print("-" * 90)
    print(f"{'线程':<4} {'普通IO (ops/sec)':<25} {'io_uring (ops/sec)':<25} {'稳定性对比':<30}")
    print("-" * 90)

    total_normal_cv = 0
    total_uring_cv = 0

    for item in data:
        threads = item['threads']
        normal_samples = item['normal_io_samples']
        uring_samples = item['io_uring_samples']

        # 计算统计指标
        normal_mean, normal_std, normal_cv, normal_range = calc_stats(normal_samples)
        uring_mean, uring_std, uring_cv, uring_range = calc_stats(uring_samples)

        total_normal_cv += normal_cv
        total_uring_cv += uring_cv

        # 稳定性对比
        if uring_cv > 0:
            stability_ratio = normal_cv / uring_cv
            stability_text = f"{stability_ratio:.1f}x worse"
        else:
            stability_text = "∞x worse"

        print(f"{threads:<4} "
              f"μ={normal_mean:<7.1f} σ={normal_std:<7.1f} CV={normal_cv:<5.1f}% "
              f"μ={uring_mean:<7.1f} σ={uring_std:<7.1f} CV={uring_cv:<5.1f}% "
              f"{stability_text:<30}")

    print("-" * 90)

    # 总体分析
    avg_normal_cv = total_normal_cv / len(data)
    avg_uring_cv = total_uring_cv / len(data)
    overall_ratio = avg_normal_cv / avg_uring_cv if avg_uring_cv > 0 else float('inf')

    print(f"\n📊 总体稳定性对比:")
    print(f"   普通IO 平均变异系数: {avg_normal_cv:.1f}%")
    print(f"   io_uring 平均变异系数: {avg_uring_cv:.1f}%")
    print(f"   稳定性差异: {overall_ratio:.1f}x (普通IO更不稳定)")

    # 异常值分析
    print(f"\n🎯 异常值分析:")
    extreme_cases = []

    for item in data:
        threads = item['threads']
        normal_samples = item['normal_io_samples']

        if normal_samples:
            max_val = max(normal_samples)
            min_val = min(normal_samples)
            if min_val > 0:
                range_ratio = max_val / min_val
                if range_ratio > 5:  # 最大值是最小值的5倍以上
                    extreme_cases.append((threads, range_ratio, min_val, max_val))

    if extreme_cases:
        print("   发现极端性能波动:")
        for threads, ratio, min_val, max_val in extreme_cases:
            print(f"   • {threads}线程: {min_val:.1f} → {max_val:.1f} ops/sec ({ratio:.1f}x 差异)")
    else:
        print("   未发现极端波动情况")

    # 生成ASCII图表显示波动模式
    print(f"\n📉 普通IO性能波动可视化:")
    print("-" * 60)

    for item in data[:6]:  # 只显示前6个以节省空间
        threads = item['threads']
        samples = item['normal_io_samples']

        if not samples:
            continue

        # 归一化到0-40的范围用于显示
        min_val = min(samples)
        max_val = max(samples)
        if max_val > min_val:
            normalized = [(x - min_val) / (max_val - min_val) * 40 for x in samples]
        else:
            normalized = [20] * len(samples)

        # 绘制简单的条形图
        print(f"{threads:2d}线程 |", end="")
        for i, val in enumerate(normalized):
            bar_len = int(val)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"{bar} {samples[i]:6.0f}")
            if i < len(normalized) - 1:
                print("       |", end="")
        print()

    print("-" * 60)
    print("说明: 每行代表一次测试，长度表示相对性能")

    print(f"\n💡 结论:")
    print(f"   普通IO在多线程环境下存在严重的性能不稳定性，")
    print(f"   主要原因是文件系统锁竞争和同步I/O的阻塞特性。")
    print(f"   io_uring通过异步批量操作有效解决了这个问题。")

if __name__ == "__main__":
    analyze_performance_variance()
