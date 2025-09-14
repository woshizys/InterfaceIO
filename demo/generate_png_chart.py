#!/usr/bin/env python3
"""
生成PNG格式的IO性能对比图表
使用Python PIL/Pillow库进行图像绘制
"""

import subprocess
import json
import sys
import os
import math

def run_benchmark():
    """运行C++测试程序并解析结果"""
    try:
        print("正在运行5轮性能测试，请稍等...")
        result = subprocess.run(['./build/test'],
                              cwd='/home/zys/interfaceIO/demo',
                              capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"测试程序执行失败: {result.stderr}")
            return None

        output = result.stdout

        # 查找JSON数据
        start_marker = "--- BENCHMARK_DATA_START ---"
        end_marker = "--- BENCHMARK_DATA_END ---"

        start_idx = output.find(start_marker)
        end_idx = output.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            print("未找到基准测试数据")
            return None

        json_data = output[start_idx + len(start_marker):end_idx].strip()

        try:
            data = json.loads(json_data)
            return data['benchmark_results']
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None

    except subprocess.TimeoutExpired:
        print("测试程序执行超时")
        return None
    except Exception as e:
        print(f"执行错误: {e}")
        return None

def install_pillow():
    """尝试安装PIL/Pillow库"""
    try:
        # 先尝试使用系统包管理器
        print("尝试安装python3-pil...")
        result = subprocess.run(['sudo', 'apt', 'install', 'python3-pil', '-y'],
                               input='zys\n', text=True, capture_output=True, timeout=120)
        if result.returncode == 0:
            print("✅ 成功通过apt安装python3-pil")
            return True
    except Exception as e:
        print(f"apt安装失败: {e}")

    # 如果系统包管理器失败，尝试pip3
    try:
        print("尝试使用pip3安装pillow...")
        subprocess.run([sys.executable, '-c', 'import pip; pip.main(["install", "--user", "pillow"])'],
                      timeout=300)
        return True
    except Exception as e:
        print(f"pip3安装失败: {e}")
        return False

def create_png_charts(data, output_file='io_performance_charts.png'):
    """创建PNG格式的性能图表"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("❌ PIL/Pillow库未安装，正在尝试安装...")
        if not install_pillow():
            print("❌ 无法安装PIL/Pillow库")
            return False

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("❌ PIL/Pillow库安装失败")
            return False

    # 图表尺寸和设置
    width, height = 1200, 800
    bg_color = (255, 255, 255)  # 白色背景

    # 创建图像
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 尝试加载字体
    try:
        # 尝试使用系统默认字体
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        # 如果没有TrueType字体，使用默认字体
        try:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_title = font_label = font_small = None

    # 提取数据
    thread_counts = [item['threads'] for item in data]
    uring_avgs = [item['io_uring_avg'] for item in data]
    normal_avgs = [item['normal_io_avg'] for item in data]
    speedups = [item['speedup'] for item in data]

    # 图表布局
    margin = 80
    chart_width = width - 2 * margin
    chart_height = height - 150

    # 绘制标题
    title = "IO_URING vs Normal IO Performance Comparison"
    if font_title:
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((width - title_width) // 2, 20), title, fill=(0, 0, 0), font=font_title)
    else:
        draw.text((width // 2 - 200, 20), title, fill=(0, 0, 0))

    # 分成两个图表区域
    chart1_y = 80  # 上半部分：性能对比
    chart2_y = 400 # 下半部分：加速比
    chart_h = 250  # 每个图表的高度

    # === 图表1: 性能对比 ===
    max_perf = max(max(uring_avgs), max(normal_avgs))
    min_perf = min(min(uring_avgs), min(normal_avgs))

    # 使用对数缩放
    if min_perf > 0:
        log_min = math.log10(min_perf)
        log_max = math.log10(max_perf)
        log_range = log_max - log_min
    else:
        log_min = 0
        log_max = math.log10(max_perf)
        log_range = log_max

    # 绘制图表1的背景和网格
    chart1_rect = [margin, chart1_y, margin + chart_width, chart1_y + chart_h]
    draw.rectangle(chart1_rect, outline=(0, 0, 0), width=2)

    # 绘制网格线
    for i in range(1, 5):
        y = chart1_y + i * chart_h // 5
        draw.line([margin, y, margin + chart_width, y], fill=(200, 200, 200))

    # 绘制数据点和线
    x_step = chart_width // (len(thread_counts) - 1) if len(thread_counts) > 1 else chart_width // 2

    uring_points = []
    normal_points = []

    for i, (threads, uring_perf, normal_perf) in enumerate(zip(thread_counts, uring_avgs, normal_avgs)):
        x = margin + i * x_step

        # 计算Y坐标（对数缩放）
        if log_range > 0:
            uring_y = chart1_y + chart_h - int((math.log10(uring_perf) - log_min) / log_range * chart_h)
            normal_y = chart1_y + chart_h - int((math.log10(normal_perf) - log_min) / log_range * chart_h)
        else:
            uring_y = chart1_y + chart_h // 2
            normal_y = chart1_y + chart_h // 2

        uring_points.append((x, uring_y))
        normal_points.append((x, normal_y))

        # 绘制数据点
        draw.ellipse([x-4, uring_y-4, x+4, uring_y+4], fill=(46, 134, 193))  # io_uring蓝色
        draw.ellipse([x-4, normal_y-4, x+4, normal_y+4], fill=(231, 76, 60))  # normal IO红色

        # 绘制X轴标签
        if font_small:
            draw.text((x-10, chart1_y + chart_h + 5), str(threads), fill=(0, 0, 0), font=font_small)

    # 绘制连接线
    if len(uring_points) > 1:
        for i in range(len(uring_points) - 1):
            draw.line([uring_points[i], uring_points[i+1]], fill=(46, 134, 193), width=2)
            draw.line([normal_points[i], normal_points[i+1]], fill=(231, 76, 60), width=2)

    # 图表1标签
    if font_label:
        draw.text((margin, chart1_y - 25), "Throughput (ops/sec, log scale)", fill=(0, 0, 0), font=font_label)
        draw.text((margin + chart_width // 2 - 50, chart1_y + chart_h + 25), "Thread Count", fill=(0, 0, 0), font=font_label)

    # 图例
    legend_y = chart1_y + 20
    draw.rectangle([margin + chart_width - 200, legend_y, margin + chart_width - 50, legend_y + 40],
                  outline=(0, 0, 0), fill=(245, 245, 245))
    draw.ellipse([margin + chart_width - 190, legend_y + 8, margin + chart_width - 182, legend_y + 16], fill=(46, 134, 193))
    draw.ellipse([margin + chart_width - 190, legend_y + 24, margin + chart_width - 182, legend_y + 32], fill=(231, 76, 60))
    if font_small:
        draw.text((margin + chart_width - 175, legend_y + 5), "io_uring", fill=(0, 0, 0), font=font_small)
        draw.text((margin + chart_width - 175, legend_y + 21), "Normal IO", fill=(0, 0, 0), font=font_small)

    # === 图表2: 加速比 ===
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    speedup_range = max_speedup - min_speedup
    if speedup_range == 0:
        speedup_range = 1

    # 绘制图表2的背景
    chart2_rect = [margin, chart2_y, margin + chart_width, chart2_y + chart_h]
    draw.rectangle(chart2_rect, outline=(0, 0, 0), width=2)

    # 绘制基准线 (1.0x)
    baseline_y = chart2_y + chart_h - int((1.0 - min_speedup) / speedup_range * chart_h)
    draw.line([margin, baseline_y, margin + chart_width, baseline_y], fill=(255, 0, 0), width=2)
    if font_small:
        draw.text((margin + chart_width + 5, baseline_y - 8), "1.0x", fill=(255, 0, 0), font=font_small)

    # 绘制加速比柱状图
    bar_width = chart_width // len(speedups) - 10

    for i, (threads, speedup) in enumerate(zip(thread_counts, speedups)):
        x = margin + i * (chart_width // len(speedups)) + 5
        bar_height = int((speedup - min_speedup) / speedup_range * chart_h)
        bar_y = chart2_y + chart_h - bar_height

        # 选择颜色
        if speedup >= 2.0:
            color = (39, 174, 96)  # 绿色：显著优势
        elif speedup >= 1.0:
            color = (52, 152, 219)  # 蓝色：轻微优势
        else:
            color = (231, 76, 60)  # 红色：劣势

        # 绘制柱子
        draw.rectangle([x, bar_y, x + bar_width, chart2_y + chart_h], fill=color, outline=(0, 0, 0))

        # 在柱子上方显示数值
        if font_small:
            speedup_text = f"{speedup:.1f}x"
            text_bbox = draw.textbbox((0, 0), speedup_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((x + (bar_width - text_width) // 2, bar_y - 15), speedup_text, fill=(0, 0, 0), font=font_small)

        # 绘制X轴标签
        if font_small:
            draw.text((x + bar_width // 2 - 10, chart2_y + chart_h + 5), str(threads), fill=(0, 0, 0), font=font_small)

    # 图表2标签
    if font_label:
        draw.text((margin, chart2_y - 25), "Speedup Ratio (io_uring / Normal IO)", fill=(0, 0, 0), font=font_label)
        draw.text((margin + chart_width // 2 - 50, chart2_y + chart_h + 25), "Thread Count", fill=(0, 0, 0), font=font_label)

    # 统计信息
    avg_speedup = sum(speedups) / len(speedups)
    max_speedup_threads = thread_counts[speedups.index(max(speedups))]
    stats_text = f"Average Speedup: {avg_speedup:.2f}x | Max: {max(speedups):.2f}x (@{max_speedup_threads} threads)"

    if font_small:
        draw.text((margin, height - 40), stats_text, fill=(0, 0, 0), font=font_small)

    # 保存图片
    try:
        img.save(output_file, 'PNG', quality=95)
        print(f"✅ PNG图表已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 保存PNG图表失败: {e}")
        return False

def main():
    print("🎨 PNG图表生成器")
    print("=" * 50)

    # 检查是否存在测试程序
    if not os.path.exists('/home/zys/interfaceIO/demo/build/test'):
        print("❌ 错误: 测试程序不存在，请先编译")
        sys.exit(1)

    # 运行基准测试
    benchmark_data = run_benchmark()

    if benchmark_data is None:
        print("❌ 无法获取基准测试数据")
        sys.exit(1)

    if not benchmark_data:
        print("❌ 基准测试数据为空")
        sys.exit(1)

    print(f"✅ 成功获取 {len(benchmark_data)} 个线程配置的数据，每个配置5组测试")

    # 生成PNG图表
    output_file = '/home/zys/interfaceIO/demo/io_performance_comparison.png'
    if create_png_charts(benchmark_data, output_file):
        print(f"\n🎉 PNG性能图表生成成功！")
        print(f"📁 文件位置: {output_file}")
        print(f"📊 图表内容:")
        print(f"   • 上半部分: io_uring vs Normal IO 性能对比曲线")
        print(f"   • 下半部分: io_uring 加速比柱状图")
        print(f"   • 分辨率: 1200x800 像素")
        print(f"   • 格式: PNG，适合插入PPT")

        # 显示文件信息
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📦 文件大小: {file_size/1024:.1f} KB")
    else:
        print("❌ PNG图表生成失败")

if __name__ == "__main__":
    main()
