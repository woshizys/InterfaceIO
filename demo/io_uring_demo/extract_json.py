import re, json
with open('test_results_shared_batch.txt', 'r') as f: content = f.read()
patterns = {
    'threads': r'\"threads\":\s*(\d+)',
    'io_uring_samples': r'\"io_uring_samples\":\s*\[([0-9., ]+)\]',
    'normal_io_samples': r'\"normal_io_samples\":\s*\[([0-9., ]+)\]',
    'io_uring_avg': r'\"io_uring_avg\":\s*([0-9.]+)',
    'normal_io_avg': r'\"normal_io_avg\":\s*([0-9.]+)',
    'speedup': r'\"speedup\":\s*([0-9.]+)'
}
data = {k: re.findall(v, content) for k, v in patterns.items()}
if all(len(v) == len(data['threads']) for v in data.values()):
    results = []
    for i in range(len(data['threads'])):
        results.append({
            'threads': int(data['threads'][i]),
            'io_uring_samples': [float(x.strip()) for x in data['io_uring_samples'][i].split(',')],
            'normal_io_samples': [float(x.strip()) for x in data['normal_io_samples'][i].split(',')],
            'io_uring_avg': float(data['io_uring_avg'][i]),
            'normal_io_avg': float(data['normal_io_avg'][i]),
            'speedup': float(data['speedup'][i])
        })
    with open('benchmark_data_shared_batch.json', 'w') as f:
        json.dump({'benchmark_results': results}, f, indent=2)
    print(f'✅ JSON数据提取成功，包含{len(results)}个测试结果')
else:
    print('❌ 数据提取失败')
