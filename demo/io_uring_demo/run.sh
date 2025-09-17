#!/bin/bash
g++ -std=c++17 -O2 -o test test.cpp io_uring_io.cpp normal_io.cpp -luring -lpthread
./test > test_results_shared_batch.txt 2>&1
python3 extract_json.py
python3 generate_png_chart.py
