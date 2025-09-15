cmake_minimum_required(VERSION 3.18)
project(torch_io_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to disable CUDA
option(USE_CUDA "Enable CUDA support" OFF)

# Set CUDA architectures if CUDA is enabled
if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86")
    enable_language(CUDA)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find PyTorch
# Set environment to handle CUDA-enabled PyTorch on systems without CUDA
set(ENV{CUDA_VISIBLE_DEVICES} "")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Python and pybind11 for Python bindings
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find liburing
pkg_check_modules(LIBURING REQUIRED liburing)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../io_uring_demo)

# Add parent directory's IO implementations
set(PARENT_IO_SOURCES
    ../io_uring_demo/io_uring_io.cpp
    ../io_uring_demo/normal_io.cpp
)

# TensorIO library sources
set(TENSOR_IO_SOURCES
    tensor_io.cpp
    ${PARENT_IO_SOURCES}
)

# Create shared library for TensorIO
add_library(tensor_io SHARED ${TENSOR_IO_SOURCES})

target_link_libraries(tensor_io
    ${TORCH_LIBRARIES}
    ${LIBURING_LIBRARIES}
)

target_include_directories(tensor_io PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${LIBURING_INCLUDE_DIRS}
)

# Set library properties
set_target_properties(tensor_io PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Create Python bindings using pybind11
pybind11_add_module(tensor_io_py
    tensor_io_python.cpp
    ${TENSOR_IO_SOURCES}
)

target_link_libraries(tensor_io_py PRIVATE
    ${TORCH_LIBRARIES}
    ${LIBURING_LIBRARIES}
)

target_include_directories(tensor_io_py PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${LIBURING_INCLUDE_DIRS}
)

# Set Python module properties
set_target_properties(tensor_io_py PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Installation
install(TARGETS tensor_io
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES tensor_io.h
    DESTINATION include
)

# Optional: Create a simple test executable
add_executable(tensor_io_test
    test_tensor_io.cpp
    ${TENSOR_IO_SOURCES}
)

target_link_libraries(tensor_io_test
    ${TORCH_LIBRARIES}
    ${LIBURING_LIBRARIES}
)

target_include_directories(tensor_io_test PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${LIBURING_INCLUDE_DIRS}
)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(tensor_io PRIVATE -Wall -Wextra -O3)
    target_compile_options(tensor_io_py PRIVATE -Wall -Wextra -O3)
    target_compile_options(tensor_io_test PRIVATE -Wall -Wextra -O3)
endif()

# Print some useful information
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "liburing include dirs: ${LIBURING_INCLUDE_DIRS}")
message(STATUS "liburing libraries: ${LIBURING_LIBRARIES}")

# Add custom target for Python setup
add_custom_target(python_setup
    COMMAND ${Python_EXECUTABLE} setup.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Python extensions"
)

# Print build instructions
message(STATUS "")
message(STATUS "Build Instructions:")
message(STATUS "==================")
message(STATUS "1. mkdir build && cd build")
message(STATUS "2. cmake ..")
message(STATUS "3. make -j$(nproc)")
message(STATUS "4. Optional: make python_setup")
message(STATUS "")
message(STATUS "Run Instructions:")
message(STATUS "================")
message(STATUS "1. ./tensor_io_test")
message(STATUS "2. cd .. && python demo_train.py")
message(STATUS "")
