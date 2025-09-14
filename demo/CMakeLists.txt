cmake_minimum_required(VERSION 3.10)
project(IOComparison)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# 查找线程库
find_package(Threads REQUIRED)

# 使用PkgConfig查找liburing（更准确的版本控制）
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBURING REQUIRED liburing>=2.0)
    set(URING_LIB ${LIBURING_LIBRARIES})
    set(URING_INCLUDE_DIRS ${LIBURING_INCLUDE_DIRS})
    message(STATUS "liburing版本: ${LIBURING_VERSION}")
    message(STATUS "liburing库路径: ${LIBURING_LIBRARY_DIRS}")
else()
    # 备用方案：直接查找库文件
    find_library(URING_LIB uring)
    if(NOT URING_LIB)
        message(FATAL_ERROR "liburing not found. Please install liburing-dev (Ubuntu/Debian) or liburing-devel (RedHat/CentOS)")
    endif()
    message(WARNING "无法获取liburing版本信息，请确保版本>=2.0")
endif()

# 检查头文件
include(CheckIncludeFileCXX)
check_include_file_cxx("liburing.h" HAVE_LIBURING_H)
if(NOT HAVE_LIBURING_H)
    message(FATAL_ERROR "liburing.h not found. Please install liburing development headers")
endif()

# 检查内核版本
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND uname -r
        OUTPUT_VARIABLE KERNEL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "内核版本: ${KERNEL_VERSION}")

    # 简单检查内核版本是否>=5.1（io_uring最低要求）
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" KERNEL_VERSION_MATCH ${KERNEL_VERSION})
    set(KERNEL_MAJOR ${CMAKE_MATCH_1})
    set(KERNEL_MINOR ${CMAKE_MATCH_2})

    if(KERNEL_MAJOR LESS 5 OR (KERNEL_MAJOR EQUAL 5 AND KERNEL_MINOR LESS 1))
        message(FATAL_ERROR "io_uring需要内核版本>=5.1，当前版本: ${KERNEL_VERSION}")
    endif()
endif()

# 添加可执行文件
add_executable(io_comparison
    main.cpp
    io_uring_io.cpp
    normal_io.cpp
)

# 添加高并发测试可执行文件
add_executable(concurrent_test
    io_uring_io.cpp
    normal_io.cpp
)

# 添加简化高并发测试 (已删除)
# add_executable(simple_concurrent
#     simple_concurrent_test.cpp
#     io_uring_io.cpp
#     normal_io.cpp
# )

# 添加IO性能测试程序
add_executable(test
    test.cpp
    io_uring_io.cpp
    normal_io.cpp
)

# 链接库
target_link_libraries(io_comparison
    ${URING_LIB}
    Threads::Threads
)

target_link_libraries(concurrent_test
    ${URING_LIB}
    Threads::Threads
)

# target_link_libraries(simple_concurrent
#     ${URING_LIB}
#     Threads::Threads
# )

target_link_libraries(test
    ${URING_LIB}
    Threads::Threads
)

# 设置包含目录
target_include_directories(io_comparison PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(concurrent_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(simple_concurrent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 如果使用pkg-config，添加包含目录和链接目录
if(PKG_CONFIG_FOUND AND LIBURING_FOUND)
    target_include_directories(io_comparison PRIVATE ${URING_INCLUDE_DIRS})
    target_include_directories(concurrent_test PRIVATE ${URING_INCLUDE_DIRS})
    # target_include_directories(simple_concurrent PRIVATE ${URING_INCLUDE_DIRS})
    target_include_directories(test PRIVATE ${URING_INCLUDE_DIRS})
    target_link_directories(io_comparison PRIVATE ${LIBURING_LIBRARY_DIRS})
    target_link_directories(concurrent_test PRIVATE ${LIBURING_LIBRARY_DIRS})
    # target_link_directories(simple_concurrent PRIVATE ${LIBURING_LIBRARY_DIRS})
    target_link_directories(test PRIVATE ${LIBURING_LIBRARY_DIRS})
endif()

# 打印编译信息
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "liburing found at: ${URING_LIB}")

# 安装规则（可选）
install(TARGETS io_comparison concurrent_test test
    RUNTIME DESTINATION bin
)

# 添加一些有用的编译目标
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMENT "Cleaning all build files"
)

# 查找clang-format
find_program(CLANG_FORMAT_EXE
    NAMES clang-format clang-format-10 clang-format-11 clang-format-12 clang-format-13 clang-format-14
    DOC "Path to clang-format executable"
)

# 格式化目标
if(CLANG_FORMAT_EXE)
    # 获取所有源文件
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    )

    # 过滤掉构建目录的文件
    list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX "${CMAKE_BINARY_DIR}/.*")

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting all source files with clang-format"
        VERBATIM
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} -style=file --dry-run --Werror ${ALL_SOURCE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking code formatting with clang-format"
        VERBATIM
    )

    message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
    message(STATUS "Format targets added: 'format' and 'format-check'")
else()
    message(WARNING "clang-format not found, format target will not be available")
endif()

# 检查系统是否支持io_uring (Linux 5.1+)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Building for Linux - io_uring should be available")
else()
    message(WARNING "This project requires Linux for io_uring support")
endif()
