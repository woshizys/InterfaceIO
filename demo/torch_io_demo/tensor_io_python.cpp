#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include "tensor_io.h"

namespace py = pybind11;

/**
 * PyBind11绑定文件，将C++ TensorIO接口暴露给Python
 */

PYBIND11_MODULE(tensor_io_py, m) {
    m.doc() = "PyTorch TensorIO Python bindings - High-performance IO backend for PyTorch";

    // 绑定IOBackend枚举
    py::enum_<TensorIO::IOBackend>(m, "IOBackend")
        .value("NORMAL", TensorIO::IOBackend::NORMAL, "Normal file IO backend")
        .value("IO_URING", TensorIO::IOBackend::IO_URING, "io_uring async IO backend")
        .export_values();

    // 绑定TensorIO类
    py::class_<TensorIO>(m, "TensorIO")
        .def(py::init<TensorIO::IOBackend>(),
             py::arg("backend") = TensorIO::IOBackend::IO_URING,
             "Initialize TensorIO with specified backend")

        .def("initialize", &TensorIO::initialize,
             "Initialize the IO backend")

        .def("load_tensor_from_file", &TensorIO::load_tensor_from_file,
             py::arg("filename"), py::arg("shape"), py::arg("dtype") = torch::kFloat32,
             "Load tensor from binary file")

        .def("save_tensor_to_file", &TensorIO::save_tensor_to_file,
             py::arg("tensor"), py::arg("filename"),
             "Save tensor to binary file")

        .def("batch_load_tensors", &TensorIO::batch_load_tensors,
             py::arg("filenames"), py::arg("shapes"), py::arg("dtype") = torch::kFloat32,
             "Batch load multiple tensors")

        .def("benchmark_io_performance", &TensorIO::benchmark_io_performance,
             py::arg("test_file"), py::arg("file_size"), py::arg("iterations") = 10,
             "Run IO performance benchmark")

        .def("get_backend", &TensorIO::get_backend,
             "Get current IO backend type");

    // 添加一些实用函数
    m.def("create_tensor_io", [](const std::string& backend_name) {
        TensorIO::IOBackend backend;
        if (backend_name == "io_uring") {
            backend = TensorIO::IOBackend::IO_URING;
        } else if (backend_name == "normal") {
            backend = TensorIO::IOBackend::NORMAL;
        } else {
            throw std::invalid_argument("Unknown backend: " + backend_name +
                                      ". Use 'io_uring' or 'normal'.");
        }

        auto tensor_io = std::make_unique<TensorIO>(backend);
        if (!tensor_io->initialize()) {
            throw std::runtime_error("Failed to initialize TensorIO backend: " + backend_name);
        }

        return tensor_io.release();
    }, py::arg("backend_name"),
       py::return_value_policy::take_ownership,
       "Create and initialize TensorIO instance by backend name");

    m.def("get_supported_backends", []() {
        return std::vector<std::string>{"normal", "io_uring"};
    }, "Get list of supported IO backend names");

    m.def("torch_dtype_to_string", [](torch::ScalarType dtype) {
        switch (dtype) {
            case torch::kFloat32: return std::string("float32");
            case torch::kFloat64: return std::string("float64");
            case torch::kInt32: return std::string("int32");
            case torch::kInt64: return std::string("int64");
            case torch::kInt8: return std::string("int8");
            case torch::kUInt8: return std::string("uint8");
            case torch::kInt16: return std::string("int16");
            case torch::kBool: return std::string("bool");
            default: return std::string("unknown");
        }
    }, py::arg("dtype"), "Convert PyTorch dtype to string");

    // 版本信息
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "TensorIO Development Team";

    // 添加一些有用的常量
    m.attr("DEFAULT_BACKEND") = "io_uring";
    m.attr("FALLBACK_BACKEND") = "normal";
}
