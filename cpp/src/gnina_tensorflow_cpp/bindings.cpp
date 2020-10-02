// @author: scantleb
// @created: 29/09/2020
// @brief: PyBind11 interface functions.
//
// calculate_distance_wrapper is the interface between data stored in python
// numpy arrays and the calculate_distance function, which takes and returns
// Eigen::Tensor objects.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "calculate_distances.h"

namespace py = pybind11;
using namespace py::literals;

template<typename T>
py::array_t<T> calculate_distance_wrapper(
        int rec_channels, py::array_t<T> input_tensor, float point_dist) {

    py::buffer_info buffer_info = input_tensor.request();

    T *data = static_cast<T *>(buffer_info.ptr);

    std::vector<py::ssize_t> input_shape = buffer_info.shape;
    std::vector<py::ssize_t> output_shape = input_shape;

    Eigen::TensorMap<Eigen::Tensor<T, 5>> input_eigen_tensor(
            data, input_shape[0], input_shape[1], input_shape[2],
            input_shape[3], input_shape[4]);

    Eigen::Tensor<T, 5> output_tensor = calculate_ligand_distances(
            rec_channels, input_eigen_tensor, point_dist);

    std::vector<py::ssize_t> stride(
            {static_cast<long>(sizeof(T)),
             static_cast<long>(output_shape[0] * sizeof(T)),
             static_cast<long>(output_shape[0] * output_shape[1] * sizeof(T)),
             static_cast<long>(output_shape[0] * output_shape[1] *
                               output_shape[2] * sizeof(T)),
             static_cast<long>(output_shape[0] * output_shape[1] *
                               output_shape[2] * output_shape[3] * sizeof(T))});

    return py::array_t<T>(output_shape, stride, output_tensor.data());
}

PYBIND11_MODULE(gnina_tensorflow_cpp, m) {
    m.def("calculate_distances", &calculate_distance_wrapper<float>,
          "rec_channels"_a, "input_tensor"_a.noconvert(),
          "point_dist"_a, py::return_value_policy::automatic);
}