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

template<typename T>
py::array_t<T> calculate_distance_wrapper(
        int rec_channels, py::array_t<T> input_tensor, float point_dist) {

    py::buffer_info buffer_info = input_tensor.request();

    T *data = static_cast<T *>(buffer_info.ptr);
    std::vector<py::ssize_t> shape = buffer_info.shape;

    std::vector<py::ssize_t> output_shape({shape[1], shape[2], shape[3]});

    Eigen::TensorMap<Eigen::Tensor<T, 4>> input_eigen_tensor(
            data, shape[0], shape[1], shape[2], shape[3]);

    Eigen::Tensor<T, 3> output_tensor = calculate_ligand_distances(
            rec_channels, input_eigen_tensor, point_dist);

    std::vector<py::ssize_t> stride(
            {static_cast<long>(sizeof(T)),
             static_cast<long>(output_shape[0] * sizeof(T)),
             static_cast<long>(output_shape[0] * output_shape[1] * sizeof(T))});

    return py::array_t<T>(output_shape, stride, output_tensor.data());
}

using namespace py::literals;

PYBIND11_MODULE(calculate_distances, m) {
    m.def(
            "calculate_distances", &calculate_distance_wrapper<float>, "rec_channels"_a, "input_tensor"_a.noconvert(),
            "point_dist"_a, py::return_value_policy::automatic);
}