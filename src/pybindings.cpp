#include <pybind11/pybind11.h>
#include <pybind11/stl.h> //access and auto convert standard library objects
#include "matrix.hpp"
#include "Network.hpp"
#include "optimizer.hpp"
#include "ReluAL.hpp"
#include "DenseLayer.hpp"
#include "SoftmaxCE.hpp"

namespace py= pybind11;

PYBIND11_MODULE(Custom_ML, m){
    m.doc()= "MLP engine";

    //Matrix Class bindings

    py::class_<Matrix>(m, "Matrix")

    //Constructors first
        .def(py::init<>()) //empty matrix constructor
        .def(py::init<size_t, size_t, double>(), //main matrix constructor for calculation
            py::arg("r"), py::arg("c"), py::arg("initial_value")=0.0)
        .def(py::init<size_t, size_t, std::initializer_list<double>>(), //already initialized vector constructor
            py::arg("r"), py::arg("c"), py::arg("user_data"))
        .def(py::init<size_t, size_t, double, double>(), //random weight generator constructor 
            py::arg("r"), py::arg("c"), py::arg("minimum"), py::arg("maximum"))
            

    //getters/index 
        .def("rows", &Matrix::rows)
    ;
}
