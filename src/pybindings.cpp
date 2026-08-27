#include <pybind11/pybind11.h>
#include <pybind11/stl.h> //access and auto convert standard library objects
#include <pybind11/operators.h> //access operators

#include "matrix.hpp"
#include "Network.hpp"
#include "optimizer.hpp"
#include "ReluAL.hpp"
#include "DenseLayer.hpp"
#include "SoftmaxCE.hpp"

namespace py= pybind11; //nanobind uses namespace py= nanobind (I only found out about nanobind after completing this)

PYBIND11_MODULE(Custom_ML, m){ //python will import Custom_ML
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
            
    //elements and index access
        .def("getIndex", &Matrix::getIndex, py::arg("row"),py::arg("col"))
        .def("findRow", &Matrix::findRow, py::arg("givenIndex"))
        .def("findCol", &Matrix::findCol, py::arg("givenIndex"))

    //2D accessors mat(r,c), lambda functions to access directly since no way to get or set values in py since cpp uses memory, py calls dunders __function__, which lambda mimicks
        //mat[r,c] in python to get read only access idx is index
        .def("__getitem__", [](const Matrix& self, std::pair<size_t,size_t> idx){ 
            return self(idx.first, idx.second);
        })
        //mat[r,c]= val in py to write access 
        .def("__setitem__", [](Matrix& self, std::pair<size_t,size_t> idx, double val){
            self(idx.first, idx.second)=val;
        })
    
    //1D accessors mat[index]

        .def("__getitem__", [](const Matrix& self, size_t idx){
            return self[idx];
        })
        .def("__setitem__", [](Matrix& self, size_t idx, double val){
            self[idx]=val;
        })
        
    //getters
        .def("rows", &Matrix::rows) //gets the address of Matrix class (&Matrix) to fetch rows getter
        .def("columns", &Matrix::columns)
        .def("size", &Matrix::size)

    //Operations
        .def("sum_columns", &Matrix::sum_columns)
        .def("sum_rows", &Matrix::sum_rows)
        .def("transposed", &Matrix::transposed)
        .def("hadamard", &Matrix::hadamard, py::arg("first_matrix"), py::arg("second_matrix"))
        .def("hadamard_inplace", &Matrix::hadamard_inplace, py::arg("other"))

    //Operator 
        .def(py::self+ py::self) // py auto calls __add__
        .def(py::self- py::self)
        .def(py::self* py::self) //matmul
        .def(py::self* double()) //scalar matmul
        .def(double()* py::self)  //reverse scalar matmul
        .def(py::self*= double()) //in place scalar matmul
    ;
}
