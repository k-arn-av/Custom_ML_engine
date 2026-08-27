#include <pybind11/pybind11.h>
#include <pybind11/stl.h> //access and auto convert standard library objects
#include <pybind11/operators.h> //access operators

#include "matrix.hpp"
#include "Network.hpp"
#include "optimizer.hpp"
#include "Layer.hpp"
#include "ReluAL.hpp"
#include "DenseLayer.hpp"
#include "SoftmaxCE.hpp"

namespace py= pybind11; //nanobind uses namespace py= nanobind (I only found out about nanobind after completing this)


//Basic Syntax for pybind11: 

// .def(py::init<(datatype)>(), py::arg("(argument)"))-> constructor
// .def("functionName", &class:: function)-> normal function call where cpp and py sync 
// .def("functionName", [](parameters){return},py::arg("") (or anything extra))-> lambda function to act where py and cpp arent in sync (uniqueptr, return by reference, element acessing (dunders), template varidic)

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
        .def(py::self*= double()); //in place scalar matmul
    
    //Layer abstract base class bindings
    py::class_<Layer>(m, "Layer") //layer doesnt have virtual or abstraction lock virtual=0
        
        .def("feedforward", &Layer::feedforward, py::arg("input"))
        .def("feedbackward", &Layer::feedbackward, py::arg("output_gradient"))

        .def("getWeight_Bias", &Layer::getWeight_Bias)
        .def("get_gradients", &Layer::get_gradients);
        

    //Denselayer bindings (inheritance syntax)
    py::class_<DenseLayer, Layer>(m, "DenseLayer")
        
        .def(py::init<std::string, size_t, size_t>(),
            py::arg("n"), py::arg("neurons"), py::arg("input_features"))
        
        //overload cast to return non const Weights since py cannot distinct between const and non const same functions
        .def("Weights", py::overload_cast<> (&DenseLayer::Weights))
        .def("Bias", py::overload_cast<> (&DenseLayer::Bias))

        .def("getName", &DenseLayer::getName)

        .def("weightGradients", &DenseLayer::weightGradients)
        .def("biasGradients", &DenseLayer::biasGradients);
        
    //Relu bindings
    py::class_<Relu, Layer>(m, "Relu")

        .def(py::init<std::string>(),
            py::arg("n"))
        
        .def("getName", &Relu::getName);
        
    
    //SoftmaxCE loss layer bindings
    py::class_<SoftmaxCrossentropy>(m, "SoftmaxCE")

        .def("calculate_loss", &SoftmaxCrossentropy::calculate_loss)
        .def("feedbackward", &SoftmaxCrossentropy::feedbackward);
    

    //Optimizer abstract base class bindings
    py::class_<optimizer>(m, "optimizer")

       .def("GradientDescent", &optimizer::GradientDescent);
    

    //Vanilla SGD bindings
    py::class_<SGD, optimizer>(m, "SGD")
        
        .def(py::init<double>(),
            py::arg("LR"));
    
    
    //Network bindings
    py::class_<Network>(m, "Network")

        .def(py::init<>())

        //set optimizer Lambda middleman function to imitate unique ptr in py 
        .def("setOptimizer", [](Network& self, std::unique_ptr<optimizer> opt){
            self.setOptimizer(std::move(opt));
        }, py::arg("optimizer"))

        //return reference
        //create a middleman lambda function since template is blueprint only, and python requires cpp compiler to complile exact create_layer during cmake build
        .def("create_DenseLayer", [](Network& self, const std::string& name, size_t neurons, size_t features)->DenseLayer& {
            return self.create_layer<DenseLayer>(name, neurons, features);
        }, py::return_value_policy::reference_internal) // prohibits python to delete the literal layer from Layers vector inside Network, and lets Network handle it
        
        .def("create_ReluLayer", [](Network& self, const std::string& name)->Relu& {
            return self.create_layer<Relu>(name);

        }, py::return_value_policy::reference_internal)
        

        //train model binding
        .def("trainModel", &Network::train_model,
            py::arg("input_matrix"), py::arg("target"));

}
