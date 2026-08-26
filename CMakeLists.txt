cmake_minimum_required(VERSION 3.20)

# Project Name and Language
project(Custom_ML LANGUAGES CXX)

#modern c++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#pybind
find_package(pybind REQUIRED)

# Collect all source files from the src directory
set(SOURCES
    src/DenseLayer.cpp
    src/matrix.cpp
    src/Network.cpp
    src/optimizer.cpp
    src/ReluAL.cpp
    src/SoftmaxCE.cpp
)

# Define Pybind11 Extension Module (.pyd on Windows)
pybind11_add_module(Custom_ML ${SOURCES})

# Tell CMake where to find header files (.hpp) in include/, use target for modern CPP isolation
target_include_directories(Custom_ML PRIVATE include)

# Optional Compiler Warnings
if(MSVC)
    target_compile_options(Custom_ML PRIVATE /W4)
else()
    target_compile_options(Custom_ML PRIVATE -Wall -Wextra -Wpedantic)
endif()
