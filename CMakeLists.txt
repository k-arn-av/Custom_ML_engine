cmake_minimum_required(VERSION 3.20)

#Project Name and Language
project(Custom_ML LANGUAGES CXX)

#modern c++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#NOTE: IMPORTED PYTHON AND PYBIND11 MANUALLY FROM PATH BECAUSE CMAKE COULDNT FIND THEM (SHUCKS), WHICH CAUSED ERROR USING PYBIND11
#use the Python installation containing Python.h and pybind11.
set(Python_ROOT_DIR "C:/Users/arnav/AppData/Local/Python/pythoncore-3.14-64")
set(Python_EXECUTABLE "${Python_ROOT_DIR}/python.exe")
set(pybind11_DIR "${Python_ROOT_DIR}/Lib/site-packages/pybind11/share/cmake/pybind11")


#find Python.h and the Python module target used by pybind11.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

#prefer the local package; use FetchContent only if it is unavailable.
find_package(pybind11 CONFIG QUIET)

#allow for external project use to fetch required library (pybind11) directly from internet so that the local device doesn't need to download pip manually
if(NOT pybind11_FOUND)
    message(STATUS "Local pybind11 not found. Fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.1
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Found local pybind11: ${pybind11_DIR}")
endif()

#collect all source files from the src directory
set(SOURCES
    src/DenseLayer.cpp
    src/matrix.cpp
    src/Network.cpp
    src/optimizer.cpp
    src/pybindings.cpp
    src/ReluAL.cpp
    src/SoftmaxCE.cpp
)

#define Pybind11 Extension Module (.pyd on Windows)
pybind11_add_module(Custom_ML ${SOURCES})

#tell CMake where to find header files (.hpp) in include/, use target for modern CPP isolation
target_include_directories(Custom_ML PRIVATE include)

#optional Compiler Warnings
if(MSVC)
    target_compile_options(Custom_ML PRIVATE /W4)
else()
    target_compile_options(Custom_ML PRIVATE -Wall -Wextra -Wpedantic)
endif()
