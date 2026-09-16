# Custom ML Engine

A lightweight machine learning engine built from scratch to demonstrate custom neural network architectures, core mathematical operations, and performant backpropagation.

---

## Overview

I am building/updating this project to gain a deep, foundational understanding of core deep learning primitives, manual differentiation, and low-level matrix operations, implementing a complete Multi-Layer Perceptron (MLP) architecture from scratch without third-party ML dependencies. I will keep updating and improving this engine to be able to perform categorization.

---

## Current Status: Active Development

**XOR Classification** (2→8→2 MLP with ReLU + Softmax-CE) is working and converges with a learning rate of 0.05. **Profiling** complete with Cachegrind and Memcheck — see the [Benchmarks & Profiling](#benchmarks--profiling) section below.

**Next:** Multi-class classification support. 

**Algorithm for the Project**

The project started with Matrix Module, that covers every operations necessary for Linear Calculations. It uses a std::vector container to store the 2D matrix into a flat 1D array, to boost CPU cache locality. I have ensured that all operations are math friendly using operator function, like Mat(r,c) to get the data on that row and column or A*B for matrix multiplication. This module changed from using nested vectors to a 1D vector, adding more needed operations like sum_columns(), allowed for weight matrix random creation using mt19937 engine and calculated weight range using Xavier's Uniform Initialization.

Moving on the next module, Layers Module- that takes the input matrix, and will have the polymorphic functions feed forward() and feed backward(), then will move to the Dense Layer—that takes the input matrix (X), weight matrix (W), bias (B), weight and bias gradients (dW, dB), and does the feed forward calculations (Z = W·X + B).

This is followed by the Activation Layer—that makes the output matrix non-linear by using the Rectified Linear Unit (ReLU) activation function (max(0, Output)), preventing output simplification and mathematical collapse.

Next comes the loss function (Cross Entropy)- that is combined with Softmax Activation function (exponential fraction conversion, converting the nogits/preactivation into probabilities) to calculate the scalar loss that will be used by the optimizer module. For backpropagation, this module calculates the error Matrix (dX/dZ) to feed backward and acts as a starting point for the output and input gradient pass down. Main reason to combine both softmax activation and loss function CE is because the equations by softmax fed into loss function cancels out each other later to form a simple formula similar to Mean Squared Error (MSE), 1/batches(prediction- target). So using this formula will make the engine less prone to errors during exponent and logarithmic calculations.

The loss from the loss module (SoftmaxCE) is then fed into the optimizer matrix, which then calculates the new weights for denselayers, and updates them. For this, network manager has to place each created layers sequentially for the optimizer to loop through them. Starting with vanilla gradient descent (SGD), I will update the optimizer later for momentum and ADAM to work.

Network Manager owns all the layers, and manages them sequentially. 

**Model Training** is done in python/ main.py via pybind11.

--- 

## Highlights

- **Custom Matrix & Linear Algebra Operations:** Optimized memory layouts and custom matrix multiplication algorithms. Uses row-major order of matrices for maximum performance, where rows are neurons/batches, and columns are features.
- **Data Oriented Approach:** Uses contiguous data structure(std::vector) to maximize CPU cache locality to perform multiple operations at once by taking advantage of SIMD (Single Instruction Multiple Data).
- **Mathematical Optimization & Manual Backpropagation:** Custom implementation of forward and backward passes.
- **Modular Neural Network Components:** Support for custom layers, activation functions, gradient optimizer and loss functions.
- **Extensibility:** Flexible architecture and optimizer class to add modern gradient optimizers like Momentum, RMSprop, or ADAM (Adaptive Moment Estimation).
- **Hybrid Performance:** High-level interfacing alongside low-level performance tuning.

---

## Hierarchy/Blueprint

```
Multi-Class Classification/ Pattern Recognition Based MLP

+-------------------------------+
|  Python High-Level Interface  |
+-------------------------------+
                |
+---------------v---------------+
|       pybind11 interop        |
+-------------------------------+
                |
+---------------v---------------+        +---------------+            
|   C++ High-Performance Core   |------->|     Matrix    |
+-------------------------------+        +--------v------+         
                |                                 |                                  
+---------------v---------------+                 |                                  
|      Network Layer(Manager)   |                 |
+-------------------------------+                 |
                |                                 |
+---------------v---------------+                 |
|            Layers             |<----------------<----------------------------------------------------------------------------------------------------------------------- 
+-------------------------------+                                                                                                                                        |
       | -------------------------------------------------------|                                                                                                        |
+------v------+  Forward Pass(FP)>, <Backpropagation(BP) +------v----------------+  FP,BP  +--------------------------------------------------------+                    |
| Dense Layer | <--------------------------------------> | ReLU Activation Layer |<------->| Final Dense Layer Output(Z)-->Softmax Activation Layer |                    |
+-------------+                                          +-----------------------+         +----------------------------^---------------------------+                    |
                                                                                                                       | Fraction conversion, Backpropagation Begin      |
                                                                                                +----------------------v---------------------+                           |     
                                                                                                |  Prediction-->Loss Function, Cross Entropy |                           |               
                                                                                                +----------------------^---------------------+                           |
                                                                                                                       |                                                 |
                                                                                                        +--------------v-----------------+              SGD              |  
                                                                                                        | Optimizer---> Gradient descent |--------------------------------
                                                                                                        +--------------------------------+


```

---

## Benchmarks & Profiling

The Matrix module's core design decision — a contiguous 1D `std::vector` with row-major indexing instead of `std::vector<std::vector<double>>` — was validated with a standalone benchmark and Valgrind Cachegrind (a cache-behavior simulator, since hardware performance counters aren't reliably available in every environment).

**Method:** Same algorithm, same loop order (r-k-c for matrix multiplication), same data — only the storage layout changes between the contiguous and nested implementations.

**Headline result:** the contiguous layout produced **82.7% fewer last-level cache misses** than the nested-vector baseline on 1024×1024 matrix multiplication, where the matrix data exceeds L1 cache capacity. On matrix transposition (used in backpropagation's `dW = X^T * dZ`), the contiguous layout showed **24.1% fewer L1 data-cache misses**.

| Operation | Metric | Contiguous | Nested | Reduction |
|---|---|---:|---:|---:|
| 1024×1024 matmul | LL misses | 19.7M | 114.1M | -82.7% |
| 512×512 transpose | D1 misses | 3.46M | 4.56M | -24.1% |
| 512×512 matmul | LL misses | 142,815 | 186,919 | -23.6% |

Zero memory leaks were also confirmed in the C++ engine via Valgrind Memcheck (`PYTHONMALLOC=malloc`) across a 200-epoch training run (37,547 allocations, 0 bytes definitely lost).

See [`benchmarks/`](benchmarks/) for the benchmark source and instructions to reproduce, and [`docs/profiling-results.md`](docs/profiling-results.md) for the full results.

---

## Future Upgrades in Mind

- **BitNet b1.58 (first choice):** Removing floating point multiplications entirely with primitive int_8t datatypes doing addition and subtraction. Significantly reduces the load in CPU. Weights are in the range [-1,0,1] (1.58 bits total) instead of double (8 bytes each). Maximizes SIMD (AVX2 CPU component) capacity through multi simple calculations.
  
- **CUDA kernels (second choice):** Enables the GPU to perform SIMT (Single Instruction Multiple Threads) for multi Tensor calculations (floating point multiplications) that CPU cannot fast enough. Significantly reduces the training time.
  
- **nanobind:** Python bindings especially for modern C++ and Python. Extremely lightweight library compared to pybind11, and compiles 3x faster.

- **Optimizers:** Momentum and RMSprop/ ADAM optimizer to fix weights and learning rate for each step in the training process (Gradient Descent upgrade), but is heavier than vanilla SGD (higher processing power needed).
  
- **Transformer Architecture (first choice):** The final goal to convert this MLP engine to a Large Language Model, combining multiple MLPS and performing the same architecture (mathematical). Uses backpropagation and Gradient Descent to reach the target.

- **NeuroEvolution Architecture (second choice):** Another choice to make generational Model that comes from reproduction (not literal) from two parent neural networks to be able to reach the target.
