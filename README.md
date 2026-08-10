# Custom ML Engine

A lightweight machine learning engine built from scratch to demonstrate custom neural network architectures, core mathematical operations, and performant backpropagation.

---

## Overview

I am building this project to gain a deep, foundational understanding of core deep learning primitives, automatic differentiation, and low-level matrix operations—implementing a complete Multi-Layer Perceptron (MLP) architecture from scratch without third-party ML dependencies.

---

## Highlights

- **Custom Matrix & Linear Algebra Operations:** Optimized memory layouts and custom matrix multiplication algorithms.
- **Data Oriented Approach:** Uses contiguous data structure(std::vector) to maximize CPU cache locality to perform multiple operations at once by taking advantage of SIMD (Single Instruction Multiple Data).
- **Backpropagation & Automatic Differentiation:** Custom implementation of forward and backward passes.
- **Modular Neural Network Components:** Support for custom layers, activation functions, gradient optimizer and loss functions.
- **Extensibility** Flexible architecture and optimizer class to add modern gradient optimizers like Momentum, RMSprop, or ADAM (Adaptive Moment Estimation).
- **Hybrid Performance:** High-level interfacing alongside low-level performance tuning.

---

## Hierarchy

```
+-------------------------------+
|  Python High-Level Interface  |
+-------------------------------+
                |
+---------------v---------------+
|       pybind11 interop        |
+-------------------------------+
                |
+---------------v---------------+        +----------------------------------+  
|   C++ High-Performance Core   |------->| Matrix, Optimizer, Loss Function |
+-------------------------------+        +------------------v---------------+
                |                                           |
+---------------v---------------+                           |
|      Network Layer(Manager)   |                           |
+-------------------------------+                           |
                |                                           |
+---------------v---------------+                           |
|            Layers             |                           |
+-------------------------------+                           |
                |                                           |
+---------------v---------------+        +------------------v------------------------------+ 
| Dense Layer, Activation Layer |<------>| Forward Pass, Backpropagation, Gradient Descent |
+-------------------------------+        +-------------------------------------------------+ 
```

---

