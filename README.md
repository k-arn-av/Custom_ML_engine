# Custom ML Engine

A lightweight machine learning engine built from scratch to demonstrate custom neural network architectures, core mathematical operations, and performant backpropagation.

---

## Overview

I am building this project to gain a deep, foundational understanding of core deep learning primitives, automatic differentiation, and low-level matrix operations, implementing a complete Multi-Layer Perceptron (MLP) architecture from scratch without third-party ML dependencies.

---

## Current Status: Active Development

Currently working on the Layers Module- that takes the input matrix, and will have the polymorphic functions feed forward() and feed backward(), then will move to the Dense Layer—that takes the input matrix (X), weight matrix (W), bias (B), weight and bias gradients (dW, dB), and does the feed forward calculations (Z = W·X + B).

This is followed by the Activation Layer—that makes the output matrix non-linear by using the Rectified Linear Unit (ReLU) activation function (max(0, Output)), preventing output simplification and mathematical collapse.

Moving next to the Loss Function—that calculates the initial error gradients required for training the model, followed by the Optimizer Module—that takes the learning rate and actually updates the weights.

Finally, tying it all together with the Network Layer—that manages the layers and runs the full pipeline.


## Highlights

- **Custom Matrix & Linear Algebra Operations:** Optimized memory layouts and custom matrix multiplication algorithms.
- **Data Oriented Approach:** Uses contiguous data structure(std::vector) to maximize CPU cache locality to perform multiple operations at once by taking advantage of SIMD (Single Instruction Multiple Data).
- **Mathematical Optimization & Manual Backpropagation.:** Custom implementation of forward and backward passes.
- **Modular Neural Network Components:** Support for custom layers, activation functions, gradient optimizer and loss functions.
- **Extensibility:** Flexible architecture and optimizer class to add modern gradient optimizers like Momentum, RMSprop, or ADAM (Adaptive Moment Estimation).
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
+---------------v---------------+        +-----------------------+         +--------------------------------+   
|   C++ High-Performance Core   |------->| Matrix, Loss Function |-------->| Optimizer---> Gradient descent |
+-------------------------------+        +--------------v--------+         +---------------v----------------+
                |                                       |                                  |
+---------------v---------------+                       |                                  |
|      Network Layer(Manager)   |<----------------------------------------------------------                     
+-------------------------------+                       |
                |                                       |
+---------------v---------------+                       |
|            Layers             |                       |
+-------------------------------+                       |
                |                                       |
+---------------v---------------+        +--------------v----------------+ 
| Dense Layer, Activation Layer |<------>| Forward Pass, Backpropagation |
+-------------------------------+        +-------------------------------+ 
```

---

