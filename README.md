# Custom ML Engine

A lightweight machine learning engine built from scratch to demonstrate custom neural network architectures, core mathematical operations, and performant backpropagation.

---

## Overview

I am building this project to gain a deep, foundational understanding of core deep learning primitives, manual differentiation, and low-level matrix operations, implementing a complete Multi-Layer Perceptron (MLP) architecture from scratch without third-party ML dependencies.

---

## Current Status: Active Development

Currently Working on **Activation Layer module**; finished Matrix, Layer and Dense Layer Modules. 

The project started with Matrix Module, that covers every operations necessary for Linear Calculations. It uses a std::vector container to store the 2D matrix into a flat 1D array, to boost CPU cache locality. I have ensured that all operations are math friendly using operator function, like Mat(r,c) to get the data on that row and column or A*B for matrix multiplication. This module changed from using nested vectors to a 1D vector, adding more needed operations like sum_columns(), allowed for weight matrix random creation using mt19937 engine and calculated weight range using Xavier's Uniform Initialization.

Moving on the next module, Layers Module- that takes the input matrix, and will have the polymorphic functions feed forward() and feed backward(), then will move to the Dense Layer—that takes the input matrix (X), weight matrix (W), bias (B), weight and bias gradients (dW, dB), and does the feed forward calculations (Z = W·X + B).

This is followed by the Activation Layer—that makes the output matrix non-linear by using the Rectified Linear Unit (ReLU) activation function (max(0, Output)), preventing output simplification and mathematical collapse.

A Softmax Activation Layer is added as the final activation layer after the last Dense Layer, which uses softmax activation function (exponential fraction conversion) which is passed down to the loss function. 

Moving next to the Loss Function (Cross Entropy)—that calculates the initial error gradients required for training the model, followed by the Optimizer Module—that takes the learning rate and actually updates the weights.

Finally, tying it all together with the Network Layer—that manages the layers and runs the full pipeline.

--- 

## Highlights

- **Custom Matrix & Linear Algebra Operations:** Optimized memory layouts and custom matrix multiplication algorithms.
- **Data Oriented Approach:** Uses contiguous data structure(std::vector) to maximize CPU cache locality to perform multiple operations at once by taking advantage of SIMD (Single Instruction Multiple Data).
- **Mathematical Optimization & Manual Backpropagation.:** Custom implementation of forward and backward passes.
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

