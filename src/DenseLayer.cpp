#include "DenseLayer.hpp"

Matrix DenseLayer::feedforward(const Matrix& input){

    input_matrix=input; 

    Matrix output_matrix= (input_matrix * W) + B; // Main feedforward calculation, Z = X*W + B, with X as batch x features

    return output_matrix;
}

Matrix DenseLayer::feedbackward(const Matrix& output_gradient){

    dW= input_matrix.transposed() * output_gradient; // dW = X^T * dZ

    dB= output_gradient.sum_rows(); // dB = sum across batch rows for each output neuron, giving a 1 x neurons row vector

    return output_gradient * W.transposed(); //returns input gradient dX that becomes the output gradient for back layer, dX = dZ * W^T
}
