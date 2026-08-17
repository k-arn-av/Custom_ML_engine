#include "DenseLayer.hpp"

Matrix DenseLayer::feedforward(const Matrix& input){

    input_matrix=input; 

    Matrix output_matrix= (W*input_matrix) + B; // Main feedforward calculation, Z= W.X+B

    return output_matrix;
}

Matrix DenseLayer::feedforward(const Matrix& output_gradient){

    dW= output_gradient* input_matrix.transposed(); // dW= dZ* X^T

    dB= output_gradient.sum_columns(); //dB= sum of all columns for each row making the matrix dB(neuron,1) with total rows neurons and column 1

    return output_gradient * W.transposed(); //returns input gradient dX that becomes the output gradient for back layer, dX= W^T * dZ    
}
