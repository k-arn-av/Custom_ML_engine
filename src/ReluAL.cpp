#include "ReluAL.hpp"
#include <algorithm>

Matrix Relu::feedforward(const Matrix& preActivation){
    input_matrix = preActivation;

    Matrix activated(preActivation.rows(), preActivation.columns());
    for (size_t i = 0; i < preActivation.size(); ++i){
        activated[i] = std::max(preActivation[i], 0.0);
    }
    return activated;
}

Matrix Relu::feedbackward(const Matrix& output_gradient){
    Matrix input_gradient(output_gradient.rows(), output_gradient.columns());

    for (size_t i = 0; i < output_gradient.size(); ++i){
        if (input_matrix[i] > 0.0){
            input_gradient[i] = output_gradient[i];
        }
        else{
            input_gradient[i] = 0.0;
        }
    }

    return input_gradient;
}