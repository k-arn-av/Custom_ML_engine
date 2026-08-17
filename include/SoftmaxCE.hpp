#pragma once
#include "matrix.hpp"

class SoftmaxCrossentropy{

    private:
        Matrix saved_probabilities;

    public:
        double calculate_loss(const Matrix& final_preactivation, const Matrix& target_matrix);

        Matrix feedbackward(const Matrix& target_matrix);

};
