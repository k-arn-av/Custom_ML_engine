#pragma once
#include "Layer.hpp"

class Relu: public Layer{ //input matrix becomes preactivation (Z)
    private:
        std::string name;

    public:

        Relu(std::string n): name(std::move(n)){}

        Matrix feedforward(const Matrix& preActivation) override; //Z(i)

        Matrix feedbackward(const Matrix& output_gradient) override;
    
};



