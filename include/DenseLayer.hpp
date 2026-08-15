#pragma once
#include "Layer.hpp"
#include <iostream>

class DenseLayer: public Layer{

    private:
        Matrix W; // Weight Matrix, generated random at first, just know the neurons and input features size. (r*c)
        Matrix dW;// Weight Gradient Matrix, calculated later
        Matrix B; // Bias column Matrix, changes the entire space in coordinate axes, meaning giving gradient descent required parameters to minimize the loss
        Matrix dB;// Bias Gradient Matrix, calculated later

    public:
        DenseLayer(std::string name, size_t neurons, size_t input_features): Layer(name),
        W(neurons, input_features, -std::sqrt(6.0/(input_features+neurons)), std::sqrt(6.0/(input_features+neurons))),
        // range calculation using Xavier Uniform Initialization that calculates boundaries using variance formula: a^2/3= 2/input_features+neurons where a is the given boundary, weights should be +- equal range
        // this formula ensures the boundaries set are correct for given rows and columns (neurons and features) so that the weights are in a suitable range to train the model
        dW(neurons, input_features), 
        B(neurons, 1), // Bias is zero matrix at first. 
        dB(neurons, 1){}

        Matrix feedforward(const Matrix& input) override{
            input_matrix=input; 
            Matrix output_matrix= (W*input_matrix) + B; // Main feedforward calculation, Z= W.X+B
            return output_matrix;
        }
        
        Matrix feedbackward(const Matrix& output_gradient) override{
            //calculate weight and bias gradients
            input_matrix.transposed();
            dW= output_gradient* input_matrix;
           
        }

        

};