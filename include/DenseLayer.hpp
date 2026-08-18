#pragma once
#include "Layer.hpp"

class DenseLayer: public Layer{

    private:
        std::string name;
        Matrix W; // Weight Matrix, generated random at first, just know the neurons and input features size. (r*c)
        Matrix dW;// Weight Gradient Matrix, calculated later
        Matrix B; // Bias column Matrix, changes the entire space in coordinate axes, meaning giving gradient descent required parameters to minimize the loss
        Matrix dB;// Bias Gradient Matrix, calculated later

    public:
        DenseLayer(std::string n, size_t neurons, size_t input_features): name(std::move(n)),
        W(input_features, neurons, -std::sqrt(6.0/(input_features+neurons)), std::sqrt(6.0/(input_features+neurons))),
        // range calculation using Xavier Uniform Initialization that calculates boundaries using variance formula: a^2/3= 2/input_features+neurons where a is the given boundary, weights should be +- equal range
        // this formula ensures the boundaries set are correct for given rows and columns (features and neurons) so that the weights are in a suitable range to train the model
        dW(input_features, neurons), 
        B(1, neurons), // Bias is zero row vector: one value per output neuron.
        dB(1, neurons){}

        Matrix feedforward(const Matrix& input) override;
        
        Matrix feedbackward(const Matrix& output_gradient) override;

        //getters and setters
        Matrix& Weights() {return W;}
        Matrix& Bias() {return B;}

        const std::string& getName() const{return name;}

        const Matrix& Weights() const {return W;}
        const Matrix& Bias() const {return B;}

        const Matrix &weightGradients() const {return dW;}
        const Matrix &biasGradients() const {return dB;}
        
        std::vector<Matrix*> getWeight_Bias()override {
        return {&W, &B}; // Hand over memory addresses of W and B
        }

        std::vector<Matrix*> get_gradients() override {
            return {&dW, &dB}; // Hand over memory addresses of the gradients
        }

};