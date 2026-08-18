#pragma once
#include "matrix.hpp"
//abstract base class Layer
class Layer{

    protected:
        Matrix input_matrix; // every layer has input matrix, that needs to be saved for later chain rule gradient calculation (dW, dB)
        
    
    public:
        virtual ~Layer()=default; //virtual destructor necessary when we are using <Layer> pointer = <child Layer> object later in the manager(Network)

        virtual Matrix feedforward(const Matrix& input)=0; //putting =0 in virtual makes the base class an abstract base class, 
                                                            // meaning you can never create its objects like Layer layer1(). input passed should be assigned to the input matrix X
        virtual Matrix feedbackward(const Matrix& output_gradient)=0;

        //virtual functions that need to filter out weights and biases for Layers needed for optimizer, Denselayer overrides it, while ReLU returns nothing
        virtual std::vector<Matrix*> getWeight_Bias(){return {};}
        virtual std::vector<Matrix*> get_gradients() { return {};}

};
