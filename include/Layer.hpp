#pragma once
#include "matrix.hpp"
//abstract base class Layer
class Layer{
    protected:
        Matrix input;
        std::string name;
    
    public:
        virtual ~Layer()=default;

        virtual Matrix feedforward(const Matrix& input)=0; //putting =0 in virtual makes the base class an abstract base class, 
                                                            // meaning you can never create its objects like Layer layer1(). input passed should be assigned to the input matrix X
        virtual Matrix feedbackward(const Matrix& output_gradient)=0;

        std::string getName(){return name;}
};
