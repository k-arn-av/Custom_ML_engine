#include "ReluAL.hpp"
#include <algorithm>

Matrix Relu::feedforward(const Matrix& preActivation){ //RELU max function if items are less than 0, converts them to 0, else leaves them unchanged

    input_matrix=preActivation;

    Matrix Activated(preActivation.rows(),preActivation.columns());

    for (size_t i=0; i< preActivation.size(); ++i){

        Activated[i]=std::max(preActivation[i],0.0);
    }

    return Activated;

}

Matrix Relu::feedbackward(const Matrix& output_gradient){ //Simplified Hadamard product between output gradient and first derivative of preactivation (input_matrix Z)

    Matrix input_gradient(output_gradient.rows(), output_gradient.columns()); 

    for (size_t i=0; i<output_gradient.size(); ++i){

        if (input_matrix[i]>0){
            input_gradient[i]=output_gradient[i]; // Since first derivative of preactivation (Z) can only be 1.0 if element>0 or 0.0 if element<=0,
        }                                         // the gradient can either have the unchanged output gradient values, or 0.0 if the Z matrix element is negative or zero.

        else{
            input_gradient[i]=0.0;
        }
    }

    return input_gradient;
}