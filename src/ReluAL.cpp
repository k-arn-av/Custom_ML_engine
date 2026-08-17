#include "ReluAL.hpp"
#include <algorithm>

Matrix Relu::feedforward(const Matrix& preActivation){

    input_matrix=preActivation;

    Matrix Activated(preActivation.rows(),preActivation.columns());

    for (int i=0; i< preActivation.size(); ++i){
        Activated[i]=std::max(preActivation[i],0.0);
    }
    return Activated;

}