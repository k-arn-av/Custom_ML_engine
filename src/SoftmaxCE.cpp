#include "SoftmaxCE.hpp"
#include <cmath>

double SoftmaxCrossentropy::calculate_loss(const Matrix& final_preactivation, const Matrix& target_matrix){

    //computes softmax function by converting the preactivation elements into fraction using exponents
    //e^(z(i)-max(z))/sum(e^(z(j)-max(z))
    
    saved_probabilities= Matrix(final_preactivation.rows(),final_preactivation.columns());
    double loss=0.0; // scalar loss calculation

    for (size_t r=0; r<final_preactivation.rows();++r){ 

        double maximum=final_preactivation(r,0); // sets initial maximum to first column of each row 

        for (size_t c=1; c<final_preactivation.columns(); ++c){ // first column loop to find the maximum column value in each row

            maximum=std::max(final_preactivation(r,c),maximum);
        }

        double expSum=0.0;

        for (size_t c=0; c<final_preactivation.columns(); ++c){ // second column loop to calculate the sum of all exponents in each row, will be used in denominator

            double expVal=std::exp(final_preactivation(r,c)-maximum);
            saved_probabilities(r,c)= expVal;
            expSum+= expVal;  //sums the exponents of columns/score for each row
        }

        for (size_t c=0; c<final_preactivation.columns(); ++c){ // third column loop to fill the elements in probability matrix with the fraction from softmax

            saved_probabilities(r,c)/=expSum;
        }  

        // LOSS CALCULATION

        for (size_t c=0; c<final_preactivation.columns(); ++c){ // fourth column loop to calculate the sum of all loss in the matrix

            loss-= target_matrix(r,c)*(std::log(saved_probabilities(r,c)+ 1e-15)); //
        }
    }
    
    double batchLoss=loss/double(target_matrix.rows());

    return batchLoss;

}

Matrix SoftmaxCrossentropy::feedbackward(const Matrix& target_matrix){

    Matrix errorMatrix= (saved_probabilities-target_matrix)*(1.0/target_matrix.rows());

    return errorMatrix; // error matrix is the outputgradient dZ for receiving matrices, and inputgradient dX for giving matrices during backpropagation
}






