#include "matrix.hpp"

Matrix Matrix::sum_columns()const {
    Matrix result_matrix(this->rows(),this->columns());
    for (size_t r=0; r< this->rows(); ++r){
        double sum=0.0;
        for (size_t c=0; c< this->columns(); ++c){
            sum+=(*this)(r,c);
        }
        result_matrix(r,0)= sum;
    }
    return result_matrix;
}
    
Matrix Matrix::transposed()const {
    Matrix result_matrix(this->columns(), this->rows());
    for (size_t r = 0; r < total_rows; ++r){
        for (size_t c = 0; c < total_columns; ++c){
            result_matrix(c, r) = (*this)(r, c);
        }
    }
    return result_matrix;
}

Matrix Matrix::hadamard(const Matrix& first_matrix, const Matrix& second_matrix){
    if (first_matrix.rows()!= second_matrix.rows() || first_matrix.columns()!=second_matrix.columns()){
        throw std::invalid_argument("error");
    }
    Matrix result_matrix(first_matrix.rows(),first_matrix.columns());
    for (size_t i=0;i<first_matrix.size();++i){ 
        result_matrix[i]= first_matrix[i]*second_matrix[i];
    }
    return result_matrix;
}

Matrix& Matrix::hadamard_inplace(const Matrix& other){
    if (this->rows()!= other.rows() || this->columns()!=other.columns()){
        throw std::invalid_argument("error");
    }
    for (size_t i=0;i<this->size();++i){ 
        (*this)[i]= (*this)[i]*other[i];
    }
    return (*this);
}

Matrix Matrix::operator+ (const Matrix& other)const {

    if (this->rows() == other.rows() && this->columns() == other.columns()){ // Normal Matrix addition for same sized matrices

        Matrix result_matrix(this->rows(), this->columns());

        for(size_t i=0;i<this->size();++i){ 
            result_matrix[i]= (*this)[i]+other[i];
        }
        return result_matrix;  
    }

    if (this->rows() == other.rows() && other.columns() == 1){ // specifically for bias addition (broadcasting) Z=WX+B where bias is column matrix, WX is this, while B is other
         
        Matrix result_matrix(this->rows(), this->columns());

        for (size_t r=0; r< this->rows(); ++r){
            for (size_t c=0; c< this->columns(); ++c){
                result_matrix(r,c)= (*this)(r,c)+ other[r];
            }
        }
        return result_matrix;
    }

    throw std::invalid_argument("error");
}

Matrix Matrix::operator- (const Matrix& other)const {
    if (this->rows()!=other.rows()|| this->columns()!=other.columns()){
        throw std::invalid_argument("error");
    }
    Matrix result_matrix(this->rows(), this->columns());

    for(size_t i=0;i<this->size();++i){ 
        result_matrix[i]= (*this)[i]-other[i];
    }
    return result_matrix;  
}

Matrix& Matrix::operator*= (double num) { 
    for(size_t i=0;i<this->size();++i){ 
        (*this)[i] *= num;
    }
    return *this;
}

Matrix Matrix::operator*(double num)const {
    Matrix result_matrix(this->rows(),this->columns());
    for(size_t i=0;i<this->size();++i){ 
        result_matrix[i]= num*(*this)[i];
    }
    return result_matrix;
}

// Matrix multiplication using r-k-c loop, because traversing by column every time is
// more efficient than traversing by rows due to cpu cache locality (chunks of elements processing at once). Traversing by row is 
// the worst, since cpu has to jump from one random space to another, and doing it at every step causes cache miss
// k is the common row and column for first and second matrices, and k only increases row-by-row

Matrix Matrix::operator* (const Matrix& other)const {
    if (this->columns()!=other.rows()){
        throw std::invalid_argument("error");
    }
    Matrix result_matrix(this->rows(),other.columns());

    for (size_t r=0;r< this->rows(); ++r){
        for(size_t k =0;k< this->columns(); ++k){
            double save_val= (*this)(r,k);
            for(size_t c=0; c< other.columns(); ++c){
                
                result_matrix(r,c)+= save_val *other(k,c);//used k, because notice that the col of A and row of B is iterating first, so replace col A and row B with k

            }
        }   
    }
    return result_matrix;

}