#include "matrix.hpp"

Matrix Matrix::transposed() const {
    Matrix result_matrix(total_columns, total_rows);
    for (size_t r = 0; r < total_rows; ++r) {
        for (size_t c = 0; c < total_columns; ++c) {
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

Matrix Matrix::hadamard_inplace(const Matrix& other){
    if (this->rows()!= other.rows() || this->columns()!=other.columns()){
        throw std::invalid_argument("error");
    }
    for (size_t i=0;i<this->size();++i){ 
        (*this)[i]= (*this)[i]*other[i];
    }
    return (*this);
}

Matrix Matrix::operator+ (const Matrix& other){
    if (this->rows()!=other.rows()|| this->columns()!=other.columns()){
        throw std::invalid_argument("error");
    }
    Matrix result_matrix(this->rows(), this->columns());

    for(size_t i=0;i<this->size();++i){ 
        result_matrix[i]= (*this)[i]+other[i];
    }
    return result_matrix;  
}

Matrix Matrix::operator- (const Matrix& other) {
    if (this->rows()!=other.rows()|| this->columns()!=other.columns()){
        throw std::invalid_argument("error");
    }
    Matrix result_matrix(this->rows(), this->columns());

    for(size_t i=0;i<this->size();++i){ 
        result_matrix[i]= (*this)[i]-other[i];
    }
    return result_matrix;  
}

Matrix& Matrix::operator*=(double num) { 
    for(size_t i=0;i<this->size();++i){ 
        (*this)[i] *= num;
    }
    return *this;
}