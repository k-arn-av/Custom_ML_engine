#include <vector>
#include <stdexcept>

class Matrix{
    private:
        size_t total_rows;
        size_t total_columns;
        std::vector<double>elements;//contiguous array for efficient processing; promotes DoD

    public:
        Matrix(size_t R, size_t C, double initial_val=0.0): total_rows(R), total_columns(C), elements(R*C,initial_val){}

        //elements and index access for row/col and index; essential for array element storage

        size_t getIndex(size_t row, size_t col)const { return (row * total_columns)+col;} //row-major order for index tracking, more efficient to prevent cache miss

        size_t findRow(size_t givenIndex)const { return givenIndex / total_columns;}

        size_t findCol(size_t givenIndex)const { return givenIndex % total_columns;}

        //2D Accessors

        double& operator()(size_t row, size_t col) { //returns reference to be able to change the value
            return elements[getIndex(row,col)]; //users call Mat(r,c) to access element
        }
        
        const double& operator()(size_t row, size_t col) const { 
            return elements[getIndex(row,col)]; //read only access
        }

        //1d accessors
        const double& operator[](size_t givenIndex) const{
            return elements[givenIndex];
        }
        double& operator[](size_t givenIndex){
            return elements[givenIndex]; // users call Mat[index] to get element
        }
        //getters
        size_t rows() const{
            return total_rows;
        }
        size_t columns() const{
            return total_columns;
        }
        size_t size() const{
            return elements.size();
        }

        Matrix transposed() const{};

        static Matrix hadamard(const Matrix& first_matrix, const Matrix& second_matrix){};

        // static class multiplication that takes two matrices
        static Matrix multiply(const Matrix& first_matrix, const Matrix& second_matrix){
            if(first_matrix.total_columns != second_matrix.total_rows)
                throw std::invalid_argument("Matrix multiply: inner dimensions must match");
            Matrix out(first_matrix.total_rows, second_matrix.total_columns);
            for(size_t r=0;r<out.total_rows;++r){
                for(size_t c=0;c<out.total_columns;++c){
                    double sum = 0.0;
                    for(size_t k=0;k<first_matrix.total_columns;++k)
                        sum += first_matrix(r,k) * second_matrix(k,c);
                    out(r,c) = sum;
                }
            }
            return out;
        }

        Matrix scalar_multiplication(double n)const {};






        







};

// non-member operator* for readability; forwards to the class static `multiply`
inline Matrix operator*(const Matrix& a, const Matrix& b){
    return Matrix::multiply(a,b);
}