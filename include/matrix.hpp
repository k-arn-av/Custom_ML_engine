#pragma once
#include <vector>
#include <stdexcept>
#include <random>

class Matrix{

    private:
        size_t total_rows;
        size_t total_columns;
        std::vector<double>elements;//contiguous array for efficient processing; promotes DoD

    public:
        //constructors: 
        //one that takes size and initial value only to make a matrix instantly, used for linear calculations
        //second that fills in the input matrix data for later use when feedforward, uses initializer list to bypass vector initialization repitition
        //third that creates a matrix with random numbers from the given range min and max using Mersenne Twister engine
        //fourth that creates an empty matrix for initialization without value declaration
        Matrix(size_t r, size_t c, double initial_val=0.0): total_rows(r), total_columns(c), elements(r*c,initial_val){}

        Matrix(size_t r, size_t c, std::initializer_list<double> user_data): total_rows(r), total_columns(c), elements(user_data){
            if (elements.size() != r * c){
                throw std::invalid_argument("Error: Initialization list size does not match matrix dimensions.");
            }
        }

        Matrix(size_t r, size_t c, double minimum, double maximum): total_rows(r), total_columns(c), elements(r*c){
            //static locks the local object creation (engine) in the memory, so whenever the function is called, the object remains the same, just its range is changed
            static thread_local std::mt19937 engine(std::random_device{}());// static in a local function preserves the variable and locks the initialization. can only be destroyed when the thread processing it ends (main()), 
                                                                            // thread local ensures the engine runs for each call separately (locally), preventing thread mixups while multithreading (processing multiple batches at once)
            std::uniform_real_distribution<double> distribute(minimum, maximum);// distributes double numbers from the given range for each object construction

            for (size_t i=0; i<elements.size(); ++i){
                elements[i]=distribute(engine);
            }
            
        }

        Matrix(): total_rows(0), total_columns(0), elements(){}

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

        //Matrix operations

        Matrix sum_columns()const;

        Matrix sum_rows()const;

        Matrix transposed()const; //transposes the this matrix

        static Matrix hadamard(const Matrix& first_matrix, const Matrix& second_matrix); // symmetrical design; Matrix::hadamard(A,B)

        Matrix& hadamard_inplace(const Matrix& other); //changes this matrix in place

        Matrix operator *(const Matrix& other) const; // Matrix Multiplication

        Matrix operator *(const double num) const; // Scalar Multiplication

        Matrix& operator *=(double num); // Scalar Multiplication in place

        Matrix operator +(const Matrix& other) const; // Matrix Addition

        Matrix operator -(const Matrix& other) const; // Matrix difference

};