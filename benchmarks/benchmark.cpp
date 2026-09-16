// Benchmark: Contiguous 1D std::vector (row-major) vs Nested std::vector<std::vector<double>>
//
// Compares cache behavior and runtime for matrix multiplication, transposition,
// and Hadamard (element-wise) product. Uses the same r-k-c loop order as the
// engine's Matrix::operator* implementation.
//
// Build:
//   g++ -O3 -DNDEBUG -std=c++23 -g benchmarks/benchmark.cpp -o benchmark
//
// Run:
//   ./benchmark contiguous 512 10 matmul
//   ./benchmark nested 512 10 matmul
//   ./benchmark contiguous 512 10 transpose
//   ./benchmark nested 512 10 transpose
//   ./benchmark contiguous 512 10 hadamard
//   ./benchmark nested 512 10 hadamard
//
// Profile with Cachegrind:
//   valgrind --tool=cachegrind --cache-sim=yes \
//     --I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64 \
//     --cachegrind-out-file=cg_cont_matmul.out \
//     ./benchmark contiguous 512 10 matmul
//
//   cg_annotate cg_cont_matmul.out

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

// ============================================================================
// ContiguousMatrix: 1D std::vector with row-major indexing (matches engine)
// ============================================================================
class ContiguousMatrix {
public:
    ContiguousMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    double& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols_ + col];
    }
    const double& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols_ + col];
    }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

// ============================================================================
// NestedMatrix: std::vector<std::vector<double>> (baseline for comparison)
// ============================================================================
class NestedMatrix {
public:
    NestedMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {}

    double& operator()(std::size_t row, std::size_t col) {
        return data_[row][col];
    }
    const double& operator()(std::size_t row, std::size_t col) const {
        return data_[row][col];
    }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::vector<double>> data_;
};

// ============================================================================
// Operations — same r-k-c loop order as the engine's Matrix::operator*
// ============================================================================
template <typename MatrixA, typename MatrixB, typename MatrixC>
void multiply_rkc(const MatrixA& a, const MatrixB& b, MatrixC& c) {
    for (std::size_t r = 0; r < a.rows(); ++r) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            double a_rk = a(r, k);
            for (std::size_t c_idx = 0; c_idx < b.cols(); ++c_idx) {
                c(r, c_idx) += a_rk * b(k, c_idx);
            }
        }
    }
}

template <typename Matrix>
void transpose(const Matrix& src, Matrix& dst) {
    for (std::size_t r = 0; r < src.rows(); ++r) {
        for (std::size_t c = 0; c < src.cols(); ++c) {
            dst(c, r) = src(r, c);
        }
    }
}

template <typename MatrixA, typename MatrixB, typename MatrixC>
void hadamard(const MatrixA& a, const MatrixB& b, MatrixC& c) {
    for (std::size_t r = 0; r < a.rows(); ++r) {
        for (std::size_t c_idx = 0; c_idx < a.cols(); ++c_idx) {
            c(r, c_idx) = a(r, c_idx) * b(r, c_idx);
        }
    }
}

template <typename Matrix>
void fill_matrix(Matrix& m, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t r = 0; r < m.rows(); ++r)
        for (std::size_t c = 0; c < m.cols(); ++c)
            m(r, c) = dist(rng);
}

// ============================================================================
// Main — run one layout + one operation per invocation
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <contiguous|nested> <N> <reps> [matmul|transpose|hadamard|all]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::size_t n = std::stoul(argv[2]);
    int reps = std::stoi(argv[3]);
    std::string op = argc > 4 ? argv[4] : "all";

    std::mt19937 rng(42);

    if (mode == "contiguous") {
        ContiguousMatrix a(n, n), b(n, n), c(n, n), t(n, n);
        fill_matrix(a, rng);
        fill_matrix(b, rng);

        if (op == "matmul" || op == "all") {
            multiply_rkc(a, b, c);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        c(i, j) = 0.0;
                multiply_rkc(a, b, c);
            }
            auto end = Clock::now();
            volatile double ck = c(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "CONTIGUOUS matmul " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
        if (op == "transpose" || op == "all") {
            transpose(a, t);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                transpose(a, t);
            }
            auto end = Clock::now();
            volatile double ck = t(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "CONTIGUOUS transpose " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
        if (op == "hadamard" || op == "all") {
            hadamard(a, b, c);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                hadamard(a, b, c);
            }
            auto end = Clock::now();
            volatile double ck = c(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "CONTIGUOUS hadamard " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
    }
    else if (mode == "nested") {
        NestedMatrix a(n, n), b(n, n), c(n, n), t(n, n);
        fill_matrix(a, rng);
        fill_matrix(b, rng);

        if (op == "matmul" || op == "all") {
            multiply_rkc(a, b, c);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        c(i, j) = 0.0;
                multiply_rkc(a, b, c);
            }
            auto end = Clock::now();
            volatile double ck = c(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "NESTED matmul " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
        if (op == "transpose" || op == "all") {
            transpose(a, t);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                transpose(a, t);
            }
            auto end = Clock::now();
            volatile double ck = t(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "NESTED transpose " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
        if (op == "hadamard" || op == "all") {
            hadamard(a, b, c);
            auto start = Clock::now();
            for (int run = 0; run < reps; ++run) {
                hadamard(a, b, c);
            }
            auto end = Clock::now();
            volatile double ck = c(0, 0);
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "NESTED hadamard " << n << "x" << n << " x" << reps
                      << " | " << ms << " ms | ck=" << ck << "\n";
        }
    }
    else {
        std::cout << "Unknown mode: " << mode << "\n";
        return 1;
    }

    return 0;
}
