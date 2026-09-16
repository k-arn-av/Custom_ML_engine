# Benchmark: Contiguous vs Nested Vector Storage

This benchmark compares two matrix storage layouts to validate the cache-locality design choice in the Custom ML engine:

- **Contiguous**: 1D `std::vector<double>` with row-major indexing (`data[row * cols + col]`) — matches the engine's `Matrix` class
- **Nested**: `std::vector<std::vector<double>>` — each row is a separate heap allocation

Both implementations use the same r-k-c loop order, same algorithm, same data, and same compiler flags. Only the storage layout differs.

## Operations tested

| Operation | Access pattern | Why it matters |
|---|---|---|
| Matrix multiplication (r-k-c) | Row-wise in both A and B | Core engine operation |
| Transposition | Column-wise read, row-wise write | Used in backpropagation (`dW = X^T * dZ`) |
| Hadamard (element-wise) product | Sequential | Used in ReLU backward pass |

## Build

```bash
g++ -O3 -DNDEBUG -std=c++23 -g benchmarks/benchmark.cpp -o benchmark
```

For Valgrind/Cachegrind compatibility, omit `-march=native` (Valgrind cannot handle AVX instructions).

## Run timing

```bash
# Matrix multiplication
./benchmark contiguous 512 10 matmul
./benchmark nested 512 10 matmul

# Transposition
./benchmark contiguous 512 10 transpose
./benchmark nested 512 10 transpose

# Hadamard product
./benchmark contiguous 512 10 hadamard
./benchmark nested 512 10 hadamard
```

## Profile with Cachegrind

```bash
valgrind --tool=cachegrind --cache-sim=yes \
  --I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64 \
  --cachegrind-out-file=cg_cont_matmul.out \
  ./benchmark contiguous 512 10 matmul

cg_annotate cg_cont_matmul.out
```

Repeat for `nested` to compare.

## Key results

See `docs/profiling-results.md` for the full table. Headline finding:

> The contiguous 1D layout produced **82.7% fewer last-level cache misses** than nested vectors on 1024×1024 matrix multiplication, where the matrix data exceeds L1 cache capacity.
