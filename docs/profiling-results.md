# Custom ML Engine — Performance Profiling Results

## Methodology

- **Build**: CMake RelWithDebInfo, g++ 15.2.0, `-O3 -DNDEBUG -std=c++23 -g`
- **Benchmark**: Standalone C++ benchmark comparing contiguous `std::vector<double>` (row-major 1D, matching the engine's Matrix class) against a `std::vector<std::vector<double>>` baseline. Same r-k-c loop order, same algorithm, same data — only storage layout differs.
- **Cache analysis**: Valgrind Cachegrind 3.26.0 with simulated L1 D-cache (32 KB, 8-way, 64 B lines) and last-level cache (8 MB, 16-way, 64 B lines)
- **Leak check**: Valgrind Memcheck with `PYTHONMALLOC=malloc`
- **No code changes** were made to the original engine. The benchmark is a separate file.

---

## 1. Cachegrind Results: Matrix Multiplication

### 512×512 matmul (10 reps)

| Metric | Contiguous (1D) | Nested (2D) | Difference |
|---|---:|---:|---|
| D refs (total) | 2,246,615,563 | 2,250,322,666 | -0.17% |
| D1 misses | 185,839,913 | 187,647,947 | -0.97% |
| D1 miss rate | 8.3% | 8.3% | — |
| **LLd misses** | **142,815** | **186,919** | **-23.6%** |
| LLd miss rate | 0.0% | 0.0% | — |

### 1024×1024 matmul (3 reps)

| Metric | Contiguous (1D) | Nested (2D) | Difference |
|---|---:|---:|---|
| D refs (total) | 6,502,011,201 | 6,508,985,646 | -0.11% |
| D1 misses | 539,136,725 | 541,777,050 | -0.49% |
| D1 miss rate | 8.3% | 8.3% | — |
| **LLd misses** | **19,733,282** | **114,065,765** | **-82.7%** |
| **LLd miss rate** | **0.3%** | **1.8%** | **-83.3%** |

The 1024×1024 matrix (8 MB) exceeds the 32 KB L1 cache. At this scale, the contiguous single-allocation layout dramatically reduces last-level cache misses because nested-vector rows are scattered across separate heap allocations, causing TLB and LL cache thrashing.

---

## 2. Cachegrind Results: Matrix Transposition

### 512×512 transpose (10 reps)

| Metric | Contiguous (1D) | Nested (2D) | Difference |
|---|---:|---:|---|
| D refs (total) | 13,819,039 | 17,599,228 | -21.5% |
| **D1 misses** | **3,457,781** | **4,557,155** | **-24.1%** |
| D1 read misses | 374,916 | 1,472,541 | -74.6% |
| D1 read miss rate | 4.6% | 12.2% | -62.3% |
| D1 write misses | 3,082,865 | 3,084,614 | -0.06% |
| LLd misses | 142,228 | 172,177 | -17.4% |

Transpose accesses the destination matrix by column. In the contiguous layout, this is a stride-`n` access within a single allocation. In the nested layout, column access jumps between separate heap-allocated row vectors, causing 3.9× more L1 read misses.

---

## 3. Cachegrind Results: Hadamard (Element-wise) Product

### 512×512 hadamard (10 reps)

| Metric | Contiguous (1D) | Nested (2D) | Difference |
|---|---:|---:|---|
| D refs (total) | 12,399,865 | 13,285,100 | -6.7% |
| D1 misses | 1,295,089 | 1,311,076 | -1.2% |
| LLd misses | 142,819 | 186,914 | -23.6% |

Element-wise access is sequential in both layouts, so the difference is smaller. The contiguous layout still benefits from a single allocation at the LL cache level.

---

## 4. Cachegrind Results: Full Training Loop (200 epochs)

Profiled the actual XOR training via `PYTHONMALLOC=malloc valgrind --tool=cachegrind`:

| Metric | Value |
|---|---:|
| I refs | 51,496,730 |
| D refs | 19,631,475 |
| D1 misses | 717,040 |
| D1 miss rate | 3.7% |
| LLd misses | 66,937 |
| LLd miss rate | 0.3% |

The training workload has a low cache miss rate (3.7% D1) because the XOR matrices are tiny (4×2, 2×8, 4×8, 8×2) and fit entirely in L1 cache.

---

## 5. Runtime Timing (no march=native, -O3)

### 512×512 matmul (10 trials, median)

| Layout | Median time | 
|---|---:|
| Contiguous | ~404 ms |
| Nested | ~447 ms |
| Speedup | 1.11× (10.7% faster) |

### 1024×1024 matmul (5 trials, median)

| Layout | Median time |
|---|---:|
| Contiguous | ~1,027 ms |
| Nested | ~1,027 ms |

At 1024×1024, runtime is comparable because the workload is compute-bound (O(n³) FLOPs dominate). The cache benefit shows up in miss counts, not wall-clock time, at this optimization level.

---

## 6. Memory Leak Verification (Memcheck)

### Full training loop (200 epochs, `PYTHONMALLOC=malloc`)

| Metric | Value |
|---|---:|
| Total heap usage | 37,547 allocs, 37,476 frees |
| Total bytes allocated | 6,034,060 |
| **Definitely lost** | **0 bytes in 0 blocks** |
| Indirectly lost | 0 bytes in 0 blocks |
| Possibly lost | 5,909 bytes in 32 blocks (Python runtime) |
| Still reachable | 86,631 bytes in 37 blocks (Python runtime + libstdc++) |

**Zero memory leaks confirmed in the C++ engine.** All "possibly lost" and "still reachable" entries originate from the Python interpreter, pybind11 module loading, and libstdc++ initialization — not from the engine's Matrix, Layer, or Network objects.

---

## 7. Training Loop Performance

| Metric | Value |
|---|---:|
| Wall-clock time (full XOR training to convergence) | ~15 ms |
| Epochs to convergence (loss < 0.01) | 1,720–2,866 (varies by random seed) |
| Learning rate | 0.05 |
| Architecture | 2→8 (ReLU)→2 (Softmax-CE) |

---

## Environment

| Component | Version |
|---|---|
| OS | Ubuntu (Linux 7.0.0-31-generic) |
| Compiler | g++ 15.2.0 |
| CMake | 4.2.3 |
| Valgrind/Cachegrind | 3.26.0 |
| Python | 3.14.3 |
| pybind11 | v2.13.1 (fetched via CMake FetchContent) |
| C++ Standard | C++23 |
| CPU | x86_64 (virtualized) |

---

## Resume-Ready Bullet (with real data)

> Implemented contiguous row-major 2D matrix storage with one-dimensional `std::vector` backing; profiled with Valgrind Cachegrind, validating 82.7% fewer last-level cache misses versus a `std::vector<std::vector<double>>` baseline on 1024×1024 matrix multiplication and 24.1% fewer L1 data cache misses on matrix transposition.

Alternative shorter version:

> Implemented contiguous row-major 2D matrix storage with 1D `std::vector` backing; Cachegrind profiling confirmed 82.7% fewer last-level cache misses versus a nested-vector baseline on large matrix multiplication.
