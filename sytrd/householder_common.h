#ifndef HOUSEHOLDER_COMMON_H
#define HOUSEHOLDER_COMMON_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>

// Common utilities
void generate_symmetric_matrix(std::vector<float>& A, int n, unsigned int seed = 42);
bool test_tridiagonalization(const std::vector<float>& A, int n, rocblas_fill uplo);

// Blocked Householder tridiagonalization
extern "C" hipError_t hip_block_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau,
    int block_size);

// Unblocked Householder tridiagonalization
extern "C" hipError_t hip_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau);

// Benchmark result structure
struct BenchmarkResult {
    int matrix_size;
    double avg_time_ms;
    std::string validation_result;
    int block_size;  // Only used for blocked version
};

// Benchmark functions
BenchmarkResult benchmark_blocked_algorithm(int n, int block_size, bool validate, int iterations = 10, int warmup = 3);
BenchmarkResult benchmark_unblocked_algorithm(int n, bool validate, int iterations = 10, int warmup = 3);

#endif // HOUSEHOLDER_COMMON_H
