#include "householder_common.h"
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

// Function to generate a random symmetric matrix on host
void generate_symmetric_matrix(std::vector<float>& A, int n, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Initialize matrix to zero
    std::fill(A.begin(), A.end(), 0.0f);
    
    // Generate symmetric matrix in column-major format
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            float value = dist(rng);
            A[i + j * n] = value;  // Column-major: A[i][j]
            if (i != j) {
                A[j + i * n] = value;  // Column-major: A[j][i] = A[i][j]
            }
        }
    }
}

// Tests whether a given matrix has been successfully reduced to tridiagonal form
bool test_tridiagonalization(const std::vector<float>& A, int n, rocblas_fill uplo) {
    const float tolerance = 1e-5f;

    // Check all elements outside the tridiagonal band of the matrix are within a specified tolerance
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (uplo == rocblas_fill_lower && i > j + 1) {
                if (std::abs(A[i + j * n]) > tolerance) {
                    return false;
                }
            } else if (uplo == rocblas_fill_upper && j > i + 1) {
                if (std::abs(A[i + j * n]) > tolerance) {
                    return false;
                }
            }
        }
    }

    return true;
}

// Benchmark blocked algorithm
BenchmarkResult benchmark_blocked_algorithm(int n, int block_size, bool validate, int iterations, int warmup) {
    BenchmarkResult result;
    result.matrix_size = n;
    result.block_size = block_size;
    
    const int lda = n;
    std::vector<float> hA(n * n);

    float *dA, *dD, *dE, *dTau;
    hipMalloc(&dA, n * lda * sizeof(float));
    hipMalloc(&dD, n * sizeof(float));
    hipMalloc(&dE, (n - 1) * sizeof(float));
    hipMalloc(&dTau, (n - 1) * sizeof(float));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Generate matrix
    generate_symmetric_matrix(hA, n);

    // Warmup
    for (int warmup_iter = 0; warmup_iter < warmup; ++warmup_iter) {
        hipMemcpy(dA, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hip_block_ssytrd(handle, rocblas_fill_lower, n, dA, lda, dD, dE, dTau, block_size);
    }

    // Benchmark
    double total_time = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        generate_symmetric_matrix(hA, n);
        hipMemcpy(dA, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        hip_block_ssytrd(handle, rocblas_fill_lower, n, dA, lda, dD, dE, dTau, block_size);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> elapsed = end - start;
        total_time += elapsed.count();
    }

    result.avg_time_ms = total_time / iterations / 1000.0; // Convert to milliseconds

    // Validation
    if (validate) {
        hipMemcpy(hA.data(), dA, n * lda * sizeof(float), hipMemcpyDeviceToHost);

        if (test_tridiagonalization(hA, n, rocblas_fill_lower)) {
            result.validation_result = "Pass";
        } else {
            result.validation_result = "Fail";
        }
    } else {
        result.validation_result = "N/A";
    }

    hipFree(dA);
    hipFree(dD);
    hipFree(dE);
    hipFree(dTau);
    rocblas_destroy_handle(handle);

    return result;
}

// Benchmark unblocked algorithm
BenchmarkResult benchmark_unblocked_algorithm(int n, bool validate, int iterations, int warmup) {
    BenchmarkResult result;
    result.matrix_size = n;
    result.block_size = 0; // Not applicable for unblocked
    
    const int lda = n;
    std::vector<float> hA(n * n);

    float *dA, *dD, *dE, *dTau;
    hipMalloc(&dA, n * lda * sizeof(float));
    hipMalloc(&dD, n * sizeof(float));
    hipMalloc(&dE, (n - 1) * sizeof(float));
    hipMalloc(&dTau, (n - 1) * sizeof(float));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Generate matrix
    generate_symmetric_matrix(hA, n);

    // Warmup
    for (int warmup_iter = 0; warmup_iter < warmup; ++warmup_iter) {
        hipMemcpy(dA, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hip_ssytrd(handle, rocblas_fill_lower, n, dA, lda, dD, dE, dTau);
    }

    // Benchmark
    double total_time = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        generate_symmetric_matrix(hA, n);
        hipMemcpy(dA, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        hip_ssytrd(handle, rocblas_fill_lower, n, dA, lda, dD, dE, dTau);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> elapsed = end - start;
        total_time += elapsed.count();
    }

    result.avg_time_ms = total_time / iterations / 1000.0; // Convert to milliseconds

    // Validation
    if (validate) {
        hipMemcpy(hA.data(), dA, n * lda * sizeof(float), hipMemcpyDeviceToHost);

        if (test_tridiagonalization(hA, n, rocblas_fill_lower)) {
            result.validation_result = "Pass";
        } else {
            result.validation_result = "Fail";
        }
    } else {
        result.validation_result = "N/A";
    }

    hipFree(dA);
    hipFree(dD);
    hipFree(dE);
    hipFree(dTau);
    rocblas_destroy_handle(handle);

    return result;
}
