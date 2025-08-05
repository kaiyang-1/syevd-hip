#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <magma_v2.h>

// Generate a random symmetric matrix
void generate_symmetric_matrix(std::vector<float>& A, int n, unsigned int seed = 42) {
    std::mt19937 rng(seed); // Random number generator
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Uniform distribution [0,1]
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float value = dist(rng);
            A[i * n + j] = value;
            if (i != j) {
                A[j * n + i] = value; // Ensure symmetry
            }
        }
    }
}

// Compute the residual: ||A - QΛQᵀ||_F / ||A||_F
float compute_residual(const std::vector<float>& A, const std::vector<float>& Q, 
                      const std::vector<float>& W, int n) {
    // Compute the Frobenius norm of A
    float norm_A = 0.0f;
    for (int i = 0; i < n * n; ++i) {
        norm_A += A[i] * A[i];
    }
    norm_A = std::sqrt(norm_A);

    // Precompute QΛ (Q times diagonal matrix of eigenvalues)
    // Q: each row is an eigenvector, W: eigenvalues
    std::vector<float> QL(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            QL[i * n + j] = Q[j * n + i] * W[j];
        }
    }

    // Compute the Frobenius norm of (A - QΛQᵀ) directly, no need to store the residual matrix
    float residual_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += QL[i * n + k] * Q[k * n + j];
            }
            float diff = A[i * n + j] - sum;
            residual_norm += diff * diff;
        }
    }
    residual_norm = std::sqrt(residual_norm);
    return residual_norm / norm_A;
}

// Compare eigenvalues from two different implementations (return the maximum absolute difference)
float compare_eigenvalues(const std::vector<float>& W1, const std::vector<float>& W2, int n) {
    // Sort eigenvalues for comparison (since order may differ)
    std::vector<float> sorted_W1 = W1;
    std::vector<float> sorted_W2 = W2;
    std::sort(sorted_W1.begin(), sorted_W1.end());
    std::sort(sorted_W2.begin(), sorted_W2.end());
    
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabs(sorted_W1[i] - sorted_W2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    return max_diff;
}

int main(int argc, char** argv) {
    // Argument parsing: support -n size1 size2 ... [-v]
    // -n: followed by one or more matrix sizes
    // -v: enable validation (residual and eigenvalue difference)
    std::vector<int> sizes;
    bool do_validation = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n") {
            // Read all following numbers until next '-' or end
            int j = i + 1;
            while (j < argc && argv[j][0] != '-') {
                sizes.push_back(std::stoi(argv[j]));
                ++j;
            }
            i = j - 1;
        } else if (arg == "-v") {
            do_validation = true;
        }
    }
    if (sizes.empty()) {
        std::cerr << "Usage: " << argv[0] << " -n <size1> <size2> ... [-v]" << std::endl;
        return 1;
    }

    const int iterations = 10; // Number of benchmark iterations
    const int warmup = 3;      // Number of warmup runs (not timed)

    // Initialize ROCm and MAGMA libraries
    hipInit(0);                // Initialize HIP runtime
    magma_init();              // Initialize MAGMA
    magma_setdevice(0);        // Set device to GPU 0

    // Create rocBLAS handle (reusable for all runs)
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Print table header
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(12) << "MatrixSize"
              << std::setw(15) << "rocSOLVER(ms)"
              << std::setw(15) << "MAGMA(ms)"
              << std::setw(30) << "SpeedRatio(rocSOLVER/MAGMA)";
    if (do_validation) {
        std::cout << std::setw(20) << "rocSOLVER_residual"
                  << std::setw(20) << "MAGMA_residual"
                  << std::setw(20) << "MaxEigDiff";
    }
    std::cout << std::endl;

    for (int n : sizes) {
        // For each matrix size, allocate and initialize data, run warmup and benchmark, and print results
        const int size = n * n;
        std::vector<float> A(size); // Host matrix
        std::vector<float> W_rocsolver(n), W_magma(n), Q_rocsolver(size), Q_magma(size); // Host eigenvalues/vectors
        generate_symmetric_matrix(A, n, 42); // Generate random symmetric matrix

        // Allocate device memory for input/output matrices and eigenvalues
        float *dA, *dW, *dA_orig;
        hipMalloc(&dA, size * sizeof(float));
        hipMalloc(&dW, n * sizeof(float));
        hipMalloc(&dA_orig, size * sizeof(float));
        hipMemcpy(dA_orig, A.data(), size * sizeof(float), hipMemcpyHostToDevice);

        // rocSOLVER-specific device allocations
        float *dE;
        rocblas_int *dInfo;
        hipMalloc(&dE, n * sizeof(float));
        hipMalloc(&dInfo, sizeof(rocblas_int));

        // MAGMA-specific workspace allocations
        magma_int_t lwork, liwork, info;
        float size_work;
        magma_int_t size_iwork;
        // Query MAGMA workspace sizes
        magma_ssyevd_gpu(MagmaVec, MagmaUpper,
                        n, nullptr, n, nullptr, 
                        nullptr, n,    // wA (host workspace)
                        &size_work, -1, 
                        &size_iwork, -1, 
                        &info);
        lwork = static_cast<magma_int_t>(size_work);
        liwork = size_iwork;
        float *wA, *work;
        magma_int_t *iwork;
        magma_smalloc_cpu(&wA, n * n);
        magma_smalloc_cpu(&work, lwork);
        magma_imalloc_cpu(&iwork, liwork);

        // Warmup runs (not timed)
        for (int i = 0; i < warmup; ++i) {
            hipMemcpy(dA, dA_orig, size * sizeof(float), hipMemcpyDeviceToDevice);
            rocsolver_ssyevd(handle, rocblas_evect_original, rocblas_fill_upper, n, dA, n, dW, dE, dInfo);

            hipMemcpy(dA, dA_orig, size * sizeof(float), hipMemcpyDeviceToDevice);
            magma_ssyevd_gpu(MagmaVec, MagmaUpper, n, dA, n, dW, wA, n, work, lwork, iwork, liwork, &info);
        }

        // Benchmark runs (timed)
        float total_rocsolver_time = 0;
        float total_magma_time = 0;
        for (int i = 0; i < iterations; ++i) {
            hipMemcpy(dA, dA_orig, size * sizeof(float), hipMemcpyDeviceToDevice);

            auto t1 = std::chrono::high_resolution_clock::now();
            rocsolver_ssyevd(handle, rocblas_evect_original, rocblas_fill_upper, n, dA, n, dW, dE, dInfo);
            hipDeviceSynchronize(); // Ensure all GPU work is done
            auto t2 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            total_rocsolver_time += us;

            // Save results for validation (only need once)
            if (do_validation && i == 0) {
                hipMemcpy(Q_rocsolver.data(), dA, size * sizeof(float), hipMemcpyDeviceToHost);
                hipMemcpy(W_rocsolver.data(), dW, n * sizeof(float), hipMemcpyDeviceToHost);
            }
        }
        for (int i = 0; i < iterations; ++i) {
            hipMemcpy(dA, dA_orig, size * sizeof(float), hipMemcpyDeviceToDevice);

            auto t1 = std::chrono::high_resolution_clock::now();
            magma_ssyevd_gpu(MagmaVec, MagmaUpper, n, dA, n, dW, wA, n, work, lwork, iwork, liwork, &info);
            hipDeviceSynchronize();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            total_magma_time += us;
            
            if (do_validation && i == 0) {
                hipMemcpy(Q_magma.data(), dA, size * sizeof(float), hipMemcpyDeviceToHost);
                hipMemcpy(W_magma.data(), dW, n * sizeof(float), hipMemcpyDeviceToHost);
            }
        }

        // Calculate average times in milliseconds
        float avg_rocsolver = total_rocsolver_time / iterations / 1000.0f;
        float avg_magma = total_magma_time / iterations / 1000.0f;
        // Print results for this matrix size
        std::cout << std::left << std::setw(12) << n
                  << std::setw(15) << avg_rocsolver
                  << std::setw(15) << avg_magma
                  << std::setw(30) << (avg_magma / avg_rocsolver);

        if (do_validation) {
            float rocsolver_residual = compute_residual(A, Q_rocsolver, W_rocsolver, n);
            float magma_residual = compute_residual(A, Q_magma, W_magma, n);
            float eigenvalue_diff = compare_eigenvalues(W_rocsolver, W_magma, n);
            char buf1[32], buf2[32], buf3[32];
            snprintf(buf1, sizeof(buf1), "%.3e", rocsolver_residual);
            snprintf(buf2, sizeof(buf2), "%.3e", magma_residual);
            snprintf(buf3, sizeof(buf3), "%.3e", eigenvalue_diff);
            std::cout << std::setw(20) << buf1
                      << std::setw(20) << buf2
                      << std::setw(20) << buf3;
        }
        std::cout << std::endl;

        // Free all allocated memory for this matrix size
        hipFree(dA);
        hipFree(dW);
        hipFree(dA_orig);
        hipFree(dE);
        hipFree(dInfo);
        magma_free_cpu(wA);
        magma_free_cpu(work);
        magma_free_cpu(iwork);
    }

    // Clean up global resources
    rocblas_destroy_handle(handle);
    magma_finalize();
    return 0;
}