#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

// Function to generate a random symmetric matrix on host
void generate_symmetric_matrix(std::vector<float>& A, int n, unsigned int seed = 42) {
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

// Kernel to build Householder vector u given alpha and precomputed u_i1
template<rocblas_fill uplo>
__global__ void build_householder_u(int n, int i, const float* A, int lda, float alpha, float u_i1, float* u) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx <= i) {
            u[idx] = 0.0f;
        } else if (idx == i + 1) {
            u[idx] = u_i1;
        } else {
            float ai;
            if constexpr (uplo == rocblas_fill_lower) {
                // For lower triangular: access A[idx, i] in column-major format
                ai = A[idx + i * lda];
            } else {
                // For upper triangular: access A[i, idx] in column-major format
                ai = A[i + idx * lda];
            }
            u[idx] = -ai / (alpha * u_i1);
        }
    }
}

// Kernel to compute v = y - 0.5 * (y^T u) * u
__global__ void compute_v(int n, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = y[idx] - 0.5f * dot * u[idx];
}

// Kernel to extract diagonal D and sub-diagonal E from tridiagonalized A
template<rocblas_fill uplo>
__global__ void extract_tridiag(int n, const float* A, int lda, float* D, float* E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = A[i + i * lda];
        if (i < n-1) {
            if constexpr (uplo == rocblas_fill_lower) {
                E[i] = A[(i+1) + i * lda];
            } else {
                E[i] = A[i + (i+1) * lda];
            }
        }
    }
}

// Host API mimicking cuSolver/rocSolver ssytrd
extern "C" hipError_t hip_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau)
{
    // Allocate workspace for u, y, v
    float *u, *y, *v;
    hipMalloc(&u, n * sizeof(float));
    hipMalloc(&y, n * sizeof(float));
    hipMalloc(&v, n * sizeof(float));

    const float one = 1.0f;
    const float minus_one = -1.0f;
    const float zero = 0.0f;

    for (int i = 0; i < n - 2; ++i) {
        int m = n - i - 1;
        // compute norm of column/row vector depending on uplo
        float norm_a;
        if (uplo == rocblas_fill_lower) {
            // For lower triangular: compute norm of A[i+1:n, i] (column vector)
            rocblas_snrm2(handle, m, dA + (i+1) + i*lda, 1, &norm_a);
        } else {
            // For upper triangular: compute norm of A[i, i+1:n] (row vector)
            rocblas_snrm2(handle, m, dA + i + (i+1)*lda, lda, &norm_a);
        }

        if (norm_a < 1e-10f) continue;

        // compute alpha and u_i1 on host
        float a0;
        if (uplo == rocblas_fill_lower) {
            hipMemcpy(&a0, dA + (i+1) + i*lda, sizeof(float), hipMemcpyDeviceToHost);
        } else {
            hipMemcpy(&a0, dA + i + (i+1)*lda, sizeof(float), hipMemcpyDeviceToHost);
        }
        float sign = (a0 >= 0) ? 1.0f : -1.0f;
        float alpha = -sign * norm_a;
        float u_i1 = sqrtf(1.0f - a0/alpha);

        // build u in parallel with precomputed u_i1
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        if (uplo == rocblas_fill_lower) {
            hipLaunchKernelGGL(build_householder_u<rocblas_fill_lower>, dim3(blocks), dim3(threads), 0, 0,
                               n, i, dA, lda, alpha, u_i1, u);
        } else {
            hipLaunchKernelGGL(build_householder_u<rocblas_fill_upper>, dim3(blocks), dim3(threads), 0, 0,
                               n, i, dA, lda, alpha, u_i1, u);
        }
        hipDeviceSynchronize();

        // y = A * u using symmetric matrix-vector multiplication
        rocblas_ssymv(handle,
                      uplo,
                      n,
                      &one,
                      dA, lda,
                      u, 1,
                      &zero,
                      y, 1);

        // dot = y^T u
        float dot;
        rocblas_sdot(handle, n, y, 1, u, 1, &dot);

        // v = y - 0.5 * dot * u
        hipLaunchKernelGGL(compute_v, dim3(blocks), dim3(threads), 0, 0,
                           n, y, u, dot, v);
        hipDeviceSynchronize();

        // symmetric rank-2 update using rocblas_ssyr2
        rocblas_ssyr2(handle,
                      uplo,
                      n,
                      &minus_one,
                      u, 1,
                      v, 1,
                      dA, lda);
    }

    // extract tridiagonal elements
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (uplo == rocblas_fill_lower) {
        hipLaunchKernelGGL(extract_tridiag<rocblas_fill_lower>, dim3(blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    } else {
        hipLaunchKernelGGL(extract_tridiag<rocblas_fill_upper>, dim3(blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    }
    hipDeviceSynchronize();

    // Free workspace
    hipFree(u);
    hipFree(y);
    hipFree(v);

    return hipSuccess;
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

int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes;
    bool validate = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        }
    }

    if (matrix_sizes.empty()) {
        printf("Usage: %s -n size1 size2 ... [-v]\n", argv[0]);
        return 1;
    }

    printf("Matrix Size | Average Time (ms)");
    if (validate) {
        printf(" | Validation\n");
        printf("---------------------------------------------\n");
    } else {
        printf("\n");
        printf("--------------------------------\n");
    }

    for (int n : matrix_sizes) {
        const int lda = n;
        const int iterations = 10; // Number of benchmark iterations
        const int warmup = 3;      // Number of warmup runs (not timed)

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

        double avg_time = total_time / iterations / 1000.0; // Convert to milliseconds

        // Validation
        std::string validation_result = "";
        if (validate) {
            hipMemcpy(hA.data(), dA, n * lda * sizeof(float), hipMemcpyDeviceToHost);

            if (test_tridiagonalization(hA, n, rocblas_fill_lower)) {
                validation_result = "Pass";
            } else {
                validation_result = "Fail";
            }
        }

        if (validate) {
            printf("%11d | %17.2f | %10s\n", n, avg_time, validation_result.c_str());
        } else {
            printf("%11d | %17.2f\n", n, avg_time);
        }

        hipFree(dA);
        hipFree(dD);
        hipFree(dE);
        hipFree(dTau);
        rocblas_destroy_handle(handle);
    }

    return 0;
}
