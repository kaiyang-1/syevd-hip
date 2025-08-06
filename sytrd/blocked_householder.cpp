#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <algorithm>

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

// Kernel to build Householder vector u given alpha and precomputed u_j1
__global__ void build_householder_vector(int n, int j, float alpha, float u_j1, const float* a_col_vec, float* u_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx <= j) {
            u_vec[idx] = 0.0f;
        } else if (idx == j + 1) {
            u_vec[idx] = u_j1;
        } else {
            float scale = -1.0f / (alpha * u_j1);
            u_vec[idx] = scale * a_col_vec[idx - (j + 1)];
        }
    }
}

// Kernel to compute v = y - 0.5 * (y^T u) * u
__global__ void compute_v(int n, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = y[idx] - 0.5f * dot * u[idx];
}

// Fused kernel to apply all accumulated updates from previous columns in the block
template<rocblas_fill uplo>
__global__ void apply_accumulated_updates(int n, int j, int j_idx, int lda,
                                          const float* A, const float* dU, const float* dV,
                                          float* a_col_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = n - j - 1;
    
    if (idx < m) {
        // Extract the original column element
        float a_val;
        if constexpr (uplo == rocblas_fill_lower) {
            a_val = A[(j+1+idx) + j*lda];
        } else {
            a_val = A[j + (j+1+idx)*lda];
        }
        
        // Apply all accumulated updates from previous columns in this block
        for (int i_idx = 0; i_idx < j_idx; ++i_idx) {
            float scalar_u = dU[j + i_idx*n];
            float scalar_v = dV[j + i_idx*n];
            float u_tail = dU[(j+1+idx) + i_idx*n];
            float v_tail = dV[(j+1+idx) + i_idx*n];
            
            a_val -= scalar_v * u_tail + scalar_u * v_tail;
        }
        
        a_col_vec[idx] = a_val;
    }
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

// Block Householder tridiagonalization implementation
extern "C" hipError_t hip_block_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau,
    int block_size)
{
    if (n < 3) return hipSuccess;
    
    const float one = 1.0f;
    const float minus_one = -1.0f;
    const float zero = 0.0f;
    
    // Calculate the number of blocks
    int num_blocks = (n - 2 + block_size - 1) / block_size;
    
    // Allocate workspace for U and V matrices (n x block_size each)
    float *dU, *dV;
    hipMalloc(&dU, n * block_size * sizeof(float));
    hipMalloc(&dV, n * block_size * sizeof(float));
    
    // Allocate workspace for temporary vectors
    float *u_vec, *y_vec, *a_col_vec, *temp_vec;
    hipMalloc(&u_vec, n * sizeof(float));
    hipMalloc(&y_vec, n * sizeof(float));
    hipMalloc(&a_col_vec, n * sizeof(float));
    hipMalloc(&temp_vec, block_size * sizeof(float));
    
    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_col = k * block_size;
        int end_col = std::min(start_col + block_size, n - 2);
        int current_block_size = end_col - start_col;
        
        if (current_block_size <= 0) break;
        
        // Initialize U and V matrices for this block to zero
        hipMemset(dU, 0, n * block_size * sizeof(float));
        hipMemset(dV, 0, n * block_size * sizeof(float));
        
        // Process each column within the block
        for (int j = start_col; j < end_col; ++j) {
            int j_idx = j - start_col;  // Index within the current block
            int m = n - j - 1;         // Size of the vector to process
            
            // Apply all accumulated updates in a single fused kernel call
            if (j_idx > 0) {
                // Use fused kernel to extract column and apply all updates at once
                int threads = 256;
                int blocks = (m + threads - 1) / threads;
                if (uplo == rocblas_fill_lower) {
                    hipLaunchKernelGGL(apply_accumulated_updates<rocblas_fill_lower>, 
                                       dim3(blocks), dim3(threads), 0, 0,
                                       n, j, j_idx, lda, dA, dU, dV, a_col_vec);
                } else {
                    hipLaunchKernelGGL(apply_accumulated_updates<rocblas_fill_upper>, 
                                       dim3(blocks), dim3(threads), 0, 0,
                                       n, j, j_idx, lda, dA, dU, dV, a_col_vec);
                }
                hipDeviceSynchronize();
            } else {
                // For the first column in the block, just extract the column
                if (uplo == rocblas_fill_lower) {
                    hipMemcpy(a_col_vec, dA + (j+1) + j*lda, m * sizeof(float), hipMemcpyDeviceToDevice);
                } else {
                    rocblas_scopy(handle, m, dA + j + (j+1)*lda, lda, a_col_vec, 1);
                }
            }
            
            // Compute norm of the updated column
            float norm_a;
            rocblas_snrm2(handle, m, a_col_vec, 1, &norm_a);
            
            if (norm_a < 1e-10f) {
                // Skip zero vector - set u[j+1] = 1, others = 0
                hipMemset(u_vec, 0, n * sizeof(float));
                float one_val = 1.0f;
                hipMemcpy(u_vec + j + 1, &one_val, sizeof(float), hipMemcpyHostToDevice);
            } else {
                // Compute Householder vector
                float a0;
                hipMemcpy(&a0, a_col_vec, sizeof(float), hipMemcpyDeviceToHost);
                
                float sign = (a0 >= 0) ? 1.0f : -1.0f;
                float alpha = -sign * norm_a;
                float u_j1 = sqrtf(1.0f - a0/alpha);
                
                int threads = 256;
                int blocks = (n + threads - 1) / threads;
                hipLaunchKernelGGL(build_householder_vector, dim3(blocks), dim3(threads), 0, 0,
                                   n, j, alpha, u_j1, a_col_vec, u_vec);
                hipDeviceSynchronize();
            }
            
            // Compute y = (A - U_prev * V_prev^T - V_prev * U_prev^T) * u
            // First compute y = A * u
            rocblas_ssymv(handle, uplo, n, &one, dA, lda, u_vec, 1, &zero, y_vec, 1);
                        
            // Apply previous updates: y -= U_prev * (V_prev^T * u) + V_prev * (U_prev^T * u)
            if (j_idx > 0) {
                // Compute V_prev^T * u
                rocblas_sgemv(handle, rocblas_operation_transpose, n, j_idx,
                              &one, dV, n, u_vec, 1, &zero, temp_vec, 1);
                // y -= U_prev * temp_vec
                rocblas_sgemv(handle, rocblas_operation_none, n, j_idx,
                              &minus_one, dU, n, temp_vec, 1, &one, y_vec, 1);

                // Compute U_prev^T * u
                rocblas_sgemv(handle, rocblas_operation_transpose, n, j_idx,
                              &one, dU, n, u_vec, 1, &zero, temp_vec, 1);
                // y -= V_prev * temp_vec
                rocblas_sgemv(handle, rocblas_operation_none, n, j_idx,
                              &minus_one, dV, n, temp_vec, 1, &one, y_vec, 1);
            }            
            
            // Compute v = y - 0.5 * (y^T * u) * u
            float dot;
            rocblas_sdot(handle, n, y_vec, 1, u_vec, 1, &dot);
            
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            // Write v directly to its position in dV matrix
            hipLaunchKernelGGL(compute_v, dim3(blocks), dim3(threads), 0, 0,
                               n, y_vec, u_vec, dot, dV + j_idx*n);
            hipDeviceSynchronize();
            
            // Store u in the U matrix
            hipMemcpy(dU + j_idx*n, u_vec, n * sizeof(float), hipMemcpyDeviceToDevice);
        }
        
        // Apply symmetric rank-2p update for the entire block
        // A -= U_block * V_block^T + V_block * U_block^T
        rocblas_ssyr2k(handle, uplo, rocblas_operation_none,
                       n, current_block_size,
                       &minus_one,
                       dU, n,
                       dV, n,
                       &one,
                       dA, lda);
    }
    
    // Extract tridiagonal elements
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
    hipFree(dU);
    hipFree(dV);
    hipFree(u_vec);
    hipFree(y_vec);
    hipFree(a_col_vec);
    hipFree(temp_vec);
    
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
    int block_size = 32;  // Default block size
    bool validate = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        }
    }

    if (matrix_sizes.empty()) {
        printf("Usage: %s -n size1 size2 ... [-b block_size] [-v]\n", argv[0]);
        printf("  -n: matrix sizes to test\n");
        printf("  -b: block size (default: 32)\n");
        printf("  -v: enable validation\n");
        return 1;
    }

    printf("Block Householder Tridiagonalization (Block Size: %d)\n", block_size);
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
