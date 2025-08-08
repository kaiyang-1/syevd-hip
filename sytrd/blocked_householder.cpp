#include "householder_common.h"
#include <algorithm>

// Kernel to apply all accumulated updates from previous columns in the block
template<rocblas_fill uplo>
__global__ void apply_accumulated_updates(int m, int j, int j_idx, const float* A, int lda,
                                          const float* dU, const float* dV, int trailing_size,
                                          float* a_col_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
            float scalar_u = dU[j_idx + i_idx*trailing_size];
            float scalar_v = dV[j_idx + i_idx*trailing_size];
            float u_tail = dU[(j_idx + 1 + idx) + i_idx*trailing_size];
            float v_tail = dV[(j_idx + 1 + idx) + i_idx*trailing_size];
            
            a_val -= scalar_v * u_tail + scalar_u * v_tail;
        }
        
        a_col_vec[idx] = a_val;
    }
}

// Kernel to build Householder vector u given alpha and precomputed u_j1
__global__ void build_householder_vector(int trailing_size, int j_idx, float alpha, float u_j1,
                                         const float* a_col_vec, float* u_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trailing_size) {
        if (idx <= j_idx) {
            u_vec[idx] = 0.0f;
        } else if (idx == j_idx + 1) {
            u_vec[idx] = u_j1;
        } else {
            float scale = -1.0f / (alpha * u_j1);
            u_vec[idx] = scale * a_col_vec[idx - j_idx - 1];
        }
    }
}

// Kernel to compute v = y - 0.5 * (y^T u) * u
__global__ void compute_v(int trailing_size, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trailing_size) v[idx] = y[idx] - 0.5f * dot * u[idx];
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
    float *y_vec, *a_col_vec, *temp_vec;
    hipMalloc(&y_vec, n * sizeof(float));
    hipMalloc(&a_col_vec, n * sizeof(float));
    hipMalloc(&temp_vec, block_size * sizeof(float));
    
    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_col = k * block_size;
        int end_col = std::min(start_col + block_size, n - 2);
        int current_block_size = end_col - start_col;
        int trailing_size = n - start_col;
        
        if (current_block_size <= 0) break;
        
        // Process each column within the block
        for (int j = start_col; j < end_col; ++j) {
            int j_idx = j - start_col;  // Column index within the current block
            int m = n - j - 1;         // Size of the column vector to process

            // Alias for the Householder vector u
            float *u_vec = dU + j_idx * trailing_size;
            
            // Apply all accumulated updates in a single fused kernel call
            if (j_idx > 0) {
                // Use fused kernel to extract column and apply all updates at once
                int threads = 256;
                int blocks = (m + threads - 1) / threads;
                if (uplo == rocblas_fill_lower) {
                    hipLaunchKernelGGL(apply_accumulated_updates<rocblas_fill_lower>, 
                                       dim3(blocks), dim3(threads), 0, 0,
                                       m, j, j_idx, dA, lda, dU, dV, trailing_size, a_col_vec);
                } else {
                    hipLaunchKernelGGL(apply_accumulated_updates<rocblas_fill_upper>, 
                                       dim3(blocks), dim3(threads), 0, 0,
                                       m, j, j_idx, dA, lda, dU, dV, trailing_size, a_col_vec);
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
                hipMemset(u_vec, 0, trailing_size * sizeof(float));
                float one_val = 1.0f;
                hipMemcpy(u_vec + j_idx + 1, &one_val, sizeof(float), hipMemcpyHostToDevice);
            } else {
                // Compute Householder vector
                float a0;
                hipMemcpy(&a0, a_col_vec, sizeof(float), hipMemcpyDeviceToHost);
                
                float sign = (a0 >= 0) ? 1.0f : -1.0f;
                float alpha = -sign * norm_a;
                float u_j1 = sqrtf(1.0f - a0/alpha);
                
                int threads = 256;
                int blocks = (trailing_size + threads - 1) / threads;
                hipLaunchKernelGGL(build_householder_vector, dim3(blocks), dim3(threads), 0, 0,
                                   trailing_size, j_idx, alpha, u_j1, a_col_vec, u_vec);
                hipDeviceSynchronize();
            }
            
            // Compute y = (A - U_prev * V_prev^T - V_prev * U_prev^T) * u
            // First compute y = trailing matrix of dA * u
            rocblas_ssymv(handle, uplo, trailing_size, &one, dA + start_col + start_col * lda, lda, u_vec, 1, &zero, y_vec, 1);
                        
            // Apply previous updates: y -= U_prev * (V_prev^T * u) + V_prev * (U_prev^T * u)
            if (j_idx > 0) {
                // Compute V_prev^T * u
                rocblas_sgemv(handle, rocblas_operation_transpose, trailing_size, j_idx,
                              &one, dV, trailing_size, u_vec, 1, &zero, temp_vec, 1);
                // y -= U_prev * temp_vec
                rocblas_sgemv(handle, rocblas_operation_none, trailing_size, j_idx,
                              &minus_one, dU, trailing_size, temp_vec, 1, &one, y_vec, 1);

                // Compute U_prev^T * u
                rocblas_sgemv(handle, rocblas_operation_transpose, trailing_size, j_idx,
                              &one, dU, trailing_size, u_vec, 1, &zero, temp_vec, 1);
                // y -= V_prev * temp_vec
                rocblas_sgemv(handle, rocblas_operation_none, trailing_size, j_idx,
                              &minus_one, dV, trailing_size, temp_vec, 1, &one, y_vec, 1);
            }            
            
            // Compute v = y - 0.5 * (y^T * u) * u
            float dot;
            rocblas_sdot(handle, trailing_size, y_vec, 1, u_vec, 1, &dot);
            
            int threads = 256;
            int blocks = (trailing_size + threads - 1) / threads;
            // Write v directly to its position in dV matrix
            hipLaunchKernelGGL(compute_v, dim3(blocks), dim3(threads), 0, 0,
                               trailing_size, y_vec, u_vec, dot, dV + j_idx*trailing_size);
            hipDeviceSynchronize(); 
        }
        
        // Apply symmetric rank-2p update for the trailing submatrix
        // A[start_col:n, start_col:n] -= U * V^T + V * U^T
        rocblas_ssyr2k(handle, uplo, rocblas_operation_none,
                       trailing_size, current_block_size,
                       &minus_one,
                       dU, trailing_size,
                       dV, trailing_size,
                       &one,
                       dA + start_col + start_col * lda, lda);
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
    hipFree(y_vec);
    hipFree(a_col_vec);
    hipFree(temp_vec);
    
    return hipSuccess;
}
