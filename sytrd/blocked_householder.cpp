// Blocked Householder tridiagonalization implementation for symmetric matrices
#include "householder_common.h"

namespace cg = cooperative_groups;

// Kernel to build the Householder vector for a given column/row of matrix A
// 'uplo' determines if the matrix is lower or upper triangular
// Applies previous block updates, computes norm, and constructs the Householder vector in parallel
template<rocblas_fill uplo>
__global__ void build_householder_vector(int m, int j, int j_idx, const float* A, int lda, const float* dU, const float* dV, int trailing_size, float* a_col_vec, float* u_out, float* block_sums)
{
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[]; // Shared memory for reduction
    float m_sq = 0.0f;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Extract column and apply accumulated updates from previous Householder vectors
    float a_val = 0.0f;
    if (gid < m) {
        if constexpr (uplo == rocblas_fill_lower) {
            a_val = A[(j + 1 + gid) + j * lda];
        } else {
            a_val = A[j + (j + 1 + gid) * lda];
        }

        // Apply previous block Householder updates
        for (int i_idx = 0; i_idx < j_idx; ++i_idx) {
            float scalar_u = dU[j_idx + i_idx * trailing_size];
            float scalar_v = dV[j_idx + i_idx * trailing_size];
            float u_tail = dU[(j_idx + 1 + gid) + i_idx * trailing_size];
            float v_tail = dV[(j_idx + 1 + gid) + i_idx * trailing_size];
            a_val -= scalar_v * u_tail + scalar_u * v_tail;
        }

        a_col_vec[gid] = a_val;

        m_sq = a_val * a_val; // Contribution to norm^2
    } else {
        m_sq = 0.0f;
    }

    // Block-level reduction to compute sum of squares
    sdata[tid] = m_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block partial sum to global block_sums
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }

    grid.sync();

    // First block reduces all partial sums to get the full norm
    float norm = 0.0f;
    if (blockIdx.x == 0) {
        float sum = 0.0f;
        for (int b = tid; b < gridDim.x; b += blockDim.x) {
            sum += block_sums[b];
        }
        sdata[tid] = sum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        // Compute the Householder reflection parameters
        if (tid == 0) {
            norm = sqrtf(sdata[0]);
            float alpha = 0.0f;
            float u_j1 = 0.0f;

            if (norm < 1e-10f) {
                alpha = 0.0f;
                u_j1 = 1.0f;
            } else {
                float a0 = a_col_vec[0];
                float sign = (a0 >= 0.0f) ? 1.0f : -1.0f;
                alpha = -sign * norm;
                u_j1 = sqrtf(1.0f - a0/alpha);
            }
            block_sums[0] = alpha;
            block_sums[1] = u_j1;
        }
    }

    grid.sync();

    // Read alpha/u_j1 from block_sums and construct Householder vector
    float alpha = block_sums[0];
    float u_j1  = block_sums[1];

    if (gid < trailing_size) {
        if (gid <= j_idx) {
            u_out[gid] = 0.0f;
        } else if (gid == (j_idx + 1)) {
            u_out[gid] = u_j1;
        } else {
            int idx_in_col = gid - (j_idx + 1);
            float scale = 0.0f;
            if (alpha == 0.0f || u_j1 == 0.0f) {
                u_out[gid] = 0.0f;
            } else {
                scale = -1.0f / (alpha * u_j1);
                u_out[gid] = scale * a_col_vec[idx_in_col];
            }
        }
    }
}

// Kernel to compute v = y - 0.5 * (y^T u) * u, used in the symmetric rank-2 update
__global__ void compute_v(int trailing_size, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trailing_size) v[idx] = y[idx] - 0.5f * dot * u[idx];
}

// Kernel to extract diagonal (D) and off-diagonal (E) elements from the tridiagonalized matrix
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

// Main function to reduce a symmetric matrix to tridiagonal form using blocked Householder transformations
// handle: rocBLAS handle
// uplo: whether matrix is lower or upper storage
// n: matrix size
// dA: device pointer to matrix
// lda: leading dimension
// dD: output diagonal elements
// dE: output off-diagonal elements
// dTau: (not used here)
// block_size: size of the block for block Householder
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

    int num_blocks = (n - 2 + block_size - 1) / block_size;

    int threads = 256;

    // Allocate workspace for U and V matrices (n x block_size each)
    float *dU, *dV;
    hipMalloc(&dU, n * block_size * sizeof(float));
    hipMalloc(&dV, n * block_size * sizeof(float));

    // Allocate workspace for temporary vectors
    float *y_vec, *a_col_vec, *temp_vec;
    hipMalloc(&y_vec, n * sizeof(float));
    hipMalloc(&a_col_vec, n * sizeof(float));
    hipMalloc(&temp_vec, block_size * sizeof(float));

    // Allocate block_sums array (length at least max(blocks,2) as we reuse indices [0],[1])
    float* d_block_sums;
    hipMalloc(&d_block_sums, sizeof(float) * max(2, int((n + threads - 1) / threads)));

    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_col = k * block_size;
        int end_col = std::min(start_col + block_size, n - 2);
        int current_block_size = end_col - start_col;
        int trailing_size = n - start_col;

        if (current_block_size <= 0) break;

        // Process each column within the block
        for (int j = start_col; j < end_col; ++j) {
            int j_idx = j - start_col;
            int m = n - j - 1;

            // Alias for the Householder vector u
            float *u_vec = dU + j_idx * trailing_size;

            int blocks = (trailing_size + threads - 1) / threads;

            void* args[] = {
                (void*)&m,
                (void*)&j,
                (void*)&j_idx,
                (void*)&dA,
                (void*)&lda,
                (void*)&dU,
                (void*)&dV,
                (void*)&trailing_size,
                (void*)&a_col_vec,
                (void*)&u_vec,
                (void*)&d_block_sums
            };

            size_t shared_bytes = threads * sizeof(float);
            // Launch as cooperative kernel for lower or upper storage
            if (uplo == rocblas_fill_lower) {
                hipLaunchCooperativeKernel((void*)build_householder_vector<rocblas_fill_lower>,
                                           dim3(blocks), dim3(threads), args, shared_bytes, 0);
            } else {
                hipLaunchCooperativeKernel((void*)build_householder_vector<rocblas_fill_upper>,
                                           dim3(blocks), dim3(threads), args, shared_bytes, 0);
            }
            hipDeviceSynchronize();

            // Compute y = (A - U_prev * V_prev^T - V_prev * U_prev^T) * u
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

            // Write v directly to its position in dV matrix
            hipLaunchKernelGGL(compute_v, dim3(blocks), dim3(threads), 0, 0,
                               trailing_size, y_vec, u_vec, dot, dV + j_idx*trailing_size);
            hipDeviceSynchronize();
        }

        // Apply symmetric rank-2k update for the trailing submatrix
        rocblas_ssyr2k(handle, uplo, rocblas_operation_none,
                       trailing_size, current_block_size,
                       &minus_one,
                       dU, trailing_size,
                       dV, trailing_size,
                       &one,
                       dA + start_col + start_col * lda, lda);
    }

    // Extract tridiagonal elements
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
    hipFree(d_block_sums);

    return hipSuccess;
}
