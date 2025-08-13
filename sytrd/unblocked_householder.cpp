#include "householder_common.h"

namespace cg = cooperative_groups;

// Warp-level reduction helper (assumes blockDim.x is a multiple of warpSize)
__device__ inline float warp_reduce_sum(float val) {
    // HIP provides __shfl_down; width defaults to warpSize
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Kernel to build the Householder vector 'u' for a given column/row of matrix A
// 'uplo' determines if the matrix is lower or upper triangular
// Computes the norm and constructs the Householder vector in parallel
template<rocblas_fill uplo>
__global__ void build_householder_u(int trailing_size, int i, const float* A, int lda, float* u, float* partial_sums, int m) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[]; // Shared memory for reduction
    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bdim = blockDim.x;
    int gid = bidx * bdim + tid;

    float local_sum = 0.0f;

    // Select the base of the vector to operate on, depending on matrix storage (lower/upper)
    const float* vec_base = nullptr;
    int stride = 1;
    if constexpr (uplo == rocblas_fill_lower) {
        vec_base = A + (i + 1) + i * lda;
        stride = 1;
    } else {
        vec_base = A + i + (i + 1) * lda;
        stride = lda;
    }

    // Compute local sum of squares for the norm
    if (gid < m) {
        float val = vec_base[gid * stride];
        local_sum += val * val;
    }

    // Warp-level reduction then consolidate warp sums
    float sum = warp_reduce_sum(local_sum);
    int lane = tid & (warpSize - 1);
    int wid  = tid / warpSize;
    if (lane == 0) sdata[wid] = sum; // one float per warp
    __syncthreads();
    // Reduce warp sums using first warp
    if (wid == 0) {
        float block_sum = (lane < (bdim + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial_sums[bidx] = block_sum;
    }

    grid.sync();

    // First block reduces all partial sums to get the full norm
    if (bidx == 0) {
        // Block 0 aggregates per-block partial sums
        float ps_local = 0.0f;
        for (int idx = tid; idx < gridDim.x; idx += bdim) ps_local += partial_sums[idx];
        // Reuse warp reduction
        float total = warp_reduce_sum(ps_local);
        if (lane == 0) sdata[wid] = total;
        __syncthreads();
        if (wid == 0) {
            float block0_sum = (lane < (bdim + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
            block0_sum = warp_reduce_sum(block0_sum);
            if (lane == 0) partial_sums[0] = block0_sum;
        }
    }

    grid.sync();

    // Compute the norm of the vector
    float norm_a = sqrtf(partial_sums[0]);

    // If the norm is very small, set u to a unit vector at position 1 (second element)
    if (norm_a < 1e-10f) {
        if (gid < trailing_size) {
            u[gid] = gid == 1 ? 1.0f : 0.0f;
        }
        return;
    }

    // Get the first element of the vector
    float a0;
    if constexpr (uplo == rocblas_fill_lower) {
        a0 = A[(i + 1) + i * lda];
    } else {
        a0 = A[i + (i + 1) * lda];
    }

    // Compute the Householder reflection parameters
    float sign = (a0 >= 0.0f) ? 1.0f : -1.0f;
    float alpha = -sign * norm_a;

    float u_i1 = sqrtf(fmaxf(0.0f, 1.0f - a0 / alpha));

    // Build the Householder vector u
    if (gid < trailing_size) {
        if (gid == 0) {
            u[gid] = 0.0f;
        } else if (gid == 1) {
            u[gid] = u_i1;
        } else {
            float ai;
            if constexpr (uplo == rocblas_fill_lower) {
                ai = A[(i + gid) + i * lda];
            } else {
                ai = A[i + (i + gid) * lda];
            }
            u[gid] = -ai / (alpha * u_i1);
        }
    }

    return;
}

// Fused kernel to compute v = y - 0.5 * (y^T * u) * u
__global__ void fused_dot_compute_v_unblocked(int n,
                                              const float* __restrict__ y,
                                              const float* __restrict__ u,
                                              float* __restrict__ v,
                                              float* __restrict__ partial_sums) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[]; // per-block reduction buffer
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load values once from global memory (or zeros if out-of-range) and keep in registers.
    float y_val = 0.0f;
    float u_val = 0.0f;
    if (gid < n) {
        y_val = y[gid];
        u_val = u[gid];
    }

    float prod = y_val * u_val; // local contribution to dot
    // Warp reduction
    float wsum = warp_reduce_sum(prod);
    int lane = tid & (warpSize - 1);
    int wid  = tid / warpSize;
    if (lane == 0) sdata[wid] = wsum;
    __syncthreads();
    if (wid == 0) {
        float block_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial_sums[blockIdx.x] = block_sum;
    }

    // Grid-wide sync so all partial sums are written
    grid.sync();

    // Block 0 performs final reduction of block partial sums
    if (blockIdx.x == 0) {
        float sum = 0.0f;
        for (int idx = tid; idx < gridDim.x; idx += blockDim.x) sum += partial_sums[idx];
        float reduced = warp_reduce_sum(sum);
        if (lane == 0) sdata[wid] = reduced;
        __syncthreads();
        if (wid == 0) {
            float final_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
            final_sum = warp_reduce_sum(final_sum);
            if (lane == 0) partial_sums[0] = final_sum;
        }
    }

    // Sync so all threads can read the final dot value
    grid.sync();

    float dot = partial_sums[0];

    // Use cached register values to compute v without reloading y/u
    if (gid < n) v[gid] = y_val - 0.5f * dot * u_val;
}

// Kernel to extract the diagonal (D) and off-diagonal (E) elements from the tridiagonalized matrix
template<rocblas_fill uplo>
__global__ void extract_tridiag_unblocked(int n, const float* A, int lda, float* D, float* E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = A[i + i * lda]; // Diagonal element
        if (i < n-1) {
            if constexpr (uplo == rocblas_fill_lower) {
                E[i] = A[(i+1) + i * lda]; // Sub-diagonal
            } else {
                E[i] = A[i + (i+1) * lda]; // Super-diagonal
            }
        }
    }
}


// Main function to reduce a symmetric matrix to tridiagonal form using Householder transformations (unblocked version)
// handle: rocBLAS handle
// uplo: whether matrix is lower or upper storage
// n: matrix size
// dA: device pointer to matrix
// lda: leading dimension
// dD: output diagonal elements
// dE: output off-diagonal elements
// dTau: (not used here)
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
    // Allocate workspace vectors on device
    float *u, *y, *v;
    hipMalloc(&u, n * sizeof(float));
    hipMalloc(&y, n * sizeof(float));
    hipMalloc(&v, n * sizeof(float));
    
    const float one = 1.0f;
    const float minus_one = -1.0f;
    const float zero = 0.0f;

    int threads = 256;
    int max_blocks = (n + threads - 1) / threads;
    float* d_partial_sums = nullptr;
    hipMalloc(&d_partial_sums, sizeof(float) * max_blocks);

    // Get warp size for the current device
    int device_warp_size;
    hipDeviceGetAttribute(&device_warp_size, hipDeviceAttributeWarpSize, 0);

    // Main loop: reduce each column/row to tridiagonal form
    for (int i = 0; i < n - 2; ++i) {
        int m = n - i - 1;
        int trailing_size = n - i;  // Size of the trailing submatrix

        // Calculate blocks needed for current trailing size
        int blocks = (trailing_size + threads - 1) / threads;
        if (blocks < 1) blocks = 1;

        {
            void* args[] = { &trailing_size, &i, &dA, &lda, &u, &d_partial_sums, &m };

            // Calculate shared memory needed: one float per warp for reduction
            size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
            size_t shmem_bytes = warps_per_block * sizeof(float);

            // Build Householder vector for current column/row
            if (uplo == rocblas_fill_lower) {
                hipLaunchCooperativeKernel((void*)build_householder_u<rocblas_fill_lower>,
                                           dim3(blocks), dim3(threads), args, shmem_bytes, 0);
            } else {
                hipLaunchCooperativeKernel((void*)build_householder_u<rocblas_fill_upper>,
                                           dim3(blocks), dim3(threads), args, shmem_bytes, 0);
            }
        }

        // Compute y = A * u
        float* dA_trailing = dA + i * lda + i;  // Pointer to A[i:, i:] submatrix
        
        rocblas_ssymv(handle, uplo, trailing_size, &one, dA_trailing, lda, u, 1, &zero, y, 1);

        // Fused computation of v = y - 0.5 * (y^T * u) * u
        {
            void* args[] = { &trailing_size, &y, &u, &v, &d_partial_sums };
            
            // Calculate shared memory needed: one float per warp for reduction
            size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
            size_t shmem_bytes = warps_per_block * sizeof(float);

            hipLaunchCooperativeKernel((void*)fused_dot_compute_v_unblocked,
                                       dim3(blocks), dim3(threads), args, shmem_bytes, 0);
        }

        // Symmetric rank-2 update: A = A - u*v^T - v*u^T
        rocblas_ssyr2(handle, uplo, trailing_size, &minus_one, u, 1, v, 1, dA_trailing, lda);
    }

    // Extract diagonal and off-diagonal elements to output arrays
    int final_blocks = (n + threads - 1) / threads;
    if (uplo == rocblas_fill_lower) {
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_lower>, dim3(final_blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    } else {
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_upper>, dim3(final_blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    }
    hipDeviceSynchronize();

    // Free device memory
    hipFree(u);
    hipFree(y);
    hipFree(v);
    hipFree(d_partial_sums);

    return hipSuccess;
}
