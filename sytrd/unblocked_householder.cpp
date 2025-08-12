#include "householder_common.h"

namespace cg = cooperative_groups;

// Kernel to build the Householder vector 'u' for a given column/row of matrix A
// 'uplo' determines if the matrix is lower or upper triangular
// Computes the norm and constructs the Householder vector in parallel
template<rocblas_fill uplo>
__global__ void build_householder_u(int n, int i, const float* A, int lda, float* u, float* partial_sums, int m) {
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

    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel reduction within block
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0) {
        partial_sums[bidx] = sdata[0];
    }

    grid.sync();

    // First block reduces all partial sums to get the full norm
    if (bidx == 0) {
        float ps_local = 0.0f;
        for (int idx = tid; idx < gridDim.x; idx += bdim) ps_local += partial_sums[idx];
        sdata[tid] = ps_local;
        __syncthreads();
        for (int s = bdim / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) partial_sums[0] = sdata[0];
    }

    grid.sync();

    // Compute the norm of the vector
    float norm_a = sqrtf(partial_sums[0]);

    // If the norm is very small, set u to a unit vector at i+1
    if (norm_a < 1e-10f) {
        if (gid < n) {
            u[gid] = gid == i + 1 ? 1.0f : 0.0f;
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
    if (gid < n) {
        if (gid <= i) {
            u[gid] = 0.0f;
        } else if (gid == i + 1) {
            u[gid] = u_i1;
        } else {
            float ai;
            if constexpr (uplo == rocblas_fill_lower) {
                ai = A[gid + i * lda];
            } else {
                ai = A[i + gid * lda];
            }
            u[gid] = -ai / (alpha * u_i1);
        }
    }

    return;
}

// Kernel to compute v = y - 0.5 * dot * u, used in the symmetric rank-2 update
__global__ void compute_v_unblocked(int n, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = y[idx] - 0.5f * dot * u[idx];
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
    int blocks = (n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    float* d_partial_sums = nullptr;
    hipMalloc(&d_partial_sums, sizeof(float) * blocks);

    // Main loop: reduce each column/row to tridiagonal form
    for (int i = 0; i < n - 2; ++i) {
        int m = n - i - 1;

        size_t shmem_bytes = threads * sizeof(float);
        void* args[] = { &n, &i, &dA, &lda, &u, &d_partial_sums, &m };

        // Build Householder vector for current column/row
        if (uplo == rocblas_fill_lower) {
            hipLaunchCooperativeKernel((void*)build_householder_u<rocblas_fill_lower>,
                                       dim3(blocks), dim3(threads), args, shmem_bytes, 0);
        } else {
            hipLaunchCooperativeKernel((void*)build_householder_u<rocblas_fill_upper>,
                                       dim3(blocks), dim3(threads), args, shmem_bytes, 0);
        }
        hipDeviceSynchronize();

        // Compute y = A * u
        rocblas_ssymv(handle, uplo, n, &one, dA, lda, u, 1, &zero, y, 1);

        // Compute dot = y^T * u
        float dot;
        rocblas_sdot(handle, n, y, 1, u, 1, &dot);

        // Compute v = y - 0.5 * dot * u
        hipLaunchKernelGGL(compute_v_unblocked, dim3(blocks), dim3(threads), 0, 0, n, y, u, dot, v);
        hipDeviceSynchronize();

        // Symmetric rank-2 update: A = A - u*v^T - v*u^T
        rocblas_ssyr2(handle, uplo, n, &minus_one, u, 1, v, 1, dA, lda);
    }

    // Extract diagonal and off-diagonal elements to output arrays
    if (uplo == rocblas_fill_lower) {
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_lower>, dim3(blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    } else {
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_upper>, dim3(blocks), dim3(threads), 0, 0,
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
