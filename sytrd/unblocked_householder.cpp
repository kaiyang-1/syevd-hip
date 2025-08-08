#include "householder_common.h"
#include <cmath>

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
__global__ void compute_v_unblocked(int n, const float* y, const float* u, float dot, float* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] = y[idx] - 0.5f * dot * u[idx];
}

// Kernel to extract diagonal D and sub-diagonal E from tridiagonalized A
template<rocblas_fill uplo>
__global__ void extract_tridiag_unblocked(int n, const float* A, int lda, float* D, float* E) {
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
        hipLaunchKernelGGL(compute_v_unblocked, dim3(blocks), dim3(threads), 0, 0,
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
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_lower>, dim3(blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    } else {
        hipLaunchKernelGGL(extract_tridiag_unblocked<rocblas_fill_upper>, dim3(blocks), dim3(threads), 0, 0,
                           n, dA, lda, dD, dE);
    }
    hipDeviceSynchronize();

    // Free workspace
    hipFree(u);
    hipFree(y);
    hipFree(v);

    return hipSuccess;
}
