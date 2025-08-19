#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace cg = cooperative_groups;

// 2D index calculation for column-major storage
__device__ __host__ inline int idx2D(const int i, const int j, const int lda) { return j * lda + i; }

// Warp-level reduction for summing values within a warp
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Kernel to compute Householder reflector (larfg) for a column vector
__global__ void larfg_kernel(int m, int i, float* __restrict__ A, int lda,
                             float* __restrict__ dE, float* __restrict__ dTau,
                             float* __restrict__ partial_sums) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int bdim = blockDim.x;
    int gid  = blockIdx.x * bdim + tid;

    float* col_base = A + idx2D(i + 1, i, lda);

    float val, local = 0.0f;

    if (gid < m) {
        val = col_base[gid];
        local = gid > 0? val * val : 0.0f;
    }
    float tail_sum = warp_reduce_sum(local);
    int lane = tid & (warpSize - 1);
    int wid  = tid / warpSize;
    if (lane == 0) sdata[wid] = tail_sum;
    __syncthreads();
    if (wid == 0) {
        float block_sum = (lane < (bdim + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial_sums[blockIdx.x] = block_sum;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        float ps = 0.0f;
        for (int idx = tid; idx < gridDim.x; idx += bdim) ps += partial_sums[idx];
        float reduced = warp_reduce_sum(ps);
        if (lane == 0) sdata[wid] = reduced;
        __syncthreads();
        if (wid == 0) {
            float final_sum = (lane < (bdim + warpSize - 1)/warpSize) ? sdata[lane] : 0.0f;
            final_sum = warp_reduce_sum(final_sum);
            if (lane == 0) partial_sums[0] = final_sum;
        }
    }
    float alpha = col_base[0];

    grid.sync();

    float sigma2 = partial_sums[0];
    float beta, tau, scale;
    if (sigma2 < 1e-10f) {
        tau   = 0.0f;
        beta  = alpha;
        scale = 0.0f;
    } else {
        float sigma = sqrtf(sigma2);
        float r = hypotf(alpha, sigma);
        beta = -copysignf(r, alpha == 0.0f ? 1.0f : alpha);
        tau  = (beta - alpha) / beta;
        scale = 1.0f / (alpha - beta);
    }

    val = gid == 0 ? 1.0f : val * scale;
    if (gid < m) {
        col_base[gid] = val;
        if (gid == 0) {
            dE[i]   = beta;
            dTau[i] = tau;
        }
    }
}

// Optimized single-block version for small vectors (m <= 2048)
__global__ void larfg_kernel_small(int m, int i, float* __restrict__ A, int lda,
                                   float* __restrict__ dE, float* __restrict__ dTau) {
    extern __shared__ float sdata[]; // size >= blockDim.x
    int tid = threadIdx.x;
    float* col_base = A + idx2D(i + 1, i, lda);

    // Accumulate sum of squares of tail (excluding first element)
    float local_sum = 0.0f;
    float alpha = 0.0f;
    for (int idx = tid; idx < m; idx += blockDim.x) {
        float val = col_base[idx];
        if (idx == 0) alpha = val; else local_sum += val * val;
    }

    // Reduction within block for local_sum
    sdata[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float sigma2 = sdata[0];

    // Broadcast alpha to all threads (alpha only known by threads that saw idx==0)
    // Use shared memory position 1 for alpha broadcast if needed.
    if (tid == 0) sdata[1] = alpha;
    __syncthreads();
    alpha = sdata[1];

    float beta, tau, scale;
    if (sigma2 < 1e-10f) {
        tau = 0.0f;
        beta = alpha;
        scale = 0.0f;
    } else {
        float sigma = sqrtf(sigma2);
        float r = hypotf(alpha, sigma);
        beta = -copysignf(r, alpha == 0.0f ? 1.0f : alpha);
        tau = (beta - alpha) / beta;
        scale = 1.0f / (alpha - beta);
    }

    // Store tau & beta
    if (tid == 0) {
        dE[i] = beta;
        dTau[i] = tau;
    }

    // Scale tail and set first element = 1
    for (int idx = tid; idx < m; idx += blockDim.x) {
        if (idx == 0) col_base[0] = 1.0f; else col_base[idx] *= scale;
    }
}

// Kernel to compute dot product, scale, and axpy in a fused manner
__global__ void fused_dot_scale_axpy(int n,
                                     const float* __restrict__ v,
                                     float* __restrict__ w,
                                     float* __restrict__ partial_sums,
                                     const float* __restrict__ tau_ptr) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float tau = *tau_ptr;

    float w_val = 0.0f;
    float v_val = 0.0f;
    if (gid < n) {
        w_val = w[gid];
        v_val = v[gid];
    }

    float prod = w_val * v_val;
    float wsum = warp_reduce_sum(prod);
    int lane = tid & (warpSize - 1);
    int wid  = tid / warpSize;
    if (lane == 0) sdata[wid] = wsum;
    __syncthreads();
    if (wid == 0) {
        float block_sum = (lane < (blockDim.x + warpSize - 1)/warpSize) ? sdata[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial_sums[blockIdx.x] = block_sum;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        float sum = 0.0f;
        for (int idx = tid; idx < gridDim.x; idx += blockDim.x) sum += partial_sums[idx];
        float red = warp_reduce_sum(sum);
        if (lane == 0) sdata[wid] = red;
        __syncthreads();
        if (wid == 0) {
            float final_sum = (lane < (blockDim.x + warpSize - 1)/warpSize) ? sdata[lane] : 0.0f;
            final_sum = warp_reduce_sum(final_sum);
            if (lane == 0) partial_sums[0] = final_sum;
        }
    }

    grid.sync();

    float dot = partial_sums[0];
    float alpha_corr = -0.5f * tau * dot;

    if (gid < n) {
        w[gid] = tau * (w_val + alpha_corr * v_val);
    }
}

// Optimized single-block version for small vectors (n <= 2048)
__global__ void fused_dot_scale_axpy_small(int n,
                                           const float* __restrict__ v,
                                           float* __restrict__ w,
                                           const float* __restrict__ tau_ptr) {
    extern __shared__ float sdata[]; // size >= blockDim.x
    int tid = threadIdx.x;
    float tau = *tau_ptr;

    // Compute dot product within single block
    float local_dot = 0.0f;
    float w_val = 0.0f;
    float v_val = 0.0f;
    
    for (int idx = tid; idx < n; idx += blockDim.x) {
        w_val = w[idx];
        v_val = v[idx];
        local_dot += w_val * v_val;
    }

    // Reduction within block for dot product
    sdata[tid] = local_dot;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    float dot = sdata[0];

    // Compute correction factor
    float alpha_corr = -0.5f * tau * dot;

    // Apply fused scale and axpy operation
    for (int idx = tid; idx < n; idx += blockDim.x) {
        w_val = w[idx];
        v_val = v[idx];
        w[idx] = tau * (w_val + alpha_corr * v_val);
    }
}

// Kernel to set up tridiagonal matrix elements D and E from A
__global__ void setup_tridiagonal(int n, float* A, int lda, float* D, float* E) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = A[idx2D(i, i, lda)];

        if (i < n - 2) {
            A[idx2D(i + 1, i, lda)] = E[i];
        } else if (i == n - 2) {
            E[i] =  A[idx2D(i + 1, i, lda)];
        }
    }
}

// Unblocked symmetric tridiagonal reduction (panel size = 1)
extern "C" hipError_t hip_sytd2(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau,
    int device_warp_size,
    float* w_vec,
    float* d_partial_sums,
    const float* d_scalars)
{
    const int threads = 256;

    for (int j = 0; j < n - 2; ++j) {
        int m = n - j - 1;
        int blocks = (m + threads - 1) / threads;

        if (m <= 2048) {
            dim3 blk(256); dim3 grd(1);
            size_t shmem_bytes = 256 * sizeof(float);
            hipLaunchKernelGGL(larfg_kernel_small, grd, blk, shmem_bytes, 0,
                               m, j, dA, lda, dE, dTau);
        } else {
            void* args[] = { &m, &j, &dA, &lda, &dE, &dTau, &d_partial_sums };
            size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
            size_t shmem_bytes = warps_per_block * sizeof(float);
            hipLaunchCooperativeKernel((void*)larfg_kernel, dim3(blocks), dim3(threads), args, shmem_bytes, 0);
        }

        float* A22 = dA + idx2D(j + 1, j + 1, lda);
        float* v_vec = dA + idx2D(j + 1, j, lda);
        rocblas_ssymv(handle, rocblas_fill_lower, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);

        {
            float *tau_ptr = dTau + j;
            if (m <= 2048) {
                // Single-block optimized path
                dim3 blk(256); dim3 grd(1);
                size_t shmem_bytes = 256 * sizeof(float);
                hipLaunchKernelGGL(fused_dot_scale_axpy_small, grd, blk, shmem_bytes, 0,
                                   m, v_vec, w_vec, tau_ptr);
            } else {
                void* args[] = { &m, &v_vec, &w_vec, &d_partial_sums, &tau_ptr };
                size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
                size_t shmem_bytes = warps_per_block * sizeof(float);
                hipLaunchCooperativeKernel((void*)fused_dot_scale_axpy, dim3(blocks), dim3(threads), args, shmem_bytes, 0);
            }
        }

        rocblas_ssyr2(handle, rocblas_fill_lower, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
    }

    return hipSuccess;
}

// Blocked panel reduction for symmetric tridiagonalization
extern "C" hipError_t hip_latrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dW,
    int ldw,
    float* dD,
    float* dE,
    float* dTau,
    int panel_size,
    int device_warp_size,
    float* d_tmp_vec,
    float* d_partial_sums,
    const float* d_scalars)
{
    const int threads = 256;

    for (int j = 0; j < panel_size; ++j) {
        int m = n - j - 1;

        if (j > 0) {
            // Compute the update to column j of A
            // A[j:, j] -= A[j:, :j] * W[j, :j]^T + W[j:, :j] * A[j, :j]^T
            int len = n - j;

            rocblas_sgemv(handle, rocblas_operation_none,
                          len, j,
                          d_scalars + 1, // -1.0f
                          dA + j, lda,
                          dW + j, ldw,
                          d_scalars, // 1.0f
                          dA + idx2D(j, j, lda), 1);

            rocblas_sgemv(handle, rocblas_operation_none,
                          len, j,
                          d_scalars + 1, // -1.0f
                          dW + j, ldw,
                          dA + j, lda,
                          d_scalars, // 1.0f
                          dA + idx2D(j, j, lda), 1);
        }

        int blocks = (m + threads - 1) / threads;

        if (m <= 2048) {
            // Single-block optimized path
            dim3 blk(256); dim3 grd(1);
            size_t shmem_bytes = 256 * sizeof(float);
            hipLaunchKernelGGL(larfg_kernel_small, grd, blk, shmem_bytes, 0,
                               m, j, dA, lda, dE, dTau);
        } else {
            void* args[] = { &m, &j, &dA, &lda, &dE, &dTau, &d_partial_sums };
            size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
            size_t shmem_bytes = warps_per_block * sizeof(float);
            hipLaunchCooperativeKernel((void*)larfg_kernel, dim3(blocks), dim3(threads), args, shmem_bytes, 0);
        }

        float* v_vec = dA + idx2D(j + 1, j, lda);
        float* w_vec = dW + idx2D(j + 1, j, ldw);

        // w = A[j+1:, j+1:] @ v
        //   - A[j+1:, :j] @ (W[j+1:, :j].T @ v)
        //   - W[j+1:, :j] @ (A[j+1:, :j].T @ v)
        rocblas_ssymv(handle, uplo, m, d_scalars + 0,
                      dA + idx2D(j + 1, j + 1, lda), lda, 
                      v_vec, 1, d_scalars + 2, w_vec, 1);

        if (j > 0) {
            rocblas_sgemv(handle, rocblas_operation_transpose, m, j,
                          d_scalars + 0, dW + j + 1, ldw, v_vec, 1, d_scalars + 2, d_tmp_vec, 1);
            rocblas_sgemv(handle, rocblas_operation_none, m, j,
                          d_scalars + 1, dA + j + 1, lda, d_tmp_vec, 1, d_scalars + 0, w_vec, 1);

            rocblas_sgemv(handle, rocblas_operation_transpose, m, j,
                          d_scalars + 0, dA + j + 1, lda, v_vec, 1, d_scalars + 2, d_tmp_vec, 1);
            rocblas_sgemv(handle, rocblas_operation_none, m, j,
                          d_scalars + 1, dW + j + 1, ldw, d_tmp_vec, 1, d_scalars + 0, w_vec, 1);
        }

        {
            float *tau_ptr = dTau + j;
            if (m <= 2048) {
                // Single-block optimized path
                dim3 blk(256); dim3 grd(1);
                size_t shmem_bytes = 256 * sizeof(float);
                hipLaunchKernelGGL(fused_dot_scale_axpy_small, grd, blk, shmem_bytes, 0,
                                   m, v_vec, w_vec, tau_ptr);
            } else {
                int blocks = (m + threads - 1) / threads;
                void* args[] = { &m, &v_vec, &w_vec, &d_partial_sums, &tau_ptr };
                size_t warps_per_block = (threads + device_warp_size - 1) / device_warp_size;
                size_t shmem_bytes = warps_per_block * sizeof(float);
                hipLaunchCooperativeKernel((void*)fused_dot_scale_axpy, dim3(blocks), dim3(threads), args, shmem_bytes, 0);
            }
        }
    }
    return hipSuccess;
}

// Main symmetric tridiagonal reduction routine (blocked + unblocked)
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
    if (n < 3) return hipSuccess;

    const int threads = 256;
    const int panel_size = 64;

    // Allocate device memory for scalar constants
    float h_scalars[3] = {1.0f, -1.0f, 0.0f};
    float *d_scalars;
    hipMalloc(&d_scalars, 3 * sizeof(float));
    hipMemcpy(d_scalars, h_scalars, 3 * sizeof(float), hipMemcpyHostToDevice);
    
    int device_warp_size;
    hipDeviceGetAttribute(&device_warp_size, hipDeviceAttributeWarpSize, 0);
    float* dW;
    hipMalloc(&dW, n * panel_size * sizeof(float));
    float* d_tmp_vec;
    hipMalloc(&d_tmp_vec, panel_size * sizeof(float));
    float* d_partial_sums;
    hipMalloc(&d_partial_sums, sizeof(float) * int((n + threads - 1) / threads));

    int j = 0;
    while(j < n - panel_size) {
        float* dA_panel = dA + idx2D(j, j, lda);
        int ldw = n - j;

        hip_latrd(handle,
                  uplo,
                  n - j,
                  dA + idx2D(j, j, lda),
                  lda,
                  dW,
                  ldw,
                  dD + j,
                  dE + j,
                  dTau + j,
                  panel_size,
                  device_warp_size,
                  d_tmp_vec,
                  d_partial_sums,
                  d_scalars);

        j += panel_size;

        rocblas_ssyr2k(handle, uplo, rocblas_operation_none,
                       n - j, panel_size,
                       d_scalars + 1, // minus_one
                       dA_panel + panel_size, lda,
                       dW + panel_size, ldw,
                       d_scalars + 0, // one
                       dA + idx2D(j, j, lda), lda);
    }
    if (j < n - 2) {
        hip_sytd2(handle,
                  uplo,
                  n - j,
                  dA + idx2D(j, j, lda),
                  lda,
                  dD + j,
                  dE + j,
                  dTau + j,
                  device_warp_size,
                  dW,
                  d_partial_sums,
                  d_scalars);
    }

    int tridiag_blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(setup_tridiagonal, dim3(tridiag_blocks), dim3(threads), 0, 0,
                       n, dA, lda, dD, dE);
    hipMemset(dTau + n - 2, 0, sizeof(float));
    hipDeviceSynchronize();
    hipFree(dW);
    hipFree(d_tmp_vec);
    hipFree(d_partial_sums);
    hipFree(d_scalars);

    return hipSuccess;
}

// Generate a random symmetric matrix (column-major)
void generate_symmetric_matrix(std::vector<float>& A, int n, unsigned int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::fill(A.begin(), A.end(), 0.0f);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            float value = dist(rng);
            A[i + j * n] = value;
            if (i != j) {
                A[j + i * n] = value;
            }
        }
    }
}

// Struct to hold benchmark results
struct BenchmarkResult {
    double hip_avg_time_ms = 0.0;
    double rocsolver_avg_time_ms = 0.0;
    double speedup_rocsolver_over_hip = 0.0;
    bool   validated = false;
    float  max_abs_diff_D = 0.0f;
    float  max_abs_diff_E = 0.0f;
    float  max_abs_diff_Tau = 0.0f;
};

// Benchmark custom and rocSOLVER tridiagonalization, optionally validate results
BenchmarkResult benchmark(int n, bool validate, int iterations, int warmup) {
    const int lda = n;
    BenchmarkResult result;

    std::vector<float> hA(n * n);
    std::vector<float> hD_hip(n), hE_hip(std::max(0, n - 1)), hTau_hip(std::max(0, n - 1));
    std::vector<float> hD_ref(n), hE_ref(std::max(0, n - 1)), hTau_ref(std::max(0, n - 1));
    float *dA_hip, *dD_hip, *dE_hip, *dTau_hip;
    hipMalloc(&dA_hip, n * lda * sizeof(float));
    hipMalloc(&dD_hip, n * sizeof(float));
    hipMalloc(&dE_hip, (n - 1) * sizeof(float));
    hipMalloc(&dTau_hip, (n - 1) * sizeof(float));
    float *dA_ref, *dD_ref, *dE_ref, *dTau_ref;
    hipMalloc(&dA_ref, n * lda * sizeof(float));
    hipMalloc(&dD_ref, n * sizeof(float));
    hipMalloc(&dE_ref, (n - 1) * sizeof(float));
    hipMalloc(&dTau_ref, (n - 1) * sizeof(float));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    generate_symmetric_matrix(hA, n);

    for (int w = 0; w < warmup; ++w) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
    }
    double hip_total_us = 0.0;
    double ref_total_us = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        auto start_hip = std::chrono::high_resolution_clock::now();
        hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        auto end_hip = std::chrono::high_resolution_clock::now();
        hip_total_us += std::chrono::duration<double, std::micro>(end_hip - start_hip).count();
        auto start_ref = std::chrono::high_resolution_clock::now();
        rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        auto end_ref = std::chrono::high_resolution_clock::now();
        ref_total_us += std::chrono::duration<double, std::micro>(end_ref - start_ref).count();
    }

    result.hip_avg_time_ms = (hip_total_us / iterations) / 1000.0;
    result.rocsolver_avg_time_ms = (ref_total_us / iterations) / 1000.0;
    if (result.hip_avg_time_ms > 0.0)
        result.speedup_rocsolver_over_hip = result.rocsolver_avg_time_ms / result.hip_avg_time_ms;

    if (validate) {
        generate_symmetric_matrix(hA, n, 12345u);
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(float), hipMemcpyHostToDevice);

        hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);

        hipMemcpy(hD_hip.data(), dD_hip, n * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(hE_hip.data(), dE_hip, (n - 1) * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_hip.data(), dTau_hip, (n - 1) * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(hD_ref.data(), dD_ref, n * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(hE_ref.data(), dE_ref, (n - 1) * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_ref.data(), dTau_ref, (n - 1) * sizeof(float), hipMemcpyDeviceToHost);

        float max_d = 0.0f, max_e = 0.0f, max_tau = 0.0f;
        for (int i = 0; i < n; ++i) {
            max_d = std::max(max_d, std::fabs(hD_hip[i] - hD_ref[i]));
        }
        for (int i = 0; i < n - 1; ++i) {
            max_e = std::max(max_e, std::fabs(hE_hip[i] - hE_ref[i]));
            max_tau = std::max(max_tau, std::fabs(hTau_hip[i] - hTau_ref[i]));
        }
        result.max_abs_diff_D = max_d;
        result.max_abs_diff_E = max_e;
        result.max_abs_diff_Tau = max_tau;
        result.validated = true;
    }

    hipFree(dA_hip); hipFree(dD_hip); hipFree(dE_hip); hipFree(dTau_hip);
    hipFree(dA_ref); hipFree(dD_ref); hipFree(dE_ref); hipFree(dTau_ref);
    rocblas_destroy_handle(handle);

    return result;
}

// Print table header for benchmark results
void print_table_header(bool validate) {
    printf("Matrix Size |   hip_ssytrd (ms) | rocSOLVER (ms) |   Speedup");
    if (validate) {
        printf(" |      max|ΔD| |      max|ΔE| |    max|ΔTau| \n");
        printf("---------------------------------------------------------------------------------------------------------\n");
    } else {
        printf("\n");
        printf("------------------------------------------------------------\n");
    }
}

// Print a single row of benchmark results
void print_comparison_result(int n, const BenchmarkResult& result, bool validate) {
    printf("%11d | %17.3f | %14.3f | %9.3f", n,
           result.hip_avg_time_ms,
           result.rocsolver_avg_time_ms,
           result.speedup_rocsolver_over_hip);
    if (validate && result.validated) {
        printf(" | %12.3e | %12.3e | %12.3e",
               result.max_abs_diff_D,
               result.max_abs_diff_E,
               result.max_abs_diff_Tau);
    }
    printf("\n");
}

// Print usage/help message
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -n size1 size2 ...    Matrix sizes to test\n");
    printf("  -v                    Enable validation\n");
    printf("  -i iterations         Number of benchmark iterations (default: 10)\n");
    printf("  -w warmup_runs        Number of warmup runs (default: 3)\n");
    printf("  -h                    Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -n 128 256 512 -v\n", program_name);
}

// Main entry point: parse arguments, run benchmarks, print results
int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes;
    bool validate = false;
    int iterations = 10;
    int warmup = 3;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (matrix_sizes.empty()) {
        printf("Error: No matrix sizes specified.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Householder Tridiagonalization Benchmark\n");
    printf("=========================================\n");
    printf("Iterations: %d\n", iterations);
    printf("Warmup runs: %d\n", warmup);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    print_table_header(validate);
    for (int n : matrix_sizes) {
        BenchmarkResult result = benchmark(n, validate, iterations, warmup);
        print_comparison_result(n, result, validate);
    }

    printf("\n");

    return 0;
}