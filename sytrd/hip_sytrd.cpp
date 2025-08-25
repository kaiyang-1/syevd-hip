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

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

// 2D index calculation for column-major storage
__device__ __host__ inline int idx2D(const int i, const int j, const int lda) { return j * lda + i; }

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction
__device__ __forceinline__ void block_reduce_sum(float val, float *smem, int tid, int blockDimX) {
    val = warp_reduce_sum(val);

    if (blockDimX > warpSize) {
        int lane = tid & (warpSize - 1);
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : 0.0f;
            val = warp_reduce_sum(val);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }

    // __syncthreads();
    // sync not needed if only thread 0 reads from smem[0]
}

// Kernel to compute Householder reflector (larfg) for a column vector
__global__ void larfg_kernel(int m, int i, float* __restrict__ A, int lda,
                             float* __restrict__ dE, float* __restrict__ dTau) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float* col_base = A + idx2D(i + 1, i, lda);

    // Accumulate sum of squares of tail (excluding first element)
    float local_sum = 0.0f;
    float alpha = 0.0f;

    int m_vec4 = m / 4;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        float4 val4 = reinterpret_cast<const float4*>(&col_base[idx4 * 4])[0];
        
        // Handle first element specially (alpha extraction)
        if (idx4 == 0) {
            alpha = val4.x;
            local_sum += (val4.y * val4.y + val4.z * val4.z + val4.w * val4.w);
        } else {
            local_sum += (val4.x * val4.x + val4.y * val4.y + 
                         val4.z * val4.z + val4.w * val4.w);
        }
    }
    
    for (int idx = 4 * m_vec4 + tid; idx < m; idx += blockDim.x) {
        float val = col_base[idx];
        if (idx == 0 && m_vec4 == 0) {
            alpha = val;
        } else if (idx > 0) {
            local_sum += val * val;
        }
    }

    // Reduction within block for local_sum
    block_reduce_sum(local_sum, sdata, tid, blockDim.x);
    __syncthreads();
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

    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        if (idx4 == 0) {
            // Handle first element specially (set to 1.0f) and scale the rest
            float4 val4 = reinterpret_cast<const float4*>(&col_base[idx4 * 4])[0];
            val4.x = 1.0f;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<float4*>(&col_base[idx4 * 4])[0] = val4;
        } else {
            float4 val4 = reinterpret_cast<const float4*>(&col_base[idx4 * 4])[0];
            val4.x *= scale;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<float4*>(&col_base[idx4 * 4])[0] = val4;
        }
    }
    
    for (int idx = 4 * m_vec4 + tid; idx < m; idx += blockDim.x) {
        if (idx == 0 && m_vec4 == 0) {
            col_base[0] = 1.0f;
        } else if (idx > 0) {
            col_base[idx] *= scale;
        }
    }
}

// Kernel to compute dot product, scale, and axpy in a fused manner
__global__ void fused_dot_scale_axpy(int n,
                                     const float* __restrict__ v,
                                     float* __restrict__ w,
                                     const float* __restrict__ tau_ptr) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float tau = *tau_ptr;

    // Compute dot product within single block
    float local_dot = 0.0f;
    
    int n_vec4 = n / 4;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        float4 w_val4 = reinterpret_cast<const float4*>(&w[idx4 * 4])[0];
        float4 v_val4 = reinterpret_cast<const float4*>(&v[idx4 * 4])[0];
        
        local_dot += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
    }
    
    // Handle remaining elements for dot product
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        float w_val = w[idx];
        float v_val = v[idx];
        local_dot += w_val * v_val;
    }

    // Reduction within block for dot product
    block_reduce_sum(local_dot, sdata, tid, blockDim.x);
    __syncthreads();
    float dot = sdata[0];

    // Compute correction factor
    float alpha_corr = -0.5f * tau * dot;

    // Apply scale and axpy operation
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        float4 w_val4 = reinterpret_cast<const float4*>(&w[idx4 * 4])[0];
        float4 v_val4 = reinterpret_cast<const float4*>(&v[idx4 * 4])[0];
        
        w_val4.x = tau * (w_val4.x + alpha_corr * v_val4.x);
        w_val4.y = tau * (w_val4.y + alpha_corr * v_val4.y);
        w_val4.z = tau * (w_val4.z + alpha_corr * v_val4.z);
        w_val4.w = tau * (w_val4.w + alpha_corr * v_val4.w);
        
        reinterpret_cast<float4*>(&w[idx4 * 4])[0] = w_val4;
    }
    
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        float w_val = w[idx];
        float v_val = v[idx];
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

// Kernel to compute small-size symmetric-matrix-vector multiplication in single block
__global__ void small_ssymv_kernel(int n,
                                   float alpha,
                                   const float* __restrict__ A, int lda,
                                   const float* __restrict__ v,
                                   float beta,
                                   float* __restrict__ w) {
    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float sv[];
    
    if (tid < n) {
        sv[tid] = v[tid];
    }
    __syncthreads();
    
    if (row >= n) return;
    
    float sum = 0.0f;
    
    #pragma unroll 8
    for (int col = 0; col <= row; ++col) {
        float a_val = A[idx2D(row, col, lda)];
        sum += a_val * sv[col];
    }
    
    #pragma unroll 8  
    for (int col = row + 1; col < n; ++col) {
        float a_val = A[idx2D(col, row, lda)];
        sum += a_val * sv[col];
    }
    
    w[row] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * w[row];
}

// Unblocked symmetric tridiagonal reduction
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
    const float* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(float);

    for (int j = 0; j < n - 2; ++j) {
        int m = n - j - 1;

        hipLaunchKernelGGL(larfg_kernel, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, j, dA, lda, dE, dTau);

        // w = A[j+1:, j+1:] @ v
        float* A22 = dA + idx2D(j + 1, j + 1, lda);
        float* v_vec = dA + idx2D(j + 1, j, lda);
        if (m > 64) {
            rocblas_ssymv(handle, rocblas_fill_lower, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);
        } else {
            int symv_blocks = CEIL_DIV(m, 64);
            hipLaunchKernelGGL(small_ssymv_kernel, dim3(symv_blocks), dim3(64), m * sizeof(float), 0,
                               m, 1.0f, A22, lda, v_vec, 0.0f, w_vec);
        }

        // w = tau * (w - 0.5 * tau * dot(w, v) * v)
        hipLaunchKernelGGL(fused_dot_scale_axpy, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, v_vec, w_vec, dTau + j);

        // A[j+1:, j+1:] -= v * w^T + w * v^T
        rocblas_ssyr2(handle, rocblas_fill_lower, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
    }

    return hipSuccess;
}

// Compute the update to column j of A
__global__ void accumulate_a_col_updates(int len, int j,
                                         float* __restrict__ dA, int lda,
                                         const float* __restrict__ dW, int ldw) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shmem[];
    float* sA = shmem;
    float* sW = shmem + j;

    for (int k = threadIdx.x; k < j; k += blockDim.x) {
        sA[k] = dA[idx2D(j, k, lda)];
        sW[k] = dW[idx2D(j, k, ldw)];
    }
    __syncthreads();

    if (tid >= len) return;

    int row = j + tid;
    float accum = 0.0f;
    for (int k = 0; k < j; ++k) {
        float a_tail = dA[idx2D(row, k, lda)];
        float w_tail = dW[idx2D(row, k, ldw)];
        accum += sA[k] * w_tail + sW[k] * a_tail;
    }
    dA[idx2D(row, j, lda)] -= accum;
}

__global__ void compute_w_col_kernel(int m, int j,
                                     const float* __restrict__ A22, int lda,
                                     const float* __restrict__ A,
                                     const float* __restrict__ W, int ldw,
                                     const float* __restrict__ v,
                                     float* __restrict__ w,
                                     float* __restrict__ tmp_vec) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];

    float w_sum = 0.0f;
    float a_sum = 0.0f;
    float symv_sum = 0.0f;

    int m_vec4 = m / 4;
        
    #pragma unroll 4
    for (int row4 = tid; row4 < m_vec4; row4 += blockDim.x) {
        float4 v_val4 = reinterpret_cast<const float4*>(&v[row4 * 4])[0];
        float4 syma_val4 = reinterpret_cast<const float4*>(&A22[idx2D(row4 * 4, col, lda)])[0];
        
        symv_sum += (syma_val4.x * v_val4.x + syma_val4.y * v_val4.y + 
                    syma_val4.z * v_val4.z + syma_val4.w * v_val4.w);

        if (col < j) {
            float4 w_val4 = reinterpret_cast<const float4*>(&W[idx2D(row4 * 4, col, ldw)])[0];
            float4 a_val4 = reinterpret_cast<const float4*>(&A[idx2D(row4 * 4, col, lda)])[0];

            w_sum += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
            a_sum += (a_val4.x * v_val4.x + a_val4.y * v_val4.y + 
                     a_val4.z * v_val4.z + a_val4.w * v_val4.w);
        }
    }

    // Handle remaining elements
    for (int row = 4 * m_vec4 + tid; row < m; row += blockDim.x) {
        float v_val = v[row];
        
        float syma_val = A22[idx2D(row, col, lda)];
        symv_sum += syma_val * v_val;

        if (col < j) {
            float w_val = W[idx2D(row, col, ldw)];
            float a_val = A[idx2D(row, col, lda)];
            w_sum += w_val * v_val;
            a_sum += a_val * v_val;
        }
    }

    if (col < j) {
        // Reduction within block for w_sum
        block_reduce_sum(w_sum, sdata, tid, blockDim.x);
        if (tid == 0) {
            tmp_vec[col] = sdata[0];
        }

        // Reduction within block for a_sum  
        block_reduce_sum(a_sum, sdata, tid, blockDim.x);
        if (tid == 0) {
            tmp_vec[col + j] = sdata[0];
        }
    }

    block_reduce_sum(symv_sum, sdata, tid, blockDim.x);
    if (tid == 0) {
        w[col] = sdata[0];
    }
}

__global__ void update_w_col_kernel(int m, int j,
                                    const float* __restrict__ A, int lda,
                                    const float* __restrict__ W, int ldw,
                                    const float* __restrict__ tmp_vec,
                                    float* __restrict__ w) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    float* stmp1 = sdata;      // tmp1 = W^T*v
    float* stmp2 = sdata + j;  // tmp2 = A^T*v
    
    for (int i = threadIdx.x; i < j; i += blockDim.x) {
        stmp1[i] = tmp_vec[i];
        stmp2[i] = tmp_vec[j + i];
    }
    __syncthreads();

    if (row >= m) return;
    
    float sum = 0.0f;
    
    for (int col = 0; col < j; ++col) {
        float a_val = A[idx2D(row, col, lda)];
        float w_val = W[idx2D(row, col, ldw)];
        sum += a_val * stmp1[col] + w_val * stmp2[col];
    }
    
    w[row] -= sum;
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
    const float* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(float);

    for (int j = 0; j < panel_size; ++j) {
        int m = n - j - 1;
        
        // A[j:, j] -= A[j:, :j] * W[j, :j]^T + W[j:, :j] * A[j, :j]^T
        if (j > 0) {
            int len = n - j;
            int blocks_updates = CEIL_DIV(len, threads);
            size_t shmem_updates = 2 * j * sizeof(float);
            hipLaunchKernelGGL(accumulate_a_col_updates, dim3(blocks_updates), dim3(threads), shmem_updates, 0,
                               len, j, dA, lda, dW, ldw);
        }

        // Single-block optimized path
        hipLaunchKernelGGL(larfg_kernel, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, j, dA, lda, dE, dTau);

        float* v_vec = dA + idx2D(j + 1, j, lda);
        float* w_vec = dW + idx2D(j + 1, j, ldw);

        // w = A[j+1:, j+1:] @ v
        //   - A[j+1:, :j] @ (W[j+1:, :j].T @ v)
        //   - W[j+1:, :j] @ (A[j+1:, :j].T @ v)
        if (j > 0) {            
            hipLaunchKernelGGL(compute_w_col_kernel, dim3(m), dim3(threads), shmem_bytes, 0,
                               m, j,
                               dA + idx2D(j + 1, j + 1, lda), lda,
                               dA + j + 1,
                               dW + j + 1, ldw,
                               v_vec, w_vec, d_tmp_vec);
            
            hipLaunchKernelGGL(update_w_col_kernel, dim3(CEIL_DIV(m, threads)), dim3(threads), 2*j*sizeof(float), 0,
                               m, j, dA + j + 1, lda, dW + j + 1, ldw, d_tmp_vec, w_vec);
        } else {
            rocblas_sgemv(handle, rocblas_operation_none, m, m, d_scalars + 0,
                          dA + idx2D(j + 1, j + 1, lda), lda, 
                          v_vec, 1, d_scalars + 2, w_vec, 1);
        }

        // w = tau * (w - 0.5 * tau * dot(w, v) * v)
        hipLaunchKernelGGL(fused_dot_scale_axpy, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, v_vec, w_vec, dTau + j);
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
    hipMalloc(&d_tmp_vec, 2 * panel_size * sizeof(float));

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
                  d_scalars);

        j += panel_size;

        // A[j:, j:] -= W[j:, :j] * A[j, :j]^T + A[j:, :j] * W[j, :j]^T
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n - j, n - j, panel_size,
                      d_scalars + 1, // -1.0f
                      dA_panel + panel_size, lda,
                      dW + panel_size, ldw,
                      d_scalars + 0, // 1.0f
                      dA + idx2D(j, j, lda), lda);
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n - j, n - j, panel_size,
                      d_scalars + 1, // -1.0f
                      dW + panel_size, ldw,
                      dA_panel + panel_size, lda,
                      d_scalars + 0, // 1.0f
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
                  d_scalars);
    }

    int tridiag_blocks = CEIL_DIV(n, threads);
    hipLaunchKernelGGL(setup_tridiagonal, dim3(tridiag_blocks), dim3(threads), 0, 0,
                       n, dA, lda, dD, dE);
    hipMemset(dTau + n - 2, 0, sizeof(float));
    hipDeviceSynchronize();
    hipFree(dW);
    hipFree(d_tmp_vec);
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