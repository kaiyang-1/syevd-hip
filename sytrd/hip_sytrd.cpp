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
#include <type_traits>

// Template trait for vector types
template<typename T> struct vec4_type {};
template<> struct vec4_type<float> { using type = float4; };
template<> struct vec4_type<double> { using type = double4; };

// Template trait for literal constants
template<typename T> struct literal_constants {
    static constexpr T zero = static_cast<T>(0);
    static constexpr T one = static_cast<T>(1);
    static constexpr T half = static_cast<T>(0.5);
    static constexpr T epsilon = std::numeric_limits<T>::epsilon();
};

// Template wrappers for math functions
template<typename T> __device__ __forceinline__ T sqrt_func(T x);
template<> __device__ __forceinline__ float sqrt_func<float>(float x) { return sqrtf(x); }
template<> __device__ __forceinline__ double sqrt_func<double>(double x) { return sqrt(x); }

template<typename T> __device__ __forceinline__ T abs_func(T x);
template<> __device__ __forceinline__ float abs_func<float>(float x) { return fabsf(x); }
template<> __device__ __forceinline__ double abs_func<double>(double x) { return fabs(x); }

#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

// 2D index calculation for column-major storage
__device__ __host__ inline int idx2D(const int i, const int j, const int lda) { return j * lda + i; }

// Warp-level reduction
template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction
template<typename scalar_t>
__device__ __forceinline__ void block_reduce_sum(scalar_t val, scalar_t *smem, int tid, int blockDimX) {
    val = warp_reduce_sum(val);

    if (blockDimX > warpSize) {
        int lane = tid & (warpSize - 1);
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : literal_constants<scalar_t>::zero;
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
template<typename scalar_t>
__global__ void larfg_kernel(int m, int i, scalar_t* __restrict__ A, int lda,
                             scalar_t* __restrict__ dE, scalar_t* __restrict__ dTau) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    scalar_t* col_base = A + idx2D(i + 1, i, lda);

    // Accumulate sum of squares of tail (excluding first element)
    scalar_t local_sum = literal_constants<scalar_t>::zero;
    scalar_t alpha = literal_constants<scalar_t>::zero;

    int m_vec4 = m / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
        
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
        scalar_t val = col_base[idx];
        if (idx == 0 && m_vec4 == 0) {
            alpha = val;
        } else if (idx > 0) {
            local_sum += val * val;
        }
    }

    // Reduction within block for local_sum
    block_reduce_sum(local_sum, sdata, tid, blockDim.x);
    __syncthreads();
    scalar_t sigma2 = sdata[0];

    // Broadcast alpha to all threads (alpha only known by threads that saw idx==0)
    // Use shared memory position 1 for alpha broadcast if needed.
    if (tid == 0) sdata[1] = alpha;
    __syncthreads();
    alpha = sdata[1];

    scalar_t beta, tau, scale;
    if (sigma2 < literal_constants<scalar_t>::epsilon) {
        tau = literal_constants<scalar_t>::zero;
        beta = alpha;
        scale = literal_constants<scalar_t>::zero;
    } else {
        scalar_t r = sqrt_func(alpha * alpha + sigma2);
        beta = alpha >= literal_constants<scalar_t>::zero ? -r : r;
        tau = (beta - alpha) / beta;
        scale = literal_constants<scalar_t>::one / (alpha - beta);
    }

    // Store tau & beta
    if (tid == 0) {
        dE[i] = beta;
        dTau[i] = tau;
    }

    #pragma unroll 4
    for (int idx4 = tid; idx4 < m_vec4; idx4 += blockDim.x) {
        if (idx4 == 0) {
            // Handle first element specially (set to 1.0) and scale the rest
            vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
            val4.x = literal_constants<scalar_t>::one;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<vec4_t*>(&col_base[idx4 * 4])[0] = val4;
        } else {
            vec4_t val4 = reinterpret_cast<const vec4_t*>(&col_base[idx4 * 4])[0];
            val4.x *= scale;
            val4.y *= scale;
            val4.z *= scale;
            val4.w *= scale;
            reinterpret_cast<vec4_t*>(&col_base[idx4 * 4])[0] = val4;
        }
    }
    
    for (int idx = 4 * m_vec4 + tid; idx < m; idx += blockDim.x) {
        if (idx == 0 && m_vec4 == 0) {
            col_base[0] = literal_constants<scalar_t>::one;
        } else if (idx > 0) {
            col_base[idx] *= scale;
        }
    }
}

// Kernel to compute dot product, scale, and axpy in a fused manner
template<typename scalar_t>
__global__ void fused_dot_scale_axpy(int n,
                                     const scalar_t* __restrict__ v,
                                     scalar_t* __restrict__ w,
                                     const scalar_t* __restrict__ tau_ptr) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    scalar_t tau = *tau_ptr;

    // Compute dot product within single block
    scalar_t local_dot = literal_constants<scalar_t>::zero;
    
    int n_vec4 = n / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
    
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&w[idx4 * 4])[0];
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[idx4 * 4])[0];
        
        local_dot += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
    }
    
    // Handle remaining elements for dot product
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        scalar_t w_val = w[idx];
        scalar_t v_val = v[idx];
        local_dot += w_val * v_val;
    }

    // Reduction within block for dot product
    block_reduce_sum(local_dot, sdata, tid, blockDim.x);
    __syncthreads();
    scalar_t dot = sdata[0];

    // Compute correction factor
    scalar_t alpha_corr = -literal_constants<scalar_t>::half * tau * dot;

    // Apply scale and axpy operation
    #pragma unroll 4
    for (int idx4 = tid; idx4 < n_vec4; idx4 += blockDim.x) {
        vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&w[idx4 * 4])[0];
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[idx4 * 4])[0];
        
        w_val4.x = tau * (w_val4.x + alpha_corr * v_val4.x);
        w_val4.y = tau * (w_val4.y + alpha_corr * v_val4.y);
        w_val4.z = tau * (w_val4.z + alpha_corr * v_val4.z);
        w_val4.w = tau * (w_val4.w + alpha_corr * v_val4.w);
        
        reinterpret_cast<vec4_t*>(&w[idx4 * 4])[0] = w_val4;
    }
    
    for (int idx = 4 * n_vec4 + tid; idx < n; idx += blockDim.x) {
        scalar_t w_val = w[idx];
        scalar_t v_val = v[idx];
        w[idx] = tau * (w_val + alpha_corr * v_val);
    }
}

// Kernel to set up tridiagonal matrix elements D and E from A
template<typename scalar_t>
__global__ void setup_tridiagonal(int n, scalar_t* A, int lda, scalar_t* D, scalar_t* E) {
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

// Fused kernel for tridiagolization of matrices of size <= warpSize
template<typename scalar_t>
__global__ void small_sytd2_kernel(int m, int j, scalar_t* __restrict__ A, int lda,
                                   scalar_t* __restrict__ dE, scalar_t* __restrict__ dTau) {
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    
    // Shared memory for Householder vector v and intermediate vector w
    scalar_t* sv = sdata;
    scalar_t* sw = sdata + m;
    
    scalar_t* col_base = A + idx2D(j + 1, j, lda);
    scalar_t* A22 = A + idx2D(j + 1, j + 1, lda);
    
    // ==================== STEP 1: Compute Householder reflector ====================
    scalar_t sigma2 = literal_constants<scalar_t>::zero;
    scalar_t alpha = literal_constants<scalar_t>::zero;
    
    // Load column vector and compute sum of squares
    for (int idx = tid; idx < m; idx += blockDim.x) {
        scalar_t val = col_base[idx];
        if (idx == 0) alpha = val; else sigma2 += val * val;
        if (idx < m) sv[idx] = val;
    }
    
    // Reduction for sum of squares
    sigma2 = warp_reduce_sum(sigma2);
    sigma2 = __shfl(sigma2, 0);
    alpha = __shfl(alpha, 0);
    
    // Compute Householder parameters
    scalar_t beta, tau, scale;
    if (sigma2 < literal_constants<scalar_t>::epsilon) {
        tau = literal_constants<scalar_t>::zero; 
        beta = alpha; 
        scale = literal_constants<scalar_t>::zero;
    } else {
        scalar_t r = sqrt_func(alpha * alpha + sigma2);
        beta = alpha >= literal_constants<scalar_t>::zero ? -r : r;
        tau = (beta - alpha) / beta;
        scale = literal_constants<scalar_t>::one / (alpha - beta);
    }
    
    // Store tau & beta
    if (tid == 0) {
        dE[j] = beta;
        dTau[j] = tau;
    }
    
    // Update Householder vector in shared memory
    for (int idx = tid; idx < m; idx += blockDim.x) {
        if (idx == 0) sv[0] = literal_constants<scalar_t>::one;
        else sv[idx] *= scale;
    }
    
    // ==================== STEP 2: Matrix-vector multiplication w = A22 * v ====================
    // Each thread computes one element of w
    if (tid < m) {
        scalar_t sum = literal_constants<scalar_t>::zero;
        for (int col = 0; col < m; col++) {
            scalar_t a_val = A22[idx2D(tid, col, lda)];
            sum += a_val * sv[col];
        }
        sw[tid] = sum;
    }
    
    // ==================== STEP 3: Dot product and scaling w = tau * (w - 0.5 * tau * dot(w,v) * v) ====================
    // Compute dot product w^T * v
    scalar_t dot = literal_constants<scalar_t>::zero;
    for (int idx = tid; idx < m; idx += blockDim.x) {
        dot += sw[idx] * sv[idx];
    }
    dot = warp_reduce_sum(dot);
    dot = __shfl(dot, 0);
    
    // Apply scaling: w = tau * (w - 0.5 * tau * dot * v)
    scalar_t alpha_corr = -literal_constants<scalar_t>::half * tau * dot;
    for (int idx = tid; idx < m; idx += blockDim.x) {
        sw[idx] = tau * (sw[idx] + alpha_corr * sv[idx]);
    }
    
    // ==================== STEP 4: Matrix update A22 -= v * w^T + w * v^T ====================
    // Each thread processes one row of the matrix
    if (tid < m) {
        scalar_t v_row = sv[tid];
        scalar_t w_row = sw[tid];
        
        for (int col = 0; col < m; col++) {
            scalar_t a_val = A22[idx2D(tid, col, lda)];
            scalar_t v_col = sv[col];
            scalar_t w_col = sw[col];
            
            // A[row][col] -= v[row] * w[col] + w[row] * v[col]
            A22[idx2D(tid, col, lda)] = a_val - (v_row * w_col + w_row * v_col);
        }
    }
    
    // ==================== STEP 5: Update original column vector ====================
    // Write back the Householder vector to the lower trianglar part of A
    for (int idx = tid; idx < m; idx += blockDim.x) {
        col_base[idx] = sv[idx];
    }
}

// Unblocked symmetric tridiagonal reduction
template<typename scalar_t>
hipError_t hip_sytd2(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau,
    int device_warp_size,
    scalar_t* w_vec,
    const scalar_t* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(scalar_t);

    for (int j = 0; j < n - 2; ++j) {
        int m = n - j - 1;

        if (m <= device_warp_size) {
            // Use fused kernel for small matrices
            size_t sytd2_shmem = 2 * m * sizeof(scalar_t);  // space for v, w vetors
            hipLaunchKernelGGL(small_sytd2_kernel<scalar_t>, dim3(1), dim3(device_warp_size), sytd2_shmem, 0,
                               m, j, dA, lda, dE, dTau);
        } else {
            // Use separate kernels for larger matrices
            hipLaunchKernelGGL(larfg_kernel<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                               m, j, dA, lda, dE, dTau);

            scalar_t* A22 = dA + idx2D(j + 1, j + 1, lda);
            scalar_t* v_vec = dA + idx2D(j + 1, j, lda);
            
            // w = A[j+1:, j+1:] @ v
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sgemv(handle, rocblas_operation_none, m, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);
            } else {
                rocblas_dgemv(handle, rocblas_operation_none, m, m, d_scalars + 0, A22, lda, v_vec, 1, d_scalars + 2, w_vec, 1);                
            }

            // w = tau * (w - 0.5 * tau * dot(w, v) * v)
            hipLaunchKernelGGL(fused_dot_scale_axpy<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                               m, v_vec, w_vec, dTau + j);
            
            // A[j+1:, j+1:] -= v * w^T + w * v^T
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sger(handle, m, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
                rocblas_sger(handle, m, m, d_scalars + 1, w_vec, 1, v_vec, 1, A22, lda);
            } else {
                rocblas_dger(handle, m, m, d_scalars + 1, v_vec, 1, w_vec, 1, A22, lda);
                rocblas_dger(handle, m, m, d_scalars + 1, w_vec, 1, v_vec, 1, A22, lda);
            }
        }
    }

    return hipSuccess;
}

// Compute the update to column j of A
template<typename scalar_t>
__global__ void accumulate_a_col_updates(int len, int j,
                                         scalar_t* __restrict__ dA, int lda,
                                         const scalar_t* __restrict__ dW, int ldw) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ char shmem_raw[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(shmem_raw);
    scalar_t* sA = shmem;
    scalar_t* sW = shmem + j;

    for (int k = threadIdx.x; k < j; k += blockDim.x) {
        sA[k] = dA[idx2D(j, k, lda)];
        sW[k] = dW[idx2D(j, k, ldw)];
    }
    __syncthreads();

    if (tid >= len) return;

    int row = j + tid;
    scalar_t accum = literal_constants<scalar_t>::zero;
    for (int k = 0; k < j; ++k) {
        scalar_t a_tail = dA[idx2D(row, k, lda)];
        scalar_t w_tail = dW[idx2D(row, k, ldw)];
        accum += sA[k] * w_tail + sW[k] * a_tail;
    }
    dA[idx2D(row, j, lda)] -= accum;
}

template<typename scalar_t>
__global__ void compute_w_col_kernel(int m, int j,
                                     const scalar_t* __restrict__ A22, int lda,
                                     const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ W, int ldw,
                                     const scalar_t* __restrict__ v,
                                     scalar_t* __restrict__ w,
                                     scalar_t* __restrict__ tmp_vec) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    scalar_t w_sum = literal_constants<scalar_t>::zero;
    scalar_t a_sum = literal_constants<scalar_t>::zero;
    scalar_t symv_sum = literal_constants<scalar_t>::zero;

    int m_vec4 = m / 4;
    using vec4_t = typename vec4_type<scalar_t>::type;
        
    #pragma unroll 4
    for (int row4 = tid; row4 < m_vec4; row4 += blockDim.x) {
        vec4_t v_val4 = reinterpret_cast<const vec4_t*>(&v[row4 * 4])[0];
        vec4_t syma_val4 = reinterpret_cast<const vec4_t*>(&A22[idx2D(row4 * 4, col, lda)])[0];
        
        symv_sum += (syma_val4.x * v_val4.x + syma_val4.y * v_val4.y + 
                    syma_val4.z * v_val4.z + syma_val4.w * v_val4.w);

        if (col < j) {
            vec4_t w_val4 = reinterpret_cast<const vec4_t*>(&W[idx2D(row4 * 4, col, ldw)])[0];
            vec4_t a_val4 = reinterpret_cast<const vec4_t*>(&A[idx2D(row4 * 4, col, lda)])[0];

            w_sum += (w_val4.x * v_val4.x + w_val4.y * v_val4.y + 
                     w_val4.z * v_val4.z + w_val4.w * v_val4.w);
            a_sum += (a_val4.x * v_val4.x + a_val4.y * v_val4.y + 
                     a_val4.z * v_val4.z + a_val4.w * v_val4.w);
        }
    }

    // Handle remaining elements
    for (int row = 4 * m_vec4 + tid; row < m; row += blockDim.x) {
        scalar_t v_val = v[row];
        
        scalar_t syma_val = A22[idx2D(row, col, lda)];
        symv_sum += syma_val * v_val;

        if (col < j) {
            scalar_t w_val = W[idx2D(row, col, ldw)];
            scalar_t a_val = A[idx2D(row, col, lda)];
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

template<typename scalar_t>
__global__ void update_w_col_kernel(int m, int j,
                                    const scalar_t* __restrict__ A, int lda,
                                    const scalar_t* __restrict__ W, int ldw,
                                    const scalar_t* __restrict__ tmp_vec,
                                    scalar_t* __restrict__ w) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ char sdata_raw[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    scalar_t* stmp1 = sdata;      // tmp1 = W^T*v
    scalar_t* stmp2 = sdata + j;  // tmp2 = A^T*v
    
    for (int i = threadIdx.x; i < j; i += blockDim.x) {
        stmp1[i] = tmp_vec[i];
        stmp2[i] = tmp_vec[j + i];
    }
    __syncthreads();

    if (row >= m) return;
    
    scalar_t sum = literal_constants<scalar_t>::zero;
    
    for (int col = 0; col < j; ++col) {
        scalar_t a_val = A[idx2D(row, col, lda)];
        scalar_t w_val = W[idx2D(row, col, ldw)];
        sum += a_val * stmp1[col] + w_val * stmp2[col];
    }
    
    w[row] -= sum;
}

// Blocked panel reduction for symmetric tridiagonalization
template<typename scalar_t>
hipError_t hip_latrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dW,
    int ldw,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau,
    int panel_size,
    int device_warp_size,
    scalar_t* d_tmp_vec,
    const scalar_t* d_scalars)
{
    const int threads = 256;
    size_t shmem_bytes = CEIL_DIV(threads, device_warp_size) * sizeof(scalar_t);

    for (int j = 0; j < panel_size; ++j) {
        int m = n - j - 1;
        
        // A[j:, j] -= A[j:, :j] * W[j, :j]^T + W[j:, :j] * A[j, :j]^T
        if (j > 0) {
            int len = n - j;
            int blocks_updates = CEIL_DIV(len, threads);
            size_t shmem_updates = 2 * j * sizeof(scalar_t);
            hipLaunchKernelGGL(accumulate_a_col_updates<scalar_t>, dim3(blocks_updates), dim3(threads), shmem_updates, 0,
                               len, j, dA, lda, dW, ldw);
        }

        // Single-block optimized path
        hipLaunchKernelGGL(larfg_kernel<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, j, dA, lda, dE, dTau);

        scalar_t* v_vec = dA + idx2D(j + 1, j, lda);
        scalar_t* w_vec = dW + idx2D(j + 1, j, ldw);

        // w = A[j+1:, j+1:] @ v
        //   - A[j+1:, :j] @ (W[j+1:, :j].T @ v)
        //   - W[j+1:, :j] @ (A[j+1:, :j].T @ v)
        if (j > 0) {            
            hipLaunchKernelGGL(compute_w_col_kernel<scalar_t>, dim3(m), dim3(threads), shmem_bytes, 0,
                               m, j,
                               dA + idx2D(j + 1, j + 1, lda), lda,
                               dA + j + 1,
                               dW + j + 1, ldw,
                               v_vec, w_vec, d_tmp_vec);
            
            hipLaunchKernelGGL(update_w_col_kernel<scalar_t>, dim3(CEIL_DIV(m, threads)), dim3(threads), 2*j*sizeof(scalar_t), 0,
                               m, j, dA + j + 1, lda, dW + j + 1, ldw, d_tmp_vec, w_vec);
        } else {
            if constexpr (std::is_same_v<scalar_t, float>) {
                rocblas_sgemv(handle, rocblas_operation_none, m, m, d_scalars + 0,
                              dA + idx2D(j + 1, j + 1, lda), lda, 
                              v_vec, 1, d_scalars + 2, w_vec, 1);
            } else {
                rocblas_dgemv(handle, rocblas_operation_none, m, m, d_scalars + 0,
                              dA + idx2D(j + 1, j + 1, lda), lda, 
                              v_vec, 1, d_scalars + 2, w_vec, 1);
            }
        }

        // w = tau * (w - 0.5 * tau * dot(w, v) * v)
        hipLaunchKernelGGL(fused_dot_scale_axpy<scalar_t>, dim3(1), dim3(threads), shmem_bytes, 0,
                           m, v_vec, w_vec, dTau + j);
    }
    return hipSuccess;
}

// Main symmetric tridiagonal reduction routine (blocked + unblocked)
template<typename scalar_t>
hipError_t hip_sytrd_template(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    scalar_t* dA,
    int lda,
    scalar_t* dD,
    scalar_t* dE,
    scalar_t* dTau)
{
    if (n < 3) return hipSuccess;

    const int threads = 256;
    const int panel_size = 64;

    // Allocate device memory for scalar constants
    scalar_t h_scalars[3];
    h_scalars[0] = literal_constants<scalar_t>::one;   // 1.0
    h_scalars[1] = -literal_constants<scalar_t>::one;  // -1.0
    h_scalars[2] = literal_constants<scalar_t>::zero;  // 0.0
    
    scalar_t *d_scalars;
    hipMalloc(&d_scalars, 3 * sizeof(scalar_t));
    hipMemcpy(d_scalars, h_scalars, 3 * sizeof(scalar_t), hipMemcpyHostToDevice);
    
    int device_warp_size;
    hipDeviceGetAttribute(&device_warp_size, hipDeviceAttributeWarpSize, 0);
    scalar_t* dW;
    hipMalloc(&dW, n * panel_size * sizeof(scalar_t));
    scalar_t* d_tmp_vec;
    hipMalloc(&d_tmp_vec, 2 * panel_size * sizeof(scalar_t));

    int j = 0;
    while(j < n - panel_size) {
        scalar_t* dA_panel = dA + idx2D(j, j, lda);
        int ldw = n - j;

        hip_latrd<scalar_t>(handle,
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
        // Use appropriate GEMM function based on scalar type
        if constexpr (std::is_same_v<scalar_t, float>) {
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
        } else {
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0
                          dA_panel + panel_size, lda,
                          dW + panel_size, ldw,
                          d_scalars + 0, // 1.0
                          dA + idx2D(j, j, lda), lda);
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          n - j, n - j, panel_size,
                          d_scalars + 1, // -1.0
                          dW + panel_size, ldw,
                          dA_panel + panel_size, lda,
                          d_scalars + 0, // 1.0
                          dA + idx2D(j, j, lda), lda);
        }
    }
    if (j < n - 2) {
        hip_sytd2<scalar_t>(handle,
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
    hipLaunchKernelGGL(setup_tridiagonal<scalar_t>, dim3(tridiag_blocks), dim3(threads), 0, 0,
                       n, dA, lda, dD, dE);
    hipMemset(dTau + n - 2, 0, sizeof(scalar_t));
    hipDeviceSynchronize();
    hipFree(dW);
    hipFree(d_tmp_vec);
    hipFree(d_scalars);

    return hipSuccess;
}

// Wrapper functions for specific precisions
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
    return hip_sytrd_template<float>(handle, uplo, n, dA, lda, dD, dE, dTau);
}

extern "C" hipError_t hip_dsytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    double* dA,
    int lda,
    double* dD,
    double* dE,
    double* dTau)
{
    return hip_sytrd_template<double>(handle, uplo, n, dA, lda, dD, dE, dTau);
}

// Generate a random symmetric matrix (column-major)
template<typename scalar_t>
void generate_symmetric_matrix(std::vector<scalar_t>& A, int n, unsigned int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar_t> dist(literal_constants<scalar_t>::zero, literal_constants<scalar_t>::one);
    std::fill(A.begin(), A.end(), literal_constants<scalar_t>::zero);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            scalar_t value = dist(rng);
            A[i + j * n] = value;
            if (i != j) {
                A[j + i * n] = value;
            }
        }
    }
}

// Struct to hold benchmark results
template<typename scalar_t>
struct BenchmarkResult {
    double hip_avg_time_ms = 0.0;
    double rocsolver_avg_time_ms = 0.0;
    double speedup_rocsolver_over_hip = 0.0;
    bool   validated = false;
    scalar_t  max_abs_diff_D = literal_constants<scalar_t>::zero;
    scalar_t  max_abs_diff_E = literal_constants<scalar_t>::zero;
    scalar_t  max_abs_diff_Tau = literal_constants<scalar_t>::zero;
};

// Benchmark custom and rocSOLVER tridiagonalization, optionally validate results
template<typename scalar_t>
BenchmarkResult<scalar_t> benchmark(int n, bool validate, int iterations, int warmup) {
    const int lda = n;
    BenchmarkResult<scalar_t> result;

    std::vector<scalar_t> hA(n * n);
    std::vector<scalar_t> hD_hip(n), hE_hip(std::max(0, n - 1)), hTau_hip(std::max(0, n - 1));
    std::vector<scalar_t> hD_ref(n), hE_ref(std::max(0, n - 1)), hTau_ref(std::max(0, n - 1));
    scalar_t *dA_hip, *dD_hip, *dE_hip, *dTau_hip;
    hipMalloc(&dA_hip, n * lda * sizeof(scalar_t));
    hipMalloc(&dD_hip, n * sizeof(scalar_t));
    hipMalloc(&dE_hip, (n - 1) * sizeof(scalar_t));
    hipMalloc(&dTau_hip, (n - 1) * sizeof(scalar_t));
    scalar_t *dA_ref, *dD_ref, *dE_ref, *dTau_ref;
    hipMalloc(&dA_ref, n * lda * sizeof(scalar_t));
    hipMalloc(&dD_ref, n * sizeof(scalar_t));
    hipMalloc(&dE_ref, (n - 1) * sizeof(scalar_t));
    hipMalloc(&dTau_ref, (n - 1) * sizeof(scalar_t));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    generate_symmetric_matrix<scalar_t>(hA, n);

    for (int w = 0; w < warmup; ++w) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        
        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_hip, lda, (double*)dD_hip, (double*)dE_hip, (double*)dTau_hip);
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_ref, lda, (double*)dD_ref, (double*)dE_ref, (double*)dTau_ref);
        }
    }
    double hip_total_us = 0.0;
    double ref_total_us = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        auto start_hip = std::chrono::high_resolution_clock::now();
        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_hip, lda, (double*)dD_hip, (double*)dE_hip, (double*)dTau_hip);
        }
        auto end_hip = std::chrono::high_resolution_clock::now();
        hip_total_us += std::chrono::duration<double, std::micro>(end_hip - start_hip).count();
        auto start_ref = std::chrono::high_resolution_clock::now();
        if constexpr (std::is_same_v<scalar_t, float>) {
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_ref, lda, (double*)dD_ref, (double*)dE_ref, (double*)dTau_ref);
        }
        auto end_ref = std::chrono::high_resolution_clock::now();
        ref_total_us += std::chrono::duration<double, std::micro>(end_ref - start_ref).count();
    }

    result.hip_avg_time_ms = (hip_total_us / iterations) / 1000.0;
    result.rocsolver_avg_time_ms = (ref_total_us / iterations) / 1000.0;
    if (result.hip_avg_time_ms > 0.0)
        result.speedup_rocsolver_over_hip = result.rocsolver_avg_time_ms / result.hip_avg_time_ms;

    if (validate) {
        generate_symmetric_matrix<scalar_t>(hA, n, 12345u);
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);

        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_hip, lda, (double*)dD_hip, (double*)dE_hip, (double*)dTau_hip);
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, (double*)dA_ref, lda, (double*)dD_ref, (double*)dE_ref, (double*)dTau_ref);
        }

        hipMemcpy(hD_hip.data(), dD_hip, n * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hE_hip.data(), dE_hip, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_hip.data(), dTau_hip, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hD_ref.data(), dD_ref, n * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hE_ref.data(), dE_ref, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_ref.data(), dTau_ref, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);

        scalar_t max_d = literal_constants<scalar_t>::zero, max_e = literal_constants<scalar_t>::zero, max_tau = literal_constants<scalar_t>::zero;
        for (int i = 0; i < n; ++i) {
            max_d = std::max(max_d, std::abs(hD_hip[i] - hD_ref[i]));
        }
        for (int i = 0; i < n - 1; ++i) {
            max_e = std::max(max_e, std::abs(hE_hip[i] - hE_ref[i]));
            max_tau = std::max(max_tau, std::abs(hTau_hip[i] - hTau_ref[i]));
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
void print_table_header(const std::string& precision, bool validate) {
    printf("Matrix Size | %s (ms) | rocSOLVER (ms) | Speedup",
           (precision == "float") ? "hip_ssytrd" : "hip_dsytrd");

    if (validate) {
        printf(" |   max|ΔD|   |   max|ΔE|   |   max|ΔTau| \n");
        printf("--------------------------------------------------------------------------------------------------\n");
    } else {
        printf("\n");
        printf("--------------------------------------------------------\n");
    }
}

// Print a single row of benchmark results
template<typename scalar_t>
void print_comparison_result(int n, const BenchmarkResult<scalar_t>& result, bool validate) {
    printf("%11d | %15.3f | %14.3f | %7.3f", n,
           result.hip_avg_time_ms,
           result.rocsolver_avg_time_ms,
           result.speedup_rocsolver_over_hip);
    if (validate && result.validated) {
        printf(" | %11.3e | %11.3e | %11.3e",
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
    printf("  -p precision          Precision: 'float' or 'double' (default: float)\n");
    printf("  -v                    Enable validation\n");
    printf("  -i iterations         Number of benchmark iterations (default: 10)\n");
    printf("  -w warmup_runs        Number of warmup runs (default: 3)\n");
    printf("  -h                    Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -n 128 256 -p float -v\n", program_name);
}

// Main entry point: parse arguments, run benchmarks, print results
int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes;
    bool validate = false;
    int iterations = 10;
    int warmup = 3;
    std::string precision = "float";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            precision = argv[++i];
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

    if (precision != "float" && precision != "double") {
        printf("Error: Invalid precision '%s'. Use 'float' or 'double'.\n\n", precision.c_str());
        print_usage(argv[0]);
        return 1;
    }

    printf("Householder Tridiagonalization Benchmark\n");
    printf("=========================================\n");
    printf("Precision: %s\n", precision.c_str());
    printf("Iterations: %d\n", iterations);
    printf("Warmup runs: %d\n", warmup);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    print_table_header(precision, validate);
    
    if (precision == "float") {
        for (int n : matrix_sizes) {
            BenchmarkResult<float> result = benchmark<float>(n, validate, iterations, warmup);
            print_comparison_result(n, result, validate);
        }
    } else {
        for (int n : matrix_sizes) {
            BenchmarkResult<double> result = benchmark<double>(n, validate, iterations, warmup);
            print_comparison_result(n, result, validate);
        }
    }

    printf("\n");

    return 0;
}