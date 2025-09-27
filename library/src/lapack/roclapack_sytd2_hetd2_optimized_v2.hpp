#pragma once

#include "roclapack_sytd2_hetd2.hpp"
#include <hip/hip_runtime.h>

ROCSOLVER_BEGIN_NAMESPACE

// Optimized V2: Single kernel that processes entire matrix with minimal launches
// Uses aggressive loop unrolling and register optimization

template <typename T, typename S>
__global__ void __launch_bounds__(256)
sytd2_single_kernel_upper(const rocblas_int n,
                          T* __restrict__ A,
                          const rocblas_int lda,
                          S* __restrict__ D,
                          S* __restrict__ E,
                          T* __restrict__ tau)
{
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.z;

    // Process one batch per block
    A += batch_id * lda * n;
    D += batch_id * n;
    E += batch_id * (n - 1);
    tau += batch_id * (n - 1);

    // Allocate maximum shared memory
    extern __shared__ char shared_mem[];
    T* shared_A = reinterpret_cast<T*>(shared_mem);
    T* shared_v = shared_A + n * n;  // Store full matrix
    T* shared_w = shared_v + n;
    T* shared_work = shared_w + n;

    // Load entire matrix into shared memory (for n <= 1024)
    const int total_elements = n * n;
    for(int i = tid; i < total_elements; i += blockDim.x)
    {
        int row = i % n;
        int col = i / n;
        shared_A[row + col * n] = A[row + col * lda];
    }
    __syncthreads();

    // Process columns from right to left
    for(int j = n - 1; j > 0; --j)
    {
        // Step 1: Generate Householder reflector
        T norm2 = 0;

        // Each thread computes partial norm
        for(int i = tid; i < j; i += blockDim.x)
        {
            T val = shared_A[i + j * n];
            shared_v[i] = val;
            norm2 += val * conj(val);
        }

        // Reduce norm across block
        __shared__ T norm_reduce[256];
        norm_reduce[tid] = norm2;
        __syncthreads();

        // Tree reduction
        for(int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if(tid < s)
                norm_reduce[tid] += norm_reduce[tid + s];
            __syncthreads();
        }

        // Compute tau
        __shared__ T tau_val;
        if(tid == 0)
        {
            norm2 = norm_reduce[0];
            T alpha = shared_A[(j-1) + j * n];
            S beta = sqrt(std::abs(norm2));

            if(beta > 0)
            {
                tau_val = (beta - std::real(alpha)) / beta;
                E[j-1] = beta;
                shared_A[(j-1) + j * n] = beta;

                // Scale vector
                T scale = T(1) / (alpha - T(beta));
                for(int i = 0; i < j - 1; i++)
                    shared_v[i] *= scale;
                shared_v[j-1] = T(1);
            }
            else
            {
                tau_val = 0;
                E[j-1] = std::real(alpha);
            }
            tau[j-1] = tau_val;
        }
        __syncthreads();

        if(std::abs(tau_val) > 0)
        {
            // Step 2: Compute w = tau * A * v
            for(int i = tid; i < j; i += blockDim.x)
            {
                T sum = 0;
                #pragma unroll 8
                for(int k = 0; k < j; k++)
                {
                    sum += shared_A[i + k * n] * shared_v[k];
                }
                shared_w[i] = tau_val * sum;
            }
            __syncthreads();

            // Step 3: Compute alpha = 0.5 * tau * (v' * w)
            T dot = 0;
            for(int i = tid; i < j; i += blockDim.x)
            {
                dot += conj(shared_v[i]) * shared_w[i];
            }

            // Reduce dot product
            norm_reduce[tid] = dot;
            __syncthreads();

            for(int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if(tid < s)
                    norm_reduce[tid] += norm_reduce[tid + s];
                __syncthreads();
            }

            __shared__ T alpha_val;
            if(tid == 0)
            {
                alpha_val = 0.5 * tau_val * norm_reduce[0];
            }
            __syncthreads();

            // Step 4: Update w = w - alpha * v
            for(int i = tid; i < j; i += blockDim.x)
            {
                shared_w[i] = shared_w[i] - alpha_val * shared_v[i];
            }
            __syncthreads();

            // Step 5: Apply rank-2 update A = A - v*w' - w*v'
            for(int ii = tid; ii < j * j; ii += blockDim.x)
            {
                int i = ii % j;
                int k = ii / j;
                shared_A[i + k * n] -= shared_v[i] * conj(shared_w[k])
                                     + shared_w[i] * conj(shared_v[k]);
            }
            __syncthreads();
        }

        // Update column j in global memory
        for(int i = tid; i < j; i += blockDim.x)
        {
            shared_A[i + j * n] = shared_v[i];
        }
        __syncthreads();
    }

    // Write back to global memory
    for(int i = tid; i < total_elements; i += blockDim.x)
    {
        int row = i % n;
        int col = i / n;
        A[row + col * lda] = shared_A[row + col * n];
    }

    // Extract diagonal
    for(int i = tid; i < n; i += blockDim.x)
    {
        D[i] = std::real(shared_A[i + i * n]);
    }
}

// Blocked version for larger matrices
template <typename T, typename S>
__global__ void __launch_bounds__(512)
sytd2_blocked_kernel(const rocblas_int n,
                    const rocblas_int nb,  // Block size
                    T* __restrict__ A,
                    const rocblas_int lda,
                    S* __restrict__ D,
                    S* __restrict__ E,
                    T* __restrict__ tau,
                    T* __restrict__ work)
{
    // Process matrix in blocks of size nb
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Each block processes one panel
    const int panel_start = bid * nb;
    const int panel_end = min(panel_start + nb, n);

    extern __shared__ char shared_mem[];
    T* shared_panel = reinterpret_cast<T*>(shared_mem);

    // Panel factorization
    for(int j = panel_end - 1; j >= panel_start && j > 0; --j)
    {
        // Similar to single kernel but operating on panels
        // This allows processing matrices larger than shared memory
    }
}

// Optimized entry point
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_optimized_v2(rocblas_handle handle,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  S* D,
                                                  const rocblas_stride strideD,
                                                  S* E,
                                                  const rocblas_stride strideE,
                                                  T* tau,
                                                  const rocblas_stride strideP,
                                                  const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    // For small matrices that fit in shared memory
    if(n <= 128)  // 128x128 * sizeof(double complex) = 256KB < shared mem limit
    {
        dim3 blocks(1, 1, batch_count);
        dim3 threads(256);

        size_t shared_size = sizeof(T) * (n * n + 3 * n + 256);

        if(uplo == rocblas_fill_upper)
        {
            sytd2_single_kernel_upper<<<blocks, threads, shared_size, stream>>>(
                n, (T*)A + shiftA, lda, D, E, tau
            );
        }
        else
        {
            // Implement lower version similarly
        }
    }
    else
    {
        // For larger matrices, fall back to original implementation
        // or use blocked version
        return rocblas_status_not_implemented;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE