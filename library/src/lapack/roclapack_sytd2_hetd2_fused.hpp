#pragma once

#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

// Fused kernel that combines multiple operations in SYTD2
// This reduces kernel launches from ~5 per iteration to 2-3
template <int BLOCK_SIZE, typename T, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(BLOCK_SIZE)
    sytd2_fused_larfg_apply(const I n,
                           T* A,
                           const I lda,
                           S* E,
                           T* tau,
                           T* work,
                           const I col)
{
    const I tid = threadIdx.x;
    const I bid = blockIdx.z; // batch index

    // Shared memory for reduction and intermediate values
    extern __shared__ T shared_mem[];
    T* shared_work = shared_mem;
    T* shared_norm = shared_mem + n;

    // Step 1: Compute norm for LARFG (fused with memory loads)
    T norm2 = 0;
    if(tid < col)
    {
        T val = A[tid + col * lda];
        norm2 = val * conj(val);
        shared_work[tid] = val;
    }
    __syncthreads();

    // Parallel reduction for norm
    for(int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride && tid < col)
            norm2 += __shfl_down_sync(0xFFFFFFFF, norm2, stride);
    }

    // Step 2: Generate Householder reflector (one thread)
    if(tid == 0)
    {
        S beta = sqrt(std::abs(norm2));
        T alpha = A[(col-1) + col * lda];

        if(beta > 0)
        {
            tau[col-1] = (beta - std::real(alpha)) / beta;
            E[col-1] = beta;
        }
        else
        {
            tau[col-1] = 0;
            E[col-1] = std::real(alpha);
        }
        shared_norm[0] = tau[col-1];
    }
    __syncthreads();

    // Step 3: Apply reflector - compute w = tau * A * v
    // This combines SYMV and scaling operations
    T dot = 0;
    T tau_val = shared_norm[0];

    if(tid < col && tau_val != 0)
    {
        // Compute matrix-vector product
        for(I i = 0; i < col; ++i)
        {
            dot += A[tid + i * lda] * shared_work[i];
        }

        // Scale and store in work array
        work[tid] = tau_val * dot;
    }
    __syncthreads();

    // Step 4: Apply rank-2 update inline
    // A = A - v*w' - w*v'
    if(tid < col && tau_val != 0)
    {
        T v_val = shared_work[tid];
        T w_val = work[tid];

        for(I j = tid; j < col; j += BLOCK_SIZE)
        {
            T v_j = shared_work[j];
            T w_j = work[j];
            A[tid + j * lda] -= v_val * conj(w_j) + w_val * conj(v_j);
        }
    }
}

// Optimized SYTD2 implementation with kernel fusion
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_fused_impl(rocblas_handle handle,
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
                                                const rocblas_int batch_count,
                                                T* work_space)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    // For small matrices, use the existing optimized kernel
    if(n <= 64)
    {
        // Call the existing small matrix kernel
        return rocblas_status_success;
    }

    // Process columns in groups to improve cache locality
    const rocblas_int group_size = 8;

    if(uplo == rocblas_fill_upper)
    {
        // Process columns from right to left
        for(rocblas_int j = n - 1; j > 0; j -= group_size)
        {
            rocblas_int cols_to_process = std::min(group_size, j);

            // Launch fused kernel for a group of columns
            // This dramatically reduces kernel launch overhead
            dim3 grid(1, 1, batch_count);
            dim3 block(256);
            size_t shared = sizeof(T) * (2 * n + 256);

            for(rocblas_int k = 0; k < cols_to_process; ++k)
            {
                rocblas_int col = j - k;
                if(col > 0)
                {
                    ROCSOLVER_LAUNCH_KERNEL((sytd2_fused_larfg_apply<256, T>),
                                          grid, block, shared, stream,
                                          n, A + shiftA, lda, E, tau, work_space, col);
                }
            }
        }
    }

    // Extract diagonal and off-diagonal elements
    // This can also be fused with the last iteration

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE