#pragma once

#include "roclapack_sytd2_hetd2.hpp"
#include <hip/hip_runtime.h>

ROCSOLVER_BEGIN_NAMESPACE

// Ultimate optimization: Minimize kernel launches by processing multiple columns per launch
// Key innovation: Process BLOCK_COLS columns in a single kernel launch

constexpr int BLOCK_COLS = 32;  // Process 32 columns per kernel launch
constexpr int BLOCK_THREADS = 256;

// Kernel that processes multiple columns in one launch
template <typename T, typename S>
__global__ void __launch_bounds__(BLOCK_THREADS)
sytd2_multi_column_kernel(const rocblas_int n,
                          const rocblas_int start_col,
                          const rocblas_int num_cols,
                          T* __restrict__ A,
                          const rocblas_int lda,
                          S* __restrict__ E,
                          T* __restrict__ tau)
{
    const int tid = threadIdx.x;

    // Large shared memory to hold working set
    extern __shared__ char shared_mem[];
    T* shared_cols = reinterpret_cast<T*>(shared_mem);
    T* shared_work = shared_cols + n * BLOCK_COLS;
    T* shared_tau = shared_work + n * BLOCK_COLS;

    // Process columns from right to left
    for(int col_idx = 0; col_idx < num_cols && (start_col - col_idx) > 0; col_idx++)
    {
        int j = start_col - col_idx;

        // Load column j into shared memory
        for(int i = tid; i < j; i += blockDim.x)
        {
            shared_cols[i + col_idx * n] = A[i + j * lda];
        }
        __syncthreads();

        // Step 1: Compute norm for LARFG
        T norm2 = 0;
        for(int i = tid; i < j; i += blockDim.x)
        {
            T val = shared_cols[i + col_idx * n];
            norm2 += val * conj(val);
        }

        // Block reduction
        __shared__ T reduction[BLOCK_THREADS];
        reduction[tid] = norm2;
        __syncthreads();

        for(int s = blockDim.x/2; s > 0; s >>= 1)
        {
            if(tid < s)
                reduction[tid] += reduction[tid + s];
            __syncthreads();
        }

        // Generate tau
        if(tid == 0)
        {
            norm2 = reduction[0];
            T alpha = A[(j-1) + j * lda];
            S beta = sqrt(std::abs(norm2));

            if(beta > 0)
            {
                shared_tau[col_idx] = (beta - std::real(alpha)) / beta;
                E[j-1] = beta;

                // Scale vector
                T scale = T(1) / (alpha - T(beta));
                for(int i = 0; i < j-1; i++)
                    shared_cols[i + col_idx * n] *= scale;
                shared_cols[(j-1) + col_idx * n] = T(1);
            }
            else
            {
                shared_tau[col_idx] = 0;
                E[j-1] = std::real(alpha);
            }
            tau[j-1] = shared_tau[col_idx];
        }
        __syncthreads();

        T tau_val = shared_tau[col_idx];

        if(std::abs(tau_val) > 0)
        {
            // Step 2: Matrix-vector product (SYMV)
            for(int i = tid; i < j; i += blockDim.x)
            {
                T sum = 0;
                for(int k = 0; k < j; k++)
                {
                    sum += A[i + k * lda] * shared_cols[k + col_idx * n];
                }
                shared_work[i + col_idx * n] = tau_val * sum;
            }
            __syncthreads();

            // Step 3: Compute dot product
            T dot = 0;
            for(int i = tid; i < j; i += blockDim.x)
            {
                dot += conj(shared_cols[i + col_idx * n]) * shared_work[i + col_idx * n];
            }

            reduction[tid] = dot;
            __syncthreads();

            for(int s = blockDim.x/2; s > 0; s >>= 1)
            {
                if(tid < s)
                    reduction[tid] += reduction[tid + s];
                __syncthreads();
            }

            __shared__ T alpha_val;
            if(tid == 0)
            {
                alpha_val = 0.5 * tau_val * reduction[0];
            }
            __syncthreads();

            // Step 4: Update w
            for(int i = tid; i < j; i += blockDim.x)
            {
                shared_work[i + col_idx * n] -= alpha_val * shared_cols[i + col_idx * n];
            }
            __syncthreads();

            // Step 5: Rank-2 update
            for(int ii = tid; ii < j*j; ii += blockDim.x)
            {
                int i = ii % j;
                int k = ii / j;
                A[i + k * lda] -= shared_cols[i + col_idx * n] * conj(shared_work[k + col_idx * n])
                                + shared_work[i + col_idx * n] * conj(shared_cols[k + col_idx * n]);
            }
        }

        // Store updated column back
        for(int i = tid; i < j; i += blockDim.x)
        {
            A[i + j * lda] = shared_cols[i + col_idx * n];
        }
        __syncthreads();
    }
}

// Kernel for extracting diagonal elements
template <typename T, typename S>
__global__ void extract_diagonal_kernel(const rocblas_int n,
                                       T* __restrict__ A,
                                       const rocblas_int lda,
                                       S* __restrict__ D)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        D[i] = std::real(A[i + i * lda]);
    }
}

// Main driver with reduced kernel launches
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_ultimate_impl(rocblas_handle handle,
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

    // Process in groups of BLOCK_COLS columns
    const int num_launches = (n - 1 + BLOCK_COLS - 1) / BLOCK_COLS;

    for(int launch = 0; launch < num_launches; launch++)
    {
        int start_col = n - 1 - launch * BLOCK_COLS;
        int num_cols = min(BLOCK_COLS, start_col + 1);

        if(start_col <= 0)
            break;

        dim3 blocks(1, 1, batch_count);
        dim3 threads(BLOCK_THREADS);

        size_t shared_size = sizeof(T) * (n * BLOCK_COLS * 2 + BLOCK_COLS + BLOCK_THREADS);

        if(uplo == rocblas_fill_upper)
        {
            sytd2_multi_column_kernel<<<blocks, threads, shared_size, stream>>>(
                n, start_col, num_cols, (T*)A + shiftA, lda, E, tau
            );
        }
    }

    // Extract diagonal
    dim3 diag_blocks((n + BLOCK_THREADS - 1) / BLOCK_THREADS, 1, batch_count);
    dim3 diag_threads(BLOCK_THREADS);
    extract_diagonal_kernel<<<diag_blocks, diag_threads, 0, stream>>>(
        n, (T*)A + shiftA, lda, D
    );

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE