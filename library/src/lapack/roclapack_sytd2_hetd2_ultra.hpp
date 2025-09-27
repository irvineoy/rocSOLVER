#pragma once

#include "roclapack_sytd2_hetd2.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Ultra optimization: Process multiple columns per kernel launch while maintaining correctness
// Key: Process BLOCK_SIZE columns in each kernel, reducing launches by BLOCK_SIZE factor

template <int BLOCK_SIZE, typename T, typename S>
__global__ void __launch_bounds__(512)
sytd2_ultra_kernel(const rocblas_int n,
                  const rocblas_int start_col,
                  const rocblas_int end_col,
                  T* __restrict__ A,
                  const rocblas_int lda,
                  S* __restrict__ E,
                  T* __restrict__ tau)
{
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.z;

    // Move to batch
    A += batch_id * lda * n;
    E += batch_id * (n - 1);
    tau += batch_id * (n - 1);

    // Large shared memory allocation
    extern __shared__ char shared_mem[];
    T* shared_work = reinterpret_cast<T*>(shared_mem);

    // Process columns sequentially but in same kernel
    for(int j = start_col; j > end_col && j > 0; --j)
    {
        // All 5 operations for column j in this single kernel

        // 1. LARFG inline
        T norm2 = 0;
        __shared__ T alpha_val;
        __shared__ T tau_val;

        for(int i = tid; i < j; i += blockDim.x)
        {
            T val = A[i + j * lda];
            shared_work[i] = val;
            if(i < j - 1)
                norm2 += val * conj(val);
        }

        // Reduce norm2
        __shared__ T reduction[512];
        reduction[tid] = norm2;
        __syncthreads();

        for(int s = blockDim.x/2; s > 0; s >>= 1)
        {
            if(tid < s)
                reduction[tid] += reduction[tid + s];
            __syncthreads();
        }

        if(tid == 0)
        {
            norm2 = reduction[0];
            alpha_val = A[(j-1) + j * lda];
            norm2 += alpha_val * conj(alpha_val);
            S beta = sqrt(std::abs(norm2));

            if(beta > 0)
            {
                if(std::real(alpha_val) > 0)
                    beta = -beta;
                tau_val = (beta - alpha_val) / beta;
                E[j-1] = beta;

                // Scale vector
                T scale = T(1) / (alpha_val - T(beta));
                for(int i = 0; i < j - 1; i++)
                    shared_work[i] *= scale;
                shared_work[j-1] = T(1);
            }
            else
            {
                tau_val = 0;
                E[j-1] = std::real(alpha_val);
            }
            tau[j-1] = tau_val;
        }
        __syncthreads();

        // Store scaled vector back
        for(int i = tid; i < j; i += blockDim.x)
        {
            A[i + j * lda] = shared_work[i];
        }

        if(std::abs(tau_val) > 0)
        {
            // 2. SYMV inline
            for(int i = tid; i < j; i += blockDim.x)
            {
                T sum = 0;
                #pragma unroll 4
                for(int k = 0; k < j; k++)
                {
                    T a_val = (i <= k) ? A[i + k * lda] : conj(A[k + i * lda]);
                    sum += a_val * shared_work[k];
                }
                shared_work[n + i] = tau_val * sum;  // Store w in second half
            }
            __syncthreads();

            // 3. DOT + AXPY combined
            T dot = 0;
            for(int i = tid; i < j; i += blockDim.x)
            {
                dot += conj(shared_work[i]) * shared_work[n + i];
            }

            reduction[tid] = dot;
            __syncthreads();

            for(int s = blockDim.x/2; s > 0; s >>= 1)
            {
                if(tid < s)
                    reduction[tid] += reduction[tid + s];
                __syncthreads();
            }

            __shared__ T alpha;
            if(tid == 0)
            {
                alpha = 0.5 * tau_val * reduction[0];
            }
            __syncthreads();

            // Update w
            for(int i = tid; i < j; i += blockDim.x)
            {
                shared_work[n + i] -= alpha * shared_work[i];
            }
            __syncthreads();

            // 4. SYR2 inline
            for(int idx = tid; idx < j * (j + 1) / 2; idx += blockDim.x)
            {
                // Convert linear index to (i,k) in upper triangle
                int i = 0;
                int linear = idx;
                while(linear >= j - i)
                {
                    linear -= (j - i);
                    i++;
                }
                int k = i + linear;

                T update = shared_work[i] * conj(shared_work[n + k])
                        + shared_work[n + i] * conj(shared_work[k]);
                A[i + k * lda] -= update;
            }
            __syncthreads();
        }
    }
}

// Ultra driver with minimal kernel launches
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_ultra_impl(rocblas_handle handle,
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

    if(uplo == rocblas_fill_upper)
    {
        // Process in blocks to minimize kernel launches
        const int cols_per_kernel = 32;  // Process 32 columns per kernel

        for(int col = n - 1; col > 0; col -= cols_per_kernel)
        {
            int start_col = col;
            int end_col = max(0, col - cols_per_kernel);

            // Check if remaining columns fit in small kernel
            if(start_col < 256)
            {
                // Use existing efficient small kernel
                const size_t lmemsize = ((512 / 64) + 2 * (start_col + 1) + 1
                                       + (start_col + 1) * (start_col + 1)) * sizeof(T);
                if(lmemsize <= 65536)
                {
                    ROCSOLVER_LAUNCH_KERNEL((sytd2_upper_kernel_small<512, T>),
                                           dim3(1, 1, batch_count),
                                           dim3(512), lmemsize, stream,
                                           start_col + 1, A, shiftA, lda, strideA,
                                           D, strideD, E, strideE, tau, strideP);
                    break;
                }
            }

            // Launch ultra kernel for this block
            dim3 blocks(1, 1, batch_count);
            dim3 threads(512);
            size_t shared_size = sizeof(T) * n * 2 + sizeof(T) * 512;

            sytd2_ultra_kernel<32><<<blocks, threads, shared_size, stream>>>(
                n, start_col, end_col, (T*)A + shiftA, lda, E, tau
            );
        }

        // Extract diagonal
        dim3 diag_blocks((n + 255) / 256, 1, batch_count);
        dim3 diag_threads(256);
        ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>,
                               diag_blocks, diag_threads, 0, stream,
                               uplo, n, A, shiftA, lda, strideA,
                               D, strideD, E, strideE);
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE