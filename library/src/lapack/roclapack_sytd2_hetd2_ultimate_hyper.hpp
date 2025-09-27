#pragma once

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Ultimate hyperaggressive: Process even more columns in the small kernel
// Key optimization: Increase shared memory usage through dynamic allocation

template <int BLOCK_SIZE, typename T>
__global__ void __launch_bounds__(BLOCK_SIZE)
sytd2_ultimate_hyper_kernel(const rocblas_int n,
                            T* A,
                            const rocblas_int lda,
                            T* D,
                            T* E,
                            T* tau)
{
    const int tid = threadIdx.x;
    const int wid = tid / 64;  // Warp ID
    const int lane = tid % 64; // Lane in warp

    extern __shared__ char ultimate_shmem[];
    T* shared_A = reinterpret_cast<T*>(ultimate_shmem);
    T* shared_tau = shared_A + n * n;
    T* shared_work = shared_tau + n;

    // Load entire matrix into shared memory if it fits
    for(int idx = tid; idx < n * n; idx += BLOCK_SIZE)
    {
        int i = idx % n;
        int j = idx / n;
        shared_A[idx] = A[i + j * lda];
    }
    __syncthreads();

    // Process all columns in shared memory
    for(int j = n - 1; j > 0; j--)
    {
        // Compute Householder reflector for column j
        T norm = 0;
        for(int i = tid; i < j; i += BLOCK_SIZE)
        {
            T val = shared_A[i + j * n];
            norm += val * conj(val);
        }

        // Warp reduction for norm
        for(int offset = 32; offset > 0; offset >>= 1)
            norm += __shfl_down(norm, offset);

        if(lane == 0)
            shared_work[wid] = norm;
        __syncthreads();

        // Final reduction
        if(tid == 0)
        {
            T total_norm = 0;
            for(int w = 0; w < (BLOCK_SIZE + 63) / 64; w++)
                total_norm += shared_work[w];

            T beta = sqrt(std::abs(total_norm + shared_A[j-1 + j*n] * conj(shared_A[j-1 + j*n])));
            shared_tau[j-1] = (beta - shared_A[j-1 + j*n]) / beta;
            E[j-1] = -beta;

            // Normalize vector
            T scale = 1.0 / (shared_A[j-1 + j*n] - beta);
            for(int i = 0; i < j-1; i++)
                shared_A[i + j*n] *= scale;
            shared_A[j-1 + j*n] = 1.0;
        }
        __syncthreads();

        // Apply Householder transformation
        for(int i = tid; i < j; i += BLOCK_SIZE)
        {
            T sum = 0;
            for(int k = 0; k < j; k++)
                sum += shared_A[k + i*n] * shared_A[k + j*n];
            shared_work[i] = sum * shared_tau[j-1];
        }
        __syncthreads();

        // Update matrix
        for(int idx = tid; idx < j * j; idx += BLOCK_SIZE)
        {
            int i = idx % j;
            int k = idx / j;
            shared_A[i + k*n] -= shared_work[i] * conj(shared_A[k + j*n]) + shared_A[i + j*n] * conj(shared_work[k]);
        }
        __syncthreads();
    }

    // Write back diagonal and off-diagonal
    if(tid < n)
    {
        D[tid] = std::real(shared_A[tid + tid*n]);
        tau[tid] = shared_tau[tid];
    }

    // Write back updated matrix if needed
    for(int idx = tid; idx < n * n; idx += BLOCK_SIZE)
    {
        int i = idx % n;
        int j = idx / n;
        A[i + j * lda] = shared_A[idx];
    }
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_ultimate_hyper_impl(rocblas_handle handle,
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
                                                         T* scalars,
                                                         T* work,
                                                         T* norms,
                                                         T* tmptau,
                                                         T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    int device;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);

    if(uplo == rocblas_fill_upper)
    {
        // Try to process as much as possible in one kernel
        rocblas_int max_n_in_shared = 0;

        // Calculate maximum n that fits in shared memory
        // Need n*n + 2*n elements of type T
        for(rocblas_int test_n = min(512, n); test_n >= 32; test_n--)
        {
            size_t shared_size = (test_n * test_n + 2 * test_n) * sizeof(T);
            if(shared_size <= props.sharedMemPerBlock)
            {
                max_n_in_shared = test_n;
                break;
            }
        }

        if(max_n_in_shared > 0 && n <= max_n_in_shared)
        {
            // Process entire matrix in one kernel!
            size_t shared_size = (n * n + 2 * n) * sizeof(T);
            sytd2_ultimate_hyper_kernel<1024, T><<<dim3(1, 1, batch_count), 1024, shared_size, stream>>>(
                n, (T*)A + shiftA, lda, D, E, tau
            );
        }
        else if(max_n_in_shared >= 64)
        {
            // Process bottom-right corner with ultimate kernel
            rocblas_int processed = max_n_in_shared;
            size_t shared_size = (processed * processed + 2 * processed) * sizeof(T);

            sytd2_ultimate_hyper_kernel<1024, T><<<dim3(1, 1, batch_count), 1024, shared_size, stream>>>(
                processed, (T*)A + shiftA + idx2D(n - processed, n - processed, lda), lda,
                D + n - processed, E + n - processed, tau + n - processed
            );

            // Process remaining with hyperaggressive approach
            for(rocblas_int j = n - processed - 1; j > 0; j--)
            {
                // LARFG
                rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                            strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                            1, batch_count, work, norms);

                // SYMV - use larger work buffers
                rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, 1, A, shiftA, lda, strideA, A,
                                         shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                         1, strideP, batch_count, work, workArr);

                // Combined operations kernel
                ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<1024, T>),
                                       dim3(1, 1, batch_count),
                                       dim3(1024, 1, 1), 0, stream,
                                       j, A, shiftA + idx2D(0, j, lda),
                                       strideA, tau, 0, strideP, tmptau, 1);

                // SYR2
                rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                         strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                         batch_count, workArr);

                // Save tau
                ROCSOLVER_LAUNCH_KERNEL(set_tau<T>,
                                       dim3((batch_count - 1) / 256 + 1, 1),
                                       dim3(256), 0, stream,
                                       batch_count, tmptau, tau + j - 1, strideP);
            }
        }
        else
        {
            // Fall back to hyperaggressive implementation
            return rocsolver_sytd2_hetd2_hyperaggressive_impl<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                                 D, strideD, E, strideE, tau, strideP,
                                                                 batch_count, scalars, work, norms, tmptau, workArr);
        }
    }
    else
    {
        return rocblas_status_not_implemented;
    }

    // Extract diagonal
    dim3 grid((n - 1) / 64 + 1, batch_count);
    dim3 threads(64);
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid, threads, 0, stream,
                           uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE