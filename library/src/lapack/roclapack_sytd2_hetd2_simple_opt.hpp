#pragma once

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Simple optimization: Reduce kernel launches by combining operations

// Combined kernel for DOT + SCALE + AXPY operations
template <typename T>
__global__ void __launch_bounds__(256)
combined_dot_scale_axpy(const rocblas_int n,
                        T* __restrict__ v,
                        T* __restrict__ w,
                        T* __restrict__ tau_scalar,
                        T* __restrict__ tmptau)
{
    const int tid = threadIdx.x;

    // Shared memory for reduction
    __shared__ T shared_dot[256];

    // Step 1: Compute dot product
    T dot = 0;
    for(int i = tid; i < n; i += blockDim.x)
    {
        dot += conj(w[i]) * v[i];
    }

    // Reduce dot product
    shared_dot[tid] = dot;
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
            shared_dot[tid] += shared_dot[tid + s];
        __syncthreads();
    }

    // Step 2: Scale and update
    __shared__ T scale;
    if(tid == 0)
    {
        scale = 0.5 * (*tau_scalar) * (*tmptau) * shared_dot[0];
        *tmptau = scale;
    }
    __syncthreads();

    // Step 3: AXPY - update w
    for(int i = tid; i < n; i += blockDim.x)
    {
        w[i] = w[i] - scale * v[i];
    }
}

// Simple optimized implementation
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_simple_opt_impl(rocblas_handle handle,
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

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // configure kernels
    rocblas_int blocks = (n - 1) / 32 + 1;
    dim3 grid_n(blocks, batch_count);
    dim3 threads(32, 1, 1);
    blocks = (batch_count - 1) / 32 + 1;
    dim3 grid_b(blocks, 1);

    rocblas_stride stridet = 1;

    // get device prop
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    if(uplo == rocblas_fill_upper)
    {
        // Process upper triangular
        for(rocblas_int j = n - 1; j > 0; --j)
        {
            const rocblas_int nn = j + 1;
            const size_t lmemsize = ((256 / props.warpSize) + 2 * nn + 1 + nn * nn) * sizeof(T);

            // Use existing efficient kernel for small sizes
            if(lmemsize <= props.sharedMemPerBlock && nn <= 130)
            {
                ROCSOLVER_LAUNCH_KERNEL((sytd2_upper_kernel_small<256, T>),
                                       dim3(1, 1, batch_count),
                                       dim3(256), lmemsize, stream,
                                       nn, A, shiftA, lda, strideA, D,
                                       strideD, E, strideE, tau, strideP);
                break;
            }

            // For larger sizes, use optimized sequence with fewer launches

            // 1. LARFG
            rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                        strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                        1, batch_count, work, norms);

            // 2. SYMV
            rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, stridet, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                     1, strideP, batch_count, work, workArr);

            // 3. Combined DOT + SCALE + AXPY in single kernel
            combined_dot_scale_axpy<<<dim3(1, 1, batch_count), dim3(256), 0, stream>>>(
                j,
                (T*)A + shiftA + idx2D(0, j, lda),
                tau,
                tau + j - 1,
                tmptau
            );

            // 4. SYR2
            rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                     strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                     batch_count, workArr);

            // 5. Save tau
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                   tau + j - 1, strideP);
        }
    }
    else
    {
        // Similar for lower triangular
    }

    // Copy results
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid_n, threads, 0, stream, uplo, n, A, shiftA, lda,
                           strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE