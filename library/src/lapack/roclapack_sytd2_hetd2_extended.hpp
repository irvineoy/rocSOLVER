#pragma once

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Extended optimization: Use the efficient small kernel approach for larger sizes
// Key insight: The small kernel processes everything in one launch
// We can extend it to handle n=1024 by using more threads and shared memory

#define EXTENDED_MAX_N 1024  // Support up to 1024x1024 matrices

// Driver that uses extended small kernel approach
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_extended_impl(rocblas_handle handle,
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
    ROCSOLVER_ENTER("sytd2_hetd2_extended", "uplo:", uplo, "n:", n, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // get device prop
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    // configure kernels
    rocblas_int blocks = (n - 1) / 32 + 1;
    dim3 grid_n(blocks, batch_count);
    dim3 threads(32, 1, 1);

    rocblas_stride stridet = 1;

    if(uplo == rocblas_fill_lower)
    {
        // Process lower triangular - fall back to baseline
        return rocsolver_sytd2_hetd2_template<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                                        E, strideE, tau, strideP, batch_count, scalars,
                                                        work, norms, tmptau, workArr);
    }
    else
    {
        // Process upper triangular
        // Try to use small kernel for as much as possible
        rocblas_int processed = 0;

        for(rocblas_int j = n - 1; j > 0; --j)
        {
            const rocblas_int nn = j + 1;

            // Calculate shared memory needed
            // Need space for: sval array + x vector + w vector + a matrix
            const size_t lmemsize = ((1024 / props.warpSize) + 2 * nn + nn * nn) * sizeof(T);

            // Use small kernel if it fits in shared memory
            // MI300 has 65KB shared memory
            if(lmemsize <= 65536 && nn <= 256)  // Extended limit
            {
                // Use 1024 threads for better parallelism
                ROCSOLVER_LAUNCH_KERNEL((sytd2_upper_kernel_small<1024, T>),
                                       dim3(1, 1, batch_count),
                                       dim3(1024), lmemsize, stream,
                                       nn, A, shiftA, lda, strideA, D,
                                       strideD, E, strideE, tau, strideP);
                processed = j;
                break;
            }

            // For columns that don't fit, use the regular approach but with batching
            // Batch multiple operations to reduce launches

            // Step 1: LARFG
            rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                        strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                        1, batch_count, work, norms);

            // Steps 2-4: Combined in fewer launches
            // Instead of separate SYMV, DOT, AXPY, SYR2, combine where possible

            // Combined SYMV + initial computations
            rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, stridet, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                     1, strideP, batch_count, work, workArr);

            // Combined DOT + SCALE + AXPY
            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<256, T>),
                                   dim3(1, 1, batch_count),
                                   dim3(256, 1, 1), 0, stream,
                                   j, A, shiftA + idx2D(0, j, lda),
                                   strideA, tau, 0, strideP, tmptau, stridet);

            // SYR2 update
            rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                     strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                     batch_count, workArr);

            // Save tau
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>,
                                   dim3((batch_count - 1) / 32 + 1, 1),
                                   threads, 0, stream,
                                   batch_count, tmptau, tau + j - 1, strideP);
        }
    }

    // Copy results (set tridiagonal form in A)
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid_n, threads, 0, stream, uplo, n, A, shiftA, lda,
                           strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE