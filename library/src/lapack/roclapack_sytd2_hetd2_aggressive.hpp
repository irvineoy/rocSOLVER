#pragma once

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Most aggressive optimization: Maximize use of the small kernel approach
// Key insight: The small kernel processes everything in ONE kernel launch
// We need to use it for as many columns as possible

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_aggressive_impl(rocblas_handle handle,
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

    // Set pointer mode
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // Get device properties
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);

    rocblas_stride stridet = 1;

    // Configure kernels - use larger blocks for better performance
    rocblas_int blocks = (n - 1) / 64 + 1;
    dim3 grid_n(blocks, batch_count);
    dim3 threads(64, 1, 1);

    if(uplo == rocblas_fill_upper)
    {
        // Aggressive strategy: Use small kernel for as much as possible
        rocblas_int processed_up_to = n;

        // Find the maximum size we can process with small kernel
        for(rocblas_int test_size = min(400, n); test_size >= 64; test_size -= 32)
        {
            const size_t lmemsize = ((1024 / props.warpSize) + 2 * test_size + 1 + test_size * test_size) * sizeof(T);
            if(lmemsize <= props.sharedMemPerBlock)
            {
                // Found maximum size - process from this point
                processed_up_to = test_size;

                // Launch small kernel for these columns
                ROCSOLVER_LAUNCH_KERNEL((sytd2_upper_kernel_small<1024, T>),
                                       dim3(1, 1, batch_count),
                                       dim3(1024), lmemsize, stream,
                                       test_size, A, shiftA + idx2D(n - test_size, n - test_size, lda),
                                       lda, strideA,
                                       D + n - test_size, strideD,
                                       E + n - test_size, strideE,
                                       tau + n - test_size, strideP);
                break;
            }
        }

        // Process remaining columns with optimized sequence
        for(rocblas_int j = n - 1; j >= processed_up_to && j > 0; --j)
        {
            // Batch operations where possible to reduce kernel launches

            // 1. LARFG
            rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                        strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                        1, batch_count, work, norms);

            // 2-4. Combined SYMV + DOT/SCALE/AXPY + SYR2
            // Use larger thread blocks for better efficiency
            rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, stridet, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                     1, strideP, batch_count, work, workArr);

            // Use 512 threads instead of 64 for better occupancy
            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<512, T>),
                                   dim3(1, 1, batch_count),
                                   dim3(512, 1, 1), 0, stream,
                                   j, A, shiftA + idx2D(0, j, lda),
                                   strideA, tau, 0, strideP, tmptau, stridet);

            rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                     strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                     batch_count, workArr);

            // Save tau - combine with other operations if possible
            if((j - 1) % 32 == 0 || j == 1)  // Batch tau saves
            {
                ROCSOLVER_LAUNCH_KERNEL(set_tau<T>,
                                       dim3((batch_count - 1) / 64 + 1, 1),
                                       dim3(64), 0, stream,
                                       batch_count, tmptau, tau + j - 1, strideP);
            }
        }

        // Process any remaining tau saves
        for(rocblas_int j = processed_up_to - 1; j > 0; j--)
        {
            if((j - 1) % 32 != 0 && j != 1)
            {
                // These were skipped above
                ROCSOLVER_LAUNCH_KERNEL(set_tau<T>,
                                       dim3((batch_count - 1) / 64 + 1, 1),
                                       dim3(64), 0, stream,
                                       batch_count, tmptau + (n - 1 - j), tau + j - 1, strideP);
            }
        }
    }
    else
    {
        // Fall back to baseline for lower triangular
        return rocsolver_sytd2_hetd2_template<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                                        E, strideE, tau, strideP, batch_count, scalars,
                                                        work, norms, tmptau, workArr);
    }

    // Extract diagonal - use larger blocks
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid_n, threads, 0, stream,
                           uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE