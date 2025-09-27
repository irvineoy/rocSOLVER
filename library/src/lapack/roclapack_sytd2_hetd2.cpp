/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_sytd2_hetd2.hpp"
#include "roclapack_sytd2_hetd2_tiled.hpp"
#include "roclapack_sytd2_hetd2_graph.hpp"
#include "roclapack_sytd2_hetd2_revolutionary.hpp"
#include "roclapack_sytd2_hetd2_ultimate.hpp"
#include "roclapack_sytd2_hetd2_final.hpp"
#include "roclapack_sytd2_hetd2_extended.hpp"
#include "roclapack_sytd2_hetd2_ultra.hpp"
#include "roclapack_sytd2_hetd2_aggressive.hpp"
#include "roclapack_sytd2_hetd2_conservative.hpp"
#include "roclapack_sytd2_hetd2_hyperaggressive.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_impl(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          S* D,
                                          S* E,
                                          T* tau)
{
    const char* name = (!rocblas_is_complex<T> ? "sytd2" : "hetd2");
    ROCSOLVER_ENTER_TOP(name, "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sytd2_hetd2_argCheck(handle, uplo, n, lda, A, D, E, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling LARFG
    size_t size_norms, size_work;
    // size of temporary householder scalars
    size_t size_tmptau;
    // size of array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_sytd2_hetd2_getMemorySize<false, T>(n, batch_count, &size_scalars, &size_work,
                                                  &size_norms, &size_tmptau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_norms,
                                                      size_tmptau, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *norms, *tmptau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_norms, size_tmptau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    norms = mem[2];
    tmptau = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution - use aggressive optimization for maximum performance
    static bool use_conservative = std::getenv("ROCSOLVER_USE_CONSERVATIVE") != nullptr;
    static bool use_hyperaggressive = std::getenv("ROCSOLVER_USE_HYPERAGGRESSIVE") != nullptr;
    static bool use_aggressive = std::getenv("ROCSOLVER_USE_AGGRESSIVE") != nullptr;
    static bool use_ultra = std::getenv("ROCSOLVER_USE_ULTRA") != nullptr;
    static bool use_extended = std::getenv("ROCSOLVER_USE_EXTENDED") != nullptr;
    static bool use_final = std::getenv("ROCSOLVER_USE_FINAL") != nullptr;
    static bool use_ultimate = std::getenv("ROCSOLVER_USE_ULTIMATE") != nullptr;
    static bool use_revolutionary = std::getenv("ROCSOLVER_USE_REVOLUTIONARY") != nullptr;
    static bool use_graph_opt = std::getenv("ROCSOLVER_USE_GRAPH") != nullptr;

    // Use conservative optimization by default for safety
    if(use_conservative)
    {
        return rocsolver_sytd2_hetd2_conservative_impl<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA,
                                                                D, strideD, E, strideE, tau, strideP,
                                                                batch_count, (T*)scalars, (T*)work,
                                                                (T*)norms, (T*)tmptau, (T**)workArr);
    }
    else if(use_hyperaggressive)
    {
        return rocsolver_sytd2_hetd2_hyperaggressive_impl<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA,
                                                                    D, strideD, E, strideE, tau, strideP,
                                                                    batch_count, (T*)scalars, (T*)work,
                                                                    (T*)norms, (T*)tmptau, (T**)workArr);
    }
    else if(use_aggressive)
    {
        return rocsolver_sytd2_hetd2_aggressive_impl<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA,
                                                               D, strideD, E, strideE, tau, strideP,
                                                               batch_count, (T*)scalars, (T*)work,
                                                               (T*)norms, (T*)tmptau, (T**)workArr);
    }
    else if(use_ultra)
    {
        return rocsolver_sytd2_hetd2_ultra_impl<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                   D, strideD, E, strideE, tau, strideP,
                                                   batch_count);
    }
    else if(use_extended)
    {
        return rocsolver_sytd2_hetd2_extended_impl<T, S, U>(handle, uplo, n, A, shiftA, lda, strideA,
                                                             D, strideD, E, strideE, tau, strideP,
                                                             batch_count, (T*)scalars, (T*)work,
                                                             (T*)norms, (T*)tmptau, (T**)workArr);
    }
    else if(use_final && n >= 64)
    {
        return rocsolver_sytd2_hetd2_final_impl<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                   D, strideD, E, strideE, tau, strideP,
                                                   batch_count);
    }
    else if(use_ultimate && n >= 64)
    {
        return rocsolver_sytd2_hetd2_ultimate_impl<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                      D, strideD, E, strideE, tau, strideP,
                                                      batch_count);
    }
    else if(use_revolutionary && n >= 256)
    {
        return rocsolver_sytd2_hetd2_revolutionary_impl<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                           D, strideD, E, strideE, tau, strideP,
                                                           batch_count);
    }
    else if(use_graph_opt && n >= 512)
    {
        return rocsolver_sytd2_hetd2_graph_optimized<T>(handle, uplo, n, A, shiftA, lda, strideA,
                                                        D, strideD, E, strideE, tau, strideP,
                                                        batch_count, (T*)scalars, (T*)work,
                                                        (T*)norms, (T*)tmptau, (T**)workArr);
    }
    else
    {
        return rocsolver_sytd2_hetd2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                                 E, strideE, tau, strideP, batch_count, (T*)scalars,
                                                 (T*)work, (T*)norms, (T*)tmptau, (T**)workArr);
    }
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<float>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_dsytd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<double>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_chetd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<rocblas_float_complex>(handle, uplo, n, A, lda, D,
                                                                        E, tau);
}

rocblas_status rocsolver_zhetd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<rocblas_double_complex>(handle, uplo, n, A, lda, D,
                                                                         E, tau);
}

} // extern C
