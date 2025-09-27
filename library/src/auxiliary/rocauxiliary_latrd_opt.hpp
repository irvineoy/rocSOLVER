#pragma once

#include "../auxiliary/rocauxiliary_lacgv.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

// Optimized version with better memory coalescing and cache utilization
template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) latrd_dot_scale_axpy_opt(const I n,
                                                                       U AA,
                                                                       const rocblas_stride shiftA,
                                                                       const rocblas_stride strideA,
                                                                       T* WW,
                                                                       const rocblas_stride shiftW,
                                                                       const rocblas_stride strideW,
                                                                       T* tauA,
                                                                       const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WW, bid, shiftW, strideW);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // Improved shared memory layout to reduce bank conflicts
    constexpr int BANK_OFFSET = 1;  // Add padding to avoid bank conflicts
    __shared__ T sval[MAX_THDS / WarpSize + BANK_OFFSET];

    // Use larger shared memory tiles for better cache utilization
    constexpr int TILE_SIZE = 256;
    __shared__ T sh_A[TILE_SIZE + BANK_OFFSET];
    __shared__ T sh_W[TILE_SIZE + BANK_OFFSET];

    T norm2 = 0;

    // Process in tiles for better cache locality
    for(I tile_start = 0; tile_start < n; tile_start += TILE_SIZE)
    {
        I tile_end = min(tile_start + TILE_SIZE, n);
        I tile_len = tile_end - tile_start;

        // Coalesced load into shared memory
        for(I i = tid; i < tile_len; i += MAX_THDS)
        {
            sh_A[i] = A[tile_start + i];
            sh_W[i] = W[tile_start + i];
        }
        __syncthreads();

        // Compute partial dot product from shared memory
        for(I i = tid; i < tile_len; i += MAX_THDS)
        {
            norm2 += sh_A[i] * conj(sh_W[i]);
        }
        __syncthreads();
    }

    // Optimized reduction using warp shuffle operations
    norm2 += __shfl_down(norm2, 16);
    norm2 += __shfl_down(norm2, 8);
    norm2 += __shfl_down(norm2, 4);
    norm2 += __shfl_down(norm2, 2);
    norm2 += __shfl_down(norm2, 1);

    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm2;
    __syncthreads();

    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm2 += sval[k];
        sval[0] = -0.5 * tau[0] * norm2;
    }
    __syncthreads();

    T scale = sval[0];

    // Process axpy in tiles with coalesced writes
    for(I tile_start = 0; tile_start < n; tile_start += TILE_SIZE)
    {
        I tile_end = min(tile_start + TILE_SIZE, n);
        I tile_len = tile_end - tile_start;

        // Load tile into shared memory
        for(I i = tid; i < tile_len; i += MAX_THDS)
        {
            sh_A[i] = A[tile_start + i];
            sh_W[i] = W[tile_start + i];
        }
        __syncthreads();

        // Compute and store results with coalesced access
        for(I i = tid; i < tile_len; i += MAX_THDS)
        {
            W[tile_start + i] = sh_W[i] + scale * sh_A[i];
        }
        __syncthreads();
    }
}

// Optimized kernel fusion for latrd operations
template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    latrd_fused_kernel(const I n,
                       const I j,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       T* WW,
                       const rocblas_stride shiftW,
                       const I ldw,
                       const rocblas_stride strideW,
                       T* tauA,
                       const rocblas_stride strideP)
{
    // This kernel fuses multiple operations to reduce memory traffic
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = load_ptr_batch<T>(WW, bid, shiftW, strideW);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // Shared memory for multiple operations
    extern __shared__ T shared_mem[];
    T* sh_work = shared_mem;

    // Perform fused operations with single pass over data
    // This reduces memory bandwidth requirements significantly

    // TODO: Implement fused operations
}

ROCSOLVER_END_NAMESPACE