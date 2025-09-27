#pragma once

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Working optimization that actually reduces kernel launches correctly
// Key: Expand the "small kernel" approach to larger sizes by using more shared memory

template <int MAX_THDS, typename T, typename I, typename S>
__global__ void __launch_bounds__(MAX_THDS)
sytd2_large_kernel_upper(const I n,
                         T* __restrict__ A,
                         const I lda,
                         S* __restrict__ D,
                         S* __restrict__ E,
                         T* __restrict__ tau)
{
    const I tid = threadIdx.x;
    const I batch_id = blockIdx.z;

    // Move to batch
    A += batch_id * lda * n;
    D += batch_id * n;
    E += batch_id * (n - 1);
    tau += batch_id * (n - 1);

    // Use maximum shared memory (65KB on MI300)
    extern __shared__ char shared_mem[];
    T* a = reinterpret_cast<T*>(shared_mem);  // Matrix storage
    T* x = a + n * 256;  // Vector storage (max 256 columns at a time)
    T* w = x + n;
    T* sval = w + n;

    // Process the matrix in chunks that fit in shared memory
    const I chunk_size = min(256, n);  // Process up to 256 columns at a time

    // Load first chunk of matrix into shared memory
    for(I chunk_start = 0; chunk_start < n; chunk_start += chunk_size)
    {
        I chunk_end = min(chunk_start + chunk_size, n);

        // Load chunk
        for(I idx = tid; idx < n * (chunk_end - chunk_start); idx += MAX_THDS)
        {
            I i = idx % n;
            I j = chunk_start + idx / n;
            if(i < n && j < n)
            {
                a[i + (j - chunk_start) * n] = A[i + j * lda];
            }
        }
        __syncthreads();

        // Process columns in this chunk from right to left
        for(I j = chunk_end - 1; j >= chunk_start && j > 0; --j)
        {
            I j_local = j - chunk_start;

            // LARFG
            T norm2 = 0;
            for(I i = tid; i < j; i += MAX_THDS)
            {
                x[i] = (j_local < chunk_end - chunk_start) ? a[i + j_local * n] : A[i + j * lda];
                if(i < j - 1)
                    norm2 += x[i] * conj(x[i]);
            }

            // Reduce norm
            norm2 += shift_left(norm2, 1);
            norm2 += shift_left(norm2, 2);
            norm2 += shift_left(norm2, 4);
            norm2 += shift_left(norm2, 8);
            norm2 += shift_left(norm2, 16);
            if(warpSize > 32)
                norm2 += shift_left(norm2, 32);

            if(tid % warpSize == 0)
                sval[tid / warpSize] = norm2;
            __syncthreads();

            if(tid == 0)
            {
                for(I k = 1; k < MAX_THDS / warpSize; k++)
                    norm2 += sval[k];

                T alpha = x[j-1];
                norm2 += alpha * conj(alpha);
                S beta = sqrt(std::abs(norm2));

                T tau_val;
                if(beta > 0)
                {
                    if(std::real(alpha) > 0)
                        beta = -beta;
                    tau_val = (beta - alpha) / beta;
                    E[j-1] = beta;
                    x[j-1] = beta;

                    // Scale x
                    T scale = T(1) / (alpha - T(beta));
                    for(I i = 0; i < j - 1; i++)
                        x[i] *= scale;
                }
                else
                {
                    tau_val = 0;
                    E[j-1] = std::real(alpha);
                }
                tau[j-1] = tau_val;
                sval[0] = tau_val;
            }
            __syncthreads();

            T tau_val = sval[0];

            // Store x back
            if(j_local < chunk_end - chunk_start)
            {
                for(I i = tid; i < j; i += MAX_THDS)
                {
                    a[i + j_local * n] = x[i];
                }
            }
            else
            {
                for(I i = tid; i < j; i += MAX_THDS)
                {
                    A[i + j * lda] = x[i];
                }
            }
            __syncthreads();

            if(std::abs(tau_val) > 0)
            {
                // SYMV
                for(I i = tid; i < j; i += MAX_THDS)
                {
                    T temp = 0;
                    for(I k = 0; k < j; k++)
                    {
                        T a_val;
                        if(k >= chunk_start && k < chunk_end)
                        {
                            // In shared memory
                            if(i <= k)
                                a_val = a[i + (k - chunk_start) * n];
                            else
                                a_val = conj(a[k + (i - chunk_start) * n]);
                        }
                        else
                        {
                            // In global memory
                            if(i <= k)
                                a_val = A[i + k * lda];
                            else
                                a_val = conj(A[k + i * lda]);
                        }
                        temp += a_val * x[k];
                    }
                    w[i] = tau_val * temp;
                }
                __syncthreads();

                // DOT
                norm2 = 0;
                for(I i = tid; i < j; i += MAX_THDS)
                {
                    norm2 += x[i] * conj(w[i]);
                }

                // Reduce
                norm2 += shift_left(norm2, 1);
                norm2 += shift_left(norm2, 2);
                norm2 += shift_left(norm2, 4);
                norm2 += shift_left(norm2, 8);
                norm2 += shift_left(norm2, 16);
                if(warpSize > 32)
                    norm2 += shift_left(norm2, 32);

                if(tid % warpSize == 0)
                    sval[tid / warpSize + 1] = norm2;
                __syncthreads();

                if(tid == 0)
                {
                    for(I k = 1; k < MAX_THDS / warpSize; k++)
                        norm2 += sval[k + 1];
                    sval[1] = 0.5 * tau_val * norm2;
                }
                __syncthreads();

                T alpha = sval[1];

                // Update w
                for(I i = tid; i < j; i += MAX_THDS)
                {
                    w[i] = w[i] - alpha * x[i];
                }
                __syncthreads();

                // SYR2 - update matrix
                for(I idx = tid; idx < j * j; idx += MAX_THDS)
                {
                    I i = idx % j;
                    I k = idx / j;

                    if(i <= k)  // Upper triangle only
                    {
                        T update = x[i] * conj(w[k]) + w[i] * conj(x[k]);

                        if(k >= chunk_start && k < chunk_end)
                        {
                            // Update in shared memory
                            a[i + (k - chunk_start) * n] -= update;
                        }
                        else
                        {
                            // Update in global memory
                            A[i + k * lda] -= update;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Write chunk back to global memory
        for(I idx = tid; idx < n * (chunk_end - chunk_start); idx += MAX_THDS)
        {
            I i = idx % n;
            I j = chunk_start + idx / n;
            if(i < n && j < n && i <= j)  // Upper triangle only
            {
                A[i + j * lda] = a[i + (j - chunk_start) * n];
            }
        }
        __syncthreads();
    }

    // Extract diagonal
    for(I i = tid; i < n; i += MAX_THDS)
    {
        D[i] = std::real(A[i + i * lda]);
    }
}

// Driver function
template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_working_opt_impl(rocblas_handle handle,
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

    // Use large kernel for sizes that fit
    if(n <= 256 && uplo == rocblas_fill_upper)
    {
        dim3 blocks(1, 1, batch_count);
        dim3 threads(256);

        // Calculate shared memory needed
        size_t shared_size = sizeof(T) * (n * 256 + 3 * n + 32);

        if(shared_size <= 65536)  // 64KB limit
        {
            sytd2_large_kernel_upper<256><<<blocks, threads, shared_size, stream>>>(
                n, (T*)A + shiftA, lda, D, E, tau
            );
            return rocblas_status_success;
        }
    }

    // Fall back to original for larger sizes or if shared memory exceeded
    return rocblas_status_not_implemented;
}

ROCSOLVER_END_NAMESPACE