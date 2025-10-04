#!/usr/bin/env python3
"""
Generator for GEQRF (blocked QR factorization) SFT dataset entries.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def generate_entries():
    entries = []

    # L1-1: Householder Reflector Generation with Warp Reduction (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L1_larfg",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the run_set_taubeta device function that computes tau and the scaling factor for a Householder reflector. This function is called during LARFG operations to generate reflectors that annihilate subdiagonal elements in QR factorization.

Requirements:
- Handle both real and complex types
- Compute norm as sqrt(norms[0] + alpha^2) where norms[0] is the squared norm of x
- Set tau = (n - alpha) / n where n is the computed norm with appropriate sign
- Set scaling factor: norms[0] = 1 / (alpha - n)
- For complex types, use proper conjugation and magnitude calculations

Code blocks for context:
```cpp
// From rocauxiliary_larfg.hpp lines 47-85 (real version)
template <typename T, typename S, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void run_set_taubeta(T* tau, T* norms, T* alpha, S* beta)
{
    const auto ignore_beta = (beta == nullptr);
    if(norms[0] > 0)
    {
        T n = sqrt(norms[0] + alpha[0] * alpha[0]);
        n = alpha[0] >= 0 ? -n : n;

        // scaling factor:
        norms[0] = 1.0 / (alpha[0] - n);

        // tau:
        tau[0] = (n - alpha[0]) / n;

        // beta:
        if(ignore_beta)
        {
            alpha[0] = n;
        }
        else
        {
            beta[0] = n;
            alpha[0] = 1;
        }
    }
    else
    {
        norms[0] = 1;
        tau[0] = 0;

        // beta:
        if(!ignore_beta)
        {
            beta[0] = alpha[0];
            alpha[0] = 1;
        }
    }
}
```

Provide the complete implementation for both real and complex types, explaining the mathematical reasoning behind the sign choice and scaling factor."""
            }]
        }
    })

    # L1-2: Warp-Level Reduction for Norm Computation (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L1_norm_reduction",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the warp-level reduction pattern used in geqr2_kernel_small to compute the squared norm of a vector. This is a critical performance optimization for QR factorization.

Requirements:
- Use shift_left operations for warp reduction (1, 2, 4, 8, 16, 32 positions)
- Handle both 32-bit and 64-bit warp sizes (check warpSize > 32)
- Store partial results in shared memory for inter-warp reduction
- Compute norm2 += x[i+1] * conj(x[i+1]) for all elements beyond the first
- Final thread 0 combines all warp results

Code blocks for context:
```cpp
// From roclapack_geqr2.hpp lines 97-122
T norm2 = 0;
for(I i = tid; i < mm - 1; i += MAX_THDS)
    norm2 += x[i + 1] * conj(x[i + 1]);

// reduce squared entries to find squared norm of x
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

    // set tau, beta, and put scaling factor into sval[0]
    run_set_taubeta<T>(tmptau, &norm2, x, diag + j);

    tau[j] = tmptau[0];
    sval[0] = norm2;

    tmptau[0] = conj(tmptau[0]);
}
__syncthreads();
```

Implement a complete warp reduction kernel for dot product computation, explaining why shift_left is more efficient than traditional shared memory reductions."""
            }]
        }
    })

    # L1-3: LARF Left-Side Kernel (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L1_larf_left",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the larf_left_kernel that applies a Householder reflector from the left side to a matrix column. This is the core operation in applying QR transformations.

Requirements:
- Each workgroup of NB_X threads operates on one column of matrix A
- Compute w = conj(A[:,j])' * x via GEMV with warp reduction
- Apply rank-1 update: A[:,j] -= (tau * w) * x via GER
- Use shared memory for both reduction and storing x vector
- Grid dimensions: dim3(1, n, batch_count), Block dimensions: dim3(NB_X)

Code blocks for context:
```cpp
// From rocauxiliary_larf.hpp lines 48-116
template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_left_kernel(const I m,
                                                               const I n,
                                                               U xx,
                                                               const rocblas_stride shiftX,
                                                               const I incX,
                                                               const rocblas_stride strideX,
                                                               const T* tauA,
                                                               const rocblas_stride strideP,
                                                               U AA,
                                                               const rocblas_stride shiftA,
                                                               const I lda,
                                                               const rocblas_stride strideA)
{
    I bid = blockIdx.z;
    I tx = threadIdx.x;
    I col = blockIdx.y;

    // select batch instance
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const T* tau = tauA + bid * strideP;

    A += col * size_t(lda);

    I start = (incX > 0 ? 0 : (m - 1) * -incX);

    T res = 0;

    extern __shared__ double smem[];
    T* sdata = reinterpret_cast<T*>(smem);
    T* xs = sdata + (NB_X / warpSize);

    for(I i = tx; i < m; i += NB_X)
        xs[i] = x[start + i * size_t(incX)];

    //
    // GEMV
    //
    for(I i = tx; i < m; i += NB_X)
        res += conj(A[i]) * xs[i];

    // reduction
    res += shift_left(res, 1);
    res += shift_left(res, 2);
    res += shift_left(res, 4);
    res += shift_left(res, 8);
    res += shift_left(res, 16);
    if(warpSize > 32)
        res += shift_left(res, 32);
    if(tx % warpSize == 0)
        sdata[tx / warpSize] = res;
    __syncthreads();
    if(tx == 0)
    {
        for(I k = 1; k < NB_X / warpSize; k++)
            res += sdata[k];

        sdata[0] = res;
    }
    __syncthreads();

    //
    // GER
    //
    res = -tau[0] * conj(sdata[0]);
    for(I i = tx; i < m; i += NB_X)
        A[i] += res * xs[i];
}
```

Provide complete implementation with detailed explanation of shared memory layout and synchronization requirements."""
            }]
        }
    })

    # L1-4: Small Matrix QR Kernel Analysis (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_geqrf_L1_small_kernel",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the geqr2_kernel_small specialized kernel that performs complete QR factorization in shared memory for small matrices. Explain the performance benefits and limitations.

The kernel performs all QR steps (for j=0 to min(m,n)-1):
1. LARFG: Generate Householder reflector for column j
2. GEMV: Compute w = tau * conj(V[:,j:n-1])' * V[:,j]
3. GER: Apply rank-1 update V[:,j+1:n-1] -= V[:,j] * w'

Code context from roclapack_geqr2.hpp lines 43-166 shows:
- Matrix loaded into shared memory: a[m*n]
- Workspace: w[n], tmptau[1], sval[MAX_THDS/warpSize]
- Condition: lmemsize <= sharedMemPerBlock && nn == mm
- Launch: dim3(1,1,batch_count), dim3(256), lmemsize

Questions to address:
1. Why is the condition nn==mm required (square submatrix)?
2. Calculate shared memory requirement for m=256, n=256, float type
3. What is the performance advantage vs. iterative geqr2_template?
4. Why is this kernel beneficial for batched operations?
5. What are the maximum matrix dimensions on typical GPUs (48KB shared memory)?"""
            }]
        }
    })

    # L2-1: LARFT Block Reflector Construction (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L2_larft_forward",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the larft_kernel_forward that constructs the triangular factor T for a block of k Householder reflectors in forward direction. This enables efficient block updates in GEQRF.

The algorithm computes T where (I - V*T*V') represents k Householder reflectors:
- T is k×k upper triangular
- T[i,i] = tau[i]
- T[0:i-1,i] = -tau[i] * T[0:i-1,0:i-1] * (V[:,0:i-1]' * V[:,i])

Requirements:
- Work entirely in shared memory (F and work arrays)
- Forward direction: process reflectors 0, 1, ..., k-1
- For each kk from 1 to k-1:
  - GEMV: work[0:kk-1] = tau[kk] * conj(V[:,kk:n-1])' * V[:,kk] + F[0:kk-1,kk]
  - TRMV: F[0:kk-1,kk] = F[0:kk-1,0:kk-1] * work[0:kk-1]
- Handle both column-wise and row-wise storage

Code blocks for context:
```cpp
// From rocauxiliary_larft.hpp lines 251-343
template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_forward(const rocblas_storev storev,
                                           const rocblas_int n,
                                           const rocblas_int k,
                                           U VA,
                                           const rocblas_int shiftV,
                                           const rocblas_int ldv,
                                           const rocblas_stride strideV,
                                           T* tauA,
                                           const rocblas_stride strideT,
                                           T* FA,
                                           const rocblas_int ldfA,
                                           const rocblas_stride strideF)
{
    const rocblas_int bid = hipBlockIdx_y;
    const rocblas_int tid = hipThreadIdx_x;
    const rocblas_int tid_inc = hipBlockDim_x;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* tau = tauA + bid * strideT;
    T* Ftemp = FA + bid * strideF;

    // shared memory setup
    extern __shared__ double lmem[];
    T* work = reinterpret_cast<T*>(lmem);
    T* F = work + k;
    rocblas_int ldf = k;

    // copy F to shared memory
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // --------- MAIN BODY ---------
    for(rocblas_int kk = 1; kk < k; kk++)
    {
        const rocblas_int mm = kk;
        const rocblas_int nn = n - 1 - kk;

        T* Fx = F + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + (kk + 1) + kk * ldv;

            // gemv (conjugate transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk + (kk + 1) * ldv;

            // gemv (no transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = i; j < mm; j++)
                temp += F[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}
```

Implement the complete kernel with detailed comments explaining the recursive construction of T."""
            }]
        }
    })

    # L2-2: LARFB Block Reflector Application (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_geqrf_L2_larfb_algorithm",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Explain the LARFB (apply block reflector) algorithm used in GEQRF to apply k Householder reflectors efficiently. The algorithm computes:

C := H * C  where H = I - V * T * V'

and V is m×k with Householder vectors, T is k×k triangular factor.

For left-side application in GEQRF, the matrix C is partitioned:
C = [C1]  where C1 is k×n,  V = [V1]  where V1 is k×k triangular
    [C2]        C2 is (m-k)×n      [V2]        V2 is (m-k)×k

The algorithm (from rocauxiliary_larfb.hpp):
1. W = C1  (copy to workspace)
2. W = V1' * C1  (TRMM with unit diagonal)
3. W = V1' * C1 + V2' * C2  (GEMM if m > k)
4. W = T' * W  (TRMM)
5. C2 = C2 - V2 * W  (GEMM if m > k)
6. W = V1 * W  (TRMM with unit diagonal)
7. C1 = C1 - W  (element-wise subtraction)

Questions to address:
1. Why is this more efficient than applying k reflectors individually?
2. What is the computational complexity: block method vs. individual reflectors?
3. Explain why V1 must be unit triangular and what the diagonal values represent
4. How does the workspace size W (k×n) compare to storing k separate work vectors?
5. Why are steps 2 and 6 TRMM instead of GEMM?
6. Trace the algorithm for a 6×4 matrix with k=2 block size"""
            }]
        }
    })

    # L2-3: Conjugation for Complex Tau Values (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L2_lacgv",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the lacgv (conjugate vector) operation used in GEQRF to conjugate tau values before and after LARF operations for complex matrices.

Requirements:
- Template should enable only for complex types (rocblas_is_complex<T>)
- For real types, the kernel should do nothing (empty implementation)
- Support both positive and negative increments
- Handle offset calculation for negative increments
- Use 64 threads with appropriate grid dimensions

Code blocks for context:
```cpp
// From rocauxiliary_lacgv.hpp lines 51-67
template <typename T, typename I, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void conj_in_place(const I m,
                                    const I n,
                                    U A,
                                    const rocblas_stride shifta,
                                    const I lda,
                                    const rocblas_stride stridea)
{
    I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    I b = hipBlockIdx_z;

    T* Ap = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < m && j < n)
        Ap[i + j * lda] = conj(Ap[i + j * (int64_t)lda]);
}

// From rocauxiliary_lacgv.hpp lines 92-118
template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_lacgv_template(rocblas_handle handle,
                                        const I n,
                                        U x,
                                        const rocblas_stride shiftx,
                                        const I incx,
                                        const rocblas_stride stridex,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("lacgv", "n:", n, "shiftX:", shiftx, "incx:", incx, "bc:", batch_count);

    // quick return
    if(n == 0 || !batch_count || !COMPLEX)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // handle negative increments
    rocblas_stride offset = incx < 0 ? shiftx - (n - 1) * incx : shiftx;

    // conjugate x
    constexpr int LACGV_NTHREADS = 64;
    I blocks = (n - 1) / LACGV_NTHREADS + 1;
    ROCSOLVER_LAUNCH_KERNEL(conj_in_place<T>, dim3(1, blocks, batch_count), dim3(1, 64, 1), 0,
                            stream, (I)1, n, x, offset, incx, stridex);

    return rocblas_status_success;
}
```

Usage in GEQR2 (roclapack_geqr2.hpp lines 286-297):
```cpp
// conjugate tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);

rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                        shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                        A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                        scalars, Abyx_norms, (T**)work_workArr);

// restore tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);
```

Explain why tau must be conjugated for complex matrices when applying Householder reflectors."""
            }]
        }
    })

    # L2-4: GEQR2 Unblocked Algorithm Flow (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_geqrf_L2_geqr2_flow",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Trace the execution flow of the unblocked GEQR2 algorithm for a 5×3 complex matrix. Show all kernel launches, memory operations, and intermediate matrix states.

Initial matrix A (5×3):
```
[ a11  a12  a13 ]
[ a21  a22  a23 ]
[ a31  a32  a33 ]
[ a41  a42  a43 ]
[ a51  a52  a53 ]
```

GEQR2 algorithm (roclapack_geqr2.hpp lines 263-299):
```cpp
I dim = std::min(m, n); // total number of pivots
for(I j = 0; j < dim; ++j)
{
    I mm = m - j;
    I nn = n - j;

    const size_t lmemsize = ((256 / props.warpSize) + mm + nn + 1 + mm * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn == mm)
    {
        ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>), dim3(1, 1, batch_count), dim3(256),
                                lmemsize, stream, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, (S*)diag + j, dim, ipiv + j, strideP);
        break;
    }

    // generate Householder reflector to work on column j
    rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the left
    if(j < n - 1)
    {
        // conjugate tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);

        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);

        // restore tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);
    }
}

// restore diagonal values of A
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>), dim3(batch_count, blocks, 1),
                        dim3(1, DIAG_NTHREADS, 1), 0, stream, (S*)diag, 0, dim, A, shiftA, lda,
                        strideA, dim);
```

For each iteration j=0,1,2 show:
1. Values of mm, nn
2. Which kernel path is taken (small kernel or larfg+larf)
3. The submatrix being processed
4. Tau value computed
5. Intermediate state of matrix A after each iteration
6. Final matrix A with R in upper triangle and Householder vectors below diagonal"""
            }]
        }
    })

    # L3-1: GEQRF Blocked Algorithm Implementation (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L3_main_algorithm",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the main blocked GEQRF algorithm that uses GEQR2-LARFT-LARFB pipeline for efficient QR factorization of large matrices.

The algorithm for m×n matrix A with block size nb:
```
for j = 0, nb, 2*nb, ..., dim-GEQxF_GEQx2_SWITCHSIZE
    jb = min(dim - j, nb)

    // Factor panel A[j:m-1, j:j+jb-1]
    GEQR2(A[j:m-1, j:j+jb-1])

    if j+jb < n:
        // Build triangular factor T
        LARFT(A[j:m-1, j:j+jb-1], tau[j:j+jb-1], T)

        // Apply block reflector to trailing matrix
        LARFB(A[j:m-1, j:j+jb-1], T, A[j:m-1, j+jb:n-1])

// Factor remaining block with GEQR2
if j < dim:
    GEQR2(A[j:m-1, j:n-1])
```

Requirements:
- Handle early exit to unblocked GEQR2 if m or n <= GEQxF_GEQx2_SWITCHSIZE
- Use GEQxF_BLOCKSIZE for nb (typically 32)
- Properly compute triangular factor workspace dimensions (ldw=nb, strideW=nb*nb)
- Support both BATCHED and STRIDED variants
- Handle memory allocation for scalars, work arrays, norms, diagonal storage

Code blocks for context:
```cpp
// From roclapack_geqrf.hpp lines 100-176
template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_geqrf_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms_trfact,
                                        T* diag_tmptr,
                                        T** workArr)
{
    ROCSOLVER_ENTER("geqrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if the matrix is small, use the unblocked (BLAS-levelII) variant of the
    // algorithm
    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        rocsolver_geqr2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count,
                                    scalars, work_workArr, Abyx_norms_trfact, diag_tmptr);
        return rocblas_status_success;
    }

    I dim = std::min(m, n); // total number of pivots
    I jb, j = 0;

    I nb = GEQxF_BLOCKSIZE;
    I ldw = GEQxF_BLOCKSIZE;
    rocblas_stride strideW = rocblas_stride(ldw) * ldw;

    while(j < dim - GEQxF_GEQx2_SWITCHSIZE)
    {
        // Factor diagonal and subdiagonal blocks
        jb = std::min(dim - j, nb); // number of columns in the block
        rocsolver_geqr2_template<T>(handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    (ipiv + j), strideP, batch_count, scalars, work_workArr,
                                    Abyx_norms_trfact, diag_tmptr);

        // apply transformation to the rest of the matrix
        if(j + jb < n)
        {
            // compute block reflector
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise,
                                        m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        (ipiv + j), strideP, Abyx_norms_trfact, ldw, strideW,
                                        batch_count, scalars, (T*)work_workArr, workArr);

            // apply the block reflector
            rocsolver_larfb_template<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_forward_direction, rocblas_column_wise, m - j, n - j - jb, jb, A,
                shiftA + idx2D(j, j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW, A,
                shiftA + idx2D(j, j + jb, lda), lda, strideA, batch_count, diag_tmptr, workArr);
        }
        j += nb;
    }

    // factor last block
    if(j < dim)
        rocsolver_geqr2_template<T>(handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, (ipiv + j), strideP, batch_count, scalars,
                                    work_workArr, Abyx_norms_trfact, diag_tmptr);

    return rocblas_status_success;
}
```

Provide complete implementation with detailed comments explaining block sizes and workspace reuse."""
            }]
        }
    })

    # L3-2: Memory Management and Workspace Calculation (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_geqrf_L3_memory_management",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the memory workspace calculation for GEQRF and explain the memory reuse strategy across different subroutines.

From roclapack_geqrf.hpp lines 43-98, the workspace consists of:
1. size_scalars: Constants for rocBLAS calls
2. size_work_workArr: Re-usable workspace and array of pointers
3. size_Abyx_norms_trfact: Norms from LARF/LARFG and triangular factor T
4. size_diag_tmptr: Diagonal storage and temporary pointers
5. size_workArr: Array of pointers for batched TRMM calls

For small matrices (m or n <= GEQxF_GEQx2_SWITCHSIZE=128):
```cpp
rocsolver_geqr2_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars,
                                          size_work_workArr, size_Abyx_norms_trfact,
                                          size_diag_tmptr);
size_workArr = 0;
```

For large matrices:
```cpp
I jb = GEQxF_BLOCKSIZE;

// Triangular factor T storage
size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

// Requirements for GEQR2 with sub-blocks
rocsolver_geqr2_getMemorySize<BATCHED, T>(m, jb, batch_count, size_scalars, &w1, &s2, &s1);
size_Abyx_norms_trfact = std::max(s2, size_Abyx_norms_trfact);

// Requirements for LARFT
rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);

// Requirements for LARFB
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                          &s2, &unused);

size_work_workArr = std::max(w1, w2);
size_diag_tmptr = std::max(s1, s2);

// Double size for batched TRMM workspace arrays
if(BATCHED)
    size_workArr *= 2;
```

Questions to address:
1. Calculate exact workspace sizes for m=1024, n=1024, batch_count=32, float type
2. Explain why size_Abyx_norms_trfact can be reused for both norms and T storage
3. How is size_work_workArr reused between GEQR2, LARFT, and LARFB?
4. Why does BATCHED mode require 2× size_workArr?
5. What is the total memory overhead compared to the input matrix size?
6. How would workspace requirements change if using the inverse-based LARFB variant?"""
            }]
        }
    })

    # L3-3: Complete Production API (CODING)
    entries.append({
        "custom_id": "roclapack_geqrf_L3_production_api",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the complete production C API wrapper for GEQRF including all four precisions (S/D/C/Z) and 64-bit integer variants.

Requirements:
- Implement rocsolver_geqrf_impl template that handles memory allocation
- Create C wrappers: sgeqrf, dgeqrf, cgeqrf, zgeqrf
- Create 64-bit variants: sgeqrf_64, dgeqrf_64, cgeqrf_64, zgeqrf_64
- Handle device memory size queries
- Support 8 memory regions allocation via rocblas_device_malloc
- Initialize scalars on device
- Proper error handling for invalid handle, argument checking, memory errors

Code blocks for context:
```cpp
// From roclapack_geqrf.cpp lines 32-98
template <typename T, typename I, typename U>
rocblas_status
    rocsolver_geqrf_impl(rocblas_handle handle, const I m, const I n, U A, const I lda, T* ipiv)
{
    ROCSOLVER_ENTER_TOP("geqrf", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geqr2_geqrf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride stridep = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    bool optim_mem;
    size_t size_work_workArr_work1, size_work2, size_work3, size_work4, size_workArr;
    // extra requirements for calling GEQR2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GEQR2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_geqrf_getMemorySize<false, false, T>(
        m, n, batch_count, &size_scalars, &size_work_workArr_work1, &size_work2, &size_work3,
        &size_work4, &size_Abyx_norms_trfact, &size_diag_tmptr, &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_workArr_work1, size_work2, size_work3, size_work4,
            size_Abyx_norms_trfact, size_diag_tmptr, size_workArr);

    // memory workspace allocation
    void *scalars, *work_workArr_work1, *work2, *work3, *work4, *Abyx_norms_trfact, *diag_tmptr,
        *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr_work1, size_work2, size_work3,
                              size_work4, size_Abyx_norms_trfact, size_diag_tmptr, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr_work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    Abyx_norms_trfact = mem[5];
    diag_tmptr = mem[6];
    workArr = mem[7];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_geqrf_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, stridep, batch_count, (T*)scalars,
        work_workArr_work1, work2, work3, work4, (T*)Abyx_norms_trfact, (T*)diag_tmptr,
        (T**)workArr, optim_mem);
}
```

C wrappers (lines 110-206):
```cpp
extern "C" {

rocblas_status rocsolver_sgeqrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geqrf_impl<float>(handle, m, n, A, lda, ipiv);
}

// ... similar for dgeqrf, cgeqrf, zgeqrf

#ifdef HAVE_ROCBLAS_64
rocblas_status rocsolver_sgeqrf_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* ipiv)
{
    return rocsolver::rocsolver_geqrf_impl<float>(handle, m, n, A, lda, ipiv);
}
#else
rocblas_status rocsolver_sgeqrf_64(...)
{
    return rocblas_status_not_implemented;
}
#endif

// ... similar for other 64-bit variants

} // extern C
```

Provide complete implementation for all precision variants with proper error handling."""
            }]
        }
    })

    # L3-4: Performance Optimization and Tuning (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_geqrf_L3_performance_tuning",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the performance characteristics and tuning parameters for GEQRF on AMD GPUs. Explain the tradeoffs and optimization strategies.

Key tuning parameters:
1. **GEQxF_GEQx2_SWITCHSIZE = 128**: Threshold to switch from blocked to unblocked algorithm
2. **GEQxF_BLOCKSIZE = 32**: Block size for panel factorization
3. **LARF_SSKER_MAX_DIM = 1024, LARF_SSKER_MIN_DIM = 64**: Small-size kernel thresholds for LARF
4. **LARFG_SSKER_MAX_N**: Small-size threshold for Householder generation
5. **LARFT_SWITCHSIZE**: Threshold to use shared memory kernel vs. iterative LARFT
6. **BS1 = block size for LARFT kernels**

Performance considerations:
1. **LARF kernel selection** (rocauxiliary_larf.hpp lines 305-346):
   - Small-size kernel: m ≤ MIN_DIM, n ≤ MIN_DIM
   - Optimized kernel: (leftside && (n ≤ 1024 || m ≥ 2048)) || (rightside && (m ≤ 1024 || n ≥ 2048))
   - Fallback: GEMV + GER using rocBLAS

2. **Memory hierarchy**:
   - L1 cache: Small matrices fit in shared memory
   - L2 cache: Panel reuse across LARFB updates
   - Global memory: Large trailing matrix updates

3. **Computational intensity**:
   - GEQR2: O(n²) with low arithmetic intensity
   - LARFT: O(k²n) with k << n
   - LARFB: O(kmn) with high arithmetic intensity (GEMM-based)

Questions to address:
1. For a 4096×4096 matrix, calculate:
   - Number of GEQR2 calls
   - Number of LARFT calls
   - Number of LARFB calls
   - Total FLOPs distribution between operations

2. Why is block size 32 optimal for CDNA architectures (MI200/MI300)?

3. Explain the memory access pattern for LARFB and cache utilization

4. How does performance scale with batch size?

5. Compare performance characteristics:
   - GEQRF vs. GELQF (QR vs. LQ)
   - Small matrices (256×256) vs. large (4096×4096)
   - Square vs. tall (m >> n) vs. wide (n >> m)

6. Propose tuning strategy for different GPU architectures (different warp sizes, shared memory sizes)"""
            }]
        }
    })

    return entries

def main():
    entries = generate_entries()
    output_file = Path(__file__).parent / "roclapack_geqrf.jsonl"
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    coding_tasks = sum(1 for e in entries
                      if 'implement' in e['body']['messages'][0]['content'].lower()[:200]
                      or 'write' in e['body']['messages'][0]['content'].lower()[:200]
                      or 'provide the complete' in e['body']['messages'][0]['content'].lower()[:300]
                      or 'provide complete' in e['body']['messages'][0]['content'].lower()[:300])

    print(f"Generated {len(entries)} entries")
    print(f"Output: {output_file}")
    print(f"Coding tasks: {coding_tasks}/{len(entries)} ({coding_tasks/len(entries)*100:.1f}%)")

if __name__ == "__main__":
    main()
