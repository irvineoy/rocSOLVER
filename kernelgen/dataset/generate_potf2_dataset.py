#!/usr/bin/env python3
"""
Generate SFT dataset for roclapack_potf2.yaml
POTF2: Unblocked Cholesky factorization (A = L*L' or A = U'*U)
"""

import json
import time

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    return {
        'id': str(int(time.time() * 1000000)),
        'level': level,
        'interface': interface,
        'instruction': instruction,
        'context_text': context_text,
        'code_blocks': code_blocks,
        'answer': answer,
        'rationale': rationale,
        'tags': tags
    }

entries = []

# L1-1: sqrtDiagOnward kernel - coding task
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Implement the positive definiteness check and diagonal update in the sqrtDiagOnward kernel for real types. Given the diagonal element after rank-1 update (M[loc] - res[id]), compute the Cholesky factor and reciprocal for scaling the column below.',
    context_text='sqrtDiagOnward kernel computes sqrt of diagonal after rank-1 update in Cholesky factorization. For column j, after DOT product, we compute L(j,j) = sqrt(A(j,j) - sum) and store 1/L(j,j) for SCAL.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''template <typename T, typename I, typename INFO, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    T t = M[loc] - res[id];

    // TODO: implement positive definiteness check and diagonal update
}'''
    }],
    answer='''if(t <= 0.0)
{
    // error for non-positive definiteness
    if(info[id] == 0)
        info[id] = j + 1; // use fortran 1-based index
    M[loc] = t;
    res[id] = 0;
}
else
{
    // minor is positive definite
    M[loc] = sqrt(t);
    res[id] = 1 / M[loc];
}''',
    rationale='If t <= 0, the matrix is not positive definite, so we set info[id] to the failing column (Fortran 1-based) and set res[id]=0 to prevent further updates. Otherwise, we compute M[loc]=sqrt(t) as the Cholesky factor and res[id]=1/M[loc] as the reciprocal for subsequent SCAL operation.',
    tags=['coding', 'cholesky', 'positive-definiteness', 'error-handling']
))

# L1-2: potf2_simple packed storage indexing - coding task
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Write the idx_lower function that maps (i,j) coordinates to packed storage index for lower triangular matrices. The function should assert i >= j and compute the linear index for column-major packed storage.',
    context_text='potf2_kernel_small uses packed storage in shared memory to reduce memory footprint. For n×n lower triangular matrix, only n(n+1)/2 elements are stored. idx_lower maps 2D coordinates to 1D packed index.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_potf2_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''/**
 * indexing for packed storage
 * for lower triangular
 *
 * ---------------------------
 * 0
 * 1      n
 * *      (n+1)
 * *
 * (n-1)  ...        n*(n+1)/2
 * ---------------------------
 **/
template <typename I>
__device__ static I idx_lower(I i, I j, I n)
{
    // TODO: implement packed index calculation
}'''
    }],
    answer='''assert((0 <= i) && (i <= (n - 1)));
assert((0 <= j) && (j <= (n - 1)));
assert(i >= j);

return ((i - j) + (j * (2 * n + 1 - j)) / 2);''',
    rationale='For column j, elements start at offset j*(2*n+1-j)/2 (sum of column lengths n, n-1, ..., n-j+1). Within column j, element at row i has offset (i-j) from the column start. Combined: (i-j) + j*(2*n+1-j)/2.',
    tags=['coding', 'packed-storage', 'indexing', 'shared-memory']
))

# L1-3: potf2_simple rank-1 update pattern
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Analyze the rank-1 update pattern in potf2_simple for lower triangular factorization. How does the kernel update A22 after computing column kcol, and why is synchronization required?',
    context_text='After computing L(kcol,kcol) and scaling L(kcol+1:n,kcol), potf2_simple performs a symmetric rank-1 update on the trailing submatrix A22.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_potf2_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
{
    auto const vj = A[idx_lower(j, kcol, lda)];
    for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
    {
        bool const lower_part = (i >= j);
        if(lower_part)
        {
            auto const vi = A[idx_lower(i, kcol, lda)];
            auto const ij = idx_lower(i, j, lda);

            A[ij] = A[ij] - vi * conj(vj);
        }
    }
}

__syncthreads();'''
    }],
    answer='The kernel computes A22 = A22 - vl21 * vl21\', where vl21 = L(kcol+1:n, kcol). Each thread processes a subset of (i,j) pairs with i >= j (lower triangular). The update is A(i,j) -= L(i,kcol) * conj(L(j,kcol)). __syncthreads() is required because: (1) all threads must finish reading L(:,kcol) before overwriting A22, and (2) all threads must complete the rank-1 update before the next iteration computes L(kcol+1,kcol+1), which depends on the updated A(kcol+1,kcol+1).',
    rationale='This implements the SYRK update A := A - L*L\' for the trailing submatrix. Synchronization prevents race conditions between reading column kcol and writing A22, and ensures the updated diagonal is ready for the next iteration.',
    tags=['rank-1-update', 'synchronization', 'cholesky', 'thread-collaboration']
))

# L1-4: potf2_kernel_small shared memory layout - coding task
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Calculate the shared memory size required for potf2_kernel_small for a 64×64 double-precision matrix. Show the packed storage size calculation.',
    context_text='potf2_kernel_small uses packed storage in LDS to fit larger matrices. The launcher allocates lmemsize bytes of shared memory.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_potf2_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''template <typename T, typename I, typename INFO, typename U>
rocblas_status potf2_run_small(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               INFO* info,
                               const I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    size_t lmemsize = sizeof(T) * (n * (n + 1)) / 2;

    bool const is_upper = (uplo == rocblas_fill_upper);
    ROCSOLVER_LAUNCH_KERNEL((potf2_kernel_small<T, I, INFO, U>), dim3(1, 1, batch_count),
                            dim3(BS2, BS2, 1), lmemsize, stream, is_upper, n, A, shiftA, lda,
                            strideA, info);
}'''
    }],
    answer='For n=64, T=double (8 bytes):\nlmemsize = 8 * (64 * 65) / 2 = 8 * 2080 = 16640 bytes = 16.25 KB\n\nThe packed storage stores only the lower (or upper) triangle plus diagonal: n*(n+1)/2 = 64*65/2 = 2080 elements. At 8 bytes per double, this is 16640 bytes.',
    rationale='Packed storage reduces memory by ~2× compared to full n×n storage (64×64×8 = 32768 bytes). This allows fitting larger matrices in the 64 KB shared memory limit per thread block on AMD GPUs.',
    tags=['coding', 'shared-memory', 'packed-storage', 'memory-optimization']
))

# L1-5: potf2_kernel_small copy-in with conjugation
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Explain why potf2_kernel_small conjugates elements when copying upper triangular data to packed storage with use_compute_lower=true. What is the mathematical justification?',
    context_text='For upper triangular input, potf2_kernel_small can transpose and conjugate to compute with lower triangular algorithm.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_potf2_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''bool const use_compute_lower = true;

if(is_lower)
{
    for(I j = j_start; j < n; j += j_inc)
    {
        for(I i = j + i_start; i < n; i += i_inc)
        {
            auto const ij = i + j * static_cast<int64_t>(lda);
            auto const ij_packed = idx_lower(i, j, n);

            Ash[ij_packed] = A[ij];
        }
    }
}
else
{
    for(I j = j_start; j < n; j += j_inc)
    {
        for(I i = i_start; i <= j; i += i_inc)
        {
            auto const ij = i + j * static_cast<int64_t>(lda);
            auto const ij_packed = (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

            auto const aij = A[ij];
            Ash[ij_packed] = (use_compute_lower) ? conj(aij) : aij;
        }
    }
}'''
    }],
    answer='For Hermitian positive definite matrices, A = A\' (conjugate transpose). Upper factorization computes A = U\'*U, while lower computes A = L*L\'. The relationship is: A\' = (U\'*U)\' = U\'*(U\')\'  = U\'*U = A, or equivalently, conj(A) = L*L\' where L = conj(U\'). By transposing indices (i,j) -> (j,i) and conjugating values, the kernel converts upper input to lower format and reuses the lower algorithm.',
    rationale='This optimization allows a single potf2_simple implementation (lower triangular) to handle both uplo modes. Conjugation is required for complex types to maintain Hermitian property; for real types, conj() is a no-op.',
    tags=['hermitian-matrices', 'conjugate-transpose', 'algorithm-reuse', 'complex-arithmetic']
))

# L1-6: rocsolver_ger kernel launch configuration
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='For a 512×128 matrix update using rocsolver_ger with BS2=32, calculate the grid dimensions (blocksx, blocksy) and the total number of threads launched.',
    context_text='rocsolver_ger performs rank-1 update A := A + alpha*x*y\'. Used in POTF2 for non-positive definiteness handling with custom stride support.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_ger_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''I blocksx = (m - 1) / BS2 + 1;
I blocksy = (n - 1) / BS2 + 1;
dim3 grid(blocksx, blocksy, batch_count);
dim3 threads(BS2, BS2, 1);'''
    }],
    answer='Given m=512, n=128, BS2=32, batch_count=1:\n\nblocksx = (512 - 1) / 32 + 1 = 511 / 32 + 1 = 15 + 1 = 16\nblocksy = (128 - 1) / 32 + 1 = 127 / 32 + 1 = 3 + 1 = 4\n\ngrid = (16, 4, 1)\nthreads = (32, 32, 1)\n\nTotal threads = 16 * 4 * 1 * 32 * 32 = 64 * 1024 = 65536 threads',
    rationale='Each block covers BS2×BS2 = 32×32 elements. For 512×128 matrix, we need ⌈512/32⌉ = 16 blocks in x and ⌈128/32⌉ = 4 blocks in y. Total threads = grid.x * grid.y * grid.z * threads.x * threads.y * threads.z.',
    tags=['coding', 'launch-configuration', 'grid-sizing', 'rank-1-update']
))

# L2-1: potf2 algorithm flow for large matrices
entries.append(create_entry(
    level='L2',
    interface='potf2',
    instruction='Trace the execution flow of rocsolver_potf2_template for a 1024×1024 lower triangular matrix. Which kernels/BLAS calls are invoked per iteration, and what are the computational complexities?',
    context_text='For n > POTRF_BLOCKSIZE, POTF2 uses unblocked column-by-column algorithm with DOT, sqrtDiagOnward, GEMV, and SCAL.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''for(I j = 0; j < n; ++j)
{
    // Compute L(J,J) and test for non-positive-definiteness.
    rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda, strideA, A,
                                shiftA + idx2D(j, 0, lda), lda, strideA, batch_count,
                                pivots, work);

    ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), dim3(batch_count), dim3(1), 0, stream,
                            A, shiftA, strideA, idx2D(j, j, lda), j, pivots, info);

    // Compute elements J+1:N of column J
    if(j < n - 1)
    {
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                        strideA, batch_count);

        rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j, scalars, 0, A,
                            shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                            shiftA + idx2D(j, 0, lda), lda, strideA, scalars + 2, 0, A,
                            shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count,
                            nullptr);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                        strideA, batch_count);

        rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A,
                            shiftA + idx2D(j + 1, j, lda), (I)1, strideA, batch_count);
    }
}'''
    }],
    answer='''Per iteration j (0 to n-1):
1. DOT(j): Compute sum of L(j,0:j-1)^2, O(j) flops
2. sqrtDiagOnward: L(j,j) = sqrt(A(j,j) - sum), O(1) flops
3. (if complex) LACGV(j): Conjugate L(j,0:j-1), O(j) flops
4. GEMV(n-j-1, j): A(j+1:n,j) -= A(j+1:n,0:j-1) * conj(L(j,0:j-1)), O(j*(n-j)) flops
5. (if complex) LACGV(j): Restore L(j,0:j-1), O(j) flops
6. SCAL(n-j-1): L(j+1:n,j) = A(j+1:n,j) / L(j,j), O(n-j) flops

Total complexity: sum_{j=0}^{n-1} O(j*(n-j)) = O(n^3/3)

For n=1024, this is ~358M flops. The algorithm is memory-bound due to small GEMV sizes (j ≤ 1024) and low arithmetic intensity.''',
    rationale='The unblocked algorithm processes one column at a time. DOT computes the diagonal update, GEMV applies the rank-j update to the trailing column, and SCAL normalizes by the diagonal. Complexity is O(n^3/3), but performance is poor due to Level-2 BLAS dominance.',
    tags=['algorithm-analysis', 'complexity', 'blas-calls', 'unblocked-algorithm']
))

# L2-2: Small-matrix kernel fusion optimization
entries.append(create_entry(
    level='L2',
    interface='potf2',
    instruction='Compare the performance benefits of potf2_run_small versus the standard BLAS-based path for n=64 matrices. Consider kernel launch overhead, memory traffic, and arithmetic intensity.',
    context_text='potf2_run_small uses a single fused kernel with packed LDS storage, while the standard path uses separate DOT/GEMV/SCAL calls per column.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''if(n <= POTRF_BLOCKSIZE(T))
{
    // ----------------------
    // use specialized kernel
    // ----------------------
    potf2_run_small<T>(handle, uplo, n, A, shiftA, lda, strideA, info, batch_count);
}
else
{
    // standard BLAS-based path with n iterations
    for(I j = 0; j < n; ++j)
    {
        rocblasCall_dot<COMPLEX, T>(...);
        ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), ...);
        rocblasCall_gemv<T>(...);
        rocblasCall_scal<T>(...);
    }
}'''
    }],
    answer='''potf2_run_small benefits for n=64:

1. **Kernel launch overhead**: 1 kernel vs. n*(3-5) kernels = 1 vs. 192-320 launches. At ~5μs/launch, this saves ~1ms.

2. **Memory traffic**: Packed LDS uses 64*65/2*8 = 16.6 KB for copy-in/out, plus in-place updates in LDS. Standard path: each DOT/GEMV reads columns from DRAM repeatedly. For n=64, total reads ~64*(64*65/2)*8 ≈ 1.7 MB vs. 33 KB. ~50× reduction.

3. **Arithmetic intensity**: Standard path has low AI due to small GEMV sizes (j=0..63). Fused kernel achieves better cache reuse in LDS with AI ≈ (n^3/3) / (n^2 * sizeof(T)) ≈ 2.7 flops/byte for n=64.

Expected speedup: 10-20× for small matrices (n ≤ 64) due to reduced launch overhead and memory traffic.''',
    rationale='Kernel fusion eliminates launch overhead, keeps intermediate results in LDS, and improves memory bandwidth utilization. The crossover point is POTRF_BLOCKSIZE, beyond which the standard path benefits from optimized BLAS libraries.',
    tags=['kernel-fusion', 'performance-optimization', 'launch-overhead', 'memory-bandwidth']
))

# L2-3: lacgv integration for complex types
entries.append(create_entry(
    level='L2',
    interface='potf2',
    instruction='Explain why lacgv (conjugate vector) is called twice per iteration for complex Hermitian matrices in POTF2. What would happen if the second lacgv call were omitted?',
    context_text='For complex types, POTF2 calls lacgv before and after GEMV to conjugate L(j,0:j-1).',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, batch_count);

rocblasCall_gemv<T>(handle, rocblas_operation_none, n - j - 1, j, scalars, 0, A,
                    shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                    shiftA + idx2D(j, 0, lda), lda, strideA, scalars + 2, 0, A,
                    shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count,
                    nullptr);

if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                                strideA, batch_count);'''
    }],
    answer='''The GEMV computes: A(j+1:n,j) := -1 * A(j+1:n,0:j-1) * x + beta * A(j+1:n,j), where x = L(j,0:j-1).

For Hermitian matrices, we need: A(j+1:n,j) -= A(j+1:n,0:j-1) * conj(L(j,0:j-1)).

GEMV with rocblas_operation_none computes A*x (no conjugation). To get conj(x), we:
1. First lacgv: L(j,0:j-1) := conj(L(j,0:j-1))
2. GEMV: Compute A(j+1:n,0:j-1) * conj(L(j,0:j-1))
3. Second lacgv: Restore L(j,0:j-1) := conj(conj(L(j,0:j-1))) = original values

If the second lacgv were omitted, L(j,0:j-1) would remain conjugated, corrupting future iterations that read this data. Iteration j+1 would compute incorrect DOT and GEMV results.''',
    rationale='The conjugation is required for Hermitian rank-1 update semantics. The second lacgv restores the matrix to its original state to maintain invariants for subsequent iterations.',
    tags=['complex-arithmetic', 'hermitian-matrices', 'conjugation', 'algorithm-correctness']
))

# L2-4: Workspace allocation strategy - coding task
entries.append(create_entry(
    level='L2',
    interface='potf2',
    instruction='Implement the workspace size calculation for POTF2. Explain the roles of size_scalars, size_work, and size_pivots.',
    context_text='POTF2 allocates temporary workspace for BLAS scalars, DOT workspace, and pivots array.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''template <typename T, typename I>
void rocsolver_potf2_getMemorySize(const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_pivots)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_pivots = 0;
        return;
    }

    // TODO: Calculate workspace sizes
}'''
    }],
    answer='''// size of scalars (constants)
*size_scalars = sizeof(T) * 3;

if(n <= POTF2_MAX_SMALL_SIZE(T))
{
    *size_work = 0;
    *size_pivots = 0;
    return;
}

// size of workspace
constexpr I ROCBLAS_DOT_NB = 512;
*size_work = sizeof(T) * ((n - 1) / ROCBLAS_DOT_NB + 2) * batch_count;

// size of array to store pivots
*size_pivots = sizeof(T) * batch_count;

// Roles:
// - size_scalars: 3 device-side constants for GEMV/SCAL (alpha=-1, beta=1, and auxiliary)
// - size_work: DOT reduction workspace, sized for ROCBLAS_DOT_NB=512 thread blocks
// - size_pivots: batch_count elements to store 1/L(j,j) for SCAL, reused across iterations''',
    rationale='Scalars must be device-side for device pointer mode. DOT uses block-level reduction with up to ⌈n/512⌉+2 partial sums. Pivots stores the reciprocal for scaling. Small matrices use potf2_run_small which requires no workspace.',
    tags=['coding', 'workspace-allocation', 'memory-management', 'blas-requirements']
))

# L3-1: Interface design and algorithm selection
entries.append(create_entry(
    level='L3',
    interface='potf2',
    instruction='How does the POTF2 interface decide between potf2_run_small and the BLAS-based path? What are the trade-offs, and how would you determine the optimal threshold POTRF_BLOCKSIZE for a new GPU architecture?',
    context_text='POTF2 provides unblocked Cholesky for small/medium matrices. It is called by POTRF for panel factorization and standalone for small problems.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''if(n <= POTRF_BLOCKSIZE(T))
{
    potf2_run_small<T>(handle, uplo, n, A, shiftA, lda, strideA, info, batch_count);
}
else
{
    // BLAS-based path with DOT/GEMV/SCAL per column
}'''
    }],
    answer='''Algorithm selection:
- **potf2_run_small**: Single fused kernel with packed LDS storage. Best for n ≤ 64-128.
- **BLAS-based path**: Iterative calls to optimized rocBLAS DOT/GEMV/SCAL. Better for n > 128.

Trade-offs:
1. **LDS capacity**: Packed storage needs n*(n+1)/2 * sizeof(T). For double, max n ≈ sqrt(2*64KB/8) ≈ 127.
2. **Launch overhead vs. BLAS efficiency**: Fused kernel saves launches but has less optimized compute. BLAS path has overhead but uses heavily tuned kernels.
3. **Arithmetic intensity**: Small n favors fusion (memory-bound). Large n favors BLAS (more compute per column).

To determine optimal threshold:
1. Benchmark both paths for n=16,32,64,128,256,512 on target GPU
2. Measure time vs. n and find crossover point
3. Consider batch_count: larger batches favor potf2_run_small (more parallelism across batches)
4. Account for LDS limits: POTRF_BLOCKSIZE ≤ sqrt(2*LDS_SIZE/sizeof(T))

Typical values: POTRF_BLOCKSIZE = 64 for double, 128 for float.''',
    rationale='The threshold balances kernel fusion benefits (launch overhead, memory reuse) against BLAS library optimizations (vectorization, cache blocking). It depends on GPU architecture (LDS size, launch latency, BLAS library quality).',
    tags=['algorithm-selection', 'performance-tuning', 'kernel-fusion', 'interface-design']
))

# L3-2: Batched execution and error handling
entries.append(create_entry(
    level='L3',
    interface='potf2',
    instruction='Describe the batched execution model for POTF2. How does the interface handle per-batch error reporting (info array), and what happens to later batches when an early batch fails the positive definiteness test?',
    context_text='POTF2 supports batched execution with strideA and info array. Each batch instance is independent.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''I blocksReset = (batch_count - 1) / BS1 + 1;
dim3 gridReset(blocksReset, 1, 1);
dim3 threads(BS1, 1, 1);

// info=0 (starting with a positive definite matrix)
ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

// ...

ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), dim3(batch_count), dim3(1), 0, stream,
                        A, shiftA, strideA, idx2D(j, j, lda), j, pivots, info);'''
    }],
    answer='''Batched execution model:
1. **Independence**: Each batch instance processes A[b*strideA:(b+1)*strideA] independently with separate info[b].
2. **Error initialization**: reset_info kernel sets info[b]=0 for all batches before factorization starts.
3. **Per-batch failure**: sqrtDiagOnward checks if(info[id]==0) before setting info[id]=j+1. Only the first failure per batch is recorded.
4. **Continuation**: Failed batches continue executing (with res[id]=0 to prevent NaN propagation), but results are invalid. This avoids branch divergence and warp serialization.

Later batches are **not affected** by early batch failures:
- Each thread block in potf2_run_small processes one batch (gridDim.z = batch_count)
- BLAS-based path uses batch_count parameter; each GEMV/SCAL operates on strided arrays independently
- info array is indexed by batch ID, so each batch has its own error state

Design rationale: Batched GPUs achieve high throughput by keeping all SMs busy. Early exit for failed batches would cause load imbalance. Better to compute all batches (with error flags) and let the caller check info[] on host.''',
    rationale='Batched execution amortizes launch overhead and improves GPU occupancy. Independent error handling ensures one bad matrix does not stall the entire batch. The interface follows LAPACK conventions: info[b]=0 for success, info[b]=j+1 for failure at column j (1-based).',
    tags=['batched-execution', 'error-handling', 'load-balancing', 'interface-design']
))

# L3-3: Integration with POTRF and memory layout - coding task
entries.append(create_entry(
    level='L3',
    interface='potf2',
    instruction='POTF2 is called by POTRF for panel factorization. Given a 512×512 matrix with nb=64 block size, show how POTRF uses POTF2 with shiftA parameter to factorize the first panel in-place. Write the function call with correct shiftA and n values.',
    context_text='POTRF uses blocked algorithm: for each panel, call POTF2 on nb columns, then update trailing matrix with GEMM. shiftA allows operating on submatrix without copying.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_potf2.hpp',
        'language': 'hip',
        'content': '''template <typename T, typename I, typename INFO, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_potf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        INFO* info,
                                        const I batch_count,
                                        T* scalars,
                                        T* work,
                                        T* pivots)'''
    }],
    answer='''For lower triangular POTRF, first panel (columns 0:63) of 512×512 matrix:

rocsolver_potf2_template<T>(
    handle,
    rocblas_fill_lower,
    n = 512,              // panel height (rows 0:511)
    A,
    shiftA = 0,           // panel starts at A[0,0]
    lda = 512,            // leading dimension of full matrix
    strideA,
    info,
    batch_count,
    scalars,
    work,
    pivots
);

Wait, n should be the panel width for POTF2. Corrected:

For first panel of width nb=64:
rocsolver_potf2_template<T>(
    handle,
    rocblas_fill_lower,
    n = 64,               // factorize 64 columns
    A,
    shiftA = 0,           // panel starts at A[0,0]
    lda = 512,            // leading dimension of full matrix
    strideA,
    info,
    batch_count,
    scalars,
    work,
    pivots
);

Actually, POTF2 processes the trailing submatrix. For panel k=0 with nb=64, we factorize A[0:63, 0:63]. But POTF2's n parameter is the size of the square submatrix, not the full panel. Let me reconsider:

The first panel factorization:
rocsolver_potf2_template<T>(handle, rocblas_fill_lower, n=64, A, shiftA=0, lda=512, strideA, info, batch_count, scalars, work, pivots);

This factorizes the first 64×64 block. Then TRSM updates A[64:511, 0:63].''',
    rationale='POTF2 is called with n=nb (panel width) to factorize the leading nb×nb block. shiftA=k*nb + k*nb*lda for panel k (both row and column offset). The lda parameter allows POTF2 to access the full matrix with correct strides.',
    tags=['coding', 'blocked-algorithm', 'memory-layout', 'interface-integration']
))

# L1-7: ger_kernel memory access pattern - coding task
entries.append(create_entry(
    level='L1',
    interface='potf2',
    instruction='Implement the core computation in ger_kernel that performs the rank-1 update A := A + alpha*x*y\' with custom inca stride support.',
    context_text='rocsolver_ger is used in POTF2 when inca != 1 (interleaved matrix layout). Each thread updates one element of A.',
    code_blocks=[{
        'path': 'library/src/specialized/roclapack_ger_specialized_kernels.hpp',
        'language': 'hip',
        'content': '''template <typename T, typename I, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void ger_kernel(I m,
                                 I n,
                                 V alpha,
                                 rocblas_stride stridea,
                                 U1 xx,
                                 rocblas_stride shiftX,
                                 I incx,
                                 rocblas_stride strideX,
                                 U2 yy,
                                 rocblas_stride shiftY,
                                 I incy,
                                 rocblas_stride strideY,
                                 U3 AA,
                                 rocblas_stride shiftA,
                                 I inca,
                                 I lda,
                                 rocblas_stride strideA)
{
    // indices
    I bid = hipBlockIdx_z;
    I i = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I j = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + hipThreadIdx_y;

    // batch instance
    T a = load_scalar(alpha, bid, stridea);
    T* A = load_ptr_batch(AA, bid, shiftA, strideA);
    T* x = load_ptr_batch(xx, bid, shiftX, strideX);
    T* y = load_ptr_batch(yy, bid, shiftY, strideY);

    // TODO: implement rank-1 update with bounds check
}'''
    }],
    answer='''if(i < m && j < n)
{
    A[i * inca + j * lda] += a * x[i * incx] * y[j * incy];
}''',
    rationale='The kernel computes A(i,j) += alpha * x(i) * y(j) for all i in [0,m) and j in [0,n). The index calculation i*inca + j*lda supports both column-major (inca=1, lda=m) and interleaved (inca>1) layouts. Bounds check prevents out-of-range writes when m,n are not multiples of block size.',
    tags=['coding', 'rank-1-update', 'memory-access', 'strided-layout']
))

# Write entries to JSONL
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_potf2.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Calculate statistics
coding_tasks = sum(1 for e in entries if 'coding' in e['tags'])
level_counts = {'L1': 0, 'L2': 0, 'L3': 0}
for e in entries:
    level_counts[e['level']] += 1

print(f"Generated {len(entries)} entries")
print(f"Saved to {output_path}")
print(f"Coding tasks: {coding_tasks}/{len(entries)} ({100*coding_tasks/len(entries):.1f}%)")
print(f"Level breakdown: L1={level_counts['L1']}, L2={level_counts['L2']}, L3={level_counts['L3']}")
