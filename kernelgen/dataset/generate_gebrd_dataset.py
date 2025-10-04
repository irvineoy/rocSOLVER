#!/usr/bin/env python3
"""
Generate JSONL dataset for roclapack_gebrd.yaml
Creates L1/L2/L3 questions with ≥50% coding tasks
"""

import json
import time

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Create a single JSONL entry"""
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

def generate_dataset():
    entries = []

    # ============================================================
    # L1 ENTRIES (Single Function/Kernel)
    # ============================================================

    # L1-1: larf_left_kernel memory access (analysis)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Analyze the memory access pattern in the larf_left_kernel GEMV phase. How does it ensure coalesced memory access when reading from matrix A?',
        context_text='The larf_left_kernel applies a Householder reflector from the left side. Each work group operates on a column of A.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larf.hpp',
            'language': 'hip',
            'content': '''template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_left_kernel(const I m, const I n, ...)
{
    I bid = blockIdx.z;
    I tx = threadIdx.x;
    I col = blockIdx.y;

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    A += col * size_t(lda);  // Point to column

    T res = 0;
    for(I i = tx; i < m; i += NB_X)
        res += conj(A[i]) * xs[i];  // GEMV operation
}'''
        }],
        answer='The kernel achieves coalesced memory access by having threads access consecutive elements along a column: thread tx accesses A[tx], A[tx+NB_X], A[tx+2*NB_X], etc. Since A is column-major (LAPACK convention), consecutive indices along a column are consecutive in memory, ensuring coalesced reads across the warp.',
        rationale='Column-major storage means A[i] and A[i+1] are adjacent in memory when traversing a column. The stride-NB_X pattern across threads creates coalesced accesses.',
        tags=['memory-coalescing', 'gemv', 'householder']
    ))
    time.sleep(0.001)

    # L1-2: set_taubeta kernel implementation (CODING)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Implement the set_taubeta device function for real types. Given norm squared and alpha, compute tau and beta for Householder reflector H = I - tau*v*v^T.',
        context_text='The set_taubeta function computes tau and beta from the norm of vector x and the pivot element alpha. It handles the case where norm > 0.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larfg.hpp',
            'language': 'hip',
            'content': '''template <typename T, typename S>
__device__ void run_set_taubeta(T* tau, T* norms, T* alpha, S* beta)
{
    // norms[0] contains ||x||^2 (squared norm of vector x)
    // alpha[0] contains the pivot element
    // Compute: n = sqrt(||x||^2 + alpha^2) with appropriate sign
    // tau = (n - alpha) / n
    // beta = n (output), or keep alpha = n if beta is null
    // scaling = 1/(alpha - n) for normalizing x
}'''
        }],
        answer='''template <typename T, typename S>
__device__ void run_set_taubeta(T* tau, T* norms, T* alpha, S* beta)
{
    const auto ignore_beta = (beta == nullptr);
    if(norms[0] > 0)
    {
        T n = sqrt(norms[0] + alpha[0] * alpha[0]);
        n = alpha[0] >= 0 ? -n : n;  // Choose sign to avoid cancellation

        // Scaling factor for normalizing x
        norms[0] = 1.0 / (alpha[0] - n);

        // Tau coefficient
        tau[0] = (n - alpha[0]) / n;

        // Beta output
        if(ignore_beta)
            alpha[0] = n;
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
        if(!ignore_beta)
        {
            beta[0] = alpha[0];
            alpha[0] = 1;
        }
    }
}''',
        rationale='The sign of n is chosen to be opposite of alpha to avoid catastrophic cancellation in (alpha - n). The scaling factor 1/(alpha-n) normalizes x so that the first element becomes 1. When norm is zero, tau=0 means no reflection is needed.',
        tags=['coding', 'householder', 'numerical-stability', 'larfg']
    ))
    time.sleep(0.001)

    # L1-3: LDS usage in larf_left_kernel (analysis)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Calculate the LDS (Local Data Share / shared memory) requirements for larf_left_kernel. How much shared memory is needed for a 1024x1024 matrix with NB_X=1024 and warpSize=64?',
        context_text='The larf_left_kernel uses shared memory to cache the Householder vector and store partial reduction results.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larf.hpp',
            'language': 'hip',
            'content': '''template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_left_kernel(...)
{
    extern __shared__ double smem[];
    T* sdata = reinterpret_cast<T*>(smem);
    T* xs = sdata + (NB_X / warpSize);  // Offset for vector x cache

    // xs needs m elements to cache the Householder vector
    for(I i = tx; i < m; i += NB_X)
        xs[i] = x[start + i * size_t(incX)];
}

// Kernel launch:
const int lds_size = (m + (NB_X / warpSize)) * sizeof(T);'''
        }],
        answer='For m=1024, NB_X=1024, warpSize=64, double precision (sizeof(T)=8): LDS = (1024 + 1024/64) * 8 = (1024 + 16) * 8 = 8320 bytes. The formula is (m + NB_X/warpSize)*sizeof(T), where m elements store the cached Householder vector x, and NB_X/warpSize elements store per-warp reduction results.',
        rationale='The shared memory layout has two regions: (1) NB_X/warpSize elements for per-warp partial sums (sdata), and (2) m elements for caching vector x (xs). This allows reuse of x across GEMV and GER phases without repeated global memory reads.',
        tags=['lds', 'shared-memory', 'memory-optimization']
    ))
    time.sleep(0.001)

    # L1-4: larf_right_kernel GER phase (CODING)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Complete the GER (rank-1 update) phase of larf_right_kernel. After computing the dot product stored in sdata[0], apply the update A = A - tau * w * x^H to matrix A.',
        context_text='The larf_right_kernel applies H = I - tau*x*x^H from the right: A := A*H = A - A*x*tau*x^H. The scalar dot product (A*x)^T is already computed in sdata[0].',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larf.hpp',
            'language': 'hip',
            'content': '''template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void larf_right_kernel(...)
{
    // After reduction, sdata[0] contains sum_j A[row, j] * x[j]
    __syncthreads();

    // GER phase: A[row, :] += factor * conj(x[:])
    // TODO: Implement the rank-1 update
}'''
        }],
        answer='''// GER phase: Apply A[row, :] -= tau * (A[row,:]*x) * conj(x[:])
T res = -tau[0] * sdata[0];  // Compute -tau * dot_product
for(I j = tx; j < n; j += NB_X)
    A[j * size_t(lda)] += res * conj(xs[j]);''',
        rationale='The formula A := A - tau*w*x^H becomes A[row, j] += (-tau * w) * conj(x[j]), where w = sdata[0] is the precomputed dot product A[row,:]*x. Each thread updates its assigned columns j with stride NB_X. The conjugate is needed for complex types to form x^H (conjugate transpose).',
        tags=['coding', 'ger', 'rank1-update', 'householder']
    ))
    time.sleep(0.001)

    # L1-5: gebd2 loop structure (analysis)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Explain why gebd2_template uses different loop structures for m >= n versus m < n. What are the resulting bidiagonal forms?',
        context_text='The gebd2 unblocked algorithm reduces a general matrix to bidiagonal form B, where B is upper bidiagonal if m >= n and lower bidiagonal if m < n.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebd2.hpp',
            'language': 'cpp',
            'content': '''template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gebd2_template(...)
{
    rocblas_int dim = std::min(m, n);

    if(m >= n)
    {
        // generate upper bidiagonal form
        for(rocblas_int j = 0; j < n; j++)
        {
            // Generate H(j) to zero column j below diagonal
            rocsolver_larfg_template(...);  // Creates reflector for column
            // Apply H(j) from left to A(j:m, j+1:n)
            rocsolver_larf_template(handle, rocblas_side_left, ...);

            if(j < n - 1)
            {
                // Generate G(j) to zero row j to the right of superdiagonal
                rocsolver_larfg_template(...);  // Creates reflector for row
                // Apply G(j) from right to A(j+1:m, j+1:n)
                rocsolver_larf_template(handle, rocblas_side_right, ...);
            }
        }
    }
    else
    {
        // generate lower bidiagonal form
        for(rocblas_int j = 0; j < m; j++)
        {
            // Generate G(j) to zero row j to the right of diagonal
            // Generate H(j) to zero column j below subdiagonal
        }
    }
}'''
        }],
        answer='For m >= n (tall matrix): Creates UPPER bidiagonal with diagonal D and superdiagonal E. Loops over n columns, applying column reflector H(j) first to zero below diagonal, then row reflector G(j) to zero to the right of superdiagonal. For m < n (wide matrix): Creates LOWER bidiagonal with diagonal D and subdiagonal E. Loops over m rows, applying row reflector G(j) first to zero to the right of diagonal, then column reflector H(j) to zero below subdiagonal.',
        rationale='The algorithm maintains the factorization A = U*B*V^H where B is bidiagonal. For tall matrices (m>=n), we get upper bidiagonal to match the natural shape. For wide matrices (m<n), lower bidiagonal is more efficient. The order of left/right reflectors differs to build the appropriate structure.',
        tags=['bidiagonal-reduction', 'algorithm', 'gebd2']
    ))
    time.sleep(0.001)

    # ============================================================
    # L2 ENTRIES (Subsystem Scope: 2-5 kernels)
    # ============================================================

    # L2-1: larfg + larf collaboration (CODING)
    entries.append(create_entry(
        level='L2',
        interface='gebrd',
        instruction='Implement a simplified version of the larfg+larf sequence for a single column reduction. Given column vector A[j:m,j], generate the Householder reflector and apply it to the trailing matrix A[j:m, j+1:n].',
        context_text='In gebd2, each column is processed by first calling larfg to generate reflector H(j), then larf to apply it. The pattern saves diagonal elements and restores them.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebd2.hpp',
            'language': 'cpp',
            'content': '''// Single step of bidiagonal reduction (column j)
void reduce_column_j(...)
{
    // Step 1: Generate Householder reflector H(j)
    rocsolver_larfg_template(handle, m - j,
        A, shiftA + idx2D(j, j, lda),           // alpha = A(j,j)
        A, shiftA + idx2D(j + 1, j, lda),       // x = A(j+1:m, j)
        1, strideA,                             // incx = 1 (column)
        (tauq + j), strideQ, ...);

    // Step 2: Copy diagonal to D, set A(j,j) = 1
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ..., D, j, strideD,
                            A, shiftA + idx2D(j, j, lda), ...);

    // Step 3: Apply H(j) from left to trailing matrix
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1,
        A, shiftA + idx2D(j, j, lda), 1, strideA,  // reflector vector
        (tauq + j), strideQ,                       // tau coefficient
        A, shiftA + idx2D(j, j + 1, lda), ...);    // target matrix

    // Step 4: Restore diagonal from D
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), ..., D, j, ...);
}'''
        }],
        answer='''// Simplified single-column reduction for m>=n case
void reduce_column_j(handle, m, n, j, A, lda, D, E, tauq, taup)
{
    // 1. Generate Householder to zero A(j+1:m, j)
    larfg(m - j, &A[j,j], &A[j+1,j], incx=1, &tauq[j]);
    // After larfg: A(j,j) = beta, A(j+1:m,j) = v (normalized), tauq[j] = tau

    // 2. Save diagonal D[j] = A(j,j), set A(j,j) = 1
    D[j] = A[j,j];
    A[j,j] = 1.0;

    // 3. Apply from left: A(j:m, j+1:n) = (I - tau*v*v^H) * A(j:m, j+1:n)
    if(j < n - 1)
    {
        // For complex matrices, conjugate tau before left application
        if(is_complex) tauq[j] = conj(tauq[j]);

        larf(side_left, m-j, n-j-1, &A[j,j], incx=1, tauq[j], &A[j, j+1], lda);

        // Restore tau
        if(is_complex) tauq[j] = conj(tauq[j]);
    }

    // 4. Restore diagonal from saved value
    A[j,j] = D[j];
}

// This pattern repeats for each column in the reduction loop.''',
        rationale='The sequence must temporarily replace A(j,j) with 1 because larfg stores the reflector norm in that position, but larf expects the vector to have first element = 1. Saving to D[] preserves the result. Complex matrices require conjugating tau for left-side application per LAPACK convention.',
        tags=['coding', 'householder', 'larfg', 'larf', 'data-flow']
    ))
    time.sleep(0.001)

    # L2-2: labrd update kernels (analysis)
    entries.append(create_entry(
        level='L2',
        interface='gebrd',
        instruction='In labrd_template, after reducing k columns to bidiagonal form, two GEMM operations update the trailing matrix. Explain the mathematical purpose of these updates and why they use X and Y matrices.',
        context_text='labrd performs panel reduction, maintaining intermediate results X and Y to defer updates to the trailing matrix until the end.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''// In gebrd_template main loop:
while(j < dim - k)
{
    jb = std::min(dim - j, nb);
    // Reduce panel of width jb using labrd
    rocsolver_labrd_template<T>(handle, m - j, n - j, jb,
        A, ..., X, ..., Y, ...);

    // Update the trailing matrix: A(j+jb:m, j+jb:n) -= ...
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   m - j - jb, n - j - jb, jb,
                   &minone,  // alpha = -1
                   A, shiftA + idx2D(j + jb, j, lda), lda, strideA,  // A(j+jb:m, j:j+jb)
                   Y, shiftY + jb, ldy, strideY,                     // Y(jb:n, 0:jb)
                   &one,     // beta = 1
                   A, shiftA + idx2D(j + jb, j + jb, lda), ...);     // A(j+jb:m, j+jb:n)

    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                   m - j - jb, n - j - jb, jb,
                   &minone,
                   X, shiftX + jb, ldx, strideX,                     // X(jb:m, 0:jb)
                   A, shiftA + idx2D(j, j + jb, lda), lda, strideA,  // A(j:j+jb, j+jb:n)
                   &one,
                   A, shiftA + idx2D(j + jb, j + jb, lda), ...);
}'''
        }],
        answer='''The two GEMM operations apply deferred updates from panel reduction. When labrd reduces panel A(j:m, j:j+jb), it generates:
- X(j:m, 0:jb): Accumulated left transformations
- Y(j:n, 0:jb): Accumulated right transformations

The updates are:
1. A(j+jb:m, j+jb:n) -= A(j+jb:m, j:j+jb) * Y(jb:n, 0:jb)^H
   This applies the deferred right-side transformations (G reflectors)

2. A(j+jb:m, j+jb:n) -= X(jb:m, 0:jb) * A(j:j+jb, j+jb:n)
   This applies the deferred left-side transformations (H reflectors)

By deferring updates, labrd uses Level-2 BLAS (gemv) within the panel and Level-3 BLAS (gemm) for the bulk trailing matrix, maximizing performance.''',
        rationale='This is the blocked algorithm optimization: panel reduction uses fast matrix-vector ops, then applies accumulated transformations to trailing matrix via matrix-matrix ops (GEMM), which achieve much higher throughput on GPUs. The X and Y matrices capture the accumulated effect of jb Householder transformations.',
        tags=['gemm', 'blocked-algorithm', 'labrd', 'performance-optimization']
    ))
    time.sleep(0.001)

    # L2-3: Diagonal restoration kernels (CODING)
    entries.append(create_entry(
        level='L2',
        interface='gebrd',
        instruction='Implement the restore_diag kernel that copies diagonal/superdiagonal elements back to matrix A after labrd completes. Handle both m >= n (upper bidiagonal) and m < n (lower bidiagonal) cases.',
        context_text='After labrd processes a panel, the bidiagonal elements stored in D and E must be restored to A before updating the trailing matrix.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''// In gebrd_template after labrd and gemm updates:
blocks = (jb - 1) / 64 + 1;
if(m >= n)
{
    // Restore diagonal D to A(j:j+jb, j:j+jb)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>),
        dim3(batch_count, blocks, 1), dim3(1, 64, 1), 0, stream,
        D, j, strideD, A, shiftA + idx2D(j, j, lda), lda, strideA, jb);

    // Restore superdiagonal E to A(j:j+jb, j+1:j+jb+1)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>),
        dim3(batch_count, blocks, 1), dim3(1, 64, 1), 0, stream,
        E, j, strideE, A, shiftA + idx2D(j, j + 1, lda), lda, strideA, jb);
}
else
{
    // Lower bidiagonal case
    // Restore diagonal and subdiagonal
}'''
        }],
        answer='''template <typename T, typename I>
ROCSOLVER_KERNEL void restore_diag(T* D, I offset_D, rocblas_stride strideD,
                                   T* A, I lda, rocblas_stride strideA, I count)
{
    I bid = hipBlockIdx_x;  // batch index
    I idx = hipBlockIdx_y * 64 + hipThreadIdx_y;  // element index

    if(idx >= count) return;

    // Select batch
    T* d_batch = D + bid * strideD;
    T* a_batch = A + bid * strideA;

    // Copy D[offset_D + idx] back to A
    a_batch[idx * (lda + 1)] = d_batch[offset_D + idx];
    // Note: idx*(lda+1) computes diagonal element position in column-major
}

// Usage for m >= n:
// Diagonal: A(j+i, j+i) = D(j+i), i=0..jb-1
restore_diag<<<...>>>(D, j, strideD, A + idx2D(j,j,lda), lda, strideA, jb);

// Superdiagonal: A(j+i, j+i+1) = E(j+i), i=0..jb-1
restore_diag<<<...>>>(E, j, strideE, A + idx2D(j,j+1,lda), lda, strideA, jb);

// Usage for m < n:
// Diagonal: A(j+i, j+i) = D(j+i)
restore_diag<<<...>>>(D, j, strideD, A + idx2D(j,j,lda), lda, strideA, jb);

// Subdiagonal: A(j+i+1, j+i) = E(j+i)
restore_diag<<<...>>>(E, j, strideE, A + idx2D(j+1,j,lda), lda, strideA, jb);''',
        rationale='The kernel uses stride (lda+1) to traverse diagonal elements efficiently in column-major layout. For a jb x jb panel, we need (jb-1)/64+1 blocks of 64 threads. The upper vs lower bidiagonal cases differ only in the starting offset into A (j+1 row vs j+1 column for the off-diagonal).',
        tags=['coding', 'kernel-implementation', 'diagonal-restore', 'batched']
    ))
    time.sleep(0.001)

    # L2-4: Kernel fusion opportunity (analysis)
    entries.append(create_entry(
        level='L2',
        interface='gebrd',
        instruction='Analyze whether the set_diag and restore_diag kernel launches in gebd2_template could be fused. What are the benefits and challenges?',
        context_text='In gebd2, for each column j, we launch set_diag (save diagonal, set to 1), then larf, then restore_diag. The pattern repeats n times.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebd2.hpp',
            'language': 'cpp',
            'content': '''for(rocblas_int j = 0; j < n; j++)
{
    rocsolver_larfg_template(...);

    // Save A(j,j) to D, set A(j,j) = 1
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, D, j, strideD, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1, true);

    if(j < n - 1)
    {
        rocsolver_larf_template(...);  // Apply reflector
    }

    // Restore A(j,j) from D
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, D, j, strideD, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);
}'''
        }],
        answer='''Fusion challenges:
1. CANNOT fuse set_diag with restore_diag for the same column j: larf must execute between them, as it requires A(j,j)=1.
2. COULD fuse restore_diag(j) with set_diag(j+1) across iterations, BUT this requires careful stream management and doesn\'t eliminate kernel launches—just reorders them.
3. Kernel launch overhead is minimal (single element, dim3(1,1,1)) compared to larf execution.

Better optimization: The entire gebd2 loop operates on small matrices (m,n <= GEBRD_GEBD2_SWITCHSIZE, typically 32-64). Instead, fuse the ENTIRE gebd2 into a single kernel that performs all j iterations on-device, eliminating host-device synchronization. This would be a specialized small-matrix kernel.

Benefits of full fusion:
- Eliminate n kernel launch overheads
- Eliminate n host-device sync points
- Keep intermediate results in registers/LDS
- 10-50x speedup for small matrices

Challenges:
- Complex control flow (nested loops, conditionals)
- LDS requirements scale with matrix size
- Limited to small matrices (n <= 32-64)''',
        rationale='For small matrices, the dominant cost is kernel launch overhead and synchronization, not compute. A fused small-matrix kernel that keeps everything on-device would be much faster. This is why many libraries implement specialized paths for small sizes.',
        tags=['kernel-fusion', 'optimization', 'launch-overhead', 'small-matrices']
    ))
    time.sleep(0.001)

    # ============================================================
    # L3 ENTRIES (Interface Scope: Full GEBRD routine)
    # ============================================================

    # L3-1: Workspace management (CODING)
    entries.append(create_entry(
        level='L3',
        interface='gebrd',
        instruction='Implement the memory allocation logic for gebrd. The routine must allocate workspace for scalars, work arrays, and temporary X/Y matrices, handling both small (gebd2-only) and large (blocked) cases.',
        context_text='gebrd_getMemorySize computes required workspace. For small matrices (m,n <= k), only gebd2 workspace is needed. For large matrices, we need max(gebd2, labrd) workspace plus X and Y matrices.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''template <bool BATCHED, typename T>
void rocsolver_gebrd_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_X,
                                   size_t* size_Y)
{
    // Quick return: no workspace
    if(m == 0 || n == 0 || batch_count == 0) { /* ... */ }

    // Threshold for switching from gebd2 to blocked gebrd
    // Typical values: GEBRD_GEBD2_SWITCHSIZE = 32, GEBRD_BLOCKSIZE = 32

    if(m <= GEBRD_GEBD2_SWITCHSIZE || n <= GEBRD_GEBD2_SWITCHSIZE)
    {
        // Small matrix: use unblocked gebd2
        // TODO: Compute workspace for gebd2 only
    }
    else
    {
        // Large matrix: use blocked algorithm
        rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
        rocblas_int d = std::min(m / k, n / k);

        // TODO: Compute max(gebd2_workspace, labrd_workspace)
        // TODO: Compute X and Y matrix sizes
    }
}'''
        }],
        answer='''template <bool BATCHED, typename T>
void rocsolver_gebrd_getMemorySize(const rocblas_int m, const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars, size_t* size_work_workArr,
                                   size_t* size_Abyx_norms, size_t* size_X, size_t* size_Y)
{
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = *size_work_workArr = *size_Abyx_norms = 0;
        *size_X = *size_Y = 0;
        return;
    }

    if(m <= GEBRD_GEBD2_SWITCHSIZE || n <= GEBRD_GEBD2_SWITCHSIZE)
    {
        // Small: use unblocked gebd2 only
        rocsolver_gebd2_getMemorySize<BATCHED, T>(m, n, batch_count,
            size_scalars, size_work_workArr, size_Abyx_norms);
        *size_X = 0;
        *size_Y = 0;
    }
    else
    {
        // Large: blocked algorithm
        size_t s1, s2, w1, w2, unused;
        rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;  // Panel width
        rocblas_int d = std::min(m / k, n / k);   // Number of panels

        // Workspace for final gebd2 call (trailing matrix after d panels)
        rocsolver_gebd2_getMemorySize<BATCHED, T>(m - d*k, n - d*k, batch_count,
            &unused, &w1, &s1);

        // Workspace for labrd panel reduction
        rocsolver_labrd_getMemorySize<BATCHED, T>(m, n, k, batch_count,
            size_scalars, &w2, &s2);

        // Take maximum requirements
        *size_work_workArr = std::max(w1, w2);
        *size_Abyx_norms = std::max(s1, s2);

        // X matrix: m x k for each batch
        *size_X = m * k * sizeof(T) * batch_count;

        // Y matrix: n x k for each batch
        *size_Y = n * k * sizeof(T) * batch_count;
    }
}''',
        rationale='The blocked algorithm alternates between labrd (panel reduction) and gebd2 (final trailing matrix), so we need max workspace. X and Y store accumulated transformations for k columns: X is m×k (left transforms), Y is n×k (right transforms). Memory is allocated contiguously for all batches.',
        tags=['coding', 'workspace-management', 'memory-allocation', 'blocked-algorithm']
    ))
    time.sleep(0.001)

    # L3-2: Algorithm selection (analysis)
    entries.append(create_entry(
        level='L3',
        interface='gebrd',
        instruction='Compare the computational complexity and memory traffic of the blocked (gebrd) versus unblocked (gebd2) bidiagonalization algorithms. When does the blocked version become worthwhile?',
        context_text='Both algorithms perform the same mathematical factorization A = U*B*V^H, but with different computational patterns.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''// Unblocked gebd2:
for(j = 0; j < dim; j++) {
    larfg(...);          // O(m-j) work, generates reflector
    larf(side_left, ...);   // O((m-j)*(n-j)) work, uses GEMV (Level-2)
    larfg(...);          // O(n-j) work
    larf(side_right, ...);  // O((m-j)*(n-j)) work, uses GEMV (Level-2)
}
// Total: ~4*m*n^2/3 FLOPs for m=n, uses Level-2 BLAS

// Blocked gebrd:
while(j < dim - k) {
    labrd(..., jb);      // O(jb*(m-j)*(n-j)) work, uses GEMV (Level-2)
    gemm(..., jb);       // O(jb*(m-j)*(n-j)) work, uses GEMM (Level-3)
    gemm(..., jb);       // O(jb*(m-j)*(n-j)) work, uses GEMM (Level-3)
}
gebd2(...);           // Final trailing matrix
// Total: same ~4*m*n^2/3 FLOPs, but most work in Level-3 BLAS'''
        }],
        answer='''Computational Complexity:
- Both: ~(4/3)*m*n^2 FLOPs for m=n (or ~(4/3)*n*m^2 for n>m)
- Same total work, but different operation breakdown

Performance characteristics:
- gebd2: ALL work in Level-2 BLAS (GEMV) → low arithmetic intensity → memory-bound
- gebrd: ~66% work in Level-3 BLAS (GEMM) → high arithmetic intensity → compute-bound

Memory traffic:
- gebd2: Each element read/written O(n) times → O(m*n^2) memory traffic
- gebrd: Panel elements read O(k) times, bulk matrix read O(k/b) times → reduced traffic

Crossover point:
- Small (m,n < 32-64): gebd2 wins due to lower overhead
- Medium (64 < m,n < 512): gebrd wins 2-5x due to GEMM efficiency
- Large (m,n > 512): gebrd wins 5-20x as GEMM reaches peak throughput

On GPUs, GEMM achieves ~80% peak FLOPs while GEMV achieves ~5-10%, making the blocked algorithm critical for performance.''',
        rationale='The blocked algorithm achieves the same asymptotic complexity but reorganizes computation to use cache-friendly Level-3 BLAS. On modern GPUs, GEMM can be 10-30x faster per FLOP than GEMV due to data reuse in shared memory. The threshold GEBRD_GEBD2_SWITCHSIZE=32 balances overhead versus GEMM efficiency.',
        tags=['algorithm-complexity', 'level3-blas', 'performance-analysis', 'cache-blocking']
    ))
    time.sleep(0.001)

    # L3-3: Batched execution (CODING)
    entries.append(create_entry(
        level='L3',
        interface='gebrd',
        instruction='Extend gebrd_template to support batched execution with non-uniform matrix sizes. Each batch entry may have different m and n, requiring dynamic workspace allocation and kernel launches.',
        context_text='Standard gebrd assumes all batches have the same dimensions. For dynamic batching, we need per-batch size checks and workspace.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''// Standard gebrd_template assumes uniform batch:
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle,
                                        const rocblas_int m,      // Same for all batches
                                        const rocblas_int n,      // Same for all batches
                                        U A, ...)
{
    // Single path for all batches
    if(m <= k || n <= k)
        return rocsolver_gebd2_template<T>(...);

    while(j < dim - k) {
        rocsolver_labrd_template<T>(...);
        rocsolver_gemm(...);
        rocsolver_gemm(...);
    }
}

// TODO: Design batched_nonuniform version that handles different sizes'''
        }],
        answer='''// Approach 1: Per-batch dispatch (simple but inefficient)
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_batched_nonuniform(
    rocblas_handle handle,
    const rocblas_int* m_array,      // Size per batch
    const rocblas_int* n_array,      // Size per batch
    U A, const rocblas_int* lda_array,
    ..., const rocblas_int batch_count)
{
    // Loop over batches (inefficient for GPUs)
    for(rocblas_int bid = 0; bid < batch_count; bid++)
    {
        rocblas_int m = m_array[bid];
        rocblas_int n = n_array[bid];
        rocblas_int lda = lda_array[bid];

        // Process single batch
        rocsolver_gebrd_template<false, false, T>(
            handle, m, n,
            A + bid*strideA, 0, lda, 0,
            D + bid*strideD, 0,
            E + bid*strideE, 0,
            ..., 1);  // batch_count=1
    }
    return rocblas_status_success;
}

// Approach 2: Grouped batching (better for GPUs)
rocblas_status rocsolver_gebrd_grouped_batched(...)
{
    // Step 1: Group batches by size bins
    std::map<std::pair<int,int>, std::vector<int>> size_groups;
    for(int bid = 0; bid < batch_count; bid++)
        size_groups[{m_array[bid], n_array[bid]}].push_back(bid);

    // Step 2: Process each group with native batched kernels
    for(auto& [size, indices] : size_groups)
    {
        auto [m, n] = size;
        int group_count = indices.size();

        // Gather pointers for this group
        std::vector<T*> A_group(group_count);
        for(int i = 0; i < group_count; i++)
            A_group[i] = A[indices[i]];

        // Call batched gebrd for uniform-size group
        rocsolver_gebrd_template<true, false, T>(
            handle, m, n, A_group.data(), ..., group_count);
    }
}

// Approach 3: Kernel padding (best for slight variations)
// Pad all matrices to max(m) x max(n), mask out-of-bounds accesses
// Achieves full batching but wastes some compute on padding
''',
        rationale='Non-uniform batching breaks the SIMD/SIMT execution model. Three strategies: (1) Sequential per-batch processing loses parallelism, (2) Grouped batching clusters similar sizes to recover batching efficiency, (3) Padding to uniform size enables native batching but wastes work. Real implementations often use grouped batching with size bins (e.g., powers of 2) as a compromise.',
        tags=['coding', 'batched-execution', 'dynamic-shapes', 'optimization']
    ))
    time.sleep(0.001)

    # L3-4: Full interface orchestration (analysis)
    entries.append(create_entry(
        level='L3',
        interface='gebrd',
        instruction='Trace the complete execution path of rocsolver_sgebrd for a 1024x1024 matrix. List all kernels launched, their grid dimensions, and data dependencies.',
        context_text='The user calls rocsolver_sgebrd(handle, 1024, 1024, A, 1024, D, E, tauq, taup). Trace the complete call stack.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.cpp',
            'language': 'cpp',
            'content': '''rocblas_status rocsolver_sgebrd(rocblas_handle handle,
                                const rocblas_int m, const rocblas_int n,
                                float* A, const rocblas_int lda,
                                float* D, float* E, float* tauq, float* taup)
{
    return rocsolver::rocsolver_gebrd_impl<float>(handle, m, n, A, lda, D, E, tauq, taup);
}

// gebrd_impl allocates workspace and calls gebrd_template
// gebrd_template decides between gebd2 (small) and blocked algorithm (large)'''
        }],
        answer='''Execution trace for m=n=1024, GEBRD_GEBD2_SWITCHSIZE=32, GEBRD_BLOCKSIZE=32:

1. rocsolver_sgebrd (C API wrapper)
   └─> rocsolver_gebrd_impl<float>
       - Allocates workspace: X(1024×32), Y(1024×32), scalars, norms
       └─> rocsolver_gebrd_template<false, false, float>
           - dim=1024, k=32, nb=32
           - Main loop: j=0 to 992, step 32 (31 iterations)

           For each j (31 panels):
           ├─> rocsolver_labrd_template(m-j, n-j, 32)
           │   - Inner loop: 32 iterations
           │   - Each iteration:
           │     • lacgv kernels (complex only, skip for float)
           │     • rocblasCall_gemv: 4-6 calls, grid(varies)
           │     • rocsolver_larfg_template:
           │       - larfg_run_small kernel: grid(batch,1), block(64)
           │       - OR rocblasCall_dot + set_taubeta kernel
           │     • set_diag kernel: grid(batch,1,1), block(1,1,1)
           │   - Total per labrd: ~200 kernel launches
           │
           ├─> rocsolver_gemm (A -= A*Y^H, Level-3 BLAS)
           │   - grid(varies), highly optimized rocBLAS GEMM
           │
           ├─> rocsolver_gemm (A -= X*A, Level-3 BLAS)
           │
           └─> restore_diag kernels: 2x grid(batch, blocks), block(1,64)

           Final trailing matrix (992:1024, 992:1024):
           └─> rocsolver_gebd2_template(32, 32)
               - Loop j=0 to 31
               - Each iteration:
                 • rocsolver_larfg_template
                   - larfg_run_small: grid(batch,1), block(64)
                 • set_diag: grid(batch,1,1), block(1,1,1)
                 • rocsolver_larf_template
                   - larf_run_small: grid(batch, m-j or n-j), block(256)
                   - OR gemv + ger: ~3 rocBLAS calls
                 • restore_diag: grid(batch,1,1), block(1,1,1)
               - Total: ~400 kernel launches

Total kernel count: ~31*200 + 400 = ~6600 kernels
GPU time breakdown: ~70% GEMM, ~20% GEMV, ~10% small kernels
Critical path: GEMM operations (largest matrices, highest arithmetic intensity)''',
        rationale='The blocked algorithm processes 31 panels of width 32, each requiring a labrd call (many GEMV operations) followed by two GEMM updates. The final 32×32 trailing matrix uses unblocked gebd2. Most computation time is in the 31 GEMM calls, which operate on progressively smaller matrices but achieve high throughput.',
        tags=['execution-trace', 'kernel-launch', 'orchestration', 'performance-analysis']
    ))
    time.sleep(0.001)

    # L3-5: Pointer mode handling (CODING)
    entries.append(create_entry(
        level='L3',
        interface='gebrd',
        instruction='gebrd_template switches pointer mode to device for internal operations, but must preserve the user\'s original mode. Implement the pointer mode management, ensuring correct behavior even if an error occurs mid-execution.',
        context_text='rocBLAS operations can use scalars from host or device memory. The user\'s mode must be preserved across the gebrd call.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebrd.hpp',
            'language': 'cpp',
            'content': '''template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_template(...)
{
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T minone = -1;
    T one = 1;

    // TODO: Save user's pointer mode, switch to host mode
    // TODO: Ensure restoration on all exit paths

    if(m <= k || n <= k)
        return rocsolver_gebd2_template<T>(...);

    while(j < dim - k)
    {
        rocsolver_labrd_template<T>(...);  // Requires device mode internally
        rocsolver_gemm(..., &minone, ..., &one, ...);  // Uses host scalars
        rocsolver_gemm(..., &minone, ..., &one, ...);
    }

    if(j < dim)
        rocsolver_gebd2_template<T>(...);

    // TODO: Restore original pointer mode
    return rocblas_status_success;
}'''
        }],
        answer='''template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_template(...)
{
    ROCSOLVER_ENTER("gebrd", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    T minone = -1;
    T one = 1;
    rocblas_int nb = GEBRD_BLOCKSIZE;
    rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
    rocblas_int dim = std::min(m, n);

    // Early exit for small matrices (before changing pointer mode)
    if(m <= k || n <= k)
        return rocsolver_gebd2_template<T>(handle, m, n, A, shiftA, lda, strideA,
                                           D, strideD, E, strideE, tauq, strideQ,
                                           taup, strideP, batch_count, scalars,
                                           work_workArr, Abyx_norms);

    // Save user's pointer mode and switch to host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status status = rocblas_status_success;
    rocblas_int j = 0;

    // Main blocked reduction loop
    while(j < dim - k)
    {
        rocblas_int jb = std::min(dim - j, nb);

        // Panel reduction (internally switches to device mode as needed)
        status = rocsolver_labrd_template<T>(...);
        if(status != rocblas_status_success) goto cleanup;

        // Update trailing matrix with host scalars
        status = rocsolver_gemm(handle, ..., &minone, ..., &one, ...);
        if(status != rocblas_status_success) goto cleanup;

        status = rocsolver_gemm(handle, ..., &minone, ..., &one, ...);
        if(status != rocblas_status_success) goto cleanup;

        // Restore diagonal elements
        // ... kernel launches ...

        j += nb;
    }

    // Final trailing block
    if(j < dim)
    {
        status = rocsolver_gebd2_template<T>(...);
    }

cleanup:
    // ALWAYS restore user's pointer mode before returning
    rocblas_set_pointer_mode(handle, old_mode);
    return status;
}

// Note: labrd_template internally manages its own pointer mode:
// - Saves current mode
// - Switches to device mode for internal GEMV calls
// - Restores mode before returning
// This allows gebrd to use host mode for GEMM scalars while labrd uses device mode for GEMV''',
        rationale='Pointer mode must be saved before any operations that might fail, and restored on ALL exit paths (success, error, early return). Using a goto cleanup pattern ensures this. The host mode is preferred for gebrd\'s GEMM calls (scalars are compile-time constants -1 and 1), while labrd internally uses device mode for efficiency. Each function manages its own pointer mode and restores it before returning.',
        tags=['coding', 'pointer-mode', 'error-handling', 'resource-management']
    ))
    time.sleep(0.001)

    # Additional L1 entries to reach >10 total and ≥50% coding

    # L1-6: lacgv for complex matrices (CODING)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='In gebd2 for complex matrices, lacgv (conjugate a vector) is called before and after row operations. Implement a simplified lacgv kernel for a row of a matrix.',
        context_text='For complex Hermitian matrices, rows must be conjugated before applying row reflectors and unconjugated after.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebd2.hpp',
            'language': 'cpp',
            'content': '''// In gebd2 for m < n case (lower bidiagonal):
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, batch_count);

// Generate reflector for row j
rocsolver_larfg_template(...);

// Apply reflector
rocsolver_larf_template(handle, rocblas_side_right, ...);

if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, batch_count);'''
        }],
        answer='''template <typename T>
ROCSOLVER_KERNEL void lacgv_kernel(const rocblas_int n,
                                   T* A,
                                   const rocblas_int incx,
                                   const rocblas_stride stride)
{
    rocblas_int bid = hipBlockIdx_x;  // batch index
    rocblas_int tid = hipThreadIdx_x + hipBlockIdx_y * hipBlockDim_x;

    // Select batch
    T* a = A + bid * stride;

    // Conjugate elements
    if(tid < n)
    {
        rocblas_int idx = tid * incx;
        a[idx] = conj(a[idx]);  // Flip sign of imaginary part
    }
}

// Launch: For a row of length n starting at A(j,j) with stride lda:
template <typename T>
void rocsolver_lacgv_template(rocblas_handle handle,
                              const rocblas_int n,
                              T* A,
                              const rocblas_int offset,
                              const rocblas_int incx,
                              const rocblas_stride stride,
                              const rocblas_int batch_count)
{
    if(n == 0 || batch_count == 0) return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const rocblas_int threads = 256;
    const rocblas_int blocks = (n + threads - 1) / threads;

    ROCSOLVER_LAUNCH_KERNEL(lacgv_kernel<T>,
                           dim3(batch_count, blocks), dim3(threads),
                           0, stream,
                           n, A + offset, incx, stride);
}

// For row j with n-j elements, incx=lda (column stride in row-wise access)''',
        rationale='lacgv conjugates a vector by flipping the sign of imaginary parts. For complex matrices, this is needed because LAPACK uses conjugate transpose (A^H) in factorizations. Rows must be conjugated before treating them as vectors for Householder generation. The stride parameter allows batched execution.',
        tags=['coding', 'complex-arithmetic', 'lacgv', 'kernel-implementation']
    ))
    time.sleep(0.001)

    # L1-7: Small-size kernel dispatch (analysis)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Explain the small-size kernel dispatch logic in larf_template. Why are there two different thresholds: LARF_SSKER_MIN_DIM and LARF_SSKER_MAX_DIM?',
        context_text='larf_template checks matrix dimensions against thresholds to decide between small-size kernels and general GEMV+GER path.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larf.hpp',
            'language': 'cpp',
            'content': '''template <typename T, typename I, typename U>
rocblas_status rocsolver_larf_template(...)
{
    if(n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    // Small-size kernel thresholds
    bool ssker_left = (side == rocblas_side_left &&
                      m <= LARF_SSKER_MAX_DIM &&
                      n <= LARF_SSKER_MIN_DIM);
    bool ssker_right = (side == rocblas_side_right &&
                       m <= LARF_SSKER_MIN_DIM &&
                       n <= LARF_SSKER_MAX_DIM);

    if(ssker_left || ssker_right)
    {
        return larf_run_small(handle, side, m, n, ...);
    }

    // General path using GEMV + GER
    // ...
}

// Typical values: LARF_SSKER_MIN_DIM = 64, LARF_SSKER_MAX_DIM = 32'''
        }],
        answer='''The two thresholds define a rectangular region where specialized small-size kernels are more efficient:

**For left-side application** (H*A where H is m×m):
- ssker_left: m <= 32 AND n <= 64
- Rationale: The reflector dimension (m) must be small enough to fit in registers/LDS, while the panel width (n) can be larger since we process it in tiles. The small-size kernel fuses GEMV (compute w=A^H*x) and GER (A += tau*x*w^H) into a single kernel, saving memory traffic and launch overhead.

**For right-side application** (A*H where H is n×n):
- ssker_right: m <= 64 AND n <= 32
- Rationale: Symmetric to left case. The reflector dimension (n) must be small, the panel height (m) can be larger.

**Why two thresholds?**
1. **MAX_DIM (32)**: Maximum size for the Householder vector to fit efficiently in LDS. Beyond this, register/LDS pressure becomes too high.
2. **MIN_DIM (64)**: Maximum size for the "other" dimension before general GEMV+GER becomes faster due to better memory coalescing.

For example, a 32×64 matrix works well for left-side (32 reflector, 64 wide), but a 64×32 matrix would use right-side kernel (32 reflector, 64 tall).

Outside this region, the standard GEMV+GER path uses highly optimized rocBLAS routines that handle large matrices better.''',
        rationale='The asymmetric thresholds optimize for the access pattern: the reflector dimension needs to fit in fast memory, while the other dimension benefits from parallelism. Small-size kernels are 2-5x faster for small matrices by eliminating memory roundtrips and kernel launches.',
        tags=['kernel-dispatch', 'optimization', 'small-matrices', 'tuning']
    ))
    time.sleep(0.001)

    # L2-5: Complex vs Real execution paths (analysis)
    entries.append(create_entry(
        level='L2',
        interface='gebrd',
        instruction='Compare the execution flow of gebrd for real (float/double) versus complex (rocblas_float_complex/rocblas_double_complex) matrices. What additional operations are required for complex?',
        context_text='The COMPLEX template parameter controls additional conjugation and transpose operations.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_gebd2.hpp',
            'language': 'cpp',
            'content': '''template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gebd2_template(...)
{
    if(m >= n)
    {
        for(rocblas_int j = 0; j < n; j++)
        {
            rocsolver_larfg_template(...);
            ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ...);

            if(j < n - 1)
            {
                // Conjugate tau for left-side application
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                rocsolver_larf_template(handle, rocblas_side_left, ...);

                // Restore tau
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, 1, tauq, j, 1, strideQ, batch_count);

                // Conjugate row before generating row reflector
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A,
                                                shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count);

                rocsolver_larfg_template(...);  // Row reflector

                // Unconjugate row after
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - j - 1, A,
                                                shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count);
            }
        }
    }
}'''
        }],
        answer='''Additional operations for complex matrices:

**1. Tau conjugation (before/after left-side larf):**
- Real: tau is real, used as-is
- Complex: tau must be conjugated before left-side application, then restored
- Reason: LAPACK convention for H = I - tau*v*v^H requires conj(tau) for left multiplication
- Cost: ~2 kernel launches per column

**2. Row/column conjugation (before/after larfg on rows):**
- Real: rows processed as-is
- Complex: Rows conjugated before larfg, unconjugated after
- Reason: larfg expects a "real" vector in the sense of working with magnitudes
- Cost: ~4 lacgv calls per column (2 for row before/after larfg, 2 for tau)

**3. Operation type in GEMV/GER:**
- Real: rocblas_operation_transpose for A^T
- Complex: rocblas_operation_conjugate_transpose for A^H
- Cost: Handled in rocBLAS, minimal overhead

**Total overhead for complex:**
- ~6 extra kernel launches per column
- For n=1024: ~6144 extra small kernel calls
- Memory traffic: negligible (conjugation is in-place)
- Time overhead: ~10-15% for complex vs real (mostly launch overhead)

**Execution time breakdown (1024×1024):**
- Real: 100% baseline
- Complex: ~110-115% (extra kernels are very fast, dominated by main computation)

The overhead is relatively small because lacgv kernels are memory-bound with perfect coalescing, and GEMV/GER dominate total time.''',
        rationale='Complex arithmetic requires Hermitian transposes (conjugate transpose) rather than regular transposes. LAPACK convention handles this through explicit conjugation of intermediate values. The overhead is modest because the additional operations are simple element-wise kernels, while the main work (GEMV, GER, GEMM) is identical.',
        tags=['complex-arithmetic', 'hermitian', 'performance-comparison']
    ))
    time.sleep(0.001)

    # L1-8: Launch configuration for larf_left_kernel (CODING)
    entries.append(create_entry(
        level='L1',
        interface='gebrd',
        instruction='Determine the optimal grid and block dimensions for launching larf_left_kernel on a 2048×512 matrix with batch_count=8. The kernel uses NB_X=1024 threads.',
        context_text='larf_left_kernel processes one column per workgroup. Each workgroup has NB_X threads.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larf.hpp',
            'language': 'cpp',
            'content': '''// Kernel signature:
template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_left_kernel(const I m, const I n, ...)
{
    I bid = blockIdx.z;   // batch index
    I tx = threadIdx.x;   // thread index within workgroup
    I col = blockIdx.y;   // column index
    // ... Each workgroup processes column 'col' of batch 'bid'
}

// Launch site:
const int NB = 1024;
const int lds_size = (m + (NB / props.warpSize)) * sizeof(T);
if(leftside && (n <= 1024 || m >= 2048))
{
    ROCSOLVER_LAUNCH_KERNEL((larf_left_kernel<NB>),
                           dim3(?, ?, ?), dim3(?),
                           lds_size, stream, m, n, ...);
}'''
        }],
        answer='''For m=2048, n=512, batch_count=8, NB_X=1024:

**Block dimensions:** dim3(1024, 1, 1)
- threadIdx.x: 1024 threads per workgroup (specified by NB_X template parameter)
- threadIdx.y, z: unused (1)

**Grid dimensions:** dim3(1, 512, 8)
- gridDim.x: 1 (not used, could be 1)
- gridDim.y: 512 (one workgroup per column, n=512 columns)
- gridDim.z: 8 (one grid layer per batch)

**Total workgroups:** 1 × 512 × 8 = 4096 workgroups
**Total threads:** 4096 × 1024 = 4,194,304 threads

**LDS per workgroup (for float, warpSize=64):**
lds_size = (m + NB/warpSize) * sizeof(float)
         = (2048 + 1024/64) * 4
         = (2048 + 16) * 4
         = 8256 bytes

**Launch call:**
```cpp
ROCSOLVER_LAUNCH_KERNEL((larf_left_kernel<1024>),
                       dim3(1, 512, 8),  // grid: (x, cols, batches)
                       dim3(1024),        // block: 1024 threads
                       8256,              // shared memory bytes
                       stream,
                       2048, 512, ...);   // m, n, ...
```

**Occupancy considerations:**
- Threads per workgroup: 1024 (high)
- LDS per workgroup: 8256 bytes (moderate, ~8KB)
- On AMD MI250 (64KB LDS per CU): ~7 workgroups per CU (limited by threads)
- Expected occupancy: ~90% (good)''',
        rationale='The grid.y dimension maps to columns (each workgroup processes one column), grid.z maps to batches (independent batch processing). With 1024 threads per workgroup and moderate LDS usage, occupancy should be good. The tuning condition (m >= 2048) ensures we have enough parallelism within each column to saturate the 1024 threads.',
        tags=['coding', 'launch-config', 'occupancy', 'grid-dimensions']
    ))
    time.sleep(0.001)

    return entries

# Main execution
if __name__ == '__main__':
    dataset = generate_dataset()

    # Write to JSONL file
    output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_gebrd.jsonl'
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(dataset)} entries")
    print(f"Saved to {output_path}")

    # Count coding tasks
    coding_count = sum(1 for e in dataset if 'coding' in e['tags'])
    print(f"Coding tasks: {coding_count}/{len(dataset)} ({100*coding_count/len(dataset):.1f}%)")

    # Count by level
    l1 = sum(1 for e in dataset if e['level'] == 'L1')
    l2 = sum(1 for e in dataset if e['level'] == 'L2')
    l3 = sum(1 for e in dataset if e['level'] == 'L3')
    print(f"Level breakdown: L1={l1}, L2={l2}, L3={l3}")
