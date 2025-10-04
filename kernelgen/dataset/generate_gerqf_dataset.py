#!/usr/bin/env python3
import json
import time

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    return {
        "id": str(int(time.time() * 1000000)),
        "level": level,
        "interface": interface,
        "instruction": instruction,
        "context_text": context_text,
        "code_blocks": code_blocks,
        "answer": answer,
        "rationale": rationale,
        "tags": tags
    }

entries = []

# L1-1: GERQ2 backward iteration for RQ factorization
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='In the GERQ2 unblocked RQ factorization, why is the iteration variable j used to index row (m-j-1) and column (n-j-1)? Explain the backward processing pattern.',
    context_text='GERQ2 performs unblocked RQ factorization processing rows from bottom to top (backward direction). Understanding the indexing is critical for correct Householder reflector application.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerq2.hpp',
        'language': 'hip',
        'content': '''rocblas_int dim = std::min(m, n); // total number of pivots

for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(m - j - 1, 0, lda),
                                    strideA, batch_count);

    // generate Householder reflector to work on row m - j - 1
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);'''
    }],
    answer='The loop iterates j from 0 to dim-1, but processes rows in reverse order: row (m-j-1) starts at the last row (m-1) when j=0 and moves upward. Column (n-j-1) similarly starts at the rightmost column. This backward pattern is essential for RQ factorization where R occupies the bottom-right corner and transformations must propagate upward.',
    rationale='RQ factorization stores R in the lower-right triangle, unlike QR which uses upper-left. The backward iteration ensures each Householder reflector is generated for the bottommost unprocessed row, annihilating elements to its left. The indexing (m-j-1, n-j-1) correctly maps the forward loop counter j to backward matrix positions.',
    tags=['indexing', 'backward-direction', 'householder', 'rq-factorization']
))

# L1-2: Conjugation pattern for complex RQ factorization (coding task)
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Write a device kernel to conjugate a row vector of length n stored with stride lda in a batched matrix. The kernel should handle complex types and be callable before/after LARFG for RQ factorization.',
    context_text='GERQ2 conjugates rows before and after Householder reflector generation because LARFG expects column-like data (unit stride), but RQ operates on rows (stride lda).',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerq2.hpp',
        'language': 'hip',
        'content': '''// conjugate the jth row of A
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(m - j - 1, 0, lda),
                                lda, strideA, batch_count);

// generate Householder reflector to work on row m - j - 1
rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), ...);

// restore the jth row of A
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(m - j - 1, 0, lda),
                                lda, strideA, batch_count);'''
    }],
    answer='''Implementation:
```cpp
template <typename T>
__global__ void conjugate_row_kernel(T* A, rocblas_int offset, rocblas_int n,
                                      rocblas_int lda, rocblas_stride strideA) {
    rocblas_int bid = hipBlockIdx_x;
    rocblas_int tid = hipThreadIdx_x;
    T* row = A + bid * strideA + offset;

    for(rocblas_int i = tid; i < n; i += hipBlockDim_x) {
        row[i * lda] = conj(row[i * lda]);
    }
}
```
Launch: `conjugate_row_kernel<<<batch_count, 256>>>(A, shiftA + idx2D(m-j-1, 0, lda), n-j, lda, strideA);`
Conjugation is needed before LARFG because it expects column data semantics, and after to restore the row.''',
    rationale='The kernel maps each block to a batch and uses threads to parallelize across the row elements. Stride lda is used because elements are in a row (non-unit stride). Conjugation is a self-inverse operation, so applying it twice restores the original values after LARFG completes.',
    tags=['coding', 'complex-arithmetic', 'conjugation', 'batched-execution', 'row-wise']
))

# L1-3: Diagonal save/restore mechanism
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Why does GERQ2 save the diagonal element A(m-j-1, n-j-1) to a temporary array before calling LARF, then restore it afterward? What would happen without this step?',
    context_text='The set_diag and restore_diag kernels are invoked around LARF application in GERQ2 to temporarily modify the diagonal element.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerq2.hpp',
        'language': 'hip',
        'content': '''// insert one in A(j,j) to build/apply the householder matrix
ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                        0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                        lda, strideA, 1, true);

// Apply Householder reflector to the rest of matrix from the right
if(j < m - 1)
{
    rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                            shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                            (ipiv + dim - j - 1), strideP, A, shiftA, lda, strideA,
                            batch_count, scalars, Abyx_norms, (T**)work_workArr);
}

// restore original value of A(j,j)
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                        dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                        shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);'''
    }],
    answer='LARFG stores the Householder reflector magnitude in the diagonal position A(m-j-1, n-j-1), but LARF needs the implicit 1 there to correctly apply H = I - tau*v*v^H where v has unit 1 in the pivot position. Without saving/restoring, the computed R factor magnitude would be overwritten. The temporary swap ensures LARF sees v[pivot]=1 while preserving the R diagonal.',
    rationale='Householder reflectors have the form H = I - tau*v*v^H where v is normalized with v[0]=1. LARFG computes tau and the rest of v, storing |v| at the pivot location for space efficiency. LARF requires v[pivot]=1 for correct matrix multiplication, so the diagonal is temporarily set to 1, then restored to preserve the R factor.',
    tags=['householder', 'diagonal-handling', 'numerical-stability', 'larf-requirements']
))

# L1-4: LARFG pivot storage in ipiv array (coding task)
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Write code to correctly index the ipiv array for storing tau values in GERQ2. Given dim=min(m,n) and loop variable j (0 to dim-1), where should tau from processing row (m-j-1) be stored?',
    context_text='GERQ2 stores Householder scalars (tau) in ipiv array. The storage order must align with how LARFT and LARFB expect to retrieve these values for blocked updates.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerq2.hpp',
        'language': 'hip',
        'content': '''rocblas_int dim = std::min(m, n);

for(rocblas_int j = 0; j < dim; ++j)
{
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);'''
    }],
    answer='''Storage location:
```cpp
// Store tau for row (m-j-1) at position (dim-j-1) in ipiv
T* tau_location = ipiv + (dim - j - 1) * strideP;
```
For j=0 (bottom row m-1): tau stored at ipiv[dim-1]
For j=dim-1 (top row m-dim): tau stored at ipiv[0]

This reverses the processing order in storage, placing the last-processed tau at the beginning of ipiv, which LARFT expects for backward direction block reflectors.''',
    rationale='The expression (ipiv + dim - j - 1) ensures taus are stored in reverse order relative to row processing. Since GERQ2 processes rows bottom-to-top (m-1 down to m-dim), but LARFT for backward direction expects taus in the order they were applied (from the perspective of the block), this indexing correctly maps physical row order to logical tau order.',
    tags=['coding', 'indexing', 'householder', 'tau-storage', 'backward-direction']
))

# L1-5: LARF right-side application dimensions
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='When GERQ2 calls LARF from the right side, it passes m-j-1 rows and n-j columns. Verify these dimensions are correct for j=1 in a 5×8 matrix (m=5, n=8, dim=5).',
    context_text='LARF applies a Householder reflector from the right to the remaining unprocessed portion of the matrix. Dimension errors cause incorrect updates or out-of-bounds access.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerq2.hpp',
        'language': 'hip',
        'content': '''// Apply Householder reflector to the rest of matrix from the right
if(j < m - 1)
{
    rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                            shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                            (ipiv + dim - j - 1), strideP, A, shiftA, lda, strideA,
                            batch_count, scalars, Abyx_norms, (T**)work_workArr);
}'''
    }],
    answer='''For j=1 with m=5, n=8, dim=5:
- m - j - 1 = 5 - 1 - 1 = 3 rows (rows 0-2, above row 3 being processed)
- n - j = 8 - 1 = 7 columns (columns 0-6, left of column 7 being processed)

LARF updates the 3×7 submatrix in the top-left by applying the reflector computed from row 3 (m-j-1=3). The reflector vector spans 7 elements (columns 0-6 of row 3). This is correct: we update rows above the current row, using all columns to the left of the pivot.''',
    rationale='LARF right-side application multiplies A[0:m-j-1, 0:n-j] by H from the right. Row (m-j-1) contains the reflector vector and is not updated (hence m-j-1 rows, not m-j). The n-j columns include all positions where the reflector has non-zero elements. For j=1, row 3 (second from bottom) was just processed, so rows 0-2 are updated.',
    tags=['dimensions', 'larf-application', 'right-side', 'matrix-geometry']
))

# L2-1: GERQF blocked algorithm loop structure
entries.append(create_entry(
    level='L2',
    interface='gerqf',
    instruction='Analyze the GERQF blocked loop: why does it initialize j=k-kk+ki and decrement j-=nb, while GERQ2 uses incrementing j? How does this relate to backward panel processing?',
    context_text='GERQF uses a blocked algorithm with GERQ2 for panels, LARFT for block reflectors, and LARFB for updates. The loop structure must correctly partition the matrix into blocks.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''rocblas_int k = std::min(m, n); // total number of pivots
rocblas_int nb = GExQF_BLOCKSIZE;
rocblas_int ki = ((k - GExQF_GExQ2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;

while(j >= k - kk)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(k - j, nb);
    rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);

    if(m - k + j > 0)
    {
        rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_row_wise, ...);
        rocsolver_larfb_template<...>(...);
    }
    j -= nb;
}'''
    }],
    answer='GERQF uses j to track the column index of the rightmost column in the current block being processed. Starting at j=k-kk+ki positions j near the right edge of the factorization region, and j-=nb moves leftward (backward) through blocks. In contrast, GERQ2 increments j but maps to (m-j-1) for backward row processing. GERQF operates on blocks of rows, so the loop variable directly represents column position, decremented to process right-to-left.',
    rationale='The initialization j=k-kk+ki computes the starting column for blocked processing after accounting for the final unblocked region (kk). The while condition j>=k-kk ensures all blocks down to the final panel are processed. Decrementing j by nb (block size) implements backward iteration at the block level, consistent with RQ factorization requiring bottom-right to top-left processing.',
    tags=['blocked-algorithm', 'loop-structure', 'backward-direction', 'panel-factorization']
))

# L2-2: LARFT backward direction for RQ (coding task)
entries.append(create_entry(
    level='L2',
    interface='gerqf',
    instruction='Implement the logic to select rocblas_backward_direction vs rocblas_forward_direction when calling LARFT in RQ factorization. What parameter determines this choice?',
    context_text='LARFT builds the triangular T factor for a block reflector. The direction parameter affects how elementary reflectors are combined: T = I - V^H * T * V for forward, different update pattern for backward.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''// compute block reflector
rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_row_wise,
                            n - k + j + jb, jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                            strideW, batch_count, scalars, (T*)work_workArr, workArr);'''
    }],
    answer='''Implementation:
```cpp
// For RQ factorization (row-wise, backward):
rocblas_direct direction = rocblas_backward_direction;

// Rule: RQ and QL use backward_direction (process from bottom/right)
//       QR and LQ use forward_direction (process from top/left)

rocsolver_larft_template<T>(handle, direction, rocblas_row_wise,
                            n - k + j + jb, jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, T_factor, ldw, strideW,
                            batch_count, scalars, work, workArr);
```
RQ always uses backward_direction because reflectors are generated from bottom row upward.''',
    rationale='rocblas_backward_direction tells LARFT that tau values in ipiv correspond to reflectors applied in backward order (last row first). This affects how T is built: T[i,j] involves combining reflectors where indices decrease. For RQ, row m-1 was processed first, so its tau is at the end logically, requiring backward accumulation in T.',
    tags=['coding', 'larft', 'block-reflector', 'backward-direction', 'triangular-factor']
))

# L2-3: GERQ2 panel row dimension calculation (coding task)
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Write a function to calculate the number of rows in the GERQ2 panel for iteration j. Given m, k, j, jb, verify the formula for m=512, k=256, j=192, jb=64.',
    context_text='GERQF processes panels of jb rows at the bottom of the matrix. The row dimension for GERQ2 must cover these rows while respecting the total factorization size.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''jb = std::min(k - j, nb);
rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                            work_workArr, Abyx_norms_trfact, diag_tmptr);'''
    }],
    answer='''Function:
```cpp
rocblas_int gerq2_panel_rows(rocblas_int m, rocblas_int k, rocblas_int j, rocblas_int jb) {
    return jb;
}

rocblas_int gerq2_panel_cols(rocblas_int n, rocblas_int k, rocblas_int j, rocblas_int jb) {
    return n - k + j + jb;
}
```
For m=512, k=256, j=192, jb=64:
- Rows: jb = 64
- Cols: n - k + j + jb = n - 256 + 192 + 64 = n
  (If n=512: 512 - 256 + 192 + 64 = 512 columns)

The panel is jb rows tall starting at row (m-k+j), spanning all columns from 0 to (n-k+j+jb-1).''',
    rationale='GERQ2 is called on a jb×(n-k+j+jb) submatrix. The row count is simply jb (block size). The column count n-k+j+jb ensures all columns from the leftmost edge to the rightmost column of the block (column n-k+j+jb-1) are included. For j=192, this spans columns 0 to 511, covering where the jb=64 rows need updating.',
    tags=['coding', 'panel-dimensions', 'indexing', 'matrix-geometry']
))

# L2-4: LARFB parameter derivation from GERQF context
entries.append(create_entry(
    level='L2',
    interface='gerqf',
    instruction='In GERQF, LARFB is called with side=right, trans=none, direct=backward, storev=row_wise. Explain why trans=none is used for RQ when QR uses trans=transpose for LARFB.',
    context_text='LARFB applies a block reflector to a matrix. The transpose parameter affects whether H or H^H is applied. The choice depends on the factorization algorithm.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''// apply the block reflector
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_right, rocblas_operation_none, rocblas_backward_direction,
    rocblas_row_wise, m - k + j, n - k + j + jb, jb, A,
    shiftA + idx2D(m - k + j, 0, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);'''
    }],
    answer='trans=none means apply H (not H^H) to the matrix from the right: C := C * H. In RQ, H is built from row reflectors stored in the bottom rows. Applying H from the right updates the rows above, transforming them toward the desired R form. QR uses trans=transpose because in column-wise storage, H^H from the left achieves the analogous effect. The difference stems from row-wise (RQ) vs column-wise (QR) reflector storage, not algorithmic intent.',
    rationale='LARFB parameters depend on reflector orientation. RQ stores reflectors in rows (storev=row_wise), so H is naturally row-oriented. Applying C*H from the right (trans=none) propagates the transformation upward through rows. QR stores reflectors in columns, requiring H^H from the left. Both achieve the same mathematical goal (triangular factorization) but with different geometric layouts.',
    tags=['larfb', 'block-reflector', 'transpose-parameter', 'row-wise', 'rq-vs-qr']
))

# L3-1: GERQF workspace allocation strategy
entries.append(create_entry(
    level='L3',
    interface='gerqf',
    instruction='Analyze GERQF workspace allocation. Why is size_Abyx_norms_trfact allocated as max(GERQ2_requirement, T_matrix_size)? What reuse pattern enables this?',
    context_text='GERQF allocates multiple workspace buffers: scalars, work_workArr, Abyx_norms_trfact, diag_tmptr, workArr. Understanding reuse is critical for memory efficiency.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''rocblas_int jb = GExQF_BLOCKSIZE;

// size to store the temporary triangular factor
*size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

// requirements for calling GERQ2 with sub blocks
rocsolver_gerq2_getMemorySize<BATCHED, T>(jb, n, batch_count, size_scalars, &w1, &s2, &s1);
*size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);'''
    }],
    answer='Abyx_norms_trfact serves dual purposes: (1) temporary norms/workspace for GERQ2/LARFG (size s2), (2) storage for the jb×jb triangular T factor used by LARFT/LARFB. These uses are non-overlapping in time: GERQ2 completes before LARFT writes T to the same buffer. Taking max(s2, jb*jb) ensures sufficient space for both. After LARFT, LARFB reads T from this buffer while GERQ2 no longer needs it.',
    rationale='Sequential execution allows workspace reuse. GERQ2 uses Abyx_norms_trfact for temporary norms during Householder generation, then LARFT overwrites it with the T matrix for block reflector application. Since GERQ2 is complete before LARFT starts, no conflict occurs. The max allocation accommodates the larger of the two requirements across all blocks.',
    tags=['workspace-allocation', 'memory-reuse', 'optimization', 'buffer-management']
))

# L3-2: GERQF algorithmic complexity (coding task)
entries.append(create_entry(
    level='L3',
    interface='gerqf',
    instruction='Write code to compute the theoretical FLOP count for GERQF on an m×n matrix with k=min(m,n) pivots. Consider blocked (Level-3 BLAS) vs unblocked (Level-2 BLAS) paths.',
    context_text='GERQF switches between blocked and unblocked algorithms based on matrix size. The blocked version uses GEMM (O(n³)) via LARFB, while unblocked uses GEMV (O(n²)) via LARF.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''// if the matrix is small, use the unblocked (BLAS-levelII) variant
if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    return rocsolver_gerq2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                       diag_tmptr);

rocblas_int k = std::min(m, n);
rocblas_int nb = GExQF_BLOCKSIZE;
// ... blocked algorithm with LARFB'''
    }],
    answer='''FLOP count estimation:
```cpp
double gerqf_flops(rocblas_int m, rocblas_int n, rocblas_int nb) {
    rocblas_int k = std::min(m, n);

    // Unblocked: ~2*k²*(3n-k) FLOPs from successive LARF calls
    if(m <= 64 || n <= 64) {
        return 2.0 * k * k * (3.0 * n - k);
    }

    // Blocked: ~2*k²*n from LARFB (GEMM-based) plus GERQ2 panel costs
    // Dominant term: LARFB does ~2*m*n*k work across all blocks
    double larfb_flops = 2.0 * k * n * (m - k/2.0);
    double panel_flops = 2.0 * k * k * n / nb; // Simplified panel cost
    return larfb_flops + panel_flops;
}
```
Blocked path is ~3-5× faster due to Level-3 BLAS efficiency.''',
    rationale='Unblocked GERQ2 performs k LARF operations, each on decreasing matrix sizes, yielding O(k²n) complexity. Blocked GERQF reduces this by batching updates via LARFB (matrix-matrix multiplies), achieving better cache reuse. The theoretical FLOP count is similar, but GEMM achieves much higher throughput than GEMV on GPUs, making the blocked path faster despite similar operation counts.',
    tags=['coding', 'complexity-analysis', 'flop-count', 'blocked-vs-unblocked', 'performance']
))

# L3-3: GERQF interface data flow
entries.append(create_entry(
    level='L3',
    interface='gerqf',
    instruction='Trace the data flow for a single batch element through GERQF: input matrix A -> GERQ2 panel factorization -> LARFT block reflector -> LARFB update -> final R and Q representation. Where is Q stored?',
    context_text='GERQF computes RQ factorization: A = R*Q where R is lower trapezoidal and Q is orthogonal. Understanding where factors are stored is essential for downstream use.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, batch_count, ...);

rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_row_wise,
                            n - k + j + jb, jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw, ...);

rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_right, rocblas_operation_none, rocblas_backward_direction,
    rocblas_row_wise, m - k + j, n - k + j + jb, jb, A,
    shiftA + idx2D(m - k + j, 0, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);'''
    }],
    answer='''Data flow:
1. **Input**: A (m×n) contains the original matrix
2. **GERQ2 panel**: Factorizes bottom jb rows, stores:
   - R factor in bottom-right of A (lower trapezoidal)
   - Householder vectors in same rows (below diagonal), overwriting A
   - Tau scalars in ipiv[j:j+jb]
3. **LARFT**: Reads reflectors from A and ipiv, computes T matrix in Abyx_norms_trfact
4. **LARFB**: Applies H=I-V*T*V^H to rows above, updates A in-place
5. **Output**:
   - R stored in lower m×k triangle of A
   - Q implicitly stored as Householder reflectors in A (below R diagonal) + ipiv (tau values)

Q is NOT explicitly formed; use ORMRQ/UNMRQ to apply Q or generate it.''',
    rationale='LAPACK-style factorizations store factors compactly: R occupies the lower triangle (its natural position for RQ), while Q is stored implicitly via the Householder representation (elementary reflectors + scaling factors). This compact storage (m*n + k) is more memory-efficient than explicit Q (m*m). The ipiv array provides the tau values needed to reconstruct Q when required.',
    tags=['data-flow', 'factorization-storage', 'implicit-q', 'compact-representation']
))

# L3-4: GERQF batched execution and striding
entries.append(create_entry(
    level='L3',
    interface='gerqf',
    instruction='How does GERQF support batched execution? Explain the difference between BATCHED=true (array of pointers) and STRIDED (strideA) modes, and when workArr size is doubled.',
    context_text='GERQF template has BATCHED and STRIDED parameters controlling how multiple matrices are processed. Workspace requirements change based on these modes.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gerqf_template(rocblas_handle handle,
                                        const rocblas_int m, const rocblas_int n, U A,
                                        const rocblas_int shiftA, const rocblas_int lda,
                                        const rocblas_stride strideA, T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count, ...);

// size of workArr is double to accommodate LARFB's TRMM calls in the batched case
if(BATCHED)
    *size_workArr *= 2;'''
    }],
    answer='''Batched execution modes:
1. **STRIDED mode** (BATCHED=false, STRIDED=true):
   - A is a single pointer, matrices accessed via A + i*strideA
   - Efficient for evenly-spaced matrices in contiguous memory
   - workArr size: 1× (single buffer reused across batches)

2. **BATCHED mode** (BATCHED=true):
   - A is T** (array of pointers to matrices)
   - Supports irregular layouts (matrices at arbitrary addresses)
   - workArr size: 2× because LARFB's TRMM needs per-batch temporary storage

3. **Non-batched** (batch_count=1): Standard mode, no batching

The doubled workArr in BATCHED mode accommodates LARFB calling TRMM, which needs workspace per batch instance for intermediate products when A is pointer-array based.''',
    rationale='GPU batched BLAS operations have two modes: strided (regular offset arithmetic) and array-of-pointers (irregular). GERQF propagates this to all sub-routines. LARFB internally uses TRMM for the block update C := C - V*T*V^H*C, and TRMM in batched mode requires per-batch workspace when pointers are irregular, hence the 2× multiplier. Strided mode can reuse a single workspace across batches since memory is predictable.',
    tags=['batched-execution', 'strided-batched', 'workspace-scaling', 'memory-layout']
))

# L1-6: Row offset calculation for GERQ2 panel (coding task)
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Write a function to calculate the starting row index for the GERQ2 panel in GERQF iteration j. Given m=1024, k=512, j=256, compute the row index and verify it points to the correct panel position.',
    context_text='GERQF calls GERQ2 on bottom panels of the matrix. The row offset must correctly position the panel for factorization.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                            work_workArr, Abyx_norms_trfact, diag_tmptr);'''
    }],
    answer='''Function:
```cpp
rocblas_int gerq2_panel_row_start(rocblas_int m, rocblas_int k, rocblas_int j) {
    return m - k + j;
}
```
For m=1024, k=512, j=256:
- Row index: m - k + j = 1024 - 512 + 256 = 768

This is row 768, which is the starting row of a panel in the bottom half of the matrix (rows 768 to 768+jb-1). Since k=512 pivots are needed and j=256 marks progress from the right, the panel is correctly positioned 256 rows above the bottom (m-k=512 is the topmost row needing factorization).''',
    rationale='The expression m-k+j computes where the current block starts. m-k is the first row needing factorization (row 512 for k=512), and adding j shifts down by the number of pivots already processed from the right. For j=256, we have processed 256 pivots, so the next panel starts at row 512+256=768, which is jb rows from where the next set of pivots will be generated.',
    tags=['coding', 'indexing', 'panel-offset', 'row-calculation']
))

# L2-5: LARFB update region calculation (coding task)
entries.append(create_entry(
    level='L2',
    interface='gerqf',
    instruction='Implement a function to calculate the LARFB update region dimensions in GERQF. Given m, k, j, jb, what are the rows and columns of the submatrix C that LARFB updates?',
    context_text='LARFB updates the portion of A above the current panel. The dimensions must exclude the panel rows but include all relevant columns.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''// apply the block reflector
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_right, rocblas_operation_none, rocblas_backward_direction,
    rocblas_row_wise, m - k + j, n - k + j + jb, jb, A,
    shiftA + idx2D(m - k + j, 0, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);'''
    }],
    answer='''Implementation:
```cpp
struct LarfbRegion {
    rocblas_int rows;
    rocblas_int cols;
    rocblas_int c_row_offset;
    rocblas_int c_col_offset;
};

LarfbRegion compute_larfb_region(rocblas_int m, rocblas_int k, rocblas_int j, rocblas_int jb) {
    LarfbRegion region;
    region.rows = m - k + j;        // Rows above the panel
    region.cols = n - k + j + jb;   // All columns from left to panel right edge
    region.c_row_offset = 0;        // C starts at row 0
    region.c_col_offset = 0;        // C starts at column 0
    return region;
}
```
For m=512, k=256, j=128, jb=64:
- C rows: 512 - 256 + 128 = 384 (rows 0-383)
- C cols: n - 256 + 128 + 64 = n - 64 (if n=512: columns 0-447)

LARFB applies H from right to this 384×448 region, using the 64-row panel as the reflector source.''',
    rationale='The update region C has (m-k+j) rows because these are all rows above the panel starting at row (m-k+j). The column count (n-k+j+jb) includes all columns from 0 to the rightmost column of the panel, since the reflector affects all these positions. The panel itself is not in C; it provides V for computing C := C * H.',
    tags=['coding', 'larfb-region', 'update-dimensions', 'matrix-partitioning']
))

# L1-8: Loop initialization ki and kk computation (coding task)
entries.append(create_entry(
    level='L1',
    interface='gerqf',
    instruction='Write a function to calculate ki and kk values for GERQF initialization. Given k=512, nb=64, switchsize=64, compute ki and kk. What do these values represent?',
    context_text='GERQF computes ki and kk to determine the starting point for blocked processing and the size of the final unblocked region.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''rocblas_int k = std::min(m, n);
rocblas_int nb = GExQF_BLOCKSIZE;
rocblas_int ki = ((k - GExQF_GExQ2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;'''
    }],
    answer='''Implementation:
```cpp
struct LoopBounds {
    rocblas_int ki;
    rocblas_int kk;
    rocblas_int j_init;
};

LoopBounds compute_gerqf_bounds(rocblas_int k, rocblas_int nb, rocblas_int switchsize) {
    LoopBounds bounds;
    bounds.ki = ((k - switchsize - 1) / nb) * nb;
    bounds.kk = std::min(k, bounds.ki + nb);
    bounds.j_init = k - bounds.kk + bounds.ki;
    return bounds;
}
```
For k=512, nb=64, switchsize=64:
- ki = ((512 - 64 - 1) / 64) * 64 = (447 / 64) * 64 = 6 * 64 = 384
- kk = min(512, 384 + 64) = min(512, 448) = 448
- j_init = 512 - 448 + 384 = 448

**ki**: Last blocked iteration starting column (384 means last block starts at col 384)
**kk**: Number of pivots processed by blocking (448 pivots via blocked algorithm)
**j_init**: Initial value of j for the loop (448 is the rightmost processed column)''',
    rationale='ki rounds down k-switchsize-1 to nearest multiple of nb, marking where blocked processing can safely end before entering unblocked region. kk=min(k, ki+nb) ensures one more block is processed if space permits. The final switchsize pivots are left for unblocked GERQ2 to handle. j_init=k-kk+ki positions the loop to start factoring from the right, processing kk pivots in blocks.',
    tags=['coding', 'loop-initialization', 'algorithm-partitioning', 'blocked-unblocked-transition']
))

# L2-6: Block size jb computation (coding task)
entries.append(create_entry(
    level='L2',
    interface='gerqf',
    instruction='Write a function to compute the actual block size jb for GERQF iteration j. Given k=512, j=480, nb=64, what is jb? Explain edge case handling.',
    context_text='Each GERQF iteration processes a block of jb rows. The block size varies near the end of the factorization to handle remaining pivots correctly.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''while(j >= k - kk)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(k - j, nb);
    rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);'''
    }],
    answer='''Implementation:
```cpp
rocblas_int compute_block_size(rocblas_int k, rocblas_int j, rocblas_int nb) {
    return std::min(k - j, nb);
}
```
For k=512, j=480, nb=64:
- jb = min(512 - 480, 64) = min(32, 64) = 32

This is an edge case: only 32 pivots remain (from j=480 to k=511), so jb=32 instead of the full nb=64. This ensures we don't exceed the factorization region.

For j=256, nb=64:
- jb = min(512 - 256, 64) = min(256, 64) = 64 (full block)

The min() handles the final partial block gracefully, adapting the block size to remaining work.''',
    rationale='The expression min(k-j, nb) ensures jb never exceeds the number of remaining pivots (k-j). When j is close to k (near loop termination), k-j is small, yielding a partial block. When j is far from k, k-j>nb, so jb=nb (full block). This dynamic sizing is essential for correctness at factorization boundaries.',
    tags=['coding', 'block-size', 'edge-case-handling', 'dynamic-sizing']
))

# L3-5: Final panel factorization logic
entries.append(create_entry(
    level='L3',
    interface='gerqf',
    instruction='After the blocked loop completes, GERQF calls GERQ2 on the final panel. Explain why this final call uses dimensions (mu, nu) instead of (jb, n-k+j+jb), and what happens when mu=0.',
    context_text='The blocked algorithm leaves a trailing panel that must be factored separately. Understanding this final step is critical for correctness.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_gerqf.hpp',
        'language': 'hip',
        'content': '''while(j >= k - kk)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(k - j, nb);
    rocsolver_gerq2_template<T>(...);

    if(m - k + j > 0)
    {
        rocsolver_larft_template<T>(...);
        rocsolver_larfb_template<...>(...);
    }
    j -= nb;
    mu = m - k + j + jb;
    nu = n - k + j + jb;
}

// factor last block
if(mu > 0 && nu > 0)
    rocsolver_gerq2_template<T>(handle, mu, nu, A, shiftA, lda, strideA, ipiv, strideP,
                                batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                diag_tmptr);'''
    }],
    answer='''The final GERQ2 call factors the remaining top-left region that wasn't processed in blocked form:

- **mu = m - k + j + jb**: rows from top (0) to the last processed panel's top edge
- **nu = n - k + j + jb**: columns from left (0) to the last processed panel's right edge

After the loop, j < k-kk means all major blocks are done, but rows [0, mu-1] and columns [0, nu-1] still need factorization. Using (mu, nu) ensures this residual region gets processed.

When mu=0: All m rows have been factored (m-k+j+jb=0 means no rows remain), so no final call is needed. The check `if(mu > 0 && nu > 0)` prevents invalid zero-dimension calls.''',
    rationale='The blocked loop processes k-kk pivots in blocks of nb, leaving kk pivots for the final unblocked call. The variables mu and nu track the unfactored region size. When j drops below k-kk, the loop exits, and mu/nu hold the residual dimensions. If mu=0, all rows were consumed by blocked processing, typically when m≤k and m is a multiple of nb.',
    tags=['final-panel', 'residual-factorization', 'edge-cases', 'algorithm-termination']
))

print(f"Generated {len(entries)} entries")

# Write to JSONL
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_gerqf.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Saved to {output_path}")

# Statistics
coding_count = sum(1 for e in entries if 'coding' in e['tags'])
level_count = {}
for e in entries:
    level_count[e['level']] = level_count.get(e['level'], 0) + 1

print(f"Coding tasks: {coding_count}/{len(entries)} ({100*coding_count/len(entries):.1f}%)")
print(f"Level breakdown: {level_count}")
