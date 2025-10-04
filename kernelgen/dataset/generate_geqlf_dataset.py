#!/usr/bin/env python3
"""
Generate SFT dataset for roclapack_geqlf.yaml
GEQLF: QL factorization with blocked algorithm (A = Q*L)
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

# L1-1: GEQL2 backward column iteration - coding task
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='In GEQL2, why does the loop iterate backwards from j=0 to dim-1, processing columns n-j-1? Write the index calculation for accessing the j-th Householder reflector in ipiv.',
    context_text='GEQL2 computes QL factorization by processing columns from right to left, unlike QR which goes left to right. Each iteration generates a Householder reflector for column n-j-1.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geql2.hpp',
        'language': 'hip',
        'content': '''rocblas_int dim = std::min(m, n); // total number of pivots

for(rocblas_int j = 0; j < dim; j++)
{
    // generate Householder reflector to work on column n - j - 1
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the left
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                            shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                            strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                            (T**)work_workArr);
}'''
    }],
    answer='''The index for j-th iteration (j=0..dim-1) is: **ipiv + dim - j - 1**

For j=0 (first iteration, rightmost column n-1):
  ipiv[dim-1] stores tau for column n-1

For j=dim-1 (last iteration, column n-dim):
  ipiv[0] stores tau for column n-dim

The backward indexing ensures ipiv[k] corresponds to column n-dim+k, maintaining consistency with QR factorization conventions where ipiv[k] relates to pivot k.''',
    rationale='QL factorization processes columns right-to-left to zero out subdiagonal elements progressively. The ipiv indexing (dim-j-1) maps loop iteration j to the actual column being processed (n-j-1), storing tau values in reverse order.',
    tags=['coding', 'ql-factorization', 'indexing', 'householder-reflectors']
))

# L1-2: GEQLF blocked algorithm structure
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='Explain the loop bounds in GEQLF\'s blocked algorithm. Given k=min(m,n)=256, nb=64, calculate ki, kk, and the initial value of j. How many block iterations will execute?',
    context_text='GEQLF uses blocked algorithm to shift work to Level-3 BLAS. It processes panels from right to left, leaving a final unblocked panel.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''rocblas_int k = std::min(m, n); // total number of pivots
rocblas_int nb = GEQxF_BLOCKSIZE;
rocblas_int ki = ((k - GEQxF_GEQx2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;

while(j >= k - kk)
{
    jb = std::min(k - j, nb);
    rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);
    // ... larft and larfb calls ...
    j -= nb;
}'''
    }],
    answer='''Given k=256, nb=64, GEQxF_GEQx2_SWITCHSIZE=64:

ki = ((256 - 64 - 1) / 64) * 64 = (191 / 64) * 64 = 2 * 64 = 128
kk = min(256, 128 + 64) = min(256, 192) = 192
j_initial = k - kk + ki = 256 - 192 + 128 = 192

Loop condition: j >= k - kk = 64
Iterations: j=192, 128, 64 (stops before j=0)
**3 block iterations**

Each iteration processes nb=64 columns. After the loop, the final 64 columns (mu=64, nu=64) are factorized with GEQL2.''',
    rationale='ki determines the last block boundary, kk is the size of the blocked portion, and j tracks the current panel position. The algorithm processes 3 blocks of 64 columns each (192-128, 128-64, 64-0), then the final 64×64 block with GEQL2.',
    tags=['coding', 'blocked-algorithm', 'loop-bounds', 'panel-factorization']
))

# L1-3: larft backward direction for QL
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='Why does GEQLF call larft with rocblas_backward_direction? How does this affect the computation of the triangular factor T compared to GEQRF?',
    context_text='LARFT computes the triangular factor T such that the block reflector H = I - V*T*V\'. The direction parameter controls how columns are processed.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''// compute block reflector
rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                            m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                            strideW, batch_count, scalars, (T*)work_workArr, workArr);'''
    }],
    answer='''rocblas_backward_direction causes larft to process reflectors in reverse order (right-to-left), matching QL factorization\'s column order.

For jb=4 columns with tau stored in ipiv[j:j+jb]:
- **GEQRF (forward)**: Processes tau[j], tau[j+1], tau[j+2], tau[j+3] → T is lower triangular
- **GEQLF (backward)**: Processes tau[j+3], tau[j+2], tau[j+1], tau[j] → T is upper triangular

The backward direction ensures H = H_{jb-1} * ... * H_1 * H_0 where H_i uses column i from right. This matches the QL convention where rightmost reflectors are applied first.

T matrix structure: Upper triangular for QL (backward) vs. lower triangular for QR (forward).''',
    rationale='The direction parameter controls the order of reflector composition. QL factorization applies reflectors from right to left, so backward direction ensures the block reflector accumulates in the correct order.',
    tags=['block-reflector', 'backward-direction', 'ql-vs-qr', 'triangular-factor']
))

# L1-4: Column offset calculation in GEQLF - coding task
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='For a 512×256 matrix in GEQLF with k=256, j=128, jb=64, calculate the exact column index and shift offset for the GEQL2 call. Show the idx2D calculation.',
    context_text='GEQLF processes panels at varying positions. The shiftA parameter points to the top-left of the current panel.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);'''
    }],
    answer='''Given m=512, n=256, k=256, j=128, jb=64, lda=512:

Column index: n - k + j = 256 - 256 + 128 = 128
Panel dimensions: (m - k + j + jb) × jb = (512 - 256 + 128 + 64) × 64 = 448 × 64
Panel starts at: A[0, 128]

shiftA calculation:
idx2D(0, n - k + j, lda) = 0 + (256 - 256 + 128) * 512 = 128 * 512 = 65536

The panel covers:
- Rows: 0 to 447 (448 rows)
- Columns: 128 to 191 (64 columns)
- ipiv storage: ipiv[128:191] stores tau values''',
    rationale='The expression (n - k + j) computes the column offset for the current panel. For j=128 in a 256-column matrix with k=256, we process columns 128-191. The row dimension (m - k + j + jb) ensures we include all rows above the diagonal of the rightmost column.',
    tags=['coding', 'indexing', 'panel-offset', 'memory-layout']
))

# L1-5: GEQL2 diagonal save/restore - coding task
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='Implement the set_diag kernel call that saves A(m-j-1, n-j-1) to the diag array and replaces it with 1 for Householder reflector application.',
    context_text='GEQL2 temporarily sets the diagonal element to 1 to build the Householder matrix H = I - tau*v*v\', then restores the original value.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geql2.hpp',
        'language': 'hip',
        'content': '''// generate Householder reflector to work on column n - j - 1
rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                         shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                         strideP, batch_count, (T*)work_workArr, Abyx_norms);

// insert one in A(m-j-1,n-j-1) to build/apply the householder matrix
ROCSOLVER_LAUNCH_KERNEL(/* TODO: complete this call */);

// Apply Householder reflector
rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                        shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                        strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                        (T**)work_workArr);

// restore original value
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                        dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                        shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);'''
    }],
    answer='''ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                        0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                        lda, strideA, 1, true);

Parameters:
- diag: Array to save original diagonal values
- diag offset: 0
- stride: 1 (one element per batch)
- A, shiftA + idx2D(m - j - 1, n - j - 1, lda): Matrix location
- lda, strideA: Matrix strides
- count: 1 (single element)
- save_restore: true (save and set to 1)''',
    rationale='set_diag saves A(m-j-1, n-j-1) to diag[batch_id] and writes 1.0 to that location. This allows larf to treat the reflector as v = [x1, x2, ..., xn-1, 1]^T. After larf, restore_diag recovers the original value from the diag array.',
    tags=['coding', 'diagonal-handling', 'householder-reflector', 'temporary-modification']
))

# L1-6: LARFB operation parameters for QL
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='Analyze the LARFB call in GEQLF. With m=512, n=256, k=256, j=128, jb=64, what matrix dimensions are passed to LARFB, and why is rocblas_operation_conjugate_transpose used?',
    context_text='LARFB applies the block reflector H to the remaining matrix. For QL factorization, H is applied from the left with conjugate transpose.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''// apply the block reflector
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_backward_direction, rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
    shiftA + idx2D(0, n - k + j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);'''
    }],
    answer='''Given m=512, n=256, k=256, j=128, jb=64:

LARFB dimensions:
- m_larfb = m - k + j + jb = 512 - 256 + 128 + 64 = 448 rows
- n_larfb = n - k + j = 256 - 256 + 128 = 128 columns
- jb = 64 reflectors

Matrix regions:
- V (reflectors): A[0:447, 128:191], 448×64 panel
- T (triangular): Abyx_norms_trfact, 64×64 upper triangular
- C (target): A[0:447, 0:127], 448×128 left submatrix

Operation: C := (I - V*T*V') * C = H' * C

rocblas_operation_conjugate_transpose means apply H' instead of H. For QL factorization, we compute Q*L where Q = H_1 * H_2 * ... * H_k. Applying H' from the left updates C := H'*C, which corresponds to the Q' operation in the factorization.''',
    rationale='LARFB updates the left portion of the matrix (columns 0:127) using the block reflector from the current panel (columns 128:191). Conjugate transpose is used because QL factorization requires applying Q^H during the panel updates.',
    tags=['larfb', 'block-reflector', 'matrix-dimensions', 'conjugate-transpose']
))

# L2-1: GEQL2 vs GEQLF algorithm comparison
entries.append(create_entry(
    level='L2',
    interface='geqlf',
    instruction='Compare the computational complexity of GEQL2 vs GEQLF for a 1024×1024 matrix. Estimate the ratio of Level-2 BLAS to Level-3 BLAS operations in GEQLF with nb=64.',
    context_text='GEQL2 uses unblocked algorithm (Level-2 BLAS), while GEQLF uses blocked algorithm (Level-3 BLAS dominant).',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geql2.hpp',
        'language': 'hip',
        'content': '''for(rocblas_int j = 0; j < dim; j++)
{
    rocsolver_larfg_template(...);  // O(m-j)
    rocsolver_larf_template(..., m - j, n - j - 1, ...);  // GEMV: O((m-j)*(n-j-1))
}'''
    }, {
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''while(j >= k - kk)
{
    jb = std::min(k - j, nb);
    rocsolver_geql2_template(..., m - k + j + jb, jb, ...);  // O(m*jb^2)
    rocsolver_larft_template(...);  // O(m*jb^2)
    rocsolver_larfb_template(..., m - k + j + jb, n - k + j, jb, ...);  // GEMM: O(m*n*jb)
    j -= nb;
}'''
    }],
    answer='''For n=m=1024:

**GEQL2**: O(2*m^2*n/3) ≈ 2*1024^3/3 ≈ 716M flops
- All Level-2 BLAS (GEMV in larf)
- Arithmetic intensity: ~1 flop/byte (memory-bound)

**GEQLF** with nb=64, ~15 block iterations:
- GEQL2 panels: 15 * O(1024*64^2) ≈ 63M flops (Level-2)
- LARFT: 15 * O(1024*64^2) ≈ 63M flops (Level-2)
- LARFB: sum of O(m*n_i*64) ≈ 15 * O(1024*640*64) ≈ 630M flops (Level-3)
- Total: ~756M flops

**Level-2 vs Level-3 ratio**: (63+63) / 630 ≈ 0.2
~83% of work is Level-3 BLAS (GEMM in larfb), achieving 7-8× higher arithmetic intensity and 5-10× speedup on GPUs.''',
    rationale='Blocked algorithm shifts ~80% of work to GEMM, which has much higher arithmetic intensity (O(n) flops per matrix element) than GEMV (O(1) flops per element). This dramatically improves GPU utilization and memory bandwidth efficiency.',
    tags=['algorithm-comparison', 'complexity-analysis', 'blocked-vs-unblocked', 'performance']
))

# L2-2: GEQL2 + LARFT + LARFB collaboration - coding task
entries.append(create_entry(
    level='L2',
    interface='geqlf',
    instruction='Trace the data flow between GEQL2, LARFT, and LARFB for one block iteration. What data is produced by each routine, and how is it consumed by the next?',
    context_text='Each GEQLF block iteration factorizes a panel with GEQL2, computes the block reflector with LARFT, then applies it with LARFB.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''jb = std::min(k - j, nb);
rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                            work_workArr, Abyx_norms_trfact, diag_tmptr);

rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                            m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                            strideW, batch_count, scalars, (T*)work_workArr, workArr);

rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_backward_direction, rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
    shiftA + idx2D(0, n - k + j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);'''
    }],
    answer='''Data flow for one block iteration:

**GEQL2 produces**:
1. V (reflectors): A[0:m-k+j+jb-1, n-k+j:n-k+j+jb-1] stores Householder vectors below diagonal
2. L (factor): Lower triangular factor stored in same panel
3. tau: ipiv[j:j+jb-1] stores scaling factors

**LARFT consumes** (from GEQL2):
- V: Reads A[0:m-k+j+jb-1, n-k+j:n-k+j+jb-1]
- tau: Reads ipiv[j:j+jb-1]

**LARFT produces**:
- T: Abyx_norms_trfact[0:jb-1, 0:jb-1], upper triangular jb×jb matrix
  Satisfies: H = I - V*T*V' where H = H_{jb-1} * ... * H_0

**LARFB consumes** (from GEQL2 + LARFT):
- V: Reads A[0:m-k+j+jb-1, n-k+j:n-k+j+jb-1] (reflectors)
- T: Reads Abyx_norms_trfact (block reflector)
- C: A[0:m-k+j+jb-1, 0:n-k+j-1] (target matrix)

**LARFB produces**:
- C := H' * C: Updates A[0:m-k+j+jb-1, 0:n-k+j-1] in-place

The V and tau from GEQL2 are reused by LARFT to build T, then V and T are used by LARFB to update the rest of the matrix.''',
    rationale='The three-stage pipeline maximizes data reuse: GEQL2 creates V and tau, LARFT compresses them into T for efficient GEMM-based updates, and LARFB applies the block reflector using Level-3 BLAS. This is the core of blocked QL factorization.',
    tags=['coding', 'data-flow', 'subsystem-collaboration', 'blocked-algorithm']
))

# L2-3: Workspace allocation strategy
entries.append(create_entry(
    level='L2',
    interface='geqlf',
    instruction='Explain the workspace allocation strategy in rocsolver_geqlf_getMemorySize. Why is size_workArr doubled for batched execution, and what is stored in size_Abyx_norms_trfact?',
    context_text='GEQLF allocates multiple workspace arrays for different purposes. The batched case has special requirements.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''// size to store the temporary triangular factor
*size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

// requirements for calling GEQL2 with sub blocks
rocsolver_geql2_getMemorySize<BATCHED, T>(m, jb, batch_count, size_scalars, &w1, &s2, &s1);
*size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

// requirements for calling LARFT
rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);

// requirements for calling LARFB
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                          &s2, &unused);

*size_work_workArr = std::max(w1, w2);
*size_diag_tmptr = std::max(s1, s2);

// size of workArr is double to accommodate
// LARFB's TRMM calls in the batched case
if(BATCHED)
    *size_workArr *= 2;'''
    }],
    answer='''Workspace purposes:

1. **size_Abyx_norms_trfact**: Stores the triangular factor T (jb × jb × batch_count). Maximum of:
   - T matrix: jb × jb × batch_count for LARFT output
   - GEQL2 requirements (norms/workspace)
   Reused across iterations.

2. **size_work_workArr**: Max of GEQL2 and LARFT workspace needs. Stores:
   - Temporary arrays for LARF/LARFG in GEQL2
   - Work arrays for LARFT computation
   Reused across calls.

3. **size_diag_tmptr**: Max of GEQL2 diag array and LARFB workspace.

4. **size_workArr**: Array of pointers for batched LARFT. **Doubled for batched** because:
   - LARFB calls TRMM internally, which needs pointer arrays
   - LARFT also needs pointer arrays
   - Both may be needed simultaneously in batched path
   - Doubling ensures no conflicts between concurrent needs

Non-batched doesn't use pointer arrays, so no doubling needed.''',
    rationale='The allocation strategy maximizes workspace reuse while ensuring sufficient memory for all call paths. Batched execution uses array-of-pointers for rocBLAS calls, requiring separate storage for nested calls. The max() operations ensure the workspace can handle the largest requirement across all iterations.',
    tags=['workspace-allocation', 'memory-management', 'batched-execution', 'resource-reuse']
))

# L2-4: Backward direction synchronization
entries.append(create_entry(
    level='L2',
    interface='geqlf',
    instruction='Why does GEQL2 conjugate tau before and after calling LARF for complex types? What would break if only one conjugation were performed?',
    context_text='GEQL2 has a conjugate-apply-restore pattern around LARF for complex Hermitian matrices.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geql2.hpp',
        'language': 'hip',
        'content': '''// conjugate tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

// Apply Householder reflector to the rest of matrix from the left
rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                        shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                        strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                        (T**)work_workArr);

// restore tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);'''
    }],
    answer='''The conjugation pattern ensures correct Hermitian reflector application:

**Why conjugate before LARF?**
LARFG produces tau such that H = I - tau*v*v' (with conjugate). LARF expects tau to be used directly in the update C := (I - tau*v*v')*C. For QL factorization from the left, we need H = I - conj(tau)*conj(v)*v^T to maintain Hermitian symmetry when processing columns right-to-left.

**Why restore after LARF?**
The ipiv array must maintain the original tau values for later use by LARFT and for API consistency. If not restored:
1. LARFT would read conj(tau) instead of tau, producing incorrect T matrix
2. User-facing API would return conjugated tau values, breaking LAPACK compatibility
3. Subsequent iterations would compound conjugation errors

**If only one conjugation:**
- Only before: ipiv permanently conjugated → LARFT fails, API broken
- Only after: LARF gets wrong tau → incorrect reflector application

The pattern ensures LARF gets conj(tau) while preserving tau in storage.''',
    rationale='This is a subtle numerical correctness requirement for complex Hermitian QL factorization. The conjugation swap allows LARF to use the correct reflector while maintaining invariants for LARFT and the user API.',
    tags=['complex-arithmetic', 'conjugation', 'numerical-correctness', 'hermitian-matrices']
))

# L3-1: GEQLF interface design and optimization
entries.append(create_entry(
    level='L3',
    interface='geqlf',
    instruction='Describe the full GEQLF execution flow for a 1024×1024 matrix. How does the interface decide between GEQL2 and blocked algorithm, and what determines the number of block iterations vs the final unblocked panel?',
    context_text='GEQLF provides QL factorization with automatic algorithm selection based on matrix size.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                       diag_tmptr);

rocblas_int k = std::min(m, n); // total number of pivots
rocblas_int nb = GEQxF_BLOCKSIZE;
rocblas_int ki = ((k - GEQxF_GEQx2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;

while(j >= k - kk)
{
    jb = std::min(k - j, nb);
    rocsolver_geql2_template<T>(...);  // panel
    rocsolver_larft_template<T>(...);   // block reflector
    rocsolver_larfb_template<...>(...); // apply to left
    j -= nb;
}

// factor last block
if(mu > 0 && nu > 0)
    rocsolver_geql2_template<T>(handle, mu, nu, A, shiftA, lda, strideA, ipiv, strideP,
                                batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                diag_tmptr);'''
    }],
    answer='''Execution flow for 1024×1024 (assuming GEQxF_GEQx2_SWITCHSIZE=64, nb=64):

**1. Algorithm selection:**
- m=1024, n=1024 > GEQxF_GEQx2_SWITCHSIZE → Use blocked GEQLF

**2. Initialization:**
- k = min(1024, 1024) = 1024
- ki = ((1024 - 64 - 1) / 64) * 64 = 14 * 64 = 896
- kk = min(1024, 896 + 64) = 960
- j_initial = 1024 - 960 + 896 = 960

**3. Blocked iterations:** j = 960, 896, ..., 64 (15 iterations, j >= 64)
Each iteration:
- GEQL2: Factorize (m-k+j+jb) × jb panel at columns [n-k+j : n-k+j+jb-1]
- LARFT: Compute block reflector T (jb × jb)
- LARFB: Apply H' to left submatrix (columns 0 to n-k+j-1)

**4. Final unblocked panel:**
- mu = m - k + j + jb = 1024 - 1024 + 0 + 64 = 64
- nu = n - k + j + jb = 64
- GEQL2: Factorize remaining 64×64 block at A[0:63, 0:63]

**Why this partitioning?**
- Blocked portion (960 columns): Use Level-3 BLAS for 90%+ of flops
- Unblocked tail (64 columns): Too small for blocking overhead to pay off
- Threshold GEQxF_GEQx2_SWITCHSIZE balances blocking benefit vs overhead''',
    rationale='The hybrid algorithm uses blocked factorization for large panels (GEMM-dominated) and unblocked for small tails (lower overhead). The partitioning ensures maximum Level-3 BLAS usage while avoiding excessive blocking overhead on small problems.',
    tags=['interface-design', 'algorithm-selection', 'execution-flow', 'hybrid-algorithm']
))

# L3-2: QL vs QR factorization comparison - coding task
entries.append(create_entry(
    level='L3',
    interface='geqlf',
    instruction='Compare GEQLF and GEQRF interfaces. How do their column iteration orders differ, and what impact does this have on matrix access patterns and cache behavior?',
    context_text='QL and QR factorizations differ in the order of column processing. This affects memory access patterns.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''// GEQLF: right-to-left iteration
rocblas_int j = k - kk + ki;
while(j >= k - kk)
{
    rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, ...);
    rocsolver_larfb_template<...>(handle, ..., m - k + j + jb, n - k + j, jb, ...);
    j -= nb;
}'''
    }],
    answer='''Key differences:

**Iteration order:**
- GEQRF: Left-to-right (j=0, nb, 2*nb, ..., k)
  - Columns: [0:nb-1], [nb:2*nb-1], ..., [k-nb:k-1]
- GEQLF: Right-to-left (j=k-nb, k-2*nb, ..., 0)
  - Columns: [n-nb:n-1], [n-2*nb:n-nb-1], ..., [0:nb-1]

**Matrix access patterns:**

GEQRF (left-to-right):
- Panel j at A[:, j:j+nb] → LARFB updates A[:, j+nb:n]
- Reads panel, writes right submatrix
- Cache: Good spatial locality when updating contiguous right columns

GEQLF (right-to-left):
- Panel j at A[:, n-k+j:n-k+j+nb] → LARFB updates A[:, 0:n-k+j]
- Reads panel, writes left submatrix
- Cache: Potentially worse locality as left submatrix may be evicted

**Performance impact:**
For column-major storage:
- GEQRF: Sequential write pattern to right → better cache reuse
- GEQLF: Updates scattered across left columns → more cache misses

However, both use GEMM (LARFB) for updates, so Level-3 cache blocking mitigates this. Performance difference is typically <5-10% on modern GPUs with large caches.''',
    rationale='The backward iteration in GEQLF is algorithmically required (QL factorization processes columns right-to-left). While this creates less favorable access patterns than QR, the Level-3 BLAS dominance and GPU cache hierarchy minimize the impact.',
    tags=['coding', 'ql-vs-qr', 'memory-access-patterns', 'cache-behavior']
))

# L1-7: Panel row dimension calculation - coding task
entries.append(create_entry(
    level='L1',
    interface='geqlf',
    instruction='Write a function to calculate the exact number of rows in the GEQL2 panel for iteration j. Given m, n, k, j, jb, verify the calculation for m=512, n=256, k=256, j=192, jb=64.',
    context_text='GEQLF processes panels of varying height. The row dimension ensures coverage of all elements above the diagonal of the rightmost column.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''jb = std::min(k - j, nb);
rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                            work_workArr, Abyx_norms_trfact, diag_tmptr);'''
    }],
    answer='''Function:
```cpp
rocblas_int panel_rows(rocblas_int m, rocblas_int k, rocblas_int j, rocblas_int jb) {
    return m - k + j + jb;
}
```

Verification for m=512, n=256, k=256, j=192, jb=64:
panel_rows = 512 - 256 + 192 + 64 = 512 rows

Explanation:
- m - k = 512 - 256 = 256: excess rows beyond square k×k portion
- j + jb = 192 + 64 = 256: columns being processed are [256-256+192 : 256-256+192+64-1] = [192:255]
- Total: 256 + 256 = 512 rows (full matrix height)

For general j:
- Columns processed: [n-k+j : n-k+j+jb-1]
- Rightmost column: n-k+j+jb-1
- Rows needed: All rows from 0 to (rightmost_col index) = m - k + j + jb
- This ensures the panel includes all elements above diagonal in column n-k+j+jb-1''',
    rationale='The row dimension m-k+j+jb ensures the panel spans from row 0 down to the diagonal of the rightmost column. For QL factorization, we need to zero subdiagonal elements, so the panel must include the full column down to its diagonal element.',
    tags=['coding', 'panel-dimensions', 'indexing', 'matrix-geometry']
))

# L3-3: Batched GEQLF execution
entries.append(create_entry(
    level='L3',
    interface='geqlf',
    instruction='For batched GEQLF with batch_count=256 and m=n=512, analyze the GPU parallelism. How are batches distributed across compute units, and what is the benefit of batching vs 256 individual GEQLF calls?',
    context_text='Batched execution processes multiple independent matrices simultaneously using strided pointers.',
    code_blocks=[{
        'path': 'library/src/lapack/roclapack_geqlf.hpp',
        'language': 'hip',
        'content': '''template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geqlf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        ...)'''
    }],
    answer='''Batched execution model (batch_count=256, m=n=512):

**Parallelism distribution:**
1. **GEQL2 small kernels** (larfg_run_small, larf_run_small):
   - Each batch instance mapped to separate thread blocks
   - Grid: (blocks_per_matrix, batch_count)
   - 256 batches × multiple blocks = high occupancy

2. **LARFT**:
   - Per-batch computation, batch_count instances in parallel
   - Each batch: ~64×64 triangular factor computation

3. **LARFB (GEMM)**:
   - rocBLAS batched GEMM: 256 × (512×512×64 operations)
   - Distributes across all GPU compute units (CUs)
   - Each CU processes tiles from multiple batches

**Batching benefits vs 256 sequential calls:**

1. **Kernel launch overhead**:
   - Batched: ~15 kernel launches total (one per blocked iteration)
   - Sequential: 256 × 15 = 3840 launches
   - Savings: ~3825 × 5μs = ~19ms launch overhead eliminated

2. **GPU utilization**:
   - Batched: All CUs busy simultaneously across batches
   - Sequential: Single matrix may underutilize GPU (especially for small m,n)
   - For m=n=512, batching achieves ~4-8× better throughput

3. **Memory allocation**:
   - Batched: Single workspace allocation for all batches
   - Sequential: 256 allocations (overhead + fragmentation)

4. **Stream efficiency**:
   - Batched: Single stream with tightly packed kernels
   - Sequential: Requires multiple streams for any parallelism

**Expected speedup**: 5-10× for batch_count=256 with m=n=512 matrices.''',
    rationale='Batching amortizes launch overhead, maximizes GPU occupancy, and enables efficient resource sharing. The strided pointer model allows rocBLAS to optimize data movement and computation across all batches simultaneously.',
    tags=['batched-execution', 'gpu-parallelism', 'performance-optimization', 'throughput']
))

# Write entries to JSONL
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_geqlf.jsonl'
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
