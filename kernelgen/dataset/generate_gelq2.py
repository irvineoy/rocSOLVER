#!/usr/bin/env python3
"""
Generator for GELQ2 dataset - LQ factorization (unblocked)

GELQ2 computes an LQ factorization of an m-by-n matrix A using Householder reflectors.
The factorization has the form A = L * Q where L is lower trapezoidal and Q is orthogonal.
Processes rows from top to bottom, applying reflectors from the RIGHT (unlike QR from left).
"""

import json
import time
from pathlib import Path

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Helper to create a dataset entry"""
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

# L1-1: CODING - Row conjugation for complex LQ
entries.append(create_entry(
    level="L1",
    interface="gelq2",
    instruction="Implement the row conjugation logic required before and after Householder reflector generation in GELQ2. Why must we conjugate row j before calling LARFG and unconjugate it afterward, and what stride pattern is used?",
    context_text="GELQ2 works on rows (not columns). For complex matrices, each row must be conjugated before LARFG, then unconjugated after the reflector is applied. This is different from column-based QR.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);

    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                             (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(j,j) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                            1, true);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }

    // restore original value of A(j,j)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);

    // restore the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);
}"""
        }
    ],
    answer="""Here's the complete row conjugation pattern for complex GELQ2:

```cpp
template <typename T, bool COMPLEX = rocblas_is_complex<T>>
void process_lq_row_with_conjugation(rocblas_handle handle,
                                     rocblas_int j, rocblas_int m, rocblas_int n,
                                     U A, rocblas_int shiftA, rocblas_int lda, rocblas_stride strideA,
                                     T* ipiv, rocblas_stride strideP,
                                     rocblas_int batch_count,
                                     T* scalars, void* work_workArr, T* Abyx_norms, T* diag)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Step 1: Conjugate row j: A(j, j:n-1) = conj(A(j, j:n-1))
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            n - j,                           // number of elements: n-j
            A,                               // matrix
            shiftA + idx2D(j, j, lda),      // start: A(j,j)
            lda,                             // increment: lda (row stride in col-major)
            strideA,                         // batch stride
            batch_count);
    }

    // Step 2: Generate Householder reflector on conjugated row
    // Input:  conj(A(j, j:n-1)) = [conj(a_jj), conj(a_j,j+1), ..., conj(a_j,n-1)]
    // Output: v stored in A(j, j+1:n-1), tau in ipiv[j], L diagonal in A(j,j)
    rocsolver_larfg_template(
        handle,
        n - j,                              // vector length
        A, shiftA + idx2D(j, j, lda),      // alpha: A(j,j)
        A, shiftA + idx2D(j, std::min(j + 1, n - 1), lda),  // x: A(j, j+1:n-1)
        lda,                                // incx: lda (row elements)
        strideA,                            // stridex
        (ipiv + j), strideP,               // tau output
        batch_count, work_workArr, Abyx_norms);

    // Step 3: Save diagonal and replace with 1.0 for LARF
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>),
                            dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            diag, 0, 1,                    // save to diag[0]
                            A, shiftA + idx2D(j, j, lda),  // A(j,j)
                            lda, strideA, 1, true);

    // Step 4: Apply H from the right: A(j+1:m-1, j:n-1) = A(j+1:m-1, j:n-1) * H
    if(j < m - 1)
    {
        rocsolver_larf_template(
            handle,
            rocblas_side_right,              // Apply from right: A*H
            m - j - 1,                       // number of rows to update
            n - j,                           // reflector size
            A, shiftA + idx2D(j, j, lda),   // Householder vector (row j)
            lda, strideA,                    // incx=lda (row stride)
            (ipiv + j), strideP,            // tau
            A, shiftA + idx2D(j + 1, j, lda),  // submatrix to update
            lda, strideA,
            batch_count, scalars, Abyx_norms, (T**)work_workArr);
    }

    // Step 5: Restore diagonal value
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>),
                            dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            diag, 0, 1,
                            A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);

    // Step 6: Unconjugate row j: A(j, j:n-1) = conj(A(j, j:n-1))
    // Restores original representation of Householder vector
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            n - j,                           // number of elements
            A,
            shiftA + idx2D(j, j, lda),      // start: A(j,j)
            lda,                             // increment: lda
            strideA,
            batch_count);
    }
}
```

**Why Row Conjugation is Necessary:**

1. **Householder reflector for rows**:
   - Column-based (QR): H = I - tau * v * v^H, v is a column
   - Row-based (LQ): H = I - tau * v^H * v, v is a row
   - For LQ with row v, we need v^H in the formula

2. **LARFG expects column vectors**:
   - LARFG is designed for column vectors: generates v such that H*x = beta*e_1
   - For row operations, we feed conj(row) as a column to LARFG
   - LARFG then computes tau and v correctly for the conjugated data

3. **Stride pattern**:
   - Row elements in column-major: A(j, j), A(j, j+1), ..., A(j, n-1)
   - Memory addresses: &A[j + j*lda], &A[j + (j+1)*lda], ..., &A[j + (n-1)*lda]
   - Stride: lda (not 1!) because we're traversing a row in column-major layout

4. **Double conjugation restores**:
   - After LARFG and LARF, row is still conjugated
   - Second LACGV restores: conj(conj(x)) = x
   - Stored Householder vector is in original (unconjugated) form

**Without Conjugation:**
- Reflector would have wrong phase for complex matrices
- LQ factorization would be incorrect: A ≠ L*Q
- Orthogonality test fails: Q*Q^H ≠ I""",
    rationale="GELQ2 processes rows, but LARFG expects column vectors. For complex matrices, we conjugate the row before LARFG (effectively transposing to a column), generate the reflector, apply it, then unconjugate to restore the original form. The stride is lda (row elements in column-major storage). This ensures correct complex Householder reflectors for row-based LQ factorization.",
    tags=["coding", "complex-arithmetic", "row-operations", "conjugation", "lq-factorization", "householder-reflectors", "stride-patterns"]
))

# L1-2: ANALYSIS - Row-based vs column-based factorization
entries.append(create_entry(
    level="L1",
    interface="gelq2",
    instruction="Explain why GELQ2 applies Householder reflectors from the RIGHT (rocblas_side_right) while GEQR2 applies from the LEFT. How does this relate to the L vs R factor structure?",
    context_text="GELQ2 processes rows top-to-bottom and applies reflectors from the right. This creates an LQ factorization where A = L*Q.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                             (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }
}"""
        }
    ],
    answer="""GELQ2's right-side application creates the LQ factorization structure:

**Left vs Right Application:**

**GEQR2 (QR factorization):**
- Processes: Columns (left to right)
- Reflector: H_j zeros column j below diagonal
- Application: H_j * A (multiply from LEFT)
- Result: A = Q * R where Q = H_0 * H_1 * ... * H_{n-1}
- Factor R (upper triangular) in top-left corner

**GELQ2 (LQ factorization):**
- Processes: Rows (top to bottom)
- Reflector: H_j zeros row j to the right of diagonal
- Application: A * H_j (multiply from RIGHT)
- Result: A = L * Q where Q = H_0 * H_1 * ... * H_{m-1}
- Factor L (lower triangular) in top-left corner

**Mathematical Reasoning:**

For row j, we want to zero out A(j, j+1:n-1):
```
Row j: [ a_jj  a_j,j+1  a_j,j+2  ...  a_j,n-1 ]
Goal:  [ l_jj    0        0      ...     0    ]
```

Using Householder H_j that acts on columns j to n-1:
```
H_j = I - tau_j * v_j * v_j^H

Where v_j is based on row j: v_j = [a_j,j:n-1]^T
```

To transform row j, we compute:
```
A(j:m-1, j:n-1) * H_j = A(j:m-1, j:n-1) * (I - tau_j * v_j * v_j^H)
                      = A(j:m-1, j:n-1) - tau_j * (A(j:m-1, j:n-1) * v_j) * v_j^H
```

This is a RIGHT-side operation because H_j post-multiplies A.

**Effect on Structure (m=5, n=4 example):**

Iteration j=0 (row 0):
```
Before: [ *  *  *  * ]    After: [ L₀  0  0  0 ]
        [ *  *  *  * ]           [ *   *  *  * ]
        [ *  *  *  * ]    H₀ →   [ *   *  *  * ]
        [ *  *  *  * ]           [ *   *  *  * ]
        [ *  *  *  * ]           [ *   *  *  * ]

H₀ zeros A(0, 1:3) by acting on columns 0:3
```

Iteration j=1 (row 1):
```
Before: [ L₀  0  0  0 ]    After: [ L₀  0   0   0  ]
        [ *   *  *  * ]           [ L₁  L₁  0   0  ]
        [ *   *  *  * ]    H₁ →   [ *   *   *   *  ]
        [ *   *  *  * ]           [ *   *   *   *  ]
        [ *   *  *  * ]           [ *   *   *   *  ]

H₁ zeros A(1, 2:3) by acting on columns 1:3
Preserves column 0 (already factored)
```

**Why RIGHT, Not LEFT?**

If we tried LEFT application (H_j * A):
- Would operate on rows, not columns
- Would destroy L factor built in previous iterations
- Would create upper triangular R, not lower triangular L

**Summary:**
| Aspect | GEQR2 (left) | GELQ2 (right) |
|--------|--------------|---------------|
| Direction | H*A | A*H |
| Zeros | Below diagonal (columns) | Right of diagonal (rows) |
| Order | Q = H₀*H₁*...*Hₙ | Q = H₀*H₁*...*Hₘ |
| Factor | R (upper) | L (lower) |
| Storage | Columns below diag | Rows right of diag |""",
    rationale="GELQ2 applies reflectors from the right because it processes rows (not columns). Each H_j zeros row j to the right of the diagonal, creating the lower triangular L factor. Right multiplication A*H_j preserves previously factored rows (unlike left multiplication H_j*A which would destroy them). This is the fundamental difference between row-based LQ and column-based QR.",
    tags=["analysis", "lq-factorization", "right-multiplication", "householder-reflectors", "matrix-structure", "left-vs-right"]
))

# L1-3: CODING - Workspace sizing for RIGHT-side LARF
entries.append(create_entry(
    level="L1",
    interface="gelq2",
    instruction="Implement workspace calculation for GELQ2. Why does it query LARF with rocblas_side_right and stride lda for the Householder vector, and how does this differ from GEQR2's workspace?",
    context_text="GELQ2 workspace must accommodate RIGHT-side LARF applications with row-strided Householder vectors. The workspace requirements differ from left-side QR factorization.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, typename T>
void rocsolver_gelq2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_right, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}"""
        }
    ],
    answer="""Here's the complete workspace calculation for GELQ2:

```cpp
template <bool BATCHED, typename T>
void rocsolver_gelq2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // Quick return for empty matrices
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // Query LARF workspace for RIGHT-side application
    size_t larf_scalars, larf_norms, larf_work;
    rocsolver_larf_getMemorySize<BATCHED, T>(
        rocblas_side_right,   // RIGHT-side: A * H (not H * A)
        m,                     // Matrix has m rows
        n,                     // Reflector size: n elements
        batch_count,
        &larf_scalars,        // Scalars for GEMV/GER
        &larf_norms,          // Temp for GEMV: m elements per batch
        &larf_work);          // Work/pointer array

    // Query LARFG workspace for row vector generation
    size_t larfg_work, larfg_norms;
    rocsolver_larfg_getMemorySize<T>(
        n,                     // Row vector length (max at j=0)
        batch_count,
        &larfg_work,          // DOT reduction workspace
        &larfg_norms);        // Norm storage

    // Take maximum (sequential reuse)
    *size_scalars = larf_scalars;
    *size_work_workArr = std::max(larf_work, larfg_work);
    *size_Abyx_norms = std::max(larf_norms, larfg_norms);

    // Diagonal backup: one element per batch per iteration
    // Unlike GEQR2 which stores ALL diagonals, GELQ2 stores one at a time
    *size_diag = sizeof(T) * batch_count;
}
```

**Why rocblas_side_right?**

LARF workspace depends on operation type:

**LEFT-side (GEQR2): H * A**
```
w = A^T * v     (GEMV: n × m, result is n elements)
A = A - v * w^T (GER: m × n)
```
Temp workspace: n * sizeof(T) * batch_count

**RIGHT-side (GELQ2): A * H**
```
w = A * v       (GEMV: m × n, result is m elements)
A = A - w * v^H (GER: m × n)
```
Temp workspace: m * sizeof(T) * batch_count

The side parameter determines temp array size!

**Comparison with GEQR2:**

| Aspect | GEQR2 | GELQ2 |
|--------|-------|-------|
| LARF side | rocblas_side_left | rocblas_side_right |
| LARF m,n | m, n | m, n |
| Temp size | n * batch_count | m * batch_count |
| LARFG size | m (column length) | n (row length) |
| Diag size | min(m,n) * batch (ALL) | 1 * batch (per-iter) |

**Why Different Diagonal Storage?**

GEQR2:
- Saves ALL diagonals during LARFG calls
- Batch restores at end: min(m,n) elements
- More efficient: 1 restore kernel

GELQ2:
- Uses set_diag/restore_diag per iteration
- Only needs 1 diagonal at a time
- Follows GEQL2 pattern (per-iteration approach)

**Workspace Breakdown (512×1024 FP64):**
```
size_scalars:    24 bytes (3 * sizeof(double))
size_work_workArr: max(LARF_work, LARFG_work)
                 = max(pointer_array, DOT_workspace)
                 ≈ 4 KB
size_Abyx_norms: max(m, n) * sizeof(T) * batch
               = max(512, 1024) * 8 * 1
               = 8 KB
size_diag:      1 * 8 * 1 = 8 bytes (per iteration, not all)

Total: ~12 KB
```""",
    rationale="GELQ2 queries LARF with rocblas_side_right because it applies reflectors from the right (A*H). This determines temp workspace size: m elements (not n). LARFG is queried with n (row length). Unlike GEQR2 which stores all diagonals, GELQ2 uses per-iteration diagonal storage (1 element), following GEQL2's pattern with set_diag/restore_diag.",
    tags=["coding", "workspace-management", "right-side-larf", "memory-optimization", "lq-factorization", "diagonal-storage"]
))

# L2-1: ANALYSIS - LACGV + LARFG + LARF pipeline for rows
entries.append(create_entry(
    level="L2",
    interface="gelq2",
    instruction="Analyze the complete LACGV→LARFG→set_diag→LARF→restore_diag→LACGV pipeline for processing row j in complex GELQ2. Trace the data transformations and explain why each step is necessary.",
    context_text="Complex GELQ2 has a 6-step pipeline per row: conjugate, generate reflector, save diagonal, apply reflector, restore diagonal, unconjugate. Each step is critical for correctness.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);

    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                             (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(j,j) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                            1, true);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }

    // restore original value of A(j,j)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);

    // restore the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);
}"""
        }
    ],
    answer="""Complete pipeline trace for complex GELQ2 row j (5×4 complex matrix, j=1):

**Initial State:**
```
A = [ L₀+0i   v₀+0i   v₀+0i   v₀+0i ]
    [ *+*i    *+*i    *+*i    *+*i  ]  ← Row 1 (j=1)
    [ *+*i    *+*i    *+*i    *+*i  ]
    [ *+*i    *+*i    *+*i    *+*i  ]
    [ *+*i    *+*i    *+*i    *+*i  ]

Row 1: [a₁₁+b₁₁i, a₁₂+b₁₂i, a₁₃+b₁₃i, a₁₄+b₁₄i]  (n-j = 3 elements to process)
```

**Step 1: LACGV (conjugate row j)**
```
Operation: A(1, 1:3) = conj(A(1, 1:3))
Result:    [a₁₁-b₁₁i, a₁₂-b₁₂i, a₁₃-b₁₃i, a₁₄-b₁₄i]

Why: LARFG expects column vectors. Conjugating a row vector converts it
     to equivalent column vector form for reflector generation.

Memory: Read/write 3 elements with stride=lda
```

**Step 2: LARFG (generate Householder reflector)**
```
Input:  alpha = a₁₁-b₁₁i (A(1,1))
        x = [a₁₂-b₁₂i, a₁₃-b₁₃i, a₁₄-b₁₄i]^T (A(1,2:3))

Compute:
  norm2 = |x|² = |a₁₂-b₁₂i|² + |a₁₃-b₁₃i|² + |a₁₄-b₁₄i|²
  beta = -sign(Re(alpha)) * sqrt(|alpha|² + norm2)  (real L diagonal)
  tau = (beta - alpha) / beta
  v = x / (alpha - beta)

Output:
  A(1,1) = beta (L diagonal, REAL even for complex)
  A(1,2:3) = v (Householder vector, still CONJUGATED)
  tau[1] = tau

Why: Creates reflector H = I - tau*v*v^H that zeros row 1 to right of diagonal
```

**Step 3: SET_DIAG (save diagonal, write 1.0)**
```
Operation: diag[batch_id] = A(1,1)  (save beta)
          A(1,1) = 1.0             (implicit v[0]=1)

Why: LARF expects v[0]=1 implicitly. Must temporarily replace diagonal.

A = [ L₀   v₀   v₀   v₀  ]
    [ L₁  1.0   v₁   v₁  ]  ← v₁ are conjugated!
    [ *    *    *    *   ]
    [ *    *    *    *   ]
    [ *    *    *    *   ]
```

**Step 4: LARF (apply from right)**
```
Operation: A(2:4, 1:3) = A(2:4, 1:3) * H
          where H = I - tau * v * v^H

Compute:
  w = A(2:4, 1:3) * v  (GEMV: 3×3 matrix × 3 vector = 3 vector)
    = [A(2,1)*1.0 + A(2,2)*v₁ + A(2,3)*v₁,    ← v are conjugated
       A(3,1)*1.0 + A(3,2)*v₁ + A(3,3)*v₁,
       A(4,1)*1.0 + A(4,2)*v₁ + A(4,3)*v₁]

  A(2:4, 1:3) -= tau * w * v^H  (GER)
              = A - tau * w * [1.0, conj(v₁), conj(v₁)]

Why: Transforms remaining rows to maintain LQ structure.
     Uses conjugated v for correct complex arithmetic.

Result:
A = [ L₀   v₀   v₀   v₀  ]
    [ L₁  1.0   v₁   v₁  ]  ← still has 1.0, v conjugated
    [ *'   *'   *'   *'  ]  ← updated by LARF
    [ *'   *'   *'   *'  ]
    [ *'   *'   *'   *'  ]
```

**Step 5: RESTORE_DIAG (restore L diagonal)**
```
Operation: A(1,1) = diag[batch_id]  (restore beta)

Why: Write back the L factor diagonal element.

A = [ L₀   v₀   v₀   v₀  ]
    [ L₁   L₁   v₁   v₁  ]  ← diagonal restored, v still conjugated
    [ *'   *'   *'   *'  ]
    [ *'   *'   *'   *'  ]
    [ *'   *'   *'   *'  ]
```

**Step 6: LACGV (unconjugate row j)**
```
Operation: A(1, 1:3) = conj(A(1, 1:3))

Before: [L₁, a₁₂-b₁₂i, a₁₃-b₁₃i, a₁₄-b₁₄i]  (v conjugated)
After:  [L₁, a₁₂+b₁₂i, a₁₃+b₁₃i, a₁₄+b₁₄i]  (v in original form)

Why: Restore Householder vector to original (unconjugated) representation.
     Storage format: row of L, followed by unconjugated reflector elements.

Final:
A = [ L₀   v₀   v₀   v₀  ]
    [ L₁   L₁   v₁   v₁  ]  ← v₁ unconjugated, ready for next iteration
    [ *'   *'   *'   *'  ]
    [ *'   *'   *'   *'  ]
    [ *'   *'   *'   *'  ]
```

**Data Flow Summary:**
```
Original row j:        [a+bi,  c+di,  e+fi,  g+hi]
After LACGV:          [a-bi,  c-di,  e-fi,  g-hi]  (conjugated)
After LARFG:          [beta,  v₁,    v₂,    v₃]    (v conjugated)
After SET_DIAG:       [1.0,   v₁,    v₂,    v₃]    (saved beta)
After LARF:           [1.0,   v₁,    v₂,    v₃]    (other rows updated)
After RESTORE_DIAG:   [beta,  v₁,    v₂,    v₃]    (beta restored)
After LACGV:          [beta,  v₁',   v₂',   v₃']   (v unconjugated)
```

**Why All 6 Steps?**
1. LACGV: Adapt row to column-based LARFG
2. LARFG: Generate reflector on conjugated data
3. SET_DIAG: Prepare for LARF (needs v[0]=1)
4. LARF: Apply transformation with conjugated v
5. RESTORE_DIAG: Write back L diagonal
6. LACGV: Restore standard storage format""",
    rationale="The 6-step pipeline handles the mismatch between row-based LQ and column-based LARFG. LACGV conjugates the row (row→column equivalence), LARFG generates the reflector, set_diag prepares for LARF (v[0]=1), LARF applies the transformation, restore_diag writes back the L diagonal, and final LACGV unconjugates for standard storage. Each step is essential for correct complex arithmetic in row-based factorization.",
    tags=["analysis", "complex-arithmetic", "lacgv-larfg-larf-pipeline", "row-operations", "data-flow", "lq-factorization"]
))

# L2-2: CODING - Per-iteration diagonal backup
entries.append(create_entry(
    level="L2",
    interface="gelq2",
    instruction="Implement the per-iteration diagonal backup strategy used by GELQ2 and explain why it differs from GEQR2's batch restore approach. Which is more efficient and why?",
    context_text="GELQ2 uses set_diag/restore_diag kernels in each iteration (like GEQL2), while GEQR2 saves all diagonals and restores in batch. This affects kernel launch count and workspace.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """// insert one in A(j,j) to build/apply the householder matrix
ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                        0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                        1, true);

// Apply Householder reflector to the rest of matrix from the right
if(j < m - 1)
{
    rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                            shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                            shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                            scalars, Abyx_norms, (T**)work_workArr);
}

// restore original value of A(j,j)
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                        dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                        lda, strideA, 1);"""
        }
    ],
    answer="""Here's the per-iteration diagonal strategy for GELQ2:

```cpp
// Workspace: diag buffer sized for ONE diagonal element per batch
// size_diag = sizeof(T) * batch_count  (not sizeof(T) * dim * batch_count)

template <typename T, typename U>
void gelq2_iteration_with_diagonal_backup(rocblas_handle handle,
                                          rocblas_int j, rocblas_int m, rocblas_int n,
                                          U A, rocblas_int shiftA, rocblas_int lda,
                                          rocblas_stride strideA,
                                          T* ipiv, rocblas_stride strideP,
                                          rocblas_int batch_count,
                                          T* scalars, void* work_workArr, T* Abyx_norms,
                                          T* diag)  // Single element per batch
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Step 1: Save A(j,j) to diag[0] and write 1.0
    ROCSOLVER_LAUNCH_KERNEL(
        (set_diag<T, rocblas_int>),
        dim3(batch_count, 1, 1),          // Grid: one block per batch
        dim3(1, 1, 1),                    // Threads: single thread
        0, stream,
        diag,                             // Save buffer
        0,                                // Offset: always 0 (reuse same slot)
        1,                                // Stride: 1 (one per batch)
        A,                                // Matrix
        shiftA + idx2D(j, j, lda),       // Diagonal position A(j,j)
        lda, strideA,
        1,                                // Number of elements: 1
        true);                            // save=true (backup mode)

    // set_diag kernel (simplified):
    // diag[batch_id * 1 + 0] = A[batch_id][j,j]  // Save
    // A[batch_id][j,j] = 1.0                      // Write

    // Step 2: Apply reflector (uses implicit v[0]=1.0)
    if(j < m - 1)
    {
        rocsolver_larf_template(
            handle,
            rocblas_side_right,
            m - j - 1,                     // Rows to update
            n - j,                         // Reflector size
            A, shiftA + idx2D(j, j, lda), // Householder vector (row j)
            lda, strideA,
            (ipiv + j), strideP,
            A, shiftA + idx2D(j + 1, j, lda),  // Submatrix
            lda, strideA,
            batch_count, scalars, Abyx_norms, (T**)work_workArr);
    }

    // Step 3: Restore A(j,j) from diag[0]
    ROCSOLVER_LAUNCH_KERNEL(
        (restore_diag<T, rocblas_int>),
        dim3(batch_count, 1, 1),
        dim3(1, 1, 1),
        0, stream,
        diag,                             // Restore buffer
        0,                                // Offset: always 0
        1,                                // Stride: 1
        A,
        shiftA + idx2D(j, j, lda),
        lda, strideA,
        1);                               // Number of elements: 1

    // restore_diag kernel (simplified):
    // A[batch_id][j,j] = diag[batch_id * 1 + 0]
}
```

**Comparison: GELQ2 vs GEQR2**

| Aspect | GELQ2 (per-iteration) | GEQR2 (batch restore) |
|--------|----------------------|----------------------|
| **Diagonal storage** | sizeof(T) * batch | sizeof(S) * min(m,n) * batch |
| **Save location** | diag[0] reused | diag[j] unique per iteration |
| **Save mechanism** | set_diag kernel | LARFG beta output |
| **Restore timing** | Each iteration | Once after loop |
| **Kernel launches** | 2*dim (set+restore) | 1 (batch restore) |
| **Memory traffic** | dim reads + dim writes | 1 read (sequential) |

**Why GELQ2 Uses Per-Iteration (Less Efficient):**

1. **LARFG doesn't save diagonals for LQ**:
   - QR: LARFG computes beta (R diagonal) and can save it
   - LQ: LARFG works on conjugated row, beta goes to A(j,j) immediately
   - No natural place to save diagonal during LARFG

2. **RIGHT-side LARF**:
   - Must have v[0]=1 before applying from right
   - Diagonal must be replaced before LARF, restored after
   - Can't defer to end-of-loop like QR

3. **Complex row handling**:
   - Row is conjugated/unconjugated around LARFG/LARF
   - Diagonal must be managed within this conjugation block

**Efficiency Analysis (512x1024 matrix):**

**GELQ2 (per-iteration):**
```
Diagonal workspace: 512 * 8 = 4 KB (single element)
Kernel launches:    512 set_diag + 512 restore_diag = 1024
Memory traffic:     512 saves + 512 restores = 1024 elements = 8 KB
Launch overhead:    1024 * 5μs = 5.12 ms
```

**GEQR2 (batch restore):**
```
Diagonal workspace: 512 * 8 = 4 KB (all elements, real-only)
Kernel launches:    1 restore_diag = 1
Memory traffic:     512 restores = 512 elements = 4 KB
Launch overhead:    1 * 5μs = 0.005 ms
```

**GEQR2 is ~1000x more efficient for diagonal handling** due to:
- 1024x fewer kernel launches
- 2x less memory traffic
- Saves during LARFG (no extra kernels)

**Why Not Change GELQ2?**

Could potentially use GEQR2 approach:
```cpp
// Hypothetical: save during LARFG
rocsolver_larfg_template(..., beta_output_array, j, dim, ...)
```

But would require:
1. Modifying LARFG to accept beta array (API change)
2. Handling conjugated row complexity
3. Non-trivial refactoring for marginal gain (diagonal ops are <1% of runtime)

Current design prioritizes code simplicity over micro-optimization.""",
    rationale="GELQ2 uses per-iteration diagonal backup with set_diag/restore_diag because: (1) LARFG works on conjugated rows and doesn't naturally save diagonals like in QR, (2) RIGHT-side LARF requires v[0]=1 before application, (3) diagonal must be managed within the conjugation block. GEQR2's batch restore is ~1000x more efficient but requires natural diagonal saving during LARFG, which LQ doesn't have. The inefficiency is acceptable since diagonal operations are <1% of total runtime.",
    tags=["coding", "diagonal-preservation", "per-iteration-backup", "kernel-launches", "gelq2-vs-geqr2", "performance-tradeoffs"]
))

# L3-1: ANALYSIS - Algorithm trace
entries.append(create_entry(
    level="L3",
    interface="gelq2",
    instruction="Trace the complete GELQ2 algorithm for a 1024x512 complex double-precision matrix, including all kernel launches, memory traffic, and handling of complex conjugation. Compare FLOPs and memory with real-valued GEQR2.",
    context_text="Complex GELQ2 has additional LACGV operations compared to real matrices. For 1024x512, analyze the complete execution including conjugation overhead.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """rocblas_int dim = std::min(m, n); // total number of pivots

for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);

    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                             (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(j,j) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                            1, true);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }

    // restore original value of A(j,j)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);

    // restore the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);
}"""
        }
    ],
    answer="""Complete GELQ2 trace for 1024x512 complex FP64:

**Problem Setup:**
- m = 1024, n = 512, dim = min(1024, 512) = 512
- Type: rocblas_double_complex (16 bytes)
- Iterations: j = 0 to 511 (512 rows)

**Per Iteration j (0 to 511):**

**1. LACGV (conjugate row j):**
```
Elements: n - j = 512 - j
Operation: A(j, j:511) = conj(A(j, j:511))
FLOPs: 0 (just flip sign of imaginary part)
Memory: Read/write (512-j) * 16 bytes
Kernel: conj_in_place with (512-j)/64 blocks
```

**2. LARFG (generate reflector):**
```
Vector length: n - j = 512 - j
DOT: norm2 = sum(|A(j, j+1:511)|^2)
  FLOPs: 4 * (512-j-1)  (complex: 4 ops per multiply-add)
SCAL: A(j, j+1:511) *= scaling
  FLOPs: 6 * (512-j-1)  (complex multiply)
Total FLOPs: ~10 * (512-j)
Memory: Read/write (512-j) * 16 bytes
```

**3. SET_DIAG:**
```
FLOPs: 0
Memory: Read 16 bytes (A(j,j)), write 16 (diag), write 16 (A(j,j)=1.0)
Kernel: 1 block, 1 thread
```

**4. LARF (apply from right): A(j+1:1023, j:511) * H:**
```
Dimensions: (1023-j) × (512-j)
GEMV: w = A * v
  FLOPs: 8 * (1023-j) * (512-j)  (complex)
  Memory: Read (1023-j)*(512-j)*16 + (512-j)*16, write (1023-j)*16
GER: A = A - tau * w * v^H
  FLOPs: 8 * (1023-j) * (512-j)  (complex)
  Memory: Read (1023-j)*(512-j)*16 + (1023-j)*16 + (512-j)*16,
          write (1023-j)*(512-j)*16
Total FLOPs: 16 * (1023-j) * (512-j)
```

**5. RESTORE_DIAG:**
```
FLOPs: 0
Memory: Read 16 bytes (diag), write 16 (A(j,j))
Kernel: 1 block, 1 thread
```

**6. LACGV (unconjugate row j):**
```
Elements: n - j = 512 - j
FLOPs: 0
Memory: Read/write (512-j) * 16 bytes
Kernel: conj_in_place with (512-j)/64 blocks
```

**Total Computation (all 512 iterations):**

**FLOPs:**
```
LACGV (×2):   0
LARFG:        Σ(j=0..511)[10*(512-j)] ≈ 1,310,720
LARF:         Σ(j=0..510)[16*(1023-j)*(512-j)]
            = 16 * Σ(k=1..512)[k*(k+511)]
            ≈ 16 * 89,913,856
            ≈ 1,438,621,696

Total: ~1.44 GFLOPs (complex double)
```

**For comparison, real GEQR2 (512×1024):**
```
GEQR2 FLOPs ≈ 4 * Σ(j=0..511)[(1024-j)*(511-j)]
            ≈ 357 MFLOPs (real double)

Ratio: 1.44 GFLOP / 0.357 GFLOP = 4x
(Expected: complex ops are 4x real ops)
```

**Memory Traffic:**

```
Per iteration j:
  LACGV (×2):   2 * 2 * (512-j) * 16 bytes
  LARFG:        2 * (512-j) * 16 bytes
  LARF:         ~4 * (1023-j) * (512-j) * 16 bytes
  SET/RESTORE:  4 * 16 bytes

Total per iter: ~4 * (1023-j) * (512-j) * 16 bytes

Sum over j=0..511:
  ≈ 4 * 16 * 89,913,856
  ≈ 5.75 GB

Real GEQR2 (512×1024):
  ≈ 1.44 GB (8 bytes vs 16 bytes)

Ratio: 5.75 GB / 1.44 GB = 4x (expected for complex)
```

**Kernel Launch Count:**

```
Per iteration:
  LACGV:      2 kernels (if complex)
  LARFG:      ~1 kernel
  SET_DIAG:   1 kernel
  LARF:       ~1 kernel
  RESTORE:    1 kernel

Total per iter: 6 kernels
Total: 512 * 6 = 3,072 kernel launches

Real GEQR2:
  Per iter: 3 kernels (LARFG, LARF, no LACGV)
  Total: 512 * 3 + 1 = 1,537 launches

Ratio: 3,072 / 1,537 = 2x (conjugation overhead)
```

**Arithmetic Intensity:**

```
Complex GELQ2: 1.44 GFLOP / 5.75 GB = 0.25 FLOP/byte
Real GEQR2:    0.357 GFLOP / 1.44 GB = 0.25 FLOP/byte

Same! Complex operations: 4x FLOPs, 4x memory → same intensity
```

**Performance Estimate (MI250X):**

```
Complex GELQ2:
  Kernel launches: 3,072 * 5μs = 15.4 ms
  Compute: 1.44 GFLOP @ 200 GFLOPS = 7.2 ms
  Memory: 5.75 GB @ 1.6 TB/s = 3.6 ms
  Total: ~26 ms (dominated by launch overhead)

Real GEQR2:
  Kernel launches: 1,537 * 5μs = 7.7 ms
  Compute: 0.357 GFLOP @ 200 GFLOPS = 1.8 ms
  Memory: 1.44 GB @ 1.6 TB/s = 0.9 ms
  Total: ~10 ms

Ratio: 26 ms / 10 ms = 2.6x slower
  (Expected ~2x due to LACGV overhead)
```

**Key Observations:**
1. Complex doubles LACGV overhead: 2x kernel launches vs real
2. Same arithmetic intensity (0.25 FLOP/byte) for complex and real
3. Complex is 2.6x slower: 2x from LACGV kernels, 1.3x from larger memory
4. Launch-bound: 15.4 ms launch overhead vs 7.2 ms compute""",
    rationale="Complex GELQ2 requires 2x kernel launches vs real due to LACGV conjugation (2 per iteration). Computational cost is 4x (complex arithmetic), memory is 4x (16-byte vs 8-byte elements), so arithmetic intensity stays 0.25 FLOP/byte. Performance is 2.6x slower than real GEQR2: 2x from extra LACGV kernels, 1.3x from memory overhead. The algorithm is launch-bound with kernel overhead dominating compute.",
    tags=["analysis", "complex-arithmetic", "performance-analysis", "lacgv-overhead", "algorithm-trace", "flop-counting", "memory-traffic"]
))

# L3-2: CODING - Production wrapper
entries.append(create_entry(
    level="L3",
    interface="gelq2",
    instruction="Implement a complete production GELQ2 wrapper with proper argument validation, workspace management, and complex type handling. Include the shared argument checker used by both GELQ2 and GELQF.",
    context_text="GELQ2 must handle complex types correctly, validate arguments in the proper order, and share validation code with GELQF for consistency.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gelq2.hpp",
            "language": "cpp",
            "content": """template <typename T, typename U>
rocblas_status rocsolver_gelq2_gelqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}"""
        }
    ],
    answer="""Complete production GELQ2 wrapper:

```cpp
// Shared argument checker (used by GELQ2 and GELQF)
template <typename T, typename U>
rocblas_status rocsolver_gelq2_gelqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // Validation order is critical for unit test compatibility

    // 1. Invalid/non-supported values (none for GELQ2/GELQF)
    // N/A

    // 2. Invalid sizes
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // Skip pointer validation during workspace query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. Invalid pointers (only when m,n > 0)
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// Main implementation template
template <typename T, typename U>
rocblas_status rocsolver_gelq2_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    // Logging
    ROCSOLVER_ENTER_TOP("gelq2", "-m", m, "-n", n, "--lda", lda);

    // Step 1: Validate handle
    if(!handle)
        return rocblas_status_invalid_handle;

    // Step 2: Argument validation
    rocblas_status st = rocsolver_gelq2_gelqf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // Step 3: Setup for non-batched execution
    const rocblas_int shiftA = 0;
    const rocblas_stride strideA = 0;
    const rocblas_stride stridep = 0;
    const rocblas_int batch_count = 1;

    // Step 4: Query workspace requirements
    size_t size_scalars, size_work_workArr, size_Abyx_norms, size_diag;
    rocsolver_gelq2_getMemorySize<false, T>(m, n, batch_count,
                                           &size_scalars,
                                           &size_work_workArr,
                                           &size_Abyx_norms,
                                           &size_diag);

    // Step 5: Workspace query mode
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle,
                                                     size_scalars,
                                                     size_work_workArr,
                                                     size_Abyx_norms,
                                                     size_diag);

    // Step 6: Allocate device workspace
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle,
                             size_scalars,
                             size_work_workArr,
                             size_Abyx_norms,
                             size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    // Step 7: Assign workspace pointers
    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];

    // Step 8: Initialize scalar constants
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // Step 9: Execute algorithm
    return rocsolver_gelq2_template<T>(handle,
                                      m, n,
                                      A, shiftA, lda, strideA,
                                      ipiv, stridep,
                                      batch_count,
                                      (T*)scalars,
                                      work_workArr,
                                      (T*)Abyx_norms,
                                      (T*)diag);
}

// C API wrappers
extern "C" {

rocblas_status rocsolver_sgelq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_gelq2_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgelq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_gelq2_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgelq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_gelq2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgelq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_gelq2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
}

} // extern "C"
```

**Key Features:**

1. **Type-generic implementation**:
   - Single template handles float, double, complex float, complex double
   - COMPLEX template parameter auto-detected via rocblas_is_complex<T>

2. **Shared argument checker**:
   - Used by both GELQ2 and GELQF for consistency
   - Validates in specific order for unit test compatibility
   - Default batch_count=1 for non-batched wrappers

3. **Complex type handling**:
   - LACGV automatically called when COMPLEX=true in template
   - No special code needed in wrapper
   - Complex requires sizeof(T)=16 bytes workspace

4. **Workspace management**:
   - Single allocation for all workspace
   - Query mode returns before allocation
   - Proper sizing for RIGHT-side LARF

5. **Error handling order**:
   - invalid_handle → invalid_size → invalid_pointer → memory_error
   - Skip pointer check during workspace query

6. **API design**:
   - Four C wrappers for different precisions
   - Consistent naming: [s|d|c|z]gelq2
   - Template backend reduces code duplication

**Usage Example:**
```cpp
// User code
rocblas_handle handle;
const int m = 1024, n = 512;
rocblas_double_complex* A;  // device pointer
rocblas_double_complex* tau;  // device pointer

// Query workspace
size_t workspace_size;
rocsolver_zgelq2(handle, m, n, nullptr, m, nullptr);  // query mode
rocblas_get_device_memory_size(handle, &workspace_size);

// Allocate workspace
rocblas_set_device_memory_size(handle, workspace_size);

// Execute
rocsolver_zgelq2(handle, m, n, A, m, tau);
```""",
    rationale="Production GELQ2 wrapper must handle all four precisions through templates, share argument validation with GELQF, manage workspace correctly for RIGHT-side operations, and automatically handle complex conjugation via template specialization. The design separates concerns: argument checking is shared and order-specific, workspace is type-aware (complex needs 16 bytes), and template automatically enables LACGV for complex types.",
    tags=["coding", "production-wrapper", "complex-type-handling", "shared-validation", "api-design", "workspace-management", "template-metaprogramming"]
))

# Write to JSONL
output_file = Path("kernelgen/dataset/roclapack_gelq2.jsonl")
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Calculate statistics
total = len(entries)
l1 = sum(1 for e in entries if e['level'] == 'L1')
l2 = sum(1 for e in entries if e['level'] == 'L2')
l3 = sum(1 for e in entries if e['level'] == 'L3')
coding = sum(1 for e in entries if 'coding' in e['tags'])
analysis = sum(1 for e in entries if 'analysis' in e['tags'])

print(f"Generated {total} entries")
print(f"Written to: {output_file.absolute()}")
print(f"\nDistribution:")
print(f"  L1: {l1} entries")
print(f"  L2: {l2} entries")
print(f"  L3: {l3} entries")
print(f"  Coding: {coding} ({coding*100//total}%)")
print(f"  Analysis: {analysis} ({analysis*100//total}%)")
