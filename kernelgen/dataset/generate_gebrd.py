#!/usr/bin/env python3
"""
Generator for GEBRD (blocked bidiagonal reduction) SFT dataset entries.
GEBRD reduces a general m×n matrix to bidiagonal form using Householder reflectors.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def generate_entries():
    entries = []

    # L1-1: Upper vs Lower Bidiagonal Form Selection (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_gebrd_L1_bidiag_form",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """In rocsolver_gebd2_template, the algorithm chooses between upper bidiagonal (m >= n) and lower bidiagonal (m < n) forms. Explain the structural differences between these two forms and why the choice depends on matrix dimensions.

Code blocks for context:
```cpp
// From roclapack_gebd2.hpp lines 133-211
rocblas_int dim = std::min(m, n); // total number of pivots

if(m >= n)
{
    // generate upper bidiagonal form
    for(rocblas_int j = 0; j < n; j++)
    {
        // generate Householder reflector H(j)
        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(std::min(j + 1, m - 1), j, lda), 1, strideA,
                                 (tauq + j), strideQ, batch_count, (T*)work_workArr, Abyx_norms);

        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

        if(j < n - 1)
        {
            // Apply Householder reflector H(j) from LEFT
            rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                    shiftA + idx2D(j, j, lda), 1, strideA, (tauq + j), strideQ,
                                    A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                    scalars, Abyx_norms, (T**)work_workArr);

            // Conjugate row j
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);

            // generate Householder reflector G(j)
            rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda), A,
                                     shiftA + idx2D(j, std::min(j + 2, n - 1), lda), lda,
                                     strideA, (taup + j), strideP, batch_count,
                                     (T*)work_workArr, Abyx_norms);

            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA, 1, true);

            // Apply Householder reflector G(j) from RIGHT
            rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j - 1, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA, (taup + j),
                                    strideP, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                    batch_count, scalars, Abyx_norms, (T**)work_workArr);

            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                            lda, strideA, batch_count);
        }
    }
}
else
{
    // generate lower bidiagonal form
    for(rocblas_int j = 0; j < m; j++)
    {
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, batch_count);

        // generate Householder reflector G(j)
        rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                                 (taup + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

        // Apply Householder reflector G(j) from RIGHT
        if(j < m - 1)
        {
            rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                    shiftA + idx2D(j, j, lda), lda, strideA, (taup + j),
                                    strideP, A, shiftA + idx2D(j + 1, j, lda), lda, strideA,
                                    batch_count, scalars, Abyx_norms, (T**)work_workArr);
        }

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                        strideA, batch_count);

        if(j < m - 1)
        {
            // generate Householder reflector H(j)
            rocsolver_larfg_template(handle, m - j - 1, A, shiftA + idx2D(j + 1, j, lda), A,
                                     shiftA + idx2D(std::min(j + 2, m - 1), j, lda), 1, strideA,
                                     (tauq + j), strideQ, batch_count, (T*)work_workArr,
                                     Abyx_norms);

            ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                    dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                                    shiftA + idx2D(j + 1, j, lda), lda, strideA, 1, true);

            // Apply Householder reflector H(j) from LEFT
            rocsolver_larf_template(handle, rocblas_side_left, m - j - 1, n - j - 1, A,
                                    shiftA + idx2D(j + 1, j, lda), 1, strideA, (tauq + j),
                                    strideQ, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                    batch_count, scalars, Abyx_norms, (T**)work_workArr);
        }
    }
}
```

Explain the diagonal/superdiagonal positions for upper bidiagonal, diagonal/subdiagonal positions for lower bidiagonal, and which reflectors (H vs G) eliminate which parts of the matrix."""
                }
            ]
        }
    })

    # L1-2: LABRD Panel Reduction with X and Y Matrices (CODING)
    entries.append({
        "custom_id": "roclapack_gebrd_L1_labrd_xy_update",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Implement the column j update in rocsolver_labrd_template for the upper bidiagonal case (m >= n). This involves updating A[j:m, j] using previously computed X and Y matrices, then computing the Householder reflector H(j).

Requirements:
- Update column j: A[j:m, j] -= Y[j, 0:j] * A[j:m, 0:j]^H - X[j:m, 0:j] * A[0:j, j]
- Conjugate Y[j, 0:j] before and after GEMV if complex
- Generate Householder reflector H(j) for column j
- Store diagonal element in D[j]

Code blocks for context:
```cpp
// From rocauxiliary_labrd.hpp lines 163-200 (upper bidiagonal, column update)
for(rocblas_int j = 0; j < k; ++j)
{
    // update column j of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                                    batch_count);

    rocblasCall_gemv<T>(handle, rocblas_operation_none, m - j, j,
                        cast2constType<T>(scalars), 0, A, shiftA + idx2D(j, 0, lda), lda,
                        strideA, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                        cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                        strideA, batch_count, (T**)work_workArr);

    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, j, Y, shiftY + idx2D(j, 0, ldy), ldy, strideY,
                                    batch_count);
    rocblasCall_gemv<T>(handle, rocblas_operation_none, m - j, j,
                        cast2constType<T>(scalars), 0, X, shiftX + idx2D(j, 0, lda), ldx,
                        strideX, A, shiftA + idx2D(0, j, lda), 1, strideA,
                        cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j, lda), 1,
                        strideA, batch_count, (T**)work_workArr);

    // generate Householder reflector to work on column j
    rocsolver_larfg_template(
        handle,
        m - j, // order of reflector
        A, shiftA + idx2D(j, j, lda), // value of alpha
        A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), // vector x to work on
        1, strideA, // inc of x
        (tauq + j), strideQ, // tau
        batch_count, (T*)work_workArr, norms);

    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                            shiftA + idx2D(j, j, lda), lda, strideA, 1, j < n - 1);
}
```

Provide the complete implementation for this column update step."""
                }
            ]
        }
    })

    # L1-3: Complex Row/Column Conjugation Pattern (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_gebrd_L1_conjugation",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Why does GEBD2 conjugate rows/columns before and after generating row-based Householder reflectors in the complex case? Compare the conjugation pattern in upper bidiagonal (conjugate row j before generating G(j)) versus lower bidiagonal (conjugate row j before generating G(j)).

Code blocks for context:
```cpp
// From roclapack_gebd2.hpp lines 171-202 (upper bidiagonal, row-based reflector)
if(j < n - 1)
{
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                    lda, strideA, batch_count);

    // generate Householder reflector G(j)
    rocsolver_larfg_template(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda), A,
                             shiftA + idx2D(j, std::min(j + 2, n - 1), lda), lda,
                             strideA, (taup + j), strideP, batch_count,
                             (T*)work_workArr, Abyx_norms);

    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, E, j, strideE, A,
                            shiftA + idx2D(j, j + 1, lda), lda, strideA, 1, true);

    // Apply Householder reflector G(j) from RIGHT
    rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j - 1, A,
                            shiftA + idx2D(j, j + 1, lda), lda, strideA, (taup + j),
                            strideP, A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                            batch_count, scalars, Abyx_norms, (T**)work_workArr);

    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(j, j + 1, lda),
                                    lda, strideA, batch_count);
}
```

```cpp
// From roclapack_gebd2.hpp lines 217-243 (lower bidiagonal, row-based reflector)
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, batch_count);

// generate Householder reflector G(j)
rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                         shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                         (taup + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                        dim3(1, 1, 1), 0, stream, D, j, strideD, A,
                        shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

// Apply Householder reflector G(j) from RIGHT
if(j < m - 1)
{
    rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                            shiftA + idx2D(j, j, lda), lda, strideA, (taup + j),
                            strideP, A, shiftA + idx2D(j + 1, j, lda), lda, strideA,
                            batch_count, scalars, Abyx_norms, (T**)work_workArr);
}

if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, batch_count);
```

Explain why LARFG requires real input for row-based reflectors, necessitating conjugation, and why column-based reflectors H(j) don't need conjugation (or conjugate tau instead)."""
                }
            ]
        }
    })

    # L2-1: LABRD-GEMM Update Pipeline (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_gebrd_L2_labrd_gemm",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Analyze the LABRD-GEMM pipeline in rocsolver_gebrd_template. Explain how LABRD reduces a panel to bidiagonal form while accumulating updates in X and Y matrices, then how two GEMM calls apply these accumulated updates to the trailing matrix.

Context: GEBRD processes the matrix in blocks of size nb. For each block j:
1. LABRD reduces A[j:m, j:j+jb] to bidiagonal, storing reflectors in A and updates in X, Y
2. First GEMM: A[j+jb:m, j+jb:n] -= A[j+jb:m, j:j+jb] * Y[j+jb:n, 0:jb]^H
3. Second GEMM: A[j+jb:m, j+jb:n] -= X[j+jb:m, 0:jb] * A[j:j+jb, j+jb:n]

Code blocks for context:
```cpp
// From roclapack_gebrd.hpp lines 153-194
while(j < dim - k)
{
    // Reduce block to bidiagonal form
    jb = std::min(dim - j, nb); // number of rows and columns in the block
    rocsolver_labrd_template<T>(handle, m - j, n - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, D + j, strideD, E + j, strideE, tauq + j, strideQ,
                                taup + j, strideP, X, shiftX, ldx, strideX, Y, shiftY, ldy,
                                strideY, batch_count, scalars, work_workArr, Abyx_norms);

    // update the rest of the matrix
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   m - j - jb, n - j - jb, jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda,
                   strideA, Y, shiftY + jb, ldy, strideY, &one, A,
                   shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count,
                   (T**)work_workArr);

    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                   jb, &minone, X, shiftX + jb, ldx, strideX, A, shiftA + idx2D(j, j + jb, lda),
                   lda, strideA, &one, A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,
                   batch_count, (T**)work_workArr);

    blocks = (jb - 1) / 64 + 1;
    if(m >= n)
    {
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                dim3(1, 64, 1), 0, stream, D, j, strideD, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, jb);
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                dim3(1, 64, 1), 0, stream, E, j, strideE, A,
                                shiftA + idx2D(j, j + 1, lda), lda, strideA, jb);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                dim3(1, 64, 1), 0, stream, D, j, strideD, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, jb);
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, blocks, 1),
                                dim3(1, 64, 1), 0, stream, E, j, strideE, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, jb);
    }

    j += nb;
}
```

Explain what updates X and Y accumulate, why two GEMM calls are needed, and the mathematical relationship between LABRD's incremental updates and GEBRD's delayed bulk updates."""
                }
            ]
        }
    })

    # L2-2: LABRD Y Matrix Construction (CODING)
    entries.append({
        "custom_id": "roclapack_gebrd_L2_labrd_y_compute",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Implement the computation of column j of the Y matrix in rocsolver_labrd_template (upper bidiagonal case). After generating reflector H(j), Y[:, j] accumulates the effect of applying H(j) to the trailing columns.

The formula for Y[:, j] is:
Y[j+1:n, j] = tau_q[j] * (A[j:m, j+1:n]^H * A[j:m, j] - Y[j+1:n, 0:j] * (A[j:m, 0:j]^H * A[j:m, j]) - A[0:j, j+1:n]^H * (Y[0:j, j] = A[j:m, 0:j]^H * A[j:m, j]))

Requirements:
- Compute Y[j+1:n, j] = A[j:m, j+1:n]^H * A[j:m, j] (GEMV conjugate_transpose)
- Compute temp = A[j:m, 0:j]^H * A[j:m, j], then Y[j+1:n, j] -= Y[j+1:n, 0:j] * temp (GEMV + GEMV)
- Compute temp = A[j:m, 0:j]^H * A[j:m, j], then Y[j+1:n, j] -= A[0:j, j+1:n]^H * temp (GEMV + GEMV)
- Compute temp = X[j:m, 0:j]^H * A[j:m, j], then Y[j+1:n, j] -= A[0:j, j+1:n]^H * temp (GEMV + GEMV)
- Scale: Y[j+1:n, j] *= tau_q[j]

Code blocks for context:
```cpp
// From rocauxiliary_labrd.hpp lines 204-231 (computing Y column j)
if(j < n - 1)
{
    // compute column j of Y
    rocblasCall_gemv<T>(
        handle, rocblas_operation_conjugate_transpose, m - j, n - j - 1,
        cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, j + 1, lda), lda, strideA,
        A, shiftA + idx2D(j, j, lda), 1, strideA, cast2constType<T>(scalars + 1), 0, Y,
        shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count, (T**)work_workArr);
    rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j, j,
                        cast2constType<T>(scalars + 2), 0, A, shiftA + idx2D(j, 0, lda),
                        lda, strideA, A, shiftA + idx2D(j, j, lda), 1, strideA,
                        cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                        1, strideY, batch_count, (T**)work_workArr);
    rocblasCall_gemv<T>(
        handle, rocblas_operation_none, n - j - 1, j, cast2constType<T>(scalars), 0, Y,
        shiftY + idx2D(j + 1, 0, ldy), ldy, strideY, Y, shiftY + idx2D(0, j, ldy), 1,
        strideY, cast2constType<T>(scalars + 2), 0, Y, shiftY + idx2D(j + 1, j, ldy), 1,
        strideY, batch_count, (T**)work_workArr);
    rocblasCall_gemv<T>(handle, rocblas_operation_conjugate_transpose, m - j, j,
                        cast2constType<T>(scalars + 2), 0, X, shiftX + idx2D(j, 0, ldx),
                        ldx, strideX, A, shiftA + idx2D(j, j, lda), 1, strideA,
                        cast2constType<T>(scalars + 1), 0, Y, shiftY + idx2D(0, j, ldy),
                        1, strideY, batch_count, (T**)work_workArr);
    rocblasCall_gemv<T>(
        handle, rocblas_operation_conjugate_transpose, j, n - j - 1,
        cast2constType<T>(scalars), 0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA,
        Y, shiftY + idx2D(0, j, ldy), 1, strideY, cast2constType<T>(scalars + 2), 0, Y,
        shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count, (T**)work_workArr);
    rocblasCall_scal<T>(handle, n - j - 1, (tauq + j), strideQ, Y,
                        shiftY + idx2D(j + 1, j, ldy), 1, strideY, batch_count);
}
```

Provide the complete implementation with all GEMV calls and explain the mathematical purpose of each step."""
                }
            ]
        }
    })

    # L3-1: Complete GEBRD Execution Trace (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_gebrd_L3_trace",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Trace the complete execution of rocsolver_gebrd_template for a 512×768 matrix (upper bidiagonal) with GEBRD_GEBD2_SWITCHSIZE=64, GEBRD_BLOCKSIZE=32.

For each iteration:
1. Block index j and block size jb
2. LABRD call: dimensions, what gets reduced, X/Y sizes
3. First GEMM: dimensions, operation (A * Y^H update)
4. Second GEMM: dimensions, operation (X * A update)
5. Diagonal restoration: which diagonals (D and E)
6. Final GEBD2 call for remaining block

Include:
- Number of LABRD iterations (blocked loop)
- Dimensions of X and Y workspace matrices
- Total FLOP count breakdown (LABRD vs GEMM)
- Why j < dim - k (not j < dim)

Code blocks for context:
```cpp
// From roclapack_gebrd.hpp lines 136-204
rocblas_int nb = GEBRD_BLOCKSIZE;
rocblas_int k = GEBRD_GEBD2_SWITCHSIZE;
rocblas_int dim = std::min(m, n); // total number of pivots
rocblas_int jb, j = 0;
rocblas_int blocks;

// if the matrix is small, use the unblocked variant of the algorithm
if(m <= k || n <= k)
    return rocsolver_gebd2_template<T>(handle, m, n, A, shiftA, lda, strideA, D, strideD, E,
                                       strideE, tauq, strideQ, taup, strideP, batch_count,
                                       scalars, work_workArr, Abyx_norms);

rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

while(j < dim - k)
{
    // Reduce block to bidiagonal form
    jb = std::min(dim - j, nb); // number of rows and columns in the block
    rocsolver_labrd_template<T>(handle, m - j, n - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, D + j, strideD, E + j, strideE, tauq + j, strideQ,
                                taup + j, strideP, X, shiftX, ldx, strideX, Y, shiftY, ldy,
                                strideY, batch_count, scalars, work_workArr, Abyx_norms);

    // update the rest of the matrix
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   m - j - jb, n - j - jb, jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda,
                   strideA, Y, shiftY + jb, ldy, strideY, &one, A,
                   shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count,
                   (T**)work_workArr);

    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                   jb, &minone, X, shiftX + jb, ldx, strideX, A, shiftA + idx2D(j, j + jb, lda),
                   lda, strideA, &one, A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,
                   batch_count, (T**)work_workArr);

    j += nb;
}

// factor last block
if(j < dim)
    rocsolver_gebd2_template<T>(handle, m - j, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                D + j, strideD, E + j, strideE, tauq + j, strideQ, taup + j,
                                strideP, batch_count, scalars, work_workArr, Abyx_norms);
```

Provide detailed trace with dimensions and FLOP estimates for each step."""
                }
            ]
        }
    })

    # L3-2: Complete Production Wrapper with Memory Management (CODING)
    entries.append({
        "custom_id": "roclapack_gebrd_L3_wrapper",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": """Implement the complete rocsolver_gebrd_impl function and all four precision wrappers (sgebrd, dgebrd, cgebrd, zgebrd) for the production GEBRD API.

Requirements:
- Argument validation with rocsolver_gebd2_gebrd_argCheck
- Workspace size calculation with rocsolver_gebrd_getMemorySize
- Memory size query support
- Workspace allocation: scalars, work_workArr, Abyx_norms, X (m×k), Y (n×k)
- Scalar initialization
- Call rocsolver_gebrd_template<false, false, T>
- All four precision wrappers

Note: D and E are real-valued even for complex matrices (S* type for complex<S>)

Code blocks for context:
```cpp
// From roclapack_gebrd.cpp lines 32-106 (impl function)
template <typename T, typename S, typename U>
rocblas_status rocsolver_gebrd_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    S* D,
                                    S* E,
                                    T* tauq,
                                    T* taup)
{
    ROCSOLVER_ENTER_TOP("gebrd", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gebd2_gebrd_argCheck(handle, m, n, lda, A, D, E, tauq, taup);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftX = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideQ = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideY = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling GEDB2 and LABRD
    size_t size_Abyx_norms;
    // size for temporary resulting orthogonal matrices when calling LABRD
    size_t size_X;
    size_t size_Y;
    rocsolver_gebrd_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_X, &size_Y);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_X, size_Y);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *X, *Y;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_X,
                              size_Y);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    X = mem[3];
    Y = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gebrd_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        (T*)X, shiftX, m, strideX, (T*)Y, shiftY, n, strideY, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms);
}
```

```cpp
// From roclapack_gebrd.cpp lines 116-172 (C wrappers)
extern "C" {

rocblas_status rocsolver_sgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tauq,
                                float* taup)
{
    return rocsolver::rocsolver_gebrd_impl<float>(handle, m, n, A, lda, D, E, tauq, taup);
}

rocblas_status rocsolver_dgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tauq,
                                double* taup)
{
    return rocsolver::rocsolver_gebrd_impl<double>(handle, m, n, A, lda, D, E, tauq, taup);
}

rocblas_status rocsolver_cgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tauq,
                                rocblas_float_complex* taup)
{
    return rocsolver::rocsolver_gebrd_impl<rocblas_float_complex>(handle, m, n, A, lda, D, E, tauq,
                                                                  taup);
}

rocblas_status rocsolver_zgebrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tauq,
                                rocblas_double_complex* taup)
{
    return rocsolver::rocsolver_gebrd_impl<rocblas_double_complex>(handle, m, n, A, lda, D, E, tauq,
                                                                   taup);
}

}
```

Provide complete implementation of the impl function and all four wrappers."""
                }
            ]
        }
    })

    return entries

def main():
    entries = generate_entries()

    output_file = Path(__file__).parent / "roclapack_gebrd.jsonl"

    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(entries)} entries")
    print(f"Output: {output_file}")

    # Statistics
    coding_count = sum(1 for e in entries if 'implement' in e['body']['messages'][0]['content'].lower()
                       or 'write' in e['body']['messages'][0]['content'].lower()
                       or 'provide the complete' in e['body']['messages'][0]['content'].lower())
    print(f"Coding tasks: {coding_count}/{len(entries)} ({coding_count/len(entries)*100:.1f}%)")

if __name__ == "__main__":
    main()
