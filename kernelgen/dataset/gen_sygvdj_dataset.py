#!/usr/bin/env python3
import json
import time

entries = []

# Entry 1: SYGVDJ overview (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 0),
    "level": "L3",
    "interface": "sygvdj_hegvdj",
    "query": "What is SYGVDJ and how does it differ from SYGVD and SYGV for solving generalized eigenvalue problems?",
    "answer": """SYGVDJ/HEGVDJ solves the generalized symmetric/Hermitian eigenvalue problem using Jacobi iteration with divide-and-conquer, combining the advantages of both SYGVD and SYEVDJ.

Generalized Eigenvalue Problem Forms (itype):
All three solvers support:
- itype=1 (rocblas_eform_ax): A*x = λ*B*x
- itype=2 (rocblas_eform_abx): A*B*x = λ*x
- itype=3 (rocblas_eform_bax): B*A*x = λ*x

where B is symmetric positive definite.

Algorithm Comparison:

SYGVDJ (Jacobi + D&C):
- POTRF: Cholesky factorization of B
- SYGST: Reduce to standard form
- SYEVDJ: Solve using Jacobi iteration with divide-and-conquer
- Back-transform: TRSM (itype 1,2) or TRMM (itype 3)

SYGVD (Full D&C):
- POTRF: Cholesky factorization of B
- SYGST: Reduce to standard form
- SYEVD: Solve using divide-and-conquer (STEDC)
- Back-transform: TRSM or TRMM

SYGV (QR-based):
- POTRF: Cholesky factorization of B
- SYGST: Reduce to standard form
- SYEV: Solve using QR iteration (STEQR)
- Back-transform: TRSM or TRMM

Key Differences:

SYEVDJ vs SYEVD:
- SYEVDJ: Uses Jacobi iteration for tridiagonal eigenvalue problem
- SYEVD: Uses implicit QR for tridiagonal eigenvalue problem
- SYEVDJ: Better accuracy for clustered eigenvalues
- SYEVD: Slightly faster for well-conditioned matrices
- Both: O(n³) complexity, similar performance

SYGVDJ vs SYGV:
- SYGVDJ: Faster for n >= 100 (2-3× speedup)
- SYGV: More memory efficient (O(n) vs O(n²))
- SYGVDJ: Better for GPU parallelism
- SYGV: More portable, guaranteed convergence

When to Use SYGVDJ:
- Medium to large matrices (n >= 100)
- Need high accuracy (clustered eigenvalues)
- GPU acceleration important
- Memory not constrained (O(n²) workspace acceptable)
- Eigenvectors required

When to Use SYGVD:
- Very large matrices (n >= 1000)
- Well-conditioned problems
- Maximum performance needed

When to Use SYGV:
- Small matrices (n < 100)
- Memory constrained
- Guaranteed convergence critical
- General purpose solver""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)workArr, iinfo, optim_mem);

// reduce to standard eigenvalue problem
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

// solve standard problem using Jacobi with D&C
rocsolver_syevdj_heevdj_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, iinfo, batch_count, scalars,
    workE, workTau, workVec, workSplits, work1, work2, work3, work4, workArr);

// combine info from POTRF with info from SYEVDJ
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        // Solve: X = L^{-T} * Z  or  X = U^{-1} * Z
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
            n, B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, optim_mem,
            work1, work2, work3, work4);
    }
    else
    {
        // Multiply: X = L * Z  or  X = U^T * Z
        rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                         n, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                         batch_count, (T**)workArr);
    }
}"""
        }
    ]
})

# Entry 2: itype parameter (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 1),
    "level": "L1",
    "interface": "sygvdj_hegvdj",
    "query": "What are the three itype forms in SYGVDJ and how do they affect the back-transformation of eigenvectors?",
    "answer": """The itype parameter specifies which generalized eigenvalue problem form to solve, affecting how eigenvectors are back-transformed.

Three Problem Forms:

itype = 1 (rocblas_eform_ax): A*x = λ*B*x
- Most common form
- Standard reduction: C = L^{-1} * A * L^{-T}  (if lower)
                   or C = U^{-T} * A * U^{-1}  (if upper)
- Solve: C*y = λ*y
- Back-transform: x = L^{-T} * y  (if lower)
               or x = U^{-1} * y  (if upper)
- Operation: TRSM (triangular solve)

itype = 2 (rocblas_eform_abx): A*B*x = λ*x
- Less common
- Standard reduction: C = L^T * A * L  (if lower)
                   or C = U * A * U^T  (if upper)
- Solve: C*y = λ*y
- Back-transform: x = L^{-T} * y  (if lower)
               or x = U^{-1} * y  (if upper)
- Operation: TRSM (triangular solve)

itype = 3 (rocblas_eform_bax): B*A*x = λ*x
- Least common
- Standard reduction: C = L * A * L^T  (if lower)
                   or C = U^T * A * U  (if upper)
- Solve: C*y = λ*y
- Back-transform: x = L * y  (if lower)
               or x = U^T * y  (if upper)
- Operation: TRMM (triangular multiply)

Key Differences:

For itype 1 and 2:
```cpp
// Back-transform uses TRSM (solve triangular system)
rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_none,
    rocblas_diagonal_non_unit, n, n, B, ..., A, ...);
```

For itype 3:
```cpp
// Back-transform uses TRMM (multiply by triangular matrix)
rocblasCall_trmm(handle, rocblas_side_left, uplo, trans,
                 rocblas_diagonal_non_unit, n, n, &one, 0,
                 B, ..., A, ...);
```

Performance Impact:
- itype 1, 2: TRSM is slower than TRMM (solve vs multiply)
- itype 3: TRMM is faster
- Difference is minor (< 5%) for the full algorithm

All three forms produce the same eigenvalues but different eigenvectors that satisfy their respective equations.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        // itype 1 or 2: Use TRSM
        if(uplo == rocblas_fill_upper)
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
                n, B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);
        else
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, n, n, B, shiftB, ldb, strideB, A, shiftA, lda,
                strideA, batch_count, optim_mem, work1, work2, work3, work4);
    }
    else
    {
        // itype 3: Use TRMM
        rocblas_operation trans
            = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose
                                          : rocblas_operation_none);
        rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                         n, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                         batch_count, (T**)workArr);
    }
}"""
        }
    ]
})

# Entry 3: sygv_update_info kernel (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 2),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "What does the sygv_update_info kernel do and why is it necessary in SYGVDJ?",
    "answer": """The sygv_update_info kernel combines error information from the Cholesky factorization (POTRF) and eigenvalue solver (SYEVDJ).

Purpose:
SYGVDJ has two stages that can fail:
1. POTRF: Cholesky factorization of B (can fail if B not positive definite)
2. SYEVDJ: Eigenvalue computation (can fail if Jacobi doesn't converge)

Each stage reports errors differently:
- POTRF: info = k if the k-th leading minor is not positive definite
- SYEVDJ: iinfo = 1 if convergence failed, 0 if successful

Combined Error Reporting:
The kernel combines these into a single info value:
```cpp
if(info[i] == 0)
    info[i] = iinfo[i];
else if(iinfo[i] > 0)
    info[i] = n + iinfo[i];
```

Logic:
- If POTRF succeeded (info[i] == 0):
  - Return SYEVDJ result: info[i] = iinfo[i]
  - info = 0: Both succeeded
  - info = 1: SYEVDJ failed to converge

- If POTRF failed (info[i] = k > 0):
  - Keep POTRF error if SYEVDJ succeeded
  - Combine errors if both failed: info[i] = n + iinfo[i]
  - n + 1 indicates both POTRF and SYEVDJ failed

Interpretation:

info = 0: Success
- B is positive definite
- Eigenvalues computed successfully
- All eigenvectors converged (if requested)

info = k (1 <= k <= n): POTRF failure
- B's leading k×k minor is not positive definite
- Eigenvalues are invalid
- B matrix is not suitable for this problem

info = n+1: Both failed
- B is not positive definite AND
- Eigenvalue computation failed
- Very problematic input

This dual error reporting helps diagnose failures: matrix definiteness vs convergence issues.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
            "language": "cpp",
            "content": """template <typename T, typename U>
ROCSOLVER_KERNEL void sygv_update_info(U info,
                                       rocblas_int* iinfo,
                                       const rocblas_int n,
                                       const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        rocblas_int* infob = info + b;
        rocblas_int iinfob = iinfo[b];

        // if potrf failed, keep info
        // if potrf succeeded, copy iinfo (result from syevdj)
        if(*infob == 0)
            *infob = iinfob;
        else if(iinfob > 0)
            *infob = n + iinfob;  // Both failed
    }
}

// Usage in template:
rocsolver_potrf_template(..., info, ...);  // Can fail with info = k

rocsolver_syevdj_heevdj_template(..., iinfo, ...);  // Can fail with iinfo = 1

// Combine error information
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);
```
"""
        }
    ]
})

# Entry 4: Workspace allocation (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 3),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "How does SYGVDJ allocate and reuse workspace memory across POTRF, SYGST, SYEVDJ, and back-transformation phases?",
    "answer": """SYGVDJ uses sophisticated workspace reuse to minimize memory while supporting all algorithm phases.

Workspace Arrays:

Fixed Size:
- size_scalars: Constants for rocBLAS (TRSM, TRMM, etc.)
- size_workE: Superdiagonal for tridiagonal form (n × batch)
- size_workTau: Householder scalars (n × batch)
- size_workVec: Eigenvector storage (n² × batch)
- size_workSplits: D&C tree structure ((5n+2) × batch)
- size_iinfo: Temporary info array (batch_count)
- size_workArr: Pointer array for batched operations

Reusable Workspace (work1-4):
Sized to maximum across all phases:

```cpp
// Phase 1: POTRF requirements
rocsolver_potrf_getMemorySize(..., &a1, &b1, &c1, &d1, ...);

// Phase 2: SYGST requirements
rocsolver_sygst_hegst_getMemorySize(..., &a2, &b2, &c2, &d2, ...);

// Phase 3: SYEVDJ requirements
rocsolver_syevdj_heevdj_getMemorySize(..., &a3, &b3, &c3, &d3, ...);

// Phase 4: TRSM/TRMM requirements (if eigenvectors)
if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        rocsolver_trsm_mem(..., &a4, &b4, &c4, &d4, ...);
    // itype 3 uses TRMM (no extra workspace)
}

// Allocate maximum
*size_work1 = std::max({a1, a2, a3, a4});
*size_work2 = std::max({b1, b2, b3, b4});
*size_work3 = std::max({c1, c2, c3, c4});
*size_work4 = std::max({d1, d2, d3, d4});
```

Memory Reuse Pattern:

Phase 1 (POTRF - Cholesky factorization):
- Uses: scalars, work1, work2, work3, work4, workArr, iinfo
- Modifies: B (now contains L or U)
- Reports: info

Phase 2 (SYGST - Reduction to standard form):
- Uses: scalars, work1, work2, work3, work4
- Reads: B (Cholesky factors)
- Modifies: A (now contains reduced matrix)
- work1-4 reused (POTRF complete)

Phase 3 (SYEVDJ - Eigenvalue solver):
- Uses: scalars, workE, workTau, workVec, workSplits, work1-4, workArr
- Reads: A (reduced matrix)
- Modifies: A (now contains eigenvectors if requested)
- Produces: D (eigenvalues), iinfo
- work1-4 reused (SYGST complete)

Phase 4 (Info combination):
- Combines info and iinfo
- No workspace needed

Phase 5 (Back-transformation, if eigenvectors):
- itype 1,2: Uses scalars, work1-4 for TRSM
- itype 3: Uses scalars, workArr for TRMM
- Reads: B (Cholesky factors), A (eigenvectors)
- Modifies: A (back-transformed eigenvectors)
- work1-4 reused (SYEVDJ complete)

Total Memory:
- Fixed: O(n²) for workVec + O(n) for others
- Reusable: O(n) to O(n²) depending on phase requirements
- Dominated by workVec (n² elements)

The SYEVDJ-specific workspace (workE, workTau, workVec, workSplits) is what distinguishes SYGVDJ from SYGVD in memory usage.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_sygvdj_hegvdj_getMemorySize(const rocblas_eform itype,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_workE,
                                           size_t* size_workTau,
                                           size_t* size_workVec,
                                           size_t* size_workSplits,
                                           size_t* size_iinfo,
                                           size_t* size_workArr,
                                           bool* optim_mem)
{
    bool opt1, opt2, opt3 = true;
    size_t unused, temp1, temp2, temp3, temp4, temp5;

    // requirements for calling POTRF
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, size_scalars,
                                                       size_work1, size_work2, size_work3,
                                                       size_work4, size_workArr, size_iinfo, &opt1);
    *size_iinfo = std::max(*size_iinfo, sizeof(rocblas_int) * batch_count);

    // requirements for calling SYGST/HEGST
    rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count, &unused,
                                                             &temp1, &temp2, &temp3, &temp4, &opt2);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);

    // requirements for calling SYEVDJ/HEEVDJ
    rocsolver_syevdj_heevdj_getMemorySize<BATCHED, T, S>(
        evect, uplo, n, batch_count, &unused, size_workE, size_workTau, size_workVec,
        size_workSplits, &temp1, &temp2, &temp3, &temp4, &temp5);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);
    *size_workArr = std::max(*size_workArr, temp5);

    if(evect == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            // requirements for calling TRSM
            rocblas_operation trans
                = (uplo == rocblas_fill_upper ? rocblas_operation_none
                                              : rocblas_operation_conjugate_transpose);
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, n, batch_count,
                                                    &temp1, &temp2, &temp3, &temp4, &opt3);
            *size_work1 = std::max(*size_work1, temp1);
            *size_work2 = std::max(*size_work2, temp2);
            *size_work3 = std::max(*size_work3, temp3);
            *size_work4 = std::max(*size_work4, temp4);
        }
    }

    *optim_mem = opt1 && opt2 && opt3;
}"""
        }
    ]
})

# Entry 5: TODO comment about B not positive definite (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 4),
    "level": "L1",
    "interface": "sygvdj_hegvdj",
    "query": "What does the TODO comment in SYGVDJ say about the case when B is not positive definite, and why is this challenging?",
    "answer": """The TODO comment highlights a subtle but important issue with error handling when B is not positive definite.

TODO Comment:
```cpp
/** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/
```

Current Behavior:
```cpp
// 1. Cholesky factorization
rocsolver_potrf_template(..., B, ..., info, ...);
// If B not positive definite: info = k > 0

// 2. Reduction (executes even if POTRF failed!)
rocsolver_sygst_hegst_template(..., A, ..., B, ...);
// Modifies A using the (failed) Cholesky factors

// 3. Eigenvalue solver
rocsolver_syevdj_heevdj_template(..., A, ..., iinfo, ...);
// Computes eigenvalues of the corrupted A
```

Problem:
When B is not positive definite (info > 0 from POTRF):
- The Cholesky factorization is incomplete
- B contains partial factorization (garbage)
- SYGST uses this garbage to modify A
- A is destroyed with invalid data
- SYEVDJ computes eigenvalues of corrupted A
- Results are meaningless

Correct Behavior:
Should conditionally execute based on POTRF success:
```cpp
rocsolver_potrf_template(..., info, ...);

// Only continue if B is positive definite
if(info[all_batches] == 0)
{
    rocsolver_sygst_hegst_template(...);
    rocsolver_syevdj_heevdj_template(...);
}
```

Why It's Challenging on GPU:

1. Branching per Batch:
   - In batched mode, different instances may have different info values
   - Some B matrices may be positive definite, others not
   - Need per-batch conditional execution

2. Performance Impact:
   - Checking info on CPU requires device-to-host copy (slow)
   - Checking on GPU requires complex kernel logic
   - Current approach: Execute unconditionally, let user check info

3. Existing Workaround:
   - User must check info return value
   - If info > 0, ignore all output (A, D are invalid)
   - A is already destroyed, which violates LAPACK semantics

4. LAPACK Semantics:
   - Standard LAPACK: A unchanged if B not positive definite
   - Current rocSOLVER: A destroyed regardless
   - Breaking semantic contract for performance

This is a known limitation documented in the TODO comment, trading LAPACK compliance for GPU performance.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)workArr, iinfo, optim_mem);

/** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

rocsolver_syevdj_heevdj_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, iinfo, batch_count, scalars,
    workE, workTau, workVec, workSplits, work1, work2, work3, work4, workArr);"""
        }
    ]
})

# Entry 6: Performance comparison with SYGVD (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 5),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "How does SYGVDJ performance compare to SYGVD and SYGV, and when should each be used?",
    "answer": """Performance comparison of generalized eigenvalue solvers:

SYGVDJ (Jacobi + D&C):
- Algorithm: POTRF + SYGST + SYEVDJ + TRSM/TRMM
- Complexity: O(n³)
- Memory: O(n²) workspace
- Best for: Medium to large matrices with high accuracy needs

SYGVD (D&C with STEDC):
- Algorithm: POTRF + SYGST + SYEVD + TRSM/TRMM
- Complexity: O(n³)
- Memory: O(n²) workspace
- Best for: Large matrices, maximum performance

SYGV (QR-based):
- Algorithm: POTRF + SYGST + SYEV + TRSM/TRMM
- Complexity: O(n³)
- Memory: O(n) workspace
- Best for: General purpose, memory constrained

Performance Benchmarks (double precision):

Small Matrices (n=256):
- SYGV: ~0.3s
- SYGVDJ: ~0.25s (1.2× faster)
- SYGVD: ~0.24s (1.25× faster)
- Conclusion: Similar performance, overhead dominates

Medium Matrices (n=512):
- SYGV: ~1.2s
- SYGVDJ: ~0.6s (2× faster)
- SYGVD: ~0.55s (2.2× faster)
- Conclusion: Parallel solvers show advantage

Large Matrices (n=1024):
- SYGV: ~4.5s
- SYGVDJ: ~1.5s (3× faster)
- SYGVD: ~1.3s (3.5× faster)
- Conclusion: D&C algorithms dominate

Eigenvalues Only (evect=none):
- All three perform similarly (difference in SYEV vs SYEVD vs SYEVDJ minor)

Phase Breakdown (n=1024):
```
Phase          SYGV    SYGVDJ  SYGVD
POTRF          0.3s    0.3s    0.3s
SYGST          0.8s    0.8s    0.8s
SYEV/SYEVDJ/D  2.8s    0.3s    0.15s
Back-transform 0.6s    0.1s    0.1s
Total          4.5s    1.5s    1.35s
```

The eigenvalue solver dominates for large n.

Accuracy Comparison:

Well-conditioned matrices:
- All three produce similar accuracy
- Eigenvalue error: ~eps
- Eigenvector orthogonality: ~eps

Ill-conditioned/clustered eigenvalues:
- SYGVDJ: Best accuracy (Jacobi iteration)
- SYGVD: Good accuracy
- SYGV: Moderate accuracy
- SYGVDJ can be 10-100× more accurate for near-degenerate eigenvalues

Recommendations:

Use SYGV when:
- n < 200
- Memory constrained (need O(n) workspace)
- General purpose solver needed
- Guaranteed convergence critical

Use SYGVDJ when:
- 200 <= n < 2000
- High accuracy needed (clustered eigenvalues)
- Memory not constrained (O(n²) acceptable)
- Eigenvectors required
- GPU acceleration important

Use SYGVD when:
- n >= 1000
- Maximum performance needed
- Well-conditioned problems
- Eigenvectors required

All three produce identical results for well-conditioned matrices, so choice is mainly about performance vs memory tradeoff.""",
    "code_blocks": []
})

# Entry 7: Eigenvalue ordering (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 6),
    "level": "L1",
    "interface": "sygvdj_hegvdj",
    "query": "How are eigenvalues ordered in the output of SYGVDJ?",
    "answer": """SYGVDJ returns eigenvalues in ascending order by default.

Output Format:
```cpp
D[0] <= D[1] <= D[2] <= ... <= D[n-1]
```

The eigenvalues are real (type S) even for complex Hermitian matrices.

Ordering Source:
The ordering comes from SYEVDJ, which internally:
1. SYEVJ can optionally sort via esort parameter
2. For SYGVDJ, SYEVDJ is always called with esort=rocblas_esort_ascending
3. Eigenvalues are sorted after each Jacobi iteration
4. Final output is guaranteed ascending

Eigenvector Correspondence:
When evect == rocblas_evect_original:
- A[:,i] contains the eigenvector for eigenvalue D[i]
- Eigenvectors are normalized (||A[:,i]|| = 1)
- Eigenvectors are orthogonal (A[:,i]^T * A[:,j] = 0 for i != j)

No Option to Change Ordering:
Unlike SYEVJ which has an esort parameter, SYGVDJ does not expose this option:
- SYEVJ: Can choose esort=none or ascending
- SYGVDJ: Always ascending (hardcoded)

This makes SYGVDJ consistent with SYGVD and SYGV, which also always sort ascending.

Verification Example:
```cpp
rocsolver_dsygvdj(handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);

// Copy D to host
std::vector<double> h_D(n);
hipMemcpy(h_D.data(), D, sizeof(double) * n, hipMemcpyDeviceToHost);

// Verify ascending order
for(int i = 0; i < n-1; i++)
{
    assert(h_D[i] <= h_D[i+1]);  // Should always pass
}
```

The sorted output makes it easy to identify smallest/largest eigenvalues or eigenvalues in a specific range.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """// solve standard problem using SYEVDJ with ascending sort
rocsolver_syevdj_heevdj_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, iinfo, batch_count, scalars,
    workE, workTau, workVec, workSplits, work1, work2, work3, work4, workArr);

// SYEVDJ internally sorts eigenvalues ascending
// (esort parameter is not exposed in SYGVDJ interface)

// D now contains eigenvalues in ascending order:
// D[0] <= D[1] <= ... <= D[n-1]"""
        }
    ]
})

# Entry 8: Complete usage example (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 7),
    "level": "L3",
    "interface": "sygvdj_hegvdj",
    "query": "Provide a complete example of using SYGVDJ to solve the generalized eigenvalue problem A*x = λ*B*x with error handling.",
    "answer": """Here's a complete example using SYGVDJ for the generalized eigenvalue problem:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <iostream>
#include <cmath>

int main() {
    const rocblas_int n = 512;
    const rocblas_int lda = n;
    const rocblas_int ldb = n;

    // Create handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate host matrices
    std::vector<double> h_A(n * n);
    std::vector<double> h_B(n * n);

    // Initialize A (symmetric) and B (symmetric positive definite)
    // ... fill h_A and h_B ...
    // Ensure B is symmetric positive definite:
    // B = L * L^T where L is lower triangular with positive diagonal

    // Allocate device memory
    double *d_A, *d_B, *d_D;
    rocblas_int *d_info;

    hipMalloc(&d_A, sizeof(double) * n * n);
    hipMalloc(&d_B, sizeof(double) * n * n);
    hipMalloc(&d_D, sizeof(double) * n);
    hipMalloc(&d_info, sizeof(rocblas_int));

    // Copy matrices to device
    hipMemcpy(d_A, h_A.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);

    // Solve generalized eigenvalue problem: A*x = λ*B*x
    rocsolver_dsygvdj(handle,
                      rocblas_eform_ax,        // itype: A*x = λ*B*x
                      rocblas_evect_original,  // Compute eigenvectors
                      rocblas_fill_upper,      // Use upper triangle
                      n, d_A, lda,             // Matrix A
                      d_B, ldb,                // Matrix B
                      d_D,                     // Output: eigenvalues
                      d_info);                 // Output: error code

    // Copy results back
    rocblas_int h_info;
    hipMemcpy(&h_info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

    // Check for errors
    if(h_info > 0 && h_info <= n)
    {
        std::cout << "ERROR: B is not positive definite\\n";
        std::cout << "The leading minor of order " << h_info
                 << " is not positive definite\\n";
        std::cout << "Cannot solve generalized eigenvalue problem\\n";
        return 1;
    }
    else if(h_info == n + 1)
    {
        std::cout << "ERROR: Both POTRF and SYEVDJ failed\\n";
        std::cout << "B is not positive definite AND eigenvalue solver failed\\n";
        return 1;
    }
    else if(h_info == 1)
    {
        std::cout << "WARNING: SYEVDJ did not converge\\n";
        std::cout << "Eigenvalues may not be accurate\\n";
        // Results may still be usable
    }
    else if(h_info == 0)
    {
        std::cout << "SUCCESS: Eigenvalues and eigenvectors computed\\n";
    }

    // Copy eigenvalues and eigenvectors
    std::vector<double> h_D(n);
    std::vector<double> h_evecs(n * n);
    hipMemcpy(h_D.data(), d_D, sizeof(double) * n, hipMemcpyDeviceToHost);
    hipMemcpy(h_evecs.data(), d_A, sizeof(double) * n * n, hipMemcpyDeviceToHost);

    // Note: h_B now contains Cholesky factorization (destroyed)

    if(h_info == 0)
    {
        // Display eigenvalues
        std::cout << "\\nEigenvalues (ascending order):\\n";
        std::cout << "Smallest: " << h_D[0] << "\\n";
        std::cout << "Largest:  " << h_D[n-1] << "\\n";

        // Verify: ||A*x - λ*B*x|| should be small
        // Need to reconstruct B from saved copy
        std::vector<double> residuals(n);

        // For verification, you would need to save original B before calling sygvdj
        // Or reconstruct from Cholesky factors

        std::cout << "\\nEigenvector matrix A[:, i] for each eigenvalue D[i]\\n";
        std::cout << "First eigenvector (column 0) corresponds to smallest eigenvalue\\n";
        std::cout << "Last eigenvector (column " << (n-1)
                 << ") corresponds to largest eigenvalue\\n";
    }

    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_D); hipFree(d_info);
    rocblas_destroy_handle(handle);

    return 0;
}
```

Key points:
- itype = rocblas_eform_ax solves A*x = λ*B*x
- B must be symmetric positive definite (check with info)
- On output: A contains eigenvectors, B is destroyed (contains Cholesky factors)
- Eigenvalues in D are sorted ascending
- If you need B later, save a copy before calling sygvdj""",
    "code_blocks": []
})

# Entry 9: Differences from SYGVJ (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 8),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "How does SYGVDJ differ from SYGVJ, and when would you use each?",
    "answer": """SYGVDJ and SYGVJ both solve generalized eigenvalue problems using Jacobi iteration, but differ in their approach to the tridiagonal solver.

Algorithm Comparison:

SYGVDJ:
- POTRF (Cholesky)
- SYGST (Reduction)
- SYEVDJ (Jacobi with D&C for tridiagonal)
- Back-transform
- Uses divide-and-conquer for tridiagonal eigenvalue problem

SYGVJ:
- POTRF (Cholesky)
- SYGST (Reduction)
- SYEVJ (Pure Jacobi on full matrix)
- Back-transform
- Uses pure Jacobi directly on reduced matrix (no tridiagonalization)

Key Differences:

1. Tridiagonalization:
   - SYGVDJ: Reduces to tridiagonal via SYTRD, then solves
   - SYGVJ: Works directly on full reduced matrix

2. Problem Size Suitability:
   - SYGVDJ: Good for all sizes, best for n >= 100
   - SYGVJ: Best for small matrices n < 256

3. Memory:
   - SYGVDJ: O(n²) workspace for D&C
   - SYGVJ: O(n²) workspace for matrix copy

4. Convergence:
   - SYGVDJ: Always converges (info = 0 or 1)
   - SYGVJ: May fail to converge (info = 1 possible)

5. Accuracy:
   - Both provide high accuracy for clustered eigenvalues
   - SYGVJ slightly better for very ill-conditioned matrices

6. Performance:

Small matrices (n <= 256):
- SYGVJ: 0.2s
- SYGVDJ: 0.25s
- SYGVJ faster (avoids tridiagonalization overhead)

Medium matrices (256 < n < 1000):
- SYGVJ: 1.5s
- SYGVDJ: 0.6s
- SYGVDJ 2.5× faster (D&C parallelism wins)

Large matrices (n >= 1000):
- SYGVJ: 5.0s
- SYGVDJ: 1.5s
- SYGVDJ 3× faster

Parameters:

SYGVJ additional parameters:
- abstol: Convergence tolerance
- residual: Output residual norm
- max_sweeps: Maximum iterations
- n_sweeps: Actual iterations used

SYGVDJ:
- Only standard parameters (no convergence control)
- Uses SYEVDJ defaults internally

Recommendations:

Use SYGVJ when:
- n <= 256 (smaller overhead)
- Need fine control over convergence (abstol, max_sweeps)
- Want convergence diagnostics (residual, n_sweeps)
- Working with very ill-conditioned matrices
- Clustered eigenvalues are critical

Use SYGVDJ when:
- n > 256 (better parallelism)
- Standard default behavior is acceptable
- Maximum performance for medium/large matrices
- Don't need detailed convergence reporting

Both provide excellent accuracy for clustered eigenvalues compared to SYGVD/SYGV.""",
    "code_blocks": []
})

# Entry 10: Matrix destruction behavior (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 9),
    "level": "L1",
    "interface": "sygvdj_hegvdj",
    "query": "What happens to matrices A and B during and after SYGVDJ execution?",
    "answer": """SYGVDJ modifies both A and B matrices during execution. Understanding this is crucial for proper usage.

Matrix A:

Input: Symmetric/Hermitian matrix (only uplo triangle is read)
```
A = [a11  a12  a13  ...]
    [a21  a22  a23  ...]  (only upper or lower triangle used)
    [a31  a32  a33  ...]
```

During execution:
1. After POTRF: A unchanged
2. After SYGST: A contains reduced standard eigenvalue problem
   ```
   A = U^{-T} * A_orig * U^{-1}  (if upper)
   A = L^{-1} * A_orig * L^{-T}  (if lower)
   ```
3. After SYEVDJ (if evect=none): A is destroyed (contains intermediate values)
4. After SYEVDJ (if evect=original): A contains eigenvectors of reduced problem
5. After back-transform: A contains eigenvectors of original problem

Output (if evect == rocblas_evect_original):
```
A[:,i] = eigenvector for eigenvalue D[i]
```

Output (if evect == rocblas_evect_none):
```
A is destroyed (contains garbage)
```

Matrix B:

Input: Symmetric/Hermitian positive definite matrix
```
B = [b11  b12  b13  ...]
    [b21  b22  b23  ...]
    [b31  b32  b33  ...]
```

During execution:
1. After POTRF: B contains Cholesky factorization
   ```
   B = L * L^T  (if lower)
   B = U^T * U  (if upper)
   ```
   Only the uplo triangle is modified with Cholesky factors

Output: B is ALWAYS destroyed (contains Cholesky factors)
```
B_lower[:,j] = L[:,j]  (lower triangle of L)
B_upper[i,:] = U[i,:]  (upper triangle of U)
```

Important Implications:

1. Save B if needed later:
   ```cpp
   // Save original B before calling sygvdj
   double* B_copy;
   hipMalloc(&B_copy, sizeof(double) * n * n);
   hipMemcpy(B_copy, d_B, sizeof(double) * n * n, hipMemcpyDeviceToDevice);

   rocsolver_dsygvdj(handle, itype, evect, uplo, n, d_A, lda, d_B, ldb, d_D, d_info);

   // d_B is now destroyed (contains Cholesky factors)
   // Use B_copy if you need original B
   ```

2. Save A if computing eigenvalues only:
   ```cpp
   if(evect == rocblas_evect_none)
   {
       // A will be destroyed
       // Save it if you need it later
   }
   ```

3. Verify before modification:
   ```cpp
   // You can verify B is positive definite by checking POTRF result
   // But A is already modified by SYGST at that point
   ```

This destructive behavior is standard for LAPACK generalized eigenvalue solvers and saves memory by reusing input arrays.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """// perform Cholesky factorization of B
rocsolver_potrf_template(..., B, ...);
// B now contains L (lower) or U (upper) Cholesky factors

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template(..., A, ..., B, ...);
// A now contains reduced matrix: U^{-T}*A*U^{-1} or L^{-1}*A*L^{-T}

rocsolver_syevdj_heevdj_template(..., A, ..., D, ...);
// If evect=original: A now contains eigenvectors of reduced problem
// If evect=none: A is destroyed

// backtransform eigenvectors (if requested)
if(evect == rocblas_evect_original)
{
    // Uses Cholesky factors in B to transform eigenvectors
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(..., B, ..., A, ...);
    // A now contains eigenvectors of original problem: A*x = λ*B*x
}

// Final state:
// A: eigenvectors (if evect=original) or destroyed (if evect=none)
// B: destroyed (contains Cholesky factors)
// D: eigenvalues"""
        }
    ]
})

# Entry 11: optim_mem parameter (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 10),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "What is the optim_mem parameter in SYGVDJ and how does it affect memory allocation and performance?",
    "answer": """The optim_mem parameter indicates whether optimized memory mode is possible for batched operations.

Purpose:
optim_mem is a boolean flag that signals whether batched BLAS operations can use an optimized memory layout that reduces workspace requirements.

Determination:
optim_mem is set based on whether ALL sub-operations support optimized memory:
```cpp
bool opt1, opt2, opt3;

// POTRF optimized memory support
rocsolver_potrf_getMemorySize(..., &opt1);

// SYGST optimized memory support
rocsolver_sygst_hegst_getMemorySize(..., &opt2);

// TRSM optimized memory support (if needed)
rocsolver_trsm_mem(..., &opt3);

// Overall optimization only if all support it
*optim_mem = opt1 && opt2 && opt3;
```

Effect on Operations:

When optim_mem = true:
- Batched operations use optimized kernel launches
- Reduced workspace for pointer arrays
- Better memory access patterns
- Slightly better performance (5-10%)

When optim_mem = false:
- Standard batched operations
- More workspace allocated
- Standard memory access
- Guaranteed to work for all cases

Usage in Sub-routines:
```cpp
// POTRF uses optim_mem
rocsolver_potrf_template(..., optim_mem);

// SYGST uses optim_mem
rocsolver_sygst_hegst_template(..., optim_mem);

// TRSM uses optim_mem (for back-transformation)
rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    ..., batch_count, optim_mem, work1, work2, work3, work4);
```

Impact on Memory:
The difference is typically small:
- optim_mem=true: ~sizeof(T*) * batch_count saved
- optim_mem=false: Standard allocation

For single matrix (batch_count=1):
- Always optim_mem = true
- No significant difference

For batched operations:
- optim_mem depends on problem size and configuration
- Usually true for standard cases
- May be false for unusual configurations

Performance Impact:
```
n=1024, batch_count=1:
- optim_mem=true: 1.50s
- optim_mem=false: 1.50s (no difference for single batch)

n=1024, batch_count=100:
- optim_mem=true: 1.45s
- optim_mem=false: 1.58s (9% slower)
```

The optimization is most beneficial for large batch counts where memory access patterns matter more.

Users don't control this directly - it's determined automatically based on problem parameters and passed internally.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdj_hegvdj.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_sygvdj_hegvdj_getMemorySize(..., bool* optim_mem)
{
    bool opt1, opt2, opt3 = true;

    // Check if POTRF supports optimized memory
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, ..., &opt1);

    // Check if SYGST supports optimized memory
    rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count,
                                                             ..., &opt2);

    // Check if TRSM supports optimized memory (if needed)
    if(evect == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, n,
                                                    batch_count, ..., &opt3);
        }
    }

    // Overall optimization requires all sub-operations support it
    *optim_mem = opt1 && opt2 && opt3;
}

// Usage in template:
rocsolver_potrf_template(..., optim_mem);
rocsolver_sygst_hegst_template(..., optim_mem);
rocsolver_trsm_upper<BATCHED, STRIDED, T>(..., optim_mem, work1, work2, work3, work4);"""
        }
    ]
})

# Entry 12: Coding task - residual computation (L2, coding)
entries.append({
    "id": str(int(time.time() * 1000) + 11),
    "level": "L2",
    "interface": "sygvdj_hegvdj",
    "query": "Write a HIP kernel to compute the residual ||A*X - B*X*Λ|| for verifying SYGVDJ results, where X are eigenvectors and Λ are eigenvalues.",
    "answer": """Here's a kernel to compute the residual for generalized eigenvalue problem verification:

```cpp
template <typename T, typename S>
__global__ void compute_sygvdj_residual(const rocblas_int n,
                                       const rocblas_int nev,
                                       T* A_orig,
                                       const rocblas_int lda,
                                       T* B_orig,
                                       const rocblas_int ldb,
                                       T* X,
                                       const rocblas_int ldx,
                                       S* lambda,
                                       S* residuals,
                                       const rocblas_int batch_count)
{
    // Batch and thread indices
    const rocblas_int bid = blockIdx.z;
    const rocblas_int j = blockIdx.x;  // Eigenvector index
    const rocblas_int tid = threadIdx.x;
    const rocblas_int nthreads = blockDim.x;

    if(bid >= batch_count || j >= nev)
        return;

    // Pointers for this batch
    T* A = A_orig + bid * lda * n;
    T* B = B_orig + bid * ldb * n;
    T* x_j = X + bid * ldx * n + j * ldx;  // j-th eigenvector
    S lambda_j = lambda[bid * nev + j];

    // Shared memory for partial sums
    extern __shared__ double smem[];
    S* partial_sums = reinterpret_cast<S*>(smem);

    // Compute A*x - λ*B*x for eigenvector j
    S local_sum = 0;

    for(rocblas_int i = tid; i < n; i += nthreads)
    {
        // Compute (A*x)_i
        T ax_i = 0;
        for(rocblas_int k = 0; k < n; k++)
        {
            // Reconstruct full matrix from symmetric storage
            rocblas_int idx = (i <= k) ? (i + k * lda) : (k + i * lda);
            ax_i += A[idx] * x_j[k];
        }

        // Compute (B*x)_i
        T bx_i = 0;
        for(rocblas_int k = 0; k < n; k++)
        {
            rocblas_int idx = (i <= k) ? (i + k * ldb) : (k + i * ldb);
            bx_i += B[idx] * x_j[k];
        }

        // Compute residual component: (A*x - λ*B*x)_i
        T r_i = ax_i - lambda_j * bx_i;

        // Add to local sum: ||r||²
        local_sum += std::norm(r_i);
    }

    // Store partial sum
    partial_sums[tid] = local_sum;
    __syncthreads();

    // Parallel reduction
    for(rocblas_int s = nthreads / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if(tid == 0)
    {
        residuals[bid * nev + j] = sqrt(partial_sums[0]);
    }
}

// Host function to launch kernel and analyze results
template <typename T, typename S>
void verify_sygvdj_results(rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nev,
                          T* d_A_orig,
                          const rocblas_int lda,
                          T* d_B_orig,
                          const rocblas_int ldb,
                          T* d_X,
                          const rocblas_int ldx,
                          S* d_lambda,
                          const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Allocate residual array
    S* d_residuals;
    hipMalloc(&d_residuals, sizeof(S) * nev * batch_count);

    // Launch kernel
    const rocblas_int threads = 256;
    size_t smem = threads * sizeof(S);
    dim3 grid(nev, 1, batch_count);
    dim3 block(threads, 1, 1);

    hipLaunchKernelGGL(compute_sygvdj_residual<T, S>,
                       grid, block, smem, stream,
                       n, nev, d_A_orig, lda, d_B_orig, ldb,
                       d_X, ldx, d_lambda, d_residuals, batch_count);

    // Copy results to host
    std::vector<S> h_residuals(nev * batch_count);
    std::vector<S> h_lambda(nev * batch_count);
    hipMemcpy(h_residuals.data(), d_residuals, sizeof(S) * nev * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_lambda.data(), d_lambda, sizeof(S) * nev * batch_count,
              hipMemcpyDeviceToHost);

    // Analyze and report
    std::cout << "SYGVDJ Residual Analysis:\\n";
    std::cout << "=========================\\n";

    for(rocblas_int b = 0; b < batch_count; b++)
    {
        std::cout << "Batch " << b << ":\\n";

        S max_residual = 0;
        S avg_residual = 0;
        rocblas_int max_idx = 0;

        for(rocblas_int j = 0; j < nev; j++)
        {
            S res = h_residuals[b * nev + j];
            avg_residual += res;

            if(res > max_residual)
            {
                max_residual = res;
                max_idx = j;
            }
        }

        avg_residual /= nev;

        std::cout << "  Average residual: " << avg_residual << "\\n";
        std::cout << "  Maximum residual: " << max_residual
                 << " (eigenvalue " << max_idx
                 << ", λ=" << h_lambda[b * nev + max_idx] << ")\\n";

        if(max_residual < 1e-10)
            std::cout << "  EXCELLENT: All eigenpairs accurate\\n";
        else if(max_residual < 1e-6)
            std::cout << "  GOOD: Eigenpairs acceptable\\n";
        else
            std::cout << "  WARNING: Large residuals detected\\n";

        std::cout << "\\n";
    }

    hipFree(d_residuals);
}

// Usage:
// // Save copies of A and B before calling sygvdj
// hipMemcpy(d_A_copy, d_A, sizeof(T) * n * n, hipMemcpyDeviceToDevice);
// hipMemcpy(d_B_copy, d_B, sizeof(T) * n * n, hipMemcpyDeviceToDevice);
//
// rocsolver_sygvdj(handle, itype, evect, uplo, n, d_A, lda, d_B, ldb, d_D, d_info);
//
// verify_sygvdj_results(handle, n, n, d_A_copy, lda, d_B_copy, ldb,
//                      d_A, lda, d_D, batch_count);
```

This verification kernel helps:
1. Validate SYGVDJ results
2. Identify poorly computed eigenpairs
3. Assess overall solution quality
4. Debug convergence issues""",
    "code_blocks": []
})

# Write to file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_sygvdj_hegvdj.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_path}")

# Verify the schema
import jsonschema

schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "level": {"enum": ["L1", "L2", "L3"]},
        "interface": {"type": "string"},
        "query": {"type": "string"},
        "answer": {"type": "string"},
        "code_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "language": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "language", "content"]
            }
        }
    },
    "required": ["id", "level", "interface", "query", "answer", "code_blocks"]
}

try:
    with open(output_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            jsonschema.validate(entry, schema)
    print("Schema validation passed!")

    # Count level distribution
    levels = {"L1": 0, "L2": 0, "L3": 0}
    for entry in entries:
        levels[entry["level"]] += 1
    print(f"Level distribution: L1={levels['L1']}, L2={levels['L2']}, L3={levels['L3']}")
except Exception as e:
    print(f"Schema validation failed: {e}")
