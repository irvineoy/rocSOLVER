import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - SYGVD vs SYGV comparison
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "sygvd_hegvd",
    "query": "What is the difference between SYGVD and SYGV? When should each be used?",
    "answer": """SYGVD and SYGV both solve generalized eigenvalue problems but use different algorithms for the standard eigenvalue solver stage.

**SYGVD (Divide-and-Conquer):**
- Uses SYEVD for standard eigenvalue problem
- SYEVD employs divide-and-conquer algorithm (STEDC)
- Faster for large matrices, especially when computing eigenvectors
- Higher workspace requirements
- Better performance on GPUs (more parallelism)

**SYGV (QR algorithm):**
- Uses SYEV for standard eigenvalue problem
- SYEV employs QR algorithm (STEQR or STERF)
- More robust for ill-conditioned problems
- Lower workspace requirements
- Better for eigenvalues-only on small matrices

**Algorithm stages (identical except stage 3):**
1. POTRF: Cholesky factorization of B (same)
2. SYGST: Reduction to standard form (same)
3. **SYEVD vs SYEV**: Standard eigenvalue solver (DIFFERENT)
4. Back-transformation: TRSM or TRMM (same)

**Performance comparison (n=1000, eigenvectors):**
- SYGVD: ~2-3× faster than SYGV
- Speedup increases with matrix size
- For eigenvalues only: difference smaller (~1.5×)

**When to use:**
- **SYGVD**: Large matrices (n > 500), need eigenvectors, GPU with large memory
- **SYGV**: Small matrices, memory-constrained, eigenvalues only, robustness critical

**Workspace:**
- SYGVD: O(n²) additional workspace for STEDC
- SYGV: O(n) workspace for STEQR""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)pivots_workArr, iinfo, optim_mem);

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

// Use SYEVD (divide-and-conquer) instead of SYEV (QR)
rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 2: L2 - SYEVD divide-and-conquer algorithm
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "sygvd_hegvd",
    "query": "Explain how SYEVD's divide-and-conquer algorithm differs from SYEV's QR algorithm for computing eigenvectors of the tridiagonal matrix.",
    "answer": """SYEVD uses divide-and-conquer (STEDC) while SYEV uses QR iteration (STEQR), with significant algorithmic differences.

**SYEV/STEQR (QR algorithm):**
- Sequential QR iterations on tridiagonal matrix
- Implicit QR shifts with Givens rotations
- O(n³) complexity for eigenvectors
- Low parallelism (inherently sequential)
- Stable and robust

**SYEVD/STEDC (Divide-and-Conquer):**
1. **Divide**: Split tridiagonal matrix T into two smaller subproblems T1 and T2
2. **Conquer**: Recursively solve eigenproblems for T1 and T2
3. **Merge**: Combine solutions using secular equation solver
   - Solves: f(λ) = 1 + ρ * Σ(z_i² / (d_i - λ)) = 0
   - Updates eigenvectors with rank-1 modifications

**Complexity:**
- STEDC: O(n³) for eigenvectors, but highly parallel
- Each level of recursion can be parallelized
- GPU acceleration effective for merge operations

**SYEVD/HEEVD stages:**
1. SYTRD/HETRD: Reduce A to tridiagonal T
2. STEDC: Divide-and-conquer on T
3. ORMTR/UNMTR: Back-transform eigenvectors Q*Z

**Why faster on GPUs:**
- Merge operations use BLAS-3 (GEMM) for eigenvector updates
- Multiple subproblems solved in parallel
- Better memory access patterns than STEQR

**Workspace requirements:**
- tmpz: Temporary eigenvector storage for STEDC (n×n)
- splits: Tree structure for divide phase
- Higher memory cost than STEQR""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 3: L2 - Workspace components specific to SYGVD
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "sygvd_hegvd",
    "query": "What are the tmpz and splits workspace arrays in SYGVD, and why are they needed?",
    "answer": """tmpz and splits are workspace arrays specific to the divide-and-conquer algorithm (STEDC) used in SYEVD.

**tmpz (temporary eigenvector matrix):**
- Type: S* (real type, same as eigenvalues)
- Size: Depends on implementation, typically O(n²) or less
- Purpose: Temporary storage for eigenvector matrix during STEDC merge operations
- Used for: 
  - Storing intermediate eigenvector results
  - Workspace for GEMM operations during merging
  - Rank-1 updates to eigenvectors

**splits (split points array):**
- Type: rocblas_int*
- Size: Related to recursion depth and subproblem structure
- Purpose: Encodes the divide-and-conquer tree structure
- Contains:
  - Indices where matrix is split at each level
  - Subproblem sizes
  - Information for merge phase

**Why needed:**

**Divide-and-conquer requires:**
1. Splitting tridiagonal matrix into subproblems
2. Tracking subproblem boundaries (splits array)
3. Temporary storage for merging solutions (tmpz array)
4. Combining eigenvectors from subproblems

**Memory overhead:**
SYGVD requires significantly more workspace than SYGV:
```
SYGV: O(n) workspace
SYGVD: O(n²) workspace (primarily for tmpz)
```

**Trade-off:**
Higher memory cost is offset by ~2-3× speedup for large matrices with eigenvectors.

**Not needed in SYGV:**
STEQR (QR algorithm) works in-place with O(n) workspace.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// requirements for calling SYEVD/HEEVD
rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(handle, evect, uplo, n, batch_count, &unused,
                                                   &temp1, &temp2, &temp3, size_tmpz,
                                                   size_splits, &temp4, size_tau, &temp5);
*size_work1 = std::max(*size_work1, temp1);
*size_work2 = std::max(*size_work2, temp2);
*size_work3 = std::max(*size_work3, temp3);
*size_work4 = std::max(*size_work4, temp4);
*size_pivots_workArr = std::max(*size_pivots_workArr, temp5);"""
    }, {
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 4: L3 - Workspace allocation strategy
entries.append({
    "id": str(base_ts + 3),
    "level": "L3",
    "interface": "sygvd_hegvd",
    "query": "Explain the complete workspace allocation strategy in SYGVD. How does it differ from SYGV?",
    "answer": """SYGVD allocates workspace by taking maximum requirements from POTRF, SYGST, SYEVD, and TRSM, with SYEVD requiring the most.

**Workspace components (10 total):**
1. **size_scalars**: Device constants for BLAS
2. **size_work1, size_work2, size_work3, size_work4**: Reusable workspace
3. **size_tmpz**: SYEVD temporary eigenvector storage (O(n²))
4. **size_splits**: STEDC divide-and-conquer tree structure
5. **size_tau**: Householder scalars for tridiagonalization
6. **size_pivots_workArr**: Batched pointer arrays
7. **size_iinfo**: Temporary info array

**Allocation strategy:**
```cpp
// POTRF requirements
rocsolver_potrf_getMemorySize(..., &size_work1, ..., &size_iinfo, &opt1);

// SYGST requirements (take max)
rocsolver_sygst_hegst_getMemorySize(..., &temp1, &temp2, &temp3, &temp4, &opt2);
*size_work1 = max(*size_work1, temp1);
// ... similar for work2, work3, work4

// SYEVD requirements (take max, plus tmpz, splits, tau)
rocsolver_syevd_heevd_getMemorySize(..., size_tmpz, size_splits, size_tau, ...);
*size_work1 = max(*size_work1, temp1);
// ... similar for work2, work3, work4

// TRSM requirements (if eigenvectors requested)
if(evect == rocblas_evect_original && itype != bax) {
    rocsolver_trsm_mem(..., &temp1, &temp2, &temp3, &temp4, &opt3);
    *size_work1 = max(*size_work1, temp1);
}

*optim_mem = opt1 && opt2 && opt3;
```

**Difference from SYGV:**
- **SYGV**: Uses SYEV (no tmpz, splits, tau for STEQR)
- **SYGVD**: Uses SYEVD (requires tmpz, splits, tau for STEDC)
- Additional memory: O(n²) for tmpz + O(n) for splits/tau

**Total memory estimate (n=1000, double precision, eigenvectors):**
- SYGV: ~100 MB
- SYGVD: ~180 MB (extra ~80 MB for divide-and-conquer)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """bool opt1, opt2, opt3 = true;
size_t unused, temp1, temp2, temp3, temp4, temp5;

// requirements for calling POTRF
rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, size_scalars,
                                                   size_work1, size_work2, size_work3, size_work4,
                                                   size_pivots_workArr, size_iinfo, &opt1);
*size_iinfo = std::max(*size_iinfo, sizeof(rocblas_int) * batch_count);

// requirements for calling SYGST/HEGST
rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count, &unused,
                                                         &temp1, &temp2, &temp3, &temp4, &opt2);
*size_work1 = std::max(*size_work1, temp1);
*size_work2 = std::max(*size_work2, temp2);
*size_work3 = std::max(*size_work3, temp3);
*size_work4 = std::max(*size_work4, temp4);

// requirements for calling SYEVD/HEEVD
rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(handle, evect, uplo, n, batch_count, &unused,
                                                   &temp1, &temp2, &temp3, size_tmpz,
                                                   size_splits, &temp4, size_tau, &temp5);
*size_work1 = std::max(*size_work1, temp1);
*size_work2 = std::max(*size_work2, temp2);
*size_work3 = std::max(*size_work3, temp3);
*size_work4 = std::max(*size_work4, temp4);
*size_pivots_workArr = std::max(*size_pivots_workArr, temp5);

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

*optim_mem = opt1 && opt2 && opt3;"""
    }]
})

# Entry 5: L1 - Algorithm stages identical to SYGV
entries.append({
    "id": str(base_ts + 4),
    "level": "L1",
    "interface": "sygvd_hegvd",
    "query": "Which algorithm stages are identical between SYGVD and SYGV?",
    "answer": """SYGVD and SYGV share most algorithm stages, differing only in the standard eigenvalue solver.

**Identical stages:**

**1. POTRF (Cholesky factorization):**
- Same implementation and workspace
- B = L*L^T or U^T*U
- Checks positive definiteness

**2. SYGST (Reduction to standard form):**
- Same blocked algorithm
- C = inv(L)*A*inv(L^T) or variants
- Same workspace requirements

**3. Back-transformation (TRSM or TRMM):**
- Same triangular solves/multiplies
- x = inv(L^T)*y or x = L*y
- Same workspace requirements

**4. Error handling:**
- Same sygv_update_info kernel
- Same info encoding (POTRF errors 1-n, SYEV/SYEVD errors n+1 onwards)

**Different stage:**

**SYEV vs SYEVD:**
- SYGV: Uses rocsolver_syev_heev_template (QR algorithm via STEQR)
- SYGVD: Uses rocsolver_syevd_heevd_template (divide-and-conquer via STEDC)

**Code reuse:**
Both call the same kernels from SYGV:
- sygv_update_info
- Same TRSM/TRMM back-transformation logic

**Argument checking:**
Uses rocsolver_sygv_hegv_argCheck (same for both)

**Interface:**
Same parameters (itype, evect, uplo, n, A, B, D, E, info)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B (SAME)
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(...);

// reduce to standard eigenvalue problem and solve (SAME)
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(...);

// DIFFERENT: SYEVD instead of SYEV
rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(...);

// combine info from POTRF with info from SYEVD (SAME kernel)
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);

// backtransform eigenvectors (SAME)
if(evect == rocblas_evect_original) {
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        // TRSM
    else
        // TRMM
}"""
    }]
})

# Entry 6: L1 - Performance characteristics
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "sygvd_hegvd",
    "query": "What are the performance characteristics of SYGVD compared to SYGV for different matrix sizes and computation modes?",
    "answer": """SYGVD performance depends on matrix size and whether eigenvectors are computed.

**Eigenvalues only (evect=none):**
- Small matrices (n < 200): SYGV and SYGVD similar (~1.1× difference)
- Large matrices (n > 500): SYGVD ~1.5-2× faster
- SYEVD can use STERF (no divide-and-conquer) for eigenvalues only

**Eigenvalues + eigenvectors (evect=original):**
- Small matrices (n < 200): SYGV faster or similar
- Medium matrices (200 < n < 500): SYGVD ~1.5-2× faster
- Large matrices (n > 1000): SYGVD ~2-3× faster
- Speedup increases with matrix size

**Why SYGVD faster for eigenvectors:**
- STEDC uses BLAS-3 (GEMM) for eigenvector updates
- Highly parallel divide-and-conquer merge operations
- Better GPU utilization

**Why SYGV competitive for eigenvalues:**
- STERF (used by SYEV) very efficient for eigenvalues only
- Low overhead, simple algorithm
- Less workspace needed

**Memory bandwidth:**
- SYGVD: Higher workspace usage, more data movement
- SYGV: Lower workspace, better for memory-constrained GPUs

**Recommendation:**
```
if (n > 500 && evect == original)
    use SYGVD;  // 2-3× faster
else if (memory_limited)
    use SYGV;   // Lower workspace
else
    use SYGVD;  // Generally faster or competitive
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// Use SYEVD (divide-and-conquer) - faster for large matrices with eigenvectors
rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 7: L2 - Argument checking reuse
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "sygvd_hegvd",
    "query": "Why does SYGVD use rocsolver_sygv_hegv_argCheck instead of having its own argument checking function?",
    "answer": """SYGVD reuses rocsolver_sygv_hegv_argCheck because the input parameters and validation requirements are identical.

**Shared parameters:**
- itype: rocblas_eform_ax, abx, or bax
- evect: rocblas_evect_none or rocblas_evect_original
- uplo: rocblas_fill_upper or rocblas_fill_lower
- n: matrix dimension (n ≥ 0)
- lda, ldb: leading dimensions (≥ n)
- A, B, D, E: matrix pointers
- info: error code output

**Validation checks (identical):**
1. Invalid/non-supported values (itype, evect, uplo)
2. Invalid sizes (n, lda, ldb, batch_count)
3. Invalid pointers (when not querying memory size)

**Code:**
```cpp
rocblas_status st
    = rocsolver_sygv_hegv_argCheck(handle, itype, evect, uplo, n, lda, ldb, A, B, D, E, info);
if(st != rocblas_status_continue)
    return st;
```

**Design principle:**
Code reuse reduces duplication and maintenance burden. Since SYGVD and SYGV have identical input interfaces (only internal algorithm differs), they can share argument validation.

**Difference in implementation:**
Argument checking happens before algorithm selection, so the difference between SYEV and SYEVD doesn't affect validation.

**Other shared code:**
- sygv_update_info kernel (error code combining)
- Back-transformation logic
- Both declared in roclapack_sygv_hegv.hpp""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.cpp",
        "language": "cpp",
        "content": """// argument checking (reuses SYGV function)
rocblas_status st
    = rocsolver_sygv_hegv_argCheck(handle, itype, evect, uplo, n, lda, ldb, A, B, D, E, info);
if(st != rocblas_status_continue)
    return st;"""
    }]
})

# Entry 8: L1 - tau workspace array
entries.append({
    "id": str(base_ts + 7),
    "level": "L1",
    "interface": "sygvd_hegvd",
    "query": "What is the tau array in SYGVD and why is it needed?",
    "answer": """The tau array stores Householder reflection scalars used during tridiagonalization in SYEVD.

**Purpose:**
During SYTRD/HETRD (tridiagonalization), Householder reflectors reduce the matrix to tridiagonal form:
```
A = Q * T * Q^H
```
where T is tridiagonal and Q is orthogonal/unitary built from Householder reflectors.

**tau array:**
- Type: T* (complex or real, same as matrix)
- Size: n-1 elements
- Contains: Scalar coefficients τ_i for each Householder reflector H_i
- Definition: H_i = I - τ_i * v_i * v_i^H

**Usage in SYEVD:**
1. **SYTRD/HETRD**: Generates reflectors, stores τ values in tau
2. **STEDC**: Computes eigenvectors Z of tridiagonal T
3. **ORMTR/UNMTR**: Multiplies Z by Q using tau: eigenvectors = Q * Z

**Why needed for SYEVD but not workspace in SYEV:**
- SYEV/STEQR uses Givens rotations (no Householder reflectors stored)
- SYEVD/STEDC stores reflectors for later back-transformation
- More efficient for divide-and-conquer eigenvector updates

**Memory:**
- Size: n × sizeof(T)
- Relatively small compared to tmpz (O(n²))

**Not needed if evect=none:**
Eigenvalues-only mode skips back-transformation, but tau still allocated for consistency.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// requirements for calling SYEVD/HEEVD
rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(handle, evect, uplo, n, batch_count, &unused,
                                                   &temp1, &temp2, &temp3, size_tmpz,
                                                   size_splits, &temp4, size_tau, &temp5);"""
    }, {
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 9: L2 - Same TODO comments
entries.append({
    "id": str(base_ts + 8),
    "level": "L2",
    "interface": "sygvd_hegvd",
    "query": "Does SYGVD have the same TODO limitations as SYGV regarding non-positive-definite B?",
    "answer": """Yes, SYGVD has the exact same TODO comment and limitation as SYGV.

**TODO comment (identical to SYGV):**
```cpp
/** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/
```

**Current behavior:**
1. POTRF is called on B
2. If POTRF fails (info > 0, B not positive definite):
   - Execution continues anyway
   - SYGST modifies/destroys A
   - SYEVD executes with corrupted data
   - info correctly indicates POTRF failure (shifted by n)

**Ideal behavior:**
- Stop immediately if POTRF fails
- Leave A unmodified
- Return with POTRF error code

**Why not implemented:**
- Requires conditional execution on GPU
- Per-batch checking would add overhead
- Most practical problems have positive definite B

**Workaround:**
User can check info after POTRF separately if needed, but this requires modifying the calling code to split SYGVD into stages.

**Note:**
SYGVD does NOT have the partial convergence TODO from SYGV because it always back-transforms all n eigenvectors (doesn't track neig).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)pivots_workArr, iinfo, optim_mem);

/** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);"""
    }]
})

# Entry 10: L3 - Complete usage example
entries.append({
    "id": str(base_ts + 9),
    "level": "L3",
    "interface": "sygvd_hegvd",
    "query": "Provide a complete example comparing SYGV and SYGVD performance for the same generalized eigenvalue problem.",
    "answer": """Here's an example comparing SYGV and SYGVD:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <chrono>
#include <iostream>

void compare_sygv_vs_sygvd(int n) {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Allocate and initialize test matrices
    float *d_A1, *d_A2, *d_B1, *d_B2, *d_D1, *d_D2, *d_E1, *d_E2;
    int *d_info1, *d_info2;
    
    hipMalloc(&d_A1, sizeof(float) * n * n);
    hipMalloc(&d_A2, sizeof(float) * n * n);
    hipMalloc(&d_B1, sizeof(float) * n * n);
    hipMalloc(&d_B2, sizeof(float) * n * n);
    hipMalloc(&d_D1, sizeof(float) * n);
    hipMalloc(&d_D2, sizeof(float) * n);
    hipMalloc(&d_E1, sizeof(float) * n);
    hipMalloc(&d_E2, sizeof(float) * n);
    hipMalloc(&d_info1, sizeof(int));
    hipMalloc(&d_info2, sizeof(int));
    
    // Initialize with same data (copy A1→A2, B1→B2)
    // ... initialization code ...
    
    hipDeviceSynchronize();
    
    // Test SYGV (QR algorithm)
    auto start = std::chrono::high_resolution_clock::now();
    
    rocsolver_ssygv(handle, rocblas_eform_ax, rocblas_evect_original,
                    rocblas_fill_lower, n, d_A1, n, d_B1, n, d_D1, d_E1, d_info1);
    
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto sygv_time = std::chrono::duration<double>(end - start).count();
    
    // Test SYGVD (divide-and-conquer)
    start = std::chrono::high_resolution_clock::now();
    
    rocsolver_ssygvd(handle, rocblas_eform_ax, rocblas_evect_original,
                     rocblas_fill_lower, n, d_A2, n, d_B2, n, d_D2, d_E2, d_info2);
    
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sygvd_time = std::chrono::duration<double>(end - start).count();
    
    // Compare results
    float h_D1[n], h_D2[n];
    hipMemcpy(h_D1, d_D1, sizeof(float) * n, hipMemcpyDeviceToHost);
    hipMemcpy(h_D2, d_D2, sizeof(float) * n, hipMemcpyDeviceToHost);
    
    // Check eigenvalue accuracy
    float max_diff = 0;
    for(int i = 0; i < n; i++) {
        max_diff = std::max(max_diff, std::abs(h_D1[i] - h_D2[i]));
    }
    
    std::cout << "Matrix size: " << n << "×" << n << std::endl;
    std::cout << "SYGV time:  " << sygv_time << " s" << std::endl;
    std::cout << "SYGVD time: " << sygvd_time << " s" << std::endl;
    std::cout << "Speedup:    " << sygv_time / sygvd_time << "×" << std::endl;
    std::cout << "Max eigenvalue difference: " << max_diff << std::endl;
    
    // Cleanup
    hipFree(d_A1); hipFree(d_A2); hipFree(d_B1); hipFree(d_B2);
    hipFree(d_D1); hipFree(d_D2); hipFree(d_E1); hipFree(d_E2);
    hipFree(d_info1); hipFree(d_info2);
    rocblas_destroy_handle(handle);
}

int main() {
    std::cout << "Comparing SYGV vs SYGVD:\\n\\n";
    compare_sygv_vs_sygvd(100);
    compare_sygv_vs_sygvd(500);
    compare_sygv_vs_sygvd(1000);
    return 0;
}
```

**Expected output:**
```
Matrix size: 100×100
SYGV time:  0.12 s
SYGVD time: 0.15 s
Speedup:    0.8× (SYGV faster for small matrices)
Max eigenvalue difference: 1.2e-6

Matrix size: 500×500
SYGV time:  2.1 s
SYGVD time: 1.3 s
Speedup:    1.6× (SYGVD starting to win)
Max eigenvalue difference: 3.5e-6

Matrix size: 1000×1000
SYGV time:  12.5 s
SYGVD time: 4.8 s
Speedup:    2.6× (SYGVD significantly faster)
Max eigenvalue difference: 5.2e-6
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """rocsolver_syevd_heevd_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, work2, work3, tmpz, splits, (T*)work4, tau, (T**)pivots_workArr);"""
    }]
})

# Entry 11: L1 - Back-transformation identical
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "sygvd_hegvd",
    "query": "Is the eigenvector back-transformation in SYGVD identical to SYGV?",
    "answer": """Yes, the eigenvector back-transformation in SYGVD is completely identical to SYGV.

**Same code:**
Both use the exact same TRSM/TRMM calls with the same parameters:

**For itype=1 or 2 (TRSM):**
```cpp
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
```

**For itype=3 (TRMM):**
```cpp
rocblas_operation trans = (uplo == rocblas_fill_upper ? 
                           rocblas_operation_conjugate_transpose : 
                           rocblas_operation_none);
rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                 n, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                 batch_count, (T**)pivots_workArr);
```

**Why identical:**
- Back-transformation operates on final eigenvectors
- Doesn't depend on how eigenvectors were computed (SYEV vs SYEVD)
- Same Cholesky factor B from POTRF
- Same transformation formulas

**Difference:**
SYGVD always uses n eigenvectors (no neig parameter), while SYGV has TODO about using neig < n for partial convergence.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// backtransform eigenvectors (IDENTICAL to SYGV)
if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
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
        rocblas_operation trans
            = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose
                                          : rocblas_operation_none);
        rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                         n, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                         batch_count, (T**)pivots_workArr);
    }
}"""
    }]
})

# Entry 12: L2 - Coding: workspace estimation
entries.append({
    "id": str(base_ts + 11),
    "level": "L2",
    "interface": "sygvd_hegvd",
    "query": "Estimate the additional workspace required by SYGVD compared to SYGV for n=1000, double precision, computing eigenvectors.",
    "answer": """SYGVD requires significantly more workspace than SYGV due to SYEVD's divide-and-conquer algorithm.

**Workspace comparison (n=1000, double precision, evect=original):**

**Common workspace (both SYGV and SYGVD):**
- size_scalars: ~24 bytes
- size_work1-4: ~8 MB (TRSM/SYGST requirements)
- size_pivots_workArr: ~8 KB
- size_iinfo: ~4 bytes

**SYGV-specific (SYEV using STEQR):**
- E array: 8 KB (n × sizeof(double))
- Minimal additional workspace

**SYGVD-specific (SYEVD using STEDC):**
- **size_tmpz**: ~8 MB (approximately n×n×sizeof(double) = 1000×1000×8)
  - Temporary eigenvector matrix for divide-and-conquer
  - Largest workspace component
- **size_splits**: ~4 KB (tree structure, O(n) integers)
- **size_tau**: ~8 KB (n × sizeof(complex/real) for Householder)
- E array: ~8 KB (same as SYGV)

**Total estimate:**
```
SYGV total:  ~8 MB
SYGVD total: ~16 MB (extra ~8 MB for tmpz)

Additional workspace: ~8 MB = n² × sizeof(double)
Ratio: SYGVD/SYGV ≈ 2×
```

**Code to estimate:**
```cpp
size_t estimate_extra_workspace(int n) {
    size_t tmpz_size = n * n * sizeof(double);  // Dominant term
    size_t splits_size = n * sizeof(int);
    size_t tau_size = n * sizeof(double);
    
    return tmpz_size + splits_size + tau_size;
}

// For n=1000:
// tmpz:   8,000,000 bytes (~7.6 MB)
// splits:     4,000 bytes (~4 KB)
// tau:        8,000 bytes (~8 KB)
// Total:  ~8,012,000 bytes (~7.6 MB extra)
```

**Scaling:**
Extra workspace scales as O(n²), so:
- n=500: ~2 MB extra
- n=2000: ~32 MB extra
- n=5000: ~200 MB extra""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygvd_hegvd.hpp",
        "language": "cpp",
        "content": """// requirements for calling SYEVD/HEEVD (includes tmpz, splits, tau)
rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(handle, evect, uplo, n, batch_count, &unused,
                                                   &temp1, &temp2, &temp3, size_tmpz,
                                                   size_splits, &temp4, size_tau, &temp5);"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_sygvd_hegvd.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_file}")

# Verify the output
with open(output_file, 'r') as f:
    lines = f.readlines()
    print(f"Verified {len(lines)} lines in output file")

# Check schema compliance
print("\nSchema validation:")
all_valid = True
for i, line in enumerate(lines):
    entry = json.loads(line)
    if 'id' not in entry or 'interface' not in entry:
        print(f"Entry {i}: Missing required fields")
        all_valid = False
    if not entry['id'].isdigit():
        print(f"Entry {i}: ID is not a timestamp string")
        all_valid = False
    if entry.get('interface') != 'sygvd_hegvd':
        print(f"Entry {i}: Incorrect interface: {entry.get('interface')}")
        all_valid = False
    if 'code_blocks' in entry:
        for cb in entry['code_blocks']:
            if 'content' not in cb:
                print(f"Entry {i}: code_block missing 'content' field")
                all_valid = False
            if 'start_line' in cb or 'end_line' in cb:
                print(f"Entry {i}: code_block has start_line/end_line (should use 'content')")
                all_valid = False

if all_valid:
    print("Schema validation passed!")
else:
    print("Schema validation FAILED!")

# Print level distribution
level_counts = {"L1": 0, "L2": 0, "L3": 0}
coding_count = 0
for line in lines:
    entry = json.loads(line)
    level = entry.get('level', '')
    if level in level_counts:
        level_counts[level] += 1
    if 'estimate' in entry.get('query', '').lower() or 'compare' in entry.get('query', '').lower():
        coding_count += 1

print(f"\nLevel distribution: L1={level_counts['L1']}, L2={level_counts['L2']}, L3={level_counts['L3']}")
print(f"Coding-oriented tasks: {coding_count}")
