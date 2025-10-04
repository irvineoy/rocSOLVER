import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYGV/HEGV algorithm overview
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "sygv_hegv",
    "query": "Explain the complete algorithm used in SYGV/HEGV to solve generalized symmetric/Hermitian eigenvalue problems A*x = λ*B*x. What are the main stages?",
    "answer": """SYGV/HEGV solves generalized eigenvalue problems A*x = λ*B*x (and variants) using a reduction to standard form.

**Algorithm stages:**

**1. Cholesky factorization of B (POTRF):**
```
B = L*L^T  (uplo=lower)
B = U^T*U  (uplo=upper)
```
- Requires B to be positive definite
- If POTRF fails (info > 0), B is not positive definite → error

**2. Reduction to standard form (SYGST/HEGST):**
Transforms A based on itype:
- itype=1 (A*x = λ*B*x): C = inv(L)*A*inv(L^T) or inv(U^T)*A*inv(U)
- itype=2 (A*B*x = λ*x): C = L^T*A*L or U*A*U^T
- itype=3 (B*A*x = λ*x): Same as itype=2
Overwrites A with C.

**3. Solve standard eigenvalue problem (SYEV/HEEV):**
```
C*y = λ*y
```
- Computes eigenvalues in D
- If evect=original, computes eigenvectors y in A
- Uses tridiagonalization + QR algorithm (STEQR or STERF)

**4. Back-transform eigenvectors (TRSM or TRMM):**
Only if evect=original:
- itype=1 or 2: x = inv(L^T)*y (TRSM) or x = inv(U)*y
- itype=3: x = L*y (TRMM) or x = U^T*y
Recovers eigenvectors of original problem.

**Info handling:**
```
info = 0: Success
info = i (1 ≤ i ≤ n): POTRF failed, B(i,i) not positive definite
info = n+i: SYEV failed, i eigenvalues did not converge
```

**Result:**
D contains eigenvalues, A contains eigenvectors (if requested).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)pivots_workArr, iinfo, optim_mem);

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

rocsolver_syev_heev_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, (T*)work2, (T*)work3, (T*)work4, (T**)pivots_workArr);

// combine info from POTRF with info from SYEV/HEEV
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);"""
    }]
})

# Entry 2: L2 - Info handling and error propagation
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "How does SYGV handle error codes from POTRF and SYEV? Explain the sygv_update_info kernel.",
    "answer": """SYGV combines error information from both POTRF (Cholesky factorization) and SYEV (eigenvalue solver) using the sygv_update_info kernel.

**Error encoding:**
- **info from POTRF**: 0 (success) or i (1 ≤ i ≤ n, B not positive definite at position i)
- **iinfo from SYEV**: 0 (success) or i (i eigenvalues failed to converge)
- **final info**: Combined error code

**sygv_update_info kernel:**
```cpp
if(info[b] != 0)
    info[b] += n;  // POTRF error: shift by n to indicate factorization failure
else
    info[b] = iinfo[b];  // No POTRF error: use SYEV result
```

**Interpretation:**
- info = 0: Complete success
- 1 ≤ info ≤ n: POTRF failed at position info (B not positive definite)
- n+1 ≤ info: SYEV failed, (info-n) eigenvalues did not converge

**Example:**
- n = 100
- POTRF succeeds (info = 0)
- SYEV fails with 3 non-converged eigenvalues (iinfo = 3)
- Final: info = 3 (3 eigenvalues did not converge)

**Example 2:**
- n = 100
- POTRF fails at position 50 (info = 50)
- Final: info = 50 + 100 = 150 (POTRF failure at position 50)

**Note:**
Current implementation continues even if POTRF fails (with TODO comment about stopping early). Matrix A is destroyed in non-positive-definite case.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """template <typename T>
ROCSOLVER_KERNEL void sygv_update_info(T* info, T* iinfo, const rocblas_int n, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] != 0)
            info[b] += n;
        else
            info[b] = iinfo[b];
    }
}"""
    }, {
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """// combine info from POTRF with info from SYEV/HEEV
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);"""
    }]
})

# Entry 3: L2 - Back-transformation for itype=1
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "Explain the eigenvector back-transformation for itype=1 in SYGV. Why is TRSM used and what system does it solve?",
    "answer": """For itype=1 (A*x = λ*B*x), eigenvector back-transformation uses TRSM to recover original eigenvectors.

**Mathematical background:**
1. SYGST transforms: C = inv(L)*A*inv(L^T) (uplo=lower)
2. SYEV solves: C*y = λ*y
3. Need to recover: x from y

**Derivation:**
```
A*x = λ*B*x
A*x = λ*(L*L^T)*x
inv(L)*A*inv(L^T) * (L^T*x) = λ*(L^T*x)
C*y = λ*y, where y = L^T*x
Therefore: x = inv(L^T)*y
```

**Implementation (uplo=lower):**
```cpp
rocsolver_trsm_lower<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_diagonal_non_unit, n, neig, B, shiftB, ldb, strideB, A, shiftA, lda,
    strideA, batch_count, optim_mem, work1, work2, work3, work4);
```

Solves: L^T * X = Y (where Y=A contains y, X overwrites with x)

**Implementation (uplo=upper):**
```cpp
rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
    neig, B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, optim_mem,
    work1, work2, work3, work4);
```

Solves: U * X = Y (since C = inv(U^T)*A*inv(U), y = U*x, so x = inv(U)*y)

**Parameter neig:**
Number of converged eigenvalues (currently set to n, but could be < n if SYEV partially failed).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """rocblas_int neig = n; //number of converged eigenvalues

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        if(uplo == rocblas_fill_upper)
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
                neig, B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, optim_mem,
                work1, work2, work3, work4);
        else
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, n, neig, B, shiftB, ldb, strideB, A, shiftA, lda,
                strideA, batch_count, optim_mem, work1, work2, work3, work4);
    }"""
    }]
})

# Entry 4: L2 - Back-transformation for itype=3
entries.append({
    "id": str(base_ts + 3),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "How does eigenvector back-transformation differ for itype=3 (B*A*x = λ*x) compared to itype=1? Why is TRMM used instead of TRSM?",
    "answer": """For itype=3 (B*A*x = λ*x), back-transformation uses TRMM (triangular matrix-matrix multiply) instead of TRSM.

**Mathematical background:**
1. SYGST transforms: C = L^T*A*L (uplo=lower, same as itype=2)
2. SYEV solves: C*y = λ*y
3. Need to recover: x from y

**Derivation:**
```
B*A*x = λ*x
(L*L^T)*A*x = λ*x
L^T*A*x = λ*inv(L)*x
L^T*A*L * (inv(L)*x) = λ*(inv(L)*x)
C*y = λ*y, where y = inv(L)*x
Therefore: x = L*y
```

**Key difference:**
- itype=1: x = inv(L^T)*y (solve, use TRSM)
- itype=3: x = L*y (multiply, use TRMM)

**Implementation (uplo=lower):**
```cpp
rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none,
                 rocblas_diagonal_non_unit, n, neig, &one, 0, B, shiftB, ldb, strideB,
                 A, shiftA, lda, strideA, batch_count, (T**)pivots_workArr);
```
Computes: X = L * Y (where Y=A contains y)

**Implementation (uplo=upper):**
```cpp
rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_conjugate_transpose,
                 rocblas_diagonal_non_unit, n, neig, &one, 0, B, shiftB, ldb, strideB,
                 A, shiftA, lda, strideA, batch_count, (T**)pivots_workArr);
```
Computes: X = U^T * Y

**Performance:**
TRMM is generally faster than TRSM (multiplication vs solving), making itype=3 slightly faster than itype=1 for back-transformation.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """else  // itype == rocblas_eform_bax
{
    rocblas_operation trans
        = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose
                                      : rocblas_operation_none);
    rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                     neig, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                     batch_count, (T**)pivots_workArr);
}"""
    }]
})

# Entry 5: L1 - evect parameter
entries.append({
    "id": str(base_ts + 4),
    "level": "L1",
    "interface": "sygv_hegv",
    "query": "What is the evect parameter in SYGV and how does it affect the computation?",
    "answer": """The evect parameter controls whether eigenvectors are computed and returned.

**Values:**
- **rocblas_evect_none**: Compute eigenvalues only (no eigenvectors)
- **rocblas_evect_original**: Compute both eigenvalues and eigenvectors

**Impact on computation:**

**evect = rocblas_evect_none:**
- SYEV computes eigenvalues only (faster)
- No back-transformation needed (TRSM/TRMM skipped)
- Matrix A is destroyed but not used for output
- Only D array (eigenvalues) contains meaningful results

**evect = rocblas_evect_original:**
- SYEV computes eigenvalues and eigenvectors
- Back-transformation applied to recover original eigenvectors
- Matrix A contains eigenvectors on output
- Both D (eigenvalues) and A (eigenvectors) contain results

**Performance:**
Computing only eigenvalues is ~2-3× faster than computing eigenvectors, since:
- SYEV can use STERF instead of STEQR
- Back-transformation (TRSM/TRMM) is skipped
- Less memory bandwidth required

**Usage:**
```cpp
// Eigenvalues only
rocsolver_ssygv(handle, itype, rocblas_evect_none, uplo, n, 
                A, lda, B, ldb, D, E, info);

// Eigenvalues + eigenvectors
rocsolver_ssygv(handle, itype, rocblas_evect_original, uplo, n, 
                A, lda, B, ldb, D, E, info);
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """rocsolver_syev_heev_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, (T*)work2, (T*)work3, (T*)work4, (T**)pivots_workArr);

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    // ... TRSM or TRMM back-transformation ...
}"""
    }, {
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """if(evect != rocblas_evect_none && evect != rocblas_evect_original)
    return rocblas_status_invalid_value;"""
    }]
})

# Entry 6: L3 - Workspace allocation
entries.append({
    "id": str(base_ts + 5),
    "level": "L3",
    "interface": "sygv_hegv",
    "query": "Explain the workspace allocation strategy in SYGV. What are the different workspace components and how are they sized?",
    "answer": """SYGV allocates workspace by taking the maximum requirements from three subroutines: POTRF, SYGST, and SYEV.

**Workspace components:**
1. **size_scalars**: Device memory for BLAS constants
2. **size_work1, size_work2, size_work3, size_work4**: Reusable workspace for TRSM and other operations
3. **size_pivots_workArr**: For batched pointer arrays (POTRF and SYEV)
4. **size_iinfo**: Temporary info array for SYEV results (batch_count × sizeof(int))

**Allocation strategy:**
```cpp
// Get POTRF requirements
rocsolver_potrf_getMemorySize(..., &size_scalars, &size_work1, &size_work2, 
                               &size_work3, &size_work4, &size_pivots_workArr, 
                               &size_iinfo, &opt1);

// Get SYGST requirements, take maximum with POTRF
rocsolver_sygst_hegst_getMemorySize(..., &temp1, &temp2, &temp3, &temp4, &opt2);
*size_work1 = max(*size_work1, temp1);
*size_work2 = max(*size_work2, temp2);
*size_work3 = max(*size_work3, temp3);
*size_work4 = max(*size_work4, temp4);

// Get SYEV requirements, take maximum with previous
rocsolver_syev_heev_getMemorySize(..., &temp1, &temp2, &temp3, &temp4, &temp5);
*size_work1 = max(*size_work1, temp1);
// ... similar for work2, work3, work4
*size_pivots_workArr = max(*size_pivots_workArr, temp5);
```

**Additional for back-transformation (if evect=original):**
For itype=1 or 2, add TRSM workspace requirements:
```cpp
rocsolver_trsm_mem(..., &temp1, &temp2, &temp3, &temp4, &opt3);
*size_work1 = max(*size_work1, temp1);
// ... similar for work2, work3, work4
```

**optim_mem flag:**
Set to true only if all three subroutines can use optimal memory allocation:
```cpp
*optim_mem = opt1 && opt2 && opt3;
```

**Total memory:**
Sum of all workspace components, reused across different algorithm stages.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
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

// requirements for calling SYEV/HEEV
rocsolver_syev_heev_getMemorySize<BATCHED, T, S>(evect, uplo, n, batch_count, &unused, &temp1,
                                                 &temp2, &temp3, &temp4, &temp5);
*size_work1 = std::max(*size_work1, temp1);
*size_work2 = std::max(*size_work2, temp2);
*size_work3 = std::max(*size_work3, temp3);
*size_work4 = std::max(*size_work4, temp4);
*size_pivots_workArr = std::max(*size_pivots_workArr, temp5);

if(evect == rocblas_evect_original)
{
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        rocblas_operation trans
            = (uplo == rocblas_fill_upper ? rocblas_operation_none
                                          : rocblas_operation_conjugate_transpose);
        // requirements for calling TRSM
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

# Entry 7: L1 - itype parameter differences
entries.append({
    "id": str(base_ts + 6),
    "level": "L1",
    "interface": "sygv_hegv",
    "query": "What are the three itype values in SYGV and how do they differ mathematically and computationally?",
    "answer": """The itype parameter specifies which generalized eigenvalue problem to solve.

**itype = 1 (rocblas_eform_ax): A*x = λ*B*x**
- Most common form
- Standard SYGST transformation: C = inv(L)*A*inv(L^T)
- Back-transformation: x = inv(L^T)*y (TRSM, solving)
- Applications: Vibration problems, quantum mechanics

**itype = 2 (rocblas_eform_abx): A*B*x = λ*x**
- Alternative form
- SYGST transformation: C = L^T*A*L
- Back-transformation: x = inv(L^T)*y (TRSM, solving)
- Same TRSM as itype=1

**itype = 3 (rocblas_eform_bax): B*A*x = λ*x**
- Another alternative form
- SYGST transformation: C = L^T*A*L (same as itype=2)
- Back-transformation: x = L*y (TRMM, multiplication)
- Different from itype=1/2, uses TRMM (faster)

**Computational differences:**

**SYGST phase:**
- itype=1: Uses TRSV/TRSM (inverse operations)
- itype=2/3: Uses TRMV/TRMM (forward operations)

**Back-transformation:**
- itype=1, 2: TRSM (solve triangular system)
- itype=3: TRMM (triangular matrix multiply, faster)

**Eigenvalues:**
All three forms produce the same eigenvalues λ.

**Choosing itype:**
Use whichever matches your problem formulation. If free to choose, itype=3 is slightly faster due to TRMM.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
    return rocblas_status_invalid_value;"""
    }, {
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
{
    // Use TRSM for back-transformation
    if(uplo == rocblas_fill_upper)
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(...);
    else
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(...);
}
else  // itype == rocblas_eform_bax
{
    // Use TRMM for back-transformation
    rocblas_operation trans = (uplo == rocblas_fill_upper ? 
                               rocblas_operation_conjugate_transpose : 
                               rocblas_operation_none);
    rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, ...);
}"""
    }]
})

# Entry 8: L2 - Output arrays D and E
entries.append({
    "id": str(base_ts + 7),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "What are the D and E output arrays in SYGV? Why are both needed?",
    "answer": """D and E are output arrays used during the eigenvalue computation process.

**D array (eigenvalues):**
- Size: n elements (real type S)
- Contains the n eigenvalues of the generalized problem on output
- Eigenvalues are in ascending order (typically)
- Always computed regardless of evect parameter

**E array (workspace/subdiagonal):**
- Size: n elements (real type S)
- Used as workspace during tridiagonalization and STEQR/STERF
- Contains off-diagonal elements of the tridiagonal matrix during intermediate stages
- On output, may contain residual information (not typically used)

**Why both needed:**

**During SYEV/HEEV:**
1. SYTRD/HETRD reduces A to tridiagonal form T with diagonal in D and off-diagonal in E
2. STEQR or STERF computes eigenvalues of T:
   - STERF (eigenvalues only): Uses D and E as input
   - STEQR (eigenvalues + vectors): Uses D, E, and matrix for transformations

**Memory consideration:**
Even though E is primarily workspace, it must be allocated by the caller to allow SYEV to use it for the tridiagonal representation.

**User perspective:**
- D: Output eigenvalues (use this)
- E: Workspace (can be discarded after call)

**Type S:**
Both D and E are real type S (float or double), even for complex Hermitian problems (complex types T), because eigenvalues of Hermitian matrices are always real.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """rocsolver_syev_heev_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, (T*)work2, (T*)work3, (T*)work4, (T**)pivots_workArr);"""
    }, {
        "path": "library/src/lapack/roclapack_sygv_hegv.cpp",
        "language": "cpp",
        "content": """template <typename T, typename S, typename U>
rocblas_status rocsolver_sygv_hegv_impl(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int lda,
                                        U B,
                                        const rocblas_int ldb,
                                        S* D,  // Real type
                                        S* E,  // Real type
                                        rocblas_int* info)"""
    }]
})

# Entry 9: L2 - TODO comments and partial convergence
entries.append({
    "id": str(base_ts + 8),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "What do the TODO comments in SYGV indicate about handling non-positive-definite B and partial eigenvalue convergence?",
    "answer": """The TODO comments highlight two current limitations in SYGV's error handling:

**TODO 1: Non-positive-definite B handling**
```cpp
/** (TODO: Strictly speaking, computations should stop here is B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/
```

**Current behavior:**
- POTRF is called on B
- If B is not positive definite (info > 0), execution continues
- SYGST and SYEV still execute, destroying A
- Final info correctly indicates POTRF failure

**Ideal behavior:**
- If POTRF fails, stop immediately
- Leave A unmodified
- Return immediately with appropriate error code
- Challenge: Requires conditional execution on GPU (inefficient)

**TODO 2: Partial convergence handling**
```cpp
/** (TODO: Similarly, if only neig < n eigenvalues converged, TRSM or TRMM below should not
    work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
    iinfo and set neig = n) **/
```

**Current behavior:**
```cpp
rocblas_int neig = n;  // Always n
```
- Back-transformation operates on all n eigenvectors
- Even if only neig < n eigenvalues converged
- Non-converged eigenvectors are back-transformed unnecessarily

**Ideal behavior:**
- Check iinfo to determine neig (number of converged eigenvalues)
- Only back-transform first neig eigenvectors
- Challenge: Variable-size TRSM/TRMM calls per batch instance

**Impact:**
- Current implementation is correct but not optimal
- Wastes computation on non-converged eigenvectors
- Most practical cases converge fully, so impact is minimal""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
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
    }, {
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """/** (TODO: Similarly, if only neig < n eigenvalues converged, TRSM or TRMM below should not
    work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
    iinfo and set neig = n) **/

rocblas_int neig = n; //number of converged eigenvalues

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    // ... uses neig for TRSM/TRMM dimensions ...
}"""
    }]
})

# Entry 10: L1 - Host pointer mode
entries.append({
    "id": str(base_ts + 9),
    "level": "L1",
    "interface": "sygv_hegv",
    "query": "Why does SYGV set pointer mode to host?",
    "answer": """SYGV sets pointer mode to host because it uses host-allocated scalar constants.

**Code:**
```cpp
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

T one = 1;

// ... TRMM call using &one ...

rocblas_set_pointer_mode(handle, old_mode);
```

**Reason:**
- TRMM (for itype=3 back-transformation) requires alpha scalar
- Alpha defined on host stack: `T one = 1;`
- rocBLAS TRMM expects host pointer when in host pointer mode
- Simpler than allocating device memory for single scalar

**Subroutine behavior:**
- POTRF, SYGST, SYEV use their own pointer modes internally
- They save/restore pointer mode, so SYGV's mode setting doesn't affect them
- Only affects direct BLAS calls (TRMM) in SYGV

**Alternative approach:**
Could allocate device memory for scalars and use device pointer mode, but:
- Requires additional device malloc
- More complex for single scalar value
- Host mode is simpler and equally efficient for one call

**Best practice:**
Always save and restore original pointer mode to maintain caller's expectations.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the host
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

// constants for rocblas functions calls
T one = 1;

// ... algorithm implementation ...

rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                 neig, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,
                 batch_count, (T**)pivots_workArr);

// ...

rocblas_set_pointer_mode(handle, old_mode);"""
    }]
})

# Entry 11: L3 - Complete usage example
entries.append({
    "id": str(base_ts + 10),
    "level": "L3",
    "interface": "sygv_hegv",
    "query": "Provide a complete example of using SYGV to solve a generalized eigenvalue problem A*x = λ*B*x on the GPU, including error handling.",
    "answer": """Here's a complete example solving A*x = λ*B*x:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <iostream>
#include <vector>

void solve_generalized_eigenproblem() {
    const int n = 100;
    
    // 1. Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // 2. Allocate host memory and initialize
    std::vector<float> h_A(n * n);  // Symmetric matrix A
    std::vector<float> h_B(n * n);  // Symmetric positive definite matrix B
    std::vector<float> h_D(n);      // Eigenvalues
    std::vector<float> h_E(n);      // Workspace
    int h_info;
    
    // Initialize h_A and h_B with problem data...
    // (Ensure B is symmetric positive definite)
    
    // 3. Allocate device memory
    float *d_A, *d_B, *d_D, *d_E;
    int *d_info;
    hipMalloc(&d_A, sizeof(float) * n * n);
    hipMalloc(&d_B, sizeof(float) * n * n);
    hipMalloc(&d_D, sizeof(float) * n);
    hipMalloc(&d_E, sizeof(float) * n);
    hipMalloc(&d_info, sizeof(int));
    
    // 4. Copy data to device
    hipMemcpy(d_A, h_A.data(), sizeof(float) * n * n, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(float) * n * n, hipMemcpyHostToDevice);
    
    // 5. Solve generalized eigenvalue problem
    // itype=1: A*x = λ*B*x
    // evect=original: Compute eigenvalues and eigenvectors
    // uplo=lower: Use lower triangle of A and B
    rocsolver_ssygv(handle, rocblas_eform_ax, rocblas_evect_original,
                    rocblas_fill_lower, n, d_A, n, d_B, n, d_D, d_E, d_info);
    
    // 6. Check for errors
    hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    
    if(h_info > 0 && h_info <= n) {
        std::cout << "Error: B is not positive definite at position " 
                  << h_info << std::endl;
        // Cleanup and exit
    }
    else if(h_info > n) {
        int failed = h_info - n;
        std::cout << "Warning: " << failed << " eigenvalues failed to converge" 
                  << std::endl;
        // Results may be partially valid
    }
    else {
        std::cout << "Success: All eigenvalues computed" << std::endl;
    }
    
    // 7. Copy results back to host
    hipMemcpy(h_D.data(), d_D, sizeof(float) * n, hipMemcpyDeviceToHost);
    hipMemcpy(h_A.data(), d_A, sizeof(float) * n * n, hipMemcpyDeviceToHost);
    
    // h_D now contains eigenvalues in ascending order
    // h_A now contains corresponding eigenvectors (columns)
    
    std::cout << "First 5 eigenvalues: ";
    for(int i = 0; i < 5; i++)
        std::cout << h_D[i] << " ";
    std::cout << std::endl;
    
    // 8. Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_D); hipFree(d_E); hipFree(d_info);
    rocblas_destroy_handle(handle);
}
```

**Key points:**
- Check info: 0=success, 1-n=POTRF error, >n=SYEV convergence failure
- Matrix A destroyed and replaced with eigenvectors
- Matrix B destroyed (contains Cholesky factor)
- E array is workspace, ignore on output""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)pivots_workArr, iinfo, optim_mem);

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

rocsolver_syev_heev_template<BATCHED, STRIDED, T>(
    handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, iinfo, batch_count,
    scalars, work1, (T*)work2, (T*)work3, (T*)work4, (T**)pivots_workArr);

// combine info from POTRF with info from SYEV/HEEV
ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threads, 0, stream, info, iinfo, n,
                        batch_count);

// backtransform eigenvectors
if(evect == rocblas_evect_original) {
    // ... TRSM or TRMM ...
}"""
    }]
})

# Entry 12: L2 - Coding: estimate operation count
entries.append({
    "id": str(base_ts + 11),
    "level": "L2",
    "interface": "sygv_hegv",
    "query": "Estimate the computational complexity (FLOPs) of SYGV for solving A*x = λ*B*x with n=1000, comparing eigenvalues-only vs eigenvalues+eigenvectors.",
    "answer": """Here's an analysis of SYGV computational complexity:

**Algorithm stages and complexity:**

**1. POTRF (Cholesky factorization): O(n³/3)**
- FLOPs ≈ n³/3
- For n=1000: ≈ 333 MFLOPs

**2. SYGST (reduction to standard form): O(n³)**
- Uses blocked BLAS-3 operations
- FLOPs ≈ 2n³ (for itype=1)
- For n=1000: ≈ 2000 MFLOPs

**3a. SYEV eigenvalues only: O(n³)**
- SYTRD (tridiagonalization): ≈ (4/3)n³
- STERF (eigenvalues of tridiagonal): ≈ 30n²
- Total: ≈ (4/3)n³
- For n=1000: ≈ 1333 MFLOPs

**3b. SYEV eigenvalues + eigenvectors: O(n³)**
- SYTRD (tridiagonalization): ≈ (4/3)n³
- ORGTR (generate Q): ≈ (4/3)n³
- STEQR (eigenvalues + vectors): ≈ 6n³
- Total: ≈ (22/3)n³
- For n=1000: ≈ 7333 MFLOPs

**4. Back-transformation (if eigenvectors): O(n³)**
- TRSM: ≈ n³
- For n=1000: ≈ 1000 MFLOPs

**Total FLOPs:**

**Eigenvalues only:**
```
Total ≈ n³/3 + 2n³ + (4/3)n³ = (11/3)n³
For n=1000: ≈ 3666 MFLOPs
```

**Eigenvalues + eigenvectors:**
```
Total ≈ n³/3 + 2n³ + (22/3)n³ + n³ = (31/3)n³
For n=1000: ≈ 10,333 MFLOPs
```

**Ratio:** Eigenvectors add ≈ 2.8× more computation

**Code estimate:**
```cpp
size_t estimate_sygv_flops(int n, bool compute_vectors) {
    size_t n3 = (size_t)n * n * n;
    
    size_t potrf_flops = n3 / 3;
    size_t sygst_flops = 2 * n3;
    
    size_t syev_flops, backtransform_flops = 0;
    
    if(compute_vectors) {
        syev_flops = (22 * n3) / 3;
        backtransform_flops = n3;
    }
    else {
        syev_flops = (4 * n3) / 3;
    }
    
    return potrf_flops + sygst_flops + syev_flops + backtransform_flops;
}
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygv_hegv.hpp",
        "language": "cpp",
        "content": """// perform Cholesky factorization of B  [O(n³/3)]
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(...);

// reduce to standard eigenvalue problem  [O(2n³)]
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(...);

// solve standard eigenvalue problem  [O(n³) to O(7n³) depending on evect]
rocsolver_syev_heev_template<BATCHED, STRIDED, T>(...);

// backtransform eigenvectors  [O(n³)]
if(evect == rocblas_evect_original) {
    // TRSM or TRMM
}"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_sygv_hegv.jsonl"
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
    if entry.get('interface') != 'sygv_hegv':
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
    if 'implement' in entry.get('query', '').lower() or 'estimate' in entry.get('query', '').lower():
        coding_count += 1

print(f"\nLevel distribution: L1={level_counts['L1']}, L2={level_counts['L2']}, L3={level_counts['L3']}")
print(f"Coding-oriented tasks: {coding_count}")
