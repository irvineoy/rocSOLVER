import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYGST/HEGST algorithm overview
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "sygst_hegst",
    "query": "Explain the difference between SYGST/HEGST (blocked) and SYGS2/HEGS2 (unblocked). How does the blocked algorithm improve performance?",
    "answer": """SYGST/HEGST is the blocked version of SYGS2/HEGS2, transforming generalized eigenvalue problems to standard form using Level 3 BLAS for better performance.

**Algorithm selection:**
- n ≤ xxGST_BLOCKSIZE: Calls SYGS2/HEGS2 (unblocked)
- n > xxGST_BLOCKSIZE: Uses blocked algorithm with block size nb = xxGST_BLOCKSIZE

**Blocked algorithm structure (itype=1, uplo=upper):**
For each block k = 0, nb, 2*nb, ...:
1. **Transform diagonal block**: Call SYGS2 on A(k:k+kb, k:k+kb) and B(k:k+kb, k:k+kb)
2. **Update off-diagonal blocks** (if k+kb < n):
   - TRSM: Solve U(k,k)^H * A(k, k+kb:n) = A(k, k+kb:n)
   - SYMM/HEMM: A(k, k+kb:n) -= 0.5 * A(k,k) * B(k, k+kb:n)
   - SYR2K/HER2K: A(k+kb:n, k+kb:n) -= A(k, k+kb:n)^H * B(k, k+kb:n) + ...
   - SYMM/HEMM: Second adjustment
   - TRSM: Solve A(k, k+kb:n) * U(k+kb:n, k+kb:n) = A(k, k+kb:n)

**Performance improvement:**
- SYGS2: O(n³) using BLAS-2 (SYR2/HER2, AXPY, SCAL, TRSV/TRMV)
- SYGST: O(n³) using BLAS-3 (SYR2K/HER2K, SYMM/HEMM, TRSM, TRMM)
- BLAS-3 operations achieve better cache utilization and higher FLOPS
- Typical speedup: 2-5× for large matrices (n > 1024)

**Block processing:**
Diagonal blocks (kb×kb) use unblocked SYGS2, off-diagonal updates use blocked BLAS-3 operations on larger submatrices.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """// if the matrix is too small, use the unblocked variant of the algorithm
if(n <= nb)
    return rocsolver_sygs2_hegs2_template<BATCHED, T>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        scalars, work_x_temp, store_wcs_invA, (T**)workArr_temp_arr);"""
    }, {
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """// Compute inv(U')*A*inv(U)
for(rocblas_int k = 0; k < n; k += nb)
{
    rocblas_int kb = std::min(n - k, nb);

    rocsolver_sygs2_hegs2_template<BATCHED, T>(
        handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
        shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
        store_wcs_invA, (T**)workArr_temp_arr);

    if(k + kb < n)
    {
        // TRSM, SYMM/HEMM, SYR2K/HER2K, SYMM/HEMM, TRSM
        // ... Level 3 BLAS operations on trailing submatrices ...
    }
}"""
    }]
})

# Entry 2: L2 - SYR2K/HER2K rank-2k update
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "Explain the SYR2K/HER2K operation in SYGST for itype=1, uplo=upper. What does it compute and why is it a rank-2k update?",
    "answer": """SYR2K/HER2K (Symmetric/Hermitian Rank-2K update) performs a critical Level 3 BLAS transformation on the trailing submatrix in blocked SYGST.

**Operation (itype=1, uplo=upper, block k):**
```
C = α*(A*B^H + B*A^H) + β*C
```

where:
- C = A(k+kb:n, k+kb:n) (trailing submatrix)
- A = A(k, k+kb:n)^H (kb×(n-k-kb) block)
- B = B(k, k+kb:n)^H (kb×(n-k-kb) block)
- α = -1 (t_minone)
- β = 1 (s_one, real type S)
- trans = conjugate_transpose

**Mathematical form:**
```
A(k+kb:n, k+kb:n) -= A(k, k+kb:n)^H * B(k, k+kb:n) + B(k, k+kb:n)^H * A(k, k+kb:n)
```

**Why "rank-2k":**
- Two matrices (A and B) each with k=kb columns
- Outer product of k-column matrices produces rank-2k update
- Symmetric/Hermitian result (sum of X*Y^H + Y*X^H is always Hermitian)

**Why needed:**
Implements the double contribution from both left and right multiplications in the similarity transformation inv(U^T)*A*inv(U). The two terms capture:
1. Left multiplication by inv(U^T)
2. Right multiplication by inv(U)

**Performance:**
- BLAS-3 operation: O(k*n²) with high computational intensity
- Much faster than equivalent BLAS-2 (HER2) loop
- Efficiently utilizes matrix multiplication hardware""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocblasCall_syr2k_her2k<BATCHED, T>(
    handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
    &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
    shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
    shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);"""
    }]
})

# Entry 3: L2 - SYMM/HEMM operations
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "What is the purpose of the two SYMM/HEMM calls in SYGST (before and after SYR2K), and why are they called with α=-0.5?",
    "answer": """The two SYMM/HEMM (Symmetric/Hermitian Matrix-Matrix multiply) calls implement a correction term in the blocked transformation.

**SYMM/HEMM operation:**
```
C = α*A*B + β*C  (side=left)
C = α*B*A + β*C  (side=right)
```

**For itype=1, uplo=upper, block k:**

**First SYMM (before SYR2K):**
```
A(k, k+kb:n) -= 0.5 * A(k,k) * B(k, k+kb:n)
```
- side: left
- α: -0.5 (t_minhalf)
- A matrix: A(k,k) (kb×kb symmetric/Hermitian block, already transformed)
- B matrix: B(k, k+kb:n) (kb×(n-k-kb) block)
- C matrix: A(k, k+kb:n) (output, in-place update)

**Second SYMM (after SYR2K, identical to first):**
```
A(k, k+kb:n) -= 0.5 * A(k,k) * B(k, k+kb:n)
```

**Why α=-0.5:**
This comes from the mathematical expansion of inv(U^T)*A*inv(U). The factor 0.5 appears because:
- The similarity transformation produces quadratic terms
- These terms need to be split between the two SYMM calls
- Each contributes half the correction

**Why two identical SYMM calls:**
Sandwiching SYR2K between two SYMM calls maintains numerical stability and follows the LAPACK reference algorithm structure. The intermediate updates ensure correct accumulation order.

**Performance:**
BLAS-3 operation with computational complexity O(kb²*(n-k-kb)).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                      &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                      shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                      shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);

rocblasCall_syr2k_her2k<BATCHED, T>(
    handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
    &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
    shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
    shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                      &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                      shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                      shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);"""
    }]
})

# Entry 4: L2 - TRSM operations in itype=1
entries.append({
    "id": str(base_ts + 3),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "Explain the two TRSM calls in SYGST for itype=1, uplo=upper. What systems do they solve?",
    "answer": """The two TRSM calls solve triangular systems to complete the transformation inv(U^T)*A*inv(U).

**First TRSM (left multiply by inv(U^H)):**
```
U(k,k)^H * X = A(k, k+kb:n)
```
Solving for X, which overwrites A(k, k+kb:n)

Parameters:
- Side: left
- Trans: conjugate_transpose (U^H)
- Diag: non_unit
- Dimensions: kb × (n-k-kb)
- Triangular matrix: B(k,k) (upper Cholesky factor)
- Right-hand side: A(k, k+kb:n) (kb rows)

**Second TRSM (right multiply by inv(U)):**
```
X * U(k+kb:n, k+kb:n) = A(k, k+kb:n)
```
Solving for X, which overwrites A(k, k+kb:n)

Parameters:
- Side: right
- Trans: none (no transpose)
- Diag: non_unit
- Dimensions: kb × (n-k-kb)
- Triangular matrix: B(k+kb:n, k+kb:n) (trailing Cholesky block)
- Right-hand side: A(k, k+kb:n) (kb rows)

**Combined effect:**
Together with SYMM and SYR2K, these implement:
```
A(k, k+kb:n) := inv(U(k,k)^T) * original_A(k, k+kb:n) * inv(U(k+kb:n, k+kb:n))
```

**Workspace:**
Both TRSM calls may need workspace (work_x_temp, invA_arr, etc.) for optimal performance.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_diagonal_non_unit, kb, n - k - kb, B, shiftB + idx2D(k, k, ldb),
    ldb, strideB, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count,
    optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);

// ... SYMM, SYR2K, SYMM operations ...

rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
    kb, n - k - kb, B, shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
    shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count, optim_mem,
    work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);"""
    }]
})

# Entry 5: L2 - itype=2 TRMM operations
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "How does itype=2 differ from itype=1 in SYGST? What are TRMM operations and when are they used?",
    "answer": """itype=2/3 computes forward transformation (U*A*U^T or L^T*A*L) using TRMM instead of TRSM.

**TRMM (Triangular Matrix-Matrix multiply):**
```
C = α*op(A)*B  (side=left)
C = α*B*op(A)  (side=right)
```
where op(A) can be A, A^T, or A^H, and A is triangular.

**For itype=2, uplo=upper, block k:**

**First TRMM (left multiply by U):**
```
Y = U(0:k, 0:k) * A(0:k, k:k+kb)
```
- Side: left
- Trans: none
- A: B(0:k, 0:k) (upper Cholesky factor)
- B: A(0:k, k:k+kb)
- Overwrites A(0:k, k:k+kb) in-place

**Second TRMM (right multiply by U^H):**
```
Y = A(0:k, k:k+kb) * U(k,k)^H
```
- Side: right
- Trans: conjugate_transpose
- A: B(k,k) (diagonal Cholesky block)
- B: A(0:k, k:k+kb)

**Key differences from itype=1:**
- itype=1: TRSM (solves) → implements inv(U)
- itype=2: TRMM (multiplies) → implements U
- itype=1: Updates trailing blocks first, diagonal last
- itype=2: Updates leading blocks first, diagonal last
- itype=2: No workspace needed for TRSM (TRMM is simpler)

**Block processing order (itype=2):**
1. TRMM operations on leading block
2. SYMM/HEMM adjustments
3. SYR2K/HER2K update
4. SYMM/HEMM adjustments
5. TRMM operations
6. SYGS2 on diagonal block (last, not first)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """// Compute U*A*U'
for(rocblas_int k = 0; k < n; k += nb)
{
    rocblas_int kb = std::min(n - k, nb);

    rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none,
                     rocblas_diagonal_non_unit, k, kb, &t_one, 0, B, shiftB, ldb,
                     strideB, A, shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                     (T**)workArr_temp_arr);

    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                          shiftA + idx2D(k, k, lda), lda, strideA, B,
                          shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                          shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

    rocblasCall_syr2k_her2k<BATCHED, T>(
        handle, uplo, rocblas_operation_none, k, kb, &t_one, A,
        shiftA + idx2D(0, k, lda), lda, strideA, B, shiftB + idx2D(0, k, ldb), ldb,
        strideB, &s_one, A, shiftA, lda, strideA, batch_count);

    // ... second SYMM, TRMM, then SYGS2 ...
}"""
    }]
})

# Entry 6: L3 - Workspace calculation
entries.append({
    "id": str(base_ts + 5),
    "level": "L3",
    "interface": "sygst_hegst",
    "query": "Explain the workspace allocation strategy in SYGST. How does it differ for n < xxGST_BLOCKSIZE vs n ≥ xxGST_BLOCKSIZE, and for itype=1 vs itype=2?",
    "answer": """SYGST workspace allocation adapts to matrix size and problem type.

**Small matrices (n < xxGST_BLOCKSIZE):**
Calls SYGS2 directly, using SYGS2 workspace:
- size_scalars: 3 * sizeof(T)
- size_work_x_temp: 0 (itype=1) or n*batch_count*sizeof(T) (itype=2/3)
- size_store_wcs_invA: 3*batch_count*sizeof(T) (plus max with TRSV/TRMV requirements)
- size_workArr_temp_arr: 0 (non-batched) or batch_count*sizeof(T*) (batched)
- size_invA_arr: 0
- optim_mem: true

**Large matrices (n ≥ xxGST_BLOCKSIZE):**

**Base requirements (both itype):**
Workspace for SYGS2 on kb×kb diagonal blocks

**Additional for itype=1:**
Two TRSM workspace requirements:
- Left TRSM: kb × (n-kb) dimensions
- Right TRSM: kb × (n-kb) dimensions
Takes maximum of both for work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr

**Code:**
```cpp
rocsolver_trsm_mem<BATCHED, STRIDED, T>(
    rocblas_side_left, rocblas_operation_conjugate_transpose, n - kb, kb,
    batch_count, &temp1, &temp2, &temp3, &temp4, optim_mem);
rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none,
                                        n - kb, kb, batch_count, &temp5, &temp6,
                                        &temp7, &temp8, optim_mem);

*size_work_x_temp = max(*size_work_x_temp, max(temp1, temp5));
// ... similar for other workspace components
```

**Additional for itype=2/3:**
None (TRMM doesn't need extra workspace beyond SYGS2 requirements)
optim_mem: true

**Workspace components:**
1. **size_scalars**: Device constants for BLAS calls
2. **size_work_x_temp**: TRSM/TRMV temporary arrays
3. **size_workArr_temp_arr**: Batched pointer arrays
4. **size_store_wcs_invA**: Coefficient storage + TRSM/TRSV workspace
5. **size_invA_arr**: TRSM inverse arrays (itype=1 only)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """if(n < xxGST_BLOCKSIZE)
{
    // requirements for calling a single SYGS2/HEGS2
    rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, n, batch_count, size_scalars,
                                                    size_work_x_temp, size_store_wcs_invA,
                                                    size_workArr_temp_arr);
    *size_invA_arr = 0;
    *optim_mem = true;
}
else
{
    rocblas_int kb = xxGST_BLOCKSIZE;
    size_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

    // requirements for calling SYGS2/HEGS2 for the subblocks
    rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, kb, batch_count, size_scalars,
                                                    size_work_x_temp, size_store_wcs_invA,
                                                    size_workArr_temp_arr);
    *size_invA_arr = 0;

    if(itype == rocblas_eform_ax)
    {
        // extra requirements for calling TRSM
        if(uplo == rocblas_fill_upper)
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_left, rocblas_operation_conjugate_transpose, n - kb, kb,
                batch_count, &temp1, &temp2, &temp3, &temp4, optim_mem);
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none,
                                                    n - kb, kb, batch_count, &temp5, &temp6,
                                                    &temp7, &temp8, optim_mem);
        }
        // ... similar for lower ...

        *size_work_x_temp = std::max(*size_work_x_temp, std::max(temp1, temp5));
        *size_workArr_temp_arr = std::max(*size_workArr_temp_arr, std::max(temp2, temp6));
        *size_store_wcs_invA = std::max(*size_store_wcs_invA, std::max(temp3, temp7));
        *size_invA_arr = std::max(*size_invA_arr, std::max(temp4, temp8));
    }
    else
        *optim_mem = true;
}"""
    }]
})

# Entry 7: L1 - Host vs device pointer mode
entries.append({
    "id": str(base_ts + 6),
    "level": "L1",
    "interface": "sygst_hegst",
    "query": "Why does SYGST use host pointer mode while SYGS2 uses device pointer mode?",
    "answer": """SYGST and SYGS2 use different pointer modes due to where their scalars are allocated.

**SYGS2 (device pointer mode):**
```cpp
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
```
- Scalars allocated in device memory via init_scalars()
- Device pointers passed to BLAS calls
- Avoids host-device transfers

**SYGST (host pointer mode):**
```cpp
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

S s_one = 1;
T t_one = 1;
T t_half = 0.5;
T t_minone = -1;
T t_minhalf = -0.5;
```
- Scalars defined on host stack
- Host pointers (&t_one, &t_minhalf, etc.) passed to BLAS calls
- Simpler allocation (no device memory needed for scalars)

**Why the difference:**

**SYGS2:**
- Column-by-column processing with many BLAS calls
- Same scalars reused frequently
- Device allocation amortizes cost

**SYGST:**
- Block-by-block processing with fewer BLAS calls
- Multiple different scalar values needed
- Host pointers simpler and sufficient

**Both:**
Save and restore original pointer mode:
```cpp
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
// ... set mode, do work ...
rocblas_set_pointer_mode(handle, old_mode);
```

**Note:** SYGST still allocates device scalars array for SYGS2 calls on diagonal blocks.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the host
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

S s_one = 1;
T t_one = 1;
T t_half = 0.5;
T t_minone = -1;
T t_minhalf = -0.5;

// ... BLAS calls using &t_one, &t_minhalf, etc. ...

rocblas_set_pointer_mode(handle, old_mode);"""
    }]
})

# Entry 8: L1 - Block size parameter
entries.append({
    "id": str(base_ts + 7),
    "level": "L1",
    "interface": "sygst_hegst",
    "query": "What is xxGST_BLOCKSIZE and how does it affect SYGST performance?",
    "answer": """xxGST_BLOCKSIZE is a compile-time constant defining the block size for SYGST's blocked algorithm.

**Role:**
- Threshold for algorithm selection: n ≤ xxGST_BLOCKSIZE → use SYGS2
- Block size for blocked algorithm: kb = xxGST_BLOCKSIZE
- Tuning parameter for performance optimization

**Performance impact:**

**Too small (e.g., 16):**
- More block iterations
- Less BLAS-3 efficiency (small matrix sizes)
- Higher kernel launch overhead

**Too large (e.g., 256):**
- Fewer blocks benefit from BLAS-3
- More work in SYGS2 (BLAS-2)
- Larger workspace requirements

**Optimal range:**
Typically 32-128 depending on:
- GPU architecture (cache sizes, warp size)
- Matrix data type (float vs double vs complex)
- BLAS library tuning

**Code usage:**
```cpp
rocblas_int nb = xxGST_BLOCKSIZE;

if(n <= nb)
    // Use unblocked SYGS2
else
    // Use blocked SYGST with block size nb
```

**Block size variability:**
Some implementations use adaptive block sizes based on matrix dimension, but rocSOLVER uses fixed xxGST_BLOCKSIZE for simplicity and predictability.

**Workspace dependency:**
Workspace size calculations use xxGST_BLOCKSIZE (referred to as kb in workspace functions).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocblas_int nb = xxGST_BLOCKSIZE;

// if the matrix is too small, use the unblocked variant of the algorithm
if(n <= nb)
    return rocsolver_sygs2_hegs2_template<BATCHED, T>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        scalars, work_x_temp, store_wcs_invA, (T**)workArr_temp_arr);"""
    }, {
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocblas_int kb = xxGST_BLOCKSIZE;
size_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

// requirements for calling SYGS2/HEGS2 for the subblocks
rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, kb, batch_count, size_scalars,
                                                size_work_x_temp, size_store_wcs_invA,
                                                size_workArr_temp_arr);"""
    }]
})

# Entry 9: L2 - Lower triangular algorithm differences
entries.append({
    "id": str(base_ts + 8),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "How does the algorithm differ between uplo=upper and uplo=lower for itype=1 in SYGST?",
    "answer": """The upper and lower algorithms are mathematically equivalent but process different triangular storage.

**uplo=upper (computes inv(U^T)*A*inv(U)):**
- Processes blocks left-to-right, top-to-bottom
- Updates A(k, k+kb:n) (row blocks to the right)
- First TRSM: side=left, trans=conjugate_transpose
- Second TRSM: side=right, trans=none
- SYMM: side=left
- SYR2K: trans=conjugate_transpose

**uplo=lower (computes inv(L)*A*inv(L^T)):**
- Processes blocks top-to-bottom, left-to-right
- Updates A(k+kb:n, k) (column blocks below)
- First TRSM: side=right, trans=conjugate_transpose
- Second TRSM: side=left, trans=none
- SYMM: side=right
- SYR2K: trans=none

**Key differences:**

**TRSM sides swapped:**
- Upper: left then right
- Lower: right then left

**SYMM side:**
- Upper: side=left (A*B where A is symmetric block)
- Lower: side=right (B*A where A is symmetric block)

**SYR2K transpose:**
- Upper: conjugate_transpose (operates on rows)
- Lower: none (operates on columns)

**Matrix indexing:**
- Upper: idx2D(k, k+kb, lda) for row k, columns k+kb:n
- Lower: idx2D(k+kb, k, lda) for rows k+kb:n, column k

**Result:**
Both produce C = inv(L*L^T) * A * inv(L*L^T) but exploit different storage patterns.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_upper)
{
    // Compute inv(U')*A*inv(U)
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
        rocblas_diagonal_non_unit, kb, n - k - kb, B, shiftB + idx2D(k, k, ldb),
        ldb, strideB, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count,
        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
    
    rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                          shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                          shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);
}
else
{
    // Compute inv(L)*A*inv(L')
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
        rocblas_diagonal_non_unit, n - k - kb, kb, B, shiftB + idx2D(k, k, ldb),
        ldb, strideB, A, shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count,
        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
    
    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, n - k - kb, kb,
                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                          shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &t_one, A,
                          shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count);
}"""
    }]
})

# Entry 10: L3 - Complete workflow
entries.append({
    "id": str(base_ts + 9),
    "level": "L3",
    "interface": "sygst_hegst",
    "query": "Provide a complete example of solving a generalized eigenvalue problem A*x = λ*B*x using POTRF, SYGST, and SYEV.",
    "answer": """Here's a complete workflow for solving generalized eigenvalue problems on GPU:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

void solve_generalized_eigenproblem(int n) {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // 1. Allocate memory (assume A, B initialized on host)
    float *d_A, *d_B;
    float *d_W;  // Eigenvalues
    int *d_info;
    hipMalloc(&d_A, sizeof(float) * n * n);
    hipMalloc(&d_B, sizeof(float) * n * n);
    hipMalloc(&d_W, sizeof(float) * n);
    hipMalloc(&d_info, sizeof(int));
    
    // Copy A and B to device...
    
    // 2. Cholesky factorization of B: B = L*L^T
    rocsolver_spotrf(handle, rocblas_fill_lower, n, d_B, n, d_info);
    
    // Check info for positive definiteness
    int h_info;
    hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    if(h_info != 0) {
        printf("POTRF failed: B is not positive definite\\n");
        // cleanup and exit
    }
    
    // 3. Reduce to standard form: C = inv(L)*A*inv(L^T)
    // d_B now contains L (lower triangle)
    // itype=1: A*x = λ*B*x
    rocsolver_ssygst(handle, rocblas_eform_ax, rocblas_fill_lower, n, 
                     d_A, n, d_B, n);
    // d_A now contains C (standard eigenvalue problem)
    
    // 4. Solve standard eigenvalue problem: C*y = λ*y
    rocsolver_ssyev(handle, rocblas_evect_original, rocblas_fill_lower,
                    n, d_A, n, d_W, d_info);
    // d_W contains eigenvalues
    // d_A contains eigenvectors y
    
    // 5. Back-transform eigenvectors: x = inv(L^T)*y
    // For itype=1, uplo=lower: x = inv(L^T)*y
    rocsolver_trsm_lower<false, false, float>(
        handle, rocblas_side_left, rocblas_operation_transpose,
        rocblas_diagonal_non_unit, n, n, d_B, 0, n, 0,
        d_A, 0, n, 0, 1, true, nullptr, nullptr, nullptr, nullptr);
    // d_A now contains original eigenvectors x
    
    // 6. Copy results back to host
    float h_W[n];
    hipMemcpy(h_W, d_W, sizeof(float) * n, hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_W); hipFree(d_info);
    rocblas_destroy_handle(handle);
}
```

**Key steps:**
1. POTRF: Factor B = L*L^T (or U^T*U)
2. SYGST: Transform A → C = inv(L)*A*inv(L^T)
3. SYEV: Solve C*y = λ*y
4. Back-transform: x = inv(L^T)*y (eigenvectors of original problem)

**Note:** Back-transformation step needed only if eigenvectors requested.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """// Compute inv(L)*A*inv(L')
for(rocblas_int k = 0; k < n; k += nb)
{
    rocblas_int kb = std::min(n - k, nb);

    rocsolver_sygs2_hegs2_template<BATCHED, T>(
        handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
        shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
        store_wcs_invA, (T**)workArr_temp_arr);

    if(k + kb < n)
    {
        // TRSM, SYMM, SYR2K, SYMM, TRSM operations on off-diagonal blocks
        // ... (transforms trailing submatrix)
    }
}"""
    }]
})

# Entry 11: L1 - Real vs complex type handling
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "sygst_hegst",
    "query": "How does SYGST handle real (symmetric) vs complex (Hermitian) matrices differently?",
    "answer": """SYGST/HEGST uses template specialization and type-aware BLAS calls to handle real and complex matrices.

**Type selection:**
```cpp
using S = decltype(std::real(T{}));
```
- T: Element type (float, double, complex<float>, complex<double>)
- S: Real type extracted from T (used for SYR2K/HER2K beta parameter)

**BLAS operation selection:**
- Real: SYR2K, SYMM (symmetric)
- Complex: HER2K, HEMM (Hermitian)

**Example - SYR2K/HER2K:**
```cpp
rocblasCall_syr2k_her2k<BATCHED, T>(
    handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
    &t_minone,  // alpha: T type
    A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
    shiftB + idx2D(k, k + kb, ldb), ldb, strideB, 
    &s_one,     // beta: S (real) type
    A, shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);
```

**Alpha vs Beta types:**
- Alpha (t_minone): Type T (can be complex)
- Beta (s_one): Type S (always real, since C is symmetric/Hermitian)

**Conjugate transpose:**
- Real: conjugate_transpose == transpose (conjugate is no-op)
- Complex: conjugate_transpose performs both transpose and conjugation

**SYGS2 calls:**
SYGS2/HEGS2 internally handles LACGV (conjugation) for complex Hermitian matrices when accessing row data stored as columns.

**C API:**
Separate functions for real and complex:
- rocsolver_ssygst, rocsolver_dsygst (symmetric)
- rocsolver_chegst, rocsolver_zhegst (Hermitian)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.cpp",
        "language": "cpp",
        "content": """template <typename T, typename U>
rocblas_status rocsolver_sygst_hegst_impl(...)
{
    const char* name = (!rocblas_is_complex<T> ? "sygst" : "hegst");
    ROCSOLVER_ENTER_TOP(name, ...);

    using S = decltype(std::real(T{}));
    
    // ... implementation uses both T and S types
}"""
    }, {
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """S s_one = 1;
T t_one = 1;
T t_half = 0.5;
T t_minone = -1;
T t_minhalf = -0.5;

// ...

rocblasCall_syr2k_her2k<BATCHED, T>(
    handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
    &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
    shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
    shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);"""
    }]
})

# Entry 12: L2 - Coding task: determine algorithm choice
entries.append({
    "id": str(base_ts + 11),
    "level": "L2",
    "interface": "sygst_hegst",
    "query": "Implement a function that determines whether SYGST will use the blocked or unblocked algorithm, and estimates the number of BLAS-3 operations for itype=1.",
    "answer": """Here's an implementation analyzing SYGST algorithm choice and operation count:

```cpp
#include <iostream>

// Assume this constant is defined in the library
#ifndef xxGST_BLOCKSIZE
#define xxGST_BLOCKSIZE 64
#endif

struct SygstAnalysis {
    bool use_blocked;
    int num_blocks;
    int num_sygs2_calls;
    int num_trsm_calls;
    int num_symm_calls;
    int num_syr2k_calls;
    size_t total_flops;  // Approximate
};

SygstAnalysis analyze_sygst(int n, int itype) {
    SygstAnalysis result;
    int nb = xxGST_BLOCKSIZE;
    
    // Algorithm selection
    result.use_blocked = (n > nb);
    
    if(!result.use_blocked) {
        // Unblocked SYGS2
        result.num_blocks = 0;
        result.num_sygs2_calls = 1;
        result.num_trsm_calls = 0;
        result.num_symm_calls = 0;
        result.num_syr2k_calls = 0;
        result.total_flops = 2 * n * n * n;  // Approximate O(n³)
    }
    else {
        // Blocked SYGST
        result.num_blocks = (n - 1) / nb + 1;
        result.num_sygs2_calls = result.num_blocks;
        
        if(itype == 1) {  // rocblas_eform_ax
            // Each block (except last) performs BLAS-3 operations
            int full_blocks = result.num_blocks - 1;
            result.num_trsm_calls = 2 * full_blocks;      // Left and right TRSM
            result.num_symm_calls = 2 * full_blocks;      // Before and after SYR2K
            result.num_syr2k_calls = full_blocks;
            
            // Approximate FLOPs (dominant terms)
            // TRSM: 2 * kb*(n-kb)^2 per call
            // SYMM: kb^2*(n-kb) per call
            // SYR2K: 2*kb*(n-kb)^2 per call
            size_t avg_trail = n - nb / 2;  // Average trailing size
            result.total_flops = full_blocks * (
                2 * (nb * avg_trail * avg_trail) +  // 2 TRSM
                2 * (nb * nb * avg_trail) +          // 2 SYMM
                2 * (nb * avg_trail * avg_trail)     // SYR2K
            );
        }
        else {  // itype=2 or 3
            int full_blocks = result.num_blocks;
            result.num_trsm_calls = 0;  // Uses TRMM instead
            result.num_symm_calls = 2 * full_blocks;
            result.num_syr2k_calls = full_blocks;
            result.total_flops = 2 * n * n * n;  // Approximate
        }
    }
    
    return result;
}

int main() {
    int sizes[] = {32, 128, 1024};
    
    for(int n : sizes) {
        SygstAnalysis a1 = analyze_sygst(n, 1);
        
        std::cout << "n=" << n << ":\\n";
        std::cout << "  Algorithm: " << (a1.use_blocked ? "Blocked" : "Unblocked") << "\\n";
        std::cout << "  Blocks: " << a1.num_blocks << "\\n";
        std::cout << "  SYGS2 calls: " << a1.num_sygs2_calls << "\\n";
        
        if(a1.use_blocked) {
            std::cout << "  TRSM calls: " << a1.num_trsm_calls << "\\n";
            std::cout << "  SYMM calls: " << a1.num_symm_calls << "\\n";
            std::cout << "  SYR2K calls: " << a1.num_syr2k_calls << "\\n";
        }
        std::cout << "\\n";
    }
    
    return 0;
}
```

**Expected output (xxGST_BLOCKSIZE=64):**
```
n=32:
  Algorithm: Unblocked
  Blocks: 0
  SYGS2 calls: 1

n=128:
  Algorithm: Blocked
  Blocks: 2
  SYGS2 calls: 2
  TRSM calls: 2
  SYMM calls: 2
  SYR2K calls: 1

n=1024:
  Algorithm: Blocked
  Blocks: 16
  SYGS2 calls: 16
  TRSM calls: 30
  SYMM calls: 30
  SYR2K calls: 15
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygst_hegst.hpp",
        "language": "cpp",
        "content": """rocblas_int nb = xxGST_BLOCKSIZE;

// if the matrix is too small, use the unblocked variant of the algorithm
if(n <= nb)
    return rocsolver_sygs2_hegs2_template<BATCHED, T>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        scalars, work_x_temp, store_wcs_invA, (T**)workArr_temp_arr);"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_sygst_hegst.jsonl"
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
    if entry.get('interface') != 'sygst_hegst':
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
    if 'implement' in entry.get('query', '').lower() or 'write' in entry.get('query', '').lower():
        coding_count += 1

print(f"\nLevel distribution: L1={level_counts['L1']}, L2={level_counts['L2']}, L3={level_counts['L3']}")
print(f"Coding-oriented tasks: {coding_count}")
