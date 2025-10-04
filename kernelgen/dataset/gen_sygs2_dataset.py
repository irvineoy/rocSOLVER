import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYGS2/HEGS2 algorithm overview
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "sygs2_hegs2",
    "query": "Explain the purpose of SYGS2/HEGS2 and the different problem types (itype) it supports. How does it reduce a generalized eigenvalue problem to standard form?",
    "answer": """SYGS2/HEGS2 reduces a generalized symmetric/Hermitian eigenvalue problem to standard form.

**Purpose:**
Transform generalized eigenvalue problems A*x = λ*B*x into standard form C*y = λ*y, where C can be solved using standard eigenvalue solvers (SYEV, HEEV, etc.).

**Three problem types (itype):**

**1. itype = rocblas_eform_ax (itype=1): A*x = λ*B*x**
- Requires B = L*L^T or B = U^T*U from Cholesky factorization (POTRF)
- uplo=upper: Computes C = inv(U^T)*A*inv(U)
- uplo=lower: Computes C = inv(L)*A*inv(L^T)
- Result: Overwrites A with C

**2. itype = rocblas_eform_abx (itype=2): A*B*x = λ*x**
- Requires B = L*L^T or B = U^T*U
- uplo=upper: Computes C = U*A*U^T
- uplo=lower: Computes C = L^T*A*L
- Result: Overwrites A with C

**3. itype = rocblas_eform_bax (itype=3): B*A*x = λ*x**
- Same as itype=2 (mathematically equivalent)
- Uses same transformations

**Workflow:**
1. User calls POTRF on B to get Cholesky factorization
2. Call SYGS2/HEGS2 to transform A in-place
3. Call SYEV/HEEV to solve standard eigenvalue problem for C
4. If needed, back-transform eigenvectors

**Algorithm:**
Unblocked column-by-column transformation using BLAS-2 operations (SCAL, AXPY, SYR2/HER2, TRSV/TRMV).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// 1. invalid/non-supported values
if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
    return rocblas_status_invalid_value;
if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    return rocblas_status_invalid_value;"""
    }, {
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """if(itype == rocblas_eform_ax)
{
    if(uplo == rocblas_fill_upper)
    {
        // Compute inv(U')*A*inv(U)
        for(rocblas_int k = 0; k < n; k++)
        {
            // ... transformation operations ...
        }
    }
    else
    {
        // Compute inv(L)*A*inv(L')
        for(rocblas_int k = 0; k < n; k++)
        {
            // ... transformation operations ...
        }
    }
}
else  // itype == abx or bax
{
    if(uplo == rocblas_fill_upper)
    {
        // Compute U*A*U'
        for(rocblas_int k = 0; k < n; k++)
        {
            // ... transformation operations ...
        }
    }
    else
    {
        // Compute L'*A*L
        for(rocblas_int k = 0; k < n; k++)
        {
            // ... transformation operations ...
        }
    }
}"""
    }]
})

# Entry 2: L2 - itype=1 upper algorithm
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "sygs2_hegs2",
    "query": "Describe the algorithm for itype=1 with uplo=upper. What operations are performed at each iteration k?",
    "answer": """For itype=1, uplo=upper, SYGS2 computes C = inv(U^T)*A*inv(U) where B = U^T*U.

**Algorithm per iteration k:**

**1. Update diagonal element A(k,k):**
```
A(k,k) = A(k,k) / (B(k,k)^2)
```
Also store: α = 1/B(k,k), β = -0.5*A(k,k) in workspace

**2. Scale column k (positions k+1:n):**
```
A(k, k+1:n) = α * A(k, k+1:n)
```

**3. Conjugate row/column if complex (LACGV)**

**4. Update with B column (AXPY):**
```
A(k, k+1:n) += β * B(k, k+1:n)
```

**5. Rank-2 update on trailing matrix (SYR2/HER2):**
```
A(k+1:n, k+1:n) -= A(k, k+1:n)^T * B(k, k+1:n) + B(k, k+1:n)^T * A(k, k+1:n)
```

**6. Second AXPY (reverse of step 4):**
```
A(k, k+1:n) += β * B(k, k+1:n)
```

**7. Triangular solve (TRSV) to finish transformation:**
```
U(k+1:n, k+1:n)^T * A(k, k+1:n)^T = A(k, k+1:n)^T
Solving for A(k, k+1:n)
```

**8. Conjugate back if complex**

The operations carefully maintain symmetry/Hermitian structure while transforming A.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// Compute inv(U')*A*inv(U)
for(rocblas_int k = 0; k < n; k++)
{
    // Set A[k, k] and store coefficients in store_wcs
    ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag1, blocks, threads, 0, stream, k, A, shiftA,
                            lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                            batch_count);

    if(k < n - 1)
    {
        rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                            shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

        if(COMPLEX)
        {
            rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                        shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                        batch_count);
            rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                        shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                        batch_count);
        }

        rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                            shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                            shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

        rocblasCall_syr2_her2<T>(
            handle, uplo, n - k - 1, scalars, A, shiftA + idx2D(k, k + 1, lda), lda,
            strideA, B, shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
            shiftA + idx2D(k + 1, k + 1, lda), lda, strideA, batch_count, workArr);

        rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                            shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                            shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                        shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                        batch_count);

        rocblasCall_trsv(handle, uplo, rocblas_operation_conjugate_transpose,
                         rocblas_diagonal_non_unit, n - k - 1, B,
                         shiftB + idx2D(k + 1, k + 1, ldb), ldb, strideB, A,
                         shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count,
                         (rocblas_int*)store_wcs, workArr);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                        shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                        batch_count);
    }
}"""
    }]
})

# Entry 3: L1 - sygs2_set_diag1 kernel
entries.append({
    "id": str(base_ts + 2),
    "level": "L1",
    "interface": "sygs2_hegs2",
    "query": "What does the sygs2_set_diag1 kernel compute and why are the coefficients stored in workspace?",
    "answer": """The sygs2_set_diag1 kernel updates the diagonal element and precomputes coefficients for itype=1.

**Computation:**
For diagonal element at position k:
1. Read A(k,k) and B(k,k)
2. Update: A(k,k) = A(k,k) / (B(k,k)^2)
3. Store coefficients in workspace (stride=3 per batch):
   - W[0] = 1 / B(k,k)  (scaling factor α)
   - W[1] = -0.5 * A(k,k)  (adjustment factor β)

**Why store coefficients:**
- These values are needed multiple times in the iteration
- W[0] is used in SCAL operation
- W[1] is used in two AXPY operations (before and after SYR2/HER2)
- Storing avoids recomputation and ensures consistency
- Device memory storage enables efficient BLAS calls with device pointers

**Parallelization:**
Each thread processes one batch instance (b = blockIdx.x * blockDim.x + threadIdx.x), operating independently.

**Memory access:**
- Reads: A[k,k], B[k,k]
- Writes: A[k,k], W[0], W[1]
- W[2] is unused in itype=1 (used in itype=2/3)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename U>
ROCSOLVER_KERNEL void sygs2_set_diag1(const rocblas_int k,
                                      U AA,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      U BB,
                                      const rocblas_int shiftB,
                                      const rocblas_int ldb,
                                      const rocblas_stride strideB,
                                      T* work,
                                      const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    constexpr rocblas_stride strideW = 3;

    if(b < batch_count)
    {
        T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
        T* B = load_ptr_batch<T>(BB, b, shiftB, strideB);
        T* W = work + b * strideW;

        T akk = A[k + k * lda];
        T bkk = B[k + k * ldb];
        akk /= bkk * bkk;
        A[k + k * lda] = akk;

        W[0] = T(1.0) / bkk;
        W[1] = T(-0.5) * akk;
    }
}"""
    }]
})

# Entry 4: L2 - SYR2/HER2 rank-2 update
entries.append({
    "id": str(base_ts + 3),
    "level": "L2",
    "interface": "sygs2_hegs2",
    "query": "Explain the SYR2/HER2 rank-2 update in SYGS2. What matrix operation does it perform and why is it needed?",
    "answer": """The SYR2/HER2 (Symmetric/Hermitian Rank-2 update) performs a critical transformation on the trailing submatrix.

**Operation (for itype=1, uplo=upper, iteration k):**
```
A(k+1:n, k+1:n) -= A(k, k+1:n)^T * B(k, k+1:n) + B(k, k+1:n)^T * A(k, k+1:n)
```

Or more generally:
```
C = C + α*(x*y^T + y*x^T)  for symmetric
C = C + α*(x*y^H + y*x^H)  for Hermitian
```

where:
- C = A(k+1:n, k+1:n) (trailing submatrix)
- x = A(k, k+1:n)^T (transformed row k)
- y = B(k, k+1:n)^T (Cholesky factor row k)
- α = -1 (from scalars array)

**Why it's needed:**
This implements the rank-2 update portion of the similarity transformation inv(U^T)*A*inv(U):
- The transformation must maintain symmetry/Hermitian structure
- Both left and right multiplications by inv(U^T) and inv(U) contribute
- The rank-2 form efficiently captures the contribution from row/column k to the trailing matrix

**Symmetry preservation:**
SYR2/HER2 only updates the upper (or lower) triangle specified by uplo, automatically maintaining the symmetric/Hermitian property.

**Performance:**
This is a BLAS-2 operation (O(n^2) per iteration), making SYGS2 O(n^3) overall.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """rocblasCall_syr2_her2<T>(
    handle, uplo, n - k - 1, scalars, A, shiftA + idx2D(k, k + 1, lda), lda,
    strideA, B, shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
    shiftA + idx2D(k + 1, k + 1, lda), lda, strideA, batch_count, workArr);"""
    }]
})

# Entry 5: L2 - itype=2 algorithm difference
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "sygs2_hegs2",
    "query": "How does the algorithm differ for itype=2 (A*B*x = λ*x) compared to itype=1? What transformation does it compute?",
    "answer": """For itype=2/3, SYGS2 computes the forward transformation (U*A*U^T or L^T*A*L) instead of the inverse transformation.

**Key differences from itype=1:**

**1. Diagonal computation (sygs2_set_diag2):**
- itype=1: A(k,k) = A(k,k) / B(k,k)^2
- itype=2: Stores W[0]=B(k,k), W[1]=0.5*A(k,k), W[2]=A(k,k)*B(k,k)^2
- Final diagonal set by sygs2_set_diag3: A(k,k) = W[2]

**2. TRMV instead of TRSV:**
- itype=1: Uses TRSV (triangular solve) to compute inv(U)*...
- itype=2: Uses TRMV (triangular matrix-vector multiply) to compute U*...

**3. Order of operations:**
For itype=2, uplo=upper (compute U*A*U^T):
- TRMV: Multiply column k by U
- AXPY: Add 0.5*A(k,k)*B column
- SYR2/HER2: Rank-2 update with α=1 (not -1)
- AXPY: Second adjustment
- SCAL: Scale by B(k,k)
- Set diagonal: A(k,k) = A(k,k)*B(k,k)^2

**4. Scalar sign:**
- itype=1: Uses α=-1 in SYR2 (subtraction)
- itype=2: Uses α=+1 in SYR2 (addition)

**Mathematical reason:**
itype=1 computes C = inv(U^T)*A*inv(U) (inverse similarity)
itype=2 computes C = U*A*U^T (forward similarity)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// Compute U*A*U'
for(rocblas_int k = 0; k < n; k++)
{
    // Store coefficients in store_wcs
    ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag2, blocks, threads, 0, stream, k, A, shiftA,
                            lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                            batch_count);

    rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, rocblas_diagonal_non_unit,
                        k, B, shiftB, ldb, strideB, A, shiftA + idx2D(0, k, lda), 1,
                        strideA, (T*)work, strideW, batch_count);

    rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                        shiftB + idx2D(0, k, ldb), 1, strideB, A,
                        shiftA + idx2D(0, k, lda), 1, strideA, batch_count);

    rocblasCall_syr2_her2<T>(handle, uplo, k, scalars + 2, A, shiftA + idx2D(0, k, lda),
                             1, strideA, B, shiftB + idx2D(0, k, ldb), 1, strideB, A,
                             shiftA, lda, strideA, batch_count, workArr);

    rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                        shiftB + idx2D(0, k, ldb), 1, strideB, A,
                        shiftA + idx2D(0, k, lda), 1, strideA, batch_count);

    rocblasCall_scal<T>(handle, k, (T*)store_wcs, strideS, A, shiftA + idx2D(0, k, lda),
                        1, strideA, batch_count);

    // Set A[k, k]
    ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag3, blocks, threads, 0, stream, k, A, shiftA,
                            lda, strideA, (T*)store_wcs, batch_count);
}"""
    }, {
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename U>
ROCSOLVER_KERNEL void sygs2_set_diag2(const rocblas_int k,
                                      U AA,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      U BB,
                                      const rocblas_int shiftB,
                                      const rocblas_int ldb,
                                      const rocblas_stride strideB,
                                      T* work,
                                      const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    constexpr rocblas_stride strideW = 3;

    if(b < batch_count)
    {
        T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
        T* B = load_ptr_batch<T>(BB, b, shiftB, strideB);
        T* W = work + b * strideW;

        T akk = A[k + k * lda];
        T bkk = B[k + k * ldb];

        W[0] = bkk;
        W[1] = T(0.5) * akk;
        W[2] = akk * (bkk * bkk);
    }
}"""
    }]
})

# Entry 6: L1 - LACGV conjugation for complex matrices
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "sygs2_hegs2",
    "query": "What is the purpose of LACGV calls in SYGS2/HEGS2, and when are they used?",
    "answer": """LACGV (Conjugate a complex vector) is used to handle complex Hermitian matrices correctly.

**Purpose:**
When processing Hermitian matrices stored in packed form (only upper or lower triangle), row operations require accessing what appears as a column (or vice versa). LACGV conjugates elements to maintain Hermitian property.

**When used:**
Only when COMPLEX=true (rocblas_float_complex or rocblas_double_complex types).

**Example in itype=1, uplo=upper:**
```cpp
if(COMPLEX)
{
    rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                batch_count);
    rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                batch_count);
}
// ... operations on conjugated elements ...
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                batch_count);
```

**Why needed:**
- Hermitian matrices satisfy A^H = A (conjugate transpose equals original)
- When stored as upper triangle, A(i,j) = conj(A(j,i))
- Row operations on row k require conjugating what's stored as column k
- After operations, conjugate back to restore storage format

**For symmetric (real) matrices:**
COMPLEX=false, so LACGV calls are skipped (no conjugation needed since elements are real).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """if(COMPLEX)
{
    rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                batch_count);
    rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                batch_count);
}

rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                    shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                    shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

// ... SYR2/HER2 and other operations ...

if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                batch_count);"""
    }]
})

# Entry 7: L3 - Workspace allocation
entries.append({
    "id": str(base_ts + 6),
    "level": "L3",
    "interface": "sygs2_hegs2",
    "query": "Explain the workspace allocation strategy in SYGS2/HEGS2. What are the different workspace components and how do their sizes depend on itype?",
    "answer": """SYGS2/HEGS2 allocates four workspace components with sizes depending on itype and matrix dimension.

**Workspace components:**

**1. scalars (size_scalars = 3 * sizeof(T)):**
Device memory for constants used in BLAS calls:
- scalars[0] = -1.0 (for itype=1 SYR2/HER2)
- scalars[2] = 1.0 (for itype=2/3 SYR2/HER2)
Initialized by init_scalars()

**2. store_wcs (size_store_wcs):**
Stores per-batch coefficients computed by sygs2_set_diag kernels:
- Minimum: 3 * batch_count * sizeof(T)
- For itype=1: max(3*batch_count*sizeof(T), sizeof(rocblas_int)*batch_count)
  (larger size needed for TRSV internal use)
- For itype=2/3: 3 * batch_count * sizeof(T)

Each batch gets 3 elements: W[0], W[1], W[2]

**3. work (size_work):**
- itype=1: 0 (TRSV doesn't need extra workspace)
- itype=2/3: n * batch_count * sizeof(T) (for TRMV result storage)

**4. workArr (size_workArr):**
- Batched mode: batch_count * sizeof(T*)
- Non-batched: 0

**Dependency on itype:**
```
if(itype == rocblas_eform_ax)  // itype=1
{
    *size_store_wcs = max(3*batch_count*sizeof(T), sizeof(rocblas_int)*batch_count);
    *size_work = 0;
}
else  // itype=2 or 3
{
    *size_work = n * batch_count * sizeof(T);
}
```

**Quick return:**
All sizes set to 0 if n=0 or batch_count=0.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// size of scalars (constants)
*size_scalars = sizeof(T) * 3;

// size of stored value array
*size_store_wcs = sizeof(T) * 3 * batch_count;

// size of array of pointers to workspace
if(BATCHED)
    *size_workArr = sizeof(T*) * batch_count;
else
    *size_workArr = 0;

if(itype == rocblas_eform_ax)
{
    // extra workspace (for calling TRSV)
    *size_store_wcs = std::max(*size_store_wcs, sizeof(rocblas_int) * batch_count);
    *size_work = 0;
}
else
{
    // extra workspace (for calling TRMV)
    *size_work = sizeof(T) * n * batch_count;
}"""
    }]
})

# Entry 8: L1 - Upper vs lower processing direction
entries.append({
    "id": str(base_ts + 7),
    "level": "L1",
    "interface": "sygs2_hegs2",
    "query": "Why does SYGS2 process the matrix differently for uplo=upper vs uplo=lower?",
    "answer": """SYGS2 processes upper vs lower triangular storage differently to match the Cholesky factor structure.

**Cholesky factorization storage:**
- uplo=upper: B = U^T*U (upper triangular U stored)
- uplo=lower: B = L*L^T (lower triangular L stored)

**Processing direction (all itype, both uplo):**
Both iterate k = 0 to n-1 (forward), but operate on different parts:

**uplo=upper (itype=1):**
- Processes row k: A(k, k+1:n)
- Updates trailing submatrix: A(k+1:n, k+1:n)
- Uses column-oriented access (lda stride)

**uplo=lower (itype=1):**
- Processes column k: A(k+1:n, k)
- Updates trailing submatrix: A(k+1:n, k+1:n)
- Uses row-oriented access (stride=1)

**Example difference in SCAL:**
```cpp
// Upper: scale row k
rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                    shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);
                    //                           ^^^^ column stride

// Lower: scale column k
rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                    shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count);
                    //                            ^ row stride
```

**Result:**
Both produce mathematically equivalent transformations, just working with different triangular storage formats.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_upper)
{
    // Compute inv(U')*A*inv(U)
    for(rocblas_int k = 0; k < n; k++)
    {
        // ... operations on A(k, k+1:n) with stride lda ...
        rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                            shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);
        // ...
    }
}
else
{
    // Compute inv(L)*A*inv(L')
    for(rocblas_int k = 0; k < n; k++)
    {
        // ... operations on A(k+1:n, k) with stride 1 ...
        rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                            shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count);
        // ...
    }
}"""
    }]
})

# Entry 9: L2 - TRSV operation
entries.append({
    "id": str(base_ts + 8),
    "level": "L2",
    "interface": "sygs2_hegs2",
    "query": "What triangular solve does TRSV perform in SYGS2 for itype=1, and why is it needed?",
    "answer": """TRSV (Triangular Solve with a vector) completes the similarity transformation for itype=1.

**For itype=1, uplo=upper:**
Solves: U(k+1:n, k+1:n)^H * x = b
where:
- U(k+1:n, k+1:n) is the trailing Cholesky factor
- b = A(k, k+1:n)^T (after previous transformations)
- x overwrites b in-place

**TRSV parameters:**
- uplo: upper (matches B storage)
- trans: conjugate_transpose (U^H)
- diag: non_unit (U has non-unit diagonal from Cholesky)
- n: n - k - 1 (size of trailing block)
- A: B matrix (Cholesky factor)
- x: A(k, k+1:n)^T

**Why needed:**
This implements the right-multiplication by inv(U):
- Previous operations handled left-multiplication by inv(U^T)
- TRSV solves U^H * x = b, giving x = inv(U^H) * b = inv(U^T) * b
- This completes: inv(U^T) * A * inv(U)

**For itype=1, uplo=lower:**
Similar but solves: L(k+1:n, k+1:n) * x = b with trans=none

**Workspace:**
store_wcs must be at least sizeof(rocblas_int)*batch_count for TRSV internal use.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """rocblasCall_trsv(handle, uplo, rocblas_operation_conjugate_transpose,
                 rocblas_diagonal_non_unit, n - k - 1, B,
                 shiftB + idx2D(k + 1, k + 1, ldb), ldb, strideB, A,
                 shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count,
                 (rocblas_int*)store_wcs, workArr);"""
    }]
})

# Entry 10: L2 - TRMV operation
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "sygs2_hegs2",
    "query": "What does TRMV compute in SYGS2 for itype=2, and how does it differ from TRSV?",
    "answer": """TRMV (Triangular Matrix-Vector multiply) performs forward multiplication for itype=2/3.

**For itype=2, uplo=upper:**
Computes: y = U(0:k-1, 0:k-1) * x
where:
- U(0:k-1, 0:k-1) is the leading k×k Cholesky factor
- x = A(0:k-1, k) (column k of A)
- y is written to workspace, then copied back

**TRMV parameters:**
- uplo: upper
- trans: none (no transpose for upper case)
- diag: non_unit
- n: k (size of leading block)
- A: B matrix (Cholesky factor)
- x: A(0:k-1, k)
- workspace: temporary storage for result

**Difference from TRSV:**
- TRSV: Solves triangular system (x = inv(T)*b), implements inverse transformation
- TRMV: Multiplies by triangular matrix (y = T*x), implements forward transformation

**Why TRMV for itype=2:**
itype=2 computes U*A*U^T (forward), not inv(U)*A*inv(U^T) (inverse)
- TRMV implements the multiplication by U
- No solving required, just matrix-vector product

**For itype=2, uplo=lower:**
Uses trans=conjugate_transpose: y = L^H * x

**Workspace usage:**
work array (size n*batch_count) stores TRMV result temporarily.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, rocblas_diagonal_non_unit,
                    k, B, shiftB, ldb, strideB, A, shiftA + idx2D(0, k, lda), 1,
                    strideA, (T*)work, strideW, batch_count);"""
    }]
})

# Entry 11: L1 - Device pointer mode
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "sygs2_hegs2",
    "query": "Why does SYGS2 use device pointer mode for scalars?",
    "answer": """SYGS2 sets device pointer mode to use pre-allocated device memory for scalar constants.

**Code:**
```cpp
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
// ... BLAS calls using device scalars ...
rocblas_set_pointer_mode(handle, old_mode);
```

**Why device mode:**
1. Scalars allocated in device memory: `init_scalars(handle, (T*)scalars)`
2. Multiple BLAS calls (SCAL, AXPY, SYR2/HER2) use same scalars
3. Device pointers avoid host-device transfers for each BLAS call
4. Improves performance by keeping scalars on device

**Scalars used:**
- scalars[0] = -1.0 (for itype=1 SYR2/HER2)
- scalars[2] = 1.0 (for itype=2/3 SYR2/HER2)
- Coefficients in store_wcs (α, β values)

**Comparison:**
- Host pointer mode: Would require copying scalar values to host before each BLAS call
- Device pointer mode: Scalars stay on device, passed as device pointers

**Best practice:**
Always save and restore original pointer mode to maintain user's expected behavior.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the device
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

// ... algorithm implementation using device scalars ...

rocblas_set_pointer_mode(handle, old_mode);"""
    }, {
        "path": "library/src/lapack/roclapack_sygs2_hegs2.cpp",
        "language": "cpp",
        "content": """scalars = mem[0];
work = mem[1];
store_wcs = mem[2];
workArr = mem[3];
if(size_scalars > 0)
    init_scalars(handle, (T*)scalars);"""
    }]
})

# Entry 12: L3 - Coding task: compute workspace size
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "sygs2_hegs2",
    "query": "Implement a function to compute the total device memory required for SYGS2 given itype, n, and batch_count. Assume non-batched mode and element type float.",
    "answer": """Here's an implementation that computes total SYGS2 workspace:

```cpp
#include <algorithm>
#include <cstddef>

enum rocblas_eform { rocblas_eform_ax = 1, rocblas_eform_abx = 2, rocblas_eform_bax = 3 };

size_t compute_sygs2_workspace(rocblas_eform itype, int n, int batch_count) {
    // Quick return for empty cases
    if(n == 0 || batch_count == 0)
        return 0;
    
    size_t size_scalars, size_work, size_store_wcs, size_workArr;
    
    // Size of scalars (constants) - always 3 floats
    size_scalars = sizeof(float) * 3;
    
    // Size of stored value array - minimum 3 per batch
    size_store_wcs = sizeof(float) * 3 * batch_count;
    
    // Size of array of pointers (0 for non-batched)
    size_workArr = 0;
    
    if(itype == rocblas_eform_ax)  // itype=1
    {
        // Extra workspace for TRSV (needs int array)
        size_store_wcs = std::max(size_store_wcs, sizeof(int) * batch_count);
        size_work = 0;  // No work array needed
    }
    else  // itype=2 or 3
    {
        // Extra workspace for TRMV
        size_work = sizeof(float) * n * batch_count;
    }
    
    // Total workspace
    size_t total = size_scalars + size_work + size_store_wcs + size_workArr;
    
    return total;
}

// Example usage:
int main() {
    int n = 1024;
    int batch_count = 1;
    
    // itype=1: needs scalars + store_wcs
    size_t bytes_itype1 = compute_sygs2_workspace(rocblas_eform_ax, n, batch_count);
    // = 12 + max(12, 4) = 12 + 12 = 24 bytes
    
    // itype=2: needs scalars + store_wcs + work
    size_t bytes_itype2 = compute_sygs2_workspace(rocblas_eform_abx, n, batch_count);
    // = 12 + 12 + 1024*4 = 24 + 4096 = 4120 bytes
    
    printf("itype=1: %zu bytes\\n", bytes_itype1);
    printf("itype=2: %zu bytes\\n", bytes_itype2);
    
    return 0;
}
```

**Key insights:**
1. itype=1 has minimal workspace (only scalars + coefficients)
2. itype=2/3 needs O(n) extra workspace for TRMV
3. Scalars are always 3 elements regardless of n
4. Store_wcs grows with batch_count, not n

**Memory breakdown for n=1024, batch=1, float:**
- itype=1: ~24 bytes
- itype=2: ~4120 bytes (1000× more due to TRMV workspace)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sygs2_hegs2.hpp",
        "language": "cpp",
        "content": """// size of scalars (constants)
*size_scalars = sizeof(T) * 3;

// size of stored value array
*size_store_wcs = sizeof(T) * 3 * batch_count;

// size of array of pointers to workspace
if(BATCHED)
    *size_workArr = sizeof(T*) * batch_count;
else
    *size_workArr = 0;

if(itype == rocblas_eform_ax)
{
    // extra workspace (for calling TRSV)
    *size_store_wcs = std::max(*size_store_wcs, sizeof(rocblas_int) * batch_count);
    *size_work = 0;
}
else
{
    // extra workspace (for calling TRMV)
    *size_work = sizeof(T) * n * batch_count;
}"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_sygs2_hegs2.jsonl"
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
    # Check required fields
    if 'id' not in entry or 'interface' not in entry:
        print(f"Entry {i}: Missing required fields")
        all_valid = False
    # Check id is timestamp
    if not entry['id'].isdigit():
        print(f"Entry {i}: ID is not a timestamp string")
        all_valid = False
    # Check interface
    if entry.get('interface') != 'sygs2_hegs2':
        print(f"Entry {i}: Incorrect interface: {entry.get('interface')}")
        all_valid = False
    # Check code_blocks have 'content' field
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
