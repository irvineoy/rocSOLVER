import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete GETRS algorithm
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "getrs",
    "query": "Explain the complete algorithm used in rocSOLVER's GETRS for solving linear systems A*X = B using LU factorization. How does it differ for trans=none vs trans=transpose?",
    "answer": """GETRS solves linear systems using the LU factorization computed by GETRF. The algorithm differs based on the trans parameter:

**trans = rocblas_operation_none (solve A*X = B):**
Given P*A = L*U from GETRF:
1. Apply row interchanges to B using ipiv (LASWP): B := P*B
2. Solve L*X = B using forward substitution (TRSM with lower, unit diagonal): B := inv(L)*B
3. Solve U*X = B using backward substitution (TRSM with upper, non-unit diagonal): B := inv(U)*B
Result: X = inv(U)*inv(L)*P*B = inv(A)*B

**trans = rocblas_operation_transpose or rocblas_operation_conjugate_transpose (solve A'*X = B or A^H*X = B):**
Given (P*A)' = U'*L' or (P*A)^H = U^H*L^H:
1. Solve U'*X = B or U^H*X = B (TRSM with upper, trans, non-unit diagonal): B := inv(U')*B
2. Solve L'*X = B or L^H*X = B (TRSM with lower, trans, unit diagonal): B := inv(L')*B
3. Apply row interchanges in reverse order (LASWP with incp=-1): B := P'*B
Result: X = P'*inv(L')*inv(U')*B = inv(A')*B

**Key differences:**
- Non-transposed: pivoting first, then L then U solves
- Transposed: U then L solves, then reverse pivoting
- All operations performed in-place on B""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """if(trans == rocblas_operation_none)
{
    // first apply row interchanges to the right hand sides
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, 1, strideP, batch_count);

    // solve L*X = B, overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                              lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // solve U*X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);
}
else
{
    // solve U'*X = B or U**H *X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // solve L'*X = B, or L**H *X = B overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                              lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // then apply row interchanges to the solution vectors
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, -1, strideP, batch_count);
}"""
    }]
})

# Entry 2: L2 - LASWP row interchange operation
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "getrs",
    "query": "What is LASWP and how does it apply row interchanges in GETRS? Explain the difference between forward (incp=1) and backward (incp=-1) pivoting.",
    "answer": """LASWP (LAPACK Swap) performs a series of row interchanges on a matrix using a pivot vector ipiv.

**Purpose in GETRS:**
Apply the permutation matrix P (encoded in ipiv from GETRF) to the right-hand side matrix B, ensuring the system is correctly permuted for solving with L and U factors.

**Forward pivoting (incp=1, used when trans=none):**
- Iterates from k1 to k2: for i = k1, k1+1, ..., k2
- Applies pivots in the order they were recorded by GETRF
- Swaps row i with row ipiv[i]
- Computes P*B

**Backward pivoting (incp=-1, used when trans!=none):**
- Iterates from k2 down to k1: for i = k2, k2-1, ..., k1
- Applies pivots in reverse order
- Computes P^T*B (transpose of permutation)

**Implementation:**
Each thread processes one column of the matrix. The kernel iterates through rows k1 to k2, swapping elements between row i and row ipiv[i] when they differ.

**Parameters in GETRS:**
- n = nrhs (number of columns in B)
- k1 = 1, k2 = n (apply all pivots)
- incp = 1 (forward) or -1 (backward)
- ipiv from GETRF (1-based indexing)""",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_laswp.hpp",
        "language": "cpp",
        "content": """I start, end, inc;
if(incp < 0)
{
    start = k2;
    end = k1 - 1;
    inc = -1;
    incp = -incp;
}
else
{
    start = k1;
    end = k2 + 1;
    inc = 1;
}

for(I i = start; i != end; i += inc)
{
    I exch = ipiv[k1 + (i - k1) * incp - 1];

    // will exchange rows i and exch if they are not the same
    if(exch != i)
        swap(A[(i - 1) * inca + tid * lda], A[(exch - 1) * inca + tid * lda]);
}"""
    }, {
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """// first apply row interchanges to the right hand sides
if(pivot)
    rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                   0, 1, strideP, batch_count);"""
    }]
})

# Entry 3: L1 - Why forward vs backward pivoting
entries.append({
    "id": str(base_ts + 2),
    "level": "L1",
    "interface": "getrs",
    "query": "Why does GETRS apply pivoting before solving when trans=none, but after solving when trans=transpose?",
    "answer": """The pivoting order depends on the mathematical structure of the transposed system:

**For A*X = B (trans=none):**
We have P*A = L*U, so A = P^T*L*U
Solving A*X = B:
- A*X = B
- (P^T*L*U)*X = B
- L*U*X = P*B (multiply both sides by P)

Therefore, we must apply P to B first, then solve L*Y = P*B, then U*X = Y.

**For A^T*X = B (trans=transpose):**
We have (P*A)^T = (L*U)^T = U^T*L^T
So A^T = U^T*L^T*P^T

Solving A^T*X = B:
- (U^T*L^T*P^T)*X = B
- U^T*L^T*Y = B (where Y = P^T*X)
- First solve U^T*Z = B
- Then solve L^T*Y = Z
- Finally X = P*Y (apply P^T^T = P in reverse)

The reverse pivoting (incp=-1) correctly computes P*Y = X.

**Summary:**
- trans=none: P*B first (forward pivoting)
- trans!=none: P*result last (backward pivoting)

This ensures mathematical correctness for both standard and transposed systems.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """if(trans == rocblas_operation_none)
{
    // first apply row interchanges to the right hand sides
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, 1, strideP, batch_count);
    // ... then solve L*X = B, then U*X = B
}
else
{
    // ... first solve U'*X = B, then L'*X = B
    // then apply row interchanges to the solution vectors
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, -1, strideP, batch_count);
}"""
    }]
})

# Entry 4: L2 - TRSM lower triangular solve
entries.append({
    "id": str(base_ts + 3),
    "level": "L2",
    "interface": "getrs",
    "query": "Explain the TRSM call that solves L*X = B in GETRS. Why is the diagonal specified as unit diagonal?",
    "answer": """The TRSM (Triangular Solve with Multiple right-hand sides) call for L solves the lower triangular system L*X = B.

**TRSM parameters:**
- Side: rocblas_side_left (L on left: L*X = B)
- Fill: lower (L is lower triangular)
- Trans: rocblas_operation_none (no transpose for trans=none case)
- Diagonal: rocblas_diagonal_unit (L has unit diagonal)
- Dimensions: m=n, n=nrhs
- Matrix L: stored in lower triangular part of A
- Matrix B: right-hand side, overwritten with solution X

**Why unit diagonal?**
In LU factorization with partial pivoting (as computed by GETRF), the convention is:
- L is lower triangular with **unit diagonal** (diagonal elements = 1)
- U is upper triangular with **non-unit diagonal**

This convention is standard in LAPACK and comes from the factorization algorithm. During LU factorization, the pivot element is divided into the rows below it, making L's diagonal implicitly 1. The actual factorization stores both L and U in the same matrix A:
- Strict lower triangle of A contains L (below diagonal)
- Upper triangle and diagonal of A contains U (diagonal and above)

**Performance benefit:**
Specifying rocblas_diagonal_unit tells TRSM to skip reading diagonal elements of L (assumed to be 1), avoiding memory accesses and enabling optimization.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """// solve L*X = B, overwriting B with X
rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                          rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                          lda, strideA, B, shiftB, incb, ldb, strideB,
                                          batch_count, optim_mem, work1, work2, work3, work4);"""
    }]
})

# Entry 5: L2 - TRSM upper triangular solve
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "getrs",
    "query": "Explain the TRSM call that solves U*X = B in GETRS. Why is the diagonal non-unit?",
    "answer": """The TRSM call for U solves the upper triangular system U*X = B.

**TRSM parameters:**
- Side: rocblas_side_left (U on left: U*X = B)
- Fill: upper (U is upper triangular)
- Trans: rocblas_operation_none (for trans=none case, or actual trans for transposed cases)
- Diagonal: rocblas_diagonal_non_unit (U has non-unit diagonal)
- Dimensions: m=n, n=nrhs
- Matrix U: stored in upper triangular part (including diagonal) of A
- Matrix B: input is result from L solve, overwritten with final solution X

**Why non-unit diagonal?**
In the LU factorization convention:
- L has unit diagonal (all 1s)
- U has **non-unit diagonal** (actual computed values)

The U diagonal contains the pivot values after factorization. These values must be:
1. Read from memory
2. Used in division during triangular solve
3. Checked for zeros (singularity detection in GETRF)

**Singularity:**
If any U(i,i) = 0, the system is singular and has no unique solution. GETRF detects this and sets info[b] = i. GETRS doesn't check singularity (assumes valid factorization from GETRF).

**Storage:**
U is stored in the upper triangle and diagonal of matrix A:
- A[i,j] for i ≤ j contains U[i,j]
- A[i,i] contains the non-unit diagonal of U""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """// solve U*X = B, overwriting B with X
rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                          rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                          inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                          batch_count, optim_mem, work1, work2, work3, work4);"""
    }]
})

# Entry 6: L3 - Workspace calculation
entries.append({
    "id": str(base_ts + 5),
    "level": "L3",
    "interface": "getrs",
    "query": "How does GETRS calculate its workspace requirements, and why does it only need TRSM workspace?",
    "answer": """GETRS workspace calculation is straightforward because it only performs TRSM operations (and LASWP which requires no workspace).

**Workspace calculation:**
```cpp
rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, nrhs, batch_count,
                                        size_work1, size_work2, size_work3, size_work4,
                                        optim_mem, lda, ldb, inca, incb);
```

**Why only TRSM workspace?**
GETRS performs three operations:
1. **LASWP** (row interchanges): No workspace needed, operates in-place by swapping rows
2. **TRSM lower** (solve L*X = B): Requires workspace for optimization
3. **TRSM upper** (solve U*X = B): Reuses the same workspace

**Workspace reuse:**
Both TRSM calls (lower and upper) execute sequentially, so they can share the same workspace. The workspace size is computed for:
- Side: left (n × nrhs problems)
- Trans: operation specified by user
- Dimensions: n × nrhs

**optim_mem flag:**
Indicates whether optimal memory allocation is used for TRSM peak performance.

**Quick return:**
If n=0 or nrhs=0 or batch_count=0, no workspace is needed (all sizes set to 0).

**Comparison to GETRI:**
GETRI needs much more workspace (for TRTRI, GEMM, TRSM), while GETRS only needs TRSM workspace, making it very memory efficient.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_getrs_getMemorySize(rocblas_operation trans,
                                   const I n,
                                   const I nrhs,
                                   const I batch_count,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   bool* optim_mem,
                                   const I lda = 1,
                                   const I ldb = 1,
                                   const I inca = 1,
                                   const I incb = 1)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *optim_mem = true;
        return;
    }

    // workspace required for calling TRSM
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, nrhs, batch_count,
                                            size_work1, size_work2, size_work3, size_work4,
                                            optim_mem, lda, ldb, inca, incb);
}"""
    }]
})

# Entry 7: L1 - 64-bit integer support
entries.append({
    "id": str(base_ts + 6),
    "level": "L1",
    "interface": "getrs",
    "query": "What is the purpose of the _64 suffix functions like rocsolver_sgetrs_64, and when are they available?",
    "answer": """The _64 suffix functions provide 64-bit integer support for solving very large linear systems.

**Purpose:**
- Support matrices with dimensions > 2^31-1 (standard 32-bit int limit)
- Use int64_t for n, nrhs, lda, ldb, and ipiv indices
- Enable solving systems with billions of equations/unknowns

**Availability:**
Controlled by HAVE_ROCBLAS_64 preprocessor macro:
```cpp
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
```

**When to use:**
- Very large-scale scientific computing (climate modeling, astrophysics, etc.)
- Systems with n > 2,147,483,647
- When using GETRF_64 for factorization (must match)

**Available variants:**
- rocsolver_sgetrs_64 (float)
- rocsolver_dgetrs_64 (double)
- rocsolver_cgetrs_64 (complex float)
- rocsolver_zgetrs_64 (complex double)

**Implementation:**
Uses template parameter I (integer type) which can be rocblas_int or int64_t. The same template implementation works for both 32-bit and 64-bit cases.

**Note:** Requires rocBLAS library built with 64-bit integer support.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgetrs_64(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   float* A,
                                   const int64_t lda,
                                   const int64_t* ipiv,
                                   float* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}"""
    }]
})

# Entry 8: L1 - Pointer mode requirement
entries.append({
    "id": str(base_ts + 7),
    "level": "L1",
    "interface": "getrs",
    "query": "Why does GETRS set the pointer mode to host during execution?",
    "answer": """GETRS sets pointer mode to host because TRSM operations require scalar parameters (alpha) that are defined on the host stack.

**Code:**
```cpp
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
// ... TRSM calls with implicit alpha = 1 ...
rocblas_set_pointer_mode(handle, old_mode);
```

**Why this is needed:**
- TRSM internally uses alpha scalars (typically alpha=1 for GETRS)
- rocsolver_trsm_lower/upper may use host scalars internally
- Pointer mode must match where scalars are stored (host vs device memory)

**Best practice:**
Always save and restore the original pointer mode to avoid altering the user's expected behavior. This ensures GETRS is transparent to the user's pointer mode configuration.

**Comparison:**
- Some routines use init_scalars() to allocate device memory for scalars
- GETRS uses simpler approach: host pointers with mode switching
- Both approaches are valid; GETRS's is simpler since TRSM alpha=1 (implicit)

**Note:**
LASWP (row interchange) doesn't require scalars, so pointer mode only matters for TRSM calls.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the host
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

if(trans == rocblas_operation_none)
{
    // ... LASWP and TRSM calls ...
}
else
{
    // ... TRSM and LASWP calls ...
}

rocblas_set_pointer_mode(handle, old_mode);"""
    }]
})

# Entry 9: L2 - LASWP kernel parallelization
entries.append({
    "id": str(base_ts + 8),
    "level": "L2",
    "interface": "getrs",
    "query": "How does the LASWP kernel parallelize row interchange operations on the GPU?",
    "answer": """The LASWP kernel parallelizes row interchanges by assigning each thread to one column of the matrix.

**Parallelization strategy:**
- Each thread processes one column across all row swaps
- Thread ID: tid = blockIdx.x * blockDim.x + threadIdx.x
- Block dimensions: LASWP_THDS = 256 threads per block
- Grid dimensions: (ceil(n/256), batch_count)

**Kernel execution:**
```cpp
I blocksPivot = (n - 1) / LASWP_THDS + 1;
dim3 gridPivot(blocksPivot, batch_count, 1);
dim3 threads(LASWP_THDS, 1, 1);
```

**Thread work:**
Each thread tid (if tid < n):
1. Iterates sequentially through row swaps: i = k1 to k2
2. For each i, swaps elements in column tid: swap(A[i-1, tid], A[exch-1, tid])
3. No synchronization needed between threads (each works on different column)

**Why this parallelization?**
- Columns are independent (no race conditions)
- Coalesced memory access: threads access consecutive columns
- Simple load balancing: each thread does equal work (k2-k1+1 swaps)
- Scales well with matrix width (nrhs in GETRS context)

**Batch processing:**
- blockIdx.y selects batch instance
- Each batch instance processed independently in parallel

**Efficiency:**
Works well when nrhs (number of columns in B) is large enough to saturate GPU. For small nrhs, there's limited parallelism.""",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_laswp.hpp",
        "language": "cpp",
        "content": """I blocksPivot = (n - 1) / LASWP_THDS + 1;
dim3 gridPivot(blocksPivot, batch_count, 1);
dim3 threads(LASWP_THDS, 1, 1);

hipStream_t stream;
rocblas_get_stream(handle, &stream);

ROCSOLVER_LAUNCH_KERNEL(laswp_kernel<T>, gridPivot, threads, 0, stream, n, A, shiftA, inca, lda,
                        strideA, k1, k2, ipiv, shiftP, incp, strideP);"""
    }, {
        "path": "library/src/auxiliary/rocauxiliary_laswp.hpp",
        "language": "cpp",
        "content": """I id = hipBlockIdx_y;
I tid = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

if(tid < n)
{
    // batch instance
    const I* ipiv = ipivA + id * strideP + shiftP;
    T* A = load_ptr_batch(AA, id, shiftA, stride);

    // ... determine start, end, inc based on incp ...

    for(I i = start; i != end; i += inc)
    {
        I exch = ipiv[k1 + (i - k1) * incp - 1];

        // will exchange rows i and exch if they are not the same
        if(exch != i)
            swap(A[(i - 1) * inca + tid * lda], A[(exch - 1) * inca + tid * lda]);
    }
}"""
    }]
})

# Entry 10: L1 - Conjugate transpose handling
entries.append({
    "id": str(base_ts + 9),
    "level": "L1",
    "interface": "getrs",
    "query": "How does GETRS handle the conjugate transpose case (trans=rocblas_operation_conjugate_transpose)?",
    "answer": """GETRS handles conjugate transpose identically to regular transpose, passing the trans parameter directly to TRSM.

**Conjugate transpose (A^H*X = B):**
For complex matrices, A^H is the conjugate transpose (transpose + complex conjugate).
From P*A = L*U, we get:
- (P*A)^H = (L*U)^H = U^H * L^H
- A^H = U^H * L^H * P^T

**Algorithm:**
1. Solve U^H*Y = B using TRSM with trans=conjugate_transpose
2. Solve L^H*Z = Y using TRSM with trans=conjugate_transpose  
3. Apply reverse pivoting: X = P*Z

**TRSM handling:**
The TRSM routines (rocsolver_trsm_upper, rocsolver_trsm_lower) handle conjugate transpose internally by:
- Transposing matrix access pattern
- Conjugating matrix elements during computation
- This is built into rocBLAS TRSM implementation

**Code path:**
```cpp
else  // trans != none (includes both transpose and conjugate_transpose)
{
    rocsolver_trsm_upper<...>(handle, rocblas_side_left, trans, ...);
    rocsolver_trsm_lower<...>(handle, rocblas_side_left, trans, ...);
    rocsolver_laswp_template<...>(..., -1, ...);  // reverse pivoting
}
```

**Validation:**
Argument checking allows conjugate_transpose:
```cpp
if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
   && trans != rocblas_operation_conjugate_transpose)
    return rocblas_status_invalid_value;
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
   && trans != rocblas_operation_conjugate_transpose)
    return rocblas_status_invalid_value;"""
    }, {
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """else  // trans != none
{
    // solve U'*X = B or U**H *X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // solve L'*X = B, or L**H *X = B overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                              lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // then apply row interchanges to the solution vectors
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, -1, strideP, batch_count);
}"""
    }]
})

# Entry 11: L3 - Complete workflow example
entries.append({
    "id": str(base_ts + 10),
    "level": "L3",
    "interface": "getrs",
    "query": "Provide a complete example of using GETRF and GETRS together to solve a linear system A*X = B on the GPU.",
    "answer": """Here's a complete workflow for solving A*X = B using GETRF + GETRS:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <vector>

void solve_linear_system(int n, int nrhs) {
    // 1. Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // 2. Allocate host memory
    std::vector<float> h_A(n * n);  // Coefficient matrix
    std::vector<float> h_B(n * nrhs);  // Right-hand sides
    std::vector<int> h_ipiv(n);  // Pivot indices
    int h_info;
    
    // Initialize h_A and h_B with problem data...
    
    // 3. Allocate device memory
    float *d_A, *d_B;
    int *d_ipiv, *d_info;
    hipMalloc(&d_A, sizeof(float) * n * n);
    hipMalloc(&d_B, sizeof(float) * n * nrhs);
    hipMalloc(&d_ipiv, sizeof(int) * n);
    hipMalloc(&d_info, sizeof(int));
    
    // 4. Copy data to device
    hipMemcpy(d_A, h_A.data(), sizeof(float) * n * n, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(float) * n * nrhs, hipMemcpyHostToDevice);
    
    // 5. LU factorization: P*A = L*U
    rocsolver_sgetrf(handle, n, n, d_A, n, d_ipiv, d_info);
    
    // Check factorization success
    hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    if(h_info != 0) {
        printf("GETRF failed: singular matrix at position %d\\n", h_info);
        // cleanup and exit
    }
    
    // 6. Solve linear system using factorization
    // Note: d_A now contains L and U, d_B will be overwritten with solution
    rocsolver_sgetrs(handle, rocblas_operation_none, n, nrhs, d_A, n, d_ipiv, d_B, n);
    
    // 7. Copy solution back to host
    hipMemcpy(h_B.data(), d_B, sizeof(float) * n * nrhs, hipMemcpyDeviceToHost);
    // h_B now contains the solution X
    
    // 8. Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_ipiv);
    hipFree(d_info);
    rocblas_destroy_handle(handle);
}
```

**Key points:**
- GETRF modifies A in-place (stores L and U)
- GETRS modifies B in-place (overwrites with solution)
- ipiv must be preserved between GETRF and GETRS
- Check info after GETRF for singularity
- For multiple right-hand sides with same A, call GETRF once, then GETRS multiple times""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """if(trans == rocblas_operation_none)
{
    // first apply row interchanges to the right hand sides
    if(pivot)
        rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                       0, 1, strideP, batch_count);

    // solve L*X = B, overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                              lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);

    // solve U*X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                              batch_count, optim_mem, work1, work2, work3, work4);
}"""
    }]
})

# Entry 12: L2 - Coding task: implement simplified GETRS
entries.append({
    "id": str(base_ts + 11),
    "level": "L2",
    "interface": "getrs",
    "query": "Implement a simplified host-side GETRS function for a single n×n system A*X = B with one right-hand side (nrhs=1). Assume A contains L and U from LU factorization, and ipiv contains pivot indices.",
    "answer": """Here's a simplified host-side implementation of GETRS for educational purposes:

```cpp
#include <vector>
#include <algorithm>

// Solve A*X = B using LU factorization (A contains L and U, B is n×1)
void simple_getrs(int n, const std::vector<float>& A, const std::vector<int>& ipiv, 
                  std::vector<float>& B) {
    // Step 1: Apply row interchanges to B (forward pivoting)
    // ipiv uses 1-based indexing (LAPACK convention)
    for(int k = 0; k < n; k++) {
        int pivot_row = ipiv[k] - 1;  // Convert to 0-based
        if(pivot_row != k) {
            std::swap(B[k], B[pivot_row]);
        }
    }
    
    // Step 2: Solve L*Y = B (forward substitution)
    // L is unit lower triangular (diagonal = 1, stored in strict lower part of A)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            B[i] -= A[i * n + j] * B[j];
        }
        // B[i] /= 1.0; (L has unit diagonal, so division is implicit)
    }
    
    // Step 3: Solve U*X = Y (backward substitution)
    // U is upper triangular (stored in upper part including diagonal of A)
    for(int i = n - 1; i >= 0; i--) {
        for(int j = i + 1; j < n; j++) {
            B[i] -= A[i * n + j] * B[j];
        }
        B[i] /= A[i * n + i];  // Divide by diagonal element
    }
    // B now contains the solution X
}

// Example usage:
int main() {
    int n = 3;
    // A contains L and U from GETRF (column-major for LAPACK compatibility)
    std::vector<float> A = {2, 0.5, 0.5, 0, 1.5, 0.333, 0, 0, 1.667}; // Example LU factors
    std::vector<int> ipiv = {1, 2, 3};  // No pivoting in this example
    std::vector<float> B = {1, 2, 3};  // Right-hand side
    
    simple_getrs(n, A, ipiv, B);
    
    // B now contains solution
    return 0;
}
```

**Key differences from rocSOLVER GETRS:**
- Host (CPU) code vs GPU kernels
- Single right-hand side vs multiple (nrhs)
- No batching support
- No transpose/conjugate transpose support
- Simplified error handling
- Row-major storage for simplicity (rocSOLVER uses column-major)

**Educational value:**
Shows the three fundamental steps: pivoting, L solve, U solve.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrs.hpp",
        "language": "cpp",
        "content": """// first apply row interchanges to the right hand sides
if(pivot)
    rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv,
                                   0, 1, strideP, batch_count);

// solve L*X = B, overwriting B with X
rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                          rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                                          lda, strideA, B, shiftB, incb, ldb, strideB,
                                          batch_count, optim_mem, work1, work2, work3, work4);

// solve U*X = B, overwriting B with X
rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, trans,
                                          rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                          inca, lda, strideA, B, shiftB, incb, ldb, strideB,
                                          batch_count, optim_mem, work1, work2, work3, work4);"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_getrs.jsonl"
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
    if entry.get('interface') != 'getrs':
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
