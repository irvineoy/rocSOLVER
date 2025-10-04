import json
import time

# Base timestamp for generating unique IDs
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete GETRI algorithm overview
entries.append({
    "id": str(base_ts + 0),
    "level": "L3",
    "interface": "getri",
    "query": "Explain the complete algorithm used in rocSOLVER's GETRI (matrix inversion from LU factorization). How does it compute A^(-1) from the LU factorization P*A = L*U?",
    "answer": """GETRI computes the inverse of a matrix A from its LU factorization P*A = L*U using a two-stage approach:

**Stage 1: Invert upper triangular matrix U**
Call TRTRI to compute U^(-1) in-place, overwriting U.

**Stage 2: Solve inv(A) * L = inv(U)**
This is done using blocked column-wise processing from right to left:

For j = n-1 down to 0 by blocks of size blk:
1. Copy lower triangular part of L block to workspace (copy_and_zero)
2. Update current columns: A[:, j:j+jb] -= A[:, j+jb:n] * L[j+jb:n, j:j+jb] (GEMM)
3. Solve triangular system: A[:, j:j+jb] = A[:, j:j+jb] * inv(L[j:j+jb, j:j+jb]) (TRSM)

**Stage 3: Apply pivoting**
Apply column interchanges using ipiv in reverse order to get A^(-1).

The algorithm is optimized with:
- Block size selection based on matrix dimension
- Small-size kernel for tiny/small matrices (≤GETRI_TINY_SIZE or ≤TRTRI_MAX_COLS)
- Singularity handling via workspace zeroing
- Batched and strided variants for multiple matrices""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """// compute inverse of U (also check singularity and update info)
rocsolver_trtri_template<BATCHED, STRIDED, T>(
    handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n, A, shiftA, lda, strideA, info,
    batch_count, work1, work2, work3, work4, tmpcopy, workArr, optim_mem);

// ************************************************ //
// Next, compute inv(A) solving inv(A) * L = inv(U) //

rocblas_int nn = ((n - 1) / blk) * blk + 1;
for(rocblas_int j = nn - 1; j >= 0; j -= blk)
{
    jb = std::min(n - j, blk);

    // copy and zero entries in case info is nonzero
    ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large1<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                            0, stream, n, j, jb, A, shiftA, lda, strideA, info, tmpcopy, strideW);

    if(j + jb < n)
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, jb,
                       n - j - jb, &minone, A, shiftA + idx2D(0, j + jb, lda), lda, strideA,
                       tmpcopy, j + jb, ldw, strideW, &one, A, shiftA + idx2D(0, j, lda), lda,
                       strideA, batch_count, workArr);

    rocblasCall_trsm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                     rocblas_diagonal_unit, n, jb, &one, tmpcopy, j, ldw, strideW, A,
                     shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem, work1,
                     work2, work3, work4, workArr);
}

// apply pivoting (column interchanges)
if(pivot)
    ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large2<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                            0, stream, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info);"""
    }]
})

# Entry 2: L2 - TRTRI stage for inverting upper triangular U
entries.append({
    "id": str(base_ts + 1),
    "level": "L2",
    "interface": "getri",
    "query": "What is the purpose of calling TRTRI in GETRI, and how does it handle singularity detection?",
    "answer": """TRTRI (Triangular Matrix Inversion) is called as the first stage of GETRI to compute the inverse of the upper triangular matrix U from the LU factorization P*A = L*U.

**Purpose:**
- Computes U^(-1) in-place, overwriting U in the input matrix A
- Detects singularity by checking diagonal elements
- Sets info[b] = k+1 if U(k,k) = 0 (singular matrix)

**Implementation details:**
- Called with rocblas_fill_upper and rocblas_diagonal_non_unit
- Uses blocked algorithm for large matrices (block size from trtri_get_blksize)
- For small matrices (n ≤ TRTRI_MAX_COLS), uses optimized trti2_run_small kernel
- Singularity information is propagated to subsequent GETRI stages

**Handling singular matrices:**
GETRI checks info after TRTRI. If info[b] != 0 (singular), the getri_kernel_large1 kernel calls zero_work() instead of copy_and_zero(), which ensures GEMM and TRSM operations don't alter the matrix (effectively skipping the inversion for that batch instance).

The invdiag kernel within TRTRI computes diagonal inverses: a[i,i] = 1.0 / a[i,i], and detects when a[i,i] == 0.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """// compute inverse of U (also check singularity and update info)
rocsolver_trtri_template<BATCHED, STRIDED, T>(
    handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n, A, shiftA, lda, strideA, info,
    batch_count, work1, work2, work3, work4, tmpcopy, workArr, optim_mem);"""
    }, {
        "path": "library/src/lapack/roclapack_trtri.hpp",
        "language": "cpp",
        "content": """template <typename T, typename U>
ROCSOLVER_KERNEL void invdiag(const rocblas_diagonal diag,
                              const rocblas_int n,
                              U A,
                              const rocblas_int shiftA,
                              const rocblas_int lda,
                              const rocblas_stride strideA,
                              T* alphas)
{
    int b = hipBlockIdx_y;
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* d = alphas + b * n;

        if(a[i + i * lda] != 0 && diag == rocblas_diagonal_non_unit)
        {
            a[i + i * lda] = 1.0 / a[i + i * lda];
            d[i] = -a[i + i * lda];
        }
        else
            d[i] = -1.0;
    }
}"""
    }]
})

# Entry 3: L1 - Block size selection
entries.append({
    "id": str(base_ts + 2),
    "level": "L1",
    "interface": "getri",
    "query": "How does GETRI select the block size for the blocked inversion algorithm?",
    "answer": """GETRI uses getri_get_blksize() to select block size based on matrix dimension and batch mode:

**Batched/Strided mode:**
- Uses GETRI_BATCH_BLKSIZES and GETRI_BATCH_INTERVALS
- Returns size[get_index(intervals, GETRI_BATCH_NUM_INTERVALS, dim)]

**Non-batched mode:**
- Uses GETRI_BLKSIZES and GETRI_INTERVALS
- Returns size[get_index(intervals, GETRI_NUM_INTERVALS, dim)]

The get_index() function finds the appropriate interval for the matrix dimension. If blk == 0 is returned, the algorithm uses blk = n (unblocked).

Block size affects performance by balancing:
- Level 3 BLAS (GEMM) utilization in trailing updates
- TRSM triangular solve efficiency
- Memory footprint of workspace (n × blk × sizeof(T) × batch_count)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """template <bool ISBATCHED>
rocblas_int getri_get_blksize(const rocblas_int dim)
{
    rocblas_int blk;

    if(ISBATCHED)
    {
        rocblas_int size[] = {GETRI_BATCH_BLKSIZES};
        rocblas_int intervals[] = {GETRI_BATCH_INTERVALS};
        rocblas_int max = GETRI_BATCH_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }
    else
    {
        rocblas_int size[] = {GETRI_BLKSIZES};
        rocblas_int intervals[] = {GETRI_INTERVALS};
        rocblas_int max = GETRI_NUM_INTERVALS;
        blk = size[get_index(intervals, max, dim)];
    }

    return blk;
}"""
    }]
})

# Entry 4: L2 - copy_and_zero kernel
entries.append({
    "id": str(base_ts + 3),
    "level": "L2",
    "interface": "getri",
    "query": "What is the purpose of the copy_and_zero device function in GETRI's getri_kernel_large1?",
    "answer": """The copy_and_zero function serves dual purposes in the blocked GETRI algorithm:

**1. Copy lower triangular L to workspace:**
Extracts the current block of the lower triangular factor L from matrix A to the workspace. This is necessary because:
- The original L factor is needed for the TRSM solve
- A is being overwritten with inv(A) during the algorithm
- The workspace preserves L for the triangular solve

**2. Zero out lower triangular part in A:**
After copying, zeros the lower triangular part in A. This prepares A for the inversion result without interference from old L values.

**Implementation:**
- Parallelized over all m×n entries using threadIdx_y
- Only processes entries where i > j (strict lower triangular)
- Uses __syncthreads() to ensure all threads complete before proceeding

The kernel getri_kernel_large1 calls either copy_and_zero (if info[b] == 0) or zero_work (if info[b] != 0, indicating singularity).""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """template <typename T>
__device__ void copy_and_zero(const rocblas_int m,
                              const rocblas_int n,
                              T* a,
                              const rocblas_int lda,
                              T* w,
                              const rocblas_int ldw)
{
    // Copies the lower triangular part of the matrix to the workspace and then
    // replaces it with zeroes
    for(int k = hipThreadIdx_y; k < m * n; k += hipBlockDim_y)
    {
        int i = k % m;
        int j = k / m;
        if(i > j)
        {
            w[i + j * ldw] = a[i + j * lda];
            a[i + j * lda] = 0;
        }
    }
    __syncthreads();
}"""
    }, {
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """template <typename T, typename U, typename V>
ROCSOLVER_KERNEL void getri_kernel_large1(const rocblas_int n,
                                          const rocblas_int j,
                                          const rocblas_int jb,
                                          U A,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          rocblas_int* info,
                                          V work,
                                          const rocblas_stride strideW)
{
    // Helper kernel for large-size matrices. Preps the matrix for calls to
    // gemm and trsm.
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* w = load_ptr_batch<T>(work, b, 0, strideW);

    if(info[b] != 0)
        zero_work(n - j, jb, w + j, n);
    else
        copy_and_zero(n - j, jb, a + j + j * lda, lda, w + j, n);
}"""
    }]
})

# Entry 5: L1 - Pivoting in reverse order
entries.append({
    "id": str(base_ts + 4),
    "level": "L1",
    "interface": "getri",
    "query": "Why does GETRI apply pivots in reverse order (from j=n-2 down to 0)?",
    "answer": """GETRI applies pivots in reverse order because the LU factorization produces P*A = L*U with forward pivots, and we need to undo the permutation P to get A^(-1).

**Mathematical reasoning:**
- GETRF computes: P*A = L*U, where P is represented by ipiv
- GETRI computes: (P*A)^(-1) = U^(-1) * L^(-1)
- To get A^(-1), we need: A^(-1) = (P*A)^(-1) * P = U^(-1) * L^(-1) * P

**Implementation:**
The getri_pivot device function iterates j from n-2 down to 0, swapping columns:
- jp = ipiv[j] - 1 (LAPACK uses 1-based indexing)
- If jp != j, swap columns j and jp

**Why reverse order:**
Pivots were applied forward during GETRF (j=0 to n-1), so reversing undoes them correctly. The loop stops at j=n-2 because the last pivot (j=n-1) doesn't require a swap.

This is only applied when pivot=true (standard GETRI). The GETRI_NPVT variant skips pivoting since it assumes A = L*U without permutation.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """template <typename T>
__device__ void getri_pivot(const rocblas_int n, T* a, const rocblas_int lda, rocblas_int* p)
{
    // Applies the pivots specified in ipiv to the inverted matrix
    rocblas_int jp;
    T temp;
    for(rocblas_int j = n - 2; j >= 0; --j)
    {
        jp = p[j] - 1;
        if(jp != j)
        {
            for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
                swap(a[i + j * lda], a[i + jp * lda]);
            __syncthreads();
        }
    }
}"""
    }]
})

# Entry 6: L3 - Workspace calculation
entries.append({
    "id": str(base_ts + 5),
    "level": "L3",
    "interface": "getri",
    "query": "Explain the workspace allocation strategy in rocsolver_getri_getMemorySize. What are the different workspace components and when are they needed?",
    "answer": """GETRI allocates multiple workspace components based on matrix size and optimization level:

**Workspace components:**
1. **work1, work2, work3, work4**: Reusable workspace for TRSM and TRTRI calls
2. **tmpcopy**: Temporary array for copying lower triangular L blocks (size: n × blk × sizeof(T) × batch_count)
3. **workArr**: Array of pointers for batched cases (size: sizeof(T*) × batch_count)

**Size-based optimization:**
- **Tiny size** (n ≤ GETRI_TINY_SIZE or GETRI_BATCH_TINY_SIZE): Zero workspace, use getri_run_small kernel
- **Small size** (n ≤ TRTRI_MAX_COLS): Only TRTRI workspace needed (w1b, w2b, w3b, w4b, t2)
- **Large size**: Full workspace including tmpcopy for blocked algorithm

**Workspace calculation:**
1. Calls rocsolver_trtri_getMemorySize for TRTRI requirements (w1b, w2b, w3b, w4b, t2)
2. Computes TRSM requirements with rocblasCall_trsm_mem (w1a, w2a, w3a, w4a)
3. Takes maximum: size_work1 = max(w1a, w1b), etc.
4. Computes tmpcopy size: n × blk × sizeof(T) × batch_count (or max with t2 from TRTRI)

**optim_mem flag:**
Set to true when optimal memory allocation is possible, ensuring TRSM has all required memory for peak performance.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """// get block size
rocblas_int blk = getri_get_blksize<ISBATCHED>(n);
if(blk == 0)
    blk = n;

// size of temporary array required for copies
t1 = n * blk * sizeof(T) * batch_count;

// requirements for calling TRSM
rocblas_int nn = (n % 128 != 0) ? n : n + 1;
rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, nn, blk + 1, 1, 1,
                                 batch_count, &w1a, &w2a, &w3a, &w4a);

*size_work1 = std::max(w1a, w1b);
*size_work2 = std::max(w2a, w2b);
*size_work3 = std::max(w3a, w3b);
*size_work4 = std::max(w4a, w4b);
*size_tmpcopy = std::max(t1, t2);

// always allocate all required memory for TRSM optimal performance
opt2 = true;

*optim_mem = opt1 && opt2;"""
    }]
})

# Entry 7: L2 - Small-size optimization
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "getri",
    "query": "How does GETRI optimize for small matrices, and what are the size thresholds?",
    "answer": """GETRI uses specialized kernels for small matrices to avoid overhead of blocked algorithms and BLAS calls.

**Size thresholds:**
- **Tiny matrices**: n ≤ GETRI_TINY_SIZE (non-batched) or n ≤ GETRI_BATCH_TINY_SIZE (batched/strided)
- **Small matrices**: n ≤ TRTRI_MAX_COLS

**Optimization strategy:**

**Tiny matrices (stage 1 & 2 combined):**
- Calls getri_run_small with two_stage=true (single kernel handles both TRTRI and solving inv(A)*L=inv(U))
- No workspace allocation needed
- Entire inversion in one optimized kernel

**Small matrices (split stages):**
- Stage 1: Uses trti2_run_small for inverting U (called within TRTRI)
- Stage 2: Uses getri_run_small with two_stage=false
- Minimal workspace (only for TRTRI)

**Large matrices:**
- Uses full blocked algorithm with TRTRI, GEMM, TRSM

The small-size kernels leverage shared memory and thread cooperation for matrices that fit within GPU shared memory limits, avoiding memory transfers and BLAS call overhead.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """#ifdef OPTIMAL
    if((n <= GETRI_TINY_SIZE && !ISBATCHED) || (n <= GETRI_BATCH_TINY_SIZE && ISBATCHED))
    {
        return getri_run_small<T>(handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                                  batch_count, true, pivot);
    }
#endif

    // compute inverse of U (also check singularity and update info)
    rocsolver_trtri_template<BATCHED, STRIDED, T>(
        handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n, A, shiftA, lda, strideA, info,
        batch_count, work1, work2, work3, work4, tmpcopy, workArr, optim_mem);

    // ************************************************ //
    // Next, compute inv(A) solving inv(A) * L = inv(U) //

#ifdef OPTIMAL
    // if small size, use optimized kernel for stage 2
    if(n <= TRTRI_MAX_COLS)
    {
        return getri_run_small<T>(handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info,
                                  batch_count, false, pivot);
    }
#endif"""
    }]
})

# Entry 8: L1 - GETRI vs GETRI_NPVT
entries.append({
    "id": str(base_ts + 7),
    "level": "L1",
    "interface": "getri",
    "query": "What is the difference between GETRI and GETRI_NPVT, and when would you use each?",
    "answer": """**GETRI (standard):**
- Computes A^(-1) from LU factorization with partial pivoting: P*A = L*U
- Requires ipiv array from GETRF
- Applies column interchanges at the end (getri_pivot)
- Use when: Matrix was factorized with GETRF (standard case)

**GETRI_NPVT (no pivot):**
- Computes A^(-1) from LU factorization without pivoting: A = L*U
- ipiv parameter is nullptr
- Skips column interchange step (pivot=false)
- Use when: Matrix was factorized with GETRF_NPVT (e.g., diagonally dominant matrices)

**Implementation difference:**
Both call the same rocsolver_getri_impl template, but with different pivot parameter:
- GETRI: pivot=true, ipiv must be provided
- GETRI_NPVT: pivot=false, ipiv=nullptr

**Performance:**
GETRI_NPVT is slightly faster (skips final pivoting kernel), but only valid for matrices that don't require pivoting for numerical stability.

**C API functions:**
rocsolver_sgetri, rocsolver_dgetri, rocsolver_cgetri, rocsolver_zgetri (standard)
rocsolver_sgetri_npvt, rocsolver_dgetri_npvt, rocsolver_cgetri_npvt, rocsolver_zgetri_npvt (no pivot)""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgetri(rocblas_handle handle,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getri_impl<float>(handle, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_sgetri_npvt(rocblas_handle handle,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_impl<float>(handle, n, A, lda, ipiv, info, false);
}"""
    }]
})

# Entry 9: L3 - GEMM update in blocked inversion
entries.append({
    "id": str(base_ts + 8),
    "level": "L3",
    "interface": "getri",
    "query": "Explain the GEMM update step in GETRI's blocked inversion loop. What matrix operation does it perform and why?",
    "answer": """The GEMM update in GETRI performs a critical trailing matrix update during the blocked solving of inv(A) * L = inv(U).

**Mathematical operation:**
For block j with size jb:
A[:, j:j+jb] -= A[:, j+jb:n] × L[j+jb:n, j:j+jb]

This can be written as:
A(1:n, j:j+jb) = A(1:n, j:j+jb) - A(1:n, j+jb:n) * L(j+jb:n, j:j+jb)

**Why this is needed:**
We're solving inv(A) * L = inv(U) column by column from right to left. For the current block at columns j:j+jb:
1. The columns to the right (j+jb:n) have already been processed and contain partial results
2. We need to account for their contribution to the current block via the L factor
3. The GEMM computes this contribution as a rank-jb update

**Implementation parameters:**
- Operation: none, none (no transposes)
- Dimensions: m=n, n=jb, k=n-j-jb
- Alpha: -1 (subtraction)
- Beta: 1 (accumulate)
- Matrix A pointer: shifted to column j+jb
- Workspace pointer: L block copied to tmpcopy

This is followed by a TRSM solve to finish processing the current block.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """rocblas_int nn = ((n - 1) / blk) * blk + 1;
for(rocblas_int j = nn - 1; j >= 0; j -= blk)
{
    jb = std::min(n - j, blk);

    // copy and zero entries in case info is nonzero
    ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large1<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                            0, stream, n, j, jb, A, shiftA, lda, strideA, info, tmpcopy, strideW);

    if(j + jb < n)
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, jb,
                       n - j - jb, &minone, A, shiftA + idx2D(0, j + jb, lda), lda, strideA,
                       tmpcopy, j + jb, ldw, strideW, &one, A, shiftA + idx2D(0, j, lda), lda,
                       strideA, batch_count, workArr);

    rocblasCall_trsm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                     rocblas_diagonal_unit, n, jb, &one, tmpcopy, j, ldw, strideW, A,
                     shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem, work1,
                     work2, work3, work4, workArr);
}"""
    }]
})

# Entry 10: L2 - TRSM solve in GETRI
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "getri",
    "query": "What triangular solve does the TRSM call in GETRI's main loop perform?",
    "answer": """The TRSM (Triangular Solve with Multiple right-hand sides) call solves:

**Mathematical operation:**
A[:, j:j+jb] = A[:, j:j+jb] × inv(L[j:j+jb, j:j+jb])

Or equivalently: X * L(j:j+jb, j:j+jb) = B, where X is the solution we're computing.

**TRSM parameters:**
- Side: rocblas_side_right (L is on the right: X*L = B)
- Fill: rocblas_fill_lower (L is lower triangular)
- Operation: rocblas_operation_none (no transpose)
- Diagonal: rocblas_diagonal_unit (L has unit diagonal from LU factorization)
- Dimensions: m=n, n=jb
- Alpha: 1
- Matrix L: from tmpcopy workspace (copied by getri_kernel_large1)
- Matrix B/X: A[:, j:j+jb] (in-place solve)

**Context:**
This is part of solving inv(A) * L = inv(U) for the current block. After the GEMM update accounts for trailing columns, TRSM solves the triangular system for the current block columns.

The unit diagonal specification means the diagonal elements of L (which are 1) are not accessed, only the strict lower triangular part.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """rocblasCall_trsm(handle, rocblas_side_right, rocblas_fill_lower, rocblas_operation_none,
                 rocblas_diagonal_unit, n, jb, &one, tmpcopy, j, ldw, strideW, A,
                 shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem, work1,
                 work2, work3, work4, workArr);"""
    }]
})

# Entry 11: L1 - Pointer mode requirement
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "getri",
    "query": "Why does GETRI temporarily switch to host pointer mode during execution?",
    "answer": """GETRI switches to rocblas_pointer_mode_host during the main blocked loop because the scalar values (minone=-1, one=1) are defined on the host stack.

**Code:**
```cpp
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

T minone = -1;
T one = 1;
// ... GEMM and TRSM calls using &minone and &one ...

rocblas_set_pointer_mode(handle, old_mode);
```

**Reasoning:**
- GEMM and TRSM require pointers to alpha/beta scalars
- Host pointer mode: scalars are read from host memory addresses
- Device pointer mode: scalars are read from device memory addresses
- Defining T minone = -1 creates a host variable, so host pointer mode is required

**Best practice:**
Always save the original pointer mode and restore it before returning, ensuring GETRI doesn't alter the user's expected pointer mode behavior.

This is different from other routines that use init_scalars() to allocate device memory for scalars.""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the host
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

T minone = -1;
T one = 1;
rocblas_int jb;

rocblas_int nn = ((n - 1) / blk) * blk + 1;
for(rocblas_int j = nn - 1; j >= 0; j -= blk)
{
    jb = std::min(n - j, blk);
    // ... GEMM and TRSM calls ...
}

// apply pivoting (column interchanges)
if(pivot)
    ROCSOLVER_LAUNCH_KERNEL(getri_kernel_large2<T>, dim3(batch_count, 1, 1), dim3(1, threads, 1),
                            0, stream, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info);

rocblas_set_pointer_mode(handle, old_mode);"""
    }]
})

# Entry 12: L3 - Coding task: implement workspace size calculation
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "getri",
    "query": "Implement a function that computes the total device memory required for GETRI given matrix dimension n and batch_count. Assume non-batched mode and include all workspace components.",
    "answer": """Here's an implementation that computes total GETRI workspace:

```cpp
#include <algorithm>

template <typename T>
size_t compute_getri_workspace(rocblas_int n, rocblas_int batch_count)
{
    // Quick return for empty cases
    if(n == 0 || batch_count == 0)
        return 0;
    
    // Tiny size optimization
    #ifdef OPTIMAL
    if(n <= GETRI_TINY_SIZE)
        return 0;  // No workspace needed
    #endif
    
    size_t total_workspace = 0;
    
    // Get block size for non-batched mode
    rocblas_int size[] = {GETRI_BLKSIZES};
    rocblas_int intervals[] = {GETRI_INTERVALS};
    rocblas_int max = GETRI_NUM_INTERVALS;
    rocblas_int blk = size[get_index(intervals, max, n)];
    
    if(blk == 0)
        blk = n;
    
    // Workspace components
    size_t size_work1 = 0, size_work2 = 0, size_work3 = 0, size_work4 = 0;
    size_t size_tmpcopy = 0;
    size_t size_workArr = 0;  // Zero for non-batched
    
    // TRTRI workspace (w1b, w2b, w3b, w4b, t2)
    // Simplified: assume standard case
    size_t w1b = n * sizeof(T) * batch_count;
    size_t w2b = 0;
    size_t w3b = n * sizeof(T) * batch_count;
    size_t w4b = 0;
    size_t t2 = n * n * sizeof(T) * batch_count;
    
    // TRSM workspace (w1a, w2a, w3a, w4a)
    rocblas_int nn = (n % 128 != 0) ? n : n + 1;
    // Simplified estimate
    size_t w1a = nn * (blk + 1) * sizeof(T) * batch_count;
    size_t w2a = 0;
    size_t w3a = nn * sizeof(T) * batch_count;
    size_t w4a = 0;
    
    // Temporary copy workspace
    size_t t1 = n * blk * sizeof(T) * batch_count;
    
    // Take maximums
    size_work1 = std::max(w1a, w1b);
    size_work2 = std::max(w2a, w2b);
    size_work3 = std::max(w3a, w3b);
    size_work4 = std::max(w4a, w4b);
    size_tmpcopy = std::max(t1, t2);
    
    total_workspace = size_work1 + size_work2 + size_work3 + size_work4 + 
                      size_tmpcopy + size_workArr;
    
    return total_workspace;
}
```

**Key aspects:**
1. Returns 0 for quick-return and tiny-size cases
2. Computes block size using tuning tables
3. Calculates TRTRI and TRSM workspace requirements
4. Takes maximum of overlapping workspaces (work1-4)
5. Sums all independent workspace components

**Usage:**
```cpp
size_t required_bytes = compute_getri_workspace<float>(1024, 1);
// Allocate: hipMalloc(&workspace, required_bytes);
```""",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getri.hpp",
        "language": "cpp",
        "content": """// get block size
rocblas_int blk = getri_get_blksize<ISBATCHED>(n);
if(blk == 0)
    blk = n;

// size of temporary array required for copies
t1 = n * blk * sizeof(T) * batch_count;

// requirements for calling TRSM
rocblas_int nn = (n % 128 != 0) ? n : n + 1;
rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, nn, blk + 1, 1, 1,
                                 batch_count, &w1a, &w2a, &w3a, &w4a);

*size_work1 = std::max(w1a, w1b);
*size_work2 = std::max(w2a, w2b);
*size_work3 = std::max(w3a, w3b);
*size_work4 = std::max(w4a, w4b);
*size_tmpcopy = std::max(t1, t2);"""
    }]
})

# Write to jsonl file
output_file = "dataset/roclapack_getri.jsonl"
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
    if entry.get('interface') != 'getri':
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
