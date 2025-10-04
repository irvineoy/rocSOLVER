#!/usr/bin/env python3
"""
Generate JSONL dataset for roclapack_geblttrf_npvt
"""

import json
from datetime import datetime

def create_entry(id_num, level, interface, instruction, context_text, code_blocks,
                answer, rationale, tags):
    """Create a dataset entry."""
    return {
        "id": f"2025-10-03T05:00:00.{id_num:03d}",
        "level": level,
        "interface": interface,
        "instruction": instruction,
        "context_text": context_text,
        "code_blocks": code_blocks,
        "answer": answer,
        "rationale": rationale,
        "tags": tags
    }

# Dataset entries
entries = []

# L1-1: geblttrf_update_info kernel
entries.append(create_entry(
    1, "L1", "geblttrf_npvt",
    "Implement the `geblttrf_update_info` kernel that updates the global info array with error codes from block LU factorizations. Each thread handles one batch instance. If the global info is zero but the local iinfo is non-zero, update info with iinfo + k_shift to indicate which block failed.",
    "The geblttrf_update_info kernel propagates factorization failure information from individual block factorizations to the global info array. Used after each block GETRF in the block tridiagonal LU factorization. Source: library/src/lapack/roclapack_geblttrf_npvt.hpp (lines 38-49)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """template <typename T>
ROCSOLVER_KERNEL void
    geblttrf_update_info(T* info, T* iinfo, const rocblas_int k_shift, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] == 0 && iinfo[b] != 0)
            info[b] = iinfo[b] + k_shift;
    }
}"""
        }
    ],
    """Implementation of geblttrf_update_info kernel:

```cpp
template <typename T>
ROCSOLVER_KERNEL void
    geblttrf_update_info(T* info, T* iinfo, const rocblas_int k_shift, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] == 0 && iinfo[b] != 0)
            info[b] = iinfo[b] + k_shift;
    }
}
```

**Key aspects:**
1. **Thread mapping**: Each thread handles one batch instance (b = blockIdx.x * blockDim.x + threadIdx.x)
2. **Conditional update**: Only update if info[b]==0 (no prior error) and iinfo[b]!=0 (current block failed)
3. **Offset calculation**: Add k_shift to iinfo[b] to indicate which block in the sequence failed
4. **Boundary check**: Guard with `if(b < bc)` to handle partial thread blocks

**Launch configuration**: dim3(blocksReset, 1, 1) with blocksReset = (batch_count-1)/BS1 + 1, threads = BS1

This ensures the first error encountered is recorded with its absolute position in the block tridiagonal matrix.""",
    "This kernel implementation is correct because: (1) The thread-to-batch mapping ensures each batch instance is processed exactly once. (2) The conditional `if(info[b] == 0 && iinfo[b] != 0)` implements the 'first error wins' semantic - once info[b] is set to non-zero, it won't be overwritten by later blocks. (3) The k_shift offset (k*nb where k is block index) converts local block indices to global matrix indices, which is required for LAPACK-compatible error reporting. (4) The boundary check prevents out-of-bounds access when batch_count is not a multiple of block size. (5) The kernel is simple and coalesced - each thread writes to a different location in info array.",
    ["coding", "error-propagation", "batch-processing", "boundary-checks"]
))

# L1-2: Memory layout and strides
entries.append(create_entry(
    2, "L1", "geblttrf_npvt",
    "Explain the memory layout and stride calculations for the block tridiagonal matrix in geblttrf_npvt. The matrix is stored as three arrays A, B, C representing off-diagonal, diagonal, and off-diagonal blocks respectively. Describe how block strides (bsa, bsb, bsc) are computed and used to access different blocks.",
    "GEBLTTRF_NPVT factorizes a block tridiagonal matrix with diagonal blocks in B, lower off-diagonal in A, and upper off-diagonal in C. Block strides determine how to access the k-th block. Source: library/src/lapack/roclapack_geblttrf_npvt.hpp (lines 209-212, 222-234)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """    // block strides
    rocblas_int bsa = lda * nb;
    rocblas_int bsb = ldb * nb;
    rocblas_int bsc = ldc * nb;

    rocsolver_getrf_template<BATCHED, STRIDED, T>(handle, nb, nb, B, shiftB, incb, ldb, strideB,
                                                  (rocblas_int*)nullptr, 0, 0, info, batch_count,
                                                  scalars, work1, work2, work3, work4, pivotval,
                                                  pivotidx, iipiv, iinfo1, optim_mem, false);

    for(rocblas_int k = 0; k < nblocks - 1; k++)
    {
        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nb, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, C, shiftC + k * bsc, incc, ldc, strideC, batch_count, work1,
            work2, work3, work4, optim_mem, false);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nb, nb, &minone,
                       A, shiftA + k * bsa, inca, lda, strideA, C, shiftC + k * bsc, incc, ldc,
                       strideC, &one, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, batch_count,
                       (T**)nullptr);"""
        }
    ],
    """Memory layout and stride calculations:

**Block Tridiagonal Structure:**
```
B[0]  C[0]   0     0    ...
A[0]  B[1]  C[1]   0    ...
 0    A[1]  B[2]  C[2]  ...
 0     0    A[2]  B[3]  ...
...
```

**Storage:**
- **A array**: Lower off-diagonal blocks A[0], A[1], ..., A[nblocks-2] stored contiguously
- **B array**: Diagonal blocks B[0], B[1], ..., B[nblocks-1] stored contiguously
- **C array**: Upper off-diagonal blocks C[0], C[1], ..., C[nblocks-2] stored contiguously

**Block Stride Calculations:**
```cpp
rocblas_int bsa = lda * nb;  // stride to next A block
rocblas_int bsb = ldb * nb;  // stride to next B block
rocblas_int bsc = ldc * nb;  // stride to next C block
```

**Accessing k-th block:**
- A[k]: `A + shiftA + k * bsa` (with inca, lda for internal layout)
- B[k]: `B + shiftB + k * bsb` (with incb, ldb for internal layout)
- C[k]: `C + shiftC + k * bsc` (with incc, ldc for internal layout)

**Why lda * nb?**
Each block is nb×nb stored in column-major with leading dimension lda. To skip one block (nb columns), advance by lda*nb elements.

**Usage in algorithm (block k):**
1. Solve B[k]*X = C[k] for X (GETRS at shiftB + k*bsb)
2. Update B[k+1] := B[k+1] - A[k]*X (GEMM using shiftA + k*bsa, shiftC + k*bsc, shiftB + (k+1)*bsb)
3. Factor updated B[k+1] (GETRF at shiftB + (k+1)*bsb)""",
    "This layout is correct because: (1) Block strides bsa=lda*nb account for column-major storage where advancing nb columns requires lda*nb elements. (2) The offset shiftB + k*bsb correctly addresses the k-th diagonal block in the B array. (3) Separate arrays A, B, C allow independent leading dimensions (lda, ldb, ldc) for flexibility. (4) The inc parameters (inca, incb, incc) support both contiguous and strided block storage within each array. (5) The strideA, strideB, strideC parameters enable batched execution where each batch instance has its own block tridiagonal matrix. (6) This layout is cache-friendly since each block is contiguous in memory (column-major within block, blocks stored sequentially).",
    ["analysis", "memory-layout", "block-tridiagonal", "stride-calculation"]
))

# L1-3: Workspace sizing strategy
entries.append(create_entry(
    3, "L1", "geblttrf_npvt",
    "Implement the workspace sizing logic for geblttrf_npvt that computes memory requirements for GETRF and GETRS operations and takes the maximum for reuse. Explain why workspace can be shared between these operations.",
    "GEBLTTRF_NPVT calls both GETRF (factorization) and GETRS (triangular solve) in its algorithm loop. The workspace sizing function computes requirements for both and allocates max(getrf_workspace, getrs_workspace) for reuse. Source: library/src/lapack/roclapack_geblttrf_npvt.hpp (lines 93-110)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """    bool unused;
    size_t a1 = 0, a2 = 0;
    size_t b1 = 0, b2 = 0;
    size_t c1 = 0, c2 = 0;
    size_t d1 = 0, d2 = 0;

    // size requirements for getrf
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T>(
        nb, nb, false, batch_count, size_scalars, &a1, &b1, &c1, &d1, size_pivotval, size_pivotidx,
        size_iipiv, size_iinfo1, optim_mem, ldb, incb);

    // size requirements for getrs
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(rocblas_operation_none, nb, nb, batch_count,
                                                       &a2, &b2, &c2, &d2, &unused, ldb, ldc, incb,
                                                       incc);

    *size_work1 = std::max(a1, a2);
    *size_work2 = std::max(b1, b2);
    *size_work3 = std::max(c1, c2);
    *size_work4 = std::max(d1, d2);

    // size for temporary info storage
    *size_iinfo2 = sizeof(rocblas_int) * batch_count;"""
        }
    ],
    """Workspace sizing implementation:

```cpp
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrf_npvt_getMemorySize(
    const rocblas_int nb, const rocblas_int nblocks,
    const rocblas_int batch_count,
    size_t* size_scalars, size_t* size_work1, size_t* size_work2,
    size_t* size_work3, size_t* size_work4,
    size_t* size_pivotval, size_t* size_pivotidx,
    size_t* size_iipiv, size_t* size_iinfo1, size_t* size_iinfo2,
    bool* optim_mem,
    const rocblas_int ldb = 1, const rocblas_int ldc = 1,
    const rocblas_int incb = 1, const rocblas_int incc = 1)
{
    // Quick return check
    if(nb == 0 || nblocks == 0 || batch_count == 0) {
        *size_scalars = 0;
        *size_work1 = *size_work2 = *size_work3 = *size_work4 = 0;
        *size_pivotval = *size_pivotidx = 0;
        *size_iipiv = *size_iinfo1 = *size_iinfo2 = 0;
        return;
    }

    size_t a1 = 0, a2 = 0, b1 = 0, b2 = 0;
    size_t c1 = 0, c2 = 0, d1 = 0, d2 = 0;
    bool unused;

    // Query GETRF workspace (nb×nb blocks)
    rocsolver_getrf_getMemorySize<BATCHED, STRIDED, T>(
        nb, nb, false, batch_count,
        size_scalars, &a1, &b1, &c1, &d1,
        size_pivotval, size_pivotidx, size_iipiv, size_iinfo1,
        optim_mem, ldb, incb);

    // Query GETRS workspace (nb×nb blocks)
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(
        rocblas_operation_none, nb, nb, batch_count,
        &a2, &b2, &c2, &d2, &unused,
        ldb, ldc, incb, incc);

    // Take maximum for each workspace category
    *size_work1 = std::max(a1, a2);
    *size_work2 = std::max(b1, b2);
    *size_work3 = std::max(c1, c2);
    *size_work4 = std::max(d1, d2);

    // Additional temporary info array
    *size_iinfo2 = sizeof(rocblas_int) * batch_count;
}
```

**Why workspace can be shared:**
1. GETRF and GETRS are **never called simultaneously** - the algorithm is sequential: GETRF(B[0]) → loop{ GETRS, GEMM, GETRF }
2. Both operations use workspace for **temporary arrays** (GEMV/TRSM intermediates, pointer arrays for batched mode)
3. Workspace is **ephemeral** - contents don't need to persist across function calls
4. Taking max(getrf_size, getrs_size) ensures sufficient space for whichever operation needs more

**Memory savings:**
- If GETRF needs 10KB and GETRS needs 8KB: allocate 10KB (shared) instead of 18KB (separate)
- Typical savings: ~30-40% reduction in total workspace allocation""",
    "This workspace sharing strategy is correct because: (1) The algorithm execution order is strictly sequential - GETRF completes before GETRS starts, and vice versa in the loop. (2) The workspace buffers (work1-4) are used for temporary storage that doesn't carry state between operations. (3) The getMemorySize functions query worst-case requirements for their respective operations at the given matrix size (nb×nb blocks). (4) Using std::max ensures the allocated buffer is large enough for whichever operation has the larger requirement. (5) The separate iinfo2 array is needed because getrf and getrs use different info arrays, and we need to preserve info while storing getrf results in iinfo2 temporarily. (6) This pattern follows rocSOLVER's design philosophy of minimizing memory footprint by identifying reuse opportunities.",
    ["coding", "workspace-management", "memory-optimization", "resource-sharing"]
))

# L2-1: GETRF and GETRS cooperation
entries.append(create_entry(
    4, "L2", "geblttrf_npvt",
    "Explain how GETRF and GETRS kernels cooperate in the block tridiagonal factorization algorithm. Describe the data flow in one iteration: how the factored B[k] is used to solve for C[k], then how the result updates B[k+1] before the next factorization.",
    "GEBLTTRF_NPVT uses a sequential algorithm where each iteration involves: (1) Factor diagonal block, (2) Solve triangular system with off-diagonal, (3) Update next diagonal block, (4) Factor updated block. Sources: library/src/lapack/roclapack_geblttrf_npvt.hpp (template function) and referenced GETRF/GETRS implementations.",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """    rocsolver_getrf_template<BATCHED, STRIDED, T>(handle, nb, nb, B, shiftB, incb, ldb, strideB,
                                                  (rocblas_int*)nullptr, 0, 0, info, batch_count,
                                                  scalars, work1, work2, work3, work4, pivotval,
                                                  pivotidx, iipiv, iinfo1, optim_mem, false);

    for(rocblas_int k = 0; k < nblocks - 1; k++)
    {
        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nb, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, C, shiftC + k * bsc, incc, ldc, strideC, batch_count, work1,
            work2, work3, work4, optim_mem, false);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nb, nb, &minone,
                       A, shiftA + k * bsa, inca, lda, strideA, C, shiftC + k * bsc, incc, ldc,
                       strideC, &one, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, batch_count,
                       (T**)nullptr);

        rocsolver_getrf_template<BATCHED, STRIDED, T>(
            handle, nb, nb, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, (rocblas_int*)nullptr, 0,
            0, iinfo2, batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, iipiv,
            iinfo1, optim_mem, false);

        ROCSOLVER_LAUNCH_KERNEL(geblttrf_update_info, gridReset, threads, 0, stream, info, iinfo2,
                                (k + 1) * nb, batch_count);
    }"""
        }
    ],
    """Data flow and kernel cooperation in block tridiagonal factorization:

**Initial Step (k=-1, before loop):**
```
GETRF(B[0]) → B[0] = L[0]*U[0]
Store L[0] in lower triangle of B[0], U[0] in upper triangle
info captures any singularity in B[0]
```

**Iteration k (k=0 to nblocks-2):**

**Step 1: Solve with factored diagonal block**
```
GETRS: B[k] * X = C[k]
Input:  B[k] already factored (L[k], U[k] from prior GETRF)
        C[k] is the upper off-diagonal block
Output: C[k] overwritten with solution X = L[k]^-1 * U[k]^-1 * C[k]
```
- Uses LU factors stored in B[k] (no pivoting version, so L and U are in-place)
- Solves: L[k]*Y = C[k] (forward substitution), then U[k]*X = Y (backward substitution)
- Result stored back in C[k]

**Step 2: Update next diagonal block**
```
GEMM: B[k+1] := B[k+1] - A[k] * C[k]
Input:  A[k] is the lower off-diagonal block (unchanged)
        C[k] now contains X from Step 1
        B[k+1] is the next diagonal block (original value)
Output: B[k+1] updated with Schur complement
```
- Implements: B[k+1] := B[k+1] - A[k] * (B[k]^-1 * C[k])
- This is the Schur complement update in block Gaussian elimination

**Step 3: Factor updated block**
```
GETRF(B[k+1]) → B[k+1] = L[k+1]*U[k+1]
Input:  B[k+1] from Step 2 (updated with Schur complement)
Output: B[k+1] overwritten with L[k+1], U[k+1] factors
        iinfo2 captures any singularity in this block
```

**Step 4: Propagate error info**
```
geblttrf_update_info: info[b] = iinfo2[b] + (k+1)*nb if needed
Updates global error indicator with block offset
```

**Synchronization:**
- All operations on same stream → implicit ordering
- GETRS must complete before GEMM reads C[k]
- GEMM must complete before GETRF reads B[k+1]
- GETRF must complete before update_info reads iinfo2

**Final state:**
- B array contains all L[k]*U[k] factors for k=0..nblocks-1
- A array unchanged (lower off-diagonal blocks)
- C array overwritten with intermediate results (not needed after factorization)
- info[b] contains first error index (if any) for each batch instance""",
    "This algorithm and cooperation is correct because: (1) The mathematical foundation is block Gaussian elimination: eliminate the lower off-diagonal A[k] by using the factored B[k] to compute Schur complement B[k+1] - A[k]*B[k]^-1*C[k]. (2) The GETRS call computes B[k]^-1*C[k] by solving the system using the LU factors in B[k]. (3) The GEMM applies the update with A[k] to get the Schur complement in B[k+1]. (4) The subsequent GETRF factors this updated B[k+1] for use in the next iteration. (5) Stream ordering ensures correct data dependencies - each operation waits for its inputs from prior kernels. (6) The no-pivoting (npvt) variant is valid here because the algorithm structure ensures diagonal dominance is preserved under certain conditions. (7) Error propagation via geblttrf_update_info correctly tracks which block failed, with offset (k+1)*nb converting local block index to global matrix index.",
    ["analysis", "kernel-cooperation", "data-flow", "block-gaussian-elimination", "schur-complement"]
))

# L2-2: Optimization - kernel fusion
entries.append(create_entry(
    5, "L2", "geblttrf_npvt",
    "Design a fused kernel that combines GETRS and the GEMM update into a single operation. The fused kernel should solve B[k]*X = C[k] and immediately apply B[k+1] := B[k+1] - A[k]*X without writing X to global memory. Explain the benefits and implementation challenges.",
    "Currently, GETRS writes the solution to C[k] in global memory, then GEMM reads it back. Fusing these operations could eliminate this round-trip. Source: library/src/lapack/roclapack_geblttrf_npvt.hpp (lines 221-229)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nb, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, C, shiftC + k * bsc, incc, ldc, strideC, batch_count, work1,
            work2, work3, work4, optim_mem, false);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nb, nb, &minone,
                       A, shiftA + k * bsa, inca, lda, strideA, C, shiftC + k * bsc, incc, ldc,
                       strideC, &one, B, shiftB + (k + 1) * bsb, incb, ldb, strideB, batch_count,
                       (T**)nullptr);"""
        }
    ],
    """Fused GETRS-GEMM kernel design:

```cpp
template <int BLK, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BLK*BLK)
geblttrf_fused_solve_update(
    const I nb,
    U BB_k,      // B[k] - already factored (L,U in-place)
    const rocblas_stride shiftB_k,
    const I ldb,
    U CC_k,      // C[k] - RHS to solve
    const rocblas_stride shiftC_k,
    const I ldc,
    U AA_k,      // A[k] - lower off-diagonal
    const rocblas_stride shiftA_k,
    const I lda,
    U BB_kp1,    // B[k+1] - next diagonal block to update
    const rocblas_stride shiftB_kp1,
    const I ldb_kp1)
{
    int bid = blockIdx.z;  // batch index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load batch pointers
    T* B_k = load_ptr_batch<T>(BB_k, bid, shiftB_k, ...);
    T* C_k = load_ptr_batch<T>(CC_k, bid, shiftC_k, ...);
    T* A_k = load_ptr_batch<T>(AA_k, bid, shiftA_k, ...);
    T* B_kp1 = load_ptr_batch<T>(BB_kp1, bid, shiftB_kp1, ...);

    extern __shared__ double smem[];
    T* X_tile = reinterpret_cast<T*>(smem);  // nb×nb tile for solution
    T* A_tile = X_tile + nb*nb;              // nb×nb tile for A[k]

    // Phase 1: Triangular solve B[k] * X = C[k]
    // Forward substitution: L*Y = C (L is unit lower in B_k)
    for(int j = 0; j < nb; j++) {
        // Load column j of C into X_tile
        if(ty == 0 && tx < nb)
            X_tile[tx + j*nb] = C_k[tx + j*ldc];
        __syncthreads();

        // Forward solve: X[i,j] = (C[i,j] - sum(L[i,k]*X[k,j])) for i>j
        if(tx > j && ty == 0) {
            T sum = 0;
            for(int k = j; k < tx; k++)
                sum += B_k[tx + k*ldb] * X_tile[k + j*nb];
            X_tile[tx + j*nb] -= sum;
        }
        __syncthreads();
    }

    // Backward substitution: U*X = Y (U is upper in B_k)
    for(int j = nb-1; j >= 0; j--) {
        if(tx <= j && ty == 0) {
            T sum = 0;
            for(int k = tx+1; k <= j; k++)
                sum += B_k[tx + k*ldb] * X_tile[k + j*nb];
            X_tile[tx + j*nb] = (X_tile[tx + j*nb] - sum) / B_k[tx + tx*ldb];
        }
        __syncthreads();
    }

    // Phase 2: GEMM update B[k+1] := B[k+1] - A[k] * X
    // Load A[k] tile into LDS
    if(tx < nb && ty < nb)
        A_tile[tx + ty*nb] = A_k[tx + ty*lda];
    __syncthreads();

    // Compute update using X_tile (already in LDS) and A_tile
    if(tx < nb && ty < nb) {
        T sum = 0;
        for(int k = 0; k < nb; k++)
            sum += A_tile[tx + k*nb] * X_tile[k + ty*nb];

        // Atomic update to B[k+1] (or use atomicAdd if T is float/double)
        B_kp1[tx + ty*ldb_kp1] -= sum;
    }
}

// Launch: dim3(1, 1, batch_count), dim3(BLK, BLK), LDS = 2*nb*nb*sizeof(T)
```

**Benefits:**
1. **Eliminates global memory traffic**: Solution X stays in LDS, saves 2× nb² words (read+write of C)
2. **Reduces kernel launch overhead**: 2 kernel launches → 1 (~2-4μs saved per iteration)
3. **Better cache locality**: A[k] loaded once, used immediately with X in LDS
4. **Lower latency**: No global memory round-trip between solve and update

**Estimated speedup**: 15-25% for nb=32-64, diminishes for large nb where GEMM dominates

**Implementation Challenges:**
1. **LDS size limit**: Needs 2×nb² elements (~32KB for nb=64, double precision). May not fit for large nb.
2. **Thread block dimensions**: BLK×BLK threads may exceed limits (1024 threads/block). For nb=64, need 64×64=4096 threads (exceeds limit).
3. **Synchronization complexity**: Multiple __syncthreads() in triangular solve loop - careful ordering required
4. **Numerical stability**: In-place triangular solve in LDS may have different rounding behavior than GETRS
5. **Generality**: Works only for nb×nb blocks. Loses flexibility of separate GETRS/GEMM for rectangular blocks.
6. **Code complexity**: Harder to maintain/debug than separate well-tested GETRS + GEMM

**When beneficial:**
- Small to medium nb (≤ 64) where LDS fits and launch overhead is significant
- Memory-bandwidth-bound scenarios
- Avoid if: large nb, or nb not known at compile time (limits template optimization)

**Recommendation**: Implement as optional tuned path for common nb values (32, 64), fall back to separate kernels otherwise.""",
    "This fusion design is sound because: (1) The triangular solve can be computed in LDS by processing columns sequentially with proper synchronization to resolve dependencies. (2) Keeping the solution X in LDS avoids the write to C[k] and subsequent read by GEMM. (3) The GEMM update can directly use X_tile and A_tile from LDS for the computation. (4) The identified challenges are real: (a) LDS size grows as 2×nb² which limits the maximum nb (e.g., 64KB LDS allows nb≤90 for double precision). (b) Thread block size BLK×BLK must handle nb×nb data, requiring careful decomposition for large nb. (c) The multiple synchronization barriers in the solve phase add overhead but are necessary for correctness. (d) The loss of generality means this fused kernel is less reusable than the modular GETRS/GEMM approach. The performance benefit analysis is realistic - launch overhead savings and memory traffic reduction are most impactful for small-to-medium nb where the kernel is not yet compute-bound.",
    ["coding", "kernel-fusion", "optimization", "lds-optimization", "trade-offs"]
))

# L3-1: Interface-wide flow
entries.append(create_entry(
    6, "L3", "geblttrf_npvt",
    "Describe the complete computational flow of the geblttrf_npvt interface from API entry to final factorization output. Include parameter validation, workspace allocation strategy, the main factorization algorithm, error handling, and the final state of the A, B, C arrays and info parameter.",
    "GEBLTTRF_NPVT computes the LU factorization of a block tridiagonal matrix without pivoting. The matrix is stored in three separate arrays representing the off-diagonal and diagonal blocks. Sources: library/src/lapack/roclapack_geblttrf_npvt.hpp (argCheck, getMemorySize, template functions)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """template <typename T>
rocblas_status rocsolver_geblttrf_npvt_argCheck(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int lda,
                                                const rocblas_int ldb,
                                                const rocblas_int ldc,
                                                T A,
                                                T B,
                                                T C,
                                                rocblas_int* info,
                                                const rocblas_int batch_count = 1,
                                                const rocblas_int inca = 1,
                                                const rocblas_int incb = 1,
                                                const rocblas_int incc = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(inca < 1 || lda < inca * nb)
        return rocblas_status_invalid_size;
    if(incb < 1 || ldb < incb * nb)
        return rocblas_status_invalid_size;
    if(incc < 1 || ldc < incc * nb)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}"""
        }
    ],
    """Complete GEBLTTRF_NPVT computational flow:

**1. API Entry & Validation** (rocsolver_[s/d/c/z]geblttrf_npvt in .cpp):
- User calls: `rocsolver_dgeblttrf_npvt(handle, nb, nblocks, A, lda, B, ldb, C, ldc, info)`
- Parameters:
  - `nb`: Size of each square block (all blocks are nb×nb)
  - `nblocks`: Number of block rows/columns in tridiagonal matrix
  - `A`: Lower off-diagonal blocks (nblocks-1 blocks)
  - `B`: Diagonal blocks (nblocks blocks)
  - `C`: Upper off-diagonal blocks (nblocks-1 blocks)
  - `info`: Error indicator array (batch_count elements)

**Validation** (`rocsolver_geblttrf_npvt_argCheck`):
- Check handle != nullptr
- Validate sizes: nb ≥ 0, nblocks ≥ 0, batch_count ≥ 0
- Validate leading dimensions: lda ≥ inca*nb, ldb ≥ incb*nb, ldc ≥ incc*nb
- Validate pointers: A, C required if nblocks > 1; B required if nblocks > 0; info required if batch_count > 0
- Return `invalid_handle`, `invalid_size`, or `invalid_pointer` on error

**2. Workspace Allocation** (`rocsolver_geblttrf_npvt_getMemorySize`):
- Query workspace for GETRF: nb×nb factorization (for diagonal blocks)
- Query workspace for GETRS: nb×nb triangular solve (for off-diagonal update)
- Allocate: max(getrf_workspace, getrs_workspace) for each category (work1-4)
- Additional allocations:
  - `pivotval`, `pivotidx`: Pivot information (not used in npvt, but allocated for compatibility)
  - `iipiv`: Internal pivot array
  - `iinfo1`: Info array for GETRF
  - `iinfo2`: Temporary info array for per-block error codes (batch_count elements)
- Memory manager handles allocation via rocBLAS

**3. Quick Return**:
- If nb=0 or nblocks=0 or batch_count=0: return success immediately

**4. Main Factorization Algorithm** (`rocsolver_geblttrf_npvt_template`):

**Initialization:**
```cpp
rocblas_int bsa = lda * nb;  // block stride for A
rocblas_int bsb = ldb * nb;  // block stride for B
rocblas_int bsc = ldc * nb;  // block stride for C
T one = 1.0, minone = -1.0;
```

**Step 0: Factor first diagonal block**
```
GETRF(B[0]) → B[0] = L[0]*U[0]
Store error in info
```

**Loop: k = 0 to nblocks-2**

a) **Solve with factored diagonal:**
```
GETRS: B[k] * X = C[k]
Input:  B[k] = L[k]*U[k] (from prior GETRF)
        C[k] = upper off-diagonal block
Output: C[k] ← X = (L[k]*U[k])^-1 * C[k]
```

b) **Schur complement update:**
```
GEMM: B[k+1] := B[k+1] - A[k] * C[k]
      B[k+1] := B[k+1] - A[k] * X
Implements Gaussian elimination to annihilate A[k]
```

c) **Factor updated diagonal:**
```
GETRF: B[k+1] = L[k+1]*U[k+1]
Store result in B[k+1]
Error code in iinfo2[batch]
```

d) **Propagate error information:**
```
geblttrf_update_info:
  if info[b]==0 and iinfo2[b]!=0:
    info[b] = iinfo2[b] + (k+1)*nb
Tracks first failure with global index
```

**5. Output State**:

**A array (unchanged):**
- Contains lower off-diagonal blocks A[0], A[1], ..., A[nblocks-2]
- Each block is nb×nb, stored column-major with leading dimension lda
- Not modified by algorithm

**B array (overwritten with LU factors):**
- B[0] = L[0]*U[0]: L[0] in strictly lower triangle (unit diagonal), U[0] in upper triangle
- B[1] = L[1]*U[1]: Factors of updated B[1] after Schur complement
- ...
- B[nblocks-1] = L[nblocks-1]*U[nblocks-1]: Factors of final block
- Each block nb×nb, column-major with leading dimension ldb

**C array (overwritten with intermediate results):**
- C[k] contains (L[k]*U[k])^-1 * C[k]_original after GETRS
- Not needed for reconstruction of factorization
- Can be discarded or reused

**info array:**
- info[b] = 0: Factorization successful for batch instance b
- info[b] = i > 0: First singularity detected at global position i (1-based index)
  - i = iinfo[b] + k*nb where k is block index, iinfo[b] is local block index
- LAPACK-compatible error reporting

**6. Cleanup & Return**:
- Workspace automatically deallocated by memory manager
- Return `rocblas_status_success`

**Mathematical Result**:
The factorization satisfies:
```
[B[0]  C[0]   0     ...  ] = [L[0]    0      0   ...] [U[0]  C̃[0]   0    ...]
[A[0]  B[1]  C[1]   ...  ]   [Ã[0]  L[1]    0   ...] [ 0    U[1]  C̃[1] ...]
[ 0    A[1]  B[2]   ...  ]   [ 0    Ã[1]  L[2]  ...] [ 0     0    U[2]  ...]
[...                     ]   [...                  ] [...                  ]
```

where:
- L[k], U[k] are stored in B[k]
- Ã[k] = A[k] * U[k]^-1 (implicitly through the algorithm)
- C̃[k] = L[k]^-1 * C[k] (computed during GETRS, stored in C[k])

This factorization enables efficient solve of block tridiagonal systems via forward/backward substitution.""",
    "This complete flow is accurate because: (1) The validation order (handle→size→pointers) matches rocSOLVER/LAPACK conventions for unit test compatibility. (2) The workspace sizing uses max(getrf, getrs) which is valid since operations are sequential and buffers are ephemeral. (3) The main algorithm implements block Gaussian elimination: factor B[k], use factors to eliminate A[k] via Schur complement B[k+1]-A[k]*B[k]^-1*C[k], then factor updated B[k+1]. (4) Error propagation adds offset (k+1)*nb to convert local block indices to global matrix indices for LAPACK compatibility. (5) The output state correctly describes: A unchanged (lower off-diagonal), B contains LU factors of all diagonal blocks, C overwritten with intermediate results. (6) The mathematical formulation shows this is equivalent to factoring the block tridiagonal as LU where L is block lower bidiagonal with unit diagonal blocks, and U is block upper bidiagonal. (7) The no-pivoting (npvt) restriction means this factorization exists if the matrix is diagonally dominant or positive definite in appropriate sense.",
    ["analysis", "end-to-end-flow", "api-design", "block-tridiagonal-factorization", "error-handling"]
))

# L3-2: Design choices and alternatives
entries.append(create_entry(
    7, "L3", "geblttrf_npvt",
    "Analyze the design choice of using separate A, B, C arrays versus a single packed array for the block tridiagonal matrix. Discuss the trade-offs in terms of memory layout, API usability, kernel performance, and potential for algorithmic variants (with/without pivoting, different block structures).",
    "GEBLTTRF_NPVT uses three separate arrays (A, B, C) to store the block tridiagonal matrix instead of a single interleaved or packed format. This design choice impacts multiple aspects of the implementation. Sources: library/src/lapack/roclapack_geblttrf_npvt.hpp (interface definition and algorithm)",
    [
        {
            "path": "library/src/lapack/roclapack_geblttrf_npvt.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrf_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                U A,
                                                const rocblas_int shiftA,
                                                const rocblas_int inca,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                U B,
                                                const rocblas_int shiftB,
                                                const rocblas_int incb,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                U C,
                                                const rocblas_int shiftC,
                                                const rocblas_int incc,
                                                const rocblas_int ldc,
                                                const rocblas_stride strideC,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                ...)"""
        }
    ],
    """Design analysis: Separate arrays (A, B, C) vs. single packed array

**Current Design: Separate Arrays**
```
A[nblocks-1 blocks]:  A[0], A[1], ..., A[nblocks-2]  (lower off-diagonal)
B[nblocks blocks]:    B[0], B[1], ..., B[nblocks-1]  (diagonal)
C[nblocks-1 blocks]:  C[0], C[1], ..., C[nblocks-2]  (upper off-diagonal)
```

**Alternative: Single Packed Array**
```
Interleaved: [B[0], C[0], A[0], B[1], C[1], A[1], B[2], ...]
or
Sequential: [B[0], B[1], ..., B[n-1], A[0], ..., A[n-2], C[0], ..., C[n-2]]
```

**Comparison Table:**

| Aspect | Separate Arrays (Current) | Single Packed Array |
|--------|---------------------------|---------------------|
| **API Usability** | ✅ Natural for users - each band is separate pointer | ❌ Requires understanding of interleaving pattern |
| **Memory Layout Flexibility** | ✅ Each array can have different leading dimension (lda, ldb, ldc) | ❌ Single leading dimension - wastes space if blocks have padding |
| **Stride Flexibility** | ✅ Independent inca, incb, incc for block-internal layout | ❌ Fixed internal layout |
| **Batch Support** | ✅ Independent strideA, strideB, strideC for batched mode | ❌ Fixed stride pattern between batches |
| **Kernel Performance** | ✅ Clean access patterns - GETRF(B), GETRS(B,C), GEMM(A,C,B) | ⚠️ More pointer arithmetic, potential cache conflicts |
| **Memory Footprint** | ⚠️ Three allocations, potential fragmentation | ✅ Single allocation, better locality for small blocks |
| **Algorithm Variants** | ✅ Easy to extend - add D array for pentadiagonal, etc. | ❌ Requires redesign for different structures |
| **With-Pivoting Version** | ✅ No change - pivoting affects B only, A/C unchanged | ✅ Also works, but pivot indices need offset |
| **Interoperability** | ✅ Matches LAPACK GT* routines (separate D, E arrays) | ❌ Non-standard format |
| **In-Place Factorization** | ⚠️ C overwritten, but A and B preserve structure | ⚠️ Entire array overwritten, harder to identify bands |

**Detailed Analysis:**

**1. API Usability:**
- **Separate**: Users naturally think of block tridiagonal as three bands. Easy to construct from existing data.
- **Packed**: Users must learn packing convention. Error-prone to get indices right.

**2. Memory Layout Flexibility:**
- **Separate**: Essential for real-world data that may have different alignments
  - Example: B blocks might be 256-byte aligned for GEMM, while A/C are not
  - Different leading dimensions allow optimal padding for each array
- **Packed**: Forces uniform layout - if B needs ldb=256 but A/C could fit in lda=128, packed format wastes 50% space

**3. Kernel Performance:**
- **Separate**:
  - GETRF operates on contiguous B array → optimal cache usage
  - GEMM with A and C as separate inputs → standard rocBLAS interface
  - No extra index calculations
- **Packed**:
  - Pointer offsets: `B[k] = base + interleave_offset(k)`
  - Possible cache line conflicts if interleaving breaks alignment
  - Potential benefit: If entire block triple (A[k], B[k], C[k]) fits in cache together for small blocks

**4. Algorithm Variants:**
- **Separate**: Extensible design
  - Pentadiagonal: Add D, E arrays for second off-diagonals
  - Block banded: Add more arrays as needed
  - Asymmetric: Allow A and C to have different nb (rectangular blocks on off-diagonal)
- **Packed**: Rigid structure
  - Any variant requires new packing convention
  - Not forward-compatible

**5. With-Pivoting Version (GEBLTTRF with pivoting):**
- **Separate**:
  - Pivoting permutes rows of B[k]
  - Same permutation applied to C[k] (upper) and A[k+1] (lower)
  - Clean to track: pivot[k] affects B[k], C[k], and A[k+1]
- **Packed**:
  - Pivoting affects interleaved entries
  - More complex index calculations
  - Could lead to unaligned accesses after pivoting

**6. Batched Execution:**
- **Separate**:
  - strideA, strideB, strideC allow different instance spacing
  - Useful if batches come from different sources (e.g., B might be in one memory pool, A/C in another)
- **Packed**:
  - Single stride limits flexibility
  - Entire batch must use same packing pattern

**Recommendation: Separate Arrays (Current Design) is Superior**

**When Packed Might Be Better:**
- **Small blocks (nb ≤ 32)** and **small nblocks (≤ 8)**: Entire structure fits in L1 cache, locality benefits outweigh flexibility loss
- **Fixed problem size**: If nb, nblocks, leading dimensions are compile-time constants, packed format can be optimized aggressively
- **Embedded systems**: Single allocation reduces memory management overhead

**Optimization for Current Design:**
- For small problems, consider **fusing A, B, C into single allocation** while keeping separate pointers:
  ```cpp
  T* base = allocate(size_A + size_B + size_C);
  A = base;
  B = base + size_A;
  C = base + size_A + size_B;
  ```
  This combines single-allocation benefit with separate-array flexibility.

**Conclusion:**
The current separate-array design is optimal for general-purpose library because:
1. It maximizes user flexibility and interoperability
2. It supports diverse memory layouts and batch configurations
3. It enables clean algorithm variants (pivoting, different block structures)
4. Performance cost is minimal - pointer indirection is negligible compared to compute
5. It matches LAPACK conventions (separate D, E, etc. arrays)

The trade-off (potential cache locality loss for tiny blocks) is acceptable because tiny blocks are better handled by dense methods anyway, not block tridiagonal algorithms.""",
    "This design analysis is grounded in practical considerations: (1) The separate array interface with independent lda/ldb/ldc parameters is evident in the template signature and enables real-world use cases with non-uniform padding. (2) The kernel calls (GETRF(B), GETRS(B,C), GEMM(A,C,B)) show clean separation that would be complicated with packed format. (3) The extensibility argument is valid - adding pentadiagonal support would simply add D, E arrays rather than redesigning storage. (4) The batched parameters (strideA, strideB, strideC) demonstrate flexibility that packed format would sacrifice. (5) The performance analysis correctly identifies that pointer indirection cost is negligible compared to O(n³) operations in GETRF/GEMM. (6) The recommendation to use single allocation with separate pointers for small problems is a best-of-both-worlds compromise used in other rocSOLVER routines. (7) The comparison to LAPACK GT* routines validates this as standard practice in numerical linear algebra libraries.",
    ["analysis", "design-trade-offs", "api-design", "memory-layout", "extensibility"]
))

# Generate JSONL file
output_file = "roclapack_geblttrf_npvt.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Generated {len(entries)} entries in {output_file}")
print(f"\nDistribution:")
print(f"  L1: {sum(1 for e in entries if e['level']=='L1')} entries")
print(f"  L2: {sum(1 for e in entries if e['level']=='L2')} entries")
print(f"  L3: {sum(1 for e in entries if e['level']=='L3')} entries")
print(f"\nTask types:")
coding = sum(1 for e in entries if 'coding' in e['tags'])
analysis = sum(1 for e in entries if 'analysis' in e['tags'])
print(f"  Coding:   {coding} ({100*coding/len(entries):.1f}%)")
print(f"  Analysis: {analysis} ({100*analysis/len(entries):.1f}%)")
