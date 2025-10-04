#!/usr/bin/env python3
"""Generate POTRF dataset entries following DATASET.md schema."""

import json
import time

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Create a dataset entry with timestamp-based ID."""
    return {
        "id": str(int(time.time() * 1000)),
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

# Entry 1: POTRF blocked algorithm - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="potrf",
    instruction="Explain the blocked Cholesky factorization algorithm in POTRF. What are the three main operations in each iteration and why does it use BLAS-3 operations (TRSM, HERK)?",
    context_text="POTRF implements blocked Cholesky factorization for efficiency. It processes the matrix in blocks, using POTF2 for diagonal blocks and BLAS-3 operations (TRSM, HERK) for off-diagonal updates to achieve high performance.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """while(j < n - POTRF_POTF2_SWITCHSIZE(T))
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(n - j, nb);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, iinfo, batch_count, 0);
    rocsolver_potf2_template<T>(handle, uplo, jb, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, iinfo, batch_count, scalars, (T*)work1, pivots);

    // test for non-positive-definiteness
    ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                            iinfo, info, j, batch_count);

    if(j + jb < n)
    {
        // update trailing submatrix
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
            rocblas_diagonal_non_unit, (n - j - jb), jb, A, shiftA + idx2D(j, j, lda), lda,
            strideA, A, shiftA + idx2D(j + jb, j, lda), lda, strideA, batch_count,
            optim_mem, work1, work2, work3, work4);

        rocblasCall_syrk_herk<BATCHED, T>(
            handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
            shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
            shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);
    }
    j += nb;
}"""
    }],
    answer="""Blocked Cholesky factorization (POTRF) algorithm:

**Overview:**

For SPD matrix A (n×n), compute A = L*L^T where L is lower triangular.

**Blocked algorithm (lower triangle case):**

```
Block size: nb (typically 64-256)
Loop: j = 0, nb, 2*nb, ..., n - nb

For each block:
```

**Operation 1: Factor diagonal block (POTF2)**

```cpp
jb = min(n - j, nb);  // Current block size

// Factor A[j:j+jb, j:j+jb] using unblocked Cholesky
rocsolver_potf2_template<T>(
    handle, uplo, jb,
    A, shiftA + idx2D(j, j, lda), lda, strideA,
    iinfo, batch_count, scalars, (T*)work1, pivots);

// Result: A[j:j+jb, j:j+jb] = L_block * L_block^T
```

Complexity: O(jb³/3) ≈ O(nb³/3) per block

**Operation 2: Solve triangular system (TRSM)**

```cpp
// Update: A[j+jb:n, j:j+jb] = A[j+jb:n, j:j+jb] * L_block^{-T}
rocsolver_trsm_lower<BATCHED, STRIDED, T>(
    handle,
    rocblas_side_right,               // L on the right
    rocblas_operation_conjugate_transpose,  // L^{-T}
    rocblas_diagonal_non_unit,
    (n - j - jb),                     // Rows to update
    jb,                               // Columns (block size)
    A, shiftA + idx2D(j, j, lda), lda, strideA,  // L_block
    A, shiftA + idx2D(j + jb, j, lda), lda, strideA,  // Block to update
    batch_count, optim_mem, work1, work2, work3, work4);

// Solves: X * L^T = B  where B = A[j+jb:n, j:j+jb]
// Result: A[j+jb:n, j:j+jb] contains L[j+jb:n, j:j+jb]
```

Complexity: O((n-j-jb) * jb²) ≈ O(n * nb²) per block

**Operation 3: Symmetric rank-k update (HERK/SYRK)**

```cpp
// Update trailing submatrix:
// A[j+jb:n, j+jb:n] -= L[j+jb:n, j:j+jb] * L[j+jb:n, j:j+jb]^T
rocblasCall_syrk_herk<BATCHED, T>(
    handle, uplo,
    rocblas_operation_none,           // No transpose on L
    n - j - jb,                       // Dimension of trailing block
    jb,                               // Rank of update
    &s_minone,                        // Alpha = -1
    A, shiftA + idx2D(j + jb, j, lda), lda, strideA,  // L[j+jb:n, j:j+jb]
    &s_one,                           // Beta = 1
    A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,  // Trailing block
    batch_count);

// Computes: A[j+jb:n, j+jb:n] = A[j+jb:n, j+jb:n] - L_new * L_new^T
```

Complexity: O((n-j-jb)² * jb) ≈ O(n² * nb) per block

**Why BLAS-3 operations?**

1. **High arithmetic intensity:**
   - TRSM: O(n²k) flops with O(n²+nk) memory → O(k) reuse
   - HERK: O(n²k) flops with O(n²+nk) memory → O(k) reuse
   - BLAS-3 achieves near-peak performance on GPUs

2. **Comparison to unblocked:**
   - Unblocked (POTF2): Uses BLAS-2 (GEMV, DOT)
   - BLAS-2: O(n²) flops with O(n²) memory → O(1) reuse (slow!)
   - BLAS-3: O(n²k) flops with O(n²+nk) memory → O(k) reuse (fast!)

3. **Performance gain:**
   - Block size nb=128: 128× more arithmetic intensity
   - Actual speedup: 10-20× on modern GPUs

**Total complexity:**

Number of blocks: ~n/nb
Per block:
- POTF2: O(nb³/3)
- TRSM: O(n*nb²)
- HERK: O(n²*nb)

Total: O(n/nb * (nb³ + n*nb² + n²*nb)) = O(n³/3)

Same asymptotic cost as unblocked, but much faster in practice!

**Algorithm trace (n=256, nb=64):**

```
Block 0 (j=0):
  POTF2(A[0:64, 0:64])       → 64³/3 = 85K flops
  TRSM(A[64:256, 0:64])      → 192*64² = 786K flops
  HERK(A[64:256, 64:256])    → 192²*64 = 2.4M flops

Block 1 (j=64):
  POTF2(A[64:128, 64:128])   → 64³/3 = 85K flops
  TRSM(A[128:256, 64:128])   → 128*64² = 524K flops
  HERK(A[128:256, 128:256])  → 128²*64 = 1.0M flops

Block 2 (j=128):
  POTF2(A[128:192, 128:192]) → 64³/3 = 85K flops
  TRSM(A[192:256, 128:192])  → 64*64² = 262K flops
  HERK(A[192:256, 192:256])  → 64²*64 = 262K flops

Block 3 (j=192):
  POTF2(A[192:256, 192:256]) → 64³/3 = 85K flops

Total: ~5.6M flops ≈ 256³/3
```

**Upper triangle case:**

Similar but with different TRSM/HERK orientation:
```cpp
// TRSM: L^T * X = B (left side, conjugate transpose)
rocsolver_trsm_upper(..., rocblas_side_left,
    rocblas_operation_conjugate_transpose, ...);

// HERK: A = A - U^T * U (conjugate transpose operation)
rocblasCall_syrk_herk(..., rocblas_operation_conjugate_transpose, ...);
```""",
    rationale="Blocked POTRF achieves high performance by using BLAS-3 operations (TRSM, HERK) which have O(nb) arithmetic intensity instead of BLAS-2 operations with O(1) intensity. Each iteration factors a diagonal block with POTF2, solves a triangular system with TRSM, and updates the trailing matrix with HERK.",
    tags=["algorithm", "potrf", "cholesky", "blocked", "blas3"]
))

# Entry 2: sqrtDiagOnward kernel - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="potrf",
    instruction="Implement the sqrtDiagOnward kernel for real matrices. How does it compute the diagonal element and detect non-positive definiteness?",
    context_text="The sqrtDiagOnward kernel computes diagonal elements of the Cholesky factor by taking square roots. It must detect when the matrix is not positive definite (when the value under the square root is non-positive) and set info accordingly.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potf2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename I, typename INFO, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    T t = M[loc] - res[id];

    if(t <= 0.0)
    {
        // error for non-positive definiteness
        if(info[id] == 0)
            info[id] = j + 1; // use fortran 1-based index
        M[loc] = t;
        res[id] = 0;
    }

    else
    {
        // minor is positive definite
        M[loc] = sqrt(t);
        res[id] = 1 / M[loc];
    }
}"""
    }],
    answer="""sqrtDiagOnward kernel implementation for real matrices:

```cpp
template <typename T, typename I, typename INFO, typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(
    U A,                              // Matrix (batched)
    const rocblas_stride shiftA,      // Offset in A
    const rocblas_stride strideA,     // Stride between batches
    const size_t loc,                 // Location of diagonal element
    const I j,                        // Column index
    T* res,                           // Workspace (per batch)
    INFO* info)                       // Error info (per batch)
{
    I id = hipBlockIdx_x;  // Batch index

    // Get pointer to current batch
    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);

    // Compute value under square root:
    // t = A[j,j] - sum(L[j,0:j-1]²)
    // res[id] contains sum from previous DOT product
    T t = M[loc] - res[id];

    if(t <= 0.0)  // Not positive definite!
    {
        // Set info to failing column (1-based index)
        if(info[id] == 0)  // Only set first failure
            info[id] = j + 1;

        // Store invalid value (for debugging)
        M[loc] = t;

        // Set res to 0 (prevent division by zero later)
        res[id] = 0;
    }
    else  // Positive definite
    {
        // Compute diagonal element: L[j,j] = sqrt(t)
        M[loc] = sqrt(t);

        // Compute reciprocal for later SCAL operations
        res[id] = 1 / M[loc];
    }
}
```

**Algorithm context:**

In unblocked Cholesky (POTF2), for each column j:

**Step 1: Compute dot product**
```cpp
// Compute: sum = L[j,0:j-1] · L[j,0:j-1]
rocblasCall_dot(handle, j, A, ..., A, ..., batch_count, res, work);
// Result: res[id] = sum of L[j,0:j-1]²
```

**Step 2: Compute diagonal (sqrtDiagOnward)**
```cpp
// Compute: L[j,j] = sqrt(A[j,j] - sum)
sqrtDiagOnward(..., loc=idx2D(j,j,lda), j, res, info);

// Mathematical formula:
// L[j,j] = sqrt(A[j,j] - Σ_{k=0}^{j-1} L[j,k]²)
```

**Step 3: Scale column below diagonal**
```cpp
// Compute: L[j+1:n, j] = (A[j+1:n, j] - ...) / L[j,j]
// Uses res[id] = 1/L[j,j] for efficiency
```

**Why t <= 0 indicates failure:**

For SPD matrix, all leading principal minors > 0:
```
det(A[0:j, 0:j]) > 0  for all j
```

The Cholesky formula:
```
L[j,j]² = A[j,j] - Σ_{k=0}^{j-1} L[j,k]²
```

If the j×j leading submatrix is positive definite:
```
L[j,j]² > 0  →  t > 0
```

If not positive definite:
```
L[j,j]² ≤ 0  →  t ≤ 0  →  Cannot compute real square root!
```

**Example:**

```
A = [4  2  1]
    [2  3  2]
    [1  2  1]  ← Not positive definite (det = 0)

j=0: t = 4 - 0 = 4 > 0
     L[0,0] = sqrt(4) = 2 ✓

j=1: t = 3 - (2/2)² = 3 - 1 = 2 > 0
     L[1,1] = sqrt(2) = 1.414 ✓

j=2: t = 1 - (1/2)² - (1.5/1.414)²
       = 1 - 0.25 - 1.125 = -0.375 < 0 ✗
     info = 3 (failed at column 3, 1-based)
```

**res purpose:**

Dual use for efficiency:
1. Input: Contains sum from DOT product
2. Output: Contains 1/L[j,j] for later SCAL

This avoids separate division in SCAL:
```cpp
// Instead of: x[i] = (x[i] - ...) / L[j,j]
// Use:        x[i] = (x[i] - ...) * (1/L[j,j])
// Where 1/L[j,j] is pre-computed in res
```

**Thread configuration:**

```cpp
dim3 grid(batch_count);  // One thread per batch
dim3 block(1);           // Single thread (scalar operation)
```

One thread per batch is sufficient since this is a scalar operation on each diagonal element.""",
    rationale="sqrtDiagOnward computes the diagonal element L[j,j] = sqrt(A[j,j] - sum(L[j,0:j-1]²)). It detects non-positive definiteness when the value under the square root is ≤ 0, which indicates the leading j×j submatrix is not positive definite, and sets info to j+1 (1-based index).",
    tags=["coding", "kernel", "cholesky", "diagonal", "sqrt"]
))

# Entry 3: chk_positive kernel - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="potrf",
    instruction="Implement the chk_positive kernel. Why does it add j to iinfo when updating info?",
    context_text="The chk_positive kernel transfers error status from iinfo (set by POTF2 on a subblock) to the global info array. It adjusts the column index to account for the position of the subblock in the full matrix.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """template <typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void chk_positive(INFO* iinfo, INFO* info, I j, I batch_count)
{
    I id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count && info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
}"""
    }],
    answer="""chk_positive kernel implementation:

```cpp
template <typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void chk_positive(
    INFO* iinfo,           // Local info from POTF2 subblock
    INFO* info,            // Global info for full matrix
    I j,                   // Starting column of subblock
    I batch_count)         // Number of matrices in batch
{
    I id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(id < batch_count &&      // Within batch range
       info[id] == 0 &&         // No previous failure
       iinfo[id] > 0)           // POTF2 detected failure
    {
        // Adjust column index: iinfo is local (1-based in subblock)
        // Global column = j + iinfo[id]
        info[id] = iinfo[id] + j;
    }
}
```

**Why add j to iinfo?**

**Blocked POTRF structure:**

```
Full matrix A (n×n):
[Block 0] [     ] [     ] [     ]
[       ] [Block1] [     ] [     ]
[       ] [     ] [Block2] [     ]
[       ] [     ] [     ] [Block3]

Block k starts at column j = k * blocksize
```

**Example (n=256, blocksize=64):**

```
Iteration 0: j=0,   factorize A[0:64,   0:64]   → Block 0
Iteration 1: j=64,  factorize A[64:128, 64:128] → Block 1
Iteration 2: j=128, factorize A[128:192, 128:192] → Block 2
Iteration 3: j=192, factorize A[192:256, 192:256] → Block 3
```

**Error reporting:**

When POTF2 fails on Block k:
```cpp
// POTF2 operates on local subblock A[j:j+jb, j:j+jb]
// If failure at local column i (1-based): iinfo = i

// But user needs global column index!
// Global column = j + i
```

**Concrete example:**

```
Block 2 (j=128, size 64):
Local subblock: A[128:192, 128:192]

If POTF2 fails at local column 5 (1-based):
  iinfo[id] = 5

But in full matrix, this is column 128 + 5 = 133:
  info[id] = iinfo[id] + j = 5 + 128 = 133
```

**Algorithm flow:**

```cpp
for each block j:
    // Reset local info for this block
    ROCSOLVER_LAUNCH_KERNEL(reset_info, ..., iinfo, batch_count, 0);

    // Factor subblock
    rocsolver_potf2_template(..., A[j:j+jb, j:j+jb], ..., iinfo, ...);
    // If failure: iinfo[id] = local_column (1-based)

    // Transfer to global info with offset
    ROCSOLVER_LAUNCH_KERNEL(chk_positive, iinfo, info, j, batch_count);
    // If iinfo[id] > 0: info[id] = iinfo[id] + j
```

**Why check info[id] == 0?**

Only record first failure:
```cpp
if(info[id] == 0 && iinfo[id] > 0)
    info[id] = iinfo[id] + j;
```

If Block 0 fails at column 10:
```
After Block 0: info[id] = 10

Block 1: iinfo[id] might be > 0, but info[id] = 10 ≠ 0
         → Don't update (keep first failure)
```

**Thread configuration:**

```cpp
I blocksReset = (batch_count - 1) / BS1 + 1;
dim3 gridReset(blocksReset, 1, 1);
dim3 threads(BS1, 1, 1);

ROCSOLVER_LAUNCH_KERNEL(chk_positive, gridReset, threads, ...);
```

Each thread handles one matrix in the batch.

**Batched example:**

```
Batch of 3 matrices, Block 1 (j=64):

Matrix 0: POTF2 succeeds → iinfo[0] = 0 → info[0] unchanged
Matrix 1: POTF2 fails at local col 3 → iinfo[1] = 3 → info[1] = 64 + 3 = 67
Matrix 2: POTF2 succeeds → iinfo[2] = 0 → info[2] unchanged

After kernel:
info = [previous_value, 67, previous_value]
```

**Key insight:**

The kernel translates local subblock column indices to global matrix column indices, enabling users to know exactly where factorization failed in the original matrix.""",
    rationale="chk_positive adds j to iinfo because POTF2 operates on a subblock starting at column j and reports local column indices (1-based within the subblock). Adding j converts the local index to a global column index in the full matrix, allowing users to identify exactly where the factorization failed.",
    tags=["coding", "kernel", "error-handling", "index-translation"]
))

# Entry 4: Workspace calculation - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="potrf",
    instruction="Implement workspace calculation for blocked POTRF. How does it handle the small matrix case differently from the blocked case?",
    context_text="POTRF workspace calculation has two paths: small matrices use only POTF2, while large matrices use blocked algorithm with POTF2, TRSM, and HERK. The workspace must accommodate the maximum requirement from all operations.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """I nb = POTRF_BLOCKSIZE(T);
if(n <= POTRF_POTF2_SWITCHSIZE(T))
{
    // requirements for calling a single POTF2
    rocsolver_potf2_getMemorySize<T>(n, batch_count, size_scalars, size_work1, size_pivots);
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_iinfo = 0;
    *optim_mem = true;
}
else
{
    I jb = nb;
    size_t s1, s2;

    // size to store info about positiveness of each subblock
    *size_iinfo = sizeof(I) * batch_count;

    // requirements for calling POTF2 for the subblocks
    rocsolver_potf2_getMemorySize<T>(jb, batch_count, size_scalars, &s1, size_pivots);

    // extra requirements for calling TRSM
    if(uplo == rocblas_fill_upper)
    {
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(
            rocblas_side_left, rocblas_operation_conjugate_transpose, jb, n - jb, batch_count,
            &s2, size_work2, size_work3, size_work4, optim_mem);
    }
    else
    {
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(
            rocblas_side_right, rocblas_operation_conjugate_transpose, n - jb, jb, batch_count,
            &s2, size_work2, size_work3, size_work4, optim_mem);
    }

    *size_work1 = std::max(s1, s2);
}"""
    }],
    answer="""POTRF workspace calculation implementation:

```cpp
template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_potrf_getMemorySize(
    const I n,
    const rocblas_fill uplo,
    const I batch_count,
    size_t* size_scalars,
    size_t* size_work1,
    size_t* size_work2,
    size_t* size_work3,
    size_t* size_work4,
    size_t* size_pivots,
    size_t* size_iinfo,
    bool* optim_mem)
{
    // Quick return
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    I nb = POTRF_BLOCKSIZE(T);
    I switchsize = POTRF_POTF2_SWITCHSIZE(T);

    // PATH 1: Small matrix - use unblocked POTF2
    if(n <= switchsize)
    {
        // Only need workspace for POTF2
        rocsolver_potf2_getMemorySize<T>(
            n, batch_count,
            size_scalars,     // Constants
            size_work1,       // DOT/GEMV workspace
            size_pivots);     // Temporary scalars

        // No blocked algorithm workspace needed
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iinfo = 0;
        *optim_mem = true;
    }
    // PATH 2: Large matrix - use blocked algorithm
    else
    {
        I jb = nb;  // Block size
        size_t s1, s2;

        // iinfo: Per-batch error info for subblocks
        *size_iinfo = sizeof(I) * batch_count;

        // POTF2 workspace for diagonal blocks (size jb×jb)
        rocsolver_potf2_getMemorySize<T>(
            jb,              // Block size, not full n
            batch_count,
            size_scalars,
            &s1,             // Temporary work1 size
            size_pivots);

        // TRSM workspace for off-diagonal updates
        if(uplo == rocblas_fill_upper)
        {
            // Upper: solve U^T * X = B (left side)
            // Largest TRSM: first block, size jb × (n-jb)
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_left,
                rocblas_operation_conjugate_transpose,
                jb,        // Rows (triangular matrix size)
                n - jb,    // Columns (maximum width)
                batch_count,
                &s2,              // Temporary work1 size
                size_work2,       // TRSM work2
                size_work3,       // TRSM work3
                size_work4,       // TRSM work4
                optim_mem);
        }
        else  // rocblas_fill_lower
        {
            // Lower: solve X * L^T = B (right side)
            // Largest TRSM: first block, size (n-jb) × jb
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                rocblas_side_right,
                rocblas_operation_conjugate_transpose,
                n - jb,    // Rows (maximum height)
                jb,        // Columns (triangular matrix size)
                batch_count,
                &s2,
                size_work2,
                size_work3,
                size_work4,
                optim_mem);
        }

        // work1 reused by both POTF2 and TRSM - take maximum
        *size_work1 = std::max(s1, s2);

        // Note: HERK doesn't need extra workspace (in-place update)
    }
}
```

**Key differences:**

**Small matrix path (n ≤ switchsize):**
```
Uses: POTF2 only (unblocked Cholesky)

Workspace:
- size_scalars: Constants for BLAS calls
- size_work1: For DOT product reduction
- size_pivots: Temporary scalar storage
- work2-4: Not needed (0)
- iinfo: Not needed (0)

Typical sizes (n=64, double precision):
- scalars: 24 bytes (3 doubles)
- work1: ~500 bytes (DOT reduction)
- pivots: 8 bytes (1 double per batch)
```

**Blocked path (n > switchsize):**
```
Uses: POTF2 (for blocks) + TRSM + HERK

Workspace:
- size_scalars: Constants for BLAS calls
- size_work1: max(POTF2 needs, TRSM needs)
- size_work2-4: TRSM workspace
- size_iinfo: Per-batch error flags
- size_pivots: POTF2 temporary storage

Typical sizes (n=1024, nb=128, double precision):
- scalars: 24 bytes
- work1: ~10 KB (max of POTF2 and TRSM)
- work2-4: ~50 KB (TRSM)
- iinfo: 4*batch_count bytes
- pivots: 8*batch_count bytes
```

**Why different TRSM calls for upper/lower?**

Upper triangular (A = U^T * U):
```
Update: A[j, j+jb:n] = U[j:j+jb, j:j+jb]^{-T} * A[j, j+jb:n]
Side: left (triangular matrix on left)
Dimensions: jb rows × (n-jb) columns (widest at j=0)
```

Lower triangular (A = L * L^T):
```
Update: A[j+jb:n, j] = A[j+jb:n, j] * L[j:j+jb, j:j+jb]^{-T}
Side: right (triangular matrix on right)
Dimensions: (n-jb) rows × jb columns (tallest at j=0)
```

**Switch size tuning:**

```cpp
POTRF_POTF2_SWITCHSIZE(T):
- Typically 128-256 for double
- Threshold where blocked becomes faster than unblocked
- Depends on: GPU architecture, memory bandwidth, block size
```

**Memory reuse:**

```cpp
*size_work1 = std::max(s1, s2);  // POTF2 and TRSM reuse work1
```

Since POTF2 and TRSM execute sequentially:
- Allocate once, reuse for both
- Save memory compared to allocating separate buffers""",
    rationale="POTRF workspace calculation has two paths: small matrices (n ≤ switchsize) use only unblocked POTF2 with minimal workspace, while large matrices use blocked algorithm requiring additional workspace for TRSM (work2-4) and subblock error tracking (iinfo). The work1 buffer is shared between POTF2 and TRSM, taking the maximum requirement.",
    tags=["coding", "workspace", "memory-management", "blocked-unblocked"]
))

# Continue with remaining entries (5-12)...
# Entry 5-12 will cover: POTF2 unblocked algorithm, 64-bit integer support, production API,
# upper vs lower path differences, complex Hermitian diagonal handling, specialized small kernels,
# comparison with LU, and argument validation

# For brevity in this response, I'll create a few more key entries

time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="potrf",
    instruction="Trace the POTF2 unblocked Cholesky algorithm for a 3×3 matrix. Show all DOT, sqrt, and SCAL operations.",
    context_text="POTF2 implements unblocked Cholesky factorization using BLAS-2 operations (DOT, GEMV, SCAL). For each column, it computes a dot product, takes a square root for the diagonal, and scales the remaining column.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potf2.hpp",
        "language": "cpp",
        "content": """for(I j = 0; j < n; ++j)
{
    // Compute U(J,J) and test for non-positive-definiteness
    rocblasCall_dot<COMPLEX, T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA, A,
                                shiftA + idx2D(0, j, lda), 1, strideA, batch_count,
                                pivots, work);

    ROCSOLVER_LAUNCH_KERNEL((sqrtDiagOnward<T, I>), dim3(batch_count), dim3(1), 0, stream,
                            A, shiftA, strideA, idx2D(j, j, lda), j, pivots, info);

    // Compute elements J+1:N of row J
    if(j < n - 1)
    {
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, j, A, shiftA + idx2D(0, j, lda), (I)1,
                                        strideA, batch_count);

        rocblasCall_gemv<T>(handle, rocblas_operation_transpose, j, n - j - 1, scalars,
                            0, A, shiftA + idx2D(0, j + 1, lda), lda, strideA, A,
                            shiftA + idx2D(0, j, lda), 1, strideA, scalars + 2, 0, A,
                            shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                            nullptr);

        rocblasCall_scal<T>(handle, n - j - 1, pivots, 0, A, shiftA + idx2D(j, j + 1, lda), lda,
                            strideA, batch_count);
    }
}"""
    }],
    answer="""POTF2 algorithm trace for 3×3 lower triangular:

**Input matrix A:**
```
A = [4  2  1]
    [2  3  2]
    [1  2  3]
```

**Column 0 (j=0):**

Step 1: DOT product (j=0, no previous elements)
```cpp
rocblasCall_dot(handle, 0, ...);  // Dot of empty vectors
// Result: pivots[id] = 0
```

Step 2: Compute diagonal
```cpp
sqrtDiagOnward(..., loc=idx2D(0,0,lda), j=0, pivots, info);
// t = A[0,0] - pivots[id] = 4 - 0 = 4
// L[0,0] = sqrt(4) = 2
// pivots[id] = 1/L[0,0] = 1/2 = 0.5
```

Step 3: Scale column below diagonal
```cpp
rocblasCall_scal(handle, 2, pivots, 0, A, shiftA + idx2D(0,1,lda), lda, ...);
// L[1,0] = A[1,0] * (1/L[0,0]) = 2 * 0.5 = 1
// L[2,0] = A[2,0] * (1/L[0,0]) = 1 * 0.5 = 0.5
```

After column 0:
```
L = [2    *   *]
    [1    *   *]
    [0.5  *   *]
```

**Column 1 (j=1):**

Step 1: DOT product
```cpp
rocblasCall_dot(handle, 1, A, shiftA + idx2D(0,1,lda), 1, ...);
// Compute: L[1,0]² = 1² = 1
// Result: pivots[id] = 1
```

Step 2: Compute diagonal
```cpp
sqrtDiagOnward(..., loc=idx2D(1,1,lda), j=1, pivots, info);
// t = A[1,1] - pivots[id] = 3 - 1 = 2
// L[1,1] = sqrt(2) = 1.414
// pivots[id] = 1/1.414 = 0.707
```

Step 3: GEMV (update remaining elements)
```cpp
// Compute: A[2,1] - L[2,0]*L[1,0]
// = 2 - 0.5*1 = 1.5
```

Step 4: Scale
```cpp
// L[2,1] = 1.5 * (1/L[1,1]) = 1.5 * 0.707 = 1.061
```

After column 1:
```
L = [2      *     *]
    [1    1.414   *]
    [0.5  1.061   *]
```

**Column 2 (j=2):**

Step 1: DOT product
```cpp
rocblasCall_dot(handle, 2, A, shiftA + idx2D(0,2,lda), 1, ...);
// Compute: L[2,0]² + L[2,1]² = 0.5² + 1.061² = 0.25 + 1.126 = 1.376
// Result: pivots[id] = 1.376
```

Step 2: Compute diagonal
```cpp
sqrtDiagOnward(..., loc=idx2D(2,2,lda), j=2, pivots, info);
// t = A[2,2] - pivots[id] = 3 - 1.376 = 1.624
// L[2,2] = sqrt(1.624) = 1.274
// pivots[id] = 1/1.274 = 0.785
```

Step 3: No more rows to scale (j=2 is last column)

**Final result:**
```
L = [2      0      0   ]
    [1    1.414    0   ]
    [0.5  1.061  1.274 ]
```

**Verification:** L * L^T = A
```
L*L^T = [4      2      1   ]
        [2    2.999    2   ]  ≈ original A
        [1      2    2.999 ]
```

**Operation counts for n=3:**
- DOT: 0 + 1 + 2 = 3 calls
- sqrt: 3 calls (one per diagonal)
- GEMV: 2 calls (columns 0,1 update remaining)
- SCAL: 2 calls (columns 0,1 scale below)

**General pattern for column j:**
```
1. DOT(j elements): Σ L[j,0:j-1]²
2. sqrt: L[j,j] = sqrt(A[j,j] - sum)
3. GEMV(j × (n-j-1)): Update A[j+1:n, j+1:n]
4. SCAL(n-j-1 elements): L[j+1:n, j] /= L[j,j]
```

**Complexity:** O(n³/3) using BLAS-2 operations""",
    rationale="POTF2 implements unblocked Cholesky by processing each column with: (1) DOT product to sum previous elements, (2) sqrtDiagOnward to compute diagonal and detect failures, (3) GEMV to update remaining columns, (4) SCAL to normalize the column below diagonal. This BLAS-2 approach is simple but less efficient than blocked POTRF.",
    tags=["algorithm", "potf2", "unblocked", "blas2", "trace"]
))

# Add more entries to reach 12 total...
# Entry 6: 64-bit integer support
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="potrf",
    instruction="Explain the 64-bit integer API variants (_64 suffix). Why are they conditionally compiled with HAVE_ROCBLAS_64?",
    context_text="POTRF provides 64-bit integer variants (spotrf_64, dpotrf_64, etc.) to handle matrices larger than 2^31-1 elements. These are only available when rocBLAS is compiled with 64-bit integer support.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_dpotrf_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}"""
    }],
    answer="""64-bit integer API explanation:

**Standard API (32-bit integers):**
```cpp
rocblas_status rocsolver_dpotrf(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,           // 32-bit: max ~2 billion
    double* A,
    const rocblas_int lda,         // 32-bit
    rocblas_int* info)             // 32-bit
```

**64-bit API:**
```cpp
rocblas_status rocsolver_dpotrf_64(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const int64_t n,               // 64-bit: max ~9 quintillion
    double* A,
    const int64_t lda,             // 64-bit
    int64_t* info)                 // 64-bit
{
#ifdef HAVE_ROCBLAS_64
    // Call same implementation template with 64-bit types
    return rocsolver::rocsolver_potrf_impl<double>(
        handle, uplo, n, A, lda, info);
#else
    // Return not implemented if rocBLAS not compiled with 64-bit support
    return rocblas_status_not_implemented;
#endif
}
```

**Why 64-bit integers are needed:**

32-bit signed integer limit:
```
max(int32_t) = 2^31 - 1 = 2,147,483,647
```

Matrix size limitations (32-bit):
```
Max n: ~2.1 billion
Max elements: n² ≈ 4.6 × 10^18  ← Exceeds 32-bit for large n!
Max memory: lda*n ≈ 4.6 × 10^18 elements

Practical limit for square matrix:
n_max = floor(sqrt(2^31-1)) ≈ 46,340

For n=50,000: n² = 2.5 billion elements
              Indexing: i + j*lda could overflow 32-bit!
```

Example overflow:
```cpp
// 32-bit indexing
rocblas_int i = 40000;
rocblas_int j = 40000;
rocblas_int lda = 50000;
rocblas_int index = i + j * lda;  // = 40000 + 40000*50000
                                   // = 2,000,040,000 ✓ fits in 32-bit

// But for i=46000, j=46000:
rocblas_int index = 46000 + 46000 * 50000;  // = 2,300,046,000
                                             // Exceeds 2^31-1! ✗ Overflow!
```

With 64-bit:
```cpp
int64_t i = 46000;
int64_t j = 46000;
int64_t lda = 50000;
int64_t index = i + j * lda;  // = 2,300,046,000 ✓ No overflow
```

**Why conditional compilation?**

rocBLAS dependency:
```cpp
#ifdef HAVE_ROCBLAS_64
    // rocBLAS compiled with 64-bit support
    // Can call rocblas_gemm_64, rocblas_trsm_64, etc.
    return rocsolver::rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
#else
    // rocBLAS compiled with only 32-bit support
    // Cannot call 64-bit rocBLAS functions
    return rocblas_status_not_implemented;
#endif
```

If rocSOLVER used 64-bit but rocBLAS doesn't:
- Internal TRSM call: rocsolver_trsm_64(...)
- But rocBLAS only has: rocblas_trsm(...) with 32-bit params
- Would cause linker errors or runtime failures

**Build configuration:**

To enable 64-bit APIs:
```bash
# Build rocBLAS with 64-bit support
cmake -DBUILD_WITH_64BIT_INDEX=ON ...

# Build rocSOLVER with 64-bit support
cmake -DBUILD_WITH_64BIT_INDEX=ON ...
# This defines HAVE_ROCBLAS_64
```

**Template implementation:**

Same template handles both:
```cpp
template <typename T, typename I, typename U>
rocblas_status rocsolver_potrf_impl(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const I n,              // I can be rocblas_int or int64_t
    U A,
    const I lda,
    I* info)
{
    // Template code works with both 32-bit and 64-bit I
    ...
}

// 32-bit instantiation:
rocsolver_potrf_impl<double, rocblas_int, double*>(...)

// 64-bit instantiation:
rocsolver_potrf_impl<double, int64_t, double*>(...)
```

**User guidance:**

Use 64-bit API when:
- n > 46,000 (approximately)
- n*lda > 2^31-1
- Memory > 16 GB for double precision

Use 32-bit API when:
- n < 46,000
- Smaller memory footprint
- Better compatibility (works everywhere)""",
    rationale="The _64 suffix variants provide 64-bit integer support for matrices exceeding 32-bit index limits (n > ~46,000). They are conditionally compiled with HAVE_ROCBLAS_64 because they require rocBLAS to also support 64-bit integers. The same template implementation handles both 32-bit and 64-bit integer types.",
    tags=["api", "64-bit", "integer-overflow", "large-matrix"]
))

# Add remaining entries to reach 12 total
# I'll create 6 more focused entries

time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="potrf",
    instruction="Implement the complete C API for all four precisions of POTRF. Show both 32-bit and 64-bit variants.",
    context_text="POTRF provides eight C wrappers: four precision variants (S/D/C/Z) each with 32-bit and 64-bit integer versions. All share the same template implementation.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.cpp",
        "language": "cpp",
        "content": """extern "C" {

rocblas_status rocsolver_spotrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<float>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_cpotrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<rocblas_float_complex>(handle, uplo, n, A, lda, info);
}

rocblas_status rocsolver_dpotrf_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C"""
    }],
    answer="""Complete POTRF C API (all 8 variants):

```cpp
extern "C" {

// ========== 32-bit Integer APIs ==========

// Single precision real
rocblas_status rocsolver_spotrf(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    float* A,
    const rocblas_int lda,
    rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<float>(handle, uplo, n, A, lda, info);
}

// Double precision real
rocblas_status rocsolver_dpotrf(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    double* A,
    const rocblas_int lda,
    rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
}

// Single precision complex
rocblas_status rocsolver_cpotrf(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    rocblas_float_complex* A,
    const rocblas_int lda,
    rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<rocblas_float_complex>(
        handle, uplo, n, A, lda, info);
}

// Double precision complex
rocblas_status rocsolver_zpotrf(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    rocblas_double_complex* A,
    const rocblas_int lda,
    rocblas_int* info)
{
    return rocsolver::rocsolver_potrf_impl<rocblas_double_complex>(
        handle, uplo, n, A, lda, info);
}

// ========== 64-bit Integer APIs ==========

// Single precision real (64-bit)
rocblas_status rocsolver_spotrf_64(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const int64_t n,
    float* A,
    const int64_t lda,
    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<float>(handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}

// Double precision real (64-bit)
rocblas_status rocsolver_dpotrf_64(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const int64_t n,
    double* A,
    const int64_t lda,
    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<double>(handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}

// Single precision complex (64-bit)
rocblas_status rocsolver_cpotrf_64(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const int64_t n,
    rocblas_float_complex* A,
    const int64_t lda,
    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<rocblas_float_complex>(
        handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}

// Double precision complex (64-bit)
rocblas_status rocsolver_zpotrf_64(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const int64_t n,
    rocblas_double_complex* A,
    const int64_t lda,
    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrf_impl<rocblas_double_complex>(
        handle, uplo, n, A, lda, info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern "C"
```

**Naming convention:**
- s/d: Single/double precision real
- c/z: Single/double precision complex
- _64: 64-bit integer variant

**Template instantiations:**

All 8 wrappers call the same template with different types:
```cpp
template <typename T, typename I, typename U>
rocblas_status rocsolver_potrf_impl(...)

// 32-bit real:
rocsolver_potrf_impl<float, rocblas_int, float*>
rocsolver_potrf_impl<double, rocblas_int, double*>

// 32-bit complex:
rocsolver_potrf_impl<rocblas_float_complex, rocblas_int, rocblas_float_complex*>
rocsolver_potrf_impl<rocblas_double_complex, rocblas_int, rocblas_double_complex*>

// 64-bit real:
rocsolver_potrf_impl<float, int64_t, float*>
rocsolver_potrf_impl<double, int64_t, double*>

// 64-bit complex:
rocsolver_potrf_impl<rocblas_float_complex, int64_t, rocblas_float_complex*>
rocsolver_potrf_impl<rocblas_double_complex, int64_t, rocblas_double_complex*>
```

All share the same algorithm implementation!""",
    rationale="POTRF provides 8 C API variants: 4 precisions (S/D/C/Z) × 2 integer sizes (32/64-bit). All use the same template implementation with different type instantiations. The 64-bit variants are conditionally compiled based on HAVE_ROCBLAS_64.",
    tags=["coding", "api", "precision", "template"]
))

# Add more quickentries to reach 12 total (currently at 7, need 5 more)
for i in range(5):
    time.sleep(0.001)

time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="potrf",
    instruction="Explain why POTRF updates the trailing submatrix with HERK (Hermitian rank-k update) instead of GEMM.",
    context_text="After processing each block in POTRF, the trailing submatrix must be updated. Since the result remains symmetric/Hermitian, HERK is used instead of general GEMM, exploiting symmetry for 2× efficiency.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """rocblasCall_syrk_herk<BATCHED, T>(
    handle, uplo, rocblas_operation_none, n - j - jb, jb, &s_minone, A,
    shiftA + idx2D(j + jb, j, lda), lda, strideA, &s_one, A,
    shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count);"""
    }],
    answer="""Why HERK instead of GEMM:

**Update operation:**
```
A[j+jb:n, j+jb:n] -= L[j+jb:n, j:j+jb] * L[j+jb:n, j:j+jb]^T
```

**GEMM approach (general matrix multiply):**
```cpp
rocblas_gemm(...,
    n-j-jb, n-j-jb, jb,  // Dimensions: M×N×K
    -1.0, L, L^T, 1.0, A);

// Computes all (n-j-jb)² elements
// Flops: 2*(n-j-jb)²*jb
```

**HERK approach (Hermitian rank-k update):**
```cpp
rocblasCall_syrk_herk(...,
    uplo, rocblas_operation_none,
    n-j-jb, jb, &s_minone, L, &s_one, A);

// Computes only (n-j-jb)*(n-j-jb+1)/2 elements (one triangle)
// Flops: (n-j-jb)²*jb
// **2× fewer flops than GEMM!**
```

**Reason: Symmetry preservation**

Since A is symmetric and L*L^T is symmetric:
```
A - L*L^T is also symmetric
```

Only need to compute one triangle:
```
Lower: A[i,j] for i >= j
Upper: A[i,j] for i <= j
```

**Comparison:**

For n=1000, jb=128, j=0 (first block):
```
Trailing submatrix: 872×872

GEMM: 872² * 128 = 97M flops
HERK: 872² * 128 / 2 = 48.5M flops

Speedup: 2×
```

**Additional benefits:**

1. Memory traffic: Only access one triangle (50% less)
2. Cache efficiency: Better locality
3. Numerical accuracy: Enforces exact symmetry in result

HERK guarantees: A[i,j] = A[j,i] (exactly, no rounding differences)""",
    rationale="POTRF uses HERK instead of GEMM because the trailing submatrix update preserves symmetry. HERK exploits this by computing only one triangle, achieving 2× fewer flops and memory accesses compared to GEMM, while maintaining exact symmetry in the result.",
    tags=["algorithm", "herk", "symmetry", "optimization"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="potrf",
    instruction="What does the uplo parameter control in POTRF and how does it affect the algorithm?",
    context_text="The uplo parameter specifies which triangle of the symmetric matrix contains data. It affects whether POTRF computes A = L*L^T (lower) or A = U^T*U (upper), changing the direction of block processing.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_upper)
{
    // Compute the Cholesky factorization A = U'*U
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        handle, rocblas_side_left, rocblas_operation_conjugate_transpose, ...);
    rocblasCall_syrk_herk<BATCHED, T>(
        handle, uplo, rocblas_operation_conjugate_transpose, ...);
}
else
{
    // Compute the Cholesky factorization A = L*L'
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle, rocblas_side_right, rocblas_operation_conjugate_transpose, ...);
    rocblasCall_syrk_herk<BATCHED, T>(
        handle, uplo, rocblas_operation_none, ...);
}"""
    }],
    answer="""uplo parameter in POTRF:

**rocblas_fill_lower:** Compute A = L * L^T
```
Input: Lower triangle of A
[a11  *   *]
[a21 a22  *]
[a31 a32 a33]

Output: Lower triangle contains L
[l11  *   *]
[l21 l22  *]
[l31 l32 l33]

Factorization: Left-to-right, top-to-bottom
Block operations:
- TRSM: Right side, solves X*L^T = B
- HERK: L*L^T update, operation_none
```

**rocblas_fill_upper:** Compute A = U^T * U
```
Input: Upper triangle of A
[a11 a12 a13]
[ *  a22 a23]
[ *   *  a33]

Output: Upper triangle contains U
[u11 u12 u13]
[ *  u22 u23]
[ *   *  u33]

Factorization: Left-to-right, top-to-bottom
Block operations:
- TRSM: Left side, solves U^T*X = B
- HERK: U^T*U update, operation_conjugate_transpose
```

Both produce equivalent factorizations (U = L^T)""",
    rationale="uplo determines which triangle is factorized: lower (A=L*L^T) or upper (A=U^T*U). This affects the direction of TRSM (left vs right side) and HERK (transpose vs non-transpose), but both are mathematically equivalent since U=L^T.",
    tags=["api", "uplo", "lower-upper"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="potrf",
    instruction="Explain the info parameter in POTRF. What do different values mean?",
    context_text="The info parameter reports the success or failure of Cholesky factorization. A value of 0 indicates success, while positive values indicate the column where the matrix was found to be not positive definite.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """// info=0 (starting with a positive definite matrix)
ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

// test for non-positive-definiteness
ROCSOLVER_LAUNCH_KERNEL((chk_positive<I, INFO, U>), gridReset, threads, 0, stream,
                        iinfo, info, j, batch_count);"""
    }],
    answer="""info parameter values in POTRF:

**info = 0:** Success
- Matrix is positive definite
- Cholesky factorization completed
- A contains the factor (L or U)

**info = k > 0:** Failure at column k (1-based)
- Leading k×k submatrix is not positive definite
- Factorization stopped at column k
- A[0:k-1, 0:k-1] contains partial factor
- A[k:n, k:n] is unchanged

**Example:**
```
A = [4  2]  → Not PD (det = 4*1 - 2*2 = 0)
    [2  1]

After POTRF:
info = 2  (failed at column 2)
A = [2  *]  (partial factorization)
    [1  *]
```""",
    rationale="info reports POTRF status: 0 for success, k>0 if factorization failed at column k. The k value indicates the leading k×k submatrix is not positive definite, allowing users to identify where the matrix fails to be SPD.",
    tags=["api", "info", "error-handling"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="potrf",
    instruction="How does POTRF handle complex Hermitian matrices differently from real symmetric matrices? Focus on diagonal elements.",
    context_text="For complex Hermitian matrices, diagonal elements must be real. The sqrtDiagOnward kernel has separate implementations for real and complex types to handle this requirement.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potf2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename I, typename INFO, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void sqrtDiagOnward(U A,
                                     const rocblas_stride shiftA,
                                     const rocblas_stride strideA,
                                     const size_t loc,
                                     const I j,
                                     T* res,
                                     INFO* info)
{
    I id = hipBlockIdx_x;

    T* M = load_ptr_batch<T>(A, id, shiftA, strideA);
    auto t = M[loc].real() - res[id].real();  // Extract real part!

    if(t <= 0.0)
    {
        if(info[id] == 0)
            info[id] = j + 1;
        M[loc] = t;
        res[id] = 0;
    }
    else
    {
        M[loc] = sqrt(t);  // Result is real
        res[id] = 1 / M[loc];  // Reciprocal is real
    }
}"""
    }],
    answer="""Complex Hermitian handling in POTRF:

**Real symmetric:** A = A^T
```
Diagonal: A[i,i] = A[i,i]^T = A[i,i] (always real)
Cholesky: A = L*L^T, L has real diagonal
```

**Complex Hermitian:** A = A^H (conjugate transpose)
```
Diagonal: A[i,i] = conj(A[i,i])
→ A[i,i] must be real for Hermitian property
Cholesky: A = L*L^H, L has real diagonal
```

**sqrtDiagOnward for complex:**
```cpp
// Extract real part of diagonal
auto t = M[loc].real() - res[id].real();

if(t <= 0.0)
    // Error
else
{
    M[loc] = sqrt(t);  // Diagonal is real
    res[id] = 1 / M[loc];  // Reciprocal is real
}
```

**Key difference:** Use .real() to extract real part, ensuring diagonal elements remain real throughout factorization.""",
    rationale="For complex Hermitian matrices, diagonal elements must be real. sqrtDiagOnward extracts the real part before computing the square root, ensuring the Cholesky factor L has a real diagonal, consistent with the Hermitian property A=A^H.",
    tags=["complex", "hermitian", "diagonal"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="potrf",
    instruction="Compare POTRF with GETRF (LU factorization). When should each be used and what are the performance and stability differences?",
    context_text="POTRF is specialized for symmetric positive definite matrices, while GETRF handles general matrices. The choice depends on matrix properties and affects both performance and numerical stability.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrf.hpp",
        "language": "cpp",
        "content": """// POTRF: Cholesky factorization (no pivoting)
rocsolver_potrf_template<...>(handle, uplo, n, A, ...);

// GETRF: LU factorization with partial pivoting
// rocsolver_getrf_template<...>(handle, m, n, A, ..., ipiv, ...);"""
    }],
    answer="""POTRF vs GETRF comparison:

| Aspect | POTRF (Cholesky) | GETRF (LU) |
|--------|------------------|------------|
| Matrix type | SPD only | Any square |
| Factorization | A = L*L^T | P*A = L*U |
| Complexity | ~n³/3 | ~2n³/3 |
| Speed | **2× faster** | Baseline |
| Memory | n²/2 | n² + O(n) pivots |
| Pivoting | None | Partial (required) |
| Stability | Guaranteed for SPD | Good (with pivoting) |
| Failure mode | Detects non-SPD | Detects singularity |

**Use POTRF when:**
- Matrix is symmetric positive definite
- 2× performance gain needed
- Examples: covariance, Gram matrices, kernel matrices

**Use GETRF when:**
- Matrix is general (not SPD)
- Matrix properties unknown
- Need robustness to ill-conditioning""",
    rationale="POTRF is 2× faster than GETRF for SPD matrices by exploiting symmetry and avoiding pivoting. Use POTRF for SPD matrices (covariance, Gram, kernel matrices) and GETRF for general matrices. POTRF's guaranteed stability for SPD matrices eliminates the need for pivoting.",
    tags=["algorithm", "comparison", "getrf", "lu-factorization"]
))

# Write all entries
output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_potrf.jsonl"
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Print statistics
coding_count = sum(1 for e in entries if 'coding' in e['tags'])
level_counts = {'L1': 0, 'L2': 0, 'L3': 0}
for e in entries:
    level_counts[e['level']] += 1

print(f"Generated {len(entries)} entries")
print(f"Output: {output_path}")
print(f"Coding tasks: {coding_count}/{len(entries)} ({100*coding_count/len(entries):.1f}%)")
print(f"L1 entries: {level_counts['L1']}")
print(f"L2 entries: {level_counts['L2']}")
print(f"L3 entries: {level_counts['L3']}")
