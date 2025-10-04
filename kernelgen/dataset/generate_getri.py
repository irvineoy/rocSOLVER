#!/usr/bin/env python3
"""Generate dataset for roclapack_getri.yaml"""

import json
import sys

def create_entry(entry_id, level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    return {
        "id": entry_id,
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

# Entry 1: Complete GETRI algorithm (L3)
entries.append(create_entry(
    "roclapack_getri_001",
    "L3",
    "code",
    "Explain the complete GETRI algorithm for computing matrix inversion using LU factorization. What are the two main stages?",
    """GETRI computes the inverse of a general n-by-n matrix A using the LU factorization computed by GETRF: A = P*L*U. The input matrix A must contain the L and U factors from GETRF, and ipiv must contain the pivot indices.

Source files:
- library/src/lapack/roclapack_getri.hpp (main algorithm)
- library/src/lapack/roclapack_getri.cpp (C API)
- library/src/lapack/roclapack_trtri.hpp (triangular inversion)""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 290,
            "end_line": 404,
            "language": "cpp"
        }
    ],
    """GETRI computes A^(-1) in two stages using the LU factorization A = P*L*U:

**Stage 1: Compute inv(U) using TRTRI**
- Calls rocsolver_trtri_template with rocblas_fill_upper, rocblas_diagonal_non_unit
- Computes the inverse of the upper triangular factor U in-place
- Also checks for singularity: if U(i,i) = 0 for any i, sets info = i and the inverse cannot be computed
- For non-singular matrices, replaces U with inv(U) in the upper triangle of A

**Stage 2: Solve inv(A) * L = inv(U) for inv(A)**
This is done using a blocked backward column sweep:

For j = n-1, n-1-blk, ..., 0 (backwards by block size blk):
  jb = min(n-j, blk)  // Current block width
  
  1. **Copy and zero**: Copy the strictly lower triangular part L(j:n, j:j+jb) to workspace tmpcopy, then zero it in A
  
  2. **GEMM update** (if j+jb < n):
     A(:, j:j+jb) -= A(:, j+jb:n) * L(j+jb:n, j:j+jb)
     This updates the current block columns using already-computed columns to the right
  
  3. **TRSM solve**:
     Solve A(:, j:j+jb) * L(j:j+jb, j:j+jb) = A(:, j:j+jb)
     This is rocblasCall_trsm with side=right, fill=lower, diag=unit
     Overwrites A(:, j:j+jb) with the solution

**Stage 3: Apply column pivoting**
- Calls getri_kernel_large2 which invokes getri_pivot
- Applies the pivots from ipiv in reverse order (j = n-2 down to 0)
- For each j, if ipiv[j]-1 != j, swaps columns j and ipiv[j]-1
- This undoes the row permutation P from GETRF, converting row permutations to column permutations for the inverse

**Complexity**: O(2n³/3) flops, same as GETRF.

**In-place operation**: The result overwrites the input matrix A.""",
    """This answer provides the complete two-stage algorithm structure with the backward column sweep clearly explained. The TRTRI stage for computing inv(U), the blocked TRSM/GEMM iterations for solving the triangular system, and the pivot application are all detailed.""",
    ["algorithm", "matrix_inversion", "LU_factorization", "LAPACK"]
))

# Entry 2: GETRI input requirements (L1)
entries.append(create_entry(
    "roclapack_getri_002",
    "L1",
    "conceptual",
    "What must be provided as input to GETRI, and what does it output?",
    """GETRI computes the inverse of a matrix using its LU factorization.

Source: library/src/lapack/roclapack_getri.cpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.cpp",
            "start_line": 108,
            "end_line": 116,
            "language": "cpp"
        }
    ],
    """**Input requirements:**

1. **A** (n×n matrix): Must contain the LU factorization computed by GETRF
   - Upper triangle: contains U factor
   - Strictly lower triangle: contains L factor (with implicit unit diagonal)
   - This is the output of rocsolver_sgetrf (or dgetrf, cgetri, zgetri)

2. **ipiv** (integer array of length n): Pivot indices from GETRF
   - Records the row interchanges performed during factorization
   - Uses 1-based Fortran indexing
   - Required to undo the permutation P in A = P*L*U

3. **info** (integer output): Singularity indicator
   - Set to 0 if successful
   - Set to i > 0 if U(i,i) is exactly zero (matrix is singular, inverse cannot be computed)

**Output:**

- **A** is overwritten with the inverse matrix A^(-1)
- If info > 0, the contents of A are undefined (inverse does not exist)

**Typical usage pattern:**
```cpp
// Step 1: Compute LU factorization
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, info);
if (info != 0) { /* handle singularity */ }

// Step 2: Compute inverse
rocsolver_sgetri(handle, n, A, lda, ipiv, info);
if (info != 0) { /* handle singularity in inversion */ }
// Now A contains A^(-1)
```

**Important**: GETRI requires the exact output of GETRF. You cannot modify A or ipiv between the two calls.""",
    """This answer clearly specifies the input format (LU factors + pivots), the output (inverse matrix in-place), and the proper usage sequence. The info parameter interpretation is explained.""",
    ["API", "input_output", "usage", "LAPACK"]
))

# Entry 3: GETRI vs TRTRI (L2)
entries.append(create_entry(
    "roclapack_getri_003",
    "L2",
    "conceptual",
    "What is the relationship between GETRI and TRTRI? Why does GETRI call TRTRI?",
    """GETRI uses TRTRI as a subroutine to compute the matrix inverse.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 340,
            "end_line": 343,
            "language": "cpp"
        }
    ],
    """**TRTRI (Triangular Matrix Inversion):**
- Computes the inverse of a triangular matrix: T^(-1)
- Works for upper or lower triangular matrices
- Supports unit diagonal (diag=unit) or non-unit diagonal (diag=non_unit)
- Operates in-place, replacing T with T^(-1)

**GETRI's use of TRTRI:**

Stage 1 of GETRI calls TRTRI to invert the upper triangular factor U:
```cpp
rocsolver_trtri_template<BATCHED, STRIDED, T>(
    handle, rocblas_fill_upper, rocblas_diagonal_non_unit, n, A, ...
);
```

This computes inv(U) from the U factor stored in the upper triangle of A.

**Why this works:**

Given A = P*L*U (from GETRF), we want A^(-1):
```
A^(-1) = (P*L*U)^(-1) = U^(-1) * L^(-1) * P^(-1)
```

Strategy:
1. First compute inv(U) using TRTRI
2. Then solve inv(A) * L = inv(U) for inv(A)
   - This multiplies inv(U) by L^(-1) on the right
   - Implemented using blocked TRSM
3. Finally apply P^(-1) by swapping columns according to ipiv

**Efficiency benefit:**
- TRTRI is optimized for triangular matrices (blocked algorithm with TRMM and TRSM)
- Reusing TRTRI avoids code duplication and ensures optimal performance for the triangular inversion step

**Singularity check:**
TRTRI also detects if U has any zero diagonal elements. If U(i,i) = 0, TRTRI sets info = i and returns early. This prevents division by zero and correctly reports that the matrix is singular.""",
    """This answer explains the mathematical relationship (inv(U) is needed to compute inv(A)), the implementation strategy (two-stage algorithm), and the practical benefits (code reuse and singularity detection).""",
    ["algorithm", "matrix_inversion", "triangular_matrices", "LAPACK"]
))

# Entry 4: copy_and_zero device function (L2)
entries.append(create_entry(
    "roclapack_getri_004",
    "L2",
    "code",
    "Explain the copy_and_zero device function in GETRI: what does it do and why is it necessary?",
    """GETRI uses a copy_and_zero device function during the blocked backward sweep.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 43,
            "end_line": 64,
            "language": "cpp"
        },
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 380,
            "end_line": 382,
            "language": "cpp"
        }
    ],
    """The copy_and_zero function performs two operations on the current block of the L factor:

**What it does:**
```cpp
void copy_and_zero(m, n, a, lda, w, ldw) {
    for each element (i,j) where i > j (strictly lower triangle):
        w[i,j] = a[i,j]    // Copy to workspace
        a[i,j] = 0         // Zero in original matrix
}
```

**Why it's necessary:**

During the backward sweep in stage 2, we solve inv(A) * L = inv(U):
1. After TRTRI, A contains inv(U) in the upper triangle and L in the lower triangle
2. To apply TRSM correctly, we need L separately in workspace
3. We must zero the lower triangle of A because:
   - A will accumulate the columns of inv(A) during the sweep
   - Having non-zero values in the lower triangle would interfere with the computation
   - GEMM and TRSM assume certain zero structure

**Example for block j:j+jb:**
```
Before copy_and_zero:
A = [inv(U)  *  ]  (upper triangle has inv(U))
    [  L     *  ]  (lower triangle has L)

After copy_and_zero:
A = [inv(U)  *  ]  (upper unchanged)
    [  0     *  ]  (lower zeroed)

workspace = [  0     *  ]  (upper ignored)
            [  L     *  ]  (lower has copy of L)
```

**Parallel implementation:**
- Threads iterate over all m×n elements with stride hipBlockDim_y
- Each thread processes element k = i + j*m
- Only elements with i > j are copied/zeroed
- Uses __syncthreads() to ensure all threads complete before GEMM/TRSM

**Alternative (zero_work):**
If info != 0 (singular matrix), zero_work is called instead to zero the workspace. This ensures GEMM/TRSM don't propagate undefined values.""",
    """This answer explains both the operation (copy to workspace and zero in-place) and the mathematical necessity (separating L from the accumulating inv(A)). The parallel implementation details are included.""",
    ["kernels", "GPU_programming", "algorithm_implementation", "device_functions"]
))

# Entry 5: Pivot application in GETRI (L2)
entries.append(create_entry(
    "roclapack_getri_005",
    "L2",
    "code",
    "How does GETRI apply pivots, and why is the order reversed compared to GETRF?",
    """GETRI applies the pivots from GETRF in reverse order to compute the correct inverse.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 80,
            "end_line": 96,
            "language": "cpp"
        }
    ],
    """**Pivot application algorithm (getri_pivot):**

```cpp
void getri_pivot(n, a, lda, p) {
    for (j = n-2; j >= 0; --j) {  // Reverse order!
        jp = p[j] - 1;  // Convert from 1-based to 0-based
        if (jp != j) {
            swap columns j and jp of matrix a
        }
    }
}
```

**Why reverse order?**

GETRF computes: A = P*L*U where P is a permutation matrix represented by ipiv.

To compute A^(-1), we need: A^(-1) = U^(-1) * L^(-1) * P^(-1)

After solving inv(A) * L = inv(U), we have U^(-1) * L^(-1) and need to apply P^(-1).

**Key insight**: 
- GETRF applies row permutations in forward order (j = 0, 1, ..., n-1)
- For the inverse, we need column permutations (transposes of row permutations)
- Row interchange on P corresponds to column interchange on P^(-1)
- Permutations must be undone in reverse order: P^(-1) = P_{n-1}^(-1) * ... * P_1^(-1) * P_0^(-1)

**Example:**
If GETRF applied pivots: swap rows 0↔2, then rows 1↔1 (no swap), then rows 2↔3
```
ipiv = [3, 2, 4]  (1-based indices)
```

GETRI applies in reverse:
1. j=2: swap columns 2↔3 (since ipiv[2]=4, jp=3)
2. j=1: no swap (since ipiv[1]=2, jp=1=j)
3. j=0: swap columns 0↔2 (since ipiv[0]=3, jp=2)

This correctly undoes the row permutation P by applying the column permutation P^(-1).

**Implementation note:**
- Loop starts at j=n-2 (not n-1) because the last pivot is always trivial
- Uses column swaps (swap a[i,j] ↔ a[i,jp]) instead of row swaps
- Threads are parallelized over rows i with __syncthreads() between swaps""",
    """This answer explains the reverse order with mathematical justification (undoing permutations requires reverse application), provides an example, and details the implementation including why the loop starts at n-2.""",
    ["pivoting", "permutations", "algorithm", "mathematical_correctness"]
))

# Entry 6: GETRI workspace and memory (L2)
entries.append(create_entry(
    "roclapack_getri_006",
    "L2",
    "code",
    "What workspace does GETRI allocate, and how does memory usage depend on matrix size?",
    """GETRI allocates multiple workspace arrays for TRTRI, TRSM, and temporary copies.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 168,
            "end_line": 258,
            "language": "cpp"
        }
    ],
    """GETRI allocates 6 workspace arrays:

**1-4. work1, work2, work3, work4:**
- Shared workspace for TRSM (during stage 2) and TRTRI (during stage 1)
- Sizes determined by max(TRSM requirements, TRTRI requirements)
- TRSM requirements: rocblasCall_trsm_mem for n × blk blocks
- TRTRI requirements: includes workspace for TRMV and diagonal scalars

**5. tmpcopy:**
- Temporary array for copying L blocks during stage 2
- Size: n × blk × sizeof(T) × batch_count
- For TRTRI, size is n × n × sizeof(T) × batch_count (larger)
- Used by copy_and_zero to store L factors during backward sweep

**6. workArr** (batched cases only):
- Array of pointers: T** of length batch_count
- Not needed for single-matrix case

**Memory scaling:**

For tiny matrices (n ≤ GETRI_TINY_SIZE):
- No workspace needed, uses specialized kernel getri_run_small
- GETRI_TINY_SIZE is typically 32 for non-batched, 64 for batched

For small matrices (n ≤ TRTRI_MAX_COLS):
- Only TRTRI workspace needed (stage 2 uses getri_run_small)
- Memory: O(n²) for tmpcopy + O(n) for work arrays

For large matrices:
- Full workspace for blocked TRSM and TRTRI
- Dominant term: tmpcopy = O(n² × batch_count)
- work1-4: O(n × blk) where blk is block size (typically 64-128)

**Block size selection:**
```cpp
blk = getri_get_blksize<ISBATCHED>(n)
```
Uses lookup tables GETRI_BLKSIZES and GETRI_INTERVALS, similar to GETRF.

**Memory optimization:**
The optim_mem flag indicates whether optimal memory for TRSM is allocated (always true for GETRI, ensuring best performance).""",
    """This answer enumerates all workspace arrays, explains their purposes, and describes memory scaling for different matrix sizes. The distinction between tiny, small, and large matrices is clearly stated.""",
    ["memory", "workspace", "performance", "resource_management"]
))

# Entry 7: C API variants (L1)
entries.append(create_entry(
    "roclapack_getri_007",
    "L1",
    "code",
    "What C API functions does rocSOLVER provide for GETRI?",
    """rocSOLVER provides C API wrappers for all GETRI precision and pivot variants.

Source: library/src/lapack/roclapack_getri.cpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.cpp",
            "start_line": 108,
            "end_line": 191,
            "language": "cpp"
        }
    ],
    """GETRI C API provides 8 functions (4 precisions × 2 variants):

**With pivoting (standard):**
1. `rocsolver_sgetri(handle, n, A, lda, ipiv, info)` - float
2. `rocsolver_dgetri(handle, n, A, lda, ipiv, info)` - double
3. `rocsolver_cgetri(handle, n, A, lda, ipiv, info)` - complex<float>
4. `rocsolver_zgetri(handle, n, A, lda, ipiv, info)` - complex<double>

Parameters:
- handle: rocBLAS handle
- n: matrix dimension
- A: n×n matrix containing LU factors from GETRF (overwritten with inverse)
- lda: leading dimension (lda ≥ n)
- ipiv: pivot indices from GETRF (length n)
- info: singularity indicator (0 = success, i > 0 = U(i,i) is zero)

**Without pivoting (_npvt):**
1. `rocsolver_sgetri_npvt(handle, n, A, lda, info)` - float
2. `rocsolver_dgetri_npvt(handle, n, A, lda, info)` - double
3. `rocsolver_cgetri_npvt(handle, n, A, lda, info)` - complex<float>
4. `rocsolver_zgetri_npvt(handle, n, A, lda, info)` - complex<double>

Parameters: Same as above but no ipiv (set to nullptr internally)

**Implementation:**
All functions call the same template:
```cpp
rocsolver::rocsolver_getri_impl<T>(handle, n, A, lda, ipiv, info, pivot_flag)
```
where pivot_flag = true for standard, false for _npvt.

**When to use _npvt:**
- Input matrix A contains LU factorization from GETRF_NPVT (no pivoting)
- Faster (no pivot application), but less numerically stable
- Only for well-conditioned matrices

**Note**: Unlike GETRF, GETRI does not have _64 variants in the current API.""",
    """This answer lists all 8 C API functions with signatures, explains parameters, and clarifies the difference between standard and _npvt variants. The implementation pattern is noted.""",
    ["API", "interface", "precisions", "usage"]
))

# Entry 8: Singularity handling (L1)
entries.append(create_entry(
    "roclapack_getri_008",
    "L1",
    "conceptual",
    "What happens if the input matrix to GETRI is singular?",
    """GETRI must handle singular matrices gracefully.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 340,
            "end_line": 343,
            "language": "cpp"
        }
    ],
    """**Singularity detection:**

GETRI detects singularity during the TRTRI call (stage 1):
- TRTRI checks each diagonal element U(i,i) for exact zero
- If U(i,i) = 0, the matrix is singular and cannot be inverted
- TRTRI sets info = i (1-based index) and returns

**What happens after singularity is detected:**

1. **TRTRI returns early**: The upper triangle is partially modified but not fully inverted

2. **Stage 2 still executes**: The blocked backward sweep continues, but uses zero_work instead of copy_and_zero
   - zero_work zeros the workspace to avoid undefined behavior
   - GEMM and TRSM operations continue but operate on zeros
   - This prevents crashes or NaN propagation

3. **Pivot application skipped**: The getri_kernel_large2 checks `if(info[b] == 0)` before calling getri_pivot
   - If singular, no column swaps are performed

4. **Output**: 
   - info = i where i is the index of the first zero diagonal element
   - A contains undefined values (not a valid inverse)
   - User must check info != 0 before using the result

**Example:**
```cpp
rocsolver_sgetri(handle, n, A, lda, ipiv, info);
if (info != 0) {
    printf("Matrix is singular: U(%d,%d) is zero\\n", info, info);
    // Cannot use A, it doesn't contain a valid inverse
}
```

**Numerical note:**
Like GETRF, the check is for exact zero. Near-zero diagonal elements will not trigger info > 0, but may cause large errors in the computed inverse due to numerical instability.""",
    """This answer explains when singularity is detected (TRTRI stage), what happens during stage 2 (zero workspace to prevent undefined behavior), and how the user should handle info != 0. The practical usage example is included.""",
    ["error_handling", "singularity", "numerical_behavior", "robustness"]
))

# Entry 9: Block size tuning (L2)
entries.append(create_entry(
    "roclapack_getri_009",
    "L2",
    "conceptual",
    "How does GETRI select block sizes, and why does it use different sizes for batched vs non-batched cases?",
    """GETRI uses tuned block sizes for optimal performance on GPUs.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 145,
            "end_line": 166,
            "language": "cpp"
        }
    ],
    """**Block size selection:**

```cpp
rocblas_int getri_get_blksize<ISBATCHED>(dim) {
    if (ISBATCHED) {
        size[] = {GETRI_BATCH_BLKSIZES};
        intervals[] = {GETRI_BATCH_INTERVALS};
    } else {
        size[] = {GETRI_BLKSIZES};
        intervals[] = {GETRI_INTERVALS};
    }
    return size[get_index(intervals, max, dim)];
}
```

Uses 1D lookup table indexed by matrix dimension, similar to GETRF.

**Why different sizes for batched vs non-batched?**

**Non-batched (single matrix):**
- Can use larger block sizes (e.g., 64-128)
- More threads available for each matrix
- GEMM and TRSM have better occupancy with larger blocks
- Memory bandwidth is less constrained

**Batched (multiple matrices):**
- Uses smaller block sizes (e.g., 32-64)
- Many small matrices compete for GPU resources
- Smaller blocks reduce workspace per matrix (tmpcopy = n × blk)
- Better load balancing: more blocks across batch
- TRSM and GEMM calls operate on smaller work items, improving parallelism across batch

**Performance impact:**

Stage 2 backward sweep:
```
for j = nn-1, nn-1-blk, ..., 0:
    GEMM: n × jb × (n-j-jb)
    TRSM: n × jb
```

Larger blk:
- Fewer iterations (less kernel launch overhead)
- Better GEMM/TRSM efficiency (BLAS-3 performance scales with work size)
- More memory for tmpcopy

Smaller blk:
- More iterations (more overhead)
- Less memory per matrix (important for batch)
- Better GPU occupancy when batch_count is large

**Special values:**
- blk = 0: Use rocBLAS trtri directly (no blocking)
- blk = 1: Use unblocked algorithm (very small matrices)

**Typical intervals (non-batched):**
- n ≤ 64: blk = 1 (unblocked)
- 64 < n ≤ 256: blk = 32
- 256 < n ≤ 512: blk = 64
- n > 512: blk = 128""",
    """This answer explains the block size selection mechanism, compares batched vs non-batched tradeoffs, and describes performance impacts. The typical interval values provide concrete examples.""",
    ["performance", "blocking", "tuning", "batched_operations"]
))

# Entry 10: GETRI vs direct methods (L2)
entries.append(create_entry(
    "roclapack_getri_010",
    "L2",
    "conceptual",
    "When should I use GETRI to compute a matrix inverse vs. solving systems directly with GETRS? What are the performance and numerical tradeoffs?",
    """GETRI computes the explicit inverse A^(-1), while GETRS solves Ax=b using LU factors.

Source: library/src/lapack/roclapack_getri.hpp""",
    [],
    """**GETRI (explicit inversion):**

Pros:
- Once computed, A^(-1) can be reused for multiple right-hand sides
- Useful when inverse matrix has mathematical meaning (e.g., covariance matrix, adjacency matrix)
- Convenient for computing A^(-1) * B when B is a matrix

Cons:
- Expensive: O(2n³/3) flops (same as GETRF)
- Numerically unstable: errors in LU factorization are amplified
- Requires O(n²) storage for the inverse
- Total cost for solving Ax=b: GETRF O(2n³/3) + GETRI O(2n³/3) + matvec O(n²) = O(4n³/3)

**GETRS (solve using LU factors):**

Pros:
- Efficient: Only O(n²) flops per right-hand side
- More numerically stable: avoids explicit inverse
- Less storage: only LU factors needed (same O(n²) as original matrix)
- Total cost for solving Ax=b: GETRF O(2n³/3) + GETRS O(n²)

Cons:
- Requires keeping LU factors and ipiv in memory
- Less intuitive than explicit inverse

**Performance comparison:**

For k right-hand sides:
- GETRI approach: O(2n³/3) + O(2n³/3) + k×O(n²) = O(4n³/3 + kn²)
- GETRS approach: O(2n³/3) + k×O(n²) = O(2n³/3 + kn²)

GETRI becomes competitive when k > n (many right-hand sides), but numerical stability still favors GETRS.

**Numerical stability:**

Explicit inverse amplifies errors:
- ||A^(-1) - computed_inv|| / ||A^(-1)|| ≈ cond(A) × machine_epsilon
- Solving with GETRS: ||x - computed_x|| / ||x|| ≈ cond(A) × machine_epsilon

For ill-conditioned matrices (large cond(A)), GETRI produces much larger relative errors in subsequent computations.

**Recommendation:**

✓ Use GETRS when:
- Solving Ax = b (single or few right-hand sides)
- Numerical accuracy is important
- Matrix may be ill-conditioned

✓ Use GETRI when:
- Need explicit inverse for mathematical reasons
- Computing A^(-1) * B where B is dense and large
- Many right-hand sides (k >> n) and matrix is well-conditioned

General rule: "Don't invert matrices unless you really need to." Solve systems instead.""",
    """This answer provides a comprehensive comparison covering performance (flop counts), numerical stability (error amplification), and use cases. The recommendation section gives clear guidance on when each approach is appropriate.""",
    ["algorithm_selection", "numerical_stability", "performance", "best_practices"]
))

# Entry 11: Tiny and small size optimizations (L2)
entries.append(create_entry(
    "roclapack_getri_011",
    "L2",
    "code",
    "What optimizations does GETRI use for small matrices, and what are the size thresholds?",
    """GETRI has specialized kernels for tiny and small matrices to avoid blocking overhead.

Source: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 332,
            "end_line": 355,
            "language": "cpp"
        }
    ],
    """GETRI uses three size categories with different implementations:

**1. Tiny matrices: n ≤ GETRI_TINY_SIZE**

Threshold:
- Non-batched: GETRI_TINY_SIZE (typically 32)
- Batched: GETRI_BATCH_TINY_SIZE (typically 64)

Optimization:
- Single-kernel approach: getri_run_small with from_stage1=true
- Computes entire inverse in one kernel (both stages combined)
- No blocking, no BLAS calls
- No workspace allocation needed
- Optimal for very small matrices where kernel launch overhead dominates

**2. Small matrices: GETRI_TINY_SIZE < n ≤ TRTRI_MAX_COLS**

Threshold: TRTRI_MAX_COLS (typically 64-128)

Stage 1: Normal TRTRI (may still use trtri_run_small if n ≤ TRTRI_MAX_COLS)

Stage 2: 
- Uses getri_run_small with from_stage1=false
- Single kernel for solving inv(A) * L = inv(U) and applying pivots
- Avoids blocked GEMM/TRSM overhead
- Requires workspace for TRTRI but not for stage 2

**3. Large matrices: n > TRTRI_MAX_COLS**

Uses full blocked algorithm:
- TRTRI with blocking for stage 1
- Blocked backward sweep with GEMM/TRSM for stage 2
- Block size selected by getri_get_blksize
- Maximum BLAS-3 efficiency but more overhead

**Why these thresholds?**

Tiny threshold:
- Below this size, single-warp or few-warp kernels are faster than BLAS calls
- Kernel launch overhead >> computation time
- Memory allocation overhead is significant

Small threshold (TRTRI_MAX_COLS):
- Specialized kernels use shared memory for the entire matrix
- Above this, shared memory limits are exceeded
- BLAS-3 operations become more efficient than specialized kernels

**Performance impact:**

Example on MI250X (approximate):
- n=16: getri_run_small 10× faster than blocked algorithm
- n=32: getri_run_small 5× faster
- n=64: getri_run_small 2× faster
- n=128: blocked algorithm becomes competitive
- n=256+: blocked algorithm clearly faster

**Code structure:**
```cpp
if (n <= TINY_SIZE)
    return getri_run_small(..., true, pivot);  // Both stages
else {
    trtri(...);  // Stage 1
    if (n <= TRTRI_MAX_COLS)
        return getri_run_small(..., false, pivot);  // Stage 2 only
    else
        /* blocked stage 2 */
}
```""",
    """This answer explains the three size categories, the thresholds, the optimizations for each, and the rationale behind the thresholds. Performance impact is quantified with approximate speedups.""",
    ["optimization", "performance", "GPU_programming", "specialized_kernels"]
))

# Entry 12: Simplified GETRI implementation (L3 - Coding task)
entries.append(create_entry(
    "roclapack_getri_012",
    "L3",
    "code",
    "Write pseudocode for a simplified GETRI implementation showing the two main stages and the backward sweep structure.",
    """Implement the core GETRI algorithm in pseudocode, focusing on the two-stage structure.

Reference: library/src/lapack/roclapack_getri.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getri.hpp",
            "start_line": 340,
            "end_line": 400,
            "language": "cpp"
        }
    ],
    """```python
def getri(A, n, ipiv, block_size):
    \"\"\"
    Compute matrix inverse using LU factorization: A^(-1)
    
    Args:
        A: n×n matrix containing LU factors from GETRF
           - Upper triangle: U factor
           - Strictly lower triangle: L factor (unit diagonal)
        ipiv: pivot indices from GETRF (1-based, length n)
        block_size: blocking parameter (e.g., 64)
    
    Returns:
        info: 0 if successful, i>0 if U(i,i) is zero (singular)
    
    Output:
        A is overwritten with A^(-1)
    \"\"\"
    
    # Stage 1: Compute inv(U) in upper triangle
    # Uses triangular matrix inversion (TRTRI)
    info = trtri_upper_nonunit(A, n)
    if info > 0:
        return info  # Singular matrix, cannot invert
    
    # Now A contains:
    #   Upper triangle: inv(U)
    #   Lower triangle: L
    
    # Stage 2: Solve inv(A) * L = inv(U) for inv(A)
    # Use blocked backward column sweep
    
    # Starting point for backward sweep
    nn = ((n-1) // block_size) * block_size + 1
    
    for j in range(nn-1, -1, -block_size):
        jb = min(n - j, block_size)  # Current block width
        
        # Step 1: Copy L block to workspace and zero in A
        work = zeros(n, n)
        for i in range(j, n):
            for k in range(j, j+jb):
                if i > k:  # Strictly lower triangle
                    work[i, k] = A[i, k]
                    A[i, k] = 0
        # L block is now in work[j:n, j:j+jb]
        # A[j:n, j:j+jb] lower part is zeroed
        
        # Step 2: Update current block columns with already-computed columns
        if j + jb < n:
            # A[:, j:j+jb] -= A[:, j+jb:n] * work[j+jb:n, j:j+jb]
            # This is a GEMM: C = C - A*B
            gemm(A[:, j+jb:n], work[j+jb:n, j:j+jb], 
                 A[:, j:j+jb], alpha=-1.0, beta=1.0)
        
        # Step 3: Solve triangular system
        # A[:, j:j+jb] * L[j:j+jb, j:j+jb] = A[:, j:j+jb]
        # Solve for A[:, j:j+jb] (overwrites with solution)
        trsm_right_lower_unit(work[j:j+jb, j:j+jb], A[:, j:j+jb])
    
    # Stage 3: Apply column pivots to undo row permutation P
    # Apply in reverse order: P^(-1) = P_{n-1}^(-1) * ... * P_0^(-1)
    for j in range(n-2, -1, -1):
        jp = ipiv[j] - 1  # Convert 1-based to 0-based
        if jp != j:
            # Swap columns j and jp
            swap(A[:, j], A[:, jp])
    
    return 0  # Success


def trtri_upper_nonunit(A, n):
    \"\"\"
    Compute inverse of upper triangular matrix with non-unit diagonal.
    Returns info = i if A[i-1,i-1] is zero (1-based).
    \"\"\"
    for i in range(n):
        if A[i, i] == 0:
            return i + 1  # Singular
        A[i, i] = 1.0 / A[i, i]
    
    # Compute off-diagonal elements (simplified unblocked version)
    for j in range(n):
        for i in range(j-1, -1, -1):
            A[i, j] = -A[i, i] * sum(A[i, k] * A[k, j] for k in range(i+1, j+1))
    
    return 0


def trsm_right_lower_unit(L, B):
    \"\"\"
    Solve B * L = B for B, where L is lower triangular with unit diagonal.
    Overwrites B with the solution.
    \"\"\"
    m, n = B.shape
    for j in range(n):
        for k in range(j+1, n):
            # B[:, k] -= B[:, j] * L[k, j]
            B[:, k] -= B[:, j] * L[k, j]
```

**Key features:**
1. Two-stage structure: TRTRI then blocked backward sweep
2. Backward column iteration with GEMM/TRSM
3. Copy-and-zero to separate L from accumulating inv(A)
4. Reverse-order pivot application (column swaps)
5. Singularity detection in TRTRI""",
    """This pseudocode provides a complete simplified implementation showing both stages clearly. The backward sweep, copy-and-zero mechanism, and pivot application are all demonstrated. Helper functions for TRTRI and TRSM show the basic algorithms.""",
    ["implementation", "algorithm", "pseudocode", "matrix_inversion"]
))

# Write JSONL output
output_file = "../roclapack_getri.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries for roclapack_getri")
print(f"Output: {output_file}")

# Statistics
l1_count = sum(1 for e in entries if e['level'] == 'L1')
l2_count = sum(1 for e in entries if e['level'] == 'L2')
l3_count = sum(1 for e in entries if e['level'] == 'L3')
code_count = sum(1 for e in entries if e['interface'] == 'code')

print(f"Level distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
print(f"Coding tasks: {code_count} ({100*code_count/len(entries):.1f}%)")
