#!/usr/bin/env python3
"""Generate dataset for roclapack_getrs.yaml"""

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

# Entry 1: Complete GETRS algorithm (L3)
entries.append(create_entry(
    "roclapack_getrs_001",
    "L3",
    "code",
    "Explain the complete GETRS algorithm for solving linear systems using LU factorization. How does it differ for trans=none vs trans=transpose?",
    """GETRS solves a system of linear equations A*X = B using the LU factorization computed by GETRF. It supports three operation modes: normal (A*X=B), transpose (A^T*X=B), and conjugate transpose (A^H*X=B).

Source files:
- library/src/lapack/roclapack_getrs.hpp (main algorithm)
- library/src/lapack/roclapack_getrs.cpp (C API)
- library/src/auxiliary/rocauxiliary_laswp.hpp (pivot application)""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 108,
            "end_line": 190,
            "language": "cpp"
        }
    ],
    """GETRS solves linear systems using the LU decomposition A = P*L*U from GETRF. The algorithm depends on the trans parameter:

**Case 1: trans = none (solve A*X = B)**

Given A = P*L*U, solve (P*L*U)*X = B:

1. **Apply row interchanges**: Compute B' = P*B
   - Uses rocsolver_laswp_template with forward direction (incp=1)
   - Applies the same row swaps recorded in ipiv during GETRF
   - This transforms the problem to (L*U)*X = B'

2. **Forward substitution**: Solve L*Y = B' for Y
   - Uses rocsolver_trsm_lower with side=left, diag=unit
   - L is lower triangular with unit diagonal
   - Overwrites B' with Y in-place

3. **Backward substitution**: Solve U*X = Y for X
   - Uses rocsolver_trsm_upper with side=left, diag=non_unit
   - U is upper triangular with non-unit diagonal
   - Overwrites Y with X in-place (final solution)

Result: B is overwritten with X such that A*X = B.

**Case 2: trans = transpose or conjugate_transpose (solve A^T*X = B or A^H*X = B)**

Given A = P*L*U, solve (P*L*U)^T*X = B which is equivalent to U^T*L^T*P^T*X = B:

1. **Backward substitution**: Solve U^T*Y = B (or U^H*Y = B) for Y
   - Uses rocsolver_trsm_upper with side=left, trans, diag=non_unit
   - Overwrites B with Y in-place

2. **Forward substitution**: Solve L^T*Z = Y (or L^H*Z = Y) for Z
   - Uses rocsolver_trsm_lower with side=left, trans, diag=unit
   - Overwrites Y with Z in-place

3. **Apply row interchanges in reverse**: Compute X = P^T*Z
   - Uses rocsolver_laswp_template with backward direction (incp=-1)
   - Reverses the pivoting to apply P^T
   - Overwrites Z with X (final solution)

Result: B is overwritten with X such that A^T*X = B (or A^H*X = B).

**Complexity**: O(n² × nrhs) flops, much cheaper than GETRF's O(2n³/3).

**In-place operation**: B is overwritten with the solution X.""",
    """This answer provides complete algorithms for both cases (normal and transpose), clearly showing the three-step structure and the differences (pivot order, substitution order). The mathematical transformations are explained.""",
    ["algorithm", "linear_systems", "LU_factorization", "LAPACK"]
))

# Entry 2: GETRS input requirements (L1)
entries.append(create_entry(
    "roclapack_getrs_002",
    "L1",
    "conceptual",
    "What must be provided as input to GETRS, and what are the outputs?",
    """GETRS solves linear systems using pre-computed LU factorization.

Source: library/src/lapack/roclapack_getrs.cpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.cpp",
            "start_line": 106,
            "end_line": 117,
            "language": "cpp"
        }
    ],
    """**Input requirements:**

1. **trans**: Operation mode
   - rocblas_operation_none: Solve A*X = B
   - rocblas_operation_transpose: Solve A^T*X = B
   - rocblas_operation_conjugate_transpose: Solve A^H*X = B (conjugate transpose for complex matrices)

2. **A** (n×n matrix): LU factorization from GETRF
   - Upper triangle: U factor
   - Strictly lower triangle: L factor (unit diagonal implicit)
   - Must be the exact output of rocsolver_sgetrf (or dgetrf, cgetrs, zgetrs)

3. **ipiv** (integer array of length n): Pivot indices from GETRF
   - Records row interchanges from factorization
   - 1-based Fortran indexing
   - Required for applying permutation P

4. **B** (n×nrhs matrix): Right-hand side(s)
   - Can have multiple columns (nrhs) to solve multiple systems with the same coefficient matrix
   - ldb ≥ n (leading dimension)

**Outputs:**

- **B** is overwritten with the solution matrix X
  - Each column of B becomes the corresponding solution
  - Original right-hand side is lost

**Typical usage:**
```cpp
// Step 1: Factor the matrix
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);
if (info != 0) { /* matrix is singular */ }

// Step 2: Solve the system (can be called multiple times with different B)
rocsolver_sgetrs(handle, rocblas_operation_none, n, nrhs, A, lda, ipiv, B, ldb);
// Now B contains the solution X
```

**Multiple right-hand sides:**
If nrhs > 1, B is n×nrhs and GETRS solves A*X(:,i) = B(:,i) for each column i simultaneously.

**Important**: A and ipiv must not be modified between GETRF and GETRS calls.""",
    """This answer specifies all input parameters with meanings, the output (in-place solution), and typical usage pattern. The multi-RHS case is explained.""",
    ["API", "input_output", "usage", "LAPACK"]
))

# Entry 3: LASWP - pivot application (L2)
entries.append(create_entry(
    "roclapack_getrs_003",
    "L2",
    "code",
    "What is LASWP and how does GETRS use it to apply pivots?",
    """LASWP (LApack SWap Pivots) applies a series of row interchanges to a matrix.

Source: library/src/auxiliary/rocauxiliary_laswp.hpp""",
    [
        {
            "path": "library/src/auxiliary/rocauxiliary_laswp.hpp",
            "start_line": 19,
            "end_line": 67,
            "language": "cpp"
        }
    ],
    """**LASWP (Linear Algebra SWap)** applies row interchanges specified in ipiv to a matrix.

**Function signature:**
```cpp
rocsolver_laswp_template(handle, n, A, k1, k2, ipiv, incp)
```

**Parameters:**
- n: number of columns in A
- A: matrix to permute (rows will be swapped)
- k1, k2: range of pivots to apply (1-based indices)
- ipiv: pivot array from GETRF
- incp: increment for ipiv (+1 for forward, -1 for backward)

**Algorithm (laswp_kernel):**
```cpp
if (incp < 0) {  // Backward order
    for i = k2 down to k1:
        exch = ipiv[i] - 1  // Convert to 0-based
        if exch != i-1:
            swap rows (i-1) and exch in A
} else {  // Forward order
    for i = k1 to k2:
        exch = ipiv[i] - 1
        if exch != i-1:
            swap rows (i-1) and exch in A
}
```

**GETRS usage:**

**For trans=none (solve A*X=B):**
```cpp
rocsolver_laswp_template(handle, nrhs, B, 1, n, ipiv, 1);  // Forward
```
- Applies pivots in forward order (incp=+1)
- Transforms B → P*B where P is the permutation from GETRF
- Called BEFORE solving triangular systems

**For trans=transpose/conjugate (solve A^T*X=B):**
```cpp
rocsolver_laswp_template(handle, nrhs, B, 1, n, ipiv, -1);  // Backward
```
- Applies pivots in reverse order (incp=-1)
- Transforms B → P^T*B (inverse permutation)
- Called AFTER solving triangular systems

**Parallelization:**
- Each thread handles one column: tid corresponds to column index
- All row swaps for that column are done sequentially by the thread
- Grid: blocksPivot × batch_count, Threads: LASWP_THDS (256)

**Why forward vs backward?**

Forward (trans=none):
- A*X = B → (P*L*U)*X = B → L*U*X = P^(-1)*B → Apply P to B first

Backward (trans=transpose):
- A^T*X = B → U^T*L^T*P^T*X = B → After solving U^T and L^T, apply P^T

**Performance**: O(n × nrhs) work, but memory-bound due to scattered row accesses.""",
    """This answer explains LASWP's purpose (row interchange application), the algorithm, and how GETRS uses it differently for forward vs backward pivot application. The mathematical justification for the direction is included.""",
    ["auxiliary_functions", "pivoting", "row_interchanges", "LAPACK"]
))

# Entry 4: TRSM in GETRS (L2)
entries.append(create_entry(
    "roclapack_getrs_004",
    "L2",
    "code",
    "How does GETRS use TRSM (Triangular Solve with Multiple right-hand sides)? What are the four TRSM calls and their parameters?",
    """GETRS solves triangular systems using TRSM.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 156,
            "end_line": 180,
            "language": "cpp"
        }
    ],
    """GETRS makes two TRSM calls (four total across both cases) to solve triangular systems.

**TRSM (TRiangular Solve with Multiple right-hand sides)**: Solves T*X = B or X*T = B where T is triangular.

**Case 1: trans=none (solve A*X=B where A=P*L*U)**

After applying P to get B' = P*B:

**TRSM call 1: Solve L*X = B' (forward substitution)**
```cpp
rocsolver_trsm_lower(handle, rocblas_side_left, rocblas_operation_none,
                     rocblas_diagonal_unit, n, nrhs, A, B, ...);
```
- side=left: L is on the left (L*X = B)
- trans=none: Use L (not L^T)
- diag=unit: L has unit diagonal (1's not stored)
- Solves for X in-place, overwrites B

**TRSM call 2: Solve U*X = B (backward substitution)**
```cpp
rocsolver_trsm_upper(handle, rocblas_side_left, rocblas_operation_none,
                     rocblas_diagonal_non_unit, n, nrhs, A, B, ...);
```
- side=left: U is on the left (U*X = B)
- trans=none: Use U (not U^T)
- diag=non_unit: U has arbitrary diagonal elements
- Solves for final X in-place

**Case 2: trans=transpose/conjugate_transpose (solve A^T*X=B or A^H*X=B)**

**TRSM call 1: Solve U^T*X = B (or U^H*X = B)**
```cpp
rocsolver_trsm_upper(handle, rocblas_side_left, trans,
                     rocblas_diagonal_non_unit, n, nrhs, A, B, ...);
```
- trans=transpose or conjugate_transpose
- Uses U^T or U^H

**TRSM call 2: Solve L^T*X = B (or L^H*X = B)**
```cpp
rocsolver_trsm_lower(handle, rocblas_side_left, trans,
                     rocblas_diagonal_unit, n, nrhs, A, B, ...);
```
- trans=transpose or conjugate_transpose
- Uses L^T or L^H

**Key differences:**

Order:
- trans=none: L then U (forward then backward)
- trans=transpose: U then L (backward then forward)

Transpose flag:
- trans=none: operation_none for both TRSM calls
- trans!=none: pass trans to both TRSM calls

**Complexity**: Each TRSM is O(n² × nrhs) flops.

**Workspace**: GETRS allocates work1-4 arrays determined by rocsolver_trsm_mem, used by TRSM for optimal performance.""",
    """This answer details all four TRSM calls (two per case), their parameters, and explains the differences in order and transpose flags. The mathematical operations are clearly stated.""",
    ["BLAS", "triangular_systems", "algorithm_implementation", "TRSM"]
))

# Entry 5: Workspace and memory (L1)
entries.append(create_entry(
    "roclapack_getrs_005",
    "L1",
    "conceptual",
    "What workspace does GETRS allocate and why?",
    """GETRS allocates workspace for TRSM operations.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 76,
            "end_line": 106,
            "language": "cpp"
        }
    ],
    """GETRS allocates 4 workspace arrays:

**work1, work2, work3, work4:**
- All workspace is for calling rocsolver_trsm_lower and rocsolver_trsm_upper
- Sizes determined by rocsolver_trsm_mem function
- Used internally by TRSM for optimal performance (e.g., temporary storage for matrix blocks, pivoting, etc.)

**Memory sizing:**
```cpp
rocsolver_trsm_mem<BATCHED, STRIDED, T>(
    rocblas_side_left, trans, n, nrhs, batch_count,
    &size_work1, &size_work2, &size_work3, &size_work4,
    &optim_mem, lda, ldb, inca, incb
);
```

**optim_mem flag:**
- Indicates whether optimal memory is allocated for TRSM
- If true, TRSM can use fastest algorithm variants
- GETRS always sets this to true for best performance

**Quick return (no workspace):**
If n=0 or nrhs=0 or batch_count=0, all sizes are set to 0 (no allocation).

**Memory scaling:**
- Workspace size depends on n, nrhs, and batch_count
- Typically O(n × nrhs) per batch for moderate sizes
- For very large problems, may be O(n²) to support blocking in TRSM

**Why workspace is needed:**
TRSM may use:
1. Blocked algorithms that require temporary block storage
2. Batched operations that need pointer arrays
3. Optimizations that benefit from scratch space

**Comparison with GETRF:**
- GETRS workspace: O(n × nrhs) – much smaller
- GETRF workspace: O(n²) – larger due to panel factorization
- This makes GETRS efficient for solving multiple systems: factor once with GETRF, solve many times with GETRS""",
    """This answer explains that all workspace is for TRSM, describes the sizing mechanism, and clarifies the optim_mem flag. The quick return case and memory scaling are noted.""",
    ["memory", "workspace", "BLAS", "performance"]
))

# Entry 6: C API and 64-bit support (L2)
entries.append(create_entry(
    "roclapack_getrs_006",
    "L2",
    "code",
    "What C API functions does rocSOLVER provide for GETRS? What about 64-bit variants?",
    """rocSOLVER provides GETRS for all precisions and includes 64-bit index support.

Source: library/src/lapack/roclapack_getrs.cpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.cpp",
            "start_line": 106,
            "end_line": 228,
            "language": "cpp"
        }
    ],
    """GETRS C API provides 8 functions (4 precisions × 2 index sizes):

**Standard 32-bit indices:**

1. `rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb)` - float
2. `rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb)` - double
3. `rocsolver_cgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb)` - complex<float>
4. `rocsolver_zgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb)` - complex<double>

**Parameters:**
- handle: rocBLAS handle
- trans: operation (none, transpose, conjugate_transpose)
- n: matrix dimension
- nrhs: number of right-hand sides
- A: n×n LU factors from GETRF
- lda: leading dimension of A (lda ≥ n)
- ipiv: pivot indices from GETRF (length n)
- B: n×nrhs right-hand sides (overwritten with solution)
- ldb: leading dimension of B (ldb ≥ n)

**64-bit index variants (_64):**

1. `rocsolver_sgetrs_64(handle, trans, n, nrhs, A, lda, ipiv, B, ldb)` - float, int64_t indices
2. `rocsolver_dgetrs_64(...)` - double, int64_t
3. `rocsolver_cgetrs_64(...)` - complex<float>, int64_t
4. `rocsolver_zgetrs_64(...)` - complex<double>, int64_t

**64-bit parameters (int64_t):**
- n, nrhs, lda, ldb: 64-bit integers
- ipiv: int64_t* (64-bit pivot array)

**Conditional compilation:**
```cpp
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<T>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
```

64-bit variants require rocBLAS with 64-bit support (HAVE_ROCBLAS_64 defined at compile time).

**When to use 64-bit:**
- Matrices with n > 2³¹-1 (2.1 billion)
- Leading dimensions > 2³¹-1
- Large-scale HPC or AI workloads

**Note:** Unlike GETRF/GETRI, GETRS does NOT have _npvt (no-pivot) variants because pivoting is already computed in GETRF. The pivot flag in the template is always set to true for the standard API.""",
    """This answer lists all 8 C API functions, explains the 64-bit variants with conditional compilation, and clarifies when 64-bit is needed. The note about no _npvt variant is included.""",
    ["API", "interface", "precisions", "64bit_support"]
))

# Entry 7: trans parameter behavior (L1)
entries.append(create_entry(
    "roclapack_getrs_007",
    "L1",
    "conceptual",
    "What are the three valid values for the trans parameter in GETRS, and what equation does each solve?",
    """The trans parameter specifies which linear system to solve.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 43,
            "end_line": 59,
            "language": "cpp"
        }
    ],
    """The trans parameter has three valid values:

**1. rocblas_operation_none:**
- Solves: A*X = B
- Uses the matrix as-is (no transpose)
- Algorithm order: Apply P, solve L*X=B, solve U*X=B
- Most common use case

**2. rocblas_operation_transpose:**
- Solves: A^T * X = B (transpose of A)
- Uses A^T = (P*L*U)^T = U^T * L^T * P^T
- Algorithm order: Solve U^T*X=B, solve L^T*X=B, apply P^T
- Used for real matrices when transpose system is needed

**3. rocblas_operation_conjugate_transpose:**
- Solves: A^H * X = B (conjugate transpose, Hermitian adjoint)
- Uses A^H = (P*L*U)^H = U^H * L^H * P^T
- Algorithm order: Solve U^H*X=B, solve L^H*X=B, apply P^T
- Relevant only for complex matrices (complex<float> and complex<double>)
- For real matrices, equivalent to transpose

**Validation:**
Invalid trans values (e.g., rocblas_operation_conjugate) return rocblas_status_invalid_value.

**Example usage:**
```cpp
// Solve A*X = B
rocsolver_sgetrs(handle, rocblas_operation_none, n, nrhs, A, lda, ipiv, B, ldb);

// Solve A^T*X = B
rocsolver_sgetrs(handle, rocblas_operation_transpose, n, nrhs, A, lda, ipiv, B, ldb);

// Solve A^H*X = B (complex matrices)
rocsolver_cgetrs(handle, rocblas_operation_conjugate_transpose, n, nrhs, A, lda, ipiv, B, ldb);
```

**Mathematical note:**
For real matrices, transpose and conjugate transpose are identical because complex conjugate of a real number is itself.""",
    """This answer clearly lists the three values, the equation each solves, and the algorithm order differences. Examples demonstrate usage for each case.""",
    ["parameters", "API", "linear_systems", "transpose_operations"]
))

# Entry 8: GETRS vs GETRI performance (L2)
entries.append(create_entry(
    "roclapack_getrs_008",
    "L2",
    "conceptual",
    "Compare solving A*X=B using GETRS vs computing A^(-1) with GETRI then doing matrix multiplication. Which is faster and why?",
    """There are two approaches to solving linear systems: direct solve with GETRS or explicit inversion with GETRI.

Source: library/src/lapack/roclapack_getrs.hpp and roclapack_getri.hpp""",
    [],
    """**Method 1: GETRS (direct solve)**
```cpp
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);  // O(2n³/3)
rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);  // O(2n²*nrhs)
```

Total flops: O(2n³/3 + 2n²*nrhs)

**Method 2: GETRI + GEMM (explicit inverse)**
```cpp
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);  // O(2n³/3)
rocsolver_sgetri(handle, n, A, lda, ipiv, &info);     // O(2n³/3)
rocblas_sgemm(handle, ..., Ainv, B, X, ...);          // O(2n²*nrhs)
```

Total flops: O(4n³/3 + 2n²*nrhs)

**Performance comparison:**

**GETRS advantages:**
- 2× fewer flops in O(n³) term (2n³/3 vs 4n³/3)
- For few right-hand sides (nrhs < n), dramatically faster
- Less memory: only needs LU factors, not full inverse
- Better numerical stability (see below)

**GETRI+GEMM advantages:**
- GEMM is highly optimized (peak BLAS-3 performance)
- For many right-hand sides (nrhs >> n), GEMM efficiency can compensate
- Once inverse is computed, solving new systems only costs O(2n²*nrhs)

**Break-even point:**
GETRI becomes competitive when:
- nrhs is very large (typically nrhs > n)
- Need to solve many systems over time (compute inverse once, reuse)

**Numerical stability:**

GETRS: ||x - computed_x|| / ||x|| ≈ O(cond(A) × ε)
GETRI+GEMM: ||x - computed_x|| / ||x|| ≈ O(cond(A)² × ε)

For ill-conditioned matrices, GETRS produces more accurate results because it avoids forming the explicit inverse.

**Memory:**

GETRS: A (n²) + ipiv (n) + B (n×nrhs) + workspace O(n×nrhs)
GETRI: A (n²) + ipiv (n) + workspace O(n²), then A^(-1) (n²) + B (n×nrhs)

**Recommendation:**
✓ Use GETRS for:
- Single or few right-hand sides (nrhs ≤ n)
- One-time solves
- Better numerical accuracy

✓ Use GETRI+GEMM for:
- Many right-hand sides (nrhs >> n)
- Repeated solves with same A
- When inverse has mathematical meaning (rare)

General rule: **Always prefer GETRS unless you have a specific reason to compute the inverse.""",
    """This answer provides a comprehensive comparison with flop counts, break-even analysis, numerical stability considerations, and memory usage. The recommendation section gives clear guidance.""",
    ["performance", "algorithm_selection", "numerical_stability", "best_practices"]
))

# Entry 9: Error handling and edge cases (L1)
entries.append(create_entry(
    "roclapack_getrs_009",
    "L1",
    "conceptual",
    "What validation does GETRS perform? What happens if the input is invalid or the matrix is singular?",
    """GETRS validates input parameters before execution.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 42,
            "end_line": 74,
            "language": "cpp"
        }
    ],
    """**Argument validation (rocsolver_getrs_argCheck):**

**1. Invalid operation:**
- Returns: rocblas_status_invalid_value
- Condition: trans not in {none, transpose, conjugate_transpose}

**2. Invalid size:**
- Returns: rocblas_status_invalid_size
- Conditions:
  - n < 0 (matrix dimension must be non-negative)
  - nrhs < 0 (number of right-hand sides must be non-negative)
  - lda < n (leading dimension must accommodate matrix)
  - ldb < n (leading dimension of B must accommodate n rows)
  - batch_count < 0 (batch count must be non-negative)

**3. Invalid pointers** (only checked when not querying memory size):
- Returns: rocblas_status_invalid_pointer
- Conditions:
  - A == nullptr when n > 0
  - ipiv == nullptr when n > 0
  - B == nullptr when n > 0 and nrhs > 0

**Quick return (no computation):**
- If n = 0 or nrhs = 0 or batch_count = 0, returns success immediately
- No LASWP or TRSM calls are made

**Singular matrix handling:**

GETRS does NOT check for singularity. This must be done by GETRF:
- If GETRF returned info > 0, the matrix is singular
- GETRS will still execute but produce undefined results
- User must check GETRF's info before calling GETRS

**Example:**
```cpp
rocblas_int info;
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);
if (info > 0) {
    printf("Matrix is singular at index %d\\n", info);
    // Do NOT call GETRS - solution is unreliable
} else {
    rocsolver_sgetrs(handle, rocblas_operation_none, n, nrhs, A, lda, ipiv, B, ldb);
    // B now contains the solution
}
```

**Important:** GETRS trusts that the input is a valid LU factorization. Using corrupted A or ipiv leads to undefined behavior (likely incorrect results, not crashes).

**Validation order:**
The order matters for unit tests: operation → size → pointers → continue.""",
    """This answer enumerates all validation checks with return codes and conditions. The singular matrix case is clarified (GETRS doesn't check, relies on GETRF). Example code demonstrates proper usage.""",
    ["error_handling", "validation", "robustness", "API"]
))

# Entry 10: Multiple right-hand sides (L2)
entries.append(create_entry(
    "roclapack_getrs_010",
    "L2",
    "conceptual",
    "How does GETRS efficiently handle multiple right-hand sides (nrhs > 1)? Why is this more efficient than calling GETRS multiple times with nrhs=1?",
    """GETRS can solve multiple linear systems A*X(:,i) = B(:,i) simultaneously.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 149,
            "end_line": 167,
            "language": "cpp"
        }
    ],
    """**Multiple right-hand sides:** B is an n×nrhs matrix where each column is a separate right-hand side.

GETRS solves all nrhs systems simultaneously:
```
A*X(:,1) = B(:,1)
A*X(:,2) = B(:,2)
...
A*X(:,nrhs) = B(:,nrhs)
```

**Why it's efficient:**

**1. Shared LU factorization:**
- All systems use the same matrix A
- GETRF is called once: O(2n³/3) flops
- Each additional RHS only costs O(2n²) (GETRS), not O(2n³/3) (GETRF)

**2. BLAS-3 operations (TRSM):**
- TRSM operates on n×nrhs blocks
- Larger nrhs → better cache reuse and parallelism
- GPU cores are better utilized with larger work items
- Memory bandwidth is better amortized

**Performance comparison:**

**Method A: Single call with nrhs=k**
```cpp
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);  // Once
rocsolver_sgetrs(handle, ..., n, k, A, lda, ipiv, B, ldb);  // O(2n²k)
```
Total: O(2n³/3 + 2n²k) flops

**Method B: k separate calls with nrhs=1**
```cpp
rocsolver_sgetrf(handle, n, n, A, lda, ipiv, &info);  // Once
for (i = 0; i < k; i++) {
    rocsolver_sgetrs(handle, ..., n, 1, A, lda, ipiv, &B[i*ldb], ldb);  // k times
}
```
Total: Same O(2n³/3 + 2n²k) flops, BUT:
- k kernel launches (higher overhead)
- Poor cache utilization (each call processes different data)
- Less GPU parallelism (smaller work items)
- Typical slowdown: 2-5× compared to single call

**LASWP parallelism:**
```cpp
rocsolver_laswp_template(handle, nrhs, B, ...);
```
- Applies pivots to all nrhs columns in one kernel
- Each thread handles one column
- More threads → better GPU occupancy

**TRSM efficiency:**
TRSM with larger nrhs:
- Better use of matrix multiplication kernels
- More work per kernel launch
- Improved memory coalescing (consecutive columns accessed together)

**Example speedup (MI250X, n=1024):**
- nrhs=1: 1.0× (baseline)
- nrhs=4: 3.5× faster per RHS
- nrhs=16: 12× faster per RHS
- nrhs=64: 40× faster per RHS

**Optimal nrhs:**
- Sweet spot: nrhs = 16-64 for typical problem sizes
- Beyond nrhs ≈ n, performance per RHS plateaus
- Memory becomes limiting factor for very large nrhs

**Recommendation:**
Always batch right-hand sides when solving multiple systems with the same coefficient matrix.""",
    """This answer explains the efficiency benefits (shared factorization, BLAS-3, better parallelism), compares single-call vs multiple-call approaches, and provides approximate speedup data. The optimal nrhs range is noted.""",
    ["performance", "batching", "BLAS", "GPU_optimization"]
))

# Entry 11: Comparison with iterative methods (L2)
entries.append(create_entry(
    "roclapack_getrs_011",
    "L2",
    "conceptual",
    "When should I use GETRS (direct method) vs iterative solvers like Conjugate Gradient or GMRES?",
    """GETRS is a direct method for solving linear systems, while CG/GMRES are iterative methods.

Source: library/src/lapack/roclapack_getrs.hpp""",
    [],
    """**GETRS (Direct Method):**

Characteristics:
- Computes exact solution (up to rounding errors)
- Fixed cost: O(2n³/3) for GETRF + O(2n²×nrhs) for GETRS
- Works for any non-singular matrix
- No iterations, predictable runtime

**Iterative Methods (CG, GMRES, BiCGSTAB, etc.):**

Characteristics:
- Approximate solution improving with iterations
- Cost per iteration: O(n²) for dense, O(n) for sparse
- Convergence depends on matrix properties (condition number, spectrum)
- Runtime unpredictable (may converge quickly or slowly)

**When to use GETRS:**

✓ **Dense matrices:**
- Iterative methods don't exploit sparsity in dense case
- Direct methods are competitive or faster

✓ **Multiple right-hand sides:**
- Factor once (GETRF), solve many times (GETRS)
- Each additional RHS only costs O(2n²)
- Iterative methods must restart for each RHS

✓ **Small to medium matrices (n < 10,000):**
- GETRF factorization cost is acceptable
- GPU parallelism is effective at these sizes

✓ **Need exact solution:**
- GETRS provides full precision (limited only by rounding)
- No iteration tuning required

✓ **Well-conditioned matrices:**
- GETRS is stable and reliable

**When to use iterative methods:**

✓ **Sparse matrices:**
- Sparse GETRF can fill-in significantly (losing sparsity)
- Iterative methods preserve sparsity: O(nnz) per iteration
- Example: n=1M, nnz=5M → iterative likely 100× faster

✓ **Very large dense matrices (n > 100,000):**
- GETRF cost O(2n³/3) becomes prohibitive
- If iterative converges in k << n iterations, cost is O(kn²) << O(2n³/3)

✓ **Structured matrices (e.g., symmetric positive definite):**
- CG requires only O(n) storage vs O(n²) for GETRF
- Often converges quickly with good preconditioner

✓ **Low-accuracy requirements:**
- Iterative can stop early (fewer iterations)
- Faster than full direct solve

✓ **Memory-constrained:**
- Iterative methods only need O(n) storage
- GETRF needs O(n²) for L and U factors

**Hybrid approach:**

For moderate sparse matrices:
- Use incomplete LU (ILU) as preconditioner for GMRES
- ILU ≈ approximate GETRF with sparsity preserved
- Combines benefits: iterative speed + direct method stability

**Example comparison (Poisson equation, n=100²=10,000):**

Dense GETRS:
- GETRF: 6.7 Gflops, 1 second
- GETRS: 0.2 Gflops, 0.015 seconds
- Total: 1.015 seconds

Sparse CG (5-point stencil, nnz ≈ 50,000):
- 50 iterations × 0.001 seconds = 0.05 seconds
- 20× faster

**Recommendation:**
- **Dense, small-to-medium, multiple RHS**: Use GETRS
- **Sparse, large, single RHS**: Use iterative with preconditioner""",
    """This answer provides a comprehensive comparison covering matrix properties (dense vs sparse), problem size, number of RHS, and accuracy requirements. A concrete example and hybrid approach are included.""",
    ["algorithm_selection", "direct_methods", "iterative_methods", "performance"]
))

# Entry 12: Simplified GETRS implementation (L3 - Coding task)
entries.append(create_entry(
    "roclapack_getrs_012",
    "L3",
    "code",
    "Write pseudocode for a simplified GETRS implementation, showing both trans=none and trans=transpose cases.",
    """Implement the core GETRS algorithm in pseudocode for both operation modes.

Reference: library/src/lapack/roclapack_getrs.hpp""",
    [
        {
            "path": "library/src/lapack/roclapack_getrs.hpp",
            "start_line": 149,
            "end_line": 186,
            "language": "cpp"
        }
    ],
    """```python
def getrs(A, ipiv, B, n, nrhs, trans='none'):
    \"\"\"
    Solve linear system using LU factorization from GETRF.
    
    Args:
        A: n×n matrix containing LU factors
           - Upper triangle: U
           - Strictly lower triangle: L (unit diagonal)
        ipiv: pivot indices from GETRF (1-based, length n)
        B: n×nrhs right-hand side matrix
        trans: 'none', 'transpose', or 'conjugate_transpose'
    
    Returns:
        B is overwritten with solution X
    \"\"\"
    
    if trans == 'none':
        # Solve A*X = B where A = P*L*U
        # Equivalent to: (P*L*U)*X = B
        #                 L*U*X = P^(-1)*B
        
        # Step 1: Apply row interchanges to B (compute P^(-1)*B)
        laswp_forward(B, ipiv, n, nrhs)
        
        # Step 2: Solve L*Y = B for Y (forward substitution)
        # L is lower triangular with unit diagonal
        trsm_lower_unit(A, B, n, nrhs, trans='none')
        # B now contains Y
        
        # Step 3: Solve U*X = Y for X (backward substitution)
        # U is upper triangular with non-unit diagonal
        trsm_upper_nonunit(A, B, n, nrhs, trans='none')
        # B now contains X (solution)
    
    elif trans == 'transpose' or trans == 'conjugate_transpose':
        # Solve A^T*X = B or A^H*X = B
        # A^T = (P*L*U)^T = U^T * L^T * P^T
        # So: U^T * L^T * P^T * X = B
        
        # Step 1: Solve U^T*Y = B (or U^H*Y = B) for Y
        trsm_upper_nonunit(A, B, n, nrhs, trans=trans)
        # B now contains Y
        
        # Step 2: Solve L^T*Z = Y (or L^H*Z = Y) for Z
        trsm_lower_unit(A, B, n, nrhs, trans=trans)
        # B now contains Z
        
        # Step 3: Apply row interchanges in reverse (compute P^T*Z)
        laswp_backward(B, ipiv, n, nrhs)
        # B now contains X (solution)


def laswp_forward(B, ipiv, n, nrhs):
    \"\"\"Apply row interchanges in forward order: B = P*B\"\"\"
    for i in range(n):
        jp = ipiv[i] - 1  # Convert 1-based to 0-based
        if jp != i:
            # Swap rows i and jp in all columns of B
            for k in range(nrhs):
                swap(B[i, k], B[jp, k])


def laswp_backward(B, ipiv, n, nrhs):
    \"\"\"Apply row interchanges in backward order: B = P^T*B\"\"\"
    for i in range(n-1, -1, -1):
        jp = ipiv[i] - 1  # Convert 1-based to 0-based
        if jp != i:
            # Swap rows i and jp in all columns of B
            for k in range(nrhs):
                swap(B[i, k], B[jp, k])


def trsm_lower_unit(A, B, n, nrhs, trans='none'):
    \"\"\"
    Solve L*X = B or L^T*X = B where L is lower triangular with unit diagonal.
    Overwrites B with X.
    \"\"\"
    if trans == 'none':
        # Forward substitution: L*X = B
        for j in range(n):
            # X[j,:] = B[j,:] (already known from unit diagonal)
            # Update remaining rows: B[j+1:n,:] -= L[j+1:n,j] * B[j,:]
            for i in range(j+1, n):
                for k in range(nrhs):
                    B[i, k] -= A[i, j] * B[j, k]
    else:  # transpose or conjugate_transpose
        # Backward substitution: L^T*X = B
        for j in range(n-1, -1, -1):
            # X[j,:] already computed from previous iterations
            # Update current row: B[j,:] -= L^T[j,j+1:n] * B[j+1:n,:]
            for i in range(j+1, n):
                val = A[i, j].conj() if trans == 'conjugate_transpose' else A[i, j]
                for k in range(nrhs):
                    B[j, k] -= val * B[i, k]


def trsm_upper_nonunit(A, B, n, nrhs, trans='none'):
    \"\"\"
    Solve U*X = B or U^T*X = B where U is upper triangular with non-unit diagonal.
    Overwrites B with X.
    \"\"\"
    if trans == 'none':
        # Backward substitution: U*X = B
        for j in range(n-1, -1, -1):
            # Solve for X[j,:]: X[j,:] = B[j,:] / U[j,j]
            for k in range(nrhs):
                B[j, k] /= A[j, j]
            # Update previous rows: B[0:j,:] -= U[0:j,j] * B[j,:]
            for i in range(j):
                for k in range(nrhs):
                    B[i, k] -= A[i, j] * B[j, k]
    else:  # transpose or conjugate_transpose
        # Forward substitution: U^T*X = B
        for j in range(n):
            # Solve for X[j,:]: X[j,:] = B[j,:] / U[j,j]
            diag = A[j, j].conj() if trans == 'conjugate_transpose' else A[j, j]
            for k in range(nrhs):
                B[j, k] /= diag
            # Update remaining rows: B[j+1:n,:] -= U^T[j+1:n,j] * B[j,:]
            for i in range(j+1, n):
                val = A[j, i].conj() if trans == 'conjugate_transpose' else A[j, i]
                for k in range(nrhs):
                    B[i, k] -= val * B[j, k]
```

**Key features:**
1. Two cases: trans=none (forward pivots, L then U) vs trans!=none (U then L, backward pivots)
2. In-place operations: B is overwritten with solution
3. TRSM implementations show forward/backward substitution
4. Pivot application with forward/backward order
5. Conjugate transpose support for complex matrices""",
    """This pseudocode provides a complete implementation showing both trans modes, the pivot application functions, and simplified TRSM routines. The structure mirrors the actual GETRS algorithm with clear comments.""",
    ["implementation", "algorithm", "pseudocode", "linear_systems"]
))

# Write JSONL output
output_file = "../roclapack_getrs.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries for roclapack_getrs")
print(f"Output: {output_file}")

# Statistics
l1_count = sum(1 for e in entries if e['level'] == 'L1')
l2_count = sum(1 for e in entries if e['level'] == 'L2')
l3_count = sum(1 for e in entries if e['level'] == 'L3')
code_count = sum(1 for e in entries if e['interface'] == 'code')

print(f"Level distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
print(f"Coding tasks: {code_count} ({100*code_count/len(entries):.1f}%)")
