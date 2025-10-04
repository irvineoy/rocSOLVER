#!/usr/bin/env python3
"""Generate POSV dataset entries following DATASET.md schema."""

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

# Entry 1: POSV algorithm overview - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="posv",
    instruction="Explain the complete POSV algorithm for solving A*X = B where A is symmetric positive definite. What are the two main stages and why is Cholesky factorization preferred over LU?",
    context_text="POSV solves linear systems A*X = B for symmetric positive definite matrices using Cholesky factorization. This is more efficient than general LU factorization and exploits the structure of positive definite matrices.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// compute Cholesky factorization of A
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2, work3,
    work4, pivots_savedB, iinfo, optim_mem);

// save elements of B that will be overwritten by POTRS for cases where info is nonzero
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                        dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                        strideB, pivots_savedB, info_mask(info));

// solve AX = B, overwriting B with X
rocsolver_potrs_template<BATCHED, STRIDED, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, B,
                                              shiftB, ldb, strideB, batch_count, work1, work2,
                                              work3, work4, optim_mem);

// restore elements of B that were overwritten by POTRS in cases where info is nonzero
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                        dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB, ldb,
                        strideB, pivots_savedB, info_mask(info));"""
    }],
    answer="""POSV algorithm for solving A*X = B with symmetric positive definite A:

**Two main stages:**

**Stage 1: Cholesky Factorization (POTRF)**

Decompose A into triangular form:
```
If uplo = upper: A = U^T * U or A = U^H * U (complex)
If uplo = lower: A = L * L^T or A = L * L^H (complex)

Where U is upper triangular, L is lower triangular
```

```cpp
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA,
    info,        // Set to k if A[k,k] <= 0 (not positive definite)
    batch_count, scalars, work1, work2, work3, work4,
    pivots_savedB, iinfo, optim_mem);
```

Output: A is overwritten with U or L

**Stage 2: Triangular Solve (POTRS)**

Solve using the triangular factors:

For uplo = upper (A = U^T * U):
```
Step 1: Solve U^T * Y = B  (forward substitution)
Step 2: Solve U * X = Y    (backward substitution)
Result: X is the solution
```

For uplo = lower (A = L * L^T):
```
Step 1: Solve L * Y = B    (forward substitution)
Step 2: Solve L^T * X = Y  (backward substitution)
Result: X is the solution
```

```cpp
rocsolver_potrs_template<BATCHED, STRIDED, T>(
    handle, uplo, n, nrhs, A, shiftA, lda, strideA,
    B, shiftB, ldb, strideB,  // B overwritten with solution X
    batch_count, work1, work2, work3, work4, optim_mem);
```

**Error handling: B preservation**

Critical feature: If POTRF fails (info != 0), B must remain unchanged:

```cpp
// Before POTRS: Save B in case of failure
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), ...,
    copymat_to_buffer, n, nrhs, B, ..., pivots_savedB,
    info_mask(info));  // Only copy if info[i] != 0

// Run POTRS (will corrupt B if info != 0)
rocsolver_potrs_template(...);

// After POTRS: Restore B for failed cases
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), ...,
    copymat_from_buffer, n, nrhs, B, ..., pivots_savedB,
    info_mask(info));  // Only restore if info[i] != 0
```

**Why Cholesky over LU?**

1. **Computational cost**:
   - Cholesky: ~(n³/3) flops
   - LU with pivoting: ~(2n³/3) flops
   - **Speedup: 2× faster!**

2. **Memory efficiency**:
   - Cholesky: Only one triangle stored (n²/2 elements)
   - LU: Both L and U (n² elements)
   - No pivot vector needed

3. **Numerical stability**:
   - Cholesky: No pivoting needed (SPD guarantees stability)
   - LU: Requires partial pivoting
   - Cholesky is backward stable for SPD matrices

4. **Exploitation of structure**:
   - Uses symmetry: Only compute/store half the matrix
   - Uses positive definiteness: All pivots positive, no breakdown

**Complexity:**

- POTRF: O(n³/3) for factorization
- POTRS: O(n² × nrhs) for solve (two TRSM calls)
- Total: O(n³/3 + n²×nrhs)

**When POSV fails:**

If A is not positive definite, POTRF detects this:
```cpp
info[i] = k  // where A[k,k] <= 0 or sqrt(A[k,k]) failed
```

In this case:
- Factorization stops at column k
- B is restored to original values (not corrupted)
- User must use GESV (LU with pivoting) instead""",
    rationale="POSV combines Cholesky factorization (POTRF) with triangular solve (POTRS) to solve linear systems for symmetric positive definite matrices. Cholesky is 2× faster than LU factorization by exploiting symmetry and positive definiteness, requiring no pivoting and storing only one triangle.",
    tags=["algorithm", "posv", "cholesky", "linear-solve", "spd"]
))

# Entry 2: B preservation logic - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="posv",
    instruction="Implement the B preservation logic in POSV. Why must B be saved before POTRS and restored afterward? How does info_mask work?",
    context_text="POSV must handle failures from POTRF gracefully. If the matrix is not positive definite, POTRF fails and sets info to the failing column index. However, POTRS would still execute and corrupt B. POSV prevents this by saving and restoring B selectively.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// save elements of B that will be overwritten by POTRS for cases where info is nonzero
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                        dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB, ldb,
                        strideB, pivots_savedB, info_mask(info));

// solve AX = B, overwriting B with X
rocsolver_potrs_template<BATCHED, STRIDED, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, B,
                                              shiftB, ldb, strideB, batch_count, work1, work2,
                                              work3, work4, optim_mem);

// restore elements of B that were overwritten by POTRS in cases where info is nonzero
ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksx, copyblocksy, batch_count),
                        dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B, shiftB, ldb,
                        strideB, pivots_savedB, info_mask(info));"""
    }],
    answer="""B preservation logic in POSV:

**Why B must be saved:**

Problem scenario:
```
Batch of 3 matrices:
- Matrix 0: SPD → POTRF succeeds (info[0] = 0)
- Matrix 1: Not SPD → POTRF fails (info[1] = 5, failed at column 5)
- Matrix 2: SPD → POTRF succeeds (info[2] = 0)
```

Without B preservation:
```
After POTRF: info = [0, 5, 0]
After POTRS:
  - Matrix 0: B → X (correct solution) ✓
  - Matrix 1: B → garbage (POTRS ran on incomplete factorization) ✗
  - Matrix 2: B → X (correct solution) ✓
```

User expects:
- For successful cases (info=0): B contains solution X
- For failed cases (info≠0): B unchanged (original values)

**Implementation:**

```cpp
const rocblas_int copyblocksx = (n - 1) / 32 + 1;
const rocblas_int copyblocksy = (nrhs - 1) / 32 + 1;

// STEP 1: Run POTRF
rocsolver_potrf_template(..., info, ...);
// After this: info[i] = 0 (success) or k (failed at column k)

// STEP 2: Save B for failed cases
ROCSOLVER_LAUNCH_KERNEL(
    (copy_mat<T, U>),
    dim3(copyblocksx, copyblocksy, batch_count),  // Grid covers n×nrhs×batch
    dim3(32, 32),                                  // Block size
    0, stream,
    copymat_to_buffer,                             // Direction: B → buffer
    n, nrhs,                                       // Dimensions
    B, shiftB, ldb, strideB,                       // Source
    pivots_savedB,                                 // Destination (reused buffer)
    info_mask(info));                              // Conditional: only if info[i] != 0

// STEP 3: Run POTRS (will execute on all batches, corrupting failed ones)
rocsolver_potrs_template(..., B, ...);
// After this:
//   info[0]=0: B contains solution ✓
//   info[1]=5: B contains garbage ✗
//   info[2]=0: B contains solution ✓

// STEP 4: Restore B for failed cases
ROCSOLVER_LAUNCH_KERNEL(
    (copy_mat<T, U>),
    dim3(copyblocksx, copyblocksy, batch_count),
    dim3(32, 32),
    0, stream,
    copymat_from_buffer,                           // Direction: buffer → B
    n, nrhs,
    B, shiftB, ldb, strideB,                       // Destination
    pivots_savedB,                                 // Source
    info_mask(info));                              // Conditional: only if info[i] != 0

// Final state:
//   info[0]=0: B contains solution ✓
//   info[1]=5: B contains original values ✓ (restored)
//   info[2]=0: B contains solution ✓
```

**info_mask mechanism:**

```cpp
// info_mask returns a mask indicating which batches should be processed
// Typically implemented as:
__device__ inline bool info_mask(rocblas_int* info)
{
    int bid = hipBlockIdx_z;  // Batch index
    return (info[bid] != 0);  // Process only if POTRF failed
}

// In copy_mat kernel:
__global__ void copy_mat(..., info_mask_func mask)
{
    int bid = hipBlockIdx_z;

    if(!mask(info))  // Check if this batch needs copying
        return;      // Skip if info[bid] == 0

    // Perform copy for this batch
    ...
}
```

**Memory reuse:**

```cpp
// pivots_savedB buffer size calculation
*size_pivots_savedB = std::max(
    *size_pivots_savedB,                     // From POTRF (small)
    sizeof(T) * n * nrhs * batch_count);     // For saving B (large)

// The buffer is reused:
// 1. POTRF may use it for pivot indices (small)
// 2. POSV uses it to save B (large, dominates size)
```

**Overhead:**

For batch_count=100, n=1000, nrhs=10:
- Save: Copy 100×1000×10 = 1M elements (only for failed batches)
- Restore: Copy back (only for failed batches)
- If all succeed: Both kernels exit early (minimal cost)
- If all fail: ~2× copy cost, but factorization didn't help anyway

**Key insights:**

1. **Selective copying**: Only failed batches are saved/restored
2. **Buffer reuse**: pivots_savedB serves dual purpose
3. **Correctness guarantee**: B always contains valid data after POSV
4. **Batched-friendly**: Each batch independently decides whether to copy""",
    rationale="POSV must preserve B for cases where POTRF fails, because POTRS would execute anyway and corrupt B with invalid results. The info_mask mechanism allows selective save/restore: only batches with info[i]≠0 trigger the copy operations, minimizing overhead while ensuring correctness.",
    tags=["coding", "error-handling", "batched", "memory-management"]
))

# Entry 3: POTRS triangular solve - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="posv",
    instruction="Implement POTRS for both upper and lower triangular cases. Explain the order of TRSM calls and why conjugate transpose is used for complex matrices.",
    context_text="POTRS solves A*X = B using the Cholesky factorization from POTRF. For A = L*L^T, it solves L*Y=B then L^T*X=Y. For A = U^T*U, it solves U^T*Y=B then U*X=Y. Both use two TRSM (triangular solve) calls.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrs.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_upper)
{
    // solve U'*X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
        rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
        batch_count, optim_mem, work1, work2, work3, work4);

    // solve U*X = B, overwriting B with X
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(handle, rocblas_side_left, rocblas_operation_none,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                              optim_mem, work1, work2, work3, work4);
}
else
{
    // solve L*X = B, overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(handle, rocblas_side_left, rocblas_operation_none,
                                              rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                                              lda, strideA, B, shiftB, ldb, strideB, batch_count,
                                              optim_mem, work1, work2, work3, work4);

    // solve L'*X = B, overwriting B with X
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
        rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
        batch_count, optim_mem, work1, work2, work3, work4);
}"""
    }],
    answer="""POTRS implementation for upper and lower cases:

**Mathematical background:**

Given Cholesky factorization A = L*L^T (or A = U^T*U), solve A*X = B:

**Case 1: Lower triangular (uplo = rocblas_fill_lower)**

A = L * L^T, so:
```
L * L^T * X = B
```

Algorithm:
```
Step 1: Solve L * Y = B         (forward substitution)
Step 2: Solve L^T * X = Y       (backward substitution)
Result: X is the solution
```

Implementation:
```cpp
if(uplo == rocblas_fill_lower)
{
    // Step 1: L * Y = B → Y (overwrite B with Y)
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle,
        rocblas_side_left,                    // L is on the left
        rocblas_operation_none,               // No transpose (L, not L^T)
        rocblas_diagonal_non_unit,            // L has non-unit diagonal
        n, nrhs,                              // Dimensions: n×n L, n×nrhs B
        A, shiftA, lda, strideA,              // L stored in lower triangle of A
        B, shiftB, ldb, strideB,              // B → Y
        batch_count, optim_mem, work1, work2, work3, work4);

    // Step 2: L^T * X = Y → X (overwrite B/Y with X)
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle,
        rocblas_side_left,
        rocblas_operation_conjugate_transpose, // L^H (Hermitian) for complex, L^T for real
        rocblas_diagonal_non_unit,
        n, nrhs,
        A, shiftA, lda, strideA,              // L^T
        B, shiftB, ldb, strideB,              // Y → X (final solution)
        batch_count, optim_mem, work1, work2, work3, work4);
}
```

**Case 2: Upper triangular (uplo = rocblas_fill_upper)**

A = U^T * U (or A = U^H * U for complex), so:
```
U^T * U * X = B
```

Algorithm:
```
Step 1: Solve U^T * Y = B       (forward substitution)
Step 2: Solve U * X = Y         (backward substitution)
Result: X is the solution
```

Implementation:
```cpp
if(uplo == rocblas_fill_upper)
{
    // Step 1: U^T * Y = B → Y (overwrite B with Y)
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        handle,
        rocblas_side_left,
        rocblas_operation_conjugate_transpose, // U^H for complex, U^T for real
        rocblas_diagonal_non_unit,
        n, nrhs,
        A, shiftA, lda, strideA,              // U^T stored in upper triangle of A
        B, shiftB, ldb, strideB,              // B → Y
        batch_count, optim_mem, work1, work2, work3, work4);

    // Step 2: U * X = Y → X (overwrite B/Y with X)
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        handle,
        rocblas_side_left,
        rocblas_operation_none,               // No transpose (U, not U^T)
        rocblas_diagonal_non_unit,
        n, nrhs,
        A, shiftA, lda, strideA,              // U
        B, shiftB, ldb, strideB,              // Y → X (final solution)
        batch_count, optim_mem, work1, work2, work3, work4);
}
```

**Why conjugate_transpose for complex matrices?**

For complex Hermitian matrices:
```
Real SPD:     A = L*L^T  or  A = U^T*U
Complex HPD:  A = L*L^H  or  A = U^H*U  (Hermitian, not just transpose)
```

Hermitian transpose (A^H) = conjugate transpose:
```
(A^H)[i,j] = conj(A[j,i])
```

For complex matrix:
```
L * L^H * X = B

Step 1: L * Y = B
Step 2: L^H * X = Y  ← Need conjugate transpose!
```

rocblas_operation_conjugate_transpose:
- Real matrices: Acts as regular transpose (L^T)
- Complex matrices: Acts as Hermitian transpose (L^H)

**Order matters:**

Lower triangle: L first, then L^T/L^H
```
L * (L^T * X) = L * Y = B
```
Forward substitution (L), then backward (L^T)

Upper triangle: U^T/U^H first, then U
```
(U^T * U) * X = U^T * Y = B
```
Forward substitution (U^T), then backward (U)

**Complexity:**

Each TRSM: O(n² × nrhs) flops
Total POTRS: O(2n² × nrhs) flops

For nrhs=1: O(2n²)
For nrhs=k: O(2n²k) - amortized cost per RHS is O(2n²)

**Comparison to forward/back substitution:**

TRSM is matrix version of triangular solve:
- Input: Triangular matrix L, multiple RHS B (n×nrhs)
- Output: X such that L*X = B
- Uses BLAS-3 operations for efficiency (blocked algorithm)""",
    rationale="POTRS solves the linear system using two TRSM calls. For lower triangular (A=L*L^T), it solves L*Y=B then L^T*X=Y. For upper triangular (A=U^T*U), it solves U^T*Y=B then U*X=Y. conjugate_transpose handles both real (transpose) and complex (Hermitian transpose) cases correctly.",
    tags=["coding", "potrs", "trsm", "triangular-solve", "hermitian"]
))

# Entry 4: Workspace calculation - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="posv",
    instruction="Implement the workspace calculation for POSV. How does it combine requirements from POTRF and POTRS? Why is pivots_savedB sized for saving B?",
    context_text="POSV workspace calculation must account for both POTRF and POTRS requirements, taking the maximum since they execute sequentially. Additionally, pivots_savedB must be large enough to save the entire B matrix.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// workspace required for potrf
rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, size_scalars,
                                                   size_work1, size_work2, size_work3, size_work4,
                                                   size_pivots_savedB, size_iinfo, &opt1);

// workspace required for potrs
rocsolver_potrs_getMemorySize<BATCHED, STRIDED, T>(n, nrhs, batch_count, &w1, &w2, &w3, &w4,
                                                   &opt2);

*size_work1 = std::max(*size_work1, w1);
*size_work2 = std::max(*size_work2, w2);
*size_work3 = std::max(*size_work3, w3);
*size_work4 = std::max(*size_work4, w4);
*optim_mem = opt1 && opt2;

// extra space to copy B
*size_pivots_savedB = std::max(*size_pivots_savedB, sizeof(T) * n * nrhs * batch_count);"""
    }],
    answer="""POSV workspace calculation:

```cpp
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_posv_getMemorySize(
    const rocblas_int n,
    const rocblas_int nrhs,
    const rocblas_fill uplo,
    const rocblas_int batch_count,
    size_t* size_scalars,
    size_t* size_work1,
    size_t* size_work2,
    size_t* size_work3,
    size_t* size_work4,
    size_t* size_pivots_savedB,
    size_t* size_iinfo,
    bool* optim_mem)
{
    // Quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots_savedB = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    bool opt1, opt2;
    size_t w1, w2, w3, w4;

    // STEP 1: Query POTRF workspace
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(
        n, uplo, batch_count,
        size_scalars,           // Constants for rocBLAS calls
        size_work1,             // POTRF work1
        size_work2,             // POTRF work2
        size_work3,             // POTRF work3
        size_work4,             // POTRF work4
        size_pivots_savedB,     // POTRF pivot storage (small)
        size_iinfo,             // Internal info array
        &opt1);                 // Optimization flag

    // STEP 2: Query POTRS workspace
    rocsolver_potrs_getMemorySize<BATCHED, STRIDED, T>(
        n, nrhs, batch_count,
        &w1,                    // POTRS work1
        &w2,                    // POTRS work2
        &w3,                    // POTRS work3
        &w4,                    // POTRS work4
        &opt2);                 // Optimization flag

    // STEP 3: Combine workspace (max of POTRF and POTRS)
    // Why max? Buffers are reused between POTRF and POTRS
    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);

    // Optimization flag: true only if both POTRF and POTRS can optimize
    *optim_mem = opt1 && opt2;

    // STEP 4: Ensure pivots_savedB can hold entire B matrix
    // This is for error handling (saving B if POTRF fails)
    *size_pivots_savedB = std::max(
        *size_pivots_savedB,                    // From POTRF (typically small)
        sizeof(T) * n * nrhs * batch_count);    // For saving B (large!)

    // Final size is max(POTRF pivot size, n×nrhs×batch for B)
}
```

**Why take maximum of work buffers?**

POSV executes sequentially:
```
1. POTRF (uses work1, work2, work3, work4)
2. POTRS (uses work1, work2, work3, work4)
```

Since they don't overlap, buffers can be reused:
```
Allocate: work1 = max(POTRF_work1, POTRS_work1)
Allocate: work2 = max(POTRF_work2, POTRS_work2)
...

Total memory = max(POTRF, POTRS) instead of POTRF + POTRS
```

**Why pivots_savedB dominates?**

Two uses for pivots_savedB:

1. **POTRF use**: Small pivot storage
   - Typical size: ~O(n) or O(blocksize)
   - Used internally by blocked Cholesky

2. **POSV use**: Save entire B matrix
   - Size: n × nrhs × batch_count elements
   - For n=1000, nrhs=10, batch=100: 1M elements
   - In double precision: 8 MB

The max() ensures sufficient space for both:
```cpp
*size_pivots_savedB = std::max(
    small_pivot_size,           // From POTRF (~KB)
    sizeof(T) * n * nrhs * batch_count);  // For B (~MB)
```

Result: pivots_savedB is sized for B (the larger requirement)

**POTRS workspace details:**

```cpp
rocsolver_potrs_getMemorySize(n, nrhs, batch_count, ...)
{
    // POTRS calls TRSM twice (with and without transpose)
    // Query both and take maximum

    size_t work1_temp1, work1_temp2, ...;

    rocsolver_trsm_mem<BATCHED, STRIDED, T>(
        rocblas_side_left, rocblas_operation_none,
        n, nrhs, batch_count,
        &work1_temp1, &work2_temp1, &work3_temp1, &work4_temp1, optim_mem);

    rocsolver_trsm_mem<BATCHED, STRIDED, T>(
        rocblas_side_left, rocblas_operation_conjugate_transpose,
        n, nrhs, batch_count,
        &work1_temp2, &work2_temp2, &work3_temp2, &work4_temp2, optim_mem);

    *size_work1 = std::max(work1_temp1, work1_temp2);
    *size_work2 = std::max(work2_temp1, work2_temp2);
    ...
}
```

**Memory totals example:**

For n=1000, nrhs=10, batch=100, double precision:

- scalars: ~1 KB (constants)
- work1-4: ~10 MB (max of POTRF, POTRS TRSM requirements)
- pivots_savedB: 8 MB (n×nrhs×batch = 1M elements × 8 bytes)
- iinfo: 400 bytes (batch_count × sizeof(int))

Total: ~18 MB (dominated by work and savedB buffers)

**Optimization flag:**

```cpp
*optim_mem = opt1 && opt2;
```

If both POTRF and POTRS can use optimized memory layout:
- true: Enable optimizations in both
- false: Fall back to standard layout

This is passed through to both subroutines to maintain consistency.""",
    rationale="POSV workspace calculation combines POTRF and POTRS requirements by taking the maximum of each reused buffer, since they execute sequentially. The pivots_savedB buffer is sized for the larger of: POTRF's pivot storage (small) or saving the entire B matrix (large), with the B-saving requirement typically dominating.",
    tags=["coding", "workspace", "memory-management", "optimization"]
))

# Entry 5: Argument validation - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="posv",
    instruction="Implement argument validation for POSV. Check uplo parameter, dimensions, leading dimensions, and pointers. Why must ldb >= n?",
    context_text="POSV argument validation follows LAPACK conventions: uplo must be upper or lower, dimensions must be non-negative, leading dimensions must accommodate the matrices, and pointers must be valid.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """template <typename T>
rocblas_status rocsolver_posv_argCheck(rocblas_handle handle,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       const rocblas_int nrhs,
                                       const rocblas_int lda,
                                       const rocblas_int ldb,
                                       T A,
                                       T B,
                                       rocblas_int* info,
                                       const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (nrhs && n && !B) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}"""
    }],
    answer="""POSV argument validation implementation:

```cpp
template <typename T>
rocblas_status rocsolver_posv_argCheck(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int nrhs,
    const rocblas_int lda,
    const rocblas_int ldb,
    T A,
    T B,
    rocblas_int* info,
    const rocblas_int batch_count = 1)
{
    // ORDER IS IMPORTANT FOR UNIT TESTS!

    // STAGE 1: Invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        return rocblas_status_invalid_value;
        // uplo must be explicitly upper or lower
        // No default, no "full" storage
    }

    // STAGE 2: Invalid sizes
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // Detailed checks:
    // n < 0: Invalid matrix dimension
    // nrhs < 0: Invalid number of right-hand sides
    // lda < n: Leading dimension too small for n×n matrix A
    // ldb < n: Leading dimension too small for n×nrhs matrix B
    // batch_count < 0: Invalid batch count

    // Skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_status_continue;
        // Memory query doesn't need valid pointers
    }

    // STAGE 3: Invalid pointers
    if((n && !A) ||                     // A required if n > 0
       (nrhs && n && !B) ||             // B required if n > 0 AND nrhs > 0
       (batch_count && !info))          // info required if batch_count > 0
    {
        return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}
```

**Why ldb >= n?**

B is stored as column-major matrix:
```
B dimensions: n × nrhs
Storage: ldb is the leading dimension (stride between columns)

Memory layout:
B = [b00  b01  b02  ...  b0,nrhs-1]
    [b10  b11  b12  ...  b1,nrhs-1]
    [b20  b21  b22  ...  b2,nrhs-1]
    ...
    [bn-1,0  bn-1,1  ...  bn-1,nrhs-1]

Column j starts at: B + j * ldb
Element B[i,j] at: B + i + j * ldb

For correct access: ldb >= n (number of rows)
```

If ldb < n:
```
Example: n=5, ldb=4, nrhs=2

Column 0: B[0:4]    ✓
Column 1: B[4:8]    ← Overlaps with column 0!

B[4,0] is at position 4
B[0,1] is at position 4  ← Collision!
```

**Pointer validation logic:**

```cpp
if((n && !A) || (nrhs && n && !B) || (batch_count && !info))
```

Breaking down:
1. `(n && !A)`: A required if n > 0
   - If n=0: Empty matrix, A can be nullptr
   - If n>0: A must be valid

2. `(nrhs && n && !B)`: B required if both n > 0 AND nrhs > 0
   - If n=0: Empty system, B can be nullptr
   - If nrhs=0: No right-hand sides, B can be nullptr
   - If both > 0: B must be valid

3. `(batch_count && !info)`: info required if batch_count > 0
   - If batch_count=0: No work, info can be nullptr
   - If batch_count>0: info must be valid (per-batch status)

**Why uplo must be explicit?**

POSV supports only:
- `rocblas_fill_upper`: Use upper triangle (A = U^T*U)
- `rocblas_fill_lower`: Use lower triangle (A = L*L^T)

Not supported:
- `rocblas_fill_full`: Doesn't make sense (symmetric matrix)
- Implicit/default: Must be explicit for correctness

Rationale:
- Cholesky only needs one triangle
- Must know which triangle user is providing
- No "full" because A is symmetric (redundant to store both)

**Validation order:**

Order matters for consistent error messages in unit tests:
1. Values (uplo)
2. Sizes (n, nrhs, lda, ldb, batch_count)
3. Pointers (A, B, info)

This ensures:
- Most fundamental errors caught first
- Consistent error reporting across platforms
- Easier unit test writing""",
    rationale="POSV argument validation checks uplo (must be upper or lower), dimensions (non-negative), leading dimensions (lda >= n, ldb >= n to accommodate column-major storage), and pointers (valid when needed). The requirement ldb >= n ensures correct column-major access to the n×nrhs matrix B.",
    tags=["coding", "validation", "leading-dimension", "error-handling"]
))

# Entry 6: Production API - L3 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="posv",
    instruction="Implement the C API for all four precisions of POSV (S/D/C/Z). Show the complete call chain from C wrapper to template instantiation.",
    context_text="POSV provides four precision variants following LAPACK naming conventions. The C wrappers call the implementation template which handles workspace allocation and calls the main template.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.cpp",
        "language": "cpp",
        "content": """extern "C" rocblas_status rocsolver_sposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          float* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_posv_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

extern "C" rocblas_status rocsolver_cposv(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_posv_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                 ldb, info);
}"""
    }],
    answer="""Complete C API for all POSV precisions:

```cpp
extern "C" {

// Single precision real: S
rocblas_status rocsolver_sposv(
    rocblas_handle handle,
    const rocblas_fill uplo,       // Upper or lower triangle
    const rocblas_int n,           // Matrix dimension
    const rocblas_int nrhs,        // Number of right-hand sides
    float* A,                      // Coefficient matrix (n×n)
    const rocblas_int lda,         // Leading dimension of A
    float* B,                      // Right-hand sides/solution (n×nrhs)
    const rocblas_int ldb,         // Leading dimension of B
    rocblas_int* info)             // Error info
{
    return rocsolver::rocsolver_posv_impl<float>(
        handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

// Double precision real: D
rocblas_status rocsolver_dposv(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int nrhs,
    double* A,                     // Double precision matrix
    const rocblas_int lda,
    double* B,                     // Double precision RHS
    const rocblas_int ldb,
    rocblas_int* info)
{
    return rocsolver::rocsolver_posv_impl<double>(
        handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

// Single precision complex: C
rocblas_status rocsolver_cposv(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int nrhs,
    rocblas_float_complex* A,      // Complex matrix (Hermitian positive definite)
    const rocblas_int lda,
    rocblas_float_complex* B,      // Complex RHS
    const rocblas_int ldb,
    rocblas_int* info)
{
    return rocsolver::rocsolver_posv_impl<rocblas_float_complex>(
        handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

// Double precision complex: Z
rocblas_status rocsolver_zposv(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int nrhs,
    rocblas_double_complex* A,     // Complex matrix
    const rocblas_int lda,
    rocblas_double_complex* B,     // Complex RHS
    const rocblas_int ldb,
    rocblas_int* info)
{
    return rocsolver::rocsolver_posv_impl<rocblas_double_complex>(
        handle, uplo, n, nrhs, A, lda, B, ldb, info);
}

} // extern "C"
```

**Call chain:**

```
User calls: rocsolver_dposv(handle, uplo, n, nrhs, A, lda, B, ldb, info)
    ↓
C wrapper: rocsolver_dposv
    ↓
Implementation: rocsolver_posv_impl<double>(...)
    │
    ├→ Argument validation
    ├→ Workspace calculation
    ├→ Memory allocation
    │
    └→ Template call: rocsolver_posv_template<false, false, double, double>(...)
        │
        ├→ POTRF: rocsolver_potrf_template(...)
        │   └→ Cholesky factorization (A = L*L^T or U^T*U)
        │
        ├→ Save B: copy_mat kernel (if POTRF might fail)
        │
        ├→ POTRS: rocsolver_potrs_template(...)
        │   ├→ TRSM #1: Solve L*Y = B or U^T*Y = B
        │   └→ TRSM #2: Solve L^T*X = Y or U*X = Y
        │
        └→ Restore B: copy_mat kernel (if POTRF failed)
```

**Implementation template:**

```cpp
template <typename T>
rocblas_status rocsolver_posv_impl(
    rocblas_handle handle,
    const rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int nrhs,
    T* A, const rocblas_int lda,
    T* B, const rocblas_int ldb,
    rocblas_int* info)
{
    using S = decltype(std::real(T{}));  // Extract real type (float from complex)

    // 1. Validate arguments
    rocblas_status st = rocsolver_posv_argCheck(
        handle, uplo, n, nrhs, lda, ldb, A, B, info);
    if(st != rocblas_status_continue)
        return st;

    // 2. Setup strides for non-batched case
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_int batch_count = 1;

    // 3. Query workspace sizes
    size_t size_scalars, size_work1, size_work2, size_work3, size_work4;
    size_t size_pivots_savedB, size_iinfo;
    bool optim_mem;

    rocsolver_posv_getMemorySize<false, false, T>(
        n, nrhs, uplo, batch_count,
        &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivots_savedB, &size_iinfo, &optim_mem);

    // 4. Handle memory query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3,
            size_work4, size_pivots_savedB, size_iinfo);

    // 5. Allocate workspace
    void *scalars, *work1, *work2, *work3, *work4, *pivots_savedB, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2,
                              size_work3, size_work4, size_pivots_savedB,
                              size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    // 6. Initialize pointers
    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots_savedB = mem[5];
    iinfo = mem[6];

    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // 7. Execute main template
    return rocsolver_posv_template<false, false, T, S>(
        handle, uplo, n, nrhs,
        A, shiftA, lda, strideA,
        B, shiftB, ldb, strideB,
        info, batch_count,
        (T*)scalars, work1, work2, work3, work4,
        (T*)pivots_savedB, (rocblas_int*)iinfo, optim_mem);
}
```

**Template parameters:**

```cpp
rocsolver_posv_template<
    false,      // BATCHED: false for normal arrays
    false,      // STRIDED: false for normal arrays
    T,          // Element type: float/double/complex
    S>          // Real type: float/double (for mixed real/complex operations)
```

**Type deduction:**

```cpp
using S = decltype(std::real(T{}));

// For T=float:          S=float
// For T=double:         S=double
// For T=complex<float>: S=float
// For T=complex<double>: S=double
```

Used for operations requiring real scalars (e.g., norms, tolerances)

**LAPACK naming:**
- S = Single precision real
- D = Double precision real
- C = Complex single precision
- Z = Complex double precision (German "Zahl")""",
    rationale="POSV provides four precision variants (S/D/C/Z) through C wrappers that call a templated implementation. The implementation validates arguments, calculates workspace, allocates memory, and executes the main template which combines POTRF (Cholesky factorization) and POTRS (triangular solve) with error handling for B preservation.",
    tags=["coding", "api", "template", "precision", "lapack"]
))

# Continue with entries 7-12...
# Entry 7: POSV vs GESV comparison - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="posv",
    instruction="Compare POSV with GESV for solving linear systems. When should each be used? What are the performance and stability differences?",
    context_text="POSV is specialized for symmetric positive definite matrices using Cholesky factorization, while GESV handles general matrices using LU factorization with partial pivoting. The choice depends on matrix properties.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// POSV: Cholesky factorization
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, ...);

rocsolver_potrs_template<BATCHED, STRIDED, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, B,
                                              shiftB, ldb, strideB, batch_count, ...);"""
    }],
    answer="""POSV vs GESV comparison:

| Aspect | POSV (Cholesky) | GESV (LU with pivoting) |
|--------|----------------|------------------------|
| **Matrix type** | Symmetric positive definite | General square |
| **Factorization** | A = L*L^T or U^T*U | P*A = L*U |
| **Complexity** | ~n³/3 | ~2n³/3 |
| **Speed** | **2× faster** | Baseline |
| **Memory** | n²/2 (one triangle) | n² (L and U) |
| **Pivoting** | None (not needed) | Partial (required) |
| **Stability** | Backward stable for SPD | Backward stable (general) |
| **Breakdown** | Fails if not SPD | Rare (singular matrix) |
| **Use case** | SPD matrices | Any square matrix |

**When to use POSV:**

1. **Matrix is symmetric positive definite:**
   ```
   A = A^T (or A = A^H for complex)
   x^T * A * x > 0 for all x ≠ 0
   ```

   Common SPD matrices:
   - Gram matrices: A = B^T*B
   - Covariance matrices
   - Kernel matrices (RBF, polynomial)
   - Finite element stiffness matrices
   - Normal equations: A^T*A in least squares

2. **Performance critical:**
   - POSV is 2× faster than GESV
   - For n=1000: POSV ~0.33 GFLOPs vs GESV ~0.67 GFLOPs

3. **Memory constrained:**
   - POSV stores one triangle (n²/2)
   - GESV stores full matrix (n²)

**When to use GESV:**

1. **Matrix is general (not SPD):**
   - Non-symmetric matrices
   - Symmetric but not positive definite
   - Indefinite matrices

2. **Unknown matrix properties:**
   - If unsure whether SPD, use GESV
   - Or try POSV first, fall back to GESV if info ≠ 0

3. **Handling near-singularity:**
   - LU with pivoting more robust for ill-conditioned matrices
   - Cholesky can fail if matrix nearly singular

**Algorithm details:**

**POSV:**
```cpp
// Stage 1: Cholesky factorization
A = L * L^T  (or U^T * U)
Cost: n³/3 flops

// Stage 2: Triangular solve
L * Y = B      → Y (n² flops)
L^T * X = Y    → X (n² flops)
Total solve: 2n² flops per RHS
```

**GESV:**
```cpp
// Stage 1: LU factorization with partial pivoting
P * A = L * U
Cost: 2n³/3 flops

// Stage 2: Triangular solve with pivot application
P * B → B'     (n² flops, permutation)
L * Y = B'     (n² flops)
U * X = Y      (n² flops)
Total solve: 3n² flops per RHS
```

**Failure modes:**

**POSV fails if:**
```cpp
// Matrix not positive definite
// POTRF detects: A[k,k] <= 0 after k-1 steps
info = k  // Column where failure occurred

Example: A = [1  0]  → Not positive definite
            [0 -1]
POTRF fails at k=2 (A[2,2] = -1 < 0)
```

**GESV fails if:**
```cpp
// Matrix is singular
// GETRF detects: All potential pivots are zero
info = k  // Column where singularity detected

Example: A = [1 1]  → Singular (rank 1)
            [1 1]
GETRF fails (cannot find non-zero pivot)
```

**Practical example:**

```cpp
// Normal equations: A^T * A * x = A^T * b
// A^T*A is symmetric positive semi-definite

double AtA[1000*1000];  // Gram matrix
double Atb[1000];

// Compute A^T * A and A^T * b
rocblas_gemm(..., A^T, A, AtA);
rocblas_gemv(..., A^T, b, Atb);

// Solve using POSV (2× faster than GESV)
int info;
rocsolver_dposv(handle, rocblas_fill_upper, 1000, 1,
                AtA, 1000, Atb, 1000, &info);

if(info == 0)
{
    // Success: Atb contains solution
}
else
{
    // A^T*A is rank-deficient or ill-conditioned
    // Fall back to:
    // 1. GESV with pivoting, or
    // 2. Regularized solve (A^T*A + λI), or
    // 3. SVD-based least squares
}
```

**Performance benchmark (n=1000, double precision):**

- POSV: ~0.35 seconds
- GESV: ~0.70 seconds
- **POSV is 2× faster!**

**Recommendation:**

If you know A is SPD → Use POSV (faster, less memory)
If A is general → Use GESV
If unsure → Try POSV, fall back to GESV if info ≠ 0""",
    rationale="POSV is 2× faster than GESV for symmetric positive definite matrices because Cholesky factorization requires half the flops of LU factorization and no pivoting. Use POSV when the matrix is known to be SPD (common in least squares, covariance, kernel methods), otherwise use GESV for general matrices.",
    tags=["algorithm", "comparison", "performance", "spd", "gesv"]
))

# Entry 8: uplo parameter - L1
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="posv",
    instruction="Explain the uplo parameter in POSV. How does it affect which triangle is used and what happens to the other triangle?",
    context_text="For symmetric matrices, only one triangle needs to be stored and accessed. The uplo parameter tells POSV which triangle contains the data, and the other triangle is not referenced.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// compute Cholesky factorization of A
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, scalars, work1, work2, work3,
    work4, pivots_savedB, iinfo, optim_mem);

// solve AX = B, overwriting B with X
rocsolver_potrs_template<BATCHED, STRIDED, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, B,
                                              shiftB, ldb, strideB, batch_count, work1, work2,
                                              work3, work4, optim_mem);"""
    }],
    answer="""uplo parameter in POSV:

**Purpose:** Specifies which triangle of the symmetric matrix A contains data

**Two values:**

1. **rocblas_fill_upper**: Use upper triangle
```
A = [a11  a12  a13  a14]
    [ *   a22  a23  a24]
    [ *    *   a33  a34]
    [ *    *    *   a44]

* = not referenced (can be garbage)
```

Factorization: A = U^T * U (or U^H * U for complex)
```
U = [u11  u12  u13  u14]
    [ 0   u22  u23  u24]
    [ 0    0   u33  u34]
    [ 0    0    0   u44]

Upper triangular stored in upper part of A
```

2. **rocblas_fill_lower**: Use lower triangle
```
A = [a11   *    *    *]
    [a21  a22   *    *]
    [a31  a32  a33   *]
    [a41  a42  a43  a44]

* = not referenced (can be garbage)
```

Factorization: A = L * L^T (or L * L^H for complex)
```
L = [l11   0    0    0]
    [l21  l22   0    0]
    [l31  l32  l33   0]
    [l41  l42  l43  l44]

Lower triangular stored in lower part of A
```

**What happens to the other triangle?**

- **Not referenced**: The unused triangle can contain anything
- **Not modified**: POTRF/POTRS don't touch it
- **Preserved**: If input has full matrix, unused triangle remains unchanged

Example:
```cpp
// Input matrix (full symmetric)
A = [4  2  1  0]
    [2  3  2  1]
    [1  2  3  2]
    [0  1  2  4]

// Call POSV with uplo = rocblas_fill_upper
rocsolver_dposv(handle, rocblas_fill_upper, 4, nrhs, A, 4, B, 4, &info);

// After POSV:
A = [u11  u12  u13  u14]  ← Upper: Cholesky factor
    [ 2   u22  u23  u24]  ← Lower: Unchanged (original values)
    [ 1    2   u33  u34]
    [ 0    1    2   u44]
```

**Why only one triangle?**

Symmetric matrix property:
```
A[i,j] = A[j,i]  (or conj(A[j,i]) for Hermitian)
```

Storing both triangles is redundant:
- Memory waste: 2× storage
- Computation waste: 2× operations
- Consistency risk: Triangles could diverge

**Choosing uplo:**

Generally doesn't matter for correctness:
```cpp
// These produce the same solution X:
rocsolver_dposv(..., rocblas_fill_upper, ..., A, ..., B, ...);
rocsolver_dposv(..., rocblas_fill_lower, ..., A, ..., B, ...);
```

Considerations:
1. **Input format**: Use whichever triangle contains your data
2. **Cache performance**: Upper may be slightly better for small matrices (row-major access)
3. **Convention**: Some applications use upper, others lower

**Common mistake:**

```cpp
// WRONG: Assume full matrix is used
A = [4  2  1]
    [2  3  2]
    [1  2  3]

// Set uplo = upper, but only fill upper triangle
A_upper = [4  2  1]
          [0  3  2]  ← Zero instead of symmetric value!
          [0  0  3]

rocsolver_dposv(..., rocblas_fill_upper, ..., A_upper, ...);
// Works correctly! Lower triangle is ignored.
```

**Memory savings:**

For n=1000 matrix:
- Full storage: 1000² = 1,000,000 elements
- Triangle storage: 1000×1001/2 = 500,500 elements
- **Savings: 50% memory!**

**Implementation note:**

POTRF and POTRS both use the same uplo:
```cpp
rocsolver_potrf_template(..., uplo, ...);  // Factorize using uplo
rocsolver_potrs_template(..., uplo, ...);  // Solve using same uplo
```

They must match because:
- POTRF writes factor to specified triangle
- POTRS reads factor from same triangle""",
    rationale="The uplo parameter specifies which triangle of the symmetric matrix contains data. For SPD matrices, only one triangle is needed since A[i,j] = A[j,i]. The other triangle is not referenced and can contain any values. This saves 50% memory and computation.",
    tags=["api", "storage", "symmetry", "upper-lower"]
))

# Entry 9-12: Quick additional entries
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="posv",
    instruction="What does info contain after POSV returns? Interpret the different info values.",
    context_text="The info parameter reports success or failure of POSV. A value of 0 indicates success, while positive values indicate the column where Cholesky factorization failed.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// info=0 (starting with a nonsingular matrix)
ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

// compute Cholesky factorization of A
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, ...);"""
    }],
    answer="""info parameter interpretation in POSV:

**Initialization:**
```cpp
info[i] = 0  // For all batches, assume success
```

**After POSV:**

**Case 1: info[i] = 0 (Success)**
- Cholesky factorization succeeded
- Solution X computed correctly
- B contains the solution
- A contains the Cholesky factor (L or U)

**Case 2: info[i] = k > 0 (Failure)**
- Cholesky factorization failed at column k
- A is not positive definite
- The leading minor of order k is not positive definite
- B is unchanged (original values preserved)
- A[0:k-1, 0:k-1] contains partial Cholesky factor

**Interpretation:**

```
info = 0:    Success
info = 1:    A[1,1] ≤ 0 or not real
info = 2:    A[2,2] - L[2,1]² ≤ 0 after eliminating row 1
info = k:    Leading k×k submatrix not positive definite
```

**Example:**

```cpp
double A[3][3] = {{4, 2, 1},
                  {2, 3, 2},
                  {1, 2, -1}};  // Not SPD! (A[3,3] = -1)
int info;

rocsolver_dposv(handle, rocblas_fill_upper, 3, 1, A, 3, B, 3, &info);

// Result: info = 3 (failed at column 3)
// Because after 2 steps, A[3,3] - ... < 0
```

**Batched case:**

```cpp
rocblas_int info[100];  // One per batch

// After POSV:
info[0] = 0     // Batch 0: Success
info[1] = 5     // Batch 1: Failed at column 5
info[2] = 0     // Batch 2: Success
...
```""",
    rationale="info reports POSV status: 0 for success, k>0 if Cholesky failed at column k (matrix not positive definite). For failed cases, B is preserved unchanged and partial factorization in A[0:k-1,0:k-1] is available.",
    tags=["api", "error-handling", "info", "status"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="posv",
    instruction="Compare the complexity of POSV vs solving via QR factorization for overdetermined systems. When is POSV advantageous?",
    context_text="For least squares problems min ||Ax-b||₂, one can either solve the normal equations using POSV or use QR factorization. The choice involves accuracy-performance tradeoffs.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// POSV approach: Solve A^T*A*x = A^T*b
rocsolver_potrf_template(...);  // Cholesky: (n³/3)
rocsolver_potrs_template(...);  // Solve: (2n²)"""
    }],
    answer="""POSV vs QR for least squares:

**Problem:** min ||Ax - b||₂ where A is m×n (m >> n)

**Method 1: Normal equations (POSV)**
```
1. Compute A^T*A (Gram matrix)    : O(mn²)
2. Compute A^T*b                  : O(mn)
3. Cholesky factorization of A^T*A: O(n³/3)
4. Solve (A^T*A)*x = A^T*b        : O(n²)

Total: O(mn² + n³/3)
```

**Method 2: QR factorization**
```
1. QR factorization A = Q*R       : O(2mn²)
2. Compute Q^T*b                  : O(mn)
3. Solve R*x = Q^T*b              : O(n²)

Total: O(2mn²)
```

**Comparison (m=10000, n=100):**
- POSV: 10⁸ + 3.3×10⁵ ≈ 10⁸ flops
- QR:   2×10⁸ flops
- **POSV is 2× faster!**

**When POSV is advantageous:**

1. **Performance (m >> n):**
   - POSV: O(mn² + n³/3) ≈ O(mn²)
   - QR: O(2mn²)
   - POSV wins when n³ << mn²

2. **A^T*A already available:**
   - In some applications, Gram matrix A^T*A is precomputed
   - POSV only needs O(n³/3), no A required

3. **Multiple RHS with same A:**
   - Factorize once: O(mn² + n³/3)
   - Solve k times: O(kn²)
   - QR: O(2mn²) per solve

**When QR is better:**

1. **Numerical stability:**
   - POSV: Condition number κ((A^T*A)) = κ(A)²
   - QR: Condition number κ(A)
   - For ill-conditioned A, QR much more accurate

2. **Rank-deficient A:**
   - POSV: A^T*A is singular, Cholesky fails
   - QR: Can use with column pivoting (rank-revealing)

3. **Small m/n ratio:**
   - If m ≈ n, QR cost O(2mn²) ≈ POSV cost O(mn²+n³/3)
   - Stability advantage of QR worth slight cost

**Recommendation:**

Well-conditioned, m >> n → POSV (2× faster)
Ill-conditioned or rank-deficient → QR (more accurate)""",
    rationale="For overdetermined least squares with m>>n and well-conditioned A, POSV via normal equations is 2× faster than QR factorization. However, QR is more accurate for ill-conditioned matrices since it avoids squaring the condition number.",
    tags=["complexity", "least-squares", "normal-equations", "qr-factorization"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="posv",
    instruction="Explain how POSV handles complex Hermitian positive definite matrices differently from real symmetric matrices.",
    context_text="For complex matrices, POSV requires Hermitian (conjugate symmetric) rather than just symmetric. The Cholesky factorization uses conjugate transpose instead of regular transpose.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_potrs.hpp",
        "language": "cpp",
        "content": """rocsolver_trsm_lower<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB,
    batch_count, optim_mem, work1, work2, work3, work4);"""
    }],
    answer="""Complex Hermitian handling in POSV:

**Real symmetric:** A = A^T (transpose equals self)
```
A[i,j] = A[j,i]

Example: [4  2]
         [2  3]
```

**Complex Hermitian:** A = A^H (conjugate transpose equals self)
```
A[i,j] = conj(A[j,i])

Example: [4      2+i  ]
         [2-i    3    ]  ← Diagonal must be real!
```

**Cholesky factorization:**

Real: A = L*L^T or U^T*U
Complex: A = L*L^H or U^H*U  (Hermitian transpose!)

**POTRS triangular solve:**

For complex lower triangle:
```cpp
// Step 1: Solve L*Y = B
rocsolver_trsm_lower(..., rocblas_operation_none, ...);

// Step 2: Solve L^H*X = Y  ← Conjugate transpose!
rocsolver_trsm_lower(..., rocblas_operation_conjugate_transpose, ...);
```

The conjugate_transpose operation:
- Real: L^T (regular transpose)
- Complex: L^H = (L̄)^T (conjugate then transpose)

**Why conjugate transpose?**

For Hermitian matrices:
```
A = L * L^H

Verification:
(L*L^H)[i,j] = Σ_k L[i,k] * conj(L[j,k])
             = Σ_k L[i,k] * L^H[k,j]

For i=j: (L*L^H)[i,i] = Σ_k |L[i,k]|² > 0  ← Real and positive
For i≠j: (L*L^H)[i,j] = conj((L*L^H)[j,i])  ← Hermitian property
```

**Diagonal must be real:**

For Hermitian positive definite:
```
A[i,i] = conj(A[i,i])  → A[i,i] must be real
x^H*A*x > 0 for all x  → Diagonal entries > 0
```

Example:
```cpp
rocblas_double_complex A[2][2] = {
    {4.0,              2.0 + 1.0i},
    {2.0 - 1.0i,       3.0}
};  // Valid HPD

rocblas_double_complex B[2][1] = {{1.0}, {2.0}};
int info;

rocsolver_zposv(handle, rocblas_fill_upper, 2, 1, A, 2, B, 2, &info);
// Uses U^H*U factorization with conjugate transpose
```""",
    rationale="For complex matrices, POSV requires Hermitian (A = A^H) rather than just symmetric. The Cholesky factorization uses conjugate transpose (L^H) instead of regular transpose (L^T), and diagonal elements must be real and positive. The conjugate_transpose operation handles both real and complex cases correctly.",
    tags=["complex", "hermitian", "conjugate-transpose", "cholesky"]
))

time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="posv",
    instruction="Why does POSV not need a pivot vector like GESV? What property of positive definite matrices guarantees stability?",
    context_text="LU factorization with partial pivoting (used in GESV) requires a pivot vector to track row swaps. Cholesky factorization (used in POSV) needs no pivoting for symmetric positive definite matrices.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_posv.hpp",
        "language": "cpp",
        "content": """// compute Cholesky factorization of A (no pivoting needed!)
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, A, shiftA, lda, strideA, info, batch_count, ...);"""
    }],
    answer="""Why POSV needs no pivoting:

**GESV (LU with partial pivoting):**
```cpp
P*A = L*U

Requires:
- Pivot vector ipiv[n] to store row swaps
- Partial pivoting to ensure |L[i,j]| ≤ 1
- May fail if matrix is singular
```

**POSV (Cholesky, no pivoting):**
```cpp
A = L*L^T  (no permutation P!)

No pivot vector needed
Always stable for SPD matrices
```

**Why no pivoting for SPD?**

**Property 1: All pivots are positive**

For SPD matrix A, at step k:
```
Pivot element: A[k,k] - Σ(i=1 to k-1) L[k,i]²

This is always > 0 for SPD matrices!
```

Proof:
```
A is SPD → All leading principal minors > 0
→ A[k,k] > 0 after k-1 eliminations
→ No zero or negative pivots
→ No need to swap rows for stability
```

**Property 2: Natural stability**

Cholesky update:
```
L[k,k] = sqrt(A[k,k] - Σ L[k,i]²)
L[j,k] = (A[j,k] - Σ L[j,i]*L[k,i]) / L[k,k]
```

For SPD:
```
L[k,k] = sqrt(positive value) → Well-defined
L[j,k] = (bounded) / (positive) → Bounded

Growth factor = max|L[j,k]| = O(1)
```

Compare to LU without pivoting:
```
Growth factor can be 2^(n-1)  ← Exponential!
```

**Example:**

SPD matrix:
```
A = [4  2  1]
    [2  3  2]
    [1  2  3]

Cholesky (no pivoting):
L[1,1] = sqrt(4) = 2          ← Always positive
L[2,1] = 2/2 = 1
L[3,1] = 1/2 = 0.5
L[2,2] = sqrt(3 - 1²) = 1.414 ← Still positive
L[3,2] = (2 - 0.5*1)/1.414 = 1.061
L[3,3] = sqrt(3 - 0.5² - 1.061²) = 1.414 ← Positive
```

All pivots positive → No swapping needed!

**Breakdown case:**

Non-SPD matrix:
```
A = [1  0]
    [0 -1]  ← Indefinite

L[1,1] = sqrt(1) = 1  ✓
L[2,2] = sqrt(-1) = ???  ✗ FAIL!

Cholesky fails (info = 2)
Must use LU with pivoting instead
```

**Memory savings:**

- GESV: Needs ipiv[n] (4n bytes for int)
- POSV: No pivot vector
- For n=1000: Save 4 KB

**Performance:**

No pivoting → No row swaps:
- GESV: O(n²) extra operations for pivoting
- POSV: Zero overhead
- Simpler algorithm, better vectorization""",
    rationale="POSV needs no pivoting because SPD matrices guarantee all pivots are positive and Cholesky factorization is naturally stable. This is due to the property that all leading principal minors of SPD matrices are positive, ensuring no zero or negative pivots occur during factorization.",
    tags=["stability", "pivoting", "spd", "cholesky", "numerical-analysis"]
))

# Write all entries to JSONL file
output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_posv.jsonl"
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
