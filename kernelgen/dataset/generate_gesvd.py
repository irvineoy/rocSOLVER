#!/usr/bin/env python3
"""Generate GESVD dataset entries following DATASET.md schema."""

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

# Entry 1: GESVD algorithm overview - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesvd",
    instruction="Explain the complete GESVD algorithm using bidiagonalization and QR iteration. What are the main stages and how does it differ from GESDD? Why is GESVD more accurate for ill-conditioned matrices?",
    context_text="GESVD computes SVD using bidiagonalization followed by QR iteration on the bidiagonal form. Unlike GESDD which uses eigendecomposition of A^*A (squaring condition number), GESVD works directly on A through bidiagonal reduction, maintaining numerical stability.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """// STAGE 3: Bidiagonalization
rocsolver_gebrd_template<BATCHED, STRIDED>(
    handle, m, n, A, shiftA, lda, strideA, S, strideS, E, strideE, tau_splits, k,
    (tau_splits + k * batch_count), k, Abyx_norms_trfact_X, shiftX, ldx, strideX,
    diag_tmptr_Y, shiftY, ldy, strideY, batch_count, scalars, work_workArr,
    Abyx_norms_tmptr_cmplt);

// STAGE 4: generate orthonormal/unitary matrices from bidiagonalization
if(leftvS || leftvA)
{
    rocsolver_orgbr_ungbr_template<false, STRIDED>(
        handle, rocblas_column_wise, m, mn, n, U, shiftU, ldu, strideU, tau_splits, k,
        batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X,
        workArr);
}

// STAGE 5: Compute singular values and vectors from the bidiagonal form
rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                            shiftV, ldv, strideV, U, shiftU, ldu, strideU, (T*)nullptr,
                            0, 1, 1, info, batch_count, (rocblas_int*)tau_splits,
                            (TT*)work_workArr, (rocblas_int*)Abyx_norms_tmptr_cmplt);"""
    }],
    answer="""GESVD algorithm for computing SVD via bidiagonalization:

**Main Stages:**

1. **Row/Column Compression** (optional, for thin SVD):
   - If m >> n: QR factorization A = Q*R
   - If n >> m: LQ factorization A = L*Q
   - Reduces to min(m,n) × min(m,n) problem

2. **Bidiagonalization** (GEBRD):
   - Reduce A (or R/L) to bidiagonal form B
   - A = U_bd * B * V_bd^* where B is bidiagonal
   - Uses Householder reflectors from both sides

3. **Generate Orthogonal Matrices** (ORGBR):
   - Build explicit U and V from Householder reflectors
   - U from column-wise reflectors (left side)
   - V from row-wise reflectors (right side)

4. **Bidiagonal SVD** (BDSQR):
   - Apply QR iteration to B: B = U_bdsqr * Σ * V_bdsqr^*
   - Compute singular values in Σ
   - Update U and V with rotations

5. **Compose Results**:
   - Final U = (Q if compressed) * U_bd * U_bdsqr
   - Final V = (Q if compressed) * V_bd * V_bdsqr
   - S contains singular values in descending order

**GESVD vs GESDD:**

| Aspect | GESVD | GESDD |
|--------|-------|-------|
| Method | Bidiagonalization + QR iteration | Eigendecomposition (A^*A) + D&C |
| Complexity | O(mn² + n³) | O(mn² + n³) (similar) |
| Speed | Slower (QR iteration) | Faster (divide-conquer) |
| Accuracy | Better (no squaring) | Good (squares cond. number) |
| Condition | Works on κ(A) | Works on κ(A²) = κ(A)² |
| Use case | Ill-conditioned matrices | Well-conditioned matrices |

**Why GESVD is more accurate:**

1. **No condition number squaring**:
   - GESVD works on A directly
   - GESDD computes A^*A with condition number κ(A)²
   - For κ(A) = 10^8, GESDD sees κ(A²) = 10^16 (near machine precision!)

2. **Backward stability**:
   - Bidiagonalization is backward stable
   - QR iteration has strong convergence guarantees

3. **Orthogonal transformations**:
   - All operations preserve singular values theoretically
   - Accumulation of rotations is numerically stable

**When to use GESVD:**
- Matrices with condition number > 10^8
- Need high accuracy
- Ill-conditioned or rank-deficient problems""",
    rationale="GESVD uses bidiagonalization and QR iteration to compute SVD without squaring the condition number. This makes it more accurate than GESDD for ill-conditioned matrices, despite being slower due to QR iteration instead of divide-and-conquer.",
    tags=["algorithm", "svd", "bidiagonalization", "qr-iteration", "numerical-stability"]
))

# Entry 2: Thin SVD optimization - L2
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Explain the thin SVD optimization in GESVD. When is it triggered, and what are the fast_thinSVD vs in-place thinSVD algorithms?",
    context_text="For matrices with sufficiently more rows than columns (or vice versa), GESVD can apply a preliminary QR or LQ factorization to reduce the problem size. The thinSVD path has two variants: fast (out-of-place) and in-place.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

if(thinSVD)
{
    //*** STAGE 1: Row (or column) compression ***//
    local_geqrlq_template<BATCHED, STRIDED>(
        handle, m, n, A, shiftA, lda, strideA, tau_splits, k, batch_count, scalars,
        work_workArr, Abyx_norms_trfact_X, diag_tmptr_Y, workArr, row);

    // ... generate Q/L matrices ...

    //*** STAGE 3: Bidiagonalization ***//
    // Work on k×k triangular factor instead of full m×n matrix
    rocsolver_gebrd_template<BATCHED, STRIDED>(
        handle, k, k, A, shiftA, lda, strideA, S, strideS, E, strideE, ...);
}"""
    }],
    answer="""Thin SVD optimization in GESVD:

**When is thin SVD triggered?**

```cpp
const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
```

- Activated when matrix is sufficiently "rectangular"
- THIN_SVD_SWITCH is a threshold (typically 1.6 or 2.0)
- Examples:
  - m=1000, n=100: ratio 10 >> threshold ✓
  - m=500, n=400: ratio 1.25 < threshold ✗

**Thin SVD algorithm:**

**Stage 1: Compression**
- If m >= n: QR factorization A = Q*R (m×n = m×n × n×n)
- If m < n: LQ factorization A = L*Q (m×n = m×m × m×n)
- Reduces problem from m×n to min(m,n)×min(m,n)

**Stage 2: Bidiagonalize compressed matrix**
- Work on k×k triangular factor (k = min(m,n))
- Much cheaper: O(k³) vs O(mn·min(m,n))

**Stage 3: Compose final result**
- Multiply Q/L with bidiagonal SVD results

**Two variants:**

**1. Fast (out-of-place) thin SVD** (`fast_alg = rocblas_outofplace`):
```cpp
const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

// Uses temporary buffers:
*size_tempArrayT = sizeof(T) * k * k * batch_count;  // Triangular factor
*size_tempArrayC = sizeof(T) * m * n * batch_count;  // Copy of A
```

Advantages:
- Preserves original A in some svect modes
- Cleaner algorithm flow
- Easier debugging

Disadvantages:
- Extra memory: O(mn + k²)
- Memory copies

**2. In-place thin SVD** (`fast_alg = rocblas_inplace`):
```cpp
// Reuse A buffer for intermediate results
// No tempArrayC allocation
```

Advantages:
- Memory efficient
- Fewer copies

Disadvantages:
- More complex buffer management
- A may be overwritten

**Complexity comparison:**

Normal SVD (m >> n, say m=10000, n=100):
- Bidiagonalization: O(10000 × 100²) = O(10^8)
- BDSQR: O(100³) = O(10^6)

Thin SVD:
- QR: O(10000 × 100²) = O(10^8)
- Bidiagonalization: O(100³) = O(10^6) ← Much smaller!
- BDSQR: O(100³) = O(10^6)

**Savings:** Bidiagonalization cost reduced by factor of m/n = 100×""",
    rationale="Thin SVD uses preliminary QR/LQ factorization to reduce a rectangular matrix to a smaller square problem. This is beneficial when m >> n or n >> m because bidiagonalization cost scales with the larger dimension. The fast variant uses extra memory for cleaner logic, while in-place reuses buffers.",
    tags=["optimization", "thin-svd", "qr-factorization", "memory-management"]
))

# Entry 3: svect_overwrite mode - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Explain the svect_overwrite mode in GESVD. How does it differ from svect_singular and svect_all? What are the buffer management implications?",
    context_text="GESVD supports svect_overwrite mode which returns singular vectors in the input matrix A, saving memory. This mode requires careful buffer management and has restrictions: both left_svect and right_svect cannot be overwrite simultaneously.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """// Argument checking
if((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
    && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none)
   || (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
       && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none)
   || (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
    return rocblas_status_invalid_value;

// Buffer assignment for m >= n case
if(row)
{
    UV = U;
    if(leftvO)
    {
        rocsolver_bdsqr_template<T>(handle, uplo, k, nv, nu, 0, S, strideS, E, strideE, V,
                                    shiftV, ldv, strideV, A, shiftA, lda, strideA, ...);
    }
}"""
    }],
    answer="""svect_overwrite mode in GESVD:

**Four svect modes:**

1. **svect_none**: Don't compute singular vectors
2. **svect_singular**: Compute first min(m,n) singular vectors (thin)
3. **svect_all**: Compute all singular vectors (full orthogonal matrix)
4. **svect_overwrite**: Return singular vectors in matrix A (save memory)

**svect_overwrite characteristics:**

**Memory savings:**
```cpp
// Without overwrite:
// Need U (m×m or m×k) + V (n×n or n×k) + A (m×n)

// With left_svect_overwrite:
// A will contain left singular vectors
// Only need V (n×n or n×k)
// Saves m×k or m×m storage

// With right_svect_overwrite:
// A will contain right singular vectors
// Only need U (m×m or m×k)
// Saves n×k or n×n storage
```

**Restriction - cannot overwrite both:**
```cpp
if(left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite)
    return rocblas_status_invalid_value;
```

Why? A can only hold one set of vectors (either U or V, not both)

**Buffer management for overwrite:**

```cpp
// For m >= n, row=true, leftvO=true (left overwrite)
if(leftvO)
{
    // Generate left vectors in A
    rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(
        handle, rocblas_column_wise, m, k, n, A, shiftA, lda, strideA, ...);

    // BDSQR updates A with left singular vectors
    rocsolver_bdsqr_template<T>(
        handle, uplo, k, nv, nu, 0, S, strideS, E, strideE,
        V, shiftV, ldv, strideV,  // Right vectors to V
        A, shiftA, lda, strideA,  // Left vectors to A (overwrite!)
        ...);
}

// For m >= n, row=true, rightvO=true (right overwrite)
if(rightvO)
{
    rocsolver_bdsqr_template<T>(
        handle, uplo, k, nv, nu, 0, S, strideS, E, strideE,
        A, shiftA, lda, strideA,  // Right vectors to A (overwrite!)
        U, shiftU, ldu, strideU,  // Left vectors to U
        ...);
}
```

**Size implications for overwrite:**

- left_svect_overwrite: A must have lda >= m (same as input)
  - Returns m×k matrix in A (first k columns are left singular vectors)

- right_svect_overwrite: A must have lda >= m, but data is k×n
  - Returns k×n matrix in A (first k rows are right singular vectors)

**Use cases:**

1. **Memory-constrained environments**: Save U or V allocation
2. **Don't need input A**: Can overwrite without loss
3. **Single-sided SVD**: Only need U or V, not both

**Implementation complexity:**

svect_overwrite adds significant branching:
- Different buffer routing in BDSQR
- Different matrix generation in ORGBR
- Different copy operations in thin SVD path

Example:
```cpp
if(!leftvO && !rightvO)
{
    rocsolver_bdsqr_template<T>(..., V, ..., U, ..., (T*)nullptr, ...);
}
else if(leftvO && !rightvO)
{
    rocsolver_bdsqr_template<T>(..., V, ..., A, ..., (W) nullptr, ...);
}
else  // rightvO && !leftvO
{
    rocsolver_bdsqr_template<T>(..., A, ..., U, ..., (W) nullptr, ...);
}
```""",
    rationale="svect_overwrite allows GESVD to return singular vectors in the input matrix A, saving memory for U or V allocation. The restriction preventing both left and right from being overwrite is because A can only hold one set of vectors. This mode requires careful buffer routing through ORGBR and BDSQR.",
    tags=["coding", "memory-optimization", "api", "buffer-management"]
))

# Entry 4: GEBRD bidiagonalization - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Trace the GEBRD bidiagonalization algorithm. How does it reduce a general matrix to bidiagonal form using Householder reflectors from both sides?",
    context_text="GEBRD reduces A to bidiagonal form B using orthogonal transformations: A = U*B*V^*. It alternates between column and row Householder reflectors using blocked LABRD for efficiency.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gebrd.hpp",
        "language": "cpp",
        "content": """while(j < dim - k)
{
    // Reduce block to bidiagonal form
    jb = std::min(dim - j, nb);
    rocsolver_labrd_template<T>(handle, m - j, n - j, jb, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, D + j, strideD, E + j, strideE, tauq + j, strideQ,
                                taup + j, strideP, X, shiftX, ldx, strideX, Y, shiftY, ldy,
                                strideY, batch_count, scalars, work_workArr, Abyx_norms);

    // update the rest of the matrix
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   m - j - jb, n - j - jb, jb, &minone, A, shiftA + idx2D(j + jb, j, lda), lda,
                   strideA, Y, shiftY + jb, ldy, strideY, &one, A,
                   shiftA + idx2D(j + jb, j + jb, lda), lda, strideA, batch_count,
                   (T**)work_workArr);

    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m - j - jb, n - j - jb,
                   jb, &minone, X, shiftX + jb, ldx, strideX, A, shiftA + idx2D(j, j + jb, lda),
                   lda, strideA, &one, A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,
                   batch_count, (T**)work_workArr);

    j += nb;
}"""
    }],
    answer="""GEBRD bidiagonalization algorithm trace:

**Goal:** Reduce A (m×n) to bidiagonal form B

For m >= n (tall matrix):
```
A = U * B * V^*  where B = [d1  e1   0   0]
                            [ 0  d2  e2   0]
                            [ 0   0  d3  e3]
                            [ 0   0   0  d4]
                            [ 0   0   0   0]
```

**Blocked algorithm:**

```cpp
j = 0;  // Column index
while(j < min(m,n) - k)  // k is switch size for unblocked
{
    jb = min(min(m,n) - j, nb);  // Block size

    // STAGE 1: Reduce block using LABRD
    // Input: A[j:m, j:n] (trailing submatrix)
    // Output:
    //   - D[j:j+jb]: Diagonal elements
    //   - E[j:j+jb]: Off-diagonal elements
    //   - tauq[j:j+jb]: Left Householder scalars
    //   - taup[j:j+jb]: Right Householder scalars
    //   - X, Y: Auxiliary matrices for updating
    rocsolver_labrd_template<T>(
        handle, m - j, n - j, jb,
        A, shiftA + idx2D(j, j, lda), lda, strideA,
        D + j, strideD,  // Diagonal
        E + j, strideE,  // Superdiagonal
        tauq + j, strideQ,  // Left reflector scalars
        taup + j, strideP,  // Right reflector scalars
        X, shiftX, ldx, strideX,  // Auxiliary for update
        Y, shiftY, ldy, strideY,  // Auxiliary for update
        ...);

    // STAGE 2: Update trailing submatrix
    // A[j+jb:m, j+jb:n] -= A[j+jb:m, j:j+jb] * Y[jb:n-j, 0:jb]^*
    rocsolver_gemm(handle, rocblas_operation_none,
                   rocblas_operation_conjugate_transpose,
                   m - j - jb, n - j - jb, jb,  // Dimensions
                   &minone,  // Alpha = -1
                   A, shiftA + idx2D(j + jb, j, lda), lda, strideA,  // Left matrix
                   Y, shiftY + jb, ldy, strideY,  // Right matrix
                   &one,  // Beta = 1 (accumulate)
                   A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,  // Output
                   ...);

    // A[j+jb:m, j+jb:n] -= X[jb:m-j, 0:jb] * A[j:j+jb, j+jb:n]
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                   m - j - jb, n - j - jb, jb,
                   &minone,
                   X, shiftX + jb, ldx, strideX,
                   A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                   &one,
                   A, shiftA + idx2D(j + jb, j + jb, lda), lda, strideA,
                   ...);

    j += nb;  // Move to next block
}

// STAGE 3: Handle last block (size < k) with unblocked GEBD2
if(j < min(m,n))
{
    rocsolver_gebd2_template<T>(
        handle, m - j, n - j,
        A, shiftA + idx2D(j, j, lda), lda, strideA,
        D + j, strideD, E + j, strideE,
        tauq + j, strideQ, taup + j, strideP,
        ...);
}
```

**How LABRD works (one block):**

For j-th block of size jb:
1. For i = 0 to jb-1:
   a. Generate left Householder reflector for column j+i (annihilate below diagonal)
   b. Apply reflector to trailing columns
   c. Generate right Householder reflector for row j+i (annihilate to right of superdiagonal)
   d. Apply reflector to trailing rows
   e. Store updates in X and Y for later GEMM

**Complexity:**

- Blocked: O(mn·min(m,n)) using BLAS-3 (GEMM) for most work
- Unblocked: O(mn·min(m,n)) using BLAS-2 (less efficient)
- Block size nb typically 64-256 for optimal GEMM performance

**Output:**
- D: Diagonal of B (length min(m,n))
- E: Superdiagonal of B (length min(m,n)-1)
- tauq: Householder scalars for left reflectors (stored in A)
- taup: Householder scalars for right reflectors (stored in A)
- A: Overwritten with Householder vectors""",
    rationale="GEBRD uses blocked bidiagonalization with LABRD to reduce a general matrix to bidiagonal form efficiently. It alternates between left and right Householder reflectors, using BLAS-3 GEMM for bulk updates to achieve high performance. The algorithm stores reflectors implicitly in A and scalars in tauq/taup.",
    tags=["coding", "bidiagonalization", "gebrd", "householder", "blocked-algorithm"]
))

# Entry 5: BDSQR QR iteration - L2
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Explain how BDSQR computes the SVD of a bidiagonal matrix using QR iteration with Wilkinson shift. What are the convergence criteria and how are singular vectors updated?",
    context_text="BDSQR applies implicit QR iteration to a bidiagonal matrix B to compute its SVD. Each iteration uses a Wilkinson shift to accelerate convergence and applies Givens rotations to diagonalize B while accumulating updates to U and V.",
    code_blocks=[{
        "path": "library/src/auxiliary/rocauxiliary_bdsqr.hpp",
        "language": "cpp",
        "content": """template <typename T>
__device__ T bdsqr_estimate(const rocblas_int n, T* D, T* E, int t2b, T tol, int conver = 0)
{
    T smin = t2b ? std::abs(D[0]) : std::abs(D[n - 1]);
    T t = smin;

    for(rocblas_int i = 1; i < n; ++i)
    {
        jd = t2b ? i : n - 1 - i;
        je = jd - t2b;
        if(conver && (std::abs(E[je]) <= tol * t))
        {
            E[je] = 0;
            smin = -1;
            break;
        }

        t = std::abs(D[jd]) * (t / (t + std::abs(E[je])));
        smin = (t < smin) ? t : smin;
    }

    return smin;
}

template <typename T, typename S>
__device__ void bdsqr_QRstep(const rocblas_int tid,
                             const rocblas_int t2b,
                             const rocblas_int n,
                             const rocblas_int nv,
                             const rocblas_int nu,
                             const rocblas_int nc,
                             S* D,
                             S* E,
                             T* V,
                             T* U,
                             T* C,
                             const S sh,
                             S* rots)
{
    // Apply implicit QR step with shift sh
    // Uses Givens rotations to chase bulge through matrix
}"""
    }],
    answer="""BDSQR QR iteration for bidiagonal SVD:

**Algorithm overview:**

BDSQR computes SVD of bidiagonal matrix B:
```
B = [d0  e0   0   0]
    [ 0  d1  e1   0]
    [ 0   0  d2  e2]
    [ 0   0   0  d3]
```

Goal: B = U * Σ * V^* where Σ diagonal

**Main loop:**

```python
while not converged:
    # 1. Test convergence: Check if |E[i]| <= tol * (|D[i]| + |D[i+1]|)
    smin = bdsqr_estimate(n, D, E, t2b, tol, conver=1)

    if smin < 0:  # Found negligible off-diagonal
        # Split into subproblems: [0:i] and [i+1:n]
        continue

    # 2. Compute Wilkinson shift
    # Uses 2×2 trailing submatrix eigenvalue
    dm = D[n-2]
    dn = D[n-1]
    em = E[n-2]
    mu = dn - (em * em) / (dm + sqrt(dm*dm - em*em))  # Wilkinson shift

    # 3. Apply implicit QR step
    bdsqr_QRstep(tid, t2b, n, nv, nu, nc, D, E, V, U, C, mu, rots)

    # 4. Update singular vectors
    # Apply accumulated rotations to U and V
```

**Convergence criteria:**

```cpp
T bdsqr_estimate(n, D, E, t2b, tol, conver=1)
{
    T t = |D[start]|;
    for i in range:
        if(|E[i]| <= tol * t)  // Negligible off-diagonal
        {
            E[i] = 0;  // Set to zero (deflate)
            return -1;  // Signal convergence
        }
        t = |D[i]| * (t / (t + |E[i]|));  // Update estimate
}
```

Tolerance: `tol = eps * max(|D[i]|)` where eps is machine precision

**QR step with Givens rotations:**

```cpp
__device__ void bdsqr_QRstep(n, D, E, V, U, shift)
{
    // Initial bulge creation
    f = D[0]^2 - shift
    g = D[0] * E[0]

    for k = 0 to n-2:
        // Compute Givens rotation to annihilate g
        [c, s, r] = lartg(f, g)

        // Apply rotation from right (update V)
        [D[k], E[k], D[k+1]] = apply_right_rotation(c, s, ...)
        if(nv > 0)
            save_rotation(rots, c, s)  // Save for updating V

        // Chase bulge: Create new g in next position
        f = ...
        g = ...

        // Compute Givens rotation to annihilate g
        [c, s, r] = lartg(f, g)

        // Apply rotation from left (update U)
        [D[k], E[k], D[k+1]] = apply_left_rotation(c, s, ...)
        if(nu > 0)
            save_rotation(rots, c, s)  // Save for updating U

        // Update f, g for next iteration
}
```

**Singular vector updates:**

After QR step, apply saved rotations:

```cpp
// Update V (right singular vectors)
for k = 0 to n-2:
    c = rots[k]
    s = rots[k + n]
    for i = 0 to nv:
        temp = c * V[i,k] - s * V[i,k+1]
        V[i,k+1] = s * V[i,k] + c * V[i,k+1]
        V[i,k] = temp

// Update U (left singular vectors)
for k = 0 to n-2:
    c = rots[k + 2*n]
    s = rots[k + 3*n]
    for i = 0 to nu:
        temp = c * U[i,k] - s * U[i,k+1]
        U[i,k+1] = s * U[i,k] + c * U[i,k+1]
        U[i,k] = temp
```

**Wilkinson shift:**

Accelerates convergence by shifting towards smallest singular value:
```cpp
// 2×2 trailing principal submatrix:
T = [dm  em]
    [em  dn]

// Eigenvalue closest to dn (Wilkinson shift)
shift = dn - em^2 / (dm + sign(dm) * sqrt(dm^2 + em^2))
```

**Complexity:**

- Average: O(n²) iterations for full convergence
- Each iteration: O(n) for bidiagonal update + O(n·nv) for V + O(n·nu) for U
- Total: O(n²(1 + nv + nu))

**Convergence properties:**

- Quadratic convergence near singular values
- Typically 2-3 iterations per singular value
- Most expensive for smallest singular values""",
    rationale="BDSQR uses implicit QR iteration with Wilkinson shift to compute the SVD of a bidiagonal matrix. Each iteration applies Givens rotations to chase a bulge through the matrix, converging quadratically. Convergence is tested by checking if off-diagonal elements are negligible relative to neighboring diagonal elements.",
    tags=["algorithm", "bdsqr", "qr-iteration", "givens-rotation", "wilkinson-shift"]
))

# Entry 6: ORGBR matrix generation - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Implement the ORGBR call that generates orthogonal matrices U and V from the Householder reflectors stored by GEBRD. How does storev (column_wise vs row_wise) affect the algorithm?",
    context_text="After GEBRD bidiagonalizes A, the Householder reflectors are stored implicitly in A along with scalars in tauq/taup. ORGBR generates the explicit orthogonal matrices U and V, with storev determining which set of reflectors to process.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """if(leftvS || leftvA)
{
    // copy data to matrix U where orthogonal matrix will be generated
    mn = (row && leftvS) ? n : m;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_m, blocks_k, batch_count),
                            dim3(thread_count, thread_count, 1), 0, stream, m, k, A, shiftA,
                            lda, strideA, U, shiftU, ldu, strideU);

    rocsolver_orgbr_ungbr_template<false, STRIDED>(
        handle, rocblas_column_wise, m, mn, n, U, shiftU, ldu, strideU, tau_splits, k,
        batch_count, scalars, (T*)work_workArr, Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X,
        workArr);
}

if(rightvS || rightvA)
{
    mn = (!row && rightvS) ? m : n;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks_k, blocks_n, batch_count),
                            dim3(thread_count, thread_count, 1), 0, stream, k, n, A, shiftA,
                            lda, strideA, V, shiftV, ldv, strideV);

    rocsolver_orgbr_ungbr_template<false, STRIDED>(
        handle, rocblas_row_wise, mn, n, m, V, shiftV, ldv, strideV,
        (tau_splits + k * batch_count), k, batch_count, scalars, (T*)work_workArr,
        Abyx_norms_tmptr_cmplt, Abyx_norms_trfact_X, workArr);
}"""
    }],
    answer="""ORGBR matrix generation from GEBRD reflectors:

**GEBRD output format:**

After GEBRD on m×n matrix (m >= n):
```
A contains:
[d0  e0  v0  v0  v0]  ← First row: e0, right reflectors v0
[u0  d1  e1  v1  v1]  ← Below diagonal: left reflectors u
[u0  u1  d2  e2  v2]
[u0  u1  u2  d3  e3]
[u0  u1  u2  u3  d4]

tauq: Householder scalars for left reflectors (U)
taup: Householder scalars for right reflectors (V)
```

**ORGBR for left singular vectors U:**

```cpp
// Copy A to U (preserves A if needed)
ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>,
    ..., m, k,  // Copy m rows, k columns
    A, shiftA, lda, strideA,
    U, shiftU, ldu, strideU);

// Generate U from column-wise reflectors
rocsolver_orgbr_ungbr_template<false, STRIDED>(
    handle,
    rocblas_column_wise,  // Process left reflectors
    m,                     // Number of rows in U
    mn,                    // Number of columns (m for svect_all, n for svect_singular)
    n,                     // Original columns (determines reflector count)
    U, shiftU, ldu, strideU,
    tau_splits,           // tauq scalars
    k,                    // Number of reflectors
    ...);
```

**Column-wise (left reflectors):**

Process reflectors stored below diagonal:
```
For j = n-1 down to 0:
    H_j = I - tauq[j] * u_j * u_j^*
    where u_j = [0, ..., 0, 1, u[j+1,j], u[j+2,j], ..., u[m-1,j]]^T

    U = U * H_j  // Apply from right
```

**ORGBR for right singular vectors V:**

```cpp
// Copy A to V
ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>,
    ..., k, n,  // Copy k rows, n columns
    A, shiftA, lda, strideA,
    V, shiftV, ldv, strideV);

// Generate V from row-wise reflectors
rocsolver_orgbr_ungbr_template<false, STRIDED>(
    handle,
    rocblas_row_wise,     // Process right reflectors
    mn,                    // Number of rows (n for svect_all, m for svect_singular)
    n,                     // Number of columns in V
    m,                     // Original rows (determines reflector count)
    V, shiftV, ldv, strideV,
    (tau_splits + k * batch_count),  // taup scalars
    k,                     // Number of reflectors
    ...);
```

**Row-wise (right reflectors):**

Process reflectors stored to right of superdiagonal:
```
For i = 0 to n-2:
    H_i = I - taup[i] * v_i * v_i^*
    where v_i = [0, ..., 0, 1, v[i,i+2], v[i,i+3], ..., v[i,n-1]]^T

    V = H_i * V  // Apply from left
```

**storev parameter:**

- **rocblas_column_wise**:
  - Process reflectors stored in columns (below diagonal)
  - Used for generating U (left singular vectors)
  - Apply reflectors from right: U = U * H_k * ... * H_1

- **rocblas_row_wise**:
  - Process reflectors stored in rows (to right of superdiagonal)
  - Used for generating V (right singular vectors)
  - Apply reflectors from left: V = H_1 * ... * H_k * V

**Size parameters:**

For m >= n case:

Left vectors (U):
- svect_singular: mn = n → Generate m×n matrix (thin U)
- svect_all: mn = m → Generate m×m matrix (full U)

Right vectors (V):
- svect_singular: mn = n → Generate n×n matrix (already thin)
- svect_all: mn = n → Generate n×n matrix (full for square B)

**Complexity:**

- Generating m×n U: O(m²n) flops
- Generating n×n V: O(n³) flops
- Uses BLAS-3 operations (GEMM) in blocked implementation""",
    rationale="ORGBR generates explicit orthogonal matrices from Householder reflectors stored by GEBRD. The storev parameter determines which set of reflectors to process: column_wise for left reflectors (U) stored below diagonal, or row_wise for right reflectors (V) stored to right of superdiagonal. The size parameters control whether to generate full or thin matrices.",
    tags=["coding", "orgbr", "householder", "matrix-generation", "storev"]
))

# Entry 7: Workspace calculation - L3 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesvd",
    instruction="Implement the workspace calculation for GESVD in the thin SVD path with fast_alg=outofplace. Account for GEQRF/GELQF, GEBRD, ORGBR, ORMBR, BDSQR, and temporary arrays.",
    context_text="GESVD workspace calculation is complex due to multiple algorithm paths (normal vs thin, fast vs in-place) and many subroutines (QR/LQ, bidiagonalization, matrix generation). The thin SVD with outofplace mode requires the most memory.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

// size of arrays to store temporary copies
*size_tempArrayT
    = (fast_thinSVD || (thinSVD && leadvO && othervN)) ? sizeof(T) * k * k * batch_count : 0;
*size_tempArrayC
    = (fast_thinSVD && (othervN || othervO || leadvO)) ? sizeof(T) * m * n * batch_count : 0;

// workspace required for the bidiagonalization
if(thinSVD)
    rocsolver_gebrd_getMemorySize<BATCHED, T>(k, k, batch_count, size_scalars, &w[0], &a[0],
                                              &x[0], &y[0]);

// extra requirements for QR/LQ factorization
if(thinSVD)
{
    if(row)
        rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                  &y[1], &unused);
    else
        rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &unused, &w[2], &x[1],
                                                  &y[1], &unused);
}"""
    }],
    answer="""GESVD workspace calculation for thin SVD with fast_alg=outofplace:

```cpp
void rocsolver_gesvd_getMemorySize(
    left_svect, right_svect, m, n, batch_count, fast_alg,
    size_scalars, size_work_workArr, size_Abyx_norms_tmptr_cmplt,
    size_Abyx_norms_trfact_X, size_diag_tmptr_Y, size_tau_splits,
    size_tempArrayT, size_tempArrayC, size_workArr)
{
    // Determine execution path
    const bool row = (m >= n);
    const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
    const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);

    const rocblas_int k = min(m, n);
    const rocblas_int kk = max(m, n);

    size_t w[6] = {0};  // work_workArr requirements
    size_t a[6] = {0};  // Abyx_norms requirements
    size_t x[6] = {0};  // trfact_X requirements
    size_t y[3] = {0};  // tmptr_Y requirements

    if(BATCHED)
        *size_workArr = 2 * sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // TEMP ARRAYS for out-of-place algorithm
    if(fast_thinSVD)
    {
        // Temporary T: k×k triangular factor from QR/LQ
        *size_tempArrayT = sizeof(T) * k * k * batch_count;

        // Temporary C: m×n copy of original matrix
        if(othervN || othervO || leadvO)
            *size_tempArrayC = sizeof(T) * m * n * batch_count;
    }
    else
    {
        *size_tempArrayT = 0;
        *size_tempArrayC = 0;
    }

    // GEBRD workspace (bidiagonalization)
    if(thinSVD)
        // Bidiagonalize k×k triangular factor
        rocsolver_gebrd_getMemorySize<BATCHED, T>(
            k, k, batch_count,
            size_scalars, &w[0], &a[0], &x[0], &y[0]);
    else
        // Bidiagonalize full m×n matrix
        rocsolver_gebrd_getMemorySize<BATCHED, T>(
            m, n, batch_count,
            size_scalars, &w[0], &a[0], &x[0], &y[0]);

    // BDSQR workspace (bidiagonal SVD)
    rocblas_int nu = leftvN ? 0 : (fast_thinSVD || (thinSVD && leadvN)) ? k : m;
    rocblas_int nv = rightvN ? 0 : (fast_thinSVD || (thinSVD && leadvN)) ? k : n;
    rocsolver_bdsqr_getMemorySize<S>(
        k, nv, nu, 0, batch_count,
        size_tau_splits, &w[1], &a[1]);

    // Override tau_splits size (needs space for Householder scalars too)
    *size_tau_splits = max(*size_tau_splits, 2 * sizeof(T) * k * batch_count);

    if(thinSVD)
    {
        // QR or LQ factorization workspace
        if(row)
            rocsolver_geqrf_getMemorySize<BATCHED, T>(
                m, n, batch_count,
                &unused, &w[2], &x[1], &y[1], &unused);
        else
            rocsolver_gelqf_getMemorySize<BATCHED, T>(
                m, n, batch_count,
                &unused, &w[2], &x[1], &y[1], &unused);

        // ORGBR workspace (generate bidiagonal matrices)
        if(!othervN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                storev_other, k, k, k, batch_count,
                &unused, &w[3], &a[3], &x[3], &unused);

        if(fast_thinSVD && !leadvN)
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                storev_lead, k, k, k, batch_count,
                &unused, &w[4], &a[4], &x[4], &unused);

        // ORGQR/ORGLQ workspace (generate Q from compression)
        if(!leadvN)
        {
            if(leadvA)
            {
                if(row)
                    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(
                        kk, kk, k, batch_count,
                        &unused, &w[5], &a[5], &x[5], &unused);
                else
                    rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(
                        kk, kk, k, batch_count,
                        &unused, &w[5], &a[5], &x[5], &unused);
            }
            else
            {
                if(row)
                    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(
                        m, n, k, batch_count,
                        &unused, &w[5], &a[5], &x[5], &unused);
                else
                    rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(
                        m, n, k, batch_count,
                        &unused, &w[5], &a[5], &x[5], &unused);
            }
        }
    }
    else
    {
        // Normal SVD: ORGBR for U and V
        if(leftvS || leftvA)
        {
            rocblas_int mn = (row && leftvS) ? n : m;
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_column_wise, m, mn, n, batch_count,
                &unused, &w[3], &a[3], &x[3], &unused);
        }

        if(rightvS || rightvA)
        {
            rocblas_int mn = (!row && rightvS) ? m : n;
            rocsolver_orgbr_ungbr_getMemorySize<BATCHED, T>(
                rocblas_row_wise, mn, n, m, batch_count,
                &unused, &w[4], &a[4], &x[4], &unused);
        }
    }

    // Combine: Take maximum of each reused buffer
    *size_work_workArr = *max_element(w, w+6);
    *size_Abyx_norms_tmptr_cmplt = *max_element(a, a+6);
    *size_Abyx_norms_trfact_X = *max_element(x, x+6);
    *size_diag_tmptr_Y = *max_element(y, y+3);
}
```

**Workspace categories:**

1. **Temporary arrays (fast_thinSVD only):**
   - tempArrayT: k×k triangular factor (R from QR or L from LQ)
   - tempArrayC: m×n copy of A (for updates)

2. **Reusable buffers (max of all subroutines):**
   - work_workArr: General workspace (GEMM, etc.)
   - Abyx_norms_tmptr_cmplt: Norms and temporary storage
   - Abyx_norms_trfact_X: Block update matrix X
   - diag_tmptr_Y: Block update matrix Y

3. **Persistent storage:**
   - tau_splits: Householder scalars (2k elements)
   - scalars: Constants for rocBLAS calls

**Memory totals for m=1000, n=100, fast_thinSVD:**

- tempArrayT: 100×100 = 10,000 elements
- tempArrayC: 1000×100 = 100,000 elements
- Reused buffers: ~max(GEQRF, GEBRD, ORGBR, BDSQR) ≈ 20,000 elements
- Total: ~130,000 elements (vs ~150,000 for normal SVD)""",
    rationale="GESVD workspace calculation must account for multiple algorithm paths and subroutines. Fast thin SVD uses extra temporary arrays (tempArrayT for triangular factor, tempArrayC for matrix copy) but reduces bidiagonalization cost. Workspace is maximized across reused buffers since subroutines execute sequentially.",
    tags=["coding", "workspace", "memory-management", "thin-svd"]
))

# Entry 8: local_geqrlq_template wrapper - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesvd",
    instruction="Implement the local_geqrlq_template wrapper that calls either GEQRF or GELQF based on the row parameter. Why is this abstraction useful?",
    context_text="GESVD needs to compress tall matrices with QR and wide matrices with LQ. The local_geqrlq_template wrapper provides a unified interface that dispatches to the appropriate factorization based on matrix shape.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """/** wrapper to GEQRF/GELQF_TEMPLATE **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_geqrlq_template(rocblas_handle handle,
                           const rocblas_int m,
                           const rocblas_int n,
                           U A,
                           const rocblas_int shiftA,
                           const rocblas_int lda,
                           const rocblas_stride strideA,
                           T* ipiv,
                           const rocblas_stride strideP,
                           const rocblas_int batch_count,
                           T* scalars,
                           void* work_workArr,
                           T* Abyx_norms_trfact,
                           T* diag_tmptr,
                           T** workArr,
                           const bool row)
{
    if(row)
        rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
    else
        rocsolver_gelqf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                   strideP, batch_count, scalars, work_workArr,
                                                   Abyx_norms_trfact, diag_tmptr, workArr);
}"""
    }],
    answer="""local_geqrlq_template wrapper implementation:

```cpp
/**
 * Unified wrapper for QR/LQ factorization
 * Dispatches based on matrix orientation (tall vs wide)
 */
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_geqrlq_template(
    rocblas_handle handle,
    const rocblas_int m,          // Rows
    const rocblas_int n,          // Columns
    U A,                          // Input/output matrix
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    T* ipiv,                      // Householder scalars (tau)
    const rocblas_stride strideP,
    const rocblas_int batch_count,
    T* scalars,
    void* work_workArr,
    T* Abyx_norms_trfact,
    T* diag_tmptr,
    T** workArr,
    const bool row)               // true: QR, false: LQ
{
    if(row)
    {
        // Tall matrix (m >= n): Use QR factorization
        // A = Q * R where Q is m×m orthogonal, R is m×n upper triangular
        rocsolver_geqrf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA,
            ipiv,            // tau: Householder scalars for Q
            strideP, batch_count,
            scalars, work_workArr, Abyx_norms_trfact, diag_tmptr, workArr);
    }
    else
    {
        // Wide matrix (m < n): Use LQ factorization
        // A = L * Q where L is m×m lower triangular, Q is m×n orthogonal
        rocsolver_gelqf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA,
            ipiv,            // tau: Householder scalars for Q
            strideP, batch_count,
            scalars, work_workArr, Abyx_norms_trfact, diag_tmptr, workArr);
    }
}

/**
 * Unified wrapper for generating Q from QR/LQ
 */
template <bool BATCHED, bool STRIDED, typename T, typename U>
void local_orgqrlq_ungqrlq_template(
    rocblas_handle handle,
    const rocblas_int m,
    const rocblas_int n,
    const rocblas_int k,
    U A,
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    T* ipiv,
    const rocblas_stride strideP,
    const rocblas_int batch_count,
    T* scalars,
    T* work,
    T* Abyx_tmptr,
    T* trfact,
    T** workArr,
    const bool row)
{
    if(row)
        // Generate Q from QR factorization
        rocsolver_orgqr_ungqr_template<BATCHED, STRIDED>(
            handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP,
            batch_count, scalars, work, Abyx_tmptr, trfact, workArr);
    else
        // Generate Q from LQ factorization
        rocsolver_orglq_unglq_template<BATCHED, STRIDED>(
            handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP,
            batch_count, scalars, work, Abyx_tmptr, trfact, workArr);
}
```

**Why this abstraction is useful:**

1. **Code reuse**: GESVD has multiple paths that need QR/LQ
   - Thin SVD compression
   - Both m>=n and m<n cases
   - Same wrapper handles both with one parameter

2. **Symmetric logic**:
```cpp
const bool row = (m >= n);

// Compression
local_geqrlq_template(..., row);

// Generate Q
local_orgqrlq_ungqrlq_template(..., row);
```

3. **Reduces code duplication**: Instead of:
```cpp
if(m >= n)
{
    rocsolver_geqrf_template(...);
    // ... later ...
    rocsolver_orgqr_ungqr_template(...);
}
else
{
    rocsolver_gelqf_template(...);
    // ... later ...
    rocsolver_orglq_unglq_template(...);
}
```

Can write:
```cpp
local_geqrlq_template(..., row);
// ... later ...
local_orgqrlq_ungqrlq_template(..., row);
```

4. **Maintainability**: Changes to calling convention only need updating in wrapper

5. **Matching parameters**: Both GEQRF and GELQF have identical signatures
   - Same m, n, A, lda, tau, workspace
   - Only difference: QR vs LQ algorithm
   - Perfect candidate for wrapper abstraction

**Usage in GESVD:**

```cpp
const bool row = (m >= n);

if(thinSVD)
{
    // Stage 1: Compress matrix
    local_geqrlq_template<BATCHED, STRIDED>(
        handle, m, n, A, ..., tau_splits, ..., row);
    //                                         ^^^^
    //                                    Determines QR vs LQ

    // Stage 2: Generate orthogonal matrix
    local_orgqrlq_ungqrlq_template<false, STRIDED>(
        handle, m, n, k, A, ..., tau_splits, ..., row);
    //                                            ^^^^
    //                                      Same parameter!
}
```""",
    rationale="The local_geqrlq_template wrapper provides a unified interface for QR/LQ factorization, dispatching based on the row parameter. This abstraction reduces code duplication in GESVD which needs to handle both tall (QR) and wide (LQ) matrices symmetrically, improving maintainability.",
    tags=["coding", "wrapper", "abstraction", "qr-lq-factorization"]
))

# Entry 9: E array (superdiagonal storage) - L1
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesvd",
    instruction="Explain the E array in GESVD. What does it store, when is it populated, and why does it have length min(m,n)-1?",
    context_text="GESVD accepts an optional E array that stores the superdiagonal (or subdiagonal) of the bidiagonal matrix B from GEBRD. This is useful for debugging or analyzing the bidiagonalization quality.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)"""
    }],
    answer="""E array in GESVD:

**What E stores:**

E contains the off-diagonal elements of the bidiagonal matrix B produced by GEBRD:

For m >= n (upper bidiagonal):
```
B = [d0  e0   0   0]
    [ 0  d1  e1   0]
    [ 0   0  d2  e2]
    [ 0   0   0  d3]

S = [d0, d1, d2, d3]          (length min(m,n))
E = [e0, e1, e2]              (length min(m,n)-1)
```

For m < n (lower bidiagonal):
```
B = [d0   0   0   0]
    [e0  d1   0   0]
    [ 0  e1  d2   0]
    [ 0   0  e2  d3]

S = [d0, d1, d2, d3]          (length min(m,n))
E = [e0, e1, e2]              (length min(m,n)-1)
```

**When E is populated:**

1. **After GEBRD**:
```cpp
rocsolver_gebrd_template<BATCHED, STRIDED>(
    handle, m, n, A, shiftA, lda, strideA,
    S, strideS,  // Diagonal → S
    E, strideE,  // Off-diagonal → E
    tau_splits, k, (tau_splits + k * batch_count), k,
    ...);
```

2. **During BDSQR**:
   - E is modified by QR iterations
   - Off-diagonal elements → 0 as singular values converge
   - Final E should be all zeros (within tolerance)

**Why length min(m,n)-1?**

Bidiagonal matrix has:
- Diagonal: min(m,n) elements
- Off-diagonal: min(m,n)-1 elements

For n×n bidiagonal:
```
[x x 0 0]  ← 3 off-diagonal elements (row 0)
[0 x x 0]  ← 1 element (row 1)
[0 0 x x]  ← 1 element (row 2)
[0 0 0 x]  ← 0 elements (row 3)

Total: 3 = n-1
```

**Use cases for E:**

1. **Convergence analysis**:
```cpp
// Check BDSQR convergence
for(int i = 0; i < k-1; i++)
{
    if(abs(E[i]) > tol * (abs(S[i]) + abs(S[i+1])))
        printf("Warning: E[%d] = %e not converged\n", i, E[i]);
}
```

2. **Debugging bidiagonalization quality**:
```cpp
// Verify GEBRD produced good bidiagonal form
// E should be relatively small compared to S
double e_norm = 0, s_norm = 0;
for(int i = 0; i < k-1; i++)
    e_norm += E[i] * E[i];
for(int i = 0; i < k; i++)
    s_norm += S[i] * S[i];
printf("Off-diagonal norm ratio: %e\n", sqrt(e_norm / s_norm));
```

3. **Reconstruct bidiagonal B**:
```cpp
// B = diag(S) + superdiag(E) for upper bidiagonal
for(int i = 0; i < k; i++)
    B[i + i*ldb] = S[i];
for(int i = 0; i < k-1; i++)
    B[i + (i+1)*ldb] = E[i];  // Superdiagonal
```

**Pointer check:**

```cpp
// E must be provided if problem is non-trivial
if((std::min(m, n) > 1 && !E) || (batch_count && !info))
    return rocblas_status_invalid_pointer;
```

E is required unless min(m,n) <= 1 (trivial case with no off-diagonal)

**Comparison to GESDD:**

GESDD doesn't have E array because:
- Uses eigendecomposition (A^*A has no off-diagonal to expose)
- E is internal to BDSQR, not exposed

GESVD exposes E because:
- Bidiagonalization is explicit algorithm step
- Useful for numerical analysis
- LAPACK compatibility (DGESVD has E parameter)""",
    rationale="The E array stores the off-diagonal elements of the bidiagonal matrix B from GEBRD. It has length min(m,n)-1 because a bidiagonal matrix has one fewer off-diagonal than diagonal elements. E is useful for debugging, convergence analysis, and LAPACK compatibility.",
    tags=["api", "bidiagonal", "array-storage", "debugging"]
))

# Entry 10: fast_alg workmode parameter - L2
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesvd",
    instruction="Explain the fast_alg parameter in GESVD. What are the differences between rocblas_inplace and rocblas_outofplace modes? When should each be used?",
    context_text="GESVD provides a fast_alg parameter controlling the thin SVD algorithm variant. rocblas_outofplace uses extra memory for cleaner logic and potentially better performance, while rocblas_inplace minimizes memory usage.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)"""
    }],
    answer="""fast_alg parameter in GESVD:

**Two modes:**

1. **rocblas_inplace**: Minimize memory, reuse buffers
2. **rocblas_outofplace**: Use extra memory for cleaner algorithm

**When modes matter:**

```cpp
const bool thinSVD = (m >= THIN_SVD_SWITCH * n || n >= THIN_SVD_SWITCH * m);
const bool fast_thinSVD = (thinSVD && fast_alg == rocblas_outofplace);
```

- Only affects thin SVD path (when matrix is sufficiently rectangular)
- Normal SVD ignores fast_alg parameter

**Memory differences:**

**rocblas_outofplace:**
```cpp
// Allocate temporary arrays
*size_tempArrayT = sizeof(T) * k * k * batch_count;         // 10,000 for k=100
*size_tempArrayC = sizeof(T) * m * n * batch_count;         // 100,000 for m=1000,n=100

// Total extra: (k² + mn) elements
// For m=1000, n=100: 110,000 elements ≈ 880 KB (double precision)
```

**rocblas_inplace:**
```cpp
// No temporary arrays
*size_tempArrayT = 0;
*size_tempArrayC = 0;

// Reuses A buffer and user-provided U/V
```

**Algorithm differences:**

**rocblas_outofplace (fast) path:**

```cpp
// Stage 1: QR/LQ compression
local_geqrlq_template(..., A, ..., tau, ...);
// A now contains Q*R or L*Q

// Copy triangular part to tempArrayT
copy_mat(..., k, k, A, ..., tempArrayT, ..., uplo);
// Preserves triangular factor separately

if(leadvA)
    // Copy full factorization to U/V for Q generation
    copy_mat(..., m, n, A, ..., UV, ...);

// Stage 2: Generate Q in A (or U/V if leadvA)
local_orgqrlq_ungqrlq_template(..., A or UV, ..., tau, ...);
// A now contains Q (or U/V contains Q)

// Stage 3: Bidiagonalize triangular factor (in tempArrayT)
rocsolver_gebrd_template(..., k, k, tempArrayT, ...);
// tempArrayT contains bidiagonal + reflectors

// Stage 4: Generate bidiagonal U and V
rocsolver_orgbr_ungbr_template(..., tempArrayT, ...);  // Lead vectors
rocsolver_orgbr_ungbr_template(..., tempArrayC, ...);  // Other vectors

// Stage 5: BDSQR
rocsolver_bdsqr_template(..., tempArrayC, ..., tempArrayT, ...);

// Stage 6: Compose final result
if(leadvO || leadvS)
{
    // Multiply: final_vectors = Q * tempArray
    rocsolver_gemm(..., A, ..., tempArrayT, ..., tempArrayC or UV, ...);
    // Copy back if needed
}
```

**rocblas_inplace path:**

```cpp
// Stage 1: QR/LQ compression
local_geqrlq_template(..., A, ..., tau, ...);

if(!leadvO)
    // Copy to U/V (not overwriting A)
    copy_mat(..., m, n, A, ..., UV, ...);

if(othervS || othervA || (leadvO && othervN))
    // Copy triangular part
    copy_mat(..., k, k, A, ..., bufferT, ..., uplo);
    // bufferT reuses tempArrayT space (small)

// Stage 2: Generate Q
if(leadvO)
    local_orgqrlq_ungqrlq_template(..., A, ..., tau, ...);
else
    local_orgqrlq_ungqrlq_template(..., UV, ..., tau, ...);

// Stage 3: Bidiagonalize (in bufferT or A depending on svect)
if(othervS || othervA || (leadvO && othervN))
    rocsolver_gebrd_template(..., k, k, bufferT, ...);
else
    rocsolver_gebrd_template(..., k, k, A, ...);

// Stage 4: Generate bidiagonal matrices
// Uses ORMBR (apply reflectors) instead of ORGBR (generate full matrix)
rocsolver_ormbr_unmbr_template(...);  // More memory efficient

// Stage 5: BDSQR (various buffer routing)
// Stage 6: No final composition needed (in-place updates)
```

**Trade-offs:**

**Use rocblas_outofplace when:**
- Memory is available (extra k² + mn elements acceptable)
- Want cleaner, easier-to-debug execution
- Potentially faster (fewer copies, better cache use)
- Need to preserve A matrix structure

**Use rocblas_inplace when:**
- Memory constrained (k² + mn too large)
- Don't need original A
- Willing to accept more complex buffer management

**Performance:**

Usually outofplace is faster despite extra memory:
- Better memory access patterns
- Fewer conditionals in hot path
- More opportunities for compiler optimization
- Cleaner GEMM calls

**Example:**

```cpp
// Memory-rich environment
rocsolver_dgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E,
                 rocblas_outofplace,  // Fast thin SVD
                 info);

// Memory-constrained environment
rocsolver_dgesvd(handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E,
                 rocblas_inplace,     // Memory-efficient thin SVD
                 info);
```""",
    rationale="The fast_alg parameter controls whether GESVD uses out-of-place (extra memory, cleaner) or in-place (minimal memory, complex) algorithm for thin SVD. rocblas_outofplace allocates temporary arrays for triangular factor and matrix copy, enabling cleaner logic and potentially better performance. rocblas_inplace minimizes memory by reusing buffers at the cost of more complex buffer management.",
    tags=["api", "memory-tradeoff", "performance", "workmode"]
))

# Entry 11: Argument validation - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesvd",
    instruction="Implement the argument validation for GESVD. Check all svect parameter combinations, leading dimensions, and pointer validity. Why can't both left_svect and right_svect be overwrite?",
    context_text="GESVD has complex argument validation due to four svect modes (none/singular/all/overwrite) for both left and right vectors, each with different leading dimension and pointer requirements.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.hpp",
        "language": "cpp",
        "content": """template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        TT* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        TT* E,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // 1. invalid/non-supported values
    if((left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
        && left_svect != rocblas_svect_overwrite && left_svect != rocblas_svect_none)
       || (right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
           && right_svect != rocblas_svect_overwrite && right_svect != rocblas_svect_none)
       || (left_svect == rocblas_svect_overwrite && right_svect == rocblas_svect_overwrite))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if((right_svect == rocblas_svect_all && ldv < n)
       || (right_svect == rocblas_svect_singular && ldv < std::min(m, n)))
        return rocblas_status_invalid_size;

    // 3. invalid pointers
    if((n && m && !A) || (std::min(m, n) > 1 && !E) || (std::min(m, n) && !S)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_all && m && !U)
       || (left_svect == rocblas_svect_singular && std::min(m, n) && !U))
        return rocblas_status_invalid_pointer;
    if((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && n && !V)
        return rocblas_status_invalid_pointer;
}"""
    }],
    answer="""GESVD argument validation implementation:

```cpp
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_argCheck(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    W A,
    const rocblas_int lda,
    TT* S,
    T* U,
    const rocblas_int ldu,
    T* V,
    const rocblas_int ldv,
    TT* E,
    rocblas_int* info,
    const rocblas_int batch_count = 1)
{
    // ORDER IS IMPORTANT FOR UNIT TESTS

    // STAGE 1: Invalid/non-supported values
    // Check valid svect enum values
    if((left_svect != rocblas_svect_all &&
        left_svect != rocblas_svect_singular &&
        left_svect != rocblas_svect_overwrite &&
        left_svect != rocblas_svect_none)
       ||
       (right_svect != rocblas_svect_all &&
        right_svect != rocblas_svect_singular &&
        right_svect != rocblas_svect_overwrite &&
        right_svect != rocblas_svect_none))
    {
        return rocblas_status_invalid_value;
    }

    // Check incompatible svect combinations
    if(left_svect == rocblas_svect_overwrite &&
       right_svect == rocblas_svect_overwrite)
    {
        return rocblas_status_invalid_value;
        // WHY? A can only hold U OR V, not both
    }

    // STAGE 2: Invalid sizes
    // Basic dimension checks
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // Leading dimension for U
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular)
       && ldu < m)
    {
        return rocblas_status_invalid_size;
        // U is m×m (all) or m×k (singular), need ldu >= m
    }

    // Leading dimension for V
    if(right_svect == rocblas_svect_all && ldv < n)
    {
        return rocblas_status_invalid_size;
        // V is n×n (all), need ldv >= n
    }

    if(right_svect == rocblas_svect_singular && ldv < std::min(m, n))
    {
        return rocblas_status_invalid_size;
        // V is k×n (singular), need ldv >= k = min(m,n)
    }

    // Skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_status_continue;
    }

    // STAGE 3: Invalid pointers
    // Always required: A, S, info
    if((n && m && !A) ||                    // A required if non-empty
       (std::min(m, n) && !S) ||            // S required if non-trivial
       (batch_count && !info))              // info required if batched
    {
        return rocblas_status_invalid_pointer;
    }

    // E required unless min(m,n) <= 1 (no off-diagonal)
    if(std::min(m, n) > 1 && !E)
    {
        return rocblas_status_invalid_pointer;
    }

    // U pointer checks
    if(left_svect == rocblas_svect_all && m && !U)
    {
        return rocblas_status_invalid_pointer;
        // svect_all requires U pointer
    }

    if(left_svect == rocblas_svect_singular && std::min(m, n) && !U)
    {
        return rocblas_status_invalid_pointer;
        // svect_singular requires U pointer
    }

    // Note: svect_overwrite doesn't need U (uses A)
    // Note: svect_none doesn't need U

    // V pointer checks
    if((right_svect == rocblas_svect_all ||
        right_svect == rocblas_svect_singular) && n && !V)
    {
        return rocblas_status_invalid_pointer;
        // svect_all and svect_singular require V pointer
    }

    // Note: svect_overwrite doesn't need V (uses A)
    // Note: svect_none doesn't need V

    return rocblas_status_continue;
}
```

**Why both can't be overwrite:**

```
Matrix A storage:
- Size: m × n
- Can hold either:
  1. Left vectors U (m × k columns), OR
  2. Right vectors V (k × n rows), OR
  3. Original matrix A

Cannot simultaneously hold:
- U AND V (would need m×k + k×n > m×n storage in many cases)

Example: m=1000, n=100, k=100
- U alone: 1000×100 = 100,000 elements ✓ fits in A
- V alone: 100×100 = 10,000 elements ✓ fits in A
- U + V: 110,000 elements ✗ doesn't fit in A
```

**Pointer requirements summary:**

| left_svect | right_svect | U needed? | V needed? | A usage |
|------------|-------------|-----------|-----------|---------|
| none       | none        | No        | No        | Destroyed |
| none       | all         | No        | Yes       | Destroyed |
| none       | singular    | No        | Yes       | Destroyed |
| none       | overwrite   | No        | No        | → V |
| all        | none        | Yes       | No        | Destroyed |
| all        | all         | Yes       | Yes       | Destroyed |
| all        | overwrite   | Yes       | No        | → V |
| singular   | singular    | Yes       | Yes       | Destroyed |
| overwrite  | none        | No        | No        | → U |
| overwrite  | all         | No        | Yes       | → U |
| overwrite  | singular    | No        | Yes       | → U |
| overwrite  | overwrite   | Invalid!  | Invalid!  | Conflict |

**Leading dimension requirements:**

- lda >= m (always)
- ldu >= m (if left_svect = all or singular)
- ldv >= n (if right_svect = all)
- ldv >= min(m,n) (if right_svect = singular)""",
    rationale="GESVD argument validation must check svect parameter validity, leading dimensions, and pointer requirements. The restriction preventing both left_svect and right_svect from being overwrite exists because A cannot simultaneously hold both U and V matrices. Different svect modes have different pointer requirements: none needs no buffer, singular/all need U/V pointers, overwrite uses A.",
    tags=["coding", "validation", "api", "error-handling"]
))

# Entry 12: Production API - L3 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesvd",
    instruction="Implement the complete C API for all four precisions of GESVD (S/D/C/Z). Show how singular values are always real even for complex matrices, and how the fast_alg parameter is exposed.",
    context_text="GESVD provides four precision variants with LAPACK-compatible interfaces. Unlike GESDD, GESVD exposes the E array and fast_alg parameter for user control over the algorithm.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesvd.cpp",
        "language": "cpp",
        "content": """extern "C" {

rocblas_status rocsolver_sgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* U,
                                const rocblas_int ldu,
                                float* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesvd_impl<float>(handle, left_svect, right_svect, m, n, A, lda, S,
                                                  U, ldu, V, ldv, E, fast_alg, info);
}

rocblas_status rocsolver_cgesvd(rocblas_handle handle,
                                const rocblas_svect left_svect,
                                const rocblas_svect right_svect,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* S,
                                rocblas_float_complex* U,
                                const rocblas_int ldu,
                                rocblas_float_complex* V,
                                const rocblas_int ldv,
                                float* E,
                                const rocblas_workmode fast_alg,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesvd_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, E, fast_alg, info);
}

} // extern C"""
    }],
    answer="""Complete C API for all GESVD precisions:

```cpp
extern "C" {

// Single precision real: S
rocblas_status rocsolver_sgesvd(
    rocblas_handle handle,
    const rocblas_svect left_svect,      // U computation mode
    const rocblas_svect right_svect,     // V computation mode
    const rocblas_int m,                 // Rows
    const rocblas_int n,                 // Columns
    float* A,                            // Input/output matrix (m×n)
    const rocblas_int lda,               // Leading dimension of A
    float* S,                            // Singular values (real, length min(m,n))
    float* U,                            // Left singular vectors (m×m or m×k)
    const rocblas_int ldu,               // Leading dimension of U
    float* V,                            // Right singular vectors (n×n or k×n)
    const rocblas_int ldv,               // Leading dimension of V
    float* E,                            // Off-diagonal (real, length min(m,n)-1)
    const rocblas_workmode fast_alg,    // Algorithm variant
    rocblas_int* info)                   // Convergence info
{
    return rocsolver::rocsolver_gesvd_impl<float>(
        handle, left_svect, right_svect, m, n, A, lda, S,
        U, ldu, V, ldv, E, fast_alg, info);
}

// Double precision real: D
rocblas_status rocsolver_dgesvd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    double* A,                           // Real matrix
    const rocblas_int lda,
    double* S,                           // Real singular values
    double* U,                           // Real left vectors
    const rocblas_int ldu,
    double* V,                           // Real right vectors
    const rocblas_int ldv,
    double* E,                           // Real off-diagonal
    const rocblas_workmode fast_alg,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesvd_impl<double>(
        handle, left_svect, right_svect, m, n, A, lda, S,
        U, ldu, V, ldv, E, fast_alg, info);
}

// Single precision complex: C
rocblas_status rocsolver_cgesvd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    rocblas_float_complex* A,            // Complex matrix
    const rocblas_int lda,
    float* S,                            // REAL singular values (not complex!)
    rocblas_float_complex* U,            // Complex left vectors
    const rocblas_int ldu,
    rocblas_float_complex* V,            // Complex right vectors
    const rocblas_int ldv,
    float* E,                            // REAL off-diagonal (not complex!)
    const rocblas_workmode fast_alg,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesvd_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S,
        U, ldu, V, ldv, E, fast_alg, info);
}

// Double precision complex: Z
rocblas_status rocsolver_zgesvd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    rocblas_double_complex* A,           // Complex matrix
    const rocblas_int lda,
    double* S,                           // REAL singular values (not complex!)
    rocblas_double_complex* U,           // Complex left vectors
    const rocblas_int ldu,
    rocblas_double_complex* V,           // Complex right vectors
    const rocblas_int ldv,
    double* E,                           // REAL off-diagonal (not complex!)
    const rocblas_workmode fast_alg,
    rocblas_workmode fast_alg,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesvd_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S,
        U, ldu, V, ldv, E, fast_alg, info);
}

} // extern "C"
```

**Key type distinctions:**

1. **Singular values always real:**
   - S/D: S is float*/double* (matches A type)
   - C/Z: S is float*/double* (NOT complex!)
   - Mathematical: σᵢ ≥ 0 are real by definition

2. **Off-diagonal always real:**
   - E has same type as S
   - Bidiagonal matrix B has real diagonal and off-diagonal
   - Even for complex A, B is real

3. **Template instantiation:**
```cpp
// Implementation uses two template parameters:
template <typename T, typename TT, typename W>
rocblas_status rocsolver_gesvd_impl(...)

// T: Matrix element type (float/double/complex)
// TT: Singular value type (float/double, always real)

// S: T=float,  TT=float
// D: T=double, TT=double
// C: T=rocblas_float_complex,  TT=float
// Z: T=rocblas_double_complex, TT=double
```

**GESVD-specific parameters:**

1. **E array**: Off-diagonal elements
   - Not in GESDD
   - Useful for debugging/analysis

2. **fast_alg**: Algorithm variant
```cpp
enum rocblas_workmode {
    rocblas_outofplace,  // Fast thin SVD (extra memory)
    rocblas_inplace      // Memory-efficient thin SVD
};
```

**Usage examples:**

```cpp
// Real matrix SVD
float A[1000*100], S[100], U[1000*100], V[100*100], E[99];
int info;

rocsolver_sgesvd(handle,
                 rocblas_svect_singular,  // Thin U
                 rocblas_svect_all,       // Full V
                 1000, 100,               // Dimensions
                 A, 1000,                 // Input matrix
                 S,                       // Singular values
                 U, 1000,                 // Left vectors
                 V, 100,                  // Right vectors
                 E,                       // Off-diagonal
                 rocblas_outofplace,      // Fast algorithm
                 &info);

// Complex matrix SVD
rocblas_double_complex A[500*500];
double S[500], E[499];  // Real, not complex!
rocblas_double_complex U[500*500], V[500*500];

rocsolver_zgesvd(handle,
                 rocblas_svect_all,
                 rocblas_svect_all,
                 500, 500,
                 A, 500,
                 S,       // Real singular values
                 U, 500,
                 V, 500,
                 E,       // Real off-diagonal
                 rocblas_inplace,
                 &info);
```

**Differences from GESDD:**

| Aspect | GESVD | GESDD |
|--------|-------|-------|
| E array | Required | Not present |
| fast_alg | User-controlled | Automatic |
| svect modes | 4 (none/singular/all/overwrite) | 3 (none/singular/all) |
| Algorithm | Bidiagonalization + QR | Eigendecomposition + D&C |
| Speed | Slower | Faster |
| Accuracy | Better (ill-conditioned) | Good (well-conditioned) |""",
    rationale="GESVD provides four precision variants (S/D/C/Z) with LAPACK-compatible interfaces. For complex matrices, singular values S and off-diagonal E are real (not complex), reflecting the mathematical property that singular values are non-negative real numbers. GESVD uniquely exposes the E array and fast_alg parameter for user control.",
    tags=["coding", "api", "precision", "template", "lapack"]
))

# Write all entries to JSONL file
output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_gesvd.jsonl"
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
