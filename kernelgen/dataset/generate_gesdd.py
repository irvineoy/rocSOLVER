#!/usr/bin/env python3
"""Generate GESDD dataset entries following DATASET.md schema."""

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

# Entry 1: gesdd_flip_signs kernel - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesdd",
    instruction="Implement the gesdd_flip_signs kernel that ensures all singular values are positive by flipping signs in the diagonal of R and corresponding columns of U and V. Explain why this is necessary after QR/LQ factorization.",
    context_text="GESDD computes SVD using divide-and-conquer. After QR factorization of AV (for m>=n), the diagonal elements of R may be negative. Since singular values must be non-negative, we flip signs in R's diagonal and corresponding columns of the orthogonal matrices.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """template <typename T, typename S>
ROCSOLVER_KERNEL void gesdd_flip_signs(const rocblas_int n,
                                       S* SS,
                                       const rocblas_stride strideS,
                                       T* R,
                                       const rocblas_int ldr,
                                       const rocblas_stride strideR,
                                       T* Q,
                                       const rocblas_int ldq,
                                       const rocblas_stride strideQ,
                                       const rocblas_int batch_count)
{
    rocblas_int tid = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int bid = blockIdx.y;

    for(b = bid; b < batch_count; b += gridDim.y)
    {
        S* Sigma = SS + b * strideS;
        T* D = R + b * strideR;
        T* U = Q + b * strideQ;

        for(j = tid; j < n; j += gridDim.x * blockDim.x)
        {
            for(k = 0; k < n; k++)
            {
                sigma = std::real(D[k + k * ldr]);
                if(sigma < 0)
                {
                    U[j + k * ldq] = -U[j + k * ldq];
                }

                if(j == 0)
                {
                    Sigma[k] = std::abs(sigma);
                }
            }
        }
    }
}"""
    }],
    answer="""The gesdd_flip_signs kernel corrects negative diagonal elements after QR/LQ factorization:

```cpp
// For each diagonal element k:
sigma = std::real(D[k + k * ldr]);  // Extract diagonal element
if(sigma < 0)
{
    U[j + k * ldq] = -U[j + k * ldq];  // Flip column k of U
}

if(j == 0)  // First thread stores absolute value
{
    Sigma[k] = std::abs(sigma);  // Store as positive singular value
}
```

Key points:
1. **QR non-uniqueness**: QR decomposition doesn't guarantee positive diagonal in R
2. **Sign flipping**: If R[k,k] < 0, negate column k of Q to maintain A = Q*R
3. **Singular value convention**: SVD requires S[k] ≥ 0 by definition
4. **Thread distribution**: tid handles rows j, all threads process all columns k
5. **Batch processing**: bid iterates over batch dimension with stride

Implementation details:
- Each thread j flips its element in column k of U if needed
- Only thread j=0 writes the absolute value to Sigma array
- Works for both m>=n path (after GEQRF) and m<n path (after GELQF)""",
    rationale="QR factorization produces A*V = Q*R where R may have negative diagonal elements. To ensure singular values are non-negative, we flip the sign of the corresponding column of Q, maintaining the equality A*V = Q*R while ensuring R has positive diagonal.",
    tags=["coding", "svd", "sign-correction", "kernel"]
))

# Entry 2: GESDD algorithm for m >= n - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesdd",
    instruction="Explain the complete GESDD algorithm for the case m >= n. Why does it compute the eigendecomposition of A^*A instead of directly computing SVD? What are the three main stages?",
    context_text="GESDD uses divide-and-conquer to compute SVD efficiently. For tall matrices (m>=n), it reduces the problem to eigendecomposition of the Gramian matrix A^*A, which is smaller and symmetric/Hermitian.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """// Stage 1: Compute -A'A (negative for descending order)
rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n,
               m, &neg_one, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, V_gemm,
               0, ldv_gemm, strideV_gemm, batch_count, (T**)workArr);

// Stage 2: Eigendecomposition V_gemm = V*D*V^* where D has eigenvalues
rocsolver_syevd_heevd_template<false, STRIDED, T>(
    handle, rocblas_evect_original, rocblas_fill_upper, n, V_gemm, 0, ldv_gemm,
    strideV_gemm, S, strideS, (SS*)workArr, strideS, info, batch_count, scalars, work1,
    work2, work3, (SS*)UVtmpZ, (rocblas_int*)splits, (T*)tmptau_W, (T*)tau, (T**)workArr2);

// Stage 3: Compute AV to get left singular vectors
rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, n, &one, A,
               shiftA, lda, strideA, V_gemm, 0, ldv_gemm, strideV_gemm, &zero, U_gemm, 0,
               ldu_gemm, strideU_gemm, batch_count, (T**)workArr);

// Stage 4: QR factorization of AV to extract U and S
rocsolver_geqrf_template<false, STRIDED, T>(handle, m, n, U_gemm, 0, ldu_gemm, strideU_gemm,
                                            (T*)work5_ipiv, n, batch_count, scalars, work2,
                                            (T*)work3, (T*)work4, (T**)workArr);"""
    }],
    answer="""GESDD algorithm for m >= n uses eigendecomposition of A^*A:

**Why eigendecomposition of A^*A?**
- SVD: A = U*Σ*V^* where U is m×n, Σ diagonal, V^* is n×n
- Property: A^*A = V*Σ²*V^* (eigendecomposition of Gramian)
- Smaller problem: n×n eigendecomposition vs m×n SVD
- Symmetric/Hermitian: A^*A allows faster divide-and-conquer (SYEVD)

**Three main stages:**

1. **Compute Gramian**: -A^*A (negative for descending eigenvalue order)
   - Output: n×n symmetric matrix
   - Why negative? SYEVD outputs ascending order, SVD needs descending

2. **Eigendecomposition**: SYEVD on -A^*A
   - Output: V (eigenvectors = right singular vectors), eigenvalues
   - Note: Eigenvalues of -A^*A are negative σ²

3. **Recover left singular vectors**: Compute A*V then QR factorize
   - A*V = U*Σ (approximate, needs normalization)
   - QR(A*V) = U*R where R diagonal ≈ Σ
   - Extract: U from Q, S from |diag(R)|

4. **Sign correction**: gesdd_flip_signs ensures S >= 0

**Complexity**: O(mn² + n³) vs O(mn² + m²n) for direct SVD when m >> n""",
    rationale="Computing eigendecomposition of the smaller Gramian matrix A^*A (n×n) is more efficient than direct SVD for tall matrices. The eigenvectors of A^*A are exactly the right singular vectors V, and the singular values can be recovered as square roots of eigenvalues.",
    tags=["algorithm", "svd", "eigendecomposition", "divide-conquer"]
))

# Entry 3: Workspace calculation - L3 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesdd",
    instruction="Implement the workspace calculation for GESDD when m >= n and left singular vectors are requested. Account for SYEVD, GEQRF, ORGQR, and temporary storage requirements.",
    context_text="GESDD requires complex workspace management, reusing buffers across SYEVD (eigendecomposition), GEQRF (QR factorization), and ORGQR (Q generation). Different paths (m>=n vs m<n) have different requirements.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """if(m >= n)
{
    // Requirements for Divide-and-Conquer eigensolver
    rocsolver_syevd_heevd_getMemorySize<BATCHED, T, SS>(
        handle, rocblas_evect_original, rocblas_fill_upper, n, batch_count, &a1, &b1, &c1, &d1,
        &e1, &f1, &g1, &h1, size_workArr2);

    // Requirements for QR factorization
    rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &a2, &b2, &c2, &d2, &f2);
    if(left_svect != rocblas_svect_none)
        rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, (left_full ? m : n), n, batch_count,
                                                        &unused, &b3, &c3, &d3, &f3);

    // Extra requirements for temporary V & U storage
    *size_VUtmp = sizeof(T) * n * n * batch_count;
    if(!leftv)
        size_UVtmp = sizeof(T) * m * n * batch_count;
}"""
    }],
    answer="""Workspace calculation for GESDD (m >= n, left_svect requested):

```cpp
// Query SYEVD workspace for n×n eigendecomposition
rocsolver_syevd_heevd_getMemorySize<BATCHED, T, SS>(
    handle, rocblas_evect_original, rocblas_fill_upper, n, batch_count,
    &a1,  // scalars
    &b1,  // work1
    &c1,  // work2
    &d1,  // work3
    &e1,  // tmpz
    &f1,  // splits
    &g1,  // tmptau_W
    &h1,  // tau
    size_workArr2);

// Query GEQRF workspace for m×n QR factorization
rocsolver_geqrf_getMemorySize<BATCHED, T>(
    m, n, batch_count,
    &a2,  // scalars
    &b2,  // work2
    &c2,  // work3
    &d2,  // work4
    &f2); // workArr

// Query ORGQR workspace if generating U
if(left_svect != rocblas_svect_none)
{
    rocblas_int ncol_u = (left_full ? m : n);  // Full U or thin U
    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(
        m, ncol_u, n, batch_count,
        &unused, &b3, &c3, &d3, &f3);
}

// Temporary storage for V (eigenvectors from SYEVD)
*size_VUtmp = sizeof(T) * n * n * batch_count;

// If not computing U, need temp storage for A*V
if(!leftv)
    size_UVtmp = sizeof(T) * m * n * batch_count;

// Combine requirements (max across reused buffers)
*size_scalars = std::max({a1, a2});
*size_work2 = std::max({c1, b2, b3});
*size_work3 = std::max({d1, c2, c3});
*size_work4 = std::max({d2, d3});
```

Key insights:
1. Buffers reused: work2/work3/work4 shared across SYEVD, GEQRF, ORGQR
2. VUtmp: Always needed for V (n×n)
3. UVtmp: Only if U not requested (user doesn't provide U buffer)
4. left_full: Determines ORGQR size (m×m vs m×n)""",
    rationale="GESDD reuses workspace across multiple stages. SYEVD, GEQRF, and ORGQR have overlapping workspace needs. By computing the maximum requirement for each reused buffer, we minimize total memory allocation while ensuring each subroutine has sufficient workspace.",
    tags=["coding", "memory", "workspace", "optimization"]
))

# Entry 4: m < n path with LQ factorization - L2
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesdd",
    instruction="Trace the algorithm execution for GESDD when m < n (wide matrices). How does it differ from the m >= n case? What factorization replaces QR?",
    context_text="For wide matrices (m < n), GESDD computes eigendecomposition of A*A^* instead of A^*A, and uses LQ factorization instead of QR to extract singular values and vectors.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """else  // m < n
{
    // Compute -AA'
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, m, m,
                   n, &neg_one, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, U_gemm,
                   0, ldu_gemm, strideU_gemm, batch_count, (T**)workArr);

    rocsolver_syevd_heevd_template<false, STRIDED, T>(
        handle, rocblas_evect_original, rocblas_fill_upper, m, U_gemm, 0, ldu_gemm,
        strideU_gemm, S, strideS, (SS*)workArr, strideS, info, batch_count, scalars, work1,
        work2, work3, (SS*)UVtmpZ, (rocblas_int*)splits, (T*)tmptau_W, (T*)tau, (T**)workArr2);

    // Compute U^*A
    rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, m, n,
                   m, &one, U_gemm, 0, ldu_gemm, strideU_gemm, A, shiftA, lda, strideA, &zero,
                   V_gemm, 0, ldv_gemm, strideV_gemm, batch_count, (T**)workArr);

    // Apply LQ factorization to U^*A, obtaining S from the diagonal of L and V^* from Q
    rocsolver_gelqf_template<false, STRIDED, T>(handle, m, n, V_gemm, 0, ldv_gemm, strideV_gemm,
                                                (T*)work5_ipiv, m, batch_count, scalars, work2,
                                                (T*)work3, (T*)work4, (T**)workArr);
}"""
    }],
    answer="""GESDD for m < n (wide matrices) algorithm trace:

**Stage 1: Compute -A*A^* (m×m matrix)**
```cpp
rocsolver_gemm(..., m, m, n, &neg_one, A, ..., A^*, ...);
// Output: -A*A^* in U_gemm buffer
```

**Stage 2: Eigendecomposition of -A*A^***
```cpp
rocsolver_syevd_heevd_template(..., m, U_gemm, ..., S, ...);
// Output: U (left singular vectors), eigenvalues in S
// Note: Eigenvectors of A*A^* are left singular vectors
```

**Stage 3: Compute U^*A**
```cpp
rocsolver_gemm(..., m, n, m, &one, U_gemm^*, ..., A, ..., V_gemm, ...);
// Output: U^*A = Σ*V^* (approximately)
```

**Stage 4: LQ factorization of U^*A**
```cpp
rocsolver_gelqf_template(..., m, n, V_gemm, ...);
// U^*A = L*Q where L is m×m lower triangular
// Output: L in V_gemm, Q implicitly stored
```

**Stage 5: Sign correction**
```cpp
gesdd_flip_signs(..., m, S, ..., V_gemm, ..., U_gemm, ...);
// Extract S from |diag(L)|, flip signs in U and V^*
```

**Key differences from m >= n:**
- Gramian: A*A^* (m×m) vs A^*A (n×n)
- Eigenvectors: U from A*A^* vs V from A^*A
- Factorization: LQ vs QR
- Product: U^*A vs A*V
- Dimension: Smaller eigenproblem (m < n)""",
    rationale="For wide matrices, computing eigendecomposition of A*A^* (m×m) is more efficient than A^*A (n×n). The eigenvectors are the left singular vectors U. LQ factorization of U^*A extracts the right singular vectors V^* (from Q) and singular values (from diagonal of L).",
    tags=["algorithm", "svd", "lq-factorization", "wide-matrix"]
))

# Entry 5: Argument validation - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesdd",
    instruction="Implement the svect parameter validation for GESDD. Handle the three options: svect_all, svect_singular, svect_none. Check leading dimension requirements for each case.",
    context_text="GESDD has flexible output options controlled by left_svect and right_svect parameters. svect_all computes full orthogonal matrices, svect_singular computes only the first min(m,n) vectors, and svect_none skips computation.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """auto is_svect_all = [](auto s) -> bool { return s == rocblas_svect_all; };
auto is_svect_singular = [](auto s) -> bool { return s == rocblas_svect_singular; };
auto is_svect_none = [](auto s) -> bool { return s == rocblas_svect_none; };

// Left singular vectors need ldu >= m
if(is_svect(left_svect) && (ldu < m))
{
    return rocblas_status_invalid_size;
}

// Right singular vectors need ldv >= n or ldv >= min(m, n),
// depending on choice of svect_all or svect_singular
if(is_svect_all(right_svect) && (ldv < n))
{
    return rocblas_status_invalid_size;
}

if(is_svect_singular(right_svect) && (ldv < std::min(m, n)))
{
    return rocblas_status_invalid_size;
}"""
    }],
    answer="""SVECT parameter validation for GESDD:

```cpp
// Define svect option checkers
auto is_svect_all = [](auto s) -> bool { return s == rocblas_svect_all; };
auto is_svect_singular = [](auto s) -> bool { return s == rocblas_svect_singular; };
auto is_svect_none = [](auto s) -> bool { return s == rocblas_svect_none; };
auto is_svect = [&](auto s) -> bool {
    return is_svect_all(s) || is_svect_singular(s);
};

// Validate svect parameters are valid enum values
auto invalid_svect_option = [&](auto s) -> bool {
    return !(is_svect_all(s) || is_svect_singular(s) || is_svect_none(s));
};

if(invalid_svect_option(left_svect) || invalid_svect_option(right_svect))
{
    return rocblas_status_invalid_value;
}

// Validate leading dimensions based on svect choice

// Left singular vectors: ldu >= m (for both svect_all and svect_singular)
if(is_svect(left_svect) && (ldu < m))
{
    return rocblas_status_invalid_size;
}

// Right singular vectors (svect_all): ldv >= n (full V is n×n)
if(is_svect_all(right_svect) && (ldv < n))
{
    return rocblas_status_invalid_size;
}

// Right singular vectors (svect_singular): ldv >= min(m,n) (thin V)
if(is_svect_singular(right_svect) && (ldv < std::min(m, n)))
{
    return rocblas_status_invalid_size;
}
```

**Leading dimension requirements:**
- `left_svect=svect_all`: U is m×m, need ldu >= m
- `left_svect=svect_singular`: U is m×min(m,n), need ldu >= m
- `right_svect=svect_all`: V is n×n, need ldv >= n
- `right_svect=svect_singular`: V is n×min(m,n), need ldv >= min(m,n)
- `svect_none`: Skip computation, no leading dimension check""",
    rationale="GESDD offers three output modes: full orthogonal matrices (svect_all), thin matrices with only min(m,n) vectors (svect_singular), or skip computation (svect_none). Leading dimension requirements differ: full V needs ldv >= n, while thin V only needs ldv >= min(m,n).",
    tags=["coding", "validation", "api", "leading-dimension"]
))

# Entry 6: Negative sign in Gramian computation - L2
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesdd",
    instruction="Why does GESDD compute -A^*A instead of A^*A? How does this affect the eigenvalues and the final singular value ordering?",
    context_text="GESDD computes the Gramian with a negative sign. This is related to the ordering convention: SVD outputs singular values in descending order, while SYEVD outputs eigenvalues in ascending order.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """// Compute -A'A; negative sign is necessary as `gesdd` outputs singular
// values in non-ascending order, while `syevd` outputs eigenvalues in
// non-decreasing order.
T neg_one = T(-1);
rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n,
               m, &neg_one, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, V_gemm,
               0, ldv_gemm, strideV_gemm, batch_count, (T**)workArr);"""
    }],
    answer="""Why compute -A^*A instead of A^*A:

**Ordering conventions:**
- **SVD convention**: Singular values in descending order σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0
- **SYEVD convention**: Eigenvalues in ascending order λ₁ ≤ λ₂ ≤ ... ≤ λₙ

**Mathematical relationship:**
- A^*A has eigenvalues λᵢ = σᵢ² (all non-negative)
- If we compute SYEVD(A^*A), we get λ₁ ≤ λ₂ ≤ ... ≤ λₙ
- This gives σ₁ ≤ σ₂ ≤ ... ≤ σₙ (ascending, wrong order!)

**Solution with negative sign:**
- Compute B = -A^*A instead
- SYEVD(B) gives eigenvalues μ₁ ≤ μ₂ ≤ ... ≤ μₙ where μᵢ = -σᵢ²
- Since μ₁ ≤ μ₂ ≤ ... ≤ μₙ and all μᵢ ≤ 0
- We have -σ₁² ≤ -σ₂² ≤ ... ≤ -σₙ²
- Therefore σ₁ ≥ σ₂ ≥ ... ≥ σₙ (descending, correct order!)

**Implementation:**
```cpp
T neg_one = T(-1);
rocsolver_gemm(..., n, n, m, &neg_one, A^*, A, &zero, V_gemm, ...);
// Result: V_gemm = -A^*A

rocsolver_syevd_heevd_template(..., V_gemm, ..., S, ...);
// SYEVD outputs: eigenvalues in S (ascending μᵢ = -σᵢ²)
//                eigenvectors in V_gemm (order matches eigenvalues)

// S now contains -σ² values in ascending order
// Later, gesdd_flip_signs computes S[k] = std::abs(diag[k]) to get σ values
```

**Result:** Singular values automatically in descending order without explicit sorting.""",
    rationale="SYEVD outputs eigenvalues in ascending order, but SVD convention requires singular values in descending order. By computing -A^*A instead of A^*A, the eigenvalues become -σ₁² ≤ -σ₂² ≤ ... ≤ -σₙ², which means σ₁ ≥ σ₂ ≥ ... ≥ σₙ, achieving the desired descending order.",
    tags=["algorithm", "eigenvalue-ordering", "svd", "mathematical-trick"]
))

# Entry 7: V transpose operation - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesdd",
    instruction="Implement the transpose operation that converts the computed eigenvectors V to V^* for output. Why is this necessary for LAPACK API consistency?",
    context_text="After computing eigenvectors, GESDD transposes V before returning to the user. This is required for LAPACK API compatibility where V^* (not V) is returned.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """// Transpose V (for consistency with LAPACK's API)
if(rightv)
{
    rocblas_int blocks_n = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks_n, blocks_n, batch_count),
                            dim3(BS2, BS2, 1), 0, stream,
                            rocblas_operation_conjugate_transpose, n, n, V_gemm, 0,
                            ldv_gemm, strideV_gemm, V, 0, ldv, strideV);
}"""
    }],
    answer="""V transpose operation for LAPACK API compatibility:

```cpp
if(rightv)  // Only if user requested right singular vectors
{
    rocblas_int blocks_n = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(
        (copy_trans_mat<T, T>),
        dim3(blocks_n, blocks_n, batch_count),  // Grid: cover n×n for each batch
        dim3(BS2, BS2, 1),                       // Block: BS2×BS2 threads
        0,                                        // Shared memory
        stream,
        rocblas_operation_conjugate_transpose,   // Conjugate transpose for complex
        n, n,                                     // Matrix dimensions
        V_gemm, 0, ldv_gemm, strideV_gemm,      // Source: computed eigenvectors
        V, 0, ldv, strideV);                     // Dest: user output buffer
}
```

**Why transpose is necessary:**

1. **SYEVD output**: Eigenvectors V where A^*A = V*Λ*V^*
   - Each column of V is an eigenvector
   - V is stored column-major in V_gemm

2. **SVD definition**: A = U*Σ*V^*
   - Right singular vectors are columns of V
   - But LAPACK API returns V^* (not V)

3. **API consistency**: LAPACK's GESDD returns V^*
   - User expects: A = U*Σ*V (where V is the output matrix)
   - So we must return V^* to maintain A = U*Σ*(output)

4. **Complex case**: Use conjugate_transpose (not just transpose)
   - For complex matrices, we need V^H (Hermitian transpose)
   - copy_trans_mat handles both real (transpose) and complex (conjugate)

**Grid/block sizing:**
- blocks_n: Covers n columns (round up division)
- BS2×BS2: Tile size for efficient coalesced memory access
- batch_count: Third dimension for batched processing""",
    rationale="LAPACK's GESDD API returns V^* (conjugate transpose of right singular vectors) rather than V. After computing eigenvectors V from SYEVD, we must transpose V to V^* before returning to the user for API compatibility.",
    tags=["coding", "transpose", "api", "lapack-compatibility"]
))

# Entry 8: Temporary storage management - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesdd",
    instruction="Explain the conditional temporary storage allocation for U and V in GESDD. When does GESDD allocate temporary buffers versus use user-provided buffers?",
    context_text="GESDD conditionally allocates temporary storage based on whether the user requests left/right singular vectors. If svect_none, GESDD allocates temporary buffers internally.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """// Extra requirements for temporary V & U storage
*size_VUtmp = sizeof(T) * n * n * batch_count;
if(!leftv)
    size_UVtmp = sizeof(T) * m * n * batch_count;

// In template:
T* U_gemm = (leftv ? U : (T*)UVtmpZ);
rocblas_int ldu_gemm = (leftv ? ldu : m);
rocblas_int strideU_gemm = (leftv ? strideU : m * n);"""
    }],
    answer="""Temporary storage management in GESDD:

**Storage allocation logic:**

```cpp
// In getMemorySize (m >= n case):
*size_VUtmp = sizeof(T) * n * n * batch_count;  // Always allocate for V
if(!leftv)
    size_UVtmp = sizeof(T) * m * n * batch_count;  // Only if U not requested

// In template execution:
T* U_gemm = (leftv ? U : (T*)UVtmpZ);
rocblas_int ldu_gemm = (leftv ? ldu : m);
rocblas_int strideU_gemm = (leftv ? strideU : m * n);
```

**Why temporary storage is needed:**

1. **VUtmp (always allocated for m>=n):**
   - Purpose: Store eigenvectors from SYEVD, then transpose to user's V
   - Size: n×n (full eigenvector matrix)
   - Why temporary? SYEVD writes column-major, need to transpose to V^*
   - Even if rightv=false, intermediate storage needed for algorithm

2. **UVtmpZ (allocated if !leftv for m>=n):**
   - Purpose: Store A*V result when user doesn't want U
   - Size: m×n
   - Why? Algorithm needs A*V for QR factorization to extract singular values
   - If leftv=true, use user's U buffer directly

3. **Conditional pointer assignment:**
```cpp
T* U_gemm = (leftv ? U : (T*)UVtmpZ);
// If user wants U: write directly to user buffer
// If user doesn't want U: write to temporary buffer
```

4. **Leading dimension adaptation:**
```cpp
rocblas_int ldu_gemm = (leftv ? ldu : m);
// User buffer: use user's leading dimension
// Temporary: use tight packing (ldu_gemm = m)
```

**Memory efficiency:**
- VUtmp: ~n² elements (always needed for V computation)
- UVtmpZ: ~m*n elements (only when user doesn't provide U)
- Total saved: ~m*n elements when leftv=true (can be large for m >> n)""",
    rationale="GESDD allocates temporary storage strategically. VUtmp is always needed to store intermediate eigenvectors before transposing to V^*. UVtmpZ is only allocated when the user doesn't request left singular vectors (svect_none), because the algorithm still needs to compute A*V internally to extract singular values from QR factorization.",
    tags=["coding", "memory", "conditional-allocation", "optimization"]
))

# Entry 9: ORGQR call for U generation - L2 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L2",
    interface="gesdd",
    instruction="Implement the ORGQR call that generates the orthogonal matrix U from the QR factorization of A*V. Handle both svect_all (full U) and svect_singular (thin U) cases.",
    context_text="After GEQRF factorizes A*V, the orthogonal factor Q (representing U) is stored implicitly as Householder reflectors. ORGQR generates the explicit matrix U, with size depending on svect parameter.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """if(leftv)
    rocsolver_orgqr_ungqr_template<false, STRIDED, T>(
        handle, m, (left_full ? m : n), n, U_gemm, 0, ldu_gemm, strideU_gemm, (T*)work5_ipiv,
        n, batch_count, scalars, (T*)work2, (T*)work3, (T*)work4, (T**)workArr);"""
    }],
    answer="""ORGQR call for U generation in GESDD:

```cpp
if(leftv)  // Only if user requested left singular vectors
{
    rocsolver_orgqr_ungqr_template<false, STRIDED, T>(
        handle,
        m,                        // Number of rows in U
        (left_full ? m : n),      // Number of columns: m (full) or n (thin)
        n,                         // Number of reflectors
        U_gemm,                    // Matrix containing QR factorization
        0,                         // Shift
        ldu_gemm,                  // Leading dimension
        strideU_gemm,              // Stride for batched
        (T*)work5_ipiv,            // Householder scalars (tau)
        n,                         // Stride for tau
        batch_count,               // Number of matrices
        scalars,                   // Workspace
        (T*)work2,                 // Workspace
        (T*)work3,                 // Workspace
        (T*)work4,                 // Workspace
        (T**)workArr);             // Array of pointers
}
```

**Parameters explained:**

1. **m (nrows)**: U always has m rows (matches A's row dimension)

2. **ncols: (left_full ? m : n)**
   - `left_full=true` (svect_all): Generate full m×m orthogonal matrix U
   - `left_full=false` (svect_singular): Generate thin m×n matrix U
   - Thin U: Only first n columns (corresponding to n singular values)

3. **n (reflectors)**: Number of Householder reflectors from GEQRF
   - GEQRF of m×n matrix produces n reflectors
   - These n reflectors generate the first n columns of Q

4. **work5_ipiv**: Contains tau values (Householder scalars) from GEQRF

**Two output modes:**

**Mode 1: svect_all (left_full=true)**
```
U is m×m, full orthogonal matrix
First n columns: Span of A*V
Last (m-n) columns: Orthogonal complement
```

**Mode 2: svect_singular (left_full=false)**
```
U is m×n, thin matrix
All n columns: Span of A*V (the actual left singular vectors)
More efficient: Saves m*(m-n) elements
```

**Why conditional:**
- If leftv=false: Skip this call entirely (user doesn't want U)
- If leftv=true: Generate U with appropriate size based on left_full""",
    rationale="ORGQR generates the explicit orthogonal matrix U from the implicit Householder reflectors produced by GEQRF. The number of columns depends on left_svect: svect_all produces the full m×m matrix, while svect_singular produces only the first n columns (thin U), which are the actual left singular vectors.",
    tags=["coding", "orgqr", "householder", "singular-vectors"]
))

# Entry 10: GESDD vs GESVD comparison - L3
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesdd",
    instruction="Compare GESDD (divide-and-conquer) with GESVD (QR iteration). What are the algorithmic differences, computational complexity, and when should each be used?",
    context_text="rocSOLVER provides two SVD implementations: GESDD uses divide-and-conquer for faster computation, while GESVD uses QR iteration for better accuracy and reliability on ill-conditioned matrices.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.hpp",
        "language": "cpp",
        "content": """// GESDD: Divide-and-conquer via eigendecomposition
rocsolver_syevd_heevd_template<false, STRIDED, T>(
    handle, rocblas_evect_original, rocblas_fill_upper, n, V_gemm, 0, ldv_gemm,
    strideV_gemm, S, strideS, (SS*)workArr, strideS, info, batch_count, ...);"""
    }],
    answer="""GESDD vs GESVD comparison:

**GESDD (Divide-and-Conquer):**

Algorithm:
1. Reduce to eigenvalue problem: Compute A^*A (or A*A^*)
2. SYEVD: Divide-and-conquer eigendecomposition
3. Recover singular vectors via QR/LQ factorization

Complexity:
- O(mn² + n³) for m >= n
- Dominated by: GEMM (mn²), SYEVD (n³), GEQRF (mn²)

Advantages:
- **Faster**: Divide-and-conquer is faster than QR iteration
- **Well-conditioned**: Works well for most matrices
- **Parallel-friendly**: SYEVD parallelizes better than QR iteration

Disadvantages:
- **Accuracy**: Squaring condition number (A^*A vs A)
- **Memory**: Requires O(n²) temporary storage for eigenvectors
- **Ill-conditioned**: May lose accuracy on matrices with large condition number

**GESVD (QR Iteration):**

Algorithm:
1. Bidiagonalization: A → U*B*V^* (B bidiagonal)
2. QR iteration: Compute SVD of B
3. Accumulate transformations

Complexity:
- O(mn² + n³) for m >= n
- Dominated by: Bidiagonalization (mn²), QR iteration (n³)

Advantages:
- **Accuracy**: Works directly on A (no squaring)
- **Robust**: Better for ill-conditioned matrices
- **Backward stable**: Strong numerical stability guarantees

Disadvantages:
- **Slower**: QR iteration slower than divide-and-conquer
- **Sequential**: QR iteration harder to parallelize

**When to use each:**

Use GESDD when:
- Speed is critical
- Matrix is well-conditioned (κ(A) < 1e8 for double)
- Standard use case (most applications)

Use GESVD when:
- Accuracy is critical
- Matrix is ill-conditioned
- Need backward stability guarantees
- Dealing with rank-deficient matrices near rank transition

**Condition number effect:**
- GESDD: Effective condition number κ(A^*A) = κ(A)²
- GESVD: Works with κ(A) directly
- For κ(A) = 1e8, GESDD sees κ(A^*A) = 1e16 (near machine precision!)""",
    rationale="GESDD is faster due to divide-and-conquer eigendecomposition but squares the condition number by computing A^*A. GESVD is more accurate and robust for ill-conditioned matrices because it works directly on A through bidiagonalization and QR iteration, avoiding condition number squaring.",
    tags=["algorithm", "svd", "comparison", "numerical-stability"]
))

# Entry 11: copymatA1 and addmatA1 kernels - L1 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L1",
    interface="gesdd",
    instruction="Implement the copymatA1 and addmatA1 kernels used by LARFB in GESDD. Explain their purpose in block reflector application.",
    context_text="LARFB (block reflector application) uses helper kernels to copy matrix blocks and add them back. These are needed for the workspace-based GEMM operations in LARFB.",
    code_blocks=[{
        "path": "library/src/auxiliary/rocauxiliary_larfb.hpp",
        "language": "cpp",
        "content": """template <typename T, typename U>
ROCSOLVER_KERNEL void copymatA1(const rocblas_int ldw,
                                const rocblas_int order,
                                U A,
                                const rocblas_int shiftA,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                T* tmptr)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Wp[i + j * ldw] = Ap[i + j * lda];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void addmatA1(const rocblas_int ldw,
                               const rocblas_int order,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* tmptr)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Ap[i + j * lda] -= Wp[i + j * ldw];
    }
}"""
    }],
    answer="""copymatA1 and addmatA1 kernels for LARFB:

**copymatA1: Copy matrix block A to workspace W**
```cpp
ROCSOLVER_KERNEL void copymatA1(
    const rocblas_int ldw,      // Leading dimension of workspace
    const rocblas_int order,    // Number of columns to copy
    U A,                         // Source matrix (batched pointer)
    const rocblas_int shiftA,   // Offset in A
    const rocblas_int lda,      // Leading dimension of A
    const rocblas_stride strideA,
    T* tmptr)                    // Workspace destination
{
    const auto b = hipBlockIdx_z;  // Batch index
    const auto j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // Column
    const auto i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  // Row

    if(i < ldw && j < order)
    {
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Wp[i + j * ldw] = Ap[i + j * lda];  // Copy: W = A
    }
}
```

**addmatA1: Subtract workspace W from matrix A**
```cpp
ROCSOLVER_KERNEL void addmatA1(/* same parameters */)
{
    // ... same indexing ...

    if(i < ldw && j < order)
    {
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Ap[i + j * lda] -= Wp[i + j * ldw];  // Update: A -= W
    }
}
```

**Purpose in LARFB (block reflector application):**

LARFB computes: C = (I - V*T*V^*) * C

Algorithm:
1. **copymatA1**: W = C (copy subset of C to workspace)
2. **GEMM**: W = V^* * W (compute V^* * C)
3. **TRMM**: W = T * W (apply triangular factor)
4. **GEMM**: W = V * W (compute V * T * V^* * C)
5. **addmatA1**: C -= W (apply update C = C - V*T*V^*C)

**Why separate kernels?**
- **copymatA1**: Preserve original C values in workspace
- **addmatA1**: Efficient in-place update without temporary
- **2D block distribution**: Each thread handles one element (i,j)
- **Batch dimension**: hipBlockIdx_z handles batch index

**Grid/block sizing:**
- Grid: (order/BS_x, ldw/BS_y, batch_count)
- Block: (BS_x, BS_y, 1) typically (16, 16, 1) or (32, 8, 1)
- Coalesced access: Column-major traversal for optimal memory throughput""",
    rationale="LARFB applies block Householder reflectors using a workspace-based algorithm. copymatA1 saves the original matrix block to workspace before transformation, while addmatA1 applies the computed update by subtracting the workspace from the matrix. This pattern avoids repeated memory reads during the multi-step LARFB computation.",
    tags=["coding", "kernel", "larfb", "block-reflector"]
))

# Entry 12: Production API with all precisions - L3 coding
time.sleep(0.001)
entries.append(create_entry(
    level="L3",
    interface="gesdd",
    instruction="Implement the C API wrappers for all four precisions (S/D/C/Z) of GESDD. Handle real vs complex singular value type distinction.",
    context_text="GESDD provides four precision variants following LAPACK naming: S (single real), D (double real), C (single complex), Z (double complex). Singular values are always real, even for complex input.",
    code_blocks=[{
        "path": "library/src/lapack/roclapack_gesdd.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgesdd(rocblas_handle handle,
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
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<float>(handle, left_svect, right_svect, m, n, A, lda, S,
                                                  U, ldu, V, ldv, info);
}

rocblas_status rocsolver_cgesdd(rocblas_handle handle,
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
                                rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}"""
    }],
    answer="""C API wrappers for all GESDD precisions:

```cpp
// Single precision real: S
rocblas_status rocsolver_sgesdd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    float* A,              // Real input matrix
    const rocblas_int lda,
    float* S,              // Real singular values
    float* U,              // Real left singular vectors
    const rocblas_int ldu,
    float* V,              // Real right singular vectors
    const rocblas_int ldv,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<float>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}

// Double precision real: D
rocblas_status rocsolver_dgesdd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    double* A,             // Real input matrix
    const rocblas_int lda,
    double* S,             // Real singular values
    double* U,             // Real left singular vectors
    const rocblas_int ldu,
    double* V,             // Real right singular vectors
    const rocblas_int ldv,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<double>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}

// Single precision complex: C
rocblas_status rocsolver_cgesdd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    rocblas_float_complex* A,    // Complex input matrix
    const rocblas_int lda,
    float* S,                     // REAL singular values (not complex!)
    rocblas_float_complex* U,    // Complex left singular vectors
    const rocblas_int ldu,
    rocblas_float_complex* V,    // Complex right singular vectors
    const rocblas_int ldv,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}

// Double precision complex: Z
rocblas_status rocsolver_zgesdd(
    rocblas_handle handle,
    const rocblas_svect left_svect,
    const rocblas_svect right_svect,
    const rocblas_int m,
    const rocblas_int n,
    rocblas_double_complex* A,   // Complex input matrix
    const rocblas_int lda,
    double* S,                    // REAL singular values (not complex!)
    rocblas_double_complex* U,   // Complex left singular vectors
    const rocblas_int ldu,
    rocblas_double_complex* V,   // Complex right singular vectors
    const rocblas_int ldv,
    rocblas_int* info)
{
    return rocsolver::rocsolver_gesdd_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv, info);
}
```

**Key distinctions:**

1. **Real vs Complex singular values:**
   - S/D (real): S is float*/double* (matches A type)
   - C/Z (complex): S is float*/double* (NOT complex!)
   - Mathematical fact: Singular values are always real ≥ 0

2. **Template instantiation:**
   - S: `rocsolver_gesdd_impl<float>`
   - D: `rocsolver_gesdd_impl<double>`
   - C: `rocsolver_gesdd_impl<rocblas_float_complex>`
   - Z: `rocsolver_gesdd_impl<rocblas_double_complex>`

3. **Type safety:**
   - Template parameter T: float/double/complex
   - Template parameter SS: float/double (singular value type)
   - Enforced in template: `template <typename T, typename SS, typename W>`

4. **LAPACK naming convention:**
   - S = Single precision real
   - D = Double precision real
   - C = Complex single precision
   - Z = Complex double precision (from German "Zahl")""",
    rationale="GESDD provides four precision variants following LAPACK conventions. For real matrices (S/D), all types match. For complex matrices (C/Z), singular values are real while U, V, and A are complex, reflecting the mathematical property that singular values are always non-negative real numbers.",
    tags=["coding", "api", "precision", "template", "lapack"]
))

# Write all entries to JSONL file
output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_gesdd.jsonl"
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
