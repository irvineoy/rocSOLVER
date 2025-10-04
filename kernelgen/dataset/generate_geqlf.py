#!/usr/bin/env python3
import json

entries = []

# ========== L1 ENTRIES ==========

# L1-1: Coding - Block partitioning for backward QL processing
entries.append({
    "level": "L1",
    "task_type": "coding",
    "instruction": "Implement the block partitioning logic for GEQLF that processes columns from right-to-left. Show how j, jb, mu, nu are calculated for backward QL factorization.",
    "code_blocks": [
        """rocblas_int k = std::min(m, n); // total number of pivots
rocblas_int nb = GEQxF_BLOCKSIZE;
rocblas_int ki = ((k - GEQxF_GEQx2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;
rocblas_int mu = m, nu = n;

while(j >= k - kk)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(k - j, nb); // number of columns in the block
    rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);

    // apply transformation to the rest of the matrix
    if(n - k + j > 0)
    {
        // compute block reflector
        rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise,
                                    m - k + j + jb, jb, A, shiftA + idx2D(0, n - k + j, lda),
                                    lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                                    strideW, batch_count, scalars, (T*)work_workArr, workArr);

        // apply the block reflector
        rocsolver_larfb_template<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
            rocblas_backward_direction, rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
            shiftA + idx2D(0, n - k + j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
            A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);
    }
    j -= nb;
    mu = m - k + j + jb;
    nu = n - k + j + jb;
}"""
    ],
    "answer": """```cpp
// GEQLF block partitioning for backward (right-to-left) QL factorization
rocblas_int k = std::min(m, n); // total pivots
rocblas_int nb = GEQxF_BLOCKSIZE; // block size (typically 32)
rocblas_int ki = ((k - GEQxF_GEQx2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;  // start from rightmost columns
rocblas_int mu = m, nu = n;

// Main loop: process column blocks from right to left
while(j >= k - kk)
{
    jb = std::min(k - j, nb); // block width (columns)

    // Factor current block: (m-k+j+jb) x jb at column (n-k+j)
    rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A,
                                shiftA + idx2D(0, n - k + j, lda),
                                lda, strideA, (ipiv + j), strideP,
                                batch_count, scalars, work_workArr,
                                Abyx_norms_trfact, diag_tmptr);

    // Apply transformation to left part of matrix
    if(n - k + j > 0)  // if there are columns to the left
    {
        // Build block reflector T (lower triangular, backward)
        rocsolver_larft_template<T>(handle, rocblas_backward_direction,
                                    rocblas_column_wise,
                                    m - k + j + jb, jb, A,
                                    shiftA + idx2D(0, n - k + j, lda),
                                    lda, strideA, (ipiv + j), strideP,
                                    Abyx_norms_trfact, ldw, strideW,
                                    batch_count, scalars,
                                    (T*)work_workArr, workArr);

        // Apply block reflector: Q'*A_left
        rocsolver_larfb_template<BATCHED, STRIDED, T>(
            handle, rocblas_side_left,
            rocblas_operation_conjugate_transpose,
            rocblas_backward_direction, rocblas_column_wise,
            m - k + j + jb,  // rows in current + above
            n - k + j,       // columns to the left
            jb, A,
            shiftA + idx2D(0, n - k + j, lda),
            lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
            A, shiftA, lda, strideA, batch_count,
            diag_tmptr, workArr);
    }

    j -= nb;  // move left by one block
    mu = m - k + j + jb;  // remaining rows
    nu = n - k + j + jb;  // remaining cols
}

// Factor final block (leftmost region)
if(mu > 0 && nu > 0)
    rocsolver_geql2_template<T>(handle, mu, nu, A, shiftA, lda, strideA,
                                ipiv, strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);
```

**Key aspects:**
- Start at j = k - kk + ki (rightmost), decrement by nb
- Block position: column n-k+j, rows 0 to m-k+j+jb-1
- Update columns 0 to n-k+j-1 (everything to the left)
- Similar structure to GERQF but column-wise instead of row-wise""",
    "rationale": "GEQLF block partitioning processes columns right-to-left to build L factor in bottom-right corner. Understanding how j/jb/mu/nu relate to block position and update regions is critical for implementing blocked QL factorization.",
    "tags": ["ql-factorization", "geqlf", "blocked-algorithm", "block-partitioning", "backward-processing", "coding"]
})

# L1-2: Analysis - LARFB trans parameter for QL vs QR
entries.append({
    "level": "L1",
    "task_type": "analysis",
    "instruction": "Explain why GEQLF uses rocblas_operation_conjugate_transpose for LARFB while GEQRF uses rocblas_operation_none. How does this relate to the matrix update formulas?",
    "code_blocks": [
        """// GEQLF: applies Q' from left with conjugate transpose
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
    rocblas_backward_direction, rocblas_column_wise, m - k + j + jb, n - k + j, jb, A,
    shiftA + idx2D(0, n - k + j, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);

// GEQRF: applies Q from left without transpose
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_none, rocblas_forward_direction,
    rocblas_column_wise, m - j, n - jb - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
    Abyx_norms_trfact, 0, ldw, strideW, A, shiftA + idx2D(j, jb + j, lda), lda, strideA,
    batch_count, diag_tmptr, workArr);"""
    ],
    "answer": """The trans parameter controls whether we apply Q or Q' (Q-conjugate-transpose) to the remaining matrix:

**GEQLF (rocblas_operation_conjugate_transpose):**
- QL factorization: `A = Q*L` where L is lower triangular in bottom-right
- Processes columns **right-to-left**: columns n-1, n-2, ..., 0
- Each block builds reflectors Q_i for rightmost columns
- Must apply `Q_i' * A_left` to update columns to the **left**
- Formula: `A_left := Q_i' * A_left` where Q_i' is the conjugate transpose
- LARFB operation: `trans = conjugate_transpose` → computes `Q'*A`

**GEQRF (rocblas_operation_none):**
- QR factorization: `A = Q*R` where R is upper triangular in top-left
- Processes columns **left-to-right**: columns 0, 1, ..., n-1
- Each block builds reflectors Q_i for leftmost columns
- Must apply `Q_i * A_right` to update columns to the **right**
- Formula: `A_right := Q_i * A_right` where Q_i is not transposed
- LARFB operation: `trans = none` → computes `Q*A`

**Why the difference:**

QL and QR factorizations build different products:
- **QR**: `A = Q*R` → successive Q_i applied from left → `A := Q_1 * Q_2 * ... * Q_k * R`
- **QL**: `A = Q*L` → successive Q_i applied from right → `A := Q_k * ... * Q_2 * Q_1 * L`

During factorization:
- **QR** accumulates: Apply Q_i to remaining A → `A_remaining := Q_i * A_remaining`
- **QL** accumulates: Apply Q_i' to remaining A → `A_remaining := Q_i' * A_remaining`

The transpose in QL ensures that when we reconstruct `A = Q*L`, the Q_i are composed in the correct order. At the end:
- QR: Q = Q_1 * Q_2 * ... * Q_k (left-to-right composition)
- QL: Q = Q_k * ... * Q_2 * Q_1 (right-to-left composition)

**Implementation detail:**
LARFB internally computes `H*A` or `H'*A` where `H = I - V*T*V'`. The trans parameter tells LARFB whether to use T or T' in the computation, which ultimately determines if we're applying Q or Q'.""",
    "rationale": "The transpose operation in LARFB is fundamental to understanding how QL vs QR factorizations compose their Householder reflectors. QL requires Q' updates during factorization to ensure proper Q*L ordering when reconstructing A.",
    "tags": ["ql-factorization", "geqlf", "larfb", "transpose-operation", "factorization-composition", "analysis"]
})

# L1-3: Coding - Tau conjugation in GEQL2 for complex matrices
entries.append({
    "level": "L1",
    "task_type": "coding",
    "instruction": "Implement the tau conjugation pattern in GEQL2 for complex matrices. Explain why tau must be conjugated before and after LARF, but not for real matrices.",
    "code_blocks": [
        """for(rocblas_int j = 0; j < dim; j++)
{
    // generate Householder reflector to work on column n - j - 1
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(m-j-1,n-j-1) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                            lda, strideA, 1, true);

    // conjugate tau
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

    // Apply Householder reflector to the rest of matrix from the left
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                            shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                            strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                            (T**)work_workArr);

    // restore original value of A(m-j-1,n-j-1)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                            shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);

    // restore tau
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);
}"""
    ],
    "answer": """```cpp
// GEQL2 main loop with tau conjugation for complex matrices
template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_geql2_template(/* ... */)
{
    // ... setup ...

    rocblas_int dim = std::min(m, n); // total pivots

    // Process columns from right to left
    for(rocblas_int j = 0; j < dim; j++)
    {
        // Step 1: Generate Householder reflector for column n-j-1
        // LARFG produces tau such that H*v = beta*e_1
        rocsolver_larfg_template(handle, m - j, A,
                                 shiftA + idx2D(m - j - 1, n - j - 1, lda),
                                 A, shiftA + idx2D(0, n - j - 1, lda),
                                 1, strideA, (ipiv + dim - j - 1), strideP,
                                 batch_count, (T*)work_workArr, Abyx_norms);

        // Step 2: Temporarily set diagonal to 1 for LARF
        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>),
                                dim3(batch_count, 1, 1), dim3(1, 1, 1),
                                0, stream, diag, 0, 1, A,
                                shiftA + idx2D(m - j - 1, n - j - 1, lda),
                                lda, strideA, 1, true);

        // Step 3: CONJUGATE tau (complex matrices only)
        // LARF expects tau for H' = I - conj(tau)*v*v'
        // But LARFG produced tau for H = I - tau*v*v'
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1,
                                        1, strideP, batch_count);

        // Step 4: Apply H' from left to remaining columns
        // H' * A_left where H' = I - conj(tau)*v*v'
        rocsolver_larf_template(handle, rocblas_side_left,
                                m - j, n - j - 1, A,
                                shiftA + idx2D(0, n - j - 1, lda),
                                1, strideA, (ipiv + dim - j - 1), strideP,
                                A, shiftA, lda, strideA,
                                batch_count, scalars, Abyx_norms,
                                (T**)work_workArr);

        // Step 5: Restore diagonal element
        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>),
                                dim3(batch_count, 1, 1), dim3(1, 1, 1),
                                0, stream, diag, 0, 1, A,
                                shiftA + idx2D(m - j - 1, n - j - 1, lda),
                                lda, strideA, 1);

        // Step 6: RESTORE tau to original value
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1,
                                        1, strideP, batch_count);
    }

    return rocblas_status_success;
}
```

**Why conjugate tau for complex matrices:**

1. **LARFG output:** Generates tau for `H = I - tau*v*v'` where v'*v = 1
   - For complex: tau is generally complex

2. **QL factorization needs H':** We need `A := Q*L` where Q is built right-to-left
   - During factorization: apply `H'` not `H`
   - `H' = (I - tau*v*v')' = I - conj(tau)*v*v'`

3. **LARF expectation:** LARF applies `H = I - alpha*v*v'` using the alpha parameter
   - We want `H' = I - conj(tau)*v*v'`
   - So we must pass `alpha = conj(tau)` to LARF

4. **Why not for real matrices:**
   - Real numbers: `conj(tau) = tau`
   - Conjugation is a no-op, so COMPLEX guard skips it

5. **Why restore:**
   - Output tau must match LAPACK convention
   - User expects tau for `H = I - tau*v*v'`, not `H'`

**Memory-efficient pattern:**
- Conjugate in-place before LARF
- Conjugate back after LARF
- Avoids allocating separate storage for conj(tau)""",
    "rationale": "Tau conjugation in GEQL2 is a subtle but critical detail for complex matrices. QL factorization applies H' during factorization but must store tau for H in the output, requiring temporary conjugation around LARF calls.",
    "tags": ["ql-factorization", "geql2", "complex-arithmetic", "tau-conjugation", "householder-reflectors", "coding"]
})

# ========== L2 ENTRIES ==========

# L2-1: Analysis - GEQLF vs GEQRF workspace differences
entries.append({
    "level": "L2",
    "task_type": "analysis",
    "instruction": "Compare GEQLF and GEQRF workspace requirements. Both use column-wise storage, so why do they have different LARFB workspace sizes?",
    "code_blocks": [
        """// GEQLF workspace
rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                          &s2, &unused);

// GEQRF workspace
rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                          &s2, &unused);"""
    ],
    "answer": """**Surprisingly, GEQLF and GEQRF have IDENTICAL workspace requirements!**

Both use column-wise Householder storage and LEFT-side LARFB application:

**LARFT workspace (both):**
- Dimension: (m, jb) for column-wise V matrix
- GEMM computes V'*V: `jb x m` times `m x jb` → `jb x jb` result
- Work buffer: `k * batch_count * sizeof(T)` (k = jb)
- T matrix: `jb x jb` (upper for QR, lower for QL)

**LARFB workspace (both):**
- Side: LEFT (applies Q or Q' from left)
- Dimensions: m rows, n-jb columns, jb reflectors
- Temp matrix W: `jb x (n-jb)` for computing `V'*A`
- Size: `jb * (n-jb) * batch_count * sizeof(T)`

**Why identical despite different processing order:**

| Aspect | GEQRF (left→right) | GEQLF (right→left) |
|--------|-------------------|-------------------|
| Storage | column-wise | column-wise |
| LARFB side | LEFT | LEFT |
| LARFT dimension | (m, jb) | (m, jb) |
| Update region | A[j:m, jb+j:n] | A[0:m-k+j+jb, 0:n-k+j] |
| W size | jb × (n-j-jb) | jb × (n-k+j) |
| Max W size | jb × (n-jb) | jb × (n-jb) |

Both have W ≈ `jb x (n-jb)` in the worst case.

**Contrast with row-wise algorithms (GELQF/GERQF):**

GELQF/GERQF use **row-wise** storage and **RIGHT-side** LARFB:
- LARFT dimension: (n, jb) instead of (m, jb)
- LARFB temp: `(m-jb) x jb` instead of `jb x (n-jb)`
- For tall matrices (m>n): GELQF/GERQF use LESS workspace
- For wide matrices (m<n): GEQLF/GEQRF use LESS workspace

**Example (1024x512 matrix, jb=32):**
- GEQLF/GEQRF: W = 32 × (512-32) = 32 × 480 = 15KB/batch
- GELQF/GERQF: W = (1024-32) × 32 = 992 × 32 = 32KB/batch

**Key insight:**
Processing order (left→right vs right→left) doesn't affect workspace when storage layout (column-wise vs row-wise) and application side (LEFT vs RIGHT) remain the same.""",
    "rationale": "Understanding that GEQLF and GEQRF have identical workspace despite different processing order highlights that workspace is determined by storage layout and application side, not by factorization direction. This is important for memory planning and algorithm selection.",
    "tags": ["ql-factorization", "workspace-management", "geqlf-vs-geqrf", "larft", "larfb", "analysis"]
})

# L2-2: Coding - Complex tau handling in blocked GEQLF
entries.append({
    "level": "L2",
    "task_type": "coding",
    "instruction": "Explain how tau conjugation from GEQL2 interacts with LARFT in blocked GEQLF. Does LARFT need special handling for complex tau values?",
    "code_blocks": [
        """// GEQL2 conjugates tau around LARF
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);
rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A, ...);
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

// GEQLF calls LARFT after GEQL2
rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A, ...);
rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_column_wise, ...);"""
    ],
    "answer": """```cpp
// GEQLF block iteration - tau handling for complex matrices

// Step 1: GEQL2 factors the current block
// Internally conjugates/restores tau around each LARF call
// Output: tau[j:j+jb] contains values for H_i = I - tau_i*v_i*v_i'
rocsolver_geql2_template<T>(handle, m - k + j + jb, jb, A,
                            shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP,
                            batch_count, scalars, work_workArr,
                            Abyx_norms_trfact, diag_tmptr);

// At this point: tau[j:j+jb] are in STANDARD form (for H, not H')

// Step 2: Build block reflector T
// LARFT builds T such that Q = H_1 * H_2 * ... * H_jb = I - V*T*V'
// where each H_i = I - tau_i * v_i * v_i'
rocsolver_larft_template<T>(handle, rocblas_backward_direction,
                            rocblas_column_wise,
                            m - k + j + jb, jb, A,
                            shiftA + idx2D(0, n - k + j, lda),
                            lda, strideA, (ipiv + j), strideP,
                            Abyx_norms_trfact, ldw, strideW,
                            batch_count, scalars,
                            (T*)work_workArr, workArr);

// LARFT INTERNALLY handles complex tau:
// 1. For complex matrices, uses conj(V) in computations
// 2. Computes T with formula: T_ii = tau_i, T_ij = -tau_i * T_jj * V_i' * V_j
// 3. Result: T is LOWER TRIANGULAR (backward direction)

// Step 3: Apply block reflector Q' from left
// LARFB applies (I - V*T*V')' = I - V*T'*V' with trans=conjugate_transpose
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_left,
    rocblas_operation_conjugate_transpose,  // Apply Q', not Q
    rocblas_backward_direction, rocblas_column_wise,
    m - k + j + jb, n - k + j, jb, A,
    shiftA + idx2D(0, n - k + j, lda),
    lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count,
    diag_tmptr, workArr);

// LARFB INTERNALLY handles the transpose:
// - Uses conj(T) when trans=conjugate_transpose
// - Applies: A_left := (I - V*conj(T)*V') * A_left
// - This is equivalent to A_left := Q' * A_left
```

**Key points about tau conjugation in blocked GEQLF:**

1. **GEQL2 layer:** Handles tau conjugation locally
   - Conjugates tau before each LARF call
   - Restores tau after LARF
   - Output tau array contains standard H-form values

2. **LARFT layer:** Receives standard tau values
   - Input: tau for H = I - tau*v*v'
   - Builds T for Q = Product of H_i
   - For complex: internally uses conj(v) in V'*V computations
   - For backward direction: T is lower triangular

3. **LARFB layer:** Uses T with transpose operation
   - trans=conjugate_transpose means apply Q', not Q
   - Internally computes with conj(T)
   - Result: A := Q' * A (what QL factorization needs)

4. **No explicit conjugation needed between layers:**
   - GEQL2 outputs standard tau
   - LARFT expects standard tau
   - LARFB handles Q vs Q' via trans parameter
   - Complex arithmetic is handled internally by each routine

5. **Advantage of this design:**
   - Separation of concerns
   - Each routine can be tested independently
   - tau array has consistent interpretation
   - No error-prone manual conjugation at caller site""",
    "rationale": "Understanding how tau conjugation is managed across GEQL2, LARFT, and LARFB layers is critical for implementing blocked QL factorization. The design cleanly separates responsibilities and avoids manual conjugation at higher levels.",
    "tags": ["ql-factorization", "geqlf", "larft", "complex-arithmetic", "tau-conjugation", "coding"]
})

# ========== L3 ENTRIES ==========

# L3-1: Analysis - Complete GEQLF algorithm trace
entries.append({
    "level": "L3",
    "task_type": "analysis",
    "instruction": "Trace the complete GEQLF algorithm for a 1024x512 double-precision complex matrix. Show all major steps, especially how backward direction differs from GEQRF.",
    "code_blocks": [
        """template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geqlf_template(rocblas_handle handle,
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
                                        T** workArr)"""
    ],
    "answer": """**GEQLF Execution Trace: 1024x512 complex double (batch=1)**

**Setup:**
- m=1024, n=512, k=min(1024,512)=512, lda=1024
- Data type: rocblas_double_complex (16 bytes)
- GEQxF_BLOCKSIZE=32, GEQxF_GEQx2_SWITCHSIZE=64
- ki = ((512-64-1)/32)*32 = 448, kk = min(512, 448+32) = 480
- Initial: j = 512-480+448 = 480

**Workspace Allocation:**
1. Abyx_norms_trfact: 32×32×16 = 16KB (T matrix)
2. work_workArr: max(GEQL2_work, LARFT_work) ≈ 512B
3. diag_tmptr: max(GEQL2_diag, LARFB_temp) = 32×(512-32)×16 ≈ 245KB
4. workArr: 32×32×16×2 = 32KB (batched TRMM)

**Main Loop (j=480 down to 32, step -32):**

**Iteration 1 (j=480, rightmost 32 columns):**
1. `jb = min(512-480, 32) = 32`
2. **GEQL2** on A[0:1024, 480:512] (1024×32 block):
   - Process 32 columns right-to-left (columns 511, 510, ..., 480)
   - For each column:
     * LARFG: generate tau for bottom m-j elements
     * LACGV: conjugate tau (complex only)
     * LARF: apply H' to columns to the left
     * LACGV: restore tau
   - 32 LARFG + 32 pairs of LACGV + 31 LARF calls
3. Update: `j = 480-32 = 448, mu = 1024-512+448+32 = 992, nu = 512-512+448+32 = 480`

**Iteration 2 (j=448, next 32 columns):**
1. `jb = min(512-448, 32) = 32`
2. **GEQL2** on A[0:992, 448:480] (992×32 block):
   - 32 LARFG + 64 LACGV + 31 LARF
3. **LARFT** (backward, column-wise, m=992, k=32):
   - GEMM: V'*V → T_init (32×992 × 992×32 → 32×32)
     * For backward + column-wise: V starts at row n-k = 512-32 = 480
     * But we're working on columns 448-479, so V is in rows 0:991
   - set_triangular kernel: diagonal = tau, zero upper triangle
   - larft_kernel_backward: compute lower triangular T
     * Processes reflectors backward: 31, 30, ..., 0
     * T[i,j] = -tau[i] * T[j,j] * (V[:,i]' * V[:,j]) for i>j
4. **LARFB** (left, conjugate_transpose, backward, 992×448, k=32):
   - copymatA1: copy A[0:31, 0:448] to W (32×448)
   - TRMM: W = V1'*W (lower tri V, conjugate transpose)
   - GEMM: W = W + V2'*A2 (32×448 + 32×960 × 960×448)
   - TRMM: W = conj(T)*W (lower tri T, 32×32 × 32×448)
   - GEMM: A2 -= V2*W (960×448 -= 960×32 × 32×448)
   - TRMM: W = V1*W (lower tri V)
   - addmatA1: A1 -= W (32×448)
5. Update: `j = 416, mu = 960, nu = 448`

**Iterations 3-14:** Similar pattern, processing 32-column blocks right-to-left

**Final Block (mu=64, nu=96):**
- **GEQL2** on A[0:64, 0:96] (64×96 leftmost block)
- No LARFT/LARFB (no columns to the left to update)

**Key Differences from GEQRF:**

| Aspect | GEQRF (QR) | GEQLF (QL) |
|--------|------------|------------|
| Processing | Left→Right | Right→Left |
| Factor location | R in top-left | L in bottom-right |
| Block position | col j, row j:m | col n-k+j, row 0:m-k+j+jb |
| Update region | A[j:m, j+jb:n] | A[0:m-k+j+jb, 0:n-k+j] |
| LARFT direction | forward | backward |
| LARFT T structure | upper triangular | lower triangular |
| LARFB trans | none (apply Q) | conj_trans (apply Q') |
| Tau conjugation | not needed | needed (complex only) |

**Total Operations:**
- GEQL2 calls: 15 (one per block + final)
- LARFG calls: ~512 total
- LARF calls: ~256 (within GEQL2)
- LACGV calls: ~1024 (tau conjugation, complex only)
- LARFT calls: 14
- LARFB calls: 14

**Output:**
- tau: 512 complex scalars (for H = I - tau*v*v')
- A: overwritten with L (bottom-right 512×512) and V (column-wise reflectors)
- L factor: 512×512 lower triangular in A[512:1024, 0:512]
- V vectors: implicit unit upper trapezoidal in A[0:1024, 0:512]

**Memory Traffic:** ~17MB per batch (similar to GEQRF)""",
    "rationale": "Complete algorithm trace shows how GEQLF's backward processing affects every layer: block indexing, LARFT direction, LARFB transpose, and tau conjugation. Understanding the full flow is essential for debugging and optimization.",
    "tags": ["ql-factorization", "geqlf", "algorithm-trace", "backward-direction", "complex-arithmetic", "analysis"]
})

# L3-2: Coding - Complete GEQLF wrapper with batched support
entries.append({
    "level": "L3",
    "task_type": "coding",
    "instruction": "Implement a complete GEQLF wrapper including strided-batched support, workspace calculation, and C API wrappers for all precisions.",
    "code_blocks": [
        """template <typename T, typename U>
rocblas_status rocsolver_geqlf_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("geqlf", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // ... workspace and execution ...
}"""
    ],
    "answer": """```cpp
// Complete GEQLF implementation with full batched/strided support

template <typename T, typename U>
rocblas_status rocsolver_geqlf_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("geqlf", "-m", m, "-n", n, "--lda", lda);

    // 1. Handle validation
    if(!handle)
        return rocblas_status_invalid_handle;

    // 2. Argument checking (shared with GEQL2)
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // 3. Setup for non-batched execution
    rocblas_int shiftA = 0;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // 4. Calculate workspace requirements
    size_t size_scalars;           // constants for rocBLAS
    size_t size_work_workArr;      // reusable work (GEQL2/LARFT)
    size_t size_Abyx_norms_trfact; // T matrix + GEQL2 buffers
    size_t size_diag_tmptr;        // GEQL2 diag + LARFB temp
    size_t size_workArr;           // array of pointers (batched)

    rocsolver_geqlf_getMemorySize<false, T>(m, n, batch_count,
                                            &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms_trfact,
                                            &size_diag_tmptr, &size_workArr);

    // 5. Handle memory size query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars,
                                                      size_work_workArr,
                                                      size_Abyx_norms_trfact,
                                                      size_diag_tmptr,
                                                      size_workArr);

    // 6. Allocate device memory
    void *scalars, *work_workArr, *Abyx_norms_trfact, *diag_tmptr, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr,
                              size_Abyx_norms_trfact, size_diag_tmptr,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_trfact = mem[2];
    diag_tmptr = mem[3];
    workArr = mem[4];

    // 7. Initialize constants on device
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // 8. Execute algorithm
    return rocsolver_geqlf_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
        batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);
}

// Strided-batched variant
template <typename T, typename U>
rocblas_status rocsolver_geqlf_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    T* ipiv,
                                                    const rocblas_stride strideP,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("geqlf_strided_batched", "-m", m, "-n", n,
                        "--lda", lda, "--strideA", strideA,
                        "--strideP", strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // Argument checking
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(handle, m, n, lda,
                                                       A, ipiv, batch_count);
    if(st != rocblas_status_continue)
        return st;

    rocblas_int shiftA = 0;

    // Workspace calculation
    size_t size_scalars, size_work_workArr, size_Abyx_norms_trfact;
    size_t size_diag_tmptr, size_workArr;
    rocsolver_geqlf_getMemorySize<false, T>(m, n, batch_count,
                                            &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms_trfact,
                                            &size_diag_tmptr, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars,
                                                      size_work_workArr,
                                                      size_Abyx_norms_trfact,
                                                      size_diag_tmptr,
                                                      size_workArr);

    // Allocate and execute
    void *scalars, *work_workArr, *Abyx_norms_trfact, *diag_tmptr, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr,
                              size_Abyx_norms_trfact, size_diag_tmptr,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_trfact = mem[2];
    diag_tmptr = mem[3];
    workArr = mem[4];

    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    return rocsolver_geqlf_template<false, true, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
        batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);
}

// C API wrappers for all precisions
extern "C" {

rocblas_status rocsolver_sgeqlf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geqlf_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgeqlf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_geqlf_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgeqlf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_geqlf_impl<rocblas_float_complex>(
        handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgeqlf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_geqlf_impl<rocblas_double_complex>(
        handle, m, n, A, lda, ipiv);
}

// Strided-batched C API
rocblas_status rocsolver_zgeqlf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* ipiv,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_geqlf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, batch_count);
}

// Similar wrappers for S/D/C precisions...

} // extern "C"
```

**Key implementation features:**
1. Unified argument checking with GEQL2
2. Workspace query support for optimal memory allocation
3. Template-based design for all precisions (S/D/C/Z)
4. Batched and strided-batched variants via template parameter
5. Proper error handling and memory management
6. LAPACK-compatible C API with S/D/C/Z prefixes
7. Consistent workspace calculation across variants""",
    "rationale": "Production GEQLF wrapper must handle all variants (regular, batched, strided-batched) and all precisions while maintaining code reuse through templates. Understanding the wrapper structure is essential for API design and maintenance.",
    "tags": ["ql-factorization", "geqlf", "production-code", "batched-operations", "workspace-management", "coding"]
})

# Write dataset
output_file = "/root/rocSOLVER/kernelgen/dataset/roclapack_geqlf.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Print statistics
coding_count = sum(1 for e in entries if e['task_type'] == 'coding')
analysis_count = sum(1 for e in entries if e['task_type'] == 'analysis')
l1_count = sum(1 for e in entries if e['level'] == 'L1')
l2_count = sum(1 for e in entries if e['level'] == 'L2')
l3_count = sum(1 for e in entries if e['level'] == 'L3')

print(f"Generated {len(entries)} entries")
print(f"Written to: {output_file}")
print(f"\nDistribution:")
print(f"  L1: {l1_count} entries")
print(f"  L2: {l2_count} entries")
print(f"  L3: {l3_count} entries")
print(f"  Coding: {coding_count} ({coding_count*100//len(entries)}%)")
print(f"  Analysis: {analysis_count} ({analysis_count*100//len(entries)}%)")
