#!/usr/bin/env python3
"""
Generator for GEQR2 dataset - QR factorization (unblocked)

GEQR2 computes a QR factorization of an m-by-n matrix A using Householder reflectors.
The factorization has the form A = Q * R where Q is orthogonal and R is upper trapezoidal.
Processes columns from left to right (j=0..dim-1).
"""

import json
import time
from pathlib import Path

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Helper to create a dataset entry"""
    return {
        "id": str(int(time.time() * 1000000)),
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

# L1-1: CODING - geqr2_kernel_small fused kernel implementation
entries.append(create_entry(
    level="L1",
    interface="geqr2",
    instruction="Implement the geqr2_kernel_small fused kernel that performs complete QR factorization in shared memory. The kernel loads the matrix into LDS, performs all LARFG+LARF iterations in a single launch, then writes back. Explain the shared memory layout.",
    context_text="GEQR2 has a specialized kernel for small square matrices (m==n, fits in shared memory). It eliminates multiple kernel launches by fusing all iterations into one kernel.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) geqr2_kernel_small(const I m,
                                                                     const I n,
                                                                     U AA,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     S* diagA,
                                                                     const rocblas_stride strideD,
                                                                     T* tauA,
                                                                     const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* diag = load_ptr_batch<S>(diagA, bid, 0, strideD);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // shared variables
    extern __shared__ double lmem[];
    T* a = reinterpret_cast<T*>(lmem);
    T* w = reinterpret_cast<T*>(a + m * n);
    T* tmptau = reinterpret_cast<T*>(w + n);
    T* sval = reinterpret_cast<T*>(tmptau + 1);

    T* x;

    I dim = std::min(m, n); // total number of pivots

    // load A to lds
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            a[i + j * m] = A[i + j * lda];
        }
    }

    __syncthreads();

    // main loop running forwards (for each column)
    for(rocblas_int j = 0; j < dim; ++j)
    {
        I mm = m - j;
        I nn = n - j;

        // ----- 1. generate Householder reflector to annihilate A(j+1:m-1,j) -----
        // load A(j:m-1,j) into x
        x = a + j + j * m;

        // larfg
        T norm2 = 0;
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            norm2 += x[i + 1] * conj(x[i + 1]);

        // reduce squared entries to find squared norm of x
        norm2 += shift_left(norm2, 1);
        norm2 += shift_left(norm2, 2);
        norm2 += shift_left(norm2, 4);
        norm2 += shift_left(norm2, 8);
        norm2 += shift_left(norm2, 16);
        if(warpSize > 32)
            norm2 += shift_left(norm2, 32);
        if(tid % warpSize == 0)
            sval[tid / warpSize] = norm2;
        __syncthreads();
        if(tid == 0)
        {
            for(I k = 1; k < MAX_THDS / warpSize; k++)
                norm2 += sval[k];

            // set tau, beta, and put scaling factor into sval[0]
            run_set_taubeta<T>(tmptau, &norm2, x, diag + j);

            tau[j] = tmptau[0];
            sval[0] = norm2;

            tmptau[0] = conj(tmptau[0]);
        }
        __syncthreads();

        // scale x by scaling factor
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            x[i + 1] *= sval[0];
        __syncthreads();

        // ----- 2. compute w = tau'*v'*A -----
        // gemv
        for(I i = tid; i < nn - 1; i += MAX_THDS)
        {
            T temp = 0;
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < mm; jj++)
                temp += Atmp[jj + (i + 1) * m] * conj(x[jj]);
            w[i] = tmptau[0] * temp;
        }
        __syncthreads();

        // ----- 3. apply the Householder reflector to A as a rank-1 update: A = A - v*w -----
        // ger
        for(I i = tid; i < mm; i += MAX_THDS)
        {
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < nn - 1; jj++)
            {
                Atmp[i + (jj + 1) * m] = Atmp[i + (jj + 1) * m] - x[i] * w[jj];
            }
        }
        __syncthreads();
    }

    // write lds back to A
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            A[i + j * lda] = a[i + j * m];
        }
    }
}"""
        }
    ],
    answer="""Here's the complete geqr2_kernel_small fused implementation:

```cpp
template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
geqr2_kernel_small(const I m, const I n,
                   U AA, const rocblas_stride shiftA, const I lda, const rocblas_stride strideA,
                   S* diagA, const rocblas_stride strideD,
                   T* tauA, const rocblas_stride strideP)
{
    const I bid = blockIdx.z;  // batch index
    const I tid = threadIdx.x; // thread index (0..MAX_THDS-1)

    // Load batch-specific pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* diag = load_ptr_batch<S>(diagA, bid, 0, strideD);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // Shared memory layout:
    // [a: m*n elements] [w: n elements] [tmptau: 1] [sval: MAX_THDS/warpSize]
    extern __shared__ double lmem[];
    T* a = reinterpret_cast<T*>(lmem);                    // Matrix copy in LDS
    T* w = reinterpret_cast<T*>(a + m * n);               // GEMV result
    T* tmptau = reinterpret_cast<T*>(w + n);              // Current tau
    T* sval = reinterpret_cast<T*>(tmptau + 1);           // Reduction scratch

    // Step 1: Cooperative load A into shared memory (column-major in LDS)
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
            a[i + j * m] = A[i + j * lda];  // LDS uses m as leading dim
    }
    __syncthreads();

    // Step 2: QR factorization loop (all iterations fused)
    const I dim = std::min(m, n);
    for(I j = 0; j < dim; ++j)
    {
        const I mm = m - j;  // remaining rows
        const I nn = n - j;  // remaining cols
        T* x = a + j + j * m;  // Householder vector location

        // 2a. LARFG: Generate Householder reflector
        // Compute norm2 = ||A(j+1:m-1, j)||^2
        T norm2 = 0;
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            norm2 += x[i + 1] * conj(x[i + 1]);

        // Warp-level reduction
        norm2 += shift_left(norm2, 1);
        norm2 += shift_left(norm2, 2);
        norm2 += shift_left(norm2, 4);
        norm2 += shift_left(norm2, 8);
        norm2 += shift_left(norm2, 16);
        if(warpSize > 32)
            norm2 += shift_left(norm2, 32);

        // Block-level reduction
        if(tid % warpSize == 0)
            sval[tid / warpSize] = norm2;
        __syncthreads();

        if(tid == 0)
        {
            for(I k = 1; k < MAX_THDS / warpSize; k++)
                norm2 += sval[k];

            // Compute tau, beta, scaling factor
            run_set_taubeta<T>(tmptau, &norm2, x, diag + j);
            tau[j] = tmptau[0];
            sval[0] = norm2;           // scaling factor
            tmptau[0] = conj(tmptau[0]);  // conjugate for LARF
        }
        __syncthreads();

        // Scale Householder vector: v(1:mm-1) *= scaling
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            x[i + 1] *= sval[0];
        __syncthreads();

        // 2b. LARF: Apply H = I - tau*v*v^H to A(j:m-1, j+1:n-1)
        // GEMV: w = A(j:m-1, j+1:n-1)^H * v
        for(I i = tid; i < nn - 1; i += MAX_THDS)
        {
            T temp = 0;
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < mm; jj++)
                temp += Atmp[jj + (i + 1) * m] * conj(x[jj]);
            w[i] = tmptau[0] * temp;  // w = conj(tau) * A^H * v
        }
        __syncthreads();

        // GER: A = A - v * w^T
        for(I i = tid; i < mm; i += MAX_THDS)
        {
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < nn - 1; jj++)
                Atmp[i + (jj + 1) * m] -= x[i] * w[jj];
        }
        __syncthreads();
    }

    // Step 3: Write LDS back to global memory
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
            A[i + j * lda] = a[i + j * m];
    }
}
```

**Shared Memory Layout (total: (256/warpSize + m*n + n + 1) * sizeof(T)):**
- `a[m*n]`: Matrix copy (column-major, ld=m)
- `w[n]`: GEMV result workspace
- `tmptau[1]`: Current tau value (conjugated)
- `sval[MAX_THDS/warpSize]`: Reduction scratch (4 or 8 elements)

**Key Optimizations:**
- Single kernel launch for entire factorization (eliminates ~2*dim kernel overhead)
- All data in shared memory (no global memory traffic during computation)
- Cooperative load/store with 2-way thread splitting
- Warp shuffle reductions for LARFG norm computation
- Used when: m==n and LDS size <= sharedMemPerBlock""",
    rationale="The fused kernel eliminates kernel launch overhead by performing all QR iterations in shared memory. This is profitable for small matrices where the entire matrix + workspace fits in LDS (typically ≤32x32). The kernel fuses LARFG (generate reflector) and LARF (apply reflector) for each column, synchronizing with __syncthreads() between stages.",
    tags=["coding", "kernel-fusion", "shared-memory", "qr-factorization", "small-size-optimization", "cooperative-loading"]
))

# L1-2: ANALYSIS - Left-to-right processing vs QL's right-to-left
entries.append(create_entry(
    level="L1",
    interface="geqr2",
    instruction="Explain why GEQR2 processes columns from left to right (j=0..dim-1) instead of right to left like GEQL2. How does this affect the final factorization structure and where the R factor appears?",
    context_text="GEQR2 computes QR factorization where A = Q*R. The loop iterates j from 0 to dim-1, processing column j in each iteration, working forward through the matrix.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """I dim = std::min(m, n); // total number of pivots
for(I j = 0; j < dim; ++j)
{
    I mm = m - j;
    I nn = n - j;

    // generate Householder reflector to work on column j
    rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the left
    if(j < n - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }
}"""
        }
    ],
    answer="""GEQR2's left-to-right processing creates a QR factorization where:

**Column Processing Order:**
- j=0: column 0 (leftmost), eliminates A(1:m-1, 0) using A(0,0) as pivot
- j=1: column 1, eliminates A(2:m-1, 1) using A(1,1) as pivot
- ...
- j=k: column k, eliminates A(k+1:m-1, k) using A(k,k) as pivot

**Why Left-to-Right:**
1. **R factor alignment**: The upper triangular R appears in the top-left corner of A, aligned with the leading rows/columns
2. **Natural QR structure**: Q = H_0 * H_1 * ... * H_{n-1} where each H_j zeros out column j below the diagonal
3. **Contrast with QL**: QL processes right-to-left, creating L in the bottom-right corner

**Final Structure (m >= n case):**
```
A = [ *   *   *   *   * ]      R = [ r₀₀ r₀₁ r₀₂ r₀₃ r₀₄ ]      Q*R = A
    [ *   *   *   *   * ]  =>      [ 0   r₁₁ r₁₂ r₁₃ r₁₄ ]
    [ *   *   *   *   * ]          [ 0   0   r₂₂ r₂₃ r₂₄ ]
    [ *   *   *   *   * ]          [ 0   0   0   r₃₃ r₃₄ ]
    [ *   *   *   *   * ]          [ 0   0   0   0   r₄₄ ]
                                   [ 0   0   0   0   0   ]  (if m > n)
```

**Householder Storage:**
- Column j: Householder vector stored below diagonal in A(j+1:m-1, j)
- Diagonal: R factor diagonals stored separately in diag[], replaced with v(0)=1 during processing
- Tau: Stored in forward order tau[0..dim-1]

**Comparison with GEQL2:**
| Aspect | GEQR2 (QR) | GEQL2 (QL) |
|--------|------------|------------|
| Order | j=0→dim-1 | j=0→dim-1, column n-j-1 |
| Factor | R (upper, top-left) | L (lower, bottom-right) |
| Q form | H₀*H₁*...*Hₙ₋₁ | Hₙ₋₁*...*H₁*H₀ |
| Updates | Columns j+1:n-1 | Columns 0:n-j-2 |""",
    rationale="Left-to-right processing ensures the R factor (upper triangular) naturally accumulates in the top-left of the matrix. Each Householder reflector H_j operates on columns j to n-1, preserving the already-factored columns 0 to j-1 on the left. This is the standard QR factorization used in linear solvers and least squares.",
    tags=["analysis", "qr-factorization", "column-ordering", "householder-reflectors", "matrix-structure"]
))

# L1-3: CODING - Diagonal backup strategy for QR
entries.append(create_entry(
    level="L1",
    interface="geqr2",
    instruction="Implement the diagonal backup and restoration strategy for GEQR2. Unlike GEQL2 which uses set_diag/restore_diag kernels per iteration, GEQR2 uses LARFG to save diagonals and a final batch restore. Why is this more efficient?",
    context_text="GEQR2 saves diagonal elements during LARFG (into diag array) and restores all of them at once after the factorization loop completes. This differs from GEQL2's per-iteration approach.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """for(I j = 0; j < dim; ++j)
{
    // generate Householder reflector to work on column j
    rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the left
    if(j < n - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }
}

// restore diagonal values of A
constexpr int DIAG_NTHREADS = 64;
I blocks = (dim - 1) / DIAG_NTHREADS + 1;
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>), dim3(batch_count, blocks, 1),
                        dim3(1, DIAG_NTHREADS, 1), 0, stream, (S*)diag, 0, dim, A, shiftA, lda,
                        strideA, dim);"""
        }
    ],
    answer="""Here's the optimized diagonal backup/restore for GEQR2:

```cpp
// Workspace allocation: diag array sized for all diagonal elements
// size_diag = sizeof(S) * min(m,n) * batch_count

const I dim = std::min(m, n);

// Main factorization loop
for(I j = 0; j < dim; ++j)
{
    const I mm = m - j;

    // LARFG saves diagonal A(j,j) into diag[j] while generating reflector
    // Alpha parameter points to A(j,j), beta points to diag[j]
    rocsolver_larfg_template<T>(
        handle,
        mm,                                           // vector length
        A, shiftA + idx2D(j, j, lda),                // alpha = A(j,j) (in/out)
        (S*)diag, j, dim,                            // beta = diag[j] (out), stride
        A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda),  // x = A(j+1:m-1, j)
        (I)1, strideA,                               // incx, stridex
        (ipiv + j), strideP,                         // tau output
        batch_count, work_workArr, Abyx_norms);

    // LARFG internally:
    // 1. Computes norm2 = ||x||^2
    // 2. Computes beta (R diagonal) from alpha and norm2
    // 3. Saves beta to diag[j]
    // 4. Overwrites alpha with 1.0 (implicit v[0])
    // 5. Computes and stores tau

    // LARF uses implicit v[0]=1.0 (no explicit diagonal set needed)
    if(j < n - 1)
    {
        rocsolver_larf_template(
            handle, rocblas_side_left,
            mm, n - j - 1,                           // Apply to (m-j) x (n-j-1) submatrix
            A, shiftA + idx2D(j, j, lda),           // Householder vector (v[0] is 1.0)
            (I)1, strideA,
            (ipiv + j), strideP,                     // tau
            A, shiftA + idx2D(j, j + 1, lda),       // Matrix to update
            lda, strideA,
            batch_count, scalars, Abyx_norms, work_workArr);
    }
}

// Batch restore all diagonals in one kernel launch
constexpr int DIAG_NTHREADS = 64;
const I blocks = (dim - 1) / DIAG_NTHREADS + 1;

ROCSOLVER_LAUNCH_KERNEL(
    (restore_diag<T, I>),
    dim3(batch_count, blocks, 1),     // Grid: batch x diagonal_blocks
    dim3(1, DIAG_NTHREADS, 1),        // Block: 64 threads for diagonals
    0, stream,
    (S*)diag,                         // Source: saved diagonals
    0, dim,                           // Offset, stride in diag array
    A, shiftA, lda, strideA,         // Destination: A matrix
    dim);                             // Number of diagonals to restore

// restore_diag kernel (simplified):
// for(i = threadIdx.y; i < dim; i += blockDim.y)
//     A[batch_id][i + i*lda] = diag[batch_id * dim + i];
```

**Why This is More Efficient Than GEQL2's Approach:**

1. **Reduced kernel launches**:
   - GEQR2: 1 restore kernel after loop (total: dim LARFG + dim LARF + 1 restore)
   - GEQL2: 2 kernels per iteration (total: dim LARFG + dim LARF + dim set_diag + dim restore_diag)
   - Savings: 2*dim - 1 kernel launches

2. **LARFG integration**:
   - LARFG already computes the diagonal value (beta) as part of reflector generation
   - Natural to save it at that point without extra work
   - GEQL2 needs explicit set_diag because QL processes columns right-to-left

3. **Coalesced restore**:
   - Batch restore processes dim diagonals in parallel across batch instances
   - Better occupancy: launches with (batch_count * blocks) rather than batch_count per iteration

4. **Memory access pattern**:
   - Sequential read from diag[0..dim-1]
   - Strided write to A[0,0], A[1,1], ..., A[dim-1, dim-1]
   - Single pass over diagonal elements""",
    rationale="GEQR2's diagonal strategy is more efficient because LARFG naturally computes and can save the diagonal during reflector generation (it's the 'beta' output). Restoring all diagonals in a single batched kernel after the loop reduces kernel launch overhead from O(dim) to O(1). GEQL2 cannot use this optimization because it processes columns right-to-left and needs the diagonal preserved during each iteration.",
    tags=["coding", "diagonal-preservation", "kernel-optimization", "larfg-integration", "batch-restore", "qr-factorization"]
))

# L2-1: ANALYSIS - Small kernel vs standard path decision logic
entries.append(create_entry(
    level="L2",
    interface="geqr2",
    instruction="Analyze the decision logic for choosing between geqr2_kernel_small (fused) and the standard LARFG+LARF path. For what matrix sizes does each path execute, and what are the performance trade-offs?",
    context_text="GEQR2 has two execution paths: a fused kernel for small square matrices and the standard iterative path using LARFG+LARF. The choice depends on matrix size and shared memory availability.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """I dim = std::min(m, n); // total number of pivots
for(I j = 0; j < dim; ++j)
{
    I mm = m - j;
    I nn = n - j;

    const size_t lmemsize = ((256 / props.warpSize) + mm + nn + 1 + mm * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn == mm)
    {
        ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>), dim3(1, 1, batch_count), dim3(256),
                                lmemsize, stream, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, (S*)diag + j, dim, ipiv + j, strideP);
        break;
    }

    // generate Householder reflector to work on column j
    rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the left
    if(j < n - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }
}"""
        }
    ],
    answer="""Complete analysis of GEQR2's execution path selection:

**Decision Criteria (checked at each iteration j):**

1. **Shared memory constraint**:
   ```
   lmemsize = (256/warpSize + mm + nn + 1 + mm*nn) * sizeof(T)
            = (4 or 8) + mm + nn + 1 + mm*nn) * sizeof(T)

   For warpSize=64 (GFX9): (4 + mm + nn + 1 + mm*nn) * sizeof(T)
   For warpSize=32 (RDNA): (8 + mm + nn + 1 + mm*nn) * sizeof(T)
   ```
   Must satisfy: `lmemsize <= props.sharedMemPerBlock`

2. **Square submatrix requirement**: `nn == mm` (remaining matrix must be square)

3. **Iteration check**: Evaluated at the start of iteration j, so processes A(j:m-1, j:n-1)

**Path Selection Examples (double precision, 64KB LDS, warpSize=64):**

| m | n | j | mm=m-j | nn=n-j | lmemsize | Square? | Path |
|---|---|---|--------|--------|----------|---------|------|
| 1024 | 512 | 0 | 1024 | 512 | 4MB | No | Standard |
| 512 | 512 | 0 | 512 | 512 | 2MB | Yes | Standard (too large) |
| 64 | 64 | 0 | 64 | 64 | 33KB | Yes | **Fused** |
| 100 | 64 | 0 | 100 | 64 | 51KB | No | Standard |
| 100 | 100 | 0 | 100 | 100 | 80KB | Yes | Standard (too large) |
| 100 | 100 | 36 | 64 | 64 | 33KB | Yes | **Fused** (from j=36 onward) |

**Maximum Fused Size (64KB LDS, sizeof(T)=8, warpSize=64):**
```
(4 + m + m + 1 + m²) * 8 <= 65536
m² + 2m + 5 <= 8192
m ≤ 89 (approximately)
```

**Performance Trade-offs:**

**Fused Path (geqr2_kernel_small):**
- ✓ Single kernel launch for remaining factorization
- ✓ All operations in shared memory (no global memory traffic)
- ✓ Eliminates 2*(dim-j) kernel launches
- ✗ Limited to small square matrices (typically ≤89×89 for FP64)
- ✗ Requires all threads idle while single block works (low GPU utilization)
- Best for: Small matrices, high batch count (many independent blocks)

**Standard Path (LARFG + LARF):**
- ✓ Works for arbitrary matrix sizes
- ✓ LARF can use optimized kernels or GEMV+GER
- ✓ Better GPU utilization for large matrices
- ✗ O(dim) kernel launches per matrix
- ✗ Global memory traffic for each operation
- Best for: Large matrices, low batch count

**Hybrid Execution:**
For non-square matrices, GEQR2 starts with standard path and switches to fused once the submatrix becomes square and small enough. Example: 1024×512 matrix processes 512 iterations with standard path (never square), but 100×100 processes iterations 0-35 standard, then switches to fused at j=36 when 64×64 submatrix fits in LDS.""",
    rationale="The decision logic optimizes for small square submatrices where kernel launch overhead dominates. Once mm==nn and the submatrix fits in shared memory, switching to the fused kernel eliminates remaining kernel launches and global memory traffic. The square requirement ensures efficient LDS usage (no wasted space for rectangular padding).",
    tags=["analysis", "kernel-selection", "shared-memory", "performance-optimization", "small-size-kernels", "execution-paths"]
))

# L2-2: CODING - Workspace sizing with LARFG+LARF cooperation
entries.append(create_entry(
    level="L2",
    interface="geqr2",
    instruction="Implement the complete workspace calculation for GEQR2 including the diagonal array size difference from GEQL2. Why does GEQR2 allocate sizeof(S)*dim instead of sizeof(T)*dim for diagonals?",
    context_text="GEQR2's workspace includes buffers for LARFG, LARF, and diagonal storage. The diagonal array stores real values even for complex matrices.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, typename T, typename I>
void rocsolver_geqr2_getMemorySize(const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    using S = decltype(std::real(T{}));

    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_left, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(m, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(S) * std::min(m, n) * batch_count;
}"""
        }
    ],
    answer="""```cpp
template <bool BATCHED, typename T, typename I>
void rocsolver_geqr2_getMemorySize(const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // Extract real type from potentially complex T
    using S = decltype(std::real(T{}));  // S=float if T=rocblas_float_complex
                                         // S=double if T=rocblas_double_complex
                                         // S=T if T is real

    // Quick return: no workspace for empty matrices
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // Query LARF workspace (for applying Householder reflectors)
    size_t larf_scalars, larf_norms, larf_work;
    rocsolver_larf_getMemorySize<BATCHED, T>(
        rocblas_side_left,    // QR applies from left: H*A
        m,                     // max reflector height
        n,                     // max matrix width
        batch_count,
        &larf_scalars,        // Constants: alpha=-1, beta=0, beta=1
        &larf_norms,          // Temp for GEMV result (n elements)
        &larf_work);          // Work array or pointer array

    // Query LARFG workspace (for generating Householder reflectors)
    size_t larfg_work, larfg_norms;
    rocsolver_larfg_getMemorySize<T>(
        m,                     // max vector length (first iteration)
        batch_count,
        &larfg_work,          // DOT reduction workspace
        &larfg_norms);        // Temp for norm storage (1 element per batch)

    // Workspace is reused between LARFG and LARF (sequential calls)
    *size_scalars = larf_scalars;  // Only LARF needs scalars
    *size_work_workArr = std::max(larf_work, larfg_work);
    *size_Abyx_norms = std::max(larf_norms, larfg_norms);

    // Diagonal storage: REAL type, not complex!
    // For complex matrices: sizeof(S) = sizeof(T)/2
    *size_diag = sizeof(S) * std::min(m, n) * batch_count;
}
```

**Why sizeof(S) Instead of sizeof(T) for Diagonals:**

1. **Mathematical reason**:
   - R factor diagonals are REAL numbers (magnitudes)
   - Even for complex QR: R_ii = ||column_i|| (always real, non-negative)
   - LARFG computes: beta = sign(alpha) * ||x||, where beta is real

2. **LARFG signature**:
   ```cpp
   template <typename T, typename S, typename U>
   rocblas_status rocsolver_larfg_template(
       ...,
       S* beta,  // Real output: R diagonal element
       ...)
   ```
   The beta parameter is type S (real), not T (potentially complex)

3. **Storage savings**:
   - Complex double (T=rocblas_double_complex): sizeof(T)=16, sizeof(S)=8
   - Complex float (T=rocblas_float_complex): sizeof(T)=8, sizeof(S)=4
   - Saves 50% memory for complex matrices

4. **Comparison with GEQL2**:
   ```cpp
   // GEQL2 uses set_diag/restore_diag which handle full complex values:
   *size_diag = sizeof(T) * batch_count;  // Per-iteration diagonal

   // GEQR2 uses LARFG beta output (real only):
   *size_diag = sizeof(S) * std::min(m,n) * batch_count;  // All diagonals
   ```

Complete Workspace Breakdown (1024x512 FP64 complex, batch=1):
- size_scalars: 24 bytes (3 * sizeof(double))
- size_work_workArr: max(512*8, DOT_workspace) = 4 KB
- size_Abyx_norms: max(512*16, 16) = 8 KB
- size_diag: 512 * 8 = 4 KB (real, not complex!)
- Total: ~16 KB

If incorrectly using sizeof(T): size_diag = 512*16 = 8 KB (wastes 4 KB)""",
    rationale="GEQR2 uses sizeof(S) for diagonals because the R factor diagonals are real-valued magnitudes, even for complex matrices. LARFG's beta output (the diagonal value) is type S (real component type). This is more memory-efficient than GEQL2's approach which stores full complex values during set_diag/restore_diag operations.",
    tags=["coding", "workspace-management", "complex-arithmetic", "real-diagonal", "memory-optimization", "larfg-integration"]
))

# L3-1: ANALYSIS - Complete algorithm trace
entries.append(create_entry(
    level="L3",
    interface="geqr2",
    instruction="Trace the complete GEQR2 execution for a 512×512 double-precision matrix, comparing the fused kernel path vs standard path. Include kernel launches, memory traffic, FLOPs, and execution time estimates.",
    context_text="GEQR2 can use either geqr2_kernel_small (fused) for small square matrices or the standard LARFG+LARF iterative path. For 512×512 FP64, analyze both scenarios.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """I dim = std::min(m, n);
for(I j = 0; j < dim; ++j)
{
    I mm = m - j;
    I nn = n - j;

    const size_t lmemsize = ((256 / props.warpSize) + mm + nn + 1 + mm * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn == mm)
    {
        ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>), dim3(1, 1, batch_count), dim3(256),
                                lmemsize, stream, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                                strideA, (S*)diag + j, dim, ipiv + j, strideP);
        break;
    }

    rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    if(j < n - 1)
    {
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);

        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);
    }
}

constexpr int DIAG_NTHREADS = 64;
I blocks = (dim - 1) / DIAG_NTHREADS + 1;
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>), dim3(batch_count, blocks, 1),
                        dim3(1, DIAG_NTHREADS, 1), 0, stream, (S*)diag, 0, dim, A, shiftA, lda,
                        strideA, dim);"""
        }
    ],
    answer="""Complete GEQR2 trace for 512×512 FP64 matrix:

**Path Determination:**
```
m=512, n=512, dim=512, sizeof(T)=8, warpSize=64, LDS=64KB

For j=0: mm=512, nn=512
lmemsize = (4 + 512 + 512 + 1 + 512*512) * 8
         = (4 + 1025 + 262144) * 8
         = 263173 * 8 = 2,105,384 bytes ≈ 2MB

2MB > 64KB → Does NOT fit in shared memory
→ Use STANDARD PATH
```

**Standard Path Execution:**

**Per Iteration j (0 to 511):**

1. **LARFG(512-j, A(j,j), A(j+1:511,j))**
   - DOT: compute ||A(j+1:511,j)||²
     - FLOPs: 2*(512-j-1) ≈ 1022 - 2j
     - Memory: read (512-j) elements
   - Compute beta, tau, scaling
     - FLOPs: ~10
   - SCAL: scale A(j+1:511,j)
     - FLOPs: 512-j-1
     - Memory: read/write (512-j-1) elements
   - Store: beta→diag[j], tau→tau[j]

2. **LARF(512-j, 511-j, v, tau, A(j:511,j+1:511))** [if j < 511]
   - GEMV: w = A(j:511,j+1:511)^T * v
     - Dimensions: (511-j) × (512-j)
     - FLOPs: 2*(512-j)*(511-j)
     - Memory: read (512-j)*(511-j) + (512-j), write (511-j)
   - GER: A = A - tau*v*w^T
     - FLOPs: 2*(512-j)*(511-j)
     - Memory: read (512-j)*(511-j) + (512-j) + (511-j), write (512-j)*(511-j)

3. **Kernel launches per iteration**:
   - LARFG: 1 call (may use larfg_run_small or DOT+set_taubeta+SCAL)
   - LARF: 1 call (larf_left_kernel or GEMV+GER)
   - Total: ~2-4 kernels per iteration

**Total FLOPs:**
```
LARFG: Σ(j=0..511)[2*(512-j) + 10] ≈ 512*1024 = 524,288

LARF:  Σ(j=0..510)[4*(512-j)*(511-j)]
     = 4*Σ(k=1..512)[k*(k-1)]
     = 4*[512*511*512/3]
     ≈ 178,782,208

Total: ~179 MFLOPs
```

**Total Memory Traffic:**
```
LARFG reads: Σ(j=0..511)[512-j] = 512*513/2 = 131,328 elements = 1 MB
LARFG writes: 512 + 512 = 1024 elements = 8 KB

LARF reads per iter j: ~2*(512-j)*(511-j)
  Total: Σ(j=0..510)[2*(512-j)*(511-j)] ≈ 89 million elements = 712 MB
LARF writes: same as reads ≈ 712 MB

Total: ~1.4 GB
```

**Kernel Launch Count:**
- LARFG: 512
- LARF: 511
- restore_diag: 1
- **Total: 1024 kernel launches**

**Hypothetical Fused Path (if it fit in LDS):**

If 512×512 could use geqr2_kernel_small:
- **Kernel launches: 1** (vs 1024)
- **Global memory traffic:
  - Load A: 512×512×8 = 2 MB
  - Write A: 512×512×8 = 2 MB
  - Write tau: 512×8 = 4 KB
  - **Total: 4 MB** (vs 1.4 GB = **350x reduction**)
- **FLOPs: same ~179 MFLOPs**
- **Arithmetic intensity: 179M / 4MB = 44.75 FLOP/byte** (vs 0.13 FLOP/byte)

**Performance Estimates (MI250X, real execution):**

Standard Path:
- Kernel launch overhead: 1024 × 5μs = 5.1 ms
- Compute (179 MFLOPs @ 100 GFLOPS): 1.8 ms
- Memory (1.4 GB @ 1.6 TB/s): 0.9 ms
- **Total: ~7.8 ms** (dominated by launch overhead)

Hypothetical Fused Path:
- Kernel launch overhead: 1 × 5μs = 0.005 ms
- Compute (179 MFLOPs @ 100 GFLOPS): 1.8 ms (LDS bottleneck)
- Memory (4 MB @ 1.6 TB/s): 0.0025 ms
- **Total: ~1.8 ms** (4.3x faster)

**Why Fused is Better (When It Fits):**
- Eliminates 99.9% of kernel launches
- Reduces memory traffic by 350x
- Improves arithmetic intensity by 344x
- But limited to small matrices (≤89×89 for FP64)""",
    rationale="For 512×512 FP64, the standard path is required because shared memory is insufficient. This results in 1024 kernel launches and 1.4 GB memory traffic with 0.13 FLOP/byte. The fused kernel, when applicable, eliminates launch overhead and achieves 44.75 FLOP/byte by keeping all data in LDS. This demonstrates the critical importance of kernel fusion for small matrices.",
    tags=["analysis", "performance-analysis", "algorithm-trace", "kernel-fusion", "memory-traffic", "flop-counting"]
))

# L3-2: CODING - Production wrapper
entries.append(create_entry(
    level="L3",
    interface="geqr2",
    instruction="Implement a complete production GEQR2 wrapper including 64-bit integer support, workspace query mode, error handling, and proper argument validation order. Handle both standard and batched execution.",
    context_text="GEQR2 needs to support both 32-bit and 64-bit integer APIs, workspace size queries, and proper error handling with the correct validation order for unit tests.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_geqr2.hpp",
            "language": "cpp",
            "content": """template <typename T, typename I, typename U>
rocblas_status rocsolver_geqr2_geqrf_argCheck(rocblas_handle handle,
                                              const I m,
                                              const I n,
                                              const I lda,
                                              T A,
                                              U ipiv,
                                              const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}"""
        }
    ],
    answer="""Complete production GEQR2 wrapper with 64-bit support:

```cpp
// Main implementation template (supports both 32-bit and 64-bit)
template <typename T, typename I, typename U>
rocblas_status rocsolver_geqr2_impl(rocblas_handle handle,
                                    const I m,
                                    const I n,
                                    U A,
                                    const I lda,
                                    T* ipiv)
{
    // Logging for debugging
    const char* api_name = (sizeof(I) == 8) ? "geqr2_64" : "geqr2";
    ROCSOLVER_ENTER_TOP(api_name, "-m", m, "-n", n, "--lda", lda);

    // Step 1: Validate handle
    if(!handle)
        return rocblas_status_invalid_handle;

    // Step 2: Argument validation (order critical for unit tests)
    rocblas_status st = rocsolver_geqr2_geqrf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // Step 3: Setup execution parameters
    const rocblas_stride shiftA = 0;     // No array offset
    const rocblas_stride strideA = 0;    // Non-strided
    const rocblas_stride stridep = 0;    // Non-strided tau
    const I batch_count = 1;             // Single matrix

    // Step 4: Query workspace requirements
    size_t size_scalars, size_work_workArr, size_Abyx_norms, size_diag;
    rocsolver_geqr2_getMemorySize<false, T>(m, n, batch_count,
                                           &size_scalars,
                                           &size_work_workArr,
                                           &size_Abyx_norms,
                                           &size_diag);

    // Step 5: Handle workspace query mode
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle,
                                                     size_scalars,
                                                     size_work_workArr,
                                                     size_Abyx_norms,
                                                     size_diag);

    // Step 6: Allocate device workspace
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle,
                             size_scalars,
                             size_work_workArr,
                             size_Abyx_norms,
                             size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    // Step 7: Assign workspace pointers
    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];

    // Step 8: Initialize scalar constants
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // Step 9: Execute algorithm
    return rocsolver_geqr2_template<T>(handle,
                                      m, n,
                                      A, shiftA, lda, strideA,
                                      ipiv, stridep,
                                      batch_count,
                                      (T*)scalars,
                                      work_workArr,
                                      (T*)Abyx_norms,
                                      diag);
}

// Argument checker (shared with GEQRF)
template <typename T, typename I, typename U>
rocblas_status rocsolver_geqr2_geqrf_argCheck(rocblas_handle handle,
                                              const I m,
                                              const I n,
                                              const I lda,
                                              T A,
                                              U ipiv,
                                              const I batch_count = 1)
{
    // Order is critical for unit test compatibility

    // 1. Invalid/non-supported values (none for GEQR2)
    // N/A

    // 2. Invalid sizes
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // Skip pointer validation during workspace query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. Invalid pointers (only when m,n > 0)
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// C API wrappers - 32-bit integer versions
extern "C" {

rocblas_status rocsolver_sgeqr2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geqr2_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgeqr2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_geqr2_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgeqr2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_geqr2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgeqr2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_geqr2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
}

// 64-bit integer versions
rocblas_status rocsolver_sgeqr2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* ipiv)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geqr2_impl<float>(handle, m, n, A, lda, ipiv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgeqr2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* ipiv)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geqr2_impl<double>(handle, m, n, A, lda, ipiv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgeqr2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   rocblas_float_complex* ipiv)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geqr2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgeqr2_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   rocblas_double_complex* ipiv)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geqr2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern "C"
```

**Key Features:**

1. **Template-based implementation**: Single impl function supports both int32 and int64
2. **Conditional 64-bit support**: Uses HAVE_ROCBLAS_64 macro
3. **Workspace query mode**: Returns early with size information
4. **Proper error handling order**:
   - invalid_handle → invalid_size → invalid_pointer → memory_error
5. **Type-generic**: Works for float, double, complex float, complex double
6. **Shared argument checker**: Used by both GEQR2 and GEQRF
7. **Logging support**: Different names for 32-bit vs 64-bit APIs""",
    rationale="Production wrappers must handle multiple data types, integer sizes, and execution modes while maintaining backward compatibility. The template-based design eliminates code duplication, while conditional compilation handles optional 64-bit support. Argument validation order is fixed for unit test compatibility, and workspace query mode allows users to pre-allocate memory efficiently.",
    tags=["coding", "production-wrapper", "64-bit-support", "workspace-query", "error-handling", "api-design", "template-metaprogramming"]
))

# Write to JSONL
output_file = Path("kernelgen/dataset/roclapack_geqr2.jsonl")
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Calculate statistics
total = len(entries)
l1 = sum(1 for e in entries if e['level'] == 'L1')
l2 = sum(1 for e in entries if e['level'] == 'L2')
l3 = sum(1 for e in entries if e['level'] == 'L3')
coding = sum(1 for e in entries if 'coding' in e['tags'])
analysis = sum(1 for e in entries if 'analysis' in e['tags'])

print(f"Generated {total} entries")
print(f"Written to: {output_file.absolute()}")
print(f"\nDistribution:")
print(f"  L1: {l1} entries")
print(f"  L2: {l2} entries")
print(f"  L3: {l3} entries")
print(f"  Coding: {coding} ({coding*100//total}%)")
print(f"  Analysis: {analysis} ({analysis*100//total}%)")
