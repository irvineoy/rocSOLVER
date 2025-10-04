#!/usr/bin/env python3
import json

entries = []

# ========== L1 ENTRIES ==========

# L1-1: Coding - Block partitioning strategy for backward processing
entries.append({
    "level": "L1",
    "task_type": "coding",
    "instruction": "Implement the block partitioning logic for GERQF's main loop that processes blocks from right-to-left. Show how j, jb, mu, nu are calculated for backward RQ processing.",
    "code_blocks": [
        """rocblas_int k = std::min(m, n); // total number of pivots
rocblas_int nb = GExQF_BLOCKSIZE;
rocblas_int ki = ((k - GExQF_GExQ2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;
rocblas_int mu = m, nu = n;

while(j >= k - kk)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(k - j, nb); // number of columns in the block
    rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A, shiftA + idx2D(m - k + j, 0, lda),
                                lda, strideA, (ipiv + j), strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);

    // ... apply block reflector ...

    j -= nb;
    mu = m - k + j + jb;
    nu = n - k + j + jb;
}

// factor last block
if(mu > 0 && nu > 0)
    rocsolver_gerq2_template<T>(handle, mu, nu, A, shiftA, lda, strideA, ipiv, strideP,
                                batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                diag_tmptr);"""
    ],
    "answer": """```cpp
// GERQF block partitioning for backward (right-to-left) RQ processing
rocblas_int k = std::min(m, n); // total pivots
rocblas_int nb = GExQF_BLOCKSIZE; // block size (e.g., 32)
rocblas_int ki = ((k - GExQF_GExQ2_SWITCHSIZE - 1) / nb) * nb;
rocblas_int kk = std::min(k, ki + nb);
rocblas_int jb, j = k - kk + ki;  // start from right side
rocblas_int mu = m, nu = n;

// Main loop: process blocks from right to left
while(j >= k - kk)
{
    jb = std::min(k - j, nb); // block width

    // Factor bottom-right block at position (m-k+j, 0)
    rocsolver_gerq2_template<T>(handle, jb, n - k + j + jb, A,
                                shiftA + idx2D(m - k + j, 0, lda),
                                lda, strideA, (ipiv + j), strideP,
                                batch_count, scalars, work_workArr,
                                Abyx_norms_trfact, diag_tmptr);

    if(m - k + j > 0)  // if there's matrix above the block
    {
        // Build and apply block reflector (LARFT + LARFB)
        // ... (block reflector application) ...
    }

    j -= nb;  // move left by one block
    mu = m - k + j + jb;  // remaining rows
    nu = n - k + j + jb;  // remaining cols
}

// Factor final block (top-left region)
if(mu > 0 && nu > 0)
    rocsolver_gerq2_template<T>(handle, mu, nu, A, shiftA, lda, strideA,
                                ipiv, strideP, batch_count, scalars,
                                work_workArr, Abyx_norms_trfact, diag_tmptr);
```

**Key differences from GEQRF (left-to-right QR):**
- GERQF starts at `j = k - kk + ki` (right side), GEQRF starts at 0
- GERQF decrements `j -= nb`, GEQRF increments `j += nb`
- Block position: `(m-k+j, 0)` for GERQF bottom rows, `(j, 0)` for GEQRF top rows
- Final block processes `(mu, nu)` instead of remaining rows/cols""",
    "rationale": "Block partitioning for GERQF must process from right-to-left (backwards) to build R factor in bottom-right corner, unlike GEQRF which processes left-to-right. Understanding j/jb/mu/nu calculation is critical for blocked RQ factorization.",
    "tags": ["rq-factorization", "blocked-algorithm", "gerqf", "block-partitioning", "backward-processing"]
})

# L1-2: Analysis - LARFT backward direction for RQ
entries.append({
    "level": "L1",
    "task_type": "analysis",
    "instruction": "Explain why GERQF uses rocblas_backward_direction for LARFT while GEQRF uses rocblas_forward_direction. What does this mean for the block reflector structure?",
    "code_blocks": [
        """// GERQF: builds block reflector with backward direction
rocsolver_larft_template<T>(handle, rocblas_backward_direction, rocblas_row_wise,
                            n - k + j + jb, jb, A, shiftA + idx2D(m - k + j, 0, lda),
                            lda, strideA, (ipiv + j), strideP, Abyx_norms_trfact, ldw,
                            strideW, batch_count, scalars, (T*)work_workArr, workArr);

// GEQRF: builds block reflector with forward direction
rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise,
                            m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                            (ipiv + j), strideP, Abyx_norms_trfact, ldw, strideW,
                            batch_count, scalars, (T*)work_workArr, workArr);"""
    ],
    "answer": """The direction parameter controls the triangular structure of the block reflector T:

**GERQF (rocblas_backward_direction):**
- Processes Householder reflectors from **right-to-left** (rows m-1, m-2, ..., m-k+j)
- Builds **lower triangular** T matrix
- T structure for k=3 reflectors:
```
  [t11  0   0 ]
  [t21 t22  0 ]
  [t31 t32 t33]
```
- LARFT accumulates: T = [T_prev; -tau*V'*V*T_prev, -tau]
- Matches RQ factorization order (bottom-to-top rows)

**GEQRF (rocblas_forward_direction):**
- Processes Householder reflectors from **left-to-right** (columns 0, 1, ..., j+jb-1)
- Builds **upper triangular** T matrix
- T structure for k=3 reflectors:
```
  [t11 t12 t13]
  [ 0  t22 t23]
  [ 0   0  t33]
```
- LARFT accumulates: T = [-tau, -tau*T_prev*V*V'; T_prev]
- Matches QR factorization order (left-to-right columns)

**Why it matters:**
- Block reflector application (LARFB) uses `H = I - V*T*V'` (right) or `H = I - V*T'*V'` (left)
- Triangular structure must match the order reflectors were generated
- TRMM operations in LARFB rely on T being either upper or lower triangular
- Backward direction ensures T is lower triangular for row-wise RQ processing""",
    "rationale": "The direction parameter in LARFT is fundamental to blocked algorithms. GERQF's backward direction creates a lower triangular T that matches the right-to-left row processing order, while GEQRF's forward direction creates upper triangular T for left-to-right column processing.",
    "tags": ["rq-factorization", "larft", "block-reflector", "backward-direction", "triangular-structure"]
})

# L1-3: Coding - LARFB application for RQ
entries.append({
    "level": "L1",
    "task_type": "coding",
    "instruction": "Implement the LARFB call for GERQF that applies the block reflector from the right side. Show the exact parameters including side, direction, and matrix dimensions.",
    "code_blocks": [
        """// apply the block reflector
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle, rocblas_side_right, rocblas_operation_none, rocblas_backward_direction,
    rocblas_row_wise, m - k + j, n - k + j + jb, jb, A,
    shiftA + idx2D(m - k + j, 0, lda), lda, strideA, Abyx_norms_trfact, 0, ldw, strideW,
    A, shiftA, lda, strideA, batch_count, diag_tmptr, workArr);"""
    ],
    "answer": """```cpp
// LARFB application for GERQF: H = I - V*T*V' from RIGHT side
rocsolver_larfb_template<BATCHED, STRIDED, T>(
    handle,
    rocblas_side_right,            // apply H from right: A_out = A_in * H
    rocblas_operation_none,         // no transpose on T
    rocblas_backward_direction,     // T is lower triangular
    rocblas_row_wise,               // V stored row-wise for RQ
    m - k + j,                      // rows of A to update (above current block)
    n - k + j + jb,                 // cols of A to update (full width)
    jb,                             // number of reflectors in block
    A,                              // V matrix (Householder vectors in rows)
    shiftA + idx2D(m - k + j, 0, lda), // V starts at row m-k+j
    lda,
    strideA,
    Abyx_norms_trfact,              // T matrix (block reflector)
    0,                              // shift for T
    ldw,                            // leading dim of T (= GExQF_BLOCKSIZE)
    strideW,                        // stride for T
    A,                              // matrix to update
    shiftA,                         // update from top-left corner
    lda,
    strideA,
    batch_count,
    diag_tmptr,                     // workspace
    workArr);                       // array workspace
```

**Dimensions for 1024x512 example with nb=32, current j:**
- Update region: `(m - k + j) x (n - k + j + jb)` rows x cols above current block
- V matrix: `jb x (n - k + j + jb)` - jb row-wise Householder vectors
- T matrix: `jb x jb` - lower triangular block reflector
- Operation: `A[0:m-k+j, 0:n-k+j+jb] = A * (I - V'*T*V)`

**Key differences from GEQRF:**
- GEQRF uses `rocblas_side_left` (H*A from left)
- GEQRF uses `rocblas_forward_direction` (T upper triangular)
- GEQRF uses `rocblas_column_wise` (V stored column-wise)""",
    "rationale": "LARFB for GERQF must apply block reflectors from the RIGHT side with BACKWARD direction because RQ factorization builds reflectors in rows from bottom-to-top. Understanding the side/direction/storev parameters is critical for correct blocked RQ implementation.",
    "tags": ["rq-factorization", "larfb", "block-reflector", "right-side-application", "coding"]
})

# ========== L2 ENTRIES ==========

# L2-1: Analysis - Workspace comparison GERQF vs GEQRF
entries.append({
    "level": "L2",
    "task_type": "analysis",
    "instruction": "Compare the workspace requirements of GERQF and GEQRF. How do the LARFT and LARFB workspace sizes differ, and why?",
    "code_blocks": [
        """// GERQF workspace calculation
rocsolver_larft_getMemorySize<BATCHED, T>(n, jb, batch_count, &unused, &w2, size_workArr);
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_right, m - jb, n, jb, batch_count,
                                          &s2, &unused);

// GEQRF workspace calculation
rocsolver_larft_getMemorySize<BATCHED, T>(m, jb, batch_count, &unused, &w2, size_workArr);
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, m, n - jb, jb, batch_count,
                                          &s2, &unused);"""
    ],
    "answer": """**GERQF (RQ factorization) workspace:**

1. **LARFT workspace** (n, jb):
   - V matrix is row-wise: `jb x n` Householder vectors
   - GEMM computes `V*V'`: `jb x n` times `n x jb` → `jb x jb`
   - Work buffer: `k * batch_count * sizeof(T)`
   - T matrix: `jb x jb`

2. **LARFB workspace** (RIGHT, m-jb, n, jb):
   - Applies `H = I - V*T*V'` from right: `A_out = A_in * H`
   - Temp matrix W: `(m-jb) x jb` for computing `A*V`
   - Size: `(m-jb) * jb * batch_count * sizeof(T)`

**GEQRF (QR factorization) workspace:**

1. **LARFT workspace** (m, jb):
   - V matrix is column-wise: `m x jb` Householder vectors
   - GEMM computes `V'*V`: `jb x m` times `m x jb` → `jb x jb`
   - Work buffer: `k * batch_count * sizeof(T)`
   - T matrix: `jb x jb`

2. **LARFB workspace** (LEFT, m, n-jb, jb):
   - Applies `H = I - V*T'*V'` from left: `A_out = H * A_in`
   - Temp matrix W: `jb x (n-jb)` for computing `V'*A`
   - Size: `jb * (n-jb) * batch_count * sizeof(T)`

**Key differences (1024x512 matrix, jb=32):**

| Component | GERQF | GEQRF |
|-----------|-------|-------|
| LARFT dimension | n=512 | m=1024 |
| LARFB temp size | (m-jb)×jb = 992×32 | jb×(n-jb) = 32×480 |
| Total LARFB | ~31KB/batch | ~15KB/batch |

**Why the difference:**
- GERQF processes **rows** → LARFB updates **columns** → temp is m x jb
- GEQRF processes **columns** → LARFB updates **rows** → temp is jb x n
- For tall matrices (m>n): GERQF uses MORE LARFB workspace
- For wide matrices (m<n): GEQRF uses MORE LARFB workspace""",
    "rationale": "Workspace requirements directly depend on which dimension (rows vs columns) is being factored. GERQF's row-based RQ needs different temp sizes than GEQRF's column-based QR, especially important for memory-constrained GPUs.",
    "tags": ["rq-factorization", "workspace-management", "larft", "larfb", "memory-optimization"]
})

# L2-2: Coding - Small matrix path selection
entries.append({
    "level": "L2",
    "task_type": "coding",
    "instruction": "Implement the logic in GERQF that decides between using the blocked algorithm (GERQF) or falling back to unblocked (GERQ2). Include workspace calculation for both paths.",
    "code_blocks": [
        """if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
{
    // requirements for a single GERQ2 call
    rocsolver_gerq2_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, size_work_workArr,
                                              size_Abyx_norms_trfact, size_diag_tmptr);
    *size_workArr = 0;
}
else
{
    size_t w1, w2, unused, s1, s2;
    rocblas_int jb = GExQF_BLOCKSIZE;

    // size to store the temporary triangular factor
    *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

    // requirements for calling GERQ2 with sub blocks
    rocsolver_gerq2_getMemorySize<BATCHED, T>(jb, n, batch_count, size_scalars, &w1, &s2, &s1);
    *size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

    // requirements for calling LARFT
    rocsolver_larft_getMemorySize<BATCHED, T>(n, jb, batch_count, &unused, &w2, size_workArr);

    // requirements for calling LARFB
    rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_right, m - jb, n, jb, batch_count,
                                              &s2, &unused);

    *size_work_workArr = std::max(w1, w2);
    *size_diag_tmptr = std::max(s1, s2);

    // size of workArr is double to accommodate
    // LARFB's TRMM calls in the batched case
    if(BATCHED)
        *size_workArr *= 2;
}"""
    ],
    "answer": """```cpp
template <bool BATCHED, typename T>
void rocsolver_gerqf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_trfact = 0;
        *size_diag_tmptr = 0;
        *size_workArr = 0;
        return;
    }

    // Small matrix path: use unblocked GERQ2
    if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    {
        rocsolver_gerq2_getMemorySize<BATCHED, T>(m, n, batch_count,
                                                  size_scalars,
                                                  size_work_workArr,
                                                  size_Abyx_norms_trfact,
                                                  size_diag_tmptr);
        *size_workArr = 0;  // GERQ2 doesn't need workArr
    }
    // Large matrix path: use blocked GERQF
    else
    {
        size_t w1, w2, unused, s1, s2;
        rocblas_int jb = GExQF_BLOCKSIZE;

        // T matrix for block reflector: jb x jb
        *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

        // GERQ2 workspace for jb x n blocks
        rocsolver_gerq2_getMemorySize<BATCHED, T>(jb, n, batch_count,
                                                  size_scalars, &w1, &s2, &s1);
        *size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

        // LARFT workspace (n, jb)
        rocsolver_larft_getMemorySize<BATCHED, T>(n, jb, batch_count,
                                                  &unused, &w2, size_workArr);

        // LARFB workspace (RIGHT side, m-jb x n, jb reflectors)
        rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_right, m - jb, n, jb,
                                                  batch_count, &s2, &unused);

        // Take max of GERQ2 and LARFT work buffers (reused)
        *size_work_workArr = std::max(w1, w2);

        // Take max of GERQ2 and LARFB temp buffers (reused)
        *size_diag_tmptr = std::max(s1, s2);

        // Batched LARFB needs 2x workArr for TRMM
        if(BATCHED)
            *size_workArr *= 2;
    }
}

// Execution path selection
if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    return rocsolver_gerq2_template<T>(handle, m, n, A, shiftA, lda, strideA,
                                       ipiv, strideP, batch_count, scalars,
                                       work_workArr, Abyx_norms_trfact, diag_tmptr);
else
    // blocked algorithm (main loop with GERQ2 + LARFT + LARFB)
    // ...
```

**Typical values (GExQF_GExQ2_SWITCHSIZE=64, GExQF_BLOCKSIZE=32):**
- 32x32 matrix → GERQ2 (unblocked)
- 128x256 matrix → GERQF (blocked)
- Threshold chosen to balance kernel launch overhead vs parallelism""",
    "rationale": "Path selection based on matrix size is critical for performance. Small matrices use simpler unblocked GERQ2 to avoid block reflector overhead, while large matrices benefit from blocked GERQF's better memory access patterns and Level-3 BLAS operations.",
    "tags": ["rq-factorization", "blocked-vs-unblocked", "workspace-management", "performance-optimization", "coding"]
})

# ========== L3 ENTRIES ==========

# L3-1: Analysis - Complete GERQF algorithm trace
entries.append({
    "level": "L3",
    "task_type": "analysis",
    "instruction": "Trace the complete GERQF algorithm for a 512x1024 double-precision complex matrix with batch size 1. Show all major steps, kernel calls, and workspace usage.",
    "code_blocks": [
        """template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gerqf_template(rocblas_handle handle,
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
    "answer": """**GERQF Execution Trace: 512x1024 complex double (batch=1)**

**Setup:**
- m=512, n=1024, k=min(512,1024)=512, lda=512
- Data type: rocblas_double_complex (16 bytes)
- GExQF_BLOCKSIZE=32, GExQF_GExQ2_SWITCHSIZE=64
- ki = ((512-64-1)/32)*32 = 448, kk = min(512, 448+32) = 480
- Initial: j = 512-480+448 = 480

**Workspace Allocation:**
1. Abyx_norms_trfact: 32×32×16 = 16KB (T matrix)
2. work_workArr: max(GERQ2_work, LARFT_work) ≈ 512B
3. diag_tmptr: max(GERQ2_diag, LARFB_temp) = (512-32)×32×16 ≈ 245KB
4. workArr: 32×32×16×2 = 32KB (batched TRMM workspace)

**Main Loop (j=480 down to 32, step -32):**

**Iteration 1 (j=480, last 32 rows):**
1. `jb = min(512-480, 32) = 32`
2. **GERQ2** on A[480:512, 0:1024] (32×1024 block, bottom 32 rows):
   - 32 LARFG calls: generate tau[480:512]
   - 31 LARF calls: apply reflectors from right
   - LACGV calls for complex conjugation (64 total: before and after each LARFG)
3. Update: `j = 480-32 = 448, mu = 512-512+448+32 = 480, nu = 1024-512+448+32 = 992`

**Iteration 2 (j=448, next 32 rows):**
1. `jb = min(512-448, 32) = 32`
2. **GERQ2** on A[448:480, 0:992] (32×992 block):
   - 32 LARFG + 31 LARF + 64 LACGV
3. **LARFT** (backward, row-wise, n=992, k=32):
   - GEMM: V×V' → T_init (32×992 × 992×32 → 32×32)
   - set_triangular kernel: fix diagonal, zero upper triangle
   - larft_kernel_backward: compute lower triangular T
4. **LARFB** (right, backward, row-wise, 448×992, k=32):
   - copymatA1: copy A[0:448, 0:32] to temp W (448×32)
   - TRMM: W = W×V1 (448×32 × triangular V)
   - GEMM: W = W + A2×V2 (448×32 + 448×960 × 960×32)
   - TRMM: W = W×T (448×32 × 32×32 lower triangular)
   - GEMM: A2 -= W×V2' (448×960 -= 448×32 × 32×960)
   - TRMM: W = W×V1' (448×32 × triangular V')
   - addmatA1: A1 -= W (448×32)
5. Update: `j = 416, mu = 448, nu = 960`

**Iterations 3-14:** Similar structure, processing 32-row blocks from bottom to top

**Final Block (mu=32, nu=544):**
- **GERQ2** on A[0:32, 0:544] (32×544 top block)
- No LARFT/LARFB (no rows above to update)

**Total Operations:**
- GERQ2 calls: 15 (one per block + final)
- LARFG calls: ~512 total
- LARF calls: ~256 (within GERQ2)
- LARFT calls: 14 (one per main iteration)
- LARFB calls: 14 (one per main iteration)
- LACGV calls: ~1024 (complex conjugation)

**Output:**
- tau: 512 complex scalars (Householder coefficients)
- A: overwritten with R (bottom-right) and V (row-wise reflectors)
- R factor: 512×512 upper triangular in A[0:512, 512:1024]
- V vectors: 512×1024 with implicit unit upper trapezoidal

**Memory Traffic (approximate):**
- Input/output: 512×1024×16×2 ≈ 16.8MB
- Workspace: ~300KB
- Total: ~17MB per batch""",
    "rationale": "Complete algorithm trace shows the interplay between GERQ2 (block factorization), LARFT (block reflector construction), and LARFB (block reflector application). Understanding the full execution flow, including workspace reuse and complex conjugation overhead, is essential for debugging and optimization.",
    "tags": ["rq-factorization", "algorithm-trace", "gerqf", "larft", "larfb", "complex-arithmetic"]
})

# L3-2: Coding - Complete production wrapper
entries.append({
    "level": "L3",
    "task_type": "coding",
    "instruction": "Implement a complete production GERQF wrapper including argument checking, workspace allocation, and execution for both regular and strided-batched cases.",
    "code_blocks": [
        """template <typename T, typename U>
rocblas_status rocsolver_gerqf_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("gerqf", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gerq2_gerqf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // extra requirements for calling GERQ2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GERQ2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_gerqf_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms_trfact, &size_diag_tmptr, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms_trfact, size_diag_tmptr,
                                                      size_workArr);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms_trfact, *diag_tmptr, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms_trfact,
                              size_diag_tmptr, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms_trfact = mem[2];
    diag_tmptr = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gerqf_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, stridep, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);
}"""
    ],
    "answer": """```cpp
// Complete production GERQF wrapper with error handling and workspace management
template <typename T, typename U>
rocblas_status rocsolver_gerqf_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("gerqf", "-m", m, "-n", n, "--lda", lda);

    // 1. Handle validation
    if(!handle)
        return rocblas_status_invalid_handle;

    // 2. Argument checking (shared with GERQ2)
    rocblas_status st = rocsolver_gerq2_gerqf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // 3. Setup for non-batched execution
    rocblas_int shiftA = 0;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // 4. Calculate workspace requirements
    size_t size_scalars;           // constants for rocBLAS calls
    size_t size_work_workArr;      // reusable work buffer (GERQ2/LARFT)
    size_t size_Abyx_norms_trfact; // T matrix + GERQ2 buffers
    size_t size_diag_tmptr;        // GERQ2 diag + LARFB temp
    size_t size_workArr;           // array of pointers (batched)

    rocsolver_gerqf_getMemorySize<false, T>(m, n, batch_count,
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
    return rocsolver_gerqf_template<false, false, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
        batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);
}

// Strided-batched variant
template <typename T, typename U>
rocblas_status rocsolver_gerqf_strided_batched_impl(rocblas_handle handle,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const rocblas_stride strideA,
                                                    T* ipiv,
                                                    const rocblas_stride strideP,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gerqf_strided_batched", "-m", m, "-n", n,
                        "--lda", lda, "--strideA", strideA,
                        "--strideP", strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // Argument checking
    rocblas_status st = rocsolver_gerq2_gerqf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    rocblas_int shiftA = 0;

    // Workspace calculation
    size_t size_scalars, size_work_workArr, size_Abyx_norms_trfact;
    size_t size_diag_tmptr, size_workArr;
    rocsolver_gerqf_getMemorySize<false, T>(m, n, batch_count,
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

    return rocsolver_gerqf_template<false, true, T>(
        handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
        batch_count, (T*)scalars, work_workArr,
        (T*)Abyx_norms_trfact, (T*)diag_tmptr, (T**)workArr);
}

// C API wrappers
extern "C" {

rocblas_status rocsolver_zgerqf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_gerqf_impl<rocblas_double_complex>(
        handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgerqf_strided_batched(rocblas_handle handle,
                                                const rocblas_int m,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* ipiv,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gerqf_strided_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, strideA, ipiv, strideP, batch_count);
}

} // extern "C"
```

**Key features:**
- Unified argument checking with GERQ2
- Workspace query support for optimal memory allocation
- Template-based implementation supporting all precisions (S/D/C/Z)
- Batched and strided-batched variants
- Proper error handling and memory management
- C API wrappers for LAPACK compatibility""",
    "rationale": "Production wrapper must handle all edge cases: invalid arguments, memory queries, workspace allocation failures. The template design allows code reuse across precisions while maintaining type safety. Strided-batched support is critical for modern GPU workloads.",
    "tags": ["rq-factorization", "gerqf", "production-code", "workspace-management", "batched-operations", "coding"]
})

# Write dataset
output_file = "/root/rocSOLVER/kernelgen/dataset/roclapack_gerqf.jsonl"
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
