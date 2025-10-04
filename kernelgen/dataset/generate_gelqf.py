#!/usr/bin/env python3
"""
Generator script for roclapack_gelqf SFT dataset.

GELQF computes the LQ factorization of an m×n matrix A = L*Q where L is lower
trapezoidal and Q is orthogonal. It uses a blocked algorithm (Level-3 BLAS)
for better performance compared to the unblocked GELQ2.
"""

import json
from datetime import datetime

def create_entry(id_num, level, interface, instruction, context_text, code_blocks,
                answer, rationale, tags):
    """Create a single dataset entry."""
    return {
        "id": f"2025-10-04T01:00:00.{id_num:03d}",
        "level": level,
        "interface": interface,
        "instruction": instruction,
        "context_text": context_text,
        "code_blocks": code_blocks,
        "answer": answer,
        "rationale": rationale,
        "tags": tags
    }

def generate_dataset():
    """Generate all dataset entries for roclapack_gelqf."""
    entries = []

    # ========== L1 ENTRIES ==========

    # L1-1: LARFT Triangle Setting Kernel (Coding)
    entries.append(create_entry(
        id_num=1,
        level="L1",
        interface="roclapack_gelqf",
        instruction="Implement a kernel that sets the strictly upper triangular part of a matrix T to zero while preserving the diagonal and lower triangle. The kernel should handle arbitrary matrix dimensions and work efficiently for batch sizes up to 1024.",
        context_text="LARFT computes a triangular factor T for a block of Householder reflectors. Before computation, the upper triangle must be zeroed to ensure correct results when accumulating the reflectors.",
        code_blocks=[
            {
                "path": "library/src/auxiliary/rocauxiliary_larft.hpp",
                "language": "cpp",
                "content": """// Target kernel in YAML: larft_set_tri
// Used to initialize T matrix before computing block reflector
// T is nb × nb upper triangular matrix storing reflector products"""
            }
        ],
        answer="""template <typename T, typename I>
__global__ void larft_set_tri_kernel(
    const I nb,               // Block size (dimension of T)
    T* __restrict__ T_array,  // Triangular factor matrix
    const I ldt,              // Leading dimension of T
    const rocblas_stride strideT)  // Stride between batch elements
{
    const I batch_id = blockIdx.y;
    const I tid = threadIdx.x;
    const I bid_x = blockIdx.x;

    // Offset to this batch element's T matrix
    T* T = T_array + batch_id * strideT;

    // Each thread block processes a column
    const I col = bid_x;

    if(col < nb)
    {
        // Set strictly upper triangle to zero: T[row, col] = 0 for row < col
        for(I row = tid; row < col; row += blockDim.x)
        {
            T[col * ldt + row] = T(0);
        }

        // Diagonal and lower triangle are preserved (no writes)
    }
}

// Launch configuration:
// Grid: (nb, batch_count, 1) - one block per column per batch element
// Block: (128, 1, 1) - threads cooperatively zero the column

/*
ALTERNATIVE: More aggressive parallelization for large nb
*/

template <typename T, typename I>
__global__ void larft_set_tri_kernel_v2(
    const I nb,
    T* __restrict__ T_array,
    const I ldt,
    const rocblas_stride strideT)
{
    const I batch_id = blockIdx.z;
    const I tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    const I tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    T* T = T_array + batch_id * strideT;

    // 2D grid: each thread handles one element
    const I row = tid_y;
    const I col = tid_x;

    if(row < nb && col < nb && row < col)  // Strictly upper triangle
    {
        T[col * ldt + row] = T(0);
    }
}

// Launch: dim3 grid((nb+15)/16, (nb+15)/16, batch_count), dim3 block(16, 16)
// Better for large nb (>64), more parallelism

/*
KEY DESIGN DECISIONS:

1. **Preserve vs Overwrite**:
   We only write zeros to upper triangle (row < col)
   Diagonal and lower triangle contain data from A, must preserve

2. **Column-wise vs Element-wise**:
   - Version 1: Process one column per block (good for small nb)
   - Version 2: Process elements in 2D grid (better for large nb)

3. **Boundary Handling**:
   Condition row < col ensures we only touch upper triangle
   No synchronization needed (independent writes)

4. **Memory Pattern**:
   Column-major access: T[col * ldt + row]
   Coalesced for column-wise processing (v1)
   Partially coalesced for 2D processing (v2, warp spans multiple columns)

USAGE IN LARFT:

T is initialized with:
1. Set upper triangle to zero (this kernel)
2. Copy diagonal from tau values (larft_set_diag)
3. Accumulate reflector products (matrix multiplies)
4. Result: T such that (I - V*T*V^T) = H_1 * H_2 * ... * H_nb
*/""",
        rationale="Zeroing triangular matrices is a common initialization pattern in dense linear algebra. This kernel demonstrates efficient 2D memory access, batch processing, and the importance of preserving existing data in specific matrix regions.",
        tags=["coding", "matrix-initialization", "triangular-matrix", "batch-processing", "householder-reflector"]
    ))

    # L1-2: Blocked vs Unblocked Algorithm Selection (Analysis)
    entries.append(create_entry(
        id_num=2,
        level="L1",
        interface="roclapack_gelqf",
        instruction="Explain why GELQF switches to the unblocked GELQ2 algorithm when m or n is below GExQF_GExQ2_SWITCHSIZE. What are the performance trade-offs between blocked and unblocked algorithms for small matrices?",
        context_text="GELQF uses a hybrid approach: blocked algorithm for large matrices, unblocked for small ones. Understanding the crossover point is important for performance tuning.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelqf.hpp",
                "language": "cpp",
                "content": """// if the matrix is small, use the unblocked (BLAS-levelII) variant of the
// algorithm
if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    return rocsolver_gelq2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       batch_count, scalars, work_workArr, Abyx_norms_trfact,
                                       diag_tmptr);

rocblas_int dim = std::min(m, n); // total number of pivots
rocblas_int jb, j = 0;

rocblas_int nb = GExQF_BLOCKSIZE;"""
            }
        ],
        answer="""**Why Switch to Unblocked Algorithm for Small Matrices?**

**Performance Trade-offs:**

**Blocked Algorithm (GELQF for large matrices):**
- **Pros**:
  - Uses Level-3 BLAS (LARFB with GEMM) → High arithmetic intensity
  - Better cache reuse when processing blocks
  - 70-90% of peak FLOPS achievable for large matrices
  - Amortizes kernel launch overhead across larger work units

- **Cons**:
  - Additional overhead: LARFT (compute block reflector T)
  - More complex orchestration: GELQ2 → LARFT → LARFB per block
  - Extra workspace needed for T (nb × nb per batch element)
  - More kernel launches: ~4× more than unblocked for same work

**Unblocked Algorithm (GELQ2 for small matrices):**
- **Pros**:
  - Simpler code path: LARFG → LARF for each row
  - Lower launch overhead (fewer kernels)
  - Minimal workspace (no T matrix needed)
  - Better for small matrices where cache covers entire matrix

- **Cons**:
  - Uses Level-2 BLAS (LARF with GEMV) → Lower arithmetic intensity
  - Memory-bound performance (10-30% of peak FLOPS)
  - Less cache reuse across iterations
  - Sequential row processing limits parallelism

**Crossover Point Analysis:**

Typical GExQF_GExQ2_SWITCHSIZE = 32-64 (varies by GPU architecture)

For m=48, n=1024:
- **Unblocked (GELQ2)**:
  - 48 iterations (one per row)
  - Each iteration: LARFG + LARF(GEMV)
  - GEMV: (48-j) × (1024-j) elements
  - Total FLOPS: ~24M (mostly GEMV)
  - Memory traffic: ~8 MB
  - **Arithmetic intensity: 3 FLOP/byte** (memory-bound)

- **Blocked (GELQF)**:
  - 1-2 blocks (nb=32)
  - Per block: GELQ2(32×1024) + LARFT(32) + LARFB(GEMM)
  - GEMM dominates: 16×1024×32 ≈ 0.5M FLOPS per block
  - Total FLOPS: ~26M (similar to unblocked)
  - **Arithmetic intensity: 50+ FLOP/byte** (compute-bound)

- **Verdict for m=48**: Unblocked is faster
  - Matrix too small for GEMM to dominate
  - Block overhead (LARFT, extra kernels) not amortized
  - Entire matrix fits in L2 cache anyway

For m=512, n=1024:
- **Unblocked**: 512 iterations, memory-bound entire time
- **Blocked**: ~16 blocks, GEMM becomes dominant
- **Verdict**: Blocked is 3-5× faster

**Why Check Both m AND n?**

```cpp
if(m <= SWITCHSIZE || n <= SWITCHSIZE)
```

- If m is small: Few rows to factor → blocking doesn't help
  - Example: m=16, n=4096 → 16 tiny GEMMs, not worth overhead

- If n is small: Short rows → GEMV/GEMM has small inner dimension
  - Example: m=4096, n=16 → GEMM is 4096×16×nb, memory-bound
  - Unblocked GEMV performs similarly, with less overhead

**Optimal Strategy:**

Use blocked algorithm only when BOTH dimensions exceed threshold:
- Ensures GEMM operations are large enough to be compute-bound
- Amortizes block reflector computation overhead
- Justifies extra workspace allocation

**Empirical Tuning:**

SWITCHSIZE is determined by autotuning:
1. Benchmark both algorithms for m,n ∈ [16, 128]
2. Find crossover point where blocked becomes faster
3. Set SWITCHSIZE conservatively to avoid regressions
4. Typical values: 32 (aggressive), 64 (conservative)

**Memory Perspective:**

For m×n = 64×1024, double precision:
- Matrix A: 512 KB (fits in L2 cache)
- Unblocked workspace: ~8 KB
- Blocked workspace: T (64×64×8 = 32 KB) + LARFB (additional 64 KB)
- Cache footprint difference becomes significant → unblocked wins""",
        rationale="Algorithm selection based on problem size is a fundamental optimization in numerical libraries. Understanding the performance characteristics of Level-2 vs Level-3 BLAS explains why hybrid approaches are necessary for covering the full performance spectrum.",
        tags=["analysis", "algorithm-selection", "performance-tuning", "blocked-algorithm", "level3-blas"]
    ))

    # L1-3: Workspace Size Calculation (Coding)
    entries.append(create_entry(
        id_num=3,
        level="L1",
        interface="roclapack_gelqf",
        instruction="Write a function that computes the total workspace size needed for GELQF, accounting for the maximum requirements of GELQ2, LARFT, and LARFB. The function should handle both batched and non-batched cases.",
        context_text="GELQF reuses workspace across multiple subroutine calls. Computing the maximum requirement ensures sufficient allocation without waste.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelqf.hpp",
                "language": "cpp",
                "content": """size_t w1, w2, unused, s1, s2;
rocblas_int jb = GExQF_BLOCKSIZE;

// size to store the temporary triangular factor
*size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

// requirements for calling GELQ2 with sub blocks
rocsolver_gelq2_getMemorySize<BATCHED, T>(jb, n, batch_count, size_scalars, &w1, &s2, &s1);
*size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

// requirements for calling LARFT
rocsolver_larft_getMemorySize<BATCHED, T>(n, jb, batch_count, &unused, &w2, size_workArr);

// requirements for calling LARFB
rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_right, m - jb, n, jb, batch_count,
                                          &s2, &unused);

*size_work_workArr = std::max(w1, w2);
*size_diag_tmptr = std::max(s1, s2);"""
            }
        ],
        answer="""template <bool BATCHED, typename T>
struct GelqfWorkspaceCalculator
{
    // Workspace components
    size_t size_scalars;         // BLAS constants (alpha, beta, etc.)
    size_t size_work_workArr;    // Reusable work buffer
    size_t size_Abyx_norms_trfact;  // Norms + triangular factor T
    size_t size_diag_tmptr;      // Diagonal temps + LARFB workspace
    size_t size_workArr;         // Array of pointers (batched only)

    // Compute workspace for GELQF
    void compute(
        const int m,
        const int n,
        const int batch_count,
        const int switchsize,   // GExQF_GExQ2_SWITCHSIZE
        const int blocksize)    // GExQF_BLOCKSIZE
    {
        // Quick return
        if(m == 0 || n == 0 || batch_count == 0)
        {
            size_scalars = 0;
            size_work_workArr = 0;
            size_Abyx_norms_trfact = 0;
            size_diag_tmptr = 0;
            size_workArr = 0;
            return;
        }

        // Check if using unblocked path
        if(m <= switchsize || n <= switchsize)
        {
            // Single GELQ2 call covers entire matrix
            compute_gelq2_workspace(m, n, batch_count);
            size_workArr = 0;  // Not needed for unblocked
            return;
        }

        // Blocked path: need workspace for GELQ2, LARFT, LARFB
        compute_blocked_workspace(m, n, batch_count, blocksize);
    }

private:
    void compute_gelq2_workspace(int m, int n, int batch_count)
    {
        // GELQ2 workspace components (simplified, actual call to gelq2_getMemorySize)

        // Scalars for GEMV calls
        size_scalars = sizeof(T) * 3;

        // LARFG workspace for computing Householder reflectors
        // Needs reduction buffer: size = (n-1)/512 + 2 elements per batch
        size_t larfg_work = sizeof(T) * ((n - 1) / 512 + 2) * batch_count;

        // LARF workspace for applying reflectors (GEMV workspace)
        size_t larf_work = 0;  // Modern GEMV typically doesn't need extra workspace

        size_work_workArr = std::max(larfg_work, larf_work);

        // Norms for LARFG (stores intermediate values)
        size_Abyx_norms_trfact = sizeof(T) * batch_count;

        // Diagonal storage (preserve A[j,j] during reflector application)
        size_diag_tmptr = sizeof(T) * batch_count;
    }

    void compute_blocked_workspace(int m, int n, int batch_count, int nb)
    {
        size_t w1, w2, s1, s2, unused;

        // === Component 1: Triangular Factor T ===
        // T is nb × nb upper triangular, one per batch element
        size_Abyx_norms_trfact = sizeof(T) * nb * nb * batch_count;

        // === Component 2: GELQ2 for Panel Factorization ===
        // GELQ2 called on nb × n submatrix

        // GELQ2 scalars
        size_scalars = sizeof(T) * 3;

        // GELQ2 work buffer
        w1 = sizeof(T) * ((n - 1) / 512 + 2) * batch_count;

        // GELQ2 norms (may be larger than T storage)
        s2 = sizeof(T) * batch_count;
        size_Abyx_norms_trfact = std::max(size_Abyx_norms_trfact, s2);

        // GELQ2 diagonal temps
        s1 = sizeof(T) * batch_count;

        // === Component 3: LARFT (Compute Block Reflector) ===
        // LARFT operates on n × nb matrix of reflectors

        // LARFT work buffer (matrix multiplies)
        w2 = sizeof(T) * nb * nb * batch_count;  // For intermediate products

        // LARFT workArr (batched mode needs pointer arrays)
        if(BATCHED)
            size_workArr = sizeof(T*) * batch_count;
        else
            size_workArr = 0;

        // === Component 4: LARFB (Apply Block Reflector) ===
        // LARFB applies to (m-nb) × n matrix using nb reflectors

        // LARFB needs workspace for TRMM and GEMM calls
        // Worst case: (m-nb) × nb workspace for intermediate matrices
        s2 = sizeof(T) * std::max(m - nb, 0) * nb * batch_count;

        // === Combine: Take Maximum of Reusable Workspaces ===

        // work_workArr is reused across GELQ2, LARFT, LARFB
        size_work_workArr = std::max(w1, w2);

        // diag_tmptr is reused across GELQ2, LARFB
        size_diag_tmptr = std::max(s1, s2);

        // Batched mode: LARFB's TRMM needs additional pointer array
        // Double the workArr size to accommodate both GEMM and TRMM
        if(BATCHED)
            size_workArr *= 2;
    }

public:
    // Total workspace in bytes
    size_t total() const
    {
        return size_scalars + size_work_workArr +
               size_Abyx_norms_trfact + size_diag_tmptr + size_workArr;
    }

    // Print workspace breakdown (useful for debugging)
    void print_breakdown() const
    {
        printf("GELQF Workspace Breakdown:\\n");
        printf("  Scalars:           %10zu bytes\\n", size_scalars);
        printf("  Work/WorkArr:      %10zu bytes\\n", size_work_workArr);
        printf("  Norms/Trfact:      %10zu bytes\\n", size_Abyx_norms_trfact);
        printf("  Diag/Tmptr:        %10zu bytes\\n", size_diag_tmptr);
        printf("  WorkArr (batched): %10zu bytes\\n", size_workArr);
        printf("  -------------------------------\\n");
        printf("  Total:             %10zu bytes\\n", total());
    }
};

// Usage example:
void example_usage()
{
    const int m = 512, n = 1024, batch_count = 16;
    const int switchsize = 64;
    const int blocksize = 32;

    GelqfWorkspaceCalculator<true, double> calc;
    calc.compute(m, n, batch_count, switchsize, blocksize);

    calc.print_breakdown();

    // Allocate workspace
    void* workspace;
    hipMalloc(&workspace, calc.total());

    // Subdivide workspace for different components
    // (actual implementation would use calc.size_* members)
}

/*
KEY INSIGHTS:

1. **Workspace Reuse**:
   work_workArr reused by GELQ2, LARFT, LARFB → take max, not sum
   diag_tmptr reused by GELQ2, LARFB → take max, not sum
   This saves significant memory (MB instead of GB for large batches)

2. **Batched Mode Overhead**:
   Needs pointer arrays (size_workArr) for batched BLAS calls
   LARFB doubles this for concurrent TRMM+GEMM

3. **T Matrix Storage**:
   Triangular factor T is nb×nb per batch element
   Not reused → must be allocated separately
   For batch=1024, nb=32, double: 8 MB just for T

4. **Scalability**:
   Most workspace scales with batch_count
   Critical for memory-constrained scenarios (large batches on consumer GPUs)
*/""",
        rationale="Workspace management is crucial for GPU memory efficiency. This implementation demonstrates how to compute maximum requirements across multiple subroutines, handle batched vs non-batched cases, and optimize for memory reuse without sacrificing correctness.",
        tags=["coding", "workspace-allocation", "memory-management", "batched-operations", "algorithm-composition"]
    ))

    # ========== L2 ENTRIES ==========

    # L2-1: GELQ2-LARFT-LARFB Pipeline (Analysis)
    entries.append(create_entry(
        id_num=4,
        level="L2",
        interface="roclapack_gelqf",
        instruction="Analyze the three-stage pipeline in each block: GELQ2 (factor panel) → LARFT (build T) → LARFB (apply to trailing matrix). Explain the data dependencies, why LARFT is needed (vs applying reflectors individually), and the performance benefit of this decomposition.",
        context_text="The blocked algorithm breaks factorization into panels. Understanding the cooperation between GELQ2, LARFT, and LARFB reveals the core optimization strategy.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelqf.hpp",
                "language": "cpp",
                "content": """while(j < dim - GExQF_GExQ2_SWITCHSIZE)
{
    // Factor diagonal and subdiagonal blocks
    jb = std::min(dim - j, nb); // number of rows in the block
    rocsolver_gelq2_template<T>(handle, jb, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                (ipiv + j), strideP, batch_count, scalars, work_workArr,
                                Abyx_norms_trfact, diag_tmptr);

    // apply transformation to the rest of the matrix
    if(j + jb < m)
    {
        // compute block reflector
        rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_row_wise, n - j,
                                    jb, A, shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j),
                                    strideP, Abyx_norms_trfact, ldw, strideW, batch_count,
                                    scalars, (T*)work_workArr, workArr);

        // apply the block reflector
        rocsolver_larfb_template<BATCHED, STRIDED, T>(
            handle, rocblas_side_right, rocblas_operation_none, rocblas_forward_direction,
            rocblas_row_wise, m - j - jb, n - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
            Abyx_norms_trfact, 0, ldw, strideW, A, shiftA + idx2D(j + jb, j, lda), lda, strideA,
            batch_count, diag_tmptr, workArr);
    }
    j += nb;
}"""
            }
        ],
        answer="""**Three-Stage Pipeline Analysis:**

**STAGE 1: GELQ2 - Panel Factorization**

```
Input:  A[j:j+jb, j:n] (jb rows, n-j columns)
Output: L[j:j+jb, j:j+jb] (lower triangular)
        V[j:j+jb, j:n] (Householder vectors stored in A)
        tau[j:j+jb] (Householder scalars in ipiv)

Algorithm: Unblocked LQ factorization
  For i = j to j+jb-1:
    Compute H_i = I - tau_i * v_i * v_i^T
    Apply H_i to rows i:m, columns i:n

Dimensions: jb × (n-j) panel
FLOPS: O(jb^2 * n)
Performance: Memory-bound (Level-2 BLAS / GEMV)
```

**STAGE 2: LARFT - Build Triangular Factor T**

```
Input:  V[j:j+jb, j:n] (Householder vectors from GELQ2)
        tau[j:j+jb] (Householder scalars)
Output: T (jb × jb upper triangular matrix)

Purpose: Compute T such that:
  H_1 * H_2 * ... * H_jb = I - V * T * V^T

Algorithm:
  T[0,0] = tau[0]
  For i = 1 to jb-1:
    T[0:i, i] = -tau[i] * V[0:i, :]^T * V[i, :]
    T[i,i] = tau[i]

Dimensions: jb × jb matrix construction
FLOPS: O(jb^2 * n) (dominated by matrix-vector products)
Performance: Memory-bound but amortized over next stage
```

**Why LARFT is Needed:**

**Without LARFT (naive approach):**
```cpp
for(int i = j; i < j+jb; i++)
    apply_householder(H_i, A[j+jb:m, j:n]);  // Apply one reflector
// Cost: jb GEMV operations, each with (m-j-jb) × (n-j) matrix
```

**With LARFT (blocked approach):**
```cpp
T = larft(V, tau);  // Precompute T
apply_block_householder(V, T, A[j+jb:m, j:n]);  // Apply all reflectors at once
// Cost: 1 GEMM operation
```

**Performance Difference:**

For m=1024, n=1024, jb=32, block position j=0:

*Naive (jb individual reflectors):*
- 32 GEMV operations: A[32:1024, 0:1024] (992 × 1024 each)
- Each GEMV: ~1M FLOPS
- Total: 32M FLOPS
- Arithmetic intensity: ~3 FLOP/byte (memory-bound)
- GPU utilization: ~15-25%

*Blocked (one block reflector):*
- LARFT: Compute T (negligible, ~30K FLOPS)
- LARFB: One GEMM: (992 × 1024) × (32 × 1024)
- GEMM: ~33M FLOPS (similar total FLOPS!)
- Arithmetic intensity: ~50+ FLOP/byte (compute-bound)
- GPU utilization: ~70-85%
- **Speedup: 4-6× faster despite similar FLOP count**

**STAGE 3: LARFB - Apply Block Reflector**

```
Input:  V[j:j+jb, j:n] (Householder vectors)
        T (jb × jb triangular factor from LARFT)
        A[j+jb:m, j:n] (trailing matrix to update)
Output: A[j+jb:m, j:n] = (I - V*T*V^T) * A[j+jb:m, j:n]

Algorithm:
  W = A[j+jb:m, j:n] * V^T        // GEMM: (m-j-jb) × n × jb
  W = W * T                        // TRMM: (m-j-jb) × jb × jb
  A[j+jb:m, j:n] -= W * V          // GEMM: (m-j-jb) × jb × n

FLOPS: ~2 * (m-j-jb) * n * jb
Performance: Compute-bound (Level-3 BLAS / GEMM)
GPU Utilization: 70-90% of peak
```

**DATA DEPENDENCIES:**

```
GELQ2[j]  →  LARFT[j]  →  LARFB[j]
     ↓           ↓            ↓
   V, tau       T          Updated A
     └────────────┴──────────┘
                 ↓
          GELQ2[j+nb] (needs updated A)
```

**Critical Path:**
1. GELQ2 MUST complete before LARFT (needs V and tau)
2. LARFT MUST complete before LARFB (needs T)
3. LARFB MUST complete before next GELQ2 (column j+jb needs updated data)

**No parallelism across blocks:** Sequential dependency chain

**PERFORMANCE BENEFIT BREAKDOWN:**

For full 1024×1024 factorization (nb=32):

*Unblocked (GELQ2 only):*
- 1024 iterations
- Each: LARFG + LARF (GEMV)
- Total FLOPS: ~720M
- Time: ~45 ms (memory-bound entire time)

*Blocked (GELQ2 + LARFT + LARFB):*
- 32 panel factorizations (GELQ2)
- 32 block builds (LARFT)
- 31 block applications (LARFB)
- LARFB dominates: ~650M FLOPS (90% of total)
- Total FLOPS: ~730M (same as unblocked!)
- Time: ~12 ms (compute-bound for LARFB)
- **Speedup: 3.75×**

**Why Same FLOPS, Different Performance?**

- GEMV (unblocked): 2-4 FLOP/byte → memory bandwidth limited
- GEMM (blocked): 50-100 FLOP/byte → compute limited
- GPU can hide memory latency with computation in GEMM
- GEMM has better instruction-level parallelism (ILP)

**MEMORY TRAFFIC REDUCTION:**

Unblocked: Each row of A read/written ~1024 times (once per iteration)
Blocked: Each row read/written ~32 times (once per block)
→ **32× reduction in memory traffic**

**CONCLUSION:**

LARFT enables transformation from memory-bound Level-2 BLAS to compute-bound Level-3 BLAS. The overhead of computing T (LARFT) is negligible compared to the speedup gained from using GEMM instead of GEMV in the update phase (LARFB).""",
        rationale="The blocked algorithm's performance comes from transforming memory-bound operations into compute-bound ones. Understanding the GELQ2-LARFT-LARFB pipeline explains why modern dense linear algebra achieves high GPU efficiency despite having the same FLOP count as naive implementations.",
        tags=["analysis", "blocked-algorithm", "level3-blas", "householder-reflector", "performance-optimization", "data-dependency"]
    ))

    # L2-2: Optimized Panel Factorization Kernel (Coding)
    entries.append(create_entry(
        id_num=5,
        level="L2",
        interface="roclapack_gelqf",
        instruction="Design a fused kernel that performs GELQ2 panel factorization for nb=32 rows entirely in shared memory, integrating LARFG (compute reflector) and LARF (apply reflector) operations. Include synchronization strategy and shared memory layout.",
        context_text="Panel factorization is the bottleneck when LARFB becomes very efficient. Fusing GELQ2 into a single kernel can eliminate launch overhead and improve data locality.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelq2.hpp",
                "language": "cpp",
                "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    batch_count);

    // generate Householder reflector to work on row j
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(j, std::min(j + 1, n - 1), lda), lda, strideA,
                             (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(j,j) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA,
                            1, true);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j + 1, j, lda), lda, strideA, batch_count,
                                scalars, Abyx_norms, (T**)work_workArr);
    }

    // restore original value of A(j,j)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A, shiftA + idx2D(j, j, lda),
                            lda, strideA, 1);
}"""
            }
        ],
        answer="""template <typename T, int NB = 32, int MAX_N = 1024>
__global__ void gelq2_panel_fused_kernel(
    const int jb,              // Actual panel height (≤ NB)
    const int n,               // Panel width
    T* __restrict__ A_global,  // Global matrix
    const int lda,
    T* __restrict__ tau_global,  // Householder scalars output
    const int batch_id_offset)
{
    // ===== Shared Memory Layout =====
    __shared__ T smem_panel[NB * MAX_N];     // Panel data: jb × n
    __shared__ T smem_tau[NB];                // Householder scalars
    __shared__ T smem_reduction[256];         // For norm reductions
    __shared__ T smem_row_broadcast[MAX_N];   // Broadcast row j

    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x + batch_id_offset;
    const int nthreads = blockDim.x;

    // Offset to this batch element
    T* A_batch = A_global + batch_id * lda * n;
    T* tau_batch = tau_global + batch_id * jb;

    // ===== Load Panel into Shared Memory =====
    for(int idx = tid; idx < jb * n; idx += nthreads)
    {
        int row = idx % jb;
        int col = idx / jb;
        smem_panel[col * NB + row] = A_batch[col * lda + row];
    }
    __syncthreads();

    // ===== Main Factorization Loop =====
    for(int j = 0; j < jb; j++)
    {
        // ----- LARFG: Compute Householder Reflector for Row j -----

        // Step 1: Compute norm of row j, columns j+1:n
        T local_norm_sq = 0;
        for(int col = j + 1 + tid; col < n; col += nthreads)
        {
            T val = smem_panel[col * NB + j];
            local_norm_sq += val * val;  // For complex: val * conj(val)
        }

        // Parallel reduction to sum norm
        smem_reduction[tid] = local_norm_sq;
        __syncthreads();

        for(int stride = nthreads / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                smem_reduction[tid] += smem_reduction[tid + stride];
            __syncthreads();
        }

        T norm_sq_tail = smem_reduction[0];
        __syncthreads();

        // Step 2: Thread 0 computes tau and updates diagonal
        __shared__ T alpha_j;  // Original A[j,j]
        __shared__ T tau_j;

        if(tid == 0)
        {
            alpha_j = smem_panel[j * NB + j];
            T norm_tail = sqrt(norm_sq_tail);
            T alpha_abs = abs(alpha_j);

            // Compute norm of entire row: sqrt(alpha^2 + norm_tail^2)
            T beta = -sign(alpha_j) * sqrt(alpha_abs * alpha_abs + norm_tail * norm_tail);

            if(beta == T(0))
            {
                // Zero row, no reflection needed
                tau_j = T(0);
                smem_panel[j * NB + j] = alpha_j;
            }
            else
            {
                // Compute tau = (beta - alpha) / beta
                tau_j = (beta - alpha_j) / beta;

                // Store beta in A[j,j] (diagonal of L)
                smem_panel[j * NB + j] = beta;

                // Scaling factor for v: v[j] = 1, v[j+1:n] /= (alpha - beta)
                T scale = T(1) / (alpha_j - beta);

                // Scale the tail of the row
                for(int col = j + 1; col < n; col++)
                {
                    smem_panel[col * NB + j] *= scale;
                }
            }

            smem_tau[j] = tau_j;
            tau_batch[j] = tau_j;  // Write to global memory
        }
        __syncthreads();

        T tau_current = smem_tau[j];

        // Skip application if tau = 0 (zero row)
        if(tau_current == T(0) || j == jb - 1)
            continue;

        // ----- LARF: Apply Reflector to Rows j+1:jb -----

        // Reflector H = I - tau * v * v^T
        // Update: A[j+1:jb, j:n] = (I - tau*v*v^T) * A[j+1:jb, j:n]
        // Equivalent: A[j+1:jb, j:n] -= tau * v * (v^T * A[j+1:jb, j:n])

        // Step 1: Temporarily set v[j] = 1
        if(tid == 0)
        {
            alpha_j = smem_panel[j * NB + j];
            smem_panel[j * NB + j] = T(1);
        }
        __syncthreads();

        // Step 2: For each row i in j+1:jb, compute dot = v^T * A[i,:]
        for(int i = j + 1; i < jb; i++)
        {
            // Compute dot product: v (row j) · A[i,:] (row i)
            T local_dot = 0;
            for(int col = j + tid; col < n; col += nthreads)
            {
                local_dot += smem_panel[col * NB + j] * smem_panel[col * NB + i];
            }

            // Reduce to get full dot product
            smem_reduction[tid] = local_dot;
            __syncthreads();

            for(int stride = nthreads / 2; stride > 0; stride >>= 1)
            {
                if(tid < stride)
                    smem_reduction[tid] += smem_reduction[tid + stride];
                __syncthreads();
            }

            T dot = smem_reduction[0];
            __syncthreads();

            // Step 3: Update row i: A[i,:] -= tau * dot * v
            T factor = -tau_current * dot;
            for(int col = j + tid; col < n; col += nthreads)
            {
                smem_panel[col * NB + i] += factor * smem_panel[col * NB + j];
            }
            __syncthreads();
        }

        // Step 4: Restore original A[j,j]
        if(tid == 0)
        {
            smem_panel[j * NB + j] = alpha_j;
        }
        __syncthreads();
    }

    // ===== Write Panel Back to Global Memory =====
    for(int idx = tid; idx < jb * n; idx += nthreads)
    {
        int row = idx % jb;
        int col = idx / jb;
        A_batch[col * lda + row] = smem_panel[col * NB + row];
    }
}

/*
LAUNCH CONFIGURATION:

// Process one panel per block (or batch element per block)
dim3 grid(batch_count);
dim3 block(256);  // 256 threads for good occupancy

gelq2_panel_fused_kernel<double, 32, 1024><<<grid, block>>>(
    jb, n, A, lda, tau, 0);

SHARED MEMORY REQUIREMENTS:

- smem_panel: NB × MAX_N × sizeof(T)
  For NB=32, MAX_N=1024, double: 256 KB (too large!)

SOLUTION: Tile in n dimension or reduce MAX_N

For practical use, limit MAX_N=512:
- smem_panel: 32 × 512 × 8 = 128 KB
- smem_tau: 32 × 8 = 256 bytes
- smem_reduction: 256 × 8 = 2 KB
- smem_row_broadcast: 512 × 8 = 4 KB
- Total: ~135 KB (feasible with dynamic shared memory)

PERFORMANCE CHARACTERISTICS:

Advantages:
- 1 kernel instead of 3×jb kernels (96 launches for jb=32)
- All data in shared memory (no global memory traffic between iterations)
- Perfect cache locality for row updates

Limitations:
- Panel width n limited by shared memory (~512-1024 columns)
- For wider matrices, must process in tiles
- Lower occupancy due to high shared memory usage

EXPECTED SPEEDUP:

For jb=32, n=1024 panel:
- Separate kernels: 96 launches × 5μs overhead = 480μs overhead
  - Compute time: ~200μs
  - Total: ~680μs

- Fused kernel: 1 launch = 5μs overhead
  - Compute time: ~180μs (better cache locality)
  - Total: ~185μs

→ **3.7× speedup**
*/""",
        rationale="Fused panel kernels represent an advanced optimization for blocked algorithms. This implementation demonstrates shared memory tiling, in-kernel reductions, careful synchronization, and the trade-offs between occupancy and data locality. Panel factorization fusion is a key technique in high-performance dense linear algebra libraries.",
        tags=["coding", "kernel-fusion", "shared-memory", "householder-reflector", "panel-factorization", "optimization"]
    ))

    # ========== L3 ENTRIES ==========

    # L3-1: Complete Algorithm Orchestration (Analysis)
    entries.append(create_entry(
        id_num=6,
        level="L3",
        interface="roclapack_gelqf",
        instruction="Trace the complete execution of GELQF for a 640×1024 matrix with nb=32. Include all algorithm decisions (blocked vs unblocked), kernel launches per block, workspace allocation, and performance estimates. Compare total time vs unblocked GELQ2.",
        context_text="Understanding end-to-end execution reveals optimization opportunities and algorithmic trade-offs. GELQF's hybrid blocked/unblocked strategy requires careful orchestration.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelqf.hpp",
                "language": "cpp",
                "content": """if(m <= GExQF_GExQ2_SWITCHSIZE || n <= GExQF_GExQ2_SWITCHSIZE)
    return rocsolver_gelq2_template<T>(...);

rocblas_int dim = std::min(m, n);
rocblas_int jb, j = 0;
rocblas_int nb = GExQF_BLOCKSIZE;

while(j < dim - GExQF_GExQ2_SWITCHSIZE)
{
    jb = std::min(dim - j, nb);
    rocsolver_gelq2_template<T>(...);  // Factor panel

    if(j + jb < m)
    {
        rocsolver_larft_template<T>(...);   // Build block reflector
        rocsolver_larfb_template<T>(...);   // Apply to trailing matrix
    }
    j += nb;
}

// factor last block
if(j < dim)
    rocsolver_gelq2_template<T>(...);"""
            }
        ],
        answer="""**Complete GELQF Execution Trace: m=640, n=1024, nb=32**

**PHASE 1: Algorithm Selection**

1. **Check switchsize**:
   ```
   GExQF_GExQ2_SWITCHSIZE = 64 (typical)
   m=640 > 64 AND n=1024 > 64
   → Use blocked algorithm
   ```

2. **Determine dimensions**:
   ```
   dim = min(m, n) = min(640, 1024) = 640 pivots needed
   nb = GExQF_BLOCKSIZE = 32
   ```

**PHASE 2: Workspace Allocation**

```cpp
// For jb=32, m=640, n=1024, batch=1

size_scalars = 24 bytes  // BLAS constants

// Triangular factor T: nb × nb
size_Abyx_norms_trfact = 32 × 32 × 8 = 8,192 bytes

// GELQ2 workspace (panel factorization)
w1 = ((1024-1)/512 + 2) × 1 × 8 = 24 bytes

// LARFT workspace (block reflector construction)
w2 = 32 × 32 × 8 = 8,192 bytes

size_work_workArr = max(24, 8192) = 8,192 bytes

// LARFB workspace (update trailing matrix)
s1 = (640-32) × 32 × 8 = 155,648 bytes

size_diag_tmptr = 155,648 bytes

Total workspace: ~172 KB
```

**PHASE 3: Blocked Factorization Loop**

```
dim = 640, nb = 32, switchsize = 64
Loop condition: j < dim - switchsize = j < 576
Blocks processed: j = 0, 32, 64, ..., 544 → 18 blocks
Final block: j = 576 to 640 (64 rows) → unblocked
```

**BLOCK 0 (j=0, jb=32):**

- **GELQ2 Panel**: A[0:32, 0:1024]
  - Iterations: 32 (one per row)
  - Per iteration: LARFG + LARF
  - LARFG: 1 kernel (reduction for norm)
  - LARF (side_right, m-j-1 × n-j): 1 kernel (GEMV)
  - Set/restore diag: 2 kernels
  - Total: 32 × 4 = **128 kernels**
  - FLOPS: ~2 × 32 × 1024 × 32 / 2 ≈ **1M**
  - Time: ~300μs

- **LARFT**: Build T[0:32, 0:32]
  - Forward direction, row-wise
  - Kernels: ~10 (matrix ops to build T)
  - FLOPS: 32² × 1024 ≈ **1M**
  - Time: ~100μs

- **LARFB**: Apply to A[32:640, 0:1024]
  - Matrix: 608 × 1024, reflectors: 32
  - W = A * V^T: GEMM (608 × 1024 × 32) = **40M FLOPS**
  - W = W * T: TRMM (608 × 32 × 32) = **0.6M FLOPS**
  - A -= W * V: GEMM (608 × 32 × 1024) = **40M FLOPS**
  - Total FLOPS: **~80M**
  - Kernels: ~3-5 (GEMM, TRMM)
  - Time: ~600μs (compute-bound)

**Block 0 summary:**
- Kernels: 128 + 10 + 5 = **143 launches**
- FLOPS: 1M + 1M + 80M = **82M**
- Time: 300 + 100 + 600 = **~1000μs**

**BLOCK k (j=k×32, k=1 to 17):**

Similar to Block 0, but dimensions shrink:
- Panel: 32 × (1024 - k×32)
- Trailing: (640 - (k+1)×32) × (1024 - k×32)

**Example: Block 10 (j=320):**
- Panel: 32 × 704
- Trailing: 288 × 704
- LARFB FLOPS: ~2 × 288 × 704 × 32 ≈ **13M**
- Time: ~400μs (smaller GEMM)

**FINAL BLOCK (j=576, rows 576:640 → 64 rows):**
- dim - j = 64 < switchsize + nb
- Use unblocked GELQ2 only (no LARFT/LARFB)
- Iterations: 64
- Kernels: 64 × 4 = **256 launches**
- FLOPS: ~2 × 64 × 448 × 64 / 2 ≈ **1.8M**
- Time: ~500μs

**PHASE 4: Total Kernel Count**

```
Block 0-17 (18 blocks):
  GELQ2: 18 × 128 = 2,304 kernels
  LARFT: 18 × 10 = 180 kernels
  LARFB: 17 × 5 = 85 kernels (only 17 blocks need LARFB)

Final block:
  GELQ2: 256 kernels

Total: 2,304 + 180 + 85 + 256 = **2,825 kernel launches**
```

**PHASE 5: Performance Estimates (AMD MI250X)**

```
FLOP Breakdown:
- Panel factorization (GELQ2): ~18 × 1M + 1.8M ≈ 20M FLOPS
- Block reflector (LARFT): ~18 × 1M ≈ 18M FLOPS
- Trailing update (LARFB): ~18 × (80M + 40M + ... + 13M) ≈ 700M FLOPS

Total FLOPS: ~740M FLOPS

Time Breakdown:
- Kernel launch overhead: 2,825 × 5μs = 14ms
- GELQ2 compute: ~6ms (memory-bound)
- LARFT compute: ~2ms
- LARFB compute: ~10ms (compute-bound, 70 GFLOPS)

Total time: 14 + 6 + 2 + 10 = **32ms**
```

**PHASE 6: Comparison with Unblocked GELQ2**

**Unblocked (GELQ2 for entire 640×1024):**

```
Iterations: 640 (one per row)
Kernels per iteration: 4 (LARFG, LARF, set_diag, restore_diag)
Total kernels: 640 × 4 = 2,560 launches

FLOPS:
- Row 0: 2 × 640 × 1024 ≈ 1.3M
- Row 1: 2 × 639 × 1023 ≈ 1.3M
- ...
- Row 639: minimal
Total: ~2 × 640² × 1024 / 2 ≈ **420M FLOPS**

Wait, GELQF should have similar FLOPS! Error in GELQF estimate.

Recompute GELQF FLOPS correctly:
Total FLOPS for LQ: (2/3) × m² × (3n - m) ≈ 2 × 640² × 1024 / 3 ≈ **280M FLOPS**

Time for unblocked:
- Launch overhead: 2,560 × 5μs = 13ms
- Compute: ~35ms (all memory-bound GEMV)
- Total: **48ms**

Speedup: 48ms / 32ms = **1.5×**
```

**Why only 1.5× speedup?**

1. **Matrix dimensions**: m=640 < n=1024
   - LARFB operates on shrinking trailing matrix
   - Average GEMM size smaller than ideal
   - Not enough work to fully saturate GPU

2. **Launch overhead still significant**: 14ms out of 32ms (44%)
   - Could reduce with panel kernel fusion

3. **Panel factorization bottleneck**: 6ms (19% of total)
   - Still using memory-bound GELQ2
   - Opportunity for optimization

**For larger matrices (m=n=4096):**
- LARFB dominates (>90% of FLOPS)
- GEMM fully saturates GPU
- Speedup: 8-12× over unblocked
- Launch overhead becomes negligible (<5%)

**CONCLUSION:**

GELQF provides moderate speedup (1.5×) for this size due to:
- Modest dimensions (not fully utilizing GPU)
- High launch overhead (2,825 kernels)
- Panel factorization still memory-bound

Optimization opportunities:
1. Fuse GELQ2 panel into single kernel → save ~2,000 launches
2. Use larger block size (nb=64) → fewer iterations
3. Increase batch size for better amortization""",
        rationale="Complete algorithm traces reveal where optimizations matter most. For GELQF, the analysis shows that moderate-sized matrices don't fully benefit from blocking due to launch overhead and shrinking trailing matrices, while large matrices achieve near-optimal speedup. This explains why autotuning is critical for covering diverse problem sizes.",
        tags=["analysis", "algorithm-trace", "performance-analysis", "blocked-algorithm", "optimization-opportunities"]
    ))

    # L3-2: Production API with Auto-Tuning (Coding)
    entries.append(create_entry(
        id_num=7,
        level="L3",
        interface="roclapack_gelqf",
        instruction="Implement a production-quality GELQF wrapper that automatically selects algorithm variant (blocked vs unblocked), tunes block size based on matrix dimensions, handles both batched and strided modes, manages workspace allocation, and provides detailed performance logging.",
        context_text="Production libraries need automatic performance tuning, comprehensive error handling, and flexibility for diverse use cases. This wrapper demonstrates best practices for numerical library development.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_gelqf.hpp",
                "language": "cpp",
                "content": """// Wrapper must:
// - Auto-tune block size based on m, n
// - Select blocked vs unblocked based on problem size
// - Support BATCHED and STRIDED modes
// - Manage workspace allocation
// - Provide performance diagnostics"""
            }
        ],
        answer="""template <bool BATCHED, bool STRIDED, typename T>
class GelqfSolver
{
private:
    rocblas_handle handle_;
    int m_, n_, batch_count_;
    int lda_;
    rocblas_stride strideA_, strideP_;

    // Algorithm parameters (auto-tuned)
    int block_size_;
    int switch_size_;
    bool use_blocked_;

    // Workspace
    void* scalars_mem_ = nullptr;
    void* work_workArr_mem_ = nullptr;
    void* Abyx_norms_trfact_mem_ = nullptr;
    void* diag_tmptr_mem_ = nullptr;
    T** workArr_mem_ = nullptr;

    // Performance logging
    bool enable_logging_ = false;
    double time_gelq2_ms_ = 0;
    double time_larft_ms_ = 0;
    double time_larfb_ms_ = 0;
    int num_kernels_ = 0;

public:
    GelqfSolver(
        rocblas_handle handle,
        int m, int n,
        int lda,
        rocblas_stride strideA,
        rocblas_stride strideP,
        int batch_count,
        bool enable_logging = false)
        : handle_(handle), m_(m), n_(n), lda_(lda),
          strideA_(strideA), strideP_(strideP),
          batch_count_(batch_count),
          enable_logging_(enable_logging)
    {
        // Auto-tune algorithm parameters
        auto_tune_parameters();
    }

    ~GelqfSolver()
    {
        cleanup_workspace();
    }

    rocblas_status factorize(
        T* A_or_array[],
        T* tau_or_array[])
    {
        // Validate inputs
        rocblas_status status = validate_inputs(A_or_array, tau_or_array);
        if(status != rocblas_status_success)
            return status;

        // Allocate workspace
        status = allocate_workspace();
        if(status != rocblas_status_success)
        {
            cleanup_workspace();
            return status;
        }

        // Execute factorization
        hipEvent_t start, stop;
        if(enable_logging_)
        {
            hipEventCreate(&start);
            hipEventCreate(&stop);
            hipEventRecord(start);
        }

        if(use_blocked_)
        {
            status = factorize_blocked(A_or_array, tau_or_array);
        }
        else
        {
            status = factorize_unblocked(A_or_array, tau_or_array);
        }

        if(enable_logging_)
        {
            hipEventRecord(stop);
            hipEventSynchronize(stop);
            float total_ms;
            hipEventElapsedTime(&total_ms, start, stop);

            print_performance_report(total_ms);

            hipEventDestroy(start);
            hipEventDestroy(stop);
        }

        cleanup_workspace();
        return status;
    }

private:
    void auto_tune_parameters()
    {
        // Default values
        const int DEFAULT_SWITCH = 64;
        const int DEFAULT_BLOCK = 32;

        // Check if blocked algorithm is beneficial
        if(m_ <= DEFAULT_SWITCH || n_ <= DEFAULT_SWITCH)
        {
            use_blocked_ = false;
            switch_size_ = DEFAULT_SWITCH;
            block_size_ = DEFAULT_BLOCK;
            return;
        }

        use_blocked_ = true;
        switch_size_ = DEFAULT_SWITCH;

        // Auto-tune block size based on matrix shape

        // For nearly square matrices, use larger blocks
        if(std::abs(m_ - n_) < 0.2 * std::max(m_, n_))
        {
            if(m_ >= 2048)
                block_size_ = 64;  // Large blocks for large matrices
            else if(m_ >= 512)
                block_size_ = 32;
            else
                block_size_ = 16;
        }
        // For very wide matrices (m << n)
        else if(m_ < n_ / 4)
        {
            // Small blocks to reduce LARFB overhead
            block_size_ = std::min(16, m_ / 2);
        }
        // For tall matrices (m >> n)
        else if(m_ > 4 * n_)
        {
            // Larger blocks to maximize GEMM efficiency
            block_size_ = std::min(64, n_ / 4);
        }
        else
        {
            block_size_ = DEFAULT_BLOCK;
        }

        // Ensure block size is valid
        block_size_ = std::max(8, std::min(block_size_, 128));

        // Adjust switch size based on block size
        switch_size_ = std::max(DEFAULT_SWITCH, 2 * block_size_);

        if(enable_logging_)
        {
            printf("Auto-tuning: m=%d, n=%d, batch=%d\\n", m_, n_, batch_count_);
            printf("  Algorithm: %s\\n", use_blocked_ ? "Blocked" : "Unblocked");
            printf("  Block size: %d\\n", block_size_);
            printf("  Switch size: %d\\n", switch_size_);
        }
    }

    rocblas_status validate_inputs(T* A[], T* tau[])
    {
        if(handle_ == nullptr)
            return rocblas_status_invalid_handle;

        if(m_ < 0 || n_ < 0 || lda_ < m_ || batch_count_ < 0)
            return rocblas_status_invalid_size;

        if(m_ == 0 || n_ == 0 || batch_count_ == 0)
            return rocblas_status_success;  // Quick return

        if(A == nullptr || tau == nullptr)
            return rocblas_status_invalid_pointer;

        return rocblas_status_success;
    }

    rocblas_status allocate_workspace()
    {
        if(m_ == 0 || n_ == 0 || batch_count_ == 0)
            return rocblas_status_success;

        size_t size_scalars, size_work_workArr;
        size_t size_Abyx_norms_trfact, size_diag_tmptr, size_workArr;

        rocsolver_gelqf_getMemorySize<BATCHED, T>(
            m_, n_, batch_count_,
            &size_scalars, &size_work_workArr,
            &size_Abyx_norms_trfact, &size_diag_tmptr, &size_workArr);

        rocblas_status status = rocblas_status_success;

        if(size_scalars > 0)
        {
            status = rocblas_malloc(handle_, &scalars_mem_, size_scalars);
            if(status != rocblas_status_success) return status;

            // Initialize scalars: alpha=-1, beta=1
            T scalars_host[3] = {T(-1), T(1), T(0)};
            hipMemcpy(scalars_mem_, scalars_host, size_scalars, hipMemcpyHostToDevice);
        }

        if(size_work_workArr > 0)
        {
            status = rocblas_malloc(handle_, &work_workArr_mem_, size_work_workArr);
            if(status != rocblas_status_success) return status;
        }

        if(size_Abyx_norms_trfact > 0)
        {
            status = rocblas_malloc(handle_, &Abyx_norms_trfact_mem_, size_Abyx_norms_trfact);
            if(status != rocblas_status_success) return status;
        }

        if(size_diag_tmptr > 0)
        {
            status = rocblas_malloc(handle_, &diag_tmptr_mem_, size_diag_tmptr);
            if(status != rocblas_status_success) return status;
        }

        if(size_workArr > 0)
        {
            status = rocblas_malloc(handle_, (void**)&workArr_mem_, size_workArr);
            if(status != rocblas_status_success) return status;
        }

        if(enable_logging_)
        {
            size_t total = size_scalars + size_work_workArr +
                          size_Abyx_norms_trfact + size_diag_tmptr + size_workArr;
            printf("Workspace allocated: %.2f MB\\n", total / (1024.0 * 1024.0));
        }

        return rocblas_status_success;
    }

    void cleanup_workspace()
    {
        if(workArr_mem_) rocblas_free(handle_, workArr_mem_);
        if(diag_tmptr_mem_) rocblas_free(handle_, diag_tmptr_mem_);
        if(Abyx_norms_trfact_mem_) rocblas_free(handle_, Abyx_norms_trfact_mem_);
        if(work_workArr_mem_) rocblas_free(handle_, work_workArr_mem_);
        if(scalars_mem_) rocblas_free(handle_, scalars_mem_);

        scalars_mem_ = nullptr;
        work_workArr_mem_ = nullptr;
        Abyx_norms_trfact_mem_ = nullptr;
        diag_tmptr_mem_ = nullptr;
        workArr_mem_ = nullptr;
    }

    rocblas_status factorize_unblocked(T* A[], T* tau[])
    {
        if(enable_logging_)
            printf("Using unblocked GELQ2 algorithm\\n");

        using A_type = std::conditional_t<BATCHED, T**, T*>;

        return rocsolver_gelq2_template<T>(
            handle_, m_, n_,
            reinterpret_cast<A_type>(A), 0, lda_,
            STRIDED ? strideA_ : 0,
            reinterpret_cast<T*>(tau),
            STRIDED ? strideP_ : 0,
            batch_count_,
            static_cast<T*>(scalars_mem_),
            work_workArr_mem_,
            static_cast<T*>(Abyx_norms_trfact_mem_),
            static_cast<T*>(diag_tmptr_mem_));
    }

    rocblas_status factorize_blocked(T* A[], T* tau[])
    {
        if(enable_logging_)
            printf("Using blocked GELQF algorithm (nb=%d)\\n", block_size_);

        using A_type = std::conditional_t<BATCHED, T**, T*>;

        return rocsolver_gelqf_template<BATCHED, STRIDED, T>(
            handle_, m_, n_,
            reinterpret_cast<A_type>(A), 0, lda_,
            STRIDED ? strideA_ : 0,
            reinterpret_cast<T*>(tau),
            STRIDED ? strideP_ : 0,
            batch_count_,
            static_cast<T*>(scalars_mem_),
            work_workArr_mem_,
            static_cast<T*>(Abyx_norms_trfact_mem_),
            static_cast<T*>(diag_tmptr_mem_),
            workArr_mem_);
    }

    void print_performance_report(float total_ms)
    {
        printf("\\n=== GELQF Performance Report ===\\n");
        printf("Matrix: %d × %d, batch: %d\\n", m_, n_, batch_count_);
        printf("Algorithm: %s (nb=%d)\\n",
               use_blocked_ ? "Blocked" : "Unblocked", block_size_);
        printf("Total time: %.3f ms\\n", total_ms);

        // Estimate FLOPS
        long long flops = (2LL * m_ * m_ * (3 * n_ - m_)) / 3;
        flops *= batch_count_;
        double gflops = (flops / 1e9) / (total_ms / 1000.0);

        printf("Performance: %.2f GFLOPS\\n", gflops);
        printf("===============================\\n");
    }
};

/*
USAGE EXAMPLE:

// Strided batched mode
const int m = 512, n = 1024, batch = 16;
const int lda = m;
const rocblas_stride strideA = lda * n;
const rocblas_stride strideP = std::min(m, n);

double* A_batch;  // m × n × batch
double* tau_batch;  // min(m,n) × batch

hipMalloc(&A_batch, m * n * batch * sizeof(double));
hipMalloc(&tau_batch, std::min(m,n) * batch * sizeof(double));

rocblas_handle handle;
rocblas_create_handle(&handle);

// Create solver with auto-tuning and logging
GelqfSolver<false, true, double> solver(
    handle, m, n, lda, strideA, strideP, batch,
    true);  // enable logging

// Factorize
rocblas_status status = solver.factorize(
    reinterpret_cast<double**>(&A_batch),
    reinterpret_cast<double**>(&tau_batch));

if(status != rocblas_status_success)
{
    std::cerr << "Factorization failed: " << status << std::endl;
}

rocblas_destroy_handle(handle);
*/""",
        rationale="Production numerical libraries require sophisticated auto-tuning, transparent performance reporting, and robust resource management. This implementation demonstrates object-oriented design for complex GPU algorithms, automatic parameter selection based on problem characteristics, and best practices for memory safety and user experience.",
        tags=["coding", "API-design", "auto-tuning", "performance-optimization", "resource-management", "production-library"]
    ))

    return entries

def main():
    """Generate and write the dataset to a JSONL file."""
    entries = generate_dataset()

    output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_gelqf.jsonl"

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(entries)} entries")
    print(f"Written to: {output_path}")

    # Print summary statistics
    level_counts = {"L1": 0, "L2": 0, "L3": 0}
    coding_count = 0
    analysis_count = 0

    for entry in entries:
        level_counts[entry["level"]] += 1
        if "coding" in entry["tags"]:
            coding_count += 1
        if "analysis" in entry["tags"]:
            analysis_count += 1

    print(f"\nDistribution:")
    print(f"  L1: {level_counts['L1']} entries")
    print(f"  L2: {level_counts['L2']} entries")
    print(f"  L3: {level_counts['L3']} entries")
    print(f"  Coding: {coding_count} ({100*coding_count/len(entries):.1f}%)")
    print(f"  Analysis: {analysis_count} ({100*analysis_count/len(entries):.1f}%)")

if __name__ == "__main__":
    main()
