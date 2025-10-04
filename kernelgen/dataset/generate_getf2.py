#!/usr/bin/env python3
"""
Generator script for roclapack_getf2 SFT dataset.

GETF2 computes the LU factorization of a general m×n matrix using partial pivoting
(with pivoting) or without pivoting. It's the unblocked version used as a building
block in blocked GETRF.
"""

import json
from datetime import datetime

def create_entry(id_num, level, interface, instruction, context_text, code_blocks,
                answer, rationale, tags):
    """Create a single dataset entry."""
    return {
        "id": f"2025-10-03T07:00:00.{id_num:03d}",
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
    """Generate all dataset entries for roclapack_getf2."""
    entries = []

    # ========== L1 ENTRIES ==========

    # L1-1: IAMAX Reduction Kernel (Analysis)
    entries.append(create_entry(
        id_num=1,
        level="L1",
        interface="roclapack_getf2",
        instruction="Analyze the getf2_iamax kernel implementation. Explain how the parallel reduction finds the maximum absolute value element and its index, and why __shared__ memory is used for both sval and sidx arrays.",
        context_text="The iamax operation (index of absolute maximum) is critical for pivot selection in LU factorization. Efficient parallel reduction is essential for performance on GPUs.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """#define IAMAX_THDS 1024

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(IAMAX_THDS) getf2_iamax(const I m,
                                                                U xx,
                                                                const rocblas_stride shiftx,
                                                                const I incx,
                                                                const rocblas_stride stridex,
                                                                I* pivotidx)
{
    using S = decltype(std::real(T{}));

    // batch instance
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    T* x = load_ptr_batch<T>(xx, bid, shiftx, stridex);

    // shared memory setup
    __shared__ S sval[IAMAX_THDS];
    __shared__ I sidx[IAMAX_THDS];

    iamax<IAMAX_THDS>(tid, m, x, incx, sval, sidx);

    // write results back to global memory
    // (after the reduction, the maximum of the elements is in sval[0] and sidx[0])
    if(tid == 0)
        pivotidx[bid] = sidx[0];
}"""
            }
        ],
        answer="""The getf2_iamax kernel performs a parallel reduction to find the index of the maximum absolute value element:

**Algorithm Structure:**

1. **Thread-Level Loading**: Each thread computes partial results for its assigned elements
   - Thread `tid` processes elements at indices: tid, tid + 1024, tid + 2048, ...
   - Stride pattern ensures coalesced memory access

2. **Parallel Reduction in Shared Memory**:
   - Each thread stores its local maximum value in `sval[tid]` and index in `sidx[tid]`
   - Reduction proceeds in log2(1024) = 10 steps using tree-based reduction
   - At each step, threads compare pairs and keep the element with larger absolute value
   - If values are equal, the smaller index is kept (stable reduction)

3. **Final Result**: After reduction, `sval[0]` contains the maximum absolute value and `sidx[0]` contains its index

**Why Shared Memory?**

1. **Speed**: Shared memory is ~100x faster than global memory (on-chip vs off-chip)
2. **Synchronization**: `__syncthreads()` ensures all threads see consistent data during reduction
3. **Parallel Access**: Multiple threads can read different locations simultaneously without bank conflicts (when properly indexed)
4. **Dual Tracking**: We need to track both value AND index during reduction - shared memory provides fast storage for both arrays

**Performance Characteristics:**
- 1024 threads per block maximizes occupancy on most GPUs
- Tree reduction completes in O(log N) steps vs O(N) for sequential
- Memory access is coalesced for both loads and stores
- Typical use: Finding pivot row in column during LU factorization""",
        rationale="Understanding parallel reduction patterns is fundamental to GPU programming. The iamax operation demonstrates key concepts: shared memory usage, tree reduction, coalesced memory access, and maintaining auxiliary data (indices) during reductions.",
        tags=["analysis", "reduction", "shared-memory", "pivot-selection", "parallel-algorithm"]
    ))

    # L1-2: Check Singularity and Pivot Kernel (Coding)
    entries.append(create_entry(
        id_num=2,
        level="L1",
        interface="roclapack_getf2",
        instruction="Implement a __device__ function that computes the reciprocal of a diagonal element for LU factorization, handling the singularity case. The function should check if the element is zero (singular), update an info flag if needed, and return either 1.0 (for zero) or 1.0/element (for non-zero).",
        context_text="During LU factorization, each diagonal element is used to scale the submatrix below it. Detecting singularities (zero diagonal elements) is critical for numerical stability and error reporting.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void getf2_npvt_check_singularity(const I j,
                                                   U AA,
                                                   const rocblas_stride shiftA,
                                                   const I inca,
                                                   const I lda,
                                                   const rocblas_stride strideA,
                                                   T* pivot_val,
                                                   INFO* info,
                                                   const I offset)
{
    using S = decltype(std::real(T{}));

    const I id = hipBlockIdx_y;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);

    // update info (check singularity)
    if(A[j * inca + j * lda] == 0)
    {
        pivot_val[id] = 1;
        if(info[id] == 0)
            info[id] = static_cast<INFO>(j + 1 + offset); // use Fortran 1-based indexing
    }
    else
        pivot_val[id] = S(1) / A[j * inca + j * lda];
}"""
            }
        ],
        answer="""template <typename T, typename INFO>
__device__ T compute_pivot_reciprocal(
    const T diagonal_element,
    const int pivot_index,
    INFO* info,
    const int offset)
{
    // Use real type for reciprocal (handles complex types)
    using RealType = decltype(std::real(T{}));

    // Check for singularity (exact zero comparison)
    if(diagonal_element == T(0))
    {
        // Mark singularity: only update info if it's the first singular pivot
        // (info[0] == 0 means no singularity detected yet)
        // Use atomic compare-and-swap to handle race conditions across threads
        INFO expected = 0;
        INFO new_val = static_cast<INFO>(pivot_index + 1 + offset);

        // Atomically update info only if it's still zero
        atomicCAS(info, expected, new_val);

        // Return 1.0 to avoid division by zero in subsequent operations
        // (the factorization is singular, but we continue for error reporting)
        return RealType(1);
    }
    else
    {
        // Compute reciprocal for non-singular case
        // This will be used to scale the submatrix: A[i,j] *= (1/A[j,j])
        return RealType(1) / diagonal_element;
    }
}

// Usage example in kernel:
template <typename T, typename INFO>
__global__ void example_usage_kernel(
    T* matrix,
    const int lda,
    const int j,  // current pivot column
    INFO* info)
{
    const int batch_id = blockIdx.y;

    if(threadIdx.x == 0)  // Only one thread per batch element
    {
        T diagonal = matrix[batch_id * lda * lda + j * lda + j];

        T reciprocal = compute_pivot_reciprocal<T, INFO>(
            diagonal,
            j,          // pivot index (0-based)
            &info[batch_id],
            0);         // offset for nested calls

        // Store reciprocal for later use in GER update
        // ...
    }
}

/*
Key Design Decisions:

1. **Exact Zero Comparison**:
   We use `diagonal_element == T(0)` rather than checking if abs(diagonal) < epsilon.
   LAPACK spec requires exact zero check for singularity detection.

2. **Fortran 1-Based Indexing**:
   Info reports pivot_index + 1 to match LAPACK convention (1-based indexing).

3. **Atomic Update**:
   atomicCAS ensures only the FIRST singularity is recorded in info.
   Subsequent singular pivots don't overwrite the first occurrence.

4. **Return Value**:
   - Non-singular: returns 1/diagonal for scaling operations
   - Singular: returns 1.0 to prevent NaN propagation (computation continues)

5. **Real Type Reciprocal**:
   For complex types, the reciprocal is computed component-wise using real division.
*/""",
        rationale="Singularity detection is a critical component of numerical linear algebra algorithms. This function demonstrates proper error handling, atomic operations for race-free updates, and the trade-off between stopping computation vs. continuing for complete error reporting.",
        tags=["coding", "singularity-detection", "device-function", "numerical-stability", "atomic-operations"]
    ))

    # L1-3: Permutation Initialization (Analysis)
    entries.append(create_entry(
        id_num=3,
        level="L1",
        interface="roclapack_getf2",
        instruction="The getf2_permut_init kernel initializes a permutation array. Explain what this permutation array tracks, why it's needed for parallel row permutations in GETRF, and how it's updated during factorization.",
        context_text="LU factorization with pivoting involves row interchanges. Tracking these permutations is essential for applying the factorization correctly and for parallel execution.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """/** this kernel initializes the permutation array
    which is instrumental for parallel row permutations in GETRF **/
template <typename T, typename I>
ROCSOLVER_KERNEL void getf2_permut_init(const I m, I* permutA, const rocblas_stride stridePI)
{
    I id = hipBlockIdx_y;
    I i = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

    // batch instance
    I* permut = permutA + id * stridePI;

    // initialize
    if(i < m)
        permut[i] = i;
}"""
            }
        ],
        answer="""**What the Permutation Array Tracks:**

The permutation array `permut[i]` stores the current row index that originally occupied position `i` in the matrix. Initially `permut[i] = i` (identity permutation).

After k row swaps, `permut[i]` tells us: "Row i in the current matrix originally came from row permut[i] in the input matrix."

**Why It's Needed for Parallel GETRF:**

1. **Sequential Dependency Problem**:
   - Traditional LAPACK applies row swaps sequentially: swap rows, update matrix, repeat
   - On GPUs, we want to defer all row swaps and apply them in parallel at the end

2. **Tracking Composition of Swaps**:
   - During panel factorization, we determine multiple row swaps: (i₀ ↔ j₀), (i₁ ↔ j₁), ...
   - These swaps don't commute: order matters!
   - The permut array tracks the cumulative effect of all swaps

3. **Parallel Row Permutation**:
   - After computing all pivots in a panel, we apply all row swaps simultaneously
   - Thread i reads from row permut[i] and writes to row i
   - This is safe because each row is read from exactly once and written to exactly once

**How It's Updated During Factorization:**

Example with 4×4 matrix, swapping rows 0↔2, then 1↔3:

```
Initial: permut = [0, 1, 2, 3]

After swap 0↔2:
  swap(permut[0], permut[2])
  permut = [2, 1, 0, 3]
  Meaning: row 0 now contains original row 2

After swap 1↔3:
  swap(permut[1], permut[3])
  permut = [2, 3, 0, 1]
  Meaning: row 0 has orig row 2, row 1 has orig row 3, etc.
```

**Update Location in Code:**

In the `getf2_check_singularity` kernel:
```cpp
if(permut_idx)
{
    I* permut = permut_idx + id * stridePI;
    if(exch != j)  // If pivot row != current row
        swap(permut[j], permut[exch]);
}
```

This incrementally updates permut to track the composition of all row swaps performed so far.

**Final Use:**

The permut array is used in GETRF's `laswp` (row permutation) kernel to apply all accumulated swaps in parallel after panel factorization completes.""",
        rationale="Understanding permutation tracking is crucial for parallel implementations of algorithms that traditionally require sequential row operations. This pattern (defer operations, track dependencies, execute in parallel) is widely applicable in GPU computing.",
        tags=["analysis", "permutation", "parallel-algorithm", "LU-factorization", "data-dependency"]
    ))

    # ========== L2 ENTRIES ==========

    # L2-1: Kernel Selection Heuristics (Analysis)
    entries.append(create_entry(
        id_num=4,
        level="L2",
        interface="roclapack_getf2",
        instruction="Analyze the select_spkernel function that chooses between specialized kernels (small kernel, panel kernel) vs normal code path. What factors drive these heuristics? Why might a panel kernel outperform the normal path for certain dimensions?",
        context_text="Performance optimization often requires different implementations for different problem sizes. The select_spkernel function encodes empirically-determined heuristics for choosing the best kernel variant.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """template <bool ISBATCHED, typename T, typename I, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
int select_spkernel(const I m, const I n, const I inca, const bool pivot)
{
    int ker = 0;

    if(m > GETF2_SPKER_MAX_M || n > GETF2_SPKER_MAX_N || inca != 1)
        return ker;

    if(ISBATCHED)
    {
        // Batch pivoting case (real precisions)
        if(pivot)
        {
            if(n <= 28)
            {
                ker = (m <= 140) ? 1 : 2;
            }
            else if(n <= 44)
            {
                ker = (m <= 256) ? 1 : 2;
            }
            else if((n <= 52 && (m <= 392 || m > 504)) || (n > 52 && n <= 60 && (m <= 392 || m > 624))
                    || (n > 60 && n <= 68 && (m <= 296 || m > 864)))
            {
                ker = (m <= 256) ? 1 : 2;
            }
            // ... more conditions
        }
    }

    if(ker == 1 && (m > GETF2_SSKER_MAX_M || n > GETF2_SSKER_MAX_N))
    {
        ker = 2;
    }

    return ker;
}"""
            }
        ],
        answer="""**Kernel Selection Strategy:**

Returns:
- **0**: Use normal iterative path (iamax → pivot → ger for each column)
- **1**: Use small specialized kernel (entire factorization in one fused kernel)
- **2**: Use panel kernel (factorize multiple columns together)

**Factors Driving Heuristics:**

1. **Problem Size Constraints**:
   - Maximum limits: GETF2_SPKER_MAX_M, GETF2_SPKER_MAX_N
   - Small kernel requires matrix to fit in shared memory (limited to ~256×256)
   - Panel kernel has relaxed constraints but still bounded

2. **Memory Access Patterns**:
   - Small matrices: Data reuse is high → fused kernel wins (avoid repeated global memory access)
   - Large matrices: Data doesn't fit in cache → normal path is fine

3. **Kernel Launch Overhead**:
   - Normal path: 3n kernel launches (iamax, pivot, ger for each of n columns)
   - Small kernel: 1 launch → huge savings for small n
   - Panel kernel: ~n/panel_width launches → moderate savings

4. **Occupancy and Resource Usage**:
   - Small kernel uses more shared memory and registers → lower occupancy
   - For very small m (rows), the small kernel's lower occupancy is acceptable
   - For larger m, panel kernel balances resource use with parallelism

**Why Panel Kernel Outperforms Normal Path:**

1. **Reduced Launch Overhead**:
   - Factorizes 4-8 columns together instead of one at a time
   - For n=64, panel width 8: 8 launches vs 192 launches (iamax+pivot+ger × 64)

2. **Better Cache Utilization**:
   - Panel stays in L2 cache across multiple column operations
   - Normal path may evict data between separate iamax/pivot/ger calls

3. **Fused Operations**:
   - Panel kernel fuses iamax+pivot within panel
   - Avoids intermediate writes to global memory

4. **Batch Processing**:
   - Panel kernel can process multiple batch elements' panels simultaneously
   - Better utilization of GPU compute units

**Observed Patterns in Heuristics:**

- **Tall matrices (m >> n)**: Prefer panel kernel (ker=2)
  - Many rows to process in GER update → amortize panel setup cost

- **Square small matrices (m ≈ n ≤ 256)**: Prefer small kernel (ker=1)
  - Entire matrix fits in shared memory → maximum fusion benefit

- **Specific gaps** (e.g., m > 504 and m <= 624 for certain n):
  - Sweet spots where panel kernel's resource usage perfectly matches GPU capabilities
  - These are found empirically through autotuning

**Non-Contiguous Data (inca != 1)**:
- Forces normal path (ker=0)
- Specialized kernels assume column-major contiguous layout for optimal memory access""",
        rationale="Kernel selection heuristics encode domain expertise and empirical performance data. Understanding the reasoning behind these decisions helps in designing performance-portable libraries and provides insight into GPU architecture constraints.",
        tags=["analysis", "performance-optimization", "kernel-selection", "heuristics", "autotuning"]
    ))

    # L2-2: Fused Panel Kernel Design (Coding)
    entries.append(create_entry(
        id_num=5,
        level="L2",
        interface="roclapack_getf2",
        instruction="Design a fused panel kernel that factorizes 8 columns simultaneously without separate iamax/pivot/ger kernel launches. Provide the kernel signature, shared memory layout, and high-level algorithm structure showing how to interleave pivot selection, row swaps, and rank-1 updates.",
        context_text="Fusing multiple steps of LU factorization into a single kernel reduces launch overhead and improves data locality. Panel kernels process several columns together while maintaining numerical correctness.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """// Normal path: for each column j
//   1. iamax: find pivot in column j
//   2. pivot: swap rows and update ipiv
//   3. ger: scale column and update trailing matrix

// Panel kernel fuses these operations for multiple columns"""
            }
        ],
        answer="""template <typename T, int PANEL_WIDTH = 8, int MAX_M = 512>
__global__ void getf2_panel_fused_kernel(
    const int m,               // Number of rows
    const int n,               // Total columns (multiple of PANEL_WIDTH)
    const int panel_start,     // Starting column of this panel
    T* __restrict__ A,         // Matrix in column-major format
    const int lda,
    int* __restrict__ ipiv,    // Pivot indices output
    int* __restrict__ info,    // Singularity info
    int* __restrict__ permut)  // Permutation tracking array
{
    // ===== Shared Memory Layout =====
    __shared__ T smem_panel[MAX_M * PANEL_WIDTH];  // Panel data (m × 8)
    __shared__ int smem_pivot_idx[PANEL_WIDTH];     // Pivot indices for panel
    __shared__ T smem_pivot_val[PANEL_WIDTH];       // Pivot reciprocals
    __shared__ int smem_local_perm[MAX_M];          // Local permutation for this panel

    const int tid = threadIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int batch_id = blockIdx.y;
    const int nthreads = blockDim.x;

    // Offset to batch element
    A += batch_id * lda * n;
    ipiv += batch_id * n;
    info += batch_id;
    permut += batch_id * m;

    // ===== Load Panel into Shared Memory =====
    // Threads cooperatively load panel columns [panel_start : panel_start+PANEL_WIDTH)
    const int panel_cols = min(PANEL_WIDTH, n - panel_start);

    for(int col = 0; col < panel_cols; col++)
    {
        for(int row = tid; row < m; row += nthreads)
        {
            smem_panel[col * MAX_M + row] = A[(panel_start + col) * lda + row];
        }
    }

    // Initialize local permutation as identity
    for(int row = tid; row < m; row += nthreads)
    {
        smem_local_perm[row] = permut[row];  // Start with current global permutation
    }

    __syncthreads();

    // ===== Factorize Panel Column-by-Column =====
    for(int j = 0; j < panel_cols; j++)
    {
        const int global_col = panel_start + j;
        const int search_start = j;  // Only search below diagonal in panel

        // ----- STEP 1: Parallel IAMAX Reduction -----
        // Find index of maximum absolute value in column j, rows [j:m)

        __shared__ T reduction_val[256];   // Assuming blockDim.x <= 256
        __shared__ int reduction_idx[256];

        T local_max_val = 0;
        int local_max_idx = j;

        // Each thread scans its portion
        for(int row = j + tid; row < m; row += nthreads)
        {
            T val = smem_panel[j * MAX_M + row];
            T abs_val = abs(val);  // Or use custom abs for complex types

            if(abs_val > local_max_val)
            {
                local_max_val = abs_val;
                local_max_idx = row;
            }
        }

        reduction_val[tid] = local_max_val;
        reduction_idx[tid] = local_max_idx;
        __syncthreads();

        // Tree reduction to find global maximum
        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
            {
                if(reduction_val[tid + stride] > reduction_val[tid])
                {
                    reduction_val[tid] = reduction_val[tid + stride];
                    reduction_idx[tid] = reduction_idx[tid + stride];
                }
            }
            __syncthreads();
        }

        // Broadcast pivot info
        if(tid == 0)
        {
            smem_pivot_idx[j] = reduction_idx[0];
            ipiv[global_col] = reduction_idx[0] + 1;  // Fortran 1-based
        }
        __syncthreads();

        int pivot_row = smem_pivot_idx[j];

        // ----- STEP 2: Row Swap -----
        // Swap rows j and pivot_row across entire panel (all columns)
        if(pivot_row != j)
        {
            for(int col = tid; col < panel_cols; col += nthreads)
            {
                swap(smem_panel[col * MAX_M + j],
                     smem_panel[col * MAX_M + pivot_row]);
            }

            // Update local permutation
            if(tid == 0)
            {
                swap(smem_local_perm[j], smem_local_perm[pivot_row]);
            }
        }
        __syncthreads();

        // ----- STEP 3: Check Singularity and Compute Reciprocal -----
        if(tid == 0)
        {
            T diag = smem_panel[j * MAX_M + j];
            if(diag == T(0))
            {
                smem_pivot_val[j] = T(1);
                if(info[0] == 0)  // First singularity
                {
                    info[0] = global_col + 1;  // Fortran 1-based
                }
            }
            else
            {
                smem_pivot_val[j] = T(1) / diag;
            }
        }
        __syncthreads();

        T pivot_recip = smem_pivot_val[j];

        // ----- STEP 4: Scale Column Below Diagonal -----
        for(int row = j + 1 + tid; row < m; row += nthreads)
        {
            smem_panel[j * MAX_M + row] *= pivot_recip;
        }
        __syncthreads();

        // ----- STEP 5: Rank-1 Update of Trailing Panel -----
        // Update columns [j+1 : panel_cols) using scaled column j
        // A[j+1:m, j+1:panel_cols] -= A[j+1:m, j] * A[j, j+1:panel_cols]

        for(int col = j + 1; col < panel_cols; col++)
        {
            T row_j_val = smem_panel[col * MAX_M + j];  // Broadcast value from row j

            for(int row = j + 1 + tid; row < m; row += nthreads)
            {
                smem_panel[col * MAX_M + row] -=
                    smem_panel[j * MAX_M + row] * row_j_val;
            }
        }
        __syncthreads();
    }

    // ===== Write Panel Back to Global Memory =====
    for(int col = 0; col < panel_cols; col++)
    {
        for(int row = tid; row < m; row += nthreads)
        {
            A[(panel_start + col) * lda + row] = smem_panel[col * MAX_M + row];
        }
    }

    // Update global permutation
    for(int row = tid; row < m; row += nthreads)
    {
        permut[row] = smem_local_perm[row];
    }
}

/*
ALGORITHM SUMMARY:

For each column j in panel:
  1. IAMAX: Parallel reduction to find pivot (max abs value in rows [j:m))
  2. PIVOT: Swap rows j ↔ pivot_row across all panel columns
  3. SINGULARITY CHECK: Test diagonal, compute reciprocal
  4. SCALE: Multiply column j below diagonal by reciprocal
  5. GER UPDATE: Rank-1 update of trailing panel submatrix

KEY OPTIMIZATIONS:

1. **Data Locality**: Entire panel stays in shared memory (m×8 elements)
2. **Fused Operations**: No intermediate global memory writes between steps
3. **Batch Parallelism**: blockIdx.y indexes independent batch elements
4. **Reduced Launches**: 1 launch per panel vs 3n launches for normal path

RESOURCE REQUIREMENTS:

- Shared memory: (m×8 + m + 8 + 8 + 256 + 256) × sizeof(T) + m×sizeof(int)
  For m=512, T=double: ~35 KB (fits in 48 KB shared memory limit)
- Threads: 128-256 recommended (tunable)
- Registers: ~40-60 per thread (high but acceptable)

LIMITATIONS:

- Maximum m limited by shared memory (typically m ≤ 512-1024)
- Panel width fixed at compile time (template parameter)
- Assumes column-major layout with lda ≥ m
*/""",
        rationale="Fused panel kernels represent an advanced GPU optimization: trading off generality and resource usage for significant performance gains on common problem sizes. This design demonstrates kernel fusion, shared memory management, parallel reductions, and careful synchronization.",
        tags=["coding", "kernel-fusion", "panel-factorization", "shared-memory", "LU-factorization", "optimization"]
    ))

    # ========== L3 ENTRIES ==========

    # L3-1: Complete GETF2 Algorithm Flow (Analysis)
    entries.append(create_entry(
        id_num=6,
        level="L3",
        interface="roclapack_getf2",
        instruction="Trace the complete execution of GETF2 for a 256×128 matrix with pivoting on GPU. Detail the kernel launches, workspace allocation, data flow between kernels, and how the final ipiv array and info status are produced. Include estimates of total kernel launches.",
        context_text="Understanding the complete algorithm flow from API call to final result is essential for debugging, performance analysis, and optimization. GETF2 involves intricate orchestration of multiple kernel types.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """// Main template function orchestrates:
// - Workspace allocation
// - Permutation initialization
// - For each column j:
//   - iamax to find pivot
//   - pivot/singularity check/row swap
//   - GER to update trailing matrix
// - Return status"""
            }
        ],
        answer="""**Complete GETF2 Execution Trace for 256×128 Matrix (m=256, n=128) with Pivoting:**

**PHASE 1: Setup and Workspace Allocation**

1. **API Entry**: `rocsolver_getf2(handle, 256, 128, A, lda=256, ipiv, info, batch_count=1)`

2. **Memory Size Query**:
   - Call `rocsolver_getf2_getMemorySize(256, 128, pivot=true, batch_count=1)`
   - Returns: size_scalars = 24 bytes (3×double), size_pivotval = 8 bytes, size_pivotidx = 8 bytes

3. **Workspace Allocation**:
   - Allocate device memory for: scalars (alpha, beta for BLAS), pivotval, pivotidx
   - Total: ~40 bytes

4. **Permutation Initialization** (if using parallel row swaps):
   - Launch `getf2_permut_init<<< (256+255)/256, batch_count >>>`
   - 1 block × 1 batch = 1 kernel launch
   - Initializes permut[i] = i for i ∈ [0, 256)

**PHASE 2: Column-by-Column Factorization (n=128 iterations)**

For each column j = 0 to 127:

  **ITERATION j:**

  **(a) Find Pivot:**
  - Launch `getf2_iamax<<< (m-j + IAMAX_THDS-1)/IAMAX_THDS, batch_count >>>`
  - Search column j, rows [j : 256) for maximum absolute value
  - Threads: 1024, Blocks: 1 (since 256-j < 1024 for all j)
  - Output: pivotidx[batch] = index of maximum element
  - **Kernel count: 128 launches**

  **(b) Check Singularity, Pivot, and Swap:**
  - Launch `getf2_check_singularity<<< blocks, batch_count >>>`
    - Blocks: getf2_get_checksingularity_blksize(n=128) = 128 (since 1024 ≤ 128 < 2048)
    - Threads per block: 128
  - Operations:
    - Read pivot index from pivotidx[batch]
    - Swap row j with pivot row across columns [j : 128)
    - Update ipiv[j] = pivot_row + 1 (Fortran indexing)
    - Check if A[j,j] == 0:
      - If singular: set info[batch] = j+1, pivotval[batch] = 1
      - If non-singular: pivotval[batch] = 1/A[j,j]
    - Update permut array: swap(permut[j], permut[pivot_row])
  - **Kernel count: 128 launches**

  **(c) Scale Column and Update Trailing Matrix (GER):**
  - Launch `rocsolver_ger<<< blocks, batch_count >>>`
    - Computes: A[j+1:256, j+1:128] -= A[j+1:256, j] * A[j, j+1:128]
    - Block dimensions from `getf2_get_ger_blksize(m=256, n=128)`:
      - For n=128 (56 < n ≤ 88): dimy=16, dimx=64
      - Grid: ((128-j-1+63)/64) × ((256-j-1+15)/16) blocks
    - First iteration (j=0): ~2 × 16 = 32 blocks
    - Last iteration (j=127): 0 blocks (no trailing matrix)
  - **Kernel count: 128 launches**

**PHASE 3: Finalization**

1. **Workspace Deallocation**:
   - Free scalars, pivotval, pivotidx device memory

2. **Output Validation**:
   - ipiv[0:127] contains pivot indices (Fortran 1-based)
   - info[0] = 0 (success) or j+1 (first singular pivot at column j)

**TOTAL KERNEL LAUNCH COUNT:**

- Permutation init: 1
- Column iterations: 128 × (iamax + check_singularity + ger) = 128 × 3 = **384 kernel launches**
- **Grand total: 385 kernel launches**

**DATA FLOW:**

```
Input: A (256×128), uninitialized ipiv, uninitialized info

For j = 0 to 127:
  iamax(A[j:256, j]) → pivotidx
  check_singularity(A, j, pivotidx) → swap rows, update ipiv[j], pivotval, info
  ger(A[j+1:256, j], A[j, j+1:128], pivotval) → A[j+1:256, j+1:128]

Output:
  - A: Lower triangle contains L (with unit diagonal), Upper triangle contains U
  - ipiv[j]: Row index that was swapped to position j (1-based)
  - info: 0 (success) or index of first singular pivot
```

**MEMORY TRAFFIC (Approximate):**

- Matrix reads/writes: Each column updated ~128 times → O(m × n²) = O(256 × 128²) ≈ 4M elements
- With double precision: ~32 MB of memory traffic
- Actual traffic reduced by L2 cache hits in GER updates

**PERFORMANCE CHARACTERISTICS:**

- Kernel launch overhead: 385 launches × ~10 μs = ~3.85 ms (significant!)
- Compute time: ~100-500 μs depending on GPU
- Memory time: ~50-200 μs
- **Conclusion**: Launch overhead dominates → specialized fused kernels (small/panel) provide huge speedups for this size

**OPTIMIZATION OPPORTUNITY:**

Using panel kernel with width=8:
- Launches: 1 + (128/8) × 2 = 33 launches (11× reduction!)
- Much better cache locality
- Expected speedup: 3-5× for this problem size""",
        rationale="Tracing complete execution flows builds intuition for performance bottlenecks and optimization opportunities. This analysis reveals why kernel fusion is critical for LU factorization on GPUs and quantifies the launch overhead problem.",
        tags=["analysis", "algorithm-trace", "data-flow", "performance-analysis", "kernel-launch-overhead"]
    ))

    # L3-2: Batched GETF2 with Error Handling (Coding)
    entries.append(create_entry(
        id_num=7,
        level="L3",
        interface="roclapack_getf2",
        instruction="Design a complete host-side wrapper function for batched GETF2 that handles workspace allocation, error checking, kernel selection (normal vs specialized), and proper cleanup. Include detailed error handling for all failure modes and comments explaining resource management.",
        context_text="Production-quality library functions require robust error handling, automatic workspace management, and transparent kernel selection. This wrapper demonstrates best practices for GPU library development.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_getf2.hpp",
                "language": "cpp",
                "content": """// Complete wrapper must handle:
// - Argument validation
// - Kernel selection heuristics
// - Workspace size computation and allocation
// - Kernel dispatch
// - Error propagation
// - Cleanup on all exit paths"""
            }
        ],
        answer="""template <bool BATCHED, typename T>
rocblas_status rocsolver_getf2_batched_wrapper(
    rocblas_handle handle,
    const int m,
    const int n,
    T* A[],                    // Array of pointers to matrices
    const int lda,
    int* ipiv[],              // Array of pointers to pivot arrays
    int* info[],              // Array of pointers to info values
    const int batch_count,
    const bool with_pivot = true)
{
    // ===== ARGUMENT VALIDATION =====

    // Check handle
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    // Check dimensions
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // Quick return for zero-sized problems
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    // Check pointers (A and info are always required; ipiv only if with_pivot)
    if(A == nullptr || info == nullptr || (with_pivot && ipiv == nullptr))
        return rocblas_status_invalid_pointer;

    // Additional validation: check that array elements are valid
    // (In production, this might be done inside kernels to avoid host-device sync)

    // ===== KERNEL SELECTION =====

    int kernel_type = 0;  // 0 = normal, 1 = small, 2 = panel

#ifdef OPTIMAL
    kernel_type = select_spkernel<BATCHED, T>(m, n, /*inca=*/1, with_pivot);
#endif

    // Log kernel selection for debugging
    ROCSOLVER_ENTER("getf2_wrapper", "m:", m, "n:", n, "lda:", lda,
                    "batch:", batch_count, "kernel:", kernel_type);

    // ===== WORKSPACE ALLOCATION =====

    size_t size_scalars = 0;
    size_t size_pivotval = 0;
    size_t size_pivotidx = 0;

    // Query workspace sizes based on kernel type
    if(kernel_type == 0)  // Normal iterative path
    {
        rocsolver_getf2_getMemorySize<BATCHED, T>(
            m, n, with_pivot, batch_count,
            &size_scalars, &size_pivotval, &size_pivotidx);
    }
    else
    {
        // Specialized kernels don't need workspace
        size_scalars = 0;
        size_pivotval = 0;
        size_pivotidx = 0;
    }

    // Allocate workspace
    void* scalars_mem = nullptr;
    void* pivotval_mem = nullptr;
    void* pivotidx_mem = nullptr;

    rocblas_status alloc_status = rocblas_status_success;

    if(size_scalars > 0)
    {
        alloc_status = rocblas_malloc(handle, &scalars_mem, size_scalars);
        if(alloc_status != rocblas_status_success)
            goto cleanup;
    }

    if(size_pivotval > 0)
    {
        alloc_status = rocblas_malloc(handle, &pivotval_mem, size_pivotval);
        if(alloc_status != rocblas_status_success)
            goto cleanup;
    }

    if(size_pivotidx > 0)
    {
        alloc_status = rocblas_malloc(handle, &pivotidx_mem, size_pivotidx);
        if(alloc_status != rocblas_status_success)
            goto cleanup;
    }

    // ===== EXECUTE FACTORIZATION =====

    {
        rocblas_status compute_status;

        switch(kernel_type)
        {
        case 1:  // Small specialized kernel
            compute_status = getf2_run_small<T>(
                handle, m, n,
                A, /*shiftA=*/0, /*inca=*/1, lda, /*strideA=*/0,
                ipiv, /*shiftP=*/0, /*strideP=*/n,
                info,
                batch_count,
                with_pivot);
            break;

        case 2:  // Panel kernel
            compute_status = getf2_run_panel<T>(
                handle, m, n,
                A, /*shiftA=*/0, /*inca=*/1, lda, /*strideA=*/0,
                ipiv, /*shiftP=*/0, /*strideP=*/n,
                info,
                batch_count,
                with_pivot);
            break;

        default:  // Normal iterative path
            compute_status = rocsolver_getf2_template<BATCHED, T>(
                handle, m, n,
                A, /*shiftA=*/0, /*inca=*/1, lda, /*strideA=*/0,
                ipiv, /*shiftP=*/0, /*strideP=*/n,
                info,
                batch_count,
                static_cast<T*>(scalars_mem),
                static_cast<T*>(pivotval_mem),
                static_cast<int*>(pivotidx_mem),
                with_pivot,
                /*offset=*/0,
                /*permut_idx=*/nullptr,
                /*stridePI=*/0);
            break;
        }

        // Check computation status
        if(compute_status != rocblas_status_success)
        {
            alloc_status = compute_status;
            goto cleanup;
        }
    }

    // ===== SUCCESS PATH =====
    alloc_status = rocblas_status_success;

cleanup:
    // ===== CLEANUP (executed on all exit paths) =====

    // Free allocated workspace in reverse order
    if(pivotidx_mem != nullptr)
        rocblas_free(handle, pivotidx_mem);

    if(pivotval_mem != nullptr)
        rocblas_free(handle, pivotval_mem);

    if(scalars_mem != nullptr)
        rocblas_free(handle, scalars_mem);

    return alloc_status;
}

/*
ERROR HANDLING STRATEGY:

1. **Early Validation**: Catch invalid arguments before any allocation
   - Minimizes cleanup complexity
   - Provides clear error messages

2. **Allocation Failure Handling**:
   - Use goto cleanup pattern for consistent cleanup
   - Free all successfully allocated resources
   - Propagate error status to caller

3. **Computation Errors**:
   - Kernel execution errors (if any) propagate through compute_status
   - Cleanup happens regardless of computation outcome

4. **Resource Leak Prevention**:
   - All allocated memory freed on every exit path (success or failure)
   - Handle freed in reverse allocation order (good practice)

5. **Info Array Semantics**:
   - info[i] = 0: successful factorization for batch element i
   - info[i] > 0: singularity detected at column info[i] (1-based) in batch element i
   - info is device memory - user must copy to host to check results

USAGE EXAMPLE:

```cpp
// Factorize 10 128×128 matrices in one call
const int batch = 10;
const int n = 128;

double* A_host[batch];
int* ipiv_host[batch];
int* info_host[batch];

// Allocate and initialize matrices (omitted)

// Copy pointer arrays to device
double** A_dev;
int** ipiv_dev;
int** info_dev;

hipMalloc(&A_dev, batch * sizeof(double*));
hipMalloc(&ipiv_dev, batch * sizeof(int*));
hipMalloc(&info_dev, batch * sizeof(int*));

hipMemcpy(A_dev, A_host, batch * sizeof(double*), hipMemcpyHostToDevice);
// ... similar for ipiv_dev, info_dev

// Call wrapper
rocblas_handle handle;
rocblas_create_handle(&handle);

rocblas_status status = rocsolver_getf2_batched_wrapper<true, double>(
    handle, n, n, A_dev, n, ipiv_dev, info_dev, batch, true);

if(status != rocblas_status_success)
{
    std::cerr << "GETF2 failed with status " << status << std::endl;
}
else
{
    // Check info array for singularities
    int info_results[batch];
    hipMemcpy(info_results, info_dev, batch * sizeof(int), hipMemcpyDeviceToHost);

    for(int i = 0; i < batch; i++)
    {
        if(info_results[i] > 0)
            std::cout << "Batch " << i << ": singular at column "
                      << info_results[i] << std::endl;
    }
}

// Cleanup (omitted)
```

THREAD SAFETY:

- rocblas_handle is NOT thread-safe (each thread needs its own handle)
- Multiple threads can call this function with different handles safely
- Device memory allocations are handle-specific (stream-ordered)

PERFORMANCE NOTES:

- Kernel selection (OPTIMAL flag) can provide 2-10× speedup for small matrices
- Workspace allocation overhead: ~10-50 μs (amortized over batch)
- For very large batches, consider stream-based pipelining to overlap compute and allocation
*/""",
        rationale="Production library code requires careful attention to resource management, error handling, and user experience. This wrapper demonstrates industry best practices: RAII-style cleanup with goto, comprehensive validation, transparent optimization selection, and clear documentation.",
        tags=["coding", "API-design", "error-handling", "resource-management", "batched-operations", "wrapper-function"]
    ))

    return entries

def main():
    """Generate and write the dataset to a JSONL file."""
    entries = generate_dataset()

    output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_getf2.jsonl"

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
