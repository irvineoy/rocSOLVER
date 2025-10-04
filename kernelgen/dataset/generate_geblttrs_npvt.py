#!/usr/bin/env python3
"""
Generator script for roclapack_geblttrs_npvt SFT dataset.

GEBLTTRS_NPVT solves a block tridiagonal system using LU factorization
without pivoting. It's the companion to GEBLTTRF_NPVT.
"""

import json
from datetime import datetime

def create_entry(id_num, level, interface, instruction, context_text, code_blocks,
                answer, rationale, tags):
    """Create a single dataset entry."""
    return {
        "id": f"2025-10-03T06:00:00.{id_num:03d}",
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
    """Generate all dataset entries for roclapack_geblttrs_npvt."""
    entries = []

    # ========== L1 ENTRIES ==========

    # L1-1: Forward/Backward Solve Loop Analysis (Analysis)
    entries.append(create_entry(
        id_num=1,
        level="L1",
        interface="roclapack_geblttrs_npvt",
        instruction="Analyze the forward and backward solve loops in the GEBLTTRS_NPVT solver. Explain the mathematical operations in each pass and why the backward solve starts at k = nblocks - 2 instead of nblocks - 1.",
        context_text="The GEBLTTRS_NPVT solver implements a two-pass algorithm to solve block tridiagonal systems: a forward solve followed by a backward solve. Understanding the indexing and loop bounds is critical for correctness.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """    // forward solve
    for(rocblas_int k = 0; k < nblocks; k++)
    {
        if(k > 0)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                           &minone, A, shiftA + (k - 1) * bsa, inca, lda, strideA, X,
                           shiftX + (k - 1) * bsx, incx, ldx, strideX, &one, X, shiftX + k * bsx,
                           incx, ldx, strideX, batch_count, (T**)nullptr);

        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nrhs, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, X, shiftX + k * bsx, incx, ldx, strideX, batch_count, work1,
            work2, work3, work4, optim_mem, false);
    }

    // backward solve
    for(rocblas_int k = nblocks - 2; k >= 0; k--)
    {
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                       &minone, C, shiftC + k * bsc, incc, ldc, strideC, X, shiftX + (k + 1) * bsx,
                       incx, ldx, strideX, &one, X, shiftX + k * bsx, incx, ldx, strideX,
                       batch_count, (T**)nullptr);
    }"""
            }
        ],
        answer="""The forward and backward solve implement a two-stage block tridiagonal solver:

**Forward Solve (k = 0 to nblocks-1):**
For each block k:
1. If k > 0: X[k] -= A[k-1] * X[k-1]  (subtract lower diagonal contribution)
2. Solve: B[k] * X[k] = X[k]  (apply diagonal block inverse via GETRS)

This eliminates the lower diagonal, transforming the system to upper triangular form.

**Backward Solve (k = nblocks-2 down to 0):**
For each block k:
- X[k] -= C[k] * X[k+1]  (subtract upper diagonal contribution)

**Why start at nblocks-2?**
The last block (k = nblocks-1) was already fully solved in the forward pass - it has no upper diagonal neighbor (C[nblocks-1] doesn't exist). Starting the backward pass at nblocks-2 is the first block that needs backward substitution.

This follows the standard LU-based block tridiagonal solver pattern.""",
        rationale="Understanding the mathematical flow and indexing conventions is essential for implementing or debugging block tridiagonal solvers. The k = nblocks - 2 starting point is a common source of off-by-one errors.",
        tags=["analysis", "algorithm", "block-tridiagonal", "forward-backward-solve"]
    ))

    # L1-2: Block Stride Calculation (Coding)
    entries.append(create_entry(
        id_num=2,
        level="L1",
        interface="roclapack_geblttrs_npvt",
        instruction="Implement a device helper function that computes the offset for accessing block (k) in a block-strided array. The function should take the block index k, block size nb, and leading dimension ld, and return the element offset.",
        context_text="GEBLTTRS_NPVT uses block strides to navigate through matrices A, B, C, and X. Understanding how to compute these offsets is fundamental for accessing the correct data.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """    // block strides
    rocblas_int bsa = lda * nb;
    rocblas_int bsb = ldb * nb;
    rocblas_int bsc = ldc * nb;
    rocblas_int bsx = ldx * nrhs;"""
            }
        ],
        answer="""template <typename I>
__device__ __host__ inline I block_offset(I block_idx, I block_size, I leading_dim)
{
    // Block stride is leading_dim * block_size
    // For block k, offset is k * (leading_dim * block_size)
    return block_idx * leading_dim * block_size;
}

// Usage example:
// To access B[k], use: B + shiftB + block_offset(k, nb, ldb)
// To access A[k], use: A + shiftA + block_offset(k, nb, lda)
// To access X[k] (RHS), use: X + shiftX + block_offset(k, nrhs, ldx)
//   Note: For X, "block_size" is nrhs, not nb

// Alternative version with inc parameter for strided access:
template <typename I>
__device__ __host__ inline I block_offset_strided(I block_idx, I block_size,
                                                   I inc, I leading_dim)
{
    return block_idx * leading_dim * block_size;
    // inc would be used for element-wise stride within blocks if needed
}""",
        rationale="Block stride calculation is a fundamental building block for block-structured algorithms. This simple helper improves code readability and reduces arithmetic errors in index calculations.",
        tags=["coding", "memory-layout", "helper-function", "block-stride"]
    ))

    # L1-3: Workspace Memory Sizing (Analysis)
    entries.append(create_entry(
        id_num=3,
        level="L1",
        interface="roclapack_geblttrs_npvt",
        instruction="Why does the rocsolver_geblttrs_npvt_getMemorySize function delegate all workspace calculations to rocsolver_getrs_getMemorySize? What does this tell us about the workspace requirements of GEBLTTRS_NPVT?",
        context_text="Understanding workspace allocation patterns helps optimize memory usage and reveals the computational structure of algorithms.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_geblttrs_npvt_getMemorySize(const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int nrhs,
                                           const rocblas_int batch_count,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           bool* optim_mem,
                                           const rocblas_int ldb = 1,
                                           const rocblas_int ldx = 1,
                                           const rocblas_int incb = 1,
                                           const rocblas_int incx = 1)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        return;
    }

    // size requirements for getrs
    rocsolver_getrs_getMemorySize<BATCHED, STRIDED, T>(rocblas_operation_none, nb, nrhs, batch_count,
                                                       size_work1, size_work2, size_work3,
                                                       size_work4, optim_mem, ldb, ldx, incb, incx);
}"""
            }
        ],
        answer="""The delegation to rocsolver_getrs_getMemorySize reveals several key insights:

1. **GETRS is the dominant workspace consumer**: The workspace is sized for GETRS operations on blocks of size nb × nrhs. GEMM operations don't require additional workspace beyond what GETRS needs.

2. **Workspace reuse across iterations**: The same workspace buffers (work1-work4) are reused for all nblocks GETRS calls in both forward and backward passes. This is safe because each GETRS call completes before the next begins.

3. **No additional overhead**: GEBLTTRS_NPVT doesn't need extra temporary storage beyond what GETRS requires. The matrix updates (GEMM) are done in-place on X.

4. **Memory efficiency**: Total workspace = max(GETRS workspace), not sum(nblocks × GETRS workspace). This is a significant savings for large nblocks.

5. **Single block sizing**: The workspace is sized for nb × nrhs (one block), not (nb × nblocks) × nrhs (full system), reflecting the block-by-block solution approach.

This pattern is typical of iterative block algorithms where workspace can be reused across iterations.""",
        rationale="Workspace analysis reveals algorithm structure and memory efficiency patterns. Understanding why workspace doesn't scale with nblocks is important for performance modeling and optimization.",
        tags=["analysis", "memory-management", "workspace", "optimization"]
    ))

    # ========== L2 ENTRIES ==========

    # L2-1: GEMM-GETRS Interleaving Pattern (Analysis)
    entries.append(create_entry(
        id_num=4,
        level="L2",
        interface="roclapack_geblttrs_npvt",
        instruction="Analyze the GEMM-GETRS interleaving pattern in the forward solve. Explain the data dependencies and identify opportunities for kernel fusion or pipelining. What prevents executing multiple blocks in parallel?",
        context_text="The forward solve alternates between GEMM and GETRS operations. Understanding data dependencies is key to identifying parallelization and optimization opportunities.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """    // forward solve
    for(rocblas_int k = 0; k < nblocks; k++)
    {
        if(k > 0)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                           &minone, A, shiftA + (k - 1) * bsa, inca, lda, strideA, X,
                           shiftX + (k - 1) * bsx, incx, ldx, strideX, &one, X, shiftX + k * bsx,
                           incx, ldx, strideX, batch_count, (T**)nullptr);

        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nrhs, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, X, shiftX + k * bsx, incx, ldx, strideX, batch_count, work1,
            work2, work3, work4, optim_mem, false);
    }"""
            }
        ],
        answer="""**Data Dependencies:**

For block k:
- GEMM: X[k] -= A[k-1] * X[k-1]  (reads X[k], X[k-1]; writes X[k])
- GETRS: Solve B[k] * X[k] = X[k]  (reads B[k]; reads/writes X[k])

**Sequential Dependency Chain:**
- Block k's GEMM requires X[k-1] from block k-1's GETRS
- Block k's GETRS requires the updated X[k] from its own GEMM
- This creates a strict sequential dependency: block k cannot start until block k-1 completes

**Why Parallel Execution is Prevented:**
Each block depends on the solution from the previous block. This is the classic forward substitution pattern - inherently sequential in the block dimension.

**Optimization Opportunities:**

1. **Kernel Fusion (GEMM+GETRS):**
   - Fuse GEMM and GETRS into a single kernel for blocks k > 0
   - Avoids intermediate X[k] writeback and reload
   - Requires inlining GETRS logic into GEMM kernel

2. **Batch-Level Parallelism:**
   - Different batch elements are independent
   - Can pipeline batch[i]'s block k with batch[i+1]'s block k-1
   - Helps amortize kernel launch overhead

3. **RHS-Level Parallelism:**
   - Multiple right-hand sides (nrhs) can share GETRS workspace
   - Already exploited by existing GEMM/GETRS implementations

4. **Asynchronous Launch:**
   - Use streams to overlap GEMM compute with GETRS memory operations
   - Limited benefit due to data dependencies

**Conclusion:** The sequential nature of forward substitution is fundamental and cannot be parallelized across blocks, but fusion and batch-level pipelining offer practical speedups.""",
        rationale="Understanding data dependencies and parallelization barriers is critical for GPU optimization. Block tridiagonal solvers have inherent sequential constraints that limit but don't eliminate optimization opportunities.",
        tags=["analysis", "data-dependency", "parallelism", "kernel-fusion", "optimization"]
    ))

    # L2-2: Optimized Forward Solve Kernel (Coding)
    entries.append(create_entry(
        id_num=5,
        level="L2",
        interface="roclapack_geblttrs_npvt",
        instruction="Design a fused GEMM+GETRS kernel for the forward solve iteration. The kernel should compute X[k] -= A[k-1] * X[k-1] followed immediately by B[k] * X[k] = X[k] without intermediate global memory traffic. Provide the kernel signature and high-level implementation strategy.",
        context_text="Fusing GEMM and GETRS operations can reduce memory traffic by keeping intermediate results in shared memory or registers. This is a common optimization for block-iterative solvers.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """    for(rocblas_int k = 0; k < nblocks; k++)
    {
        if(k > 0)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                           &minone, A, shiftA + (k - 1) * bsa, inca, lda, strideA, X,
                           shiftX + (k - 1) * bsx, incx, ldx, strideX, &one, X, shiftX + k * bsx,
                           incx, ldx, strideX, batch_count, (T**)nullptr);

        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nrhs, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, X, shiftX + k * bsx, incx, ldx, strideX, batch_count, work1,
            work2, work3, work4, optim_mem, false);
    }"""
            }
        ],
        answer="""template <typename T, int NB_MAX = 256, int NRHS_MAX = 16>
__global__ void geblttrs_forward_fused_kernel(
    const int nb,
    const int nrhs,
    const T* __restrict__ A_k_minus_1,  // A[k-1], size nb × nb
    const T* __restrict__ B_k,          // B[k] (LU factored), size nb × nb
    const T* __restrict__ X_k_minus_1,  // X[k-1] (already solved), size nb × nrhs
    T* __restrict__ X_k,                // X[k] (input/output), size nb × nrhs
    const int lda,
    const int ldb,
    const int ldx)
{
    // Shared memory for intermediate results and block data
    __shared__ T smem_X_local[NB_MAX * NRHS_MAX];  // Local copy of X[k] after GEMM
    __shared__ T smem_B[NB_MAX * NB_MAX];          // B[k] matrix
    __shared__ T smem_A[NB_MAX * NB_MAX];          // A[k-1] matrix

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // ===== STEP 1: Load B[k] and A[k-1] into shared memory =====
    for(int idx = tid; idx < nb * nb; idx += nthreads)
    {
        int row = idx % nb;
        int col = idx / nb;
        smem_B[col * nb + row] = B_k[col * ldb + row];
        smem_A[col * nb + row] = A_k_minus_1[col * lda + row];
    }

    // ===== STEP 2: Load X[k] into shared memory =====
    for(int idx = tid; idx < nb * nrhs; idx += nthreads)
    {
        int row = idx % nb;
        int col = idx / nb;
        smem_X_local[col * nb + row] = X_k[col * ldx + row];
    }

    __syncthreads();

    // ===== STEP 3: GEMM - Compute X[k] -= A[k-1] * X[k-1] =====
    // Use block-wise GEMM with registers for accumulation
    for(int rhs_col = 0; rhs_col < nrhs; rhs_col++)
    {
        for(int row = tid; row < nb; row += nthreads)
        {
            T sum = smem_X_local[rhs_col * nb + row];

            // Compute dot product: A[k-1][row,:] · X[k-1][:,rhs_col]
            for(int k_inner = 0; k_inner < nb; k_inner++)
            {
                sum -= smem_A[k_inner * nb + row] *
                       X_k_minus_1[rhs_col * ldx + k_inner];
            }

            smem_X_local[rhs_col * nb + row] = sum;
        }
    }

    __syncthreads();

    // ===== STEP 4: GETRS - Solve B[k] * X[k] = X[k] in-place =====
    // Forward substitution (L * y = X[k])
    for(int col = 0; col < nb; col++)
    {
        // Broadcast pivot row element
        __shared__ T pivot_recip;
        if(tid == 0)
            pivot_recip = T(1.0) / smem_B[col * nb + col];
        __syncthreads();

        for(int rhs_col = tid; rhs_col < nrhs; rhs_col += nthreads)
        {
            T x_val = smem_X_local[rhs_col * nb + col];

            // Apply pivot scaling
            x_val *= pivot_recip;
            smem_X_local[rhs_col * nb + col] = x_val;

            // Update remaining rows
            for(int row = col + 1; row < nb; row++)
            {
                smem_X_local[rhs_col * nb + row] -=
                    smem_B[col * nb + row] * x_val;
            }
        }
        __syncthreads();
    }

    // Backward substitution (U * X[k] = y)
    for(int col = nb - 1; col >= 0; col--)
    {
        for(int rhs_col = tid; rhs_col < nrhs; rhs_col += nthreads)
        {
            T x_val = smem_X_local[rhs_col * nb + col];

            for(int row = 0; row < col; row++)
            {
                smem_X_local[rhs_col * nb + row] -=
                    smem_B[col * nb + row] * x_val;
            }
        }
        __syncthreads();
    }

    // ===== STEP 5: Write back results to global memory =====
    for(int idx = tid; idx < nb * nrhs; idx += nthreads)
    {
        int row = idx % nb;
        int col = idx / nb;
        X_k[col * ldx + row] = smem_X_local[col * nb + row];
    }
}

// Launch configuration:
// - One block per batch element
// - Threads: 128-256 (tunable based on nb and nrhs)
// - Shared memory: 2*nb^2 + nb*nrhs elements of type T
// - Call once per iteration k > 0 in the forward solve""",
        rationale="Fused kernels are a key GPU optimization technique. This implementation demonstrates how to combine GEMM and triangular solve while keeping intermediate data in fast shared memory, reducing global memory traffic by ~50% for this operation.",
        tags=["coding", "kernel-fusion", "optimization", "shared-memory", "GEMM", "GETRS"]
    ))

    # ========== L3 ENTRIES ==========

    # L3-1: Complete Algorithm Flow (Analysis)
    entries.append(create_entry(
        id_num=6,
        level="L3",
        interface="roclapack_geblttrs_npvt",
        instruction="Trace the complete data flow for solving a 3-block tridiagonal system (nblocks=3, nb=256, nrhs=1) from API entry to final solution. Include all GEMM and GETRS calls with their dimensions and the state of X after each operation.",
        context_text="Understanding the complete execution flow helps debug issues and optimize performance. GEBLTTRS_NPVT is the solve phase for systems factorized by GEBLTTRF_NPVT.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int nrhs,
                                                U A, const rocblas_int shiftA, const rocblas_int inca,
                                                const rocblas_int lda, const rocblas_stride strideA,
                                                U B, const rocblas_int shiftB, const rocblas_int incb,
                                                const rocblas_int ldb, const rocblas_stride strideB,
                                                U C, const rocblas_int shiftC, const rocblas_int incc,
                                                const rocblas_int ldc, const rocblas_stride strideC,
                                                U X, const rocblas_int shiftX, const rocblas_int incx,
                                                const rocblas_int ldx, const rocblas_stride strideX,
                                                const rocblas_int batch_count,
                                                void* work1, void* work2, void* work3, void* work4,
                                                bool optim_mem)
{
    // forward solve
    for(rocblas_int k = 0; k < nblocks; k++)
    {
        if(k > 0)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                           &minone, A, shiftA + (k - 1) * bsa, inca, lda, strideA, X,
                           shiftX + (k - 1) * bsx, incx, ldx, strideX, &one, X, shiftX + k * bsx,
                           incx, ldx, strideX, batch_count, (T**)nullptr);

        rocsolver_getrs_template<BATCHED, STRIDED, T>(
            handle, rocblas_operation_none, nb, nrhs, B, shiftB + k * bsb, incb, ldb, strideB,
            (rocblas_int*)nullptr, 0, X, shiftX + k * bsx, incx, ldx, strideX, batch_count, work1,
            work2, work3, work4, optim_mem, false);
    }

    // backward solve
    for(rocblas_int k = nblocks - 2; k >= 0; k--)
    {
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb, nrhs, nb,
                       &minone, C, shiftC + k * bsc, incc, ldc, strideC, X, shiftX + (k + 1) * bsx,
                       incx, ldx, strideX, &one, X, shiftX + k * bsx, incx, ldx, strideX,
                       batch_count, (T**)nullptr);
    }
}"""
            }
        ],
        answer="""**Complete Execution Trace for nblocks=3, nb=256, nrhs=1:**

**Initial State:**
- Input: X[0:767] contains right-hand side b
- Matrices: A[0:1] (subdiagonal), B[0:2] (diagonal, LU factored), C[0:1] (superdiagonal)
- System to solve: [B0  C0   0 ] [x0]   [b0]
                   [A0  B1  C1 ] [x1] = [b1]
                   [ 0  A1  B2 ] [x2]   [b2]

**FORWARD SOLVE (k = 0 to 2):**

**k=0:**
- GETRS(B0, X[0:255]): Solve B0 * X[0:255] = X[0:255]
  - Dimensions: 256×256 matrix, 256×1 RHS
  - X[0:255] now contains partially solved x0

**k=1:**
- GEMM: X[256:511] -= A0 * X[0:255]
  - Dimensions: (256×256) × (256×1) = 256×1
  - Alpha=-1, Beta=1
  - Eliminates A0 contribution: X[1] = b1 - A0*x0
- GETRS(B1, X[256:511]): Solve B1 * X[256:511] = X[256:511]
  - Dimensions: 256×256 matrix, 256×1 RHS
  - X[256:511] now contains partially solved x1

**k=2:**
- GEMM: X[512:767] -= A1 * X[256:511]
  - Dimensions: (256×256) × (256×1) = 256×1
  - Eliminates A1 contribution: X[2] = b2 - A1*x1
- GETRS(B2, X[512:767]): Solve B2 * X[512:767] = X[512:767]
  - Dimensions: 256×256 matrix, 256×1 RHS
  - X[512:767] now contains fully solved x2 (last block is complete)

**After Forward Solve:**
- X[512:767] = x2 (final)
- X[256:511] = x1 (partial - still needs C1*x2 subtracted)
- X[0:255] = x0 (partial - still needs C0*x1 subtracted)

**BACKWARD SOLVE (k = 1 down to 0):**

**k=1:**
- GEMM: X[256:511] -= C1 * X[512:767]
  - Dimensions: (256×256) × (256×1) = 256×1
  - Eliminates C1 contribution
  - X[256:511] now contains final x1

**k=0:**
- GEMM: X[0:255] -= C0 * X[256:511]
  - Dimensions: (256×256) × (256×1) = 256×1
  - Eliminates C0 contribution
  - X[0:255] now contains final x0

**Final State:**
- X[0:255] = x0 (final solution for block 0)
- X[256:511] = x1 (final solution for block 1)
- X[512:767] = x2 (final solution for block 2)

**Total Operations:**
- Forward: 3 GETRS (256×256×1 each) + 2 GEMM (256×256×1 each)
- Backward: 2 GEMM (256×256×1 each)
- Total: 3 GETRS + 4 GEMM calls""",
        rationale="Tracing complete execution flows is essential for debugging, performance analysis, and understanding algorithm correctness. This level of detail helps identify optimization opportunities and verify implementation correctness.",
        tags=["analysis", "algorithm-trace", "data-flow", "block-tridiagonal"]
    ))

    # L3-2: Integration with GEBLTTRF_NPVT (Coding)
    entries.append(create_entry(
        id_num=7,
        level="L3",
        interface="roclapack_geblttrs_npvt",
        instruction="Design a wrapper function that performs both factorization (GEBLTTRF_NPVT) and solve (GEBLTTRS_NPVT) for a block tridiagonal system. Include proper error handling, workspace allocation, and memory layout documentation. The function should handle batched operations.",
        context_text="GEBLTTRF_NPVT and GEBLTTRS_NPVT are typically used together. A combined interface simplifies usage and ensures correct memory layouts and workspace handling.",
        code_blocks=[
            {
                "path": "library/src/lapack/roclapack_geblttrs_npvt.hpp",
                "language": "cpp",
                "content": """template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_template(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                const rocblas_int nrhs,
                                                U A, U B, U C, U X,
                                                /* ... parameters ... */,
                                                void* work1, void* work2, void* work3, void* work4,
                                                bool optim_mem)"""
            }
        ],
        answer="""template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gebltsolve_npvt(
    rocblas_handle handle,
    const rocblas_int nb,          // Block size
    const rocblas_int nblocks,     // Number of diagonal blocks
    const rocblas_int nrhs,        // Number of right-hand sides
    U A,                           // Subdiagonal blocks (nblocks-1 blocks)
    const rocblas_int lda,
    const rocblas_stride strideA,
    U B,                           // Diagonal blocks (nblocks blocks) - overwritten with LU
    const rocblas_int ldb,
    const rocblas_stride strideB,
    U C,                           // Superdiagonal blocks (nblocks-1 blocks)
    const rocblas_int ldc,
    const rocblas_stride strideC,
    U X,                           // Right-hand side (input) / Solution (output)
    const rocblas_int ldx,
    const rocblas_stride strideX,
    rocblas_int* info,             // Info array for factorization status
    const rocblas_int batch_count)
{
    // ===== Input Validation =====
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    if(nb < 0 || nblocks < 0 || nrhs < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    if(lda < nb || ldb < nb || ldc < nb || ldx < nb)
        return rocblas_status_invalid_size;

    if((nblocks > 1 && (!A || !C)) || !B || (nrhs > 0 && !X) || !info)
        return rocblas_status_invalid_pointer;

    // Quick return
    if(nb == 0 || nblocks == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    // ===== Workspace Allocation =====
    size_t size_work_trf1, size_work_trf2, size_work_trf3, size_work_trf4;
    size_t size_work_trs1, size_work_trs2, size_work_trs3, size_work_trs4;
    bool optim_mem_trf, optim_mem_trs;

    // Get workspace sizes for factorization
    rocsolver_geblttrf_npvt_getMemorySize<BATCHED, STRIDED, T>(
        nb, nblocks, batch_count,
        &size_work_trf1, &size_work_trf2, &size_work_trf3, &size_work_trf4,
        &optim_mem_trf, ldb);

    // Get workspace sizes for solve
    rocsolver_geblttrs_npvt_getMemorySize<BATCHED, STRIDED, T>(
        nb, nblocks, nrhs, batch_count,
        &size_work_trs1, &size_work_trs2, &size_work_trs3, &size_work_trs4,
        &optim_mem_trs, ldb, ldx);

    // Allocate workspace (use max of both requirements since they don't overlap)
    size_t size_work1 = std::max(size_work_trf1, size_work_trs1);
    size_t size_work2 = std::max(size_work_trf2, size_work_trs2);
    size_t size_work3 = std::max(size_work_trf3, size_work_trs3);
    size_t size_work4 = std::max(size_work_trf4, size_work_trs4);

    void *work1 = nullptr, *work2 = nullptr, *work3 = nullptr, *work4 = nullptr;

    rocblas_status status;
    if(size_work1 > 0)
        status = rocblas_malloc(handle, &work1, size_work1);
    if(size_work2 > 0 && status == rocblas_status_success)
        status = rocblas_malloc(handle, &work2, size_work2);
    if(size_work3 > 0 && status == rocblas_status_success)
        status = rocblas_malloc(handle, &work3, size_work3);
    if(size_work4 > 0 && status == rocblas_status_success)
        status = rocblas_malloc(handle, &work4, size_work4);

    if(status != rocblas_status_success)
    {
        // Cleanup on allocation failure
        if(work1) rocblas_free(handle, work1);
        if(work2) rocblas_free(handle, work2);
        if(work3) rocblas_free(handle, work3);
        return status;
    }

    // ===== PHASE 1: Factorization =====
    // Compute LU factorization of block tridiagonal matrix
    // B[k] is overwritten with LU factors
    status = rocsolver_geblttrf_npvt_template<BATCHED, STRIDED, T>(
        handle, nb, nblocks,
        A, 0, 1, lda, strideA,
        B, 0, 1, ldb, strideB,
        C, 0, 1, ldc, strideC,
        info, batch_count,
        work1, work2, work3, work4, optim_mem_trf);

    if(status != rocblas_status_success)
    {
        rocblas_free(handle, work1);
        rocblas_free(handle, work2);
        rocblas_free(handle, work3);
        rocblas_free(handle, work4);
        return status;
    }

    // Check for singularities - if any batch element failed, return error
    rocblas_int info_host[batch_count];
    hipMemcpy(info_host, info, batch_count * sizeof(rocblas_int), hipMemcpyDeviceToHost);

    bool has_singularity = false;
    for(int i = 0; i < batch_count; i++)
    {
        if(info_host[i] != 0)
        {
            has_singularity = true;
            break;
        }
    }

    if(has_singularity)
    {
        rocblas_free(handle, work1);
        rocblas_free(handle, work2);
        rocblas_free(handle, work3);
        rocblas_free(handle, work4);
        return rocblas_status_invalid_value;  // Or custom error code
    }

    // ===== PHASE 2: Solve =====
    // Solve the factorized system: (L*U) * X = RHS
    status = rocsolver_geblttrs_npvt_template<BATCHED, STRIDED, T>(
        handle, nb, nblocks, nrhs,
        A, 0, 1, lda, strideA,
        B, 0, 1, ldb, strideB,
        C, 0, 1, ldc, strideC,
        X, 0, 1, ldx, strideX,
        batch_count,
        work1, work2, work3, work4, optim_mem_trs);

    // ===== Cleanup =====
    rocblas_free(handle, work1);
    rocblas_free(handle, work2);
    rocblas_free(handle, work3);
    rocblas_free(handle, work4);

    return status;
}

/*
MEMORY LAYOUT DOCUMENTATION:

Input Arrays:
- A: Subdiagonal blocks, size nb×nb per block, (nblocks-1) blocks total
  Layout: [A[0] | A[1] | ... | A[nblocks-2]]
  Each block A[k] is size nb×nb with leading dimension lda

- B: Diagonal blocks, size nb×nb per block, nblocks blocks total
  Layout: [B[0] | B[1] | ... | B[nblocks-1]]
  Each block B[k] is size nb×nb with leading dimension ldb
  OVERWRITTEN with LU factors during factorization

- C: Superdiagonal blocks, size nb×nb per block, (nblocks-1) blocks total
  Layout: [C[0] | C[1] | ... | C[nblocks-2]]
  Each block C[k] is size nb×nb with leading dimension ldc

- X: Right-hand side (input) and solution (output)
  Size: (nb × nblocks) × nrhs with leading dimension ldx
  Layout: [X[0:nb-1,:] | X[nb:2*nb-1,:] | ... ]

Batching:
- For BATCHED: arrays are arrays of pointers
- For STRIDED: stride parameters control batch element spacing
- batch_count controls number of independent systems

Error Handling:
- info[i] = 0: success for batch element i
- info[i] > 0: singularity detected at block info[i] in batch element i
- Return status indicates global success/failure
*/""",
        rationale="Wrapper functions that combine related operations improve usability and reduce user errors. This implementation demonstrates proper resource management, error handling, and documentation practices for library development.",
        tags=["coding", "API-design", "wrapper", "error-handling", "workspace-management"]
    ))

    return entries

def main():
    """Generate and write the dataset to a JSONL file."""
    entries = generate_dataset()

    output_path = "/root/rocSOLVER/kernelgen/dataset/roclapack_geblttrs_npvt.jsonl"

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
