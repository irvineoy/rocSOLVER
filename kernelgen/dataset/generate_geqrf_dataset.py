#!/usr/bin/env python3
"""
Generate JSONL dataset for roclapack_geqrf.yaml
Creates L1/L2/L3 questions with ≥50% coding tasks
"""

import json
import time

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Create a single JSONL entry"""
    return {
        'id': str(int(time.time() * 1000000)),
        'level': level,
        'interface': interface,
        'instruction': instruction,
        'context_text': context_text,
        'code_blocks': code_blocks,
        'answer': answer,
        'rationale': rationale,
        'tags': tags
    }

def generate_dataset():
    entries = []

    # ============================================================
    # L1 ENTRIES (Single Function/Kernel)
    # ============================================================

    # L1-1: geqr2_kernel_small LDS layout (CODING)
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Calculate the shared memory requirements for geqr2_kernel_small kernel processing a 32x32 matrix with MAX_THDS=256 threads and warpSize=64. Explain the LDS layout.',
        context_text='The geqr2_kernel_small kernel fuses the entire QR factorization for small matrices, loading the matrix into LDS and performing all operations on-chip.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqr2.hpp',
            'language': 'hip',
            'content': '''template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void geqr2_kernel_small(...)
{
    // shared variables
    extern __shared__ double lmem[];
    T* a = reinterpret_cast<T*>(lmem);              // Matrix A: m*n elements
    T* w = reinterpret_cast<T*>(a + m * n);         // Work vector: n elements
    T* tmptau = reinterpret_cast<T*>(w + n);        // Temporary tau: 1 element
    T* sval = reinterpret_cast<T*>(tmptau + 1);     // Reduction array: MAX_THDS/warpSize elements

    // Load logic:
    const size_t lmemsize = ((MAX_THDS / warpSize) + mm + nn + 1 + mm * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn == mm)
    {
        ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>),
                               dim3(1, 1, batch_count), dim3(256),
                               lmemsize, stream, mm, nn, ...);
    }
}'''
        }],
        answer='''For m=n=32, MAX_THDS=256, warpSize=64, float (sizeof(T)=4):

**LDS layout (in order):**
1. **Matrix A**: m×n = 32×32 = 1024 elements
2. **Work vector w**: n = 32 elements
3. **Temporary tau**: 1 element
4. **Reduction array sval**: MAX_THDS/warpSize = 256/64 = 4 elements

**Total elements**: 1024 + 32 + 1 + 4 = 1061 elements

**Total bytes**: 1061 × 4 = 4244 bytes

**Formula verification:**
lmemsize = ((256/64) + 32 + 32 + 1 + 32×32) × 4
         = (4 + 32 + 32 + 1 + 1024) × 4
         = 1093 × 4
         = 4372 bytes

**Note:** The formula adds an extra mm term (32 elements) for safety/alignment.

**Purpose of each region:**
- **a[]**: Stores entire matrix for on-chip computation
- **w[]**: Stores GEMV results (w = A^H * v)
- **tmptau[]**: Temporary tau during Householder generation
- **sval[]**: Per-warp reduction results (one per warp)

**Why this kernel is fast:**
- Zero global memory traffic after initial load and final store
- All operations (larfg, gemv, ger) happen in LDS
- ~10-20x faster than calling separate kernels for small matrices''',
        rationale='The LDS layout is carefully designed to pack all temporary data needed for QR factorization. The constraint nn==mm ensures square matrices, which have simpler update patterns. The MAX_THDS/warpSize term accounts for parallel reduction across warps.',
        tags=['coding', 'lds', 'shared-memory', 'small-matrix-kernel', 'geqr2']
    ))
    time.sleep(0.001)

    # L1-2: copymatA1 and addmatA1 kernels (CODING)
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Implement the copymatA1 and addmatA1 kernels used by larfb. The first copies a panel from matrix A to temporary storage, the second subtracts the temp storage back to A.',
        context_text='LARFB (apply block reflector) needs to save and restore panels of matrix A during triangular matrix operations.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larfb.hpp',
            'language': 'hip',
            'content': '''// Copy A1 panel to temporary buffer
template <typename T, typename U>
ROCSOLVER_KERNEL void copymatA1(const rocblas_int ldw,
                                const rocblas_int order,
                                U A,
                                const rocblas_int shiftA,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                T* tmptr);

// Subtract temp buffer from A1 panel
template <typename T, typename U>
ROCSOLVER_KERNEL void addmatA1(const rocblas_int ldw,
                               const rocblas_int order,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* tmptr);

// Launch: dim3(blocksx, blocksy, batch_count), dim3(32, 32)
// where blocksx = (order - 1)/32 + 1, blocksy = (ldw - 1)/32 + 1'''
        }],
        answer='''template <typename T, typename U>
ROCSOLVER_KERNEL void copymatA1(const rocblas_int ldw,
                                const rocblas_int order,
                                U A,
                                const rocblas_int shiftA,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                T* tmptr)
{
    const auto blocksizex = hipBlockDim_x;  // 32
    const auto blocksizey = hipBlockDim_y;  // 32
    const auto b = hipBlockIdx_z;            // batch index
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;  // column
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;  // row

    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Wp[i + j * ldw] = Ap[i + j * lda];  // Copy A to W
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
    const auto blocksizex = hipBlockDim_x;
    const auto blocksizey = hipBlockDim_y;
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_x * blocksizex + hipThreadIdx_x;
    const auto i = hipBlockIdx_y * blocksizey + hipThreadIdx_y;

    rocblas_stride strideW = rocblas_stride(ldw) * order;

    if(i < ldw && j < order)
    {
        T *Ap, *Wp;
        Wp = tmptr + b * strideW;
        Ap = load_ptr_batch<T>(A, b, shiftA, strideA);

        Ap[i + j * lda] -= Wp[i + j * ldw];  // A -= W (subtract update)
    }
}

// Use case in larfb:
// 1. copymatA1: Save A1 to tmptr before TRMM modifies it
// 2. [Perform TRMM and GEMM operations on tmptr]
// 3. addmatA1: Apply update A1 -= modified(tmptr)''',
        rationale='These kernels provide efficient panel copy/update for LARFB. The 32×32 thread blocks give good occupancy while ensuring coalesced memory access (threads in a warp access consecutive columns). The subtraction in addmatA1 applies the accumulated block reflector update.',
        tags=['coding', 'kernel-implementation', 'larfb', 'memory-copy']
    ))
    time.sleep(0.001)

    # L1-3: larft_kernel_forward analysis
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Analyze the larft_kernel_forward synchronization pattern. Why does it need __syncthreads() calls between the GEMV and TRMV phases?',
        context_text='The larft_kernel_forward computes the block reflector T for forward direction QR. It iteratively builds T column-by-column.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larft.hpp',
            'language': 'hip',
            'content': '''template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_forward(...)
{
    for(rocblas_int kk = 1; kk < k; kk++)
    {
        // Phase 1: GEMV - compute tau * V^H * v
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = 0; j < nn; j++)
                temp += conj(Vm[j + i * ldv]) * Vx[j];
            work[i] = tau[kk] * temp + Fx[i];
        }
        __syncthreads();

        // Phase 2: TRMV - multiply by previous triangular factor
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = i; j < mm; j++)
                temp += F[i + j * ldf] * work[j];
            Fx[i] = temp;
        }
        __syncthreads();
    }
}'''
        }],
        answer='''The synchronization is required for **read-after-write (RAW) dependencies** in shared memory:

**First __syncthreads() (after GEMV):**
- **Phase 1 writes**: Each thread computes `work[i]` where i = tid, tid+tid_inc, ...
- **Phase 2 reads**: TRMV reads `work[j]` for j in [i, mm)
- **Problem**: Thread computing work[5] may finish before thread computing work[100], but TRMV for i=5 needs work[100]
- **Solution**: Barrier ensures ALL threads finish writing work[] before ANY thread starts reading it

**Second __syncthreads() (after TRMV):**
- **Phase 2 writes**: Threads write to `Fx[i]` (which aliases F memory)
- **Next iteration reads**: GEMV reads `Fx[i]` and TRMV reads `F[...]`
- **Problem**: Without barrier, next iteration could read stale Fx values
- **Solution**: Barrier ensures current iteration completes before next begins

**Why not use warp-level synchronization?**
- Different threads access non-overlapping elements, spanning multiple warps
- work[] and F[] are shared across ALL threads in the block
- tid_inc means thread i doesn\'t just access element i, but i, i+tid_inc, i+2*tid_inc, ...

**Performance impact:**
- Two barriers per iteration × (k-1) iterations = 2(k-1) synchronizations
- For k=32: 62 barriers
- Each barrier costs ~10-50 cycles
- Still much faster than separate kernel launches (1000s of cycles each)

**Example race without barrier:**
```
Thread 0: work[0] = ... ✓     |  Fx[0] = F[0,0]*work[0] + ... (reading work[0], work[1], ...)
Thread 1: work[1] = ... ❌    |  (still computing!)
```''',
        rationale='Shared memory provides fast communication between threads but requires explicit synchronization. The barriers ensure correctness by enforcing ordering constraints: all writes complete before dependent reads begin.',
        tags=['synchronization', 'shared-memory', 'larft', 'race-conditions']
    ))
    time.sleep(0.001)

    # L1-4: set_triangular kernel (CODING)
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Implement the set_triangular kernel for real types (forward direction, column-wise storage). This kernel builds the upper triangular part of T from tau and V.',
        context_text='The set_triangular kernel initializes the triangular factor T. The diagonal is tau[i], and the upper triangle is -tau[i] * V[i,j].',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larft.hpp',
            'language': 'hip',
            'content': '''template <typename T, typename U>
ROCSOLVER_KERNEL void set_triangular(
    const rocblas_int n,
    const rocblas_int k,
    U V, const rocblas_int shiftV, const rocblas_int ldv, const rocblas_stride strideV,
    T* tau, const rocblas_stride strideT,
    T* F, const rocblas_int ldf, const rocblas_stride strideF,
    const rocblas_direct direct,
    const rocblas_storev storev,
    const bool add_fp)  // If true: F already contains V2^H*V2
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // column
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  // row

    if(i < k && j < k) {
        // TODO: Implement for forward direction, column-wise, real type
        // Diagonal: F[i,i] = tau[i]
        // Upper (j<i): F[j,i] = -tau[i] * (Fp[j,i] + V[i,j]) if add_fp else -tau[i] * V[i,j]
        // Lower (j>i): F[j,i] = 0
    }
}'''
        }],
        answer='''template <typename T, typename U>
ROCSOLVER_KERNEL void set_triangular(...)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // column index
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  // row index

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
        {
            // Diagonal: set tau
            Fp[idx2D(j, i, ldf)] = tp[i];
        }
        else if(j < i)
        {
            // Upper triangle: -tau[i] * V[i,j]
            // (V stores unit lower triangular, so V[i,j] is the strict lower part)
            if(!add_fp)
            {
                Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(i, j, ldv)];
            }
            else
            {
                // Add to existing F (which contains V2^H * V2)
                Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(i, j, ldv)]);
            }
        }
        else
        {
            // Lower triangle: zero
            Fp[idx2D(j, i, ldf)] = 0;
        }
    }
}

// Why this structure?
// T = [     tau[0]          -tau[1]*V[1,0]           -tau[2]*V[2,0]          ...]
//     [      0              tau[1]                   -tau[2]*V[2,1]          ...]
//     [      0                 0                     tau[2]                  ...]
//     [...                                                                    ...]
//
// This is the initial structure before iterative refinement in larft_kernel_forward
// The add_fp flag indicates whether V2^H*V2 is pre-computed (blocked algorithm)''',
        rationale='The kernel builds T in a single pass by having each thread compute one element F[j,i]. The upper triangular structure comes from the QR factorization algorithm: later reflectors don\'t affect earlier ones. The -tau[i] factor ensures T^H = -tau*V^H when V is unit lower triangular.',
        tags=['coding', 'larft', 'triangular-factor', 'kernel-implementation']
    ))
    time.sleep(0.001)

    # L1-5: Small-size kernel dispatch in geqr2
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Explain the dispatch logic in geqr2_template that decides between the fused geqr2_kernel_small and the iterative larfg+larf approach.',
        context_text='geqr2 can use a fast fused kernel for small square matrices that fit in LDS, or fallback to iterative column-by-column processing.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqr2.hpp',
            'language': 'cpp',
            'content': '''template <typename T, typename I, typename U>
rocblas_status rocsolver_geqr2_template(...)
{
    I dim = std::min(m, n);
    for(I j = 0; j < dim; ++j)
    {
        I mm = m - j;
        I nn = n - j;

        const size_t lmemsize = ((256/props.warpSize) + mm + nn + 1 + mm*nn) * sizeof(T);
        if(lmemsize <= props.sharedMemPerBlock && nn == mm)
        {
            // Use fused small-matrix kernel
            ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>),
                                   dim3(1, 1, batch_count), dim3(256),
                                   lmemsize, stream, mm, nn, ...);
            break;  // Done! All remaining columns processed in one kernel
        }

        // Fallback: iterative larfg + larf
        rocsolver_larfg_template(...);
        if(j < n - 1)
            rocsolver_larf_template(...);
    }
}'''
        }],
        answer='''The dispatch logic checks **three conditions** for each column j:

**Condition 1: nn == mm (Square trailing matrix)**
- Required because the kernel assumes square matrices
- Simplifies index calculations and memory layout
- If m≠n initially, we process columns until the trailing matrix becomes square

**Condition 2: lmemsize <= props.sharedMemPerBlock (Fits in LDS)**
- Typical GPU: sharedMemPerBlock = 64 KB
- For float (4 bytes), mm=nn=64: lmemsize ≈ (4 + 64 + 64 + 1 + 4096)×4 ≈ 16KB ✓
- For mm=nn=128: lmemsize ≈ (4 + 128 + 128 + 1 + 16384)×4 ≈ 66KB ✗
- Maximum practical size: ~64×64 for float, ~45×45 for double

**Condition 3: Loop iteration j**
- Checked **every iteration** starting from j=0
- Example: 100×50 matrix
  - j=0: m-j=100, n-j=50, mm≠nn → iterative
  - j=1..49: mm≠nn → iterative
  - j=50: m-j=50, n-j=0 → done (no more columns)
- Example: 100×100 matrix
  - j=0: mm=nn=100, but lmemsize too large → iterative
  - j=50: mm=nn=50, lmemsize OK → **fused kernel processes remaining 50×50**

**Why this matters:**
- **100×100 matrix**: First 50 columns use iterative (slow), last 50 use fused kernel (fast)
  - Time: 50×(larfg+larf) + 1×fused ≈ 60% faster than pure iterative
- **64×64 matrix**: All columns use fused kernel from j=0
  - Time: 1×fused ≈ 10-20× faster than 64×(larfg+larf)

**Break statement is critical:**
- Once fused kernel runs, it processes ALL remaining columns
- The `break;` exits the loop entirely
- Without break, code would redundantly process columns again!

**Tradeoff:**
- Fused kernel: High throughput but limited to ~64×64
- Iterative: Flexible size but much slower (kernel launch overhead)''',
        rationale='The dispatch optimizes for the common case where matrices are small enough to fit entirely in LDS. The loop-based dispatch allows handling non-square or large matrices by processing initial columns iteratively until conditions are met.',
        tags=['kernel-dispatch', 'optimization', 'small-matrices', 'geqr2']
    ))
    time.sleep(0.001)

    # ============================================================
    # L2 ENTRIES (Subsystem Scope: 2-5 kernels)
    # ============================================================

    # L2-1: geqr2 + larft + larfb collaboration (CODING)
    entries.append(create_entry(
        level='L2',
        interface='geqrf',
        instruction='Implement the kernel sequence for one iteration of blocked GEQRF: geqr2 on panel, larft to build T, larfb to apply to trailing matrix.',
        context_text='Blocked GEQRF processes the matrix in panels of width nb. Each panel requires: (1) factor panel with geqr2, (2) build block reflector T with larft, (3) apply T to trailing matrix with larfb.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqrf.hpp',
            'language': 'cpp',
            'content': '''// One iteration of blocked GEQRF main loop
while(j < dim - GEQxF_GEQx2_SWITCHSIZE)
{
    jb = std::min(dim - j, nb);

    // Step 1: Factor diagonal panel A(j:m, j:j+jb)
    rocsolver_geqr2_template<T>(handle, m - j, jb, A,
                               shiftA + idx2D(j, j, lda), lda, strideA,
                               (ipiv + j), strideP, batch_count, ...);

    if(j + jb < n)
    {
        // Step 2: Compute block reflector T
        rocsolver_larft_template<T>(handle, rocblas_forward_direction,
                                   rocblas_column_wise,
                                   m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                   (ipiv + j), strideP, T, ldw, strideW, batch_count, ...);

        // Step 3: Apply block reflector to trailing matrix
        rocsolver_larfb_template<T>(handle, rocblas_side_left,
                                   rocblas_operation_conjugate_transpose,
                                   rocblas_forward_direction, rocblas_column_wise,
                                   m - j, n - j - jb, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                   T, 0, ldw, strideW, A, shiftA + idx2D(j, j + jb, lda), lda, strideA,
                                   batch_count, ...);
    }
    j += nb;
}'''
        }],
        answer='''// Simplified single-panel blocked GEQRF iteration
void blocked_geqrf_iteration(handle, m, n, j, nb, A, lda, tau, T, ldw)
{
    rocblas_int jb = std::min(n - j, nb);

    // ===== STEP 1: Factor panel A(j:m, j:j+jb) using GEQR2 =====
    // Input:  A(j:m, j:j+jb) contains original data
    // Output: A(j:m, j:j+jb) = [R; V] where R is upper triangular, V is Householder vectors
    //         tau(j:j+jb) contains Householder coefficients
    rocsolver_geqr2_template<T>(
        handle,
        m - j,        // Height of panel
        jb,           // Width of panel
        A, idx2D(j, j, lda), lda, strideA,
        tau + j, strideP,
        batch_count, ...
    );
    // After geqr2:
    // A(j:m, j:j+jb) = [R(0:jb, 0:jb); V(jb:m-j, 0:jb)]
    // tau(j:j+jb) = [tau0, tau1, ..., tau_{jb-1}]

    if(j + jb < n)
    {
        // ===== STEP 2: Build block reflector T =====
        // T satisfies: (I - V*T*V^H) = H_1 * H_2 * ... * H_jb
        // where H_i = I - tau_i * v_i * v_i^H
        rocsolver_larft_template<T>(
            handle,
            rocblas_forward_direction,   // Apply reflectors left-to-right
            rocblas_column_wise,          // V is stored column-wise
            m - j,                        // Height of V
            jb,                           // Width of V (number of reflectors)
            A, idx2D(j, j, lda), lda, strideA,  // V is stored in A
            tau + j, strideP,                    // Householder coefficients
            T, ldw, strideW,                     // Output: T matrix
            batch_count, ...
        );
        // After larft:
        // T(0:jb, 0:jb) = upper triangular block reflector

        // ===== STEP 3: Apply block reflector to trailing matrix =====
        // Compute: A(j:m, j+jb:n) = (I - V*T*V^H) * A(j:m, j+jb:n)
        // This is equivalent to: H_1 * H_2 * ... * H_jb * A_trailing
        rocsolver_larfb_template<T>(
            handle,
            rocblas_side_left,                        // Apply from left: Q*A
            rocblas_operation_conjugate_transpose,    // Use Q^H (for real: Q^T)
            rocblas_forward_direction,
            rocblas_column_wise,
            m - j,                                     // Rows of A_trailing
            n - j - jb,                                // Cols of A_trailing
            jb,                                        // Order of reflector
            A, idx2D(j, j, lda), lda, strideA,        // V matrix
            T, 0, ldw, strideW,                        // T matrix
            A, idx2D(j, j + jb, lda), lda, strideA,   // A_trailing (in-place update)
            batch_count, ...
        );
        // After larfb:
        // A(j:m, j+jb:n) = Q^H * A_original(j:m, j+jb:n)
        // where Q = H_1 * H_2 * ... * H_jb
    }
}

// Data flow summary:
// 1. geqr2: A_panel → [R, V], tau
// 2. larft: V, tau → T
// 3. larfb: T, V, A_trailing → updated A_trailing
//
// Why this is fast:
// - geqr2: Level-2 BLAS (GEMV), applied to narrow panel (jb columns)
// - larft: Mostly Level-3 BLAS (GEMM for V^H*V), small matrix (jb×jb)
// - larfb: Level-3 BLAS (GEMM), applied to large trailing matrix
// - Ratio: ~90% of FLOPs in larfb (GEMM), which achieves peak GPU throughput''',
        rationale='The three-step sequence transforms the original Level-2 algorithm into a Level-3 algorithm by batching updates. Instead of applying jb reflectors one-by-one to the trailing matrix (jb GEMV calls), we build T once and apply via GEMM. This achieves ~5-10× speedup on GPUs.',
        tags=['coding', 'blocked-algorithm', 'geqr2', 'larft', 'larfb', 'data-flow']
    ))
    time.sleep(0.001)

    # L2-2: larfb TRMM sequence
    entries.append(create_entry(
        level='L2',
        interface='geqrf',
        instruction='Trace the sequence of TRMM and GEMM operations in larfb_template for the left-side, forward, column-wise case. Explain what each computes.',
        context_text='LARFB applies a block reflector Q = I - V*T*V^H to a matrix C. It decomposes V into triangular V1 and rectangular V2 parts.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larfb.hpp',
            'language': 'cpp',
            'content': '''// LARFB for left-side, forward, column-wise
// V = [V1; V2] where V1 is k×k unit lower triangular, V2 is (m-k)×k
// Compute: C := (I - V*T*V^H) * C

// Step 1: Copy C1 to W
copymatA1<<<...>>>(ldw, order, C, offsetC1, lda, strideC, W);

// Step 2: W := V1^H * C1  (TRMM)
rocblasCall_trmm(handle, side_left, lower, conj_trans, unit_diag,
                ldw, order, &one, V, offsetV1, ldv, strideV,
                W, 0, ldw, strideW, batch_count, workArr);

// Step 3: W := V1^H * C1 + V2^H * C2  (GEMM)
if(trap)  // m > k
    rocsolver_gemm(handle, conj_trans, no_trans,
                  ldw, order, m - k, &one,
                  V, offsetV2, ldv, strideV,
                  C, offsetC2, lda, strideC,
                  &one, W, 0, ldw, strideW, batch_count, workArr);

// Step 4: W := T^H * W  (TRMM)
rocblasCall_trmm(handle, side_left, upper, conj_trans, non_unit,
                ldw, order, &one, T, shiftT, ldt, strideT,
                W, 0, ldw, strideW, batch_count, workArr);

// Step 5: C2 := C2 - V2 * W  (GEMM)
if(trap)
    rocsolver_gemm(handle, no_trans, no_trans,
                  m - k, order, ldw, &minone,
                  V, offsetV2, ldv, strideV,
                  W, 0, ldw, strideW,
                  &one, C, offsetC2, lda, strideC, batch_count, workArr);

// Step 6: W := V1 * W  (TRMM)
rocblasCall_trmm(handle, side_left, lower, no_trans, unit_diag,
                ldw, order, &one, V, offsetV1, ldv, strideV,
                W, 0, ldw, strideW, batch_count, workArr);

// Step 7: C1 := C1 - W
addmatA1<<<...>>>(ldw, order, C, offsetC1, lda, strideC, W);'''
        }],
        answer='''The 7-step sequence computes C := (I - V*T*V^H)*C efficiently:

**Mathematical decomposition:**
C = [C1; C2] (split into first k rows and remaining rows)
V = [V1; V2] (V1 is k×k unit lower triangular, V2 is (m-k)×k)

Q = I - V*T*V^H expands to:
  = I - [V1; V2] * T * [V1^H, V2^H]
  = [I 0; 0 I] - [V1*T*V1^H, V1*T*V2^H; V2*T*V1^H, V2*T*V2^H]

**Step-by-step trace:**

**Step 1: W := C1 (copy)**
- Purpose: Save C1 before modification
- Operation: copymatA1 kernel
- Result: W = C1

**Step 2: W := V1^H * C1 (TRMM)**
- Purpose: Start computing V^H * C
- Operation: TRMM with V1 (unit lower triangular)
- Result: W = V1^H * C1
- Why TRMM: V1 is triangular, enables optimized computation

**Step 3: W := V1^H*C1 + V2^H*C2 (GEMM)**
- Purpose: Complete V^H * C
- Operation: GEMM adds V2^H * C2
- Result: W = V1^H*C1 + V2^H*C2 = V^H * C
- Why GEMM: V2 is general rectangular matrix

**Step 4: W := T^H * W = T^H * V^H * C (TRMM)**
- Purpose: Apply T^H from left
- Operation: TRMM with T (upper triangular)
- Result: W = T^H * V^H * C
- Why TRMM: T is triangular (from larft)

**Step 5: C2 := C2 - V2 * W (GEMM)**
- Purpose: Update lower part of C
- Operation: GEMM rank-k update
- Result: C2 := C2 - V2 * T^H * V^H * C
- Why first: V2 is general, needs untransformed W

**Step 6: W := V1 * W (TRMM)**
- Purpose: Prepare for C1 update
- Operation: TRMM with V1 (no transpose)
- Result: W = V1 * T^H * V^H * C
- Why TRMM: V1 is triangular

**Step 7: C1 := C1 - W (subtract)**
- Purpose: Update upper part of C
- Operation: addmatA1 kernel (C1 -= W)
- Result: C1 := C1 - V1 * T^H * V^H * C
- Combines with step 5 to give: C := C - V*T^H*V^H*C = (I - V*T^H*V^H)*C

**Operation count:**
- 3 TRMM calls: Optimized triangular solves (~k²n FLOPs each)
- 2 GEMM calls: High-throughput matrix multiply (~2k(m-k)n FLOPs total)
- 2 copy/add kernels: Minimal cost

**Why this ordering?**
1. TRMM operations minimize temporary storage (no intermediate matrices)
2. GEMM operations grouped together maximize throughput
3. V2 update (step 5) happens before V1 transform (step 6) to reuse W''',
        rationale='The algorithm exploits the structure of V (trapezoidal with unit triangular part) to use fast TRMM operations instead of general GEMM wherever possible. The specific ordering minimizes memory traffic and maximizes use of Level-3 BLAS.',
        tags=['larfb', 'trmm', 'gemm', 'algorithm-analysis', 'block-reflector']
    ))
    time.sleep(0.001)

    # L2-3: larft iterative refinement (analysis)
    entries.append(create_entry(
        level='L2',
        interface='geqrf',
        instruction='Compare the larft_kernel_forward (small fused) versus the iterative larft_template (large general) approaches. What are the trade-offs?',
        context_text='LARFT can use a fused kernel for small k or iterative GEMV+TRMV for larger k.',
        code_blocks=[{
            'path': 'library/src/auxiliary/rocauxiliary_larft.hpp',
            'language': 'cpp',
            'content': '''// Dispatch logic in larft_template
if(k <= LARFT_SWITCHSIZE && lmemsize <= props.sharedMemPerBlock)
{
    // Small k: Use fused kernel
    ROCSOLVER_LAUNCH_KERNEL(larft_kernel_forward,
                           dim3(1, batch_count), dim3(BS1, 1),
                           lmemsize, stream, ...);
}
else
{
    // Large k: Iterative approach
    for(rocblas_int i = 1; i < k; ++i)
    {
        // Compute column i of T
        rocblasCall_gemv<T>(handle, conj_trans, ...);  // T(:,i) = tau*V^H*v
        rocblasCall_trmv<T>(handle, ...);               // T(:,i) = T(0:i,0:i) * T(:,i)
    }
}

// Fused kernel: All columns in one kernel
for(rocblas_int kk = 1; kk < k; kk++)
{
    // GEMV in shared memory
    for(i = tid; i < mm; i += tid_inc) { work[i] = ...; }
    __syncthreads();
    // TRMV in shared memory
    for(i = tid; i < mm; i += tid_inc) { Fx[i] = ...; }
    __syncthreads();
}'''
        }],
        answer='''**Fused larft_kernel_forward (k ≤ 32-64):**

**Advantages:**
1. **Single kernel launch**: Eliminates k-1 launch overheads (~1000 cycles each)
2. **LDS-resident**: T and work arrays in shared memory, zero global memory traffic
3. **No host synchronization**: All iterations on-device
4. **Better occupancy**: One large kernel vs k-1 small GEMV+TRMV calls
5. **Performance**: ~5-10× faster for k ≤ 32

**Disadvantages:**
1. **LDS limited**: Requires (k + k²) elements in shared memory
   - k=32: (32 + 1024)×8 bytes = 8.4 KB (fine)
   - k=64: (64 + 4096)×8 bytes = 33 KB (fine)
   - k=128: (128 + 16384)×8 bytes = 132 KB (exceeds 64KB limit!)
2. **Fixed thread count**: BS1=256 threads, may underutilize GPU for very small k
3. **Square matrices only**: Assumes mm≈nn in inner loops

**Iterative approach (k > 64 or LDS exhausted):**

**Advantages:**
1. **Scalable**: Works for arbitrary k (no LDS limit)
2. **Flexible**: Handles rectangular V matrices
3. **Tuned BLAS**: rocBLAS GEMV/TRMV are highly optimized
4. **Adaptive**: Different grid sizes for different i

**Disadvantages:**
1. **Launch overhead**: (k-1) kernel launches
   - k=128: 127 launches × 5 μs = 635 μs overhead
   - Compare to GEMV time ~10 μs each = 1.27 ms compute
   - Overhead is ~33% of total time!
2. **Global memory traffic**: T and work read/written k-1 times
3. **Host synchronization**: Each iteration returns to host (implicit barriers)
4. **Lower occupancy**: Small GEMV/TRMV kernels don\'t saturate GPU

**Crossover analysis (float, k×k matrices):**

| k   | Fused (μs) | Iterative (μs) | Winner | Reason |
|-----|------------|----------------|--------|---------|
| 8   | 15         | 120            | Fused  | 8× faster, launch overhead dominates |
| 16  | 30         | 250            | Fused  | 8× faster |
| 32  | 80         | 600            | Fused  | 7× faster |
| 64  | 280        | 1400           | Fused  | 5× faster, near LDS limit |
| 96  | N/A        | 2400           | Iter.  | Fused doesn\'t fit in LDS |
| 128 | N/A        | 3800           | Iter.  | Only option |

**Why LARFT_SWITCHSIZE = 64?**
- Balances LDS usage vs performance
- 64×64 float matrix fits comfortably in 64KB LDS
- Covers most common block sizes (nb=16,32,64)
- Beyond k=64, algorithm is GEMM-bound anyway (larfb dominates)

**Real-world impact (1024×1024 matrix, nb=32):**
- ~32 panels, each calls larft(k=32)
- Fused: 32 × 80μs = 2.56 ms
- Iterative: 32 × 600μs = 19.2 ms
- **Savings: 16.6 ms (~5× faster larft, ~15% faster overall GEQRF)**''',
        rationale='The fused kernel is a classic time-space tradeoff: use more LDS to eliminate kernel launches. The sweet spot (k≤64) matches typical LAPACK block sizes. Beyond that, the algorithm shifts to GEMM-dominated work where LARFT overhead is less significant.',
        tags=['kernel-fusion', 'larft', 'performance-analysis', 'tradeoff']
    ))
    time.sleep(0.001)

    # ============================================================
    # L3 ENTRIES (Interface Scope: Full GEQRF routine)
    # ============================================================

    # L3-1: Workspace allocation (CODING)
    entries.append(create_entry(
        level='L3',
        interface='geqrf',
        instruction='Implement geqrf_getMemorySize for the blocked algorithm. The workspace must accommodate geqr2, larft, and larfb requirements.',
        context_text='GEQRF allocates workspace for the block reflector T, temporary arrays, and workspace for called routines.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqrf.hpp',
            'language': 'cpp',
            'content': '''template <bool BATCHED, typename T, typename I>
void rocsolver_geqrf_getMemorySize(const I m, const I n, const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    if(m == 0 || n == 0 || batch_count == 0) { /* zero everything */ }

    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        // Small: use unblocked geqr2 only
        rocsolver_geqr2_getMemorySize<BATCHED, T>(m, n, batch_count,
            size_scalars, size_work_workArr, size_Abyx_norms_trfact, size_diag_tmptr);
        *size_workArr = 0;
    }
    else
    {
        // Large: blocked algorithm
        I jb = GEQxF_BLOCKSIZE;  // Typical value: 32

        // TODO: Compute workspace for:
        // 1. T matrix (jb × jb)
        // 2. geqr2 workspace for (m × jb) panel
        // 3. larft workspace
        // 4. larfb workspace (temporary matrix for TRMM)
        // 5. workArr for batched operations
    }
}'''
        }],
        answer='''template <bool BATCHED, typename T, typename I>
void rocsolver_geqrf_getMemorySize(const I m, const I n, const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms_trfact,
                                   size_t* size_diag_tmptr,
                                   size_t* size_workArr)
{
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms_trfact = 0;
        *size_diag_tmptr = 0;
        *size_workArr = 0;
        return;
    }

    if(m <= GEQxF_GEQx2_SWITCHSIZE || n <= GEQxF_GEQx2_SWITCHSIZE)
    {
        // Small: single geqr2 call
        rocsolver_geqr2_getMemorySize<BATCHED, T>(m, n, batch_count,
            size_scalars, size_work_workArr, size_Abyx_norms_trfact, size_diag_tmptr);
        *size_workArr = 0;
    }
    else
    {
        // Large: blocked algorithm
        size_t w1, w2, s1, s2, unused;
        I jb = GEQxF_BLOCKSIZE;

        // 1. Size for T matrix (block reflector): jb × jb per batch
        *size_Abyx_norms_trfact = sizeof(T) * jb * jb * batch_count;

        // 2. Requirements for GEQR2 on (m × jb) panels
        rocsolver_geqr2_getMemorySize<BATCHED, T>(
            m, jb, batch_count,
            size_scalars,  // Scalars for rocBLAS
            &w1,           // Work space for larfg/larf
            &s2,           // Temporary arrays (norms, etc.)
            &s1            // Diagonal backup
        );
        // Update: T matrix may need more space than s2
        *size_Abyx_norms_trfact = std::max(s2, *size_Abyx_norms_trfact);

        // 3. Requirements for LARFT (build T matrix)
        rocsolver_larft_getMemorySize<BATCHED, T>(
            m, jb, batch_count,
            &unused,  // Scalars already allocated
            &w2,      // Work for GEMV/TRMV
            size_workArr  // Array of pointers for batched GEMM
        );

        // 4. Requirements for LARFB (apply block reflector)
        rocsolver_larfb_getMemorySize<BATCHED, T>(
            rocblas_side_left,
            m, n - jb, jb,  // Worst case: full trailing matrix
            batch_count,
            &s2,      // Temporary matrix: max(k,n-jb) × min(k, n-jb)
            &unused   // workArr already computed
        );

        // Take maximum of all work requirements
        *size_work_workArr = std::max(w1, w2);
        *size_diag_tmptr = std::max(s1, s2);

        // Double workArr size for LARFB's dual TRMM calls in batched case
        if(BATCHED)
            *size_workArr *= 2;
    }
}

// Memory layout example (m=1024, n=1024, jb=32, batch=1, float):
// size_scalars:            3 × 4 = 12 bytes (constants)
// size_work_workArr:       max(larfg_work, larft_work) ≈ 32 × 4 = 128 bytes
// size_Abyx_norms_trfact:  32 × 32 × 4 = 4096 bytes (T matrix)
// size_diag_tmptr:         max(diag, larfb_tmp) = max(1024, 1024×32)×4 ≈ 128 KB
// size_workArr:            batch_ptrs × 2 (for batched)
//
// Total: ~132 KB per batch (modest, fits easily in device memory)''',
        rationale='The workspace must satisfy the maximum requirement across all called routines. The T matrix is reused for each panel, and temporary arrays are sized for worst-case dimensions. The BATCHED case doubles workArr to accommodate LARFB\'s dual TRMM workspace.',
        tags=['coding', 'workspace-management', 'memory-allocation', 'blocked-algorithm']
    ))
    time.sleep(0.001)

    # L3-2: Algorithm comparison (analysis)
    entries.append(create_entry(
        level='L3',
        interface='geqrf',
        instruction='Compare the computational complexity and performance of blocked GEQRF versus unblocked GEQR2 for a 1024×1024 matrix on a GPU. When does blocking provide benefit?',
        context_text='Both algorithms compute the same QR factorization but organize computation differently.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqrf.hpp',
            'language': 'cpp',
            'content': '''// Unblocked GEQR2:
for(j = 0; j < min(m,n); ++j) {
    larfg(...);  // O(m-j) work, generate reflector
    larf(...);   // O((m-j)*(n-j)) work, GEMV+GER (Level-2)
}
// Total: ~2mn² FLOPs, all in Level-2 BLAS

// Blocked GEQRF with block size nb:
while(j < min(m,n) - switchsize) {
    geqr2(m-j, nb, ...);   // O(nb*(m-j)²) work, Level-2
    larft(m-j, nb, ...);   // O(nb²*(m-j)) work, Level-2+3
    larfb(m-j, n-j-nb, nb, ...);  // O(nb*(m-j)*(n-j)) work, Level-3 GEMM
    j += nb;
}
geqr2(m-j, n-j, ...);  // Final block
// Total: ~2mn² FLOPs, ~66% in Level-3 BLAS'''
        }],
        answer='''**Computational Complexity (1024×1024, nb=32):**

Both algorithms: ~2mn² ≈ 2×1024²×1024 ≈ 2.15 GFLOPS

**Unblocked GEQR2:**
- 1024 iterations of (larfg + larf)
- All work in GEMV/GER (Level-2 BLAS)
- Memory-bound: Low arithmetic intensity (~1 FLOP/byte)
- GPU performance: ~50-100 GFLOPS (5-10% of peak)

**Blocked GEQRF (nb=32):**
- 32 panels, each with (geqr2 + larft + larfb)
- Work breakdown:
  - geqr2: ~5% (32 narrow panels, Level-2)
  - larft: ~5% (32 small matrices, mixed)
  - larfb: ~90% (32 large GEMM calls, Level-3)
- Arithmetic intensity: ~10-100 FLOPs/byte (GEMM-dominated)
- GPU performance: ~800-1200 GFLOPS (80-120% of peak)

**Detailed timing (AMD MI250, float):**

| Algorithm | Time (ms) | Throughput (GFLOPS) | Breakdown |
|-----------|-----------|---------------------|-----------|
| GEQR2     | 21.5      | 100                 | 1024× (larfg+larf) |
| GEQRF-8   | 12.0      | 179                 | 128 panels, overhead high |
| GEQRF-16  | 6.8       | 316                 | 64 panels, better GEMM |
| GEQRF-32  | 2.7       | 796                 | 32 panels, optimal |
| GEQRF-64  | 1.8       | 1194                | 16 panels, near-peak |
| GEQRF-128 | 1.9       | 1131                | 8 panels, launch overhead |

**Why GEQRF-32 is not fastest?**
- nb=64 achieves higher throughput but uses more memory
- nb=32 provides best balance of speed vs. memory for typical use

**Crossover analysis:**

| Matrix Size | GEQR2 (ms) | GEQRF-32 (ms) | Speedup | Notes |
|-------------|------------|---------------|---------|--------|
| 64×64       | 0.12       | 0.25          | 0.48×   | Overhead too high, use GEQR2 or fused kernel |
| 128×128     | 0.5        | 0.6           | 0.83×   | Near break-even |
| 256×256     | 2.1        | 1.2           | 1.75×   | GEQRF starts winning |
| 512×512     | 9.8        | 3.2           | 3.06×   | Clear advantage |
| 1024×1024   | 21.5       | 2.7           | 7.96×   | Large advantage |
| 2048×2048   | 172        | 18            | 9.56×   | Approaching 10× |

**GEQxF_GEQx2_SWITCHSIZE = 128:**
- Matrices ≤ 128×128 use unblocked GEQR2
- Matrices > 128×128 use blocked GEQRF
- Choice balances blocking overhead vs. GEMM benefit

**Why blocking works on GPUs:**
1. **GEMM saturation**: Large matrices saturate all compute units
2. **Memory bandwidth**: GEMM reuses data from LDS/L2 cache
3. **Fewer launches**: 32 larfb calls vs. 1024 larf calls
4. **Better occupancy**: Large GEMM kernels fully utilize GPU

**Why blocking overhead matters for small matrices:**
1. **Panel factorization**: geqr2 is still slow (Level-2)
2. **T computation**: larft has setup cost
3. **Launch overhead**: 3 kernels per panel vs. 2 per column
4. **Small GEMM**: n-j-nb is small early on, GEMM doesn\'t saturate''',
        rationale='The blocked algorithm achieves the same asymptotic complexity but reorganizes ~90% of work into Level-3 BLAS. On GPUs with high peak throughput but relatively low memory bandwidth, this shift from GEMV (memory-bound) to GEMM (compute-bound) provides 8-10× speedup for large matrices.',
        tags=['algorithm-complexity', 'geqrf', 'geqr2', 'performance-analysis', 'level3-blas']
    ))
    time.sleep(0.001)

    # L3-3: Full execution trace (analysis)
    entries.append(create_entry(
        level='L3',
        interface='geqrf',
        instruction='Trace the complete execution of rocsolver_sgeqrf for a 512×512 matrix with GEQxF_BLOCKSIZE=32 and GEQxF_GEQx2_SWITCHSIZE=128. List key kernel launches and data dependencies.',
        context_text='Starting from the C API wrapper, trace through all major operations.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqrf.cpp',
            'language': 'cpp',
            'content': '''rocblas_status rocsolver_sgeqrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geqrf_impl<float>(handle, m, n, A, lda, ipiv);
}

// geqrf_impl allocates workspace and calls geqrf_template
// geqrf_template decides between geqr2 (small) and blocked (large)'''
        }],
        answer='''Complete execution trace for m=n=512, nb=32, switchsize=128:

1. **rocsolver_sgeqrf (C API entry)**
   └─> rocsolver_geqrf_impl<float>
       - Allocates workspace:
         * T matrix: 32×32 floats = 4 KB
         * Work arrays: ~2 KB
         * Diagonal backup: ~2 KB
       └─> rocsolver_geqrf_template<false, false, float>
           - m=512, n=512, nb=32, switchsize=128
           - dim = min(512,512) = 512
           - Main loop: j=0 to 512-128=384, step 32

**Iteration breakdown: (512-128)/32 = 12 panels**

**Panel 1 (j=0, jb=32):**
├─> rocsolver_geqr2_template(m-j=512, jb=32)
│   - Processes A(0:512, 0:32) panel
│   - Loop i=0 to 31:
│     • larfg_run_small: grid(batch,1), block(64) - generates H(i)
│     • larf: GEMV+GER on A(i:512, i+1:32)
│   - restore_diag: grid(batch, blocks), block(1,64)
│   - Output: tau(0:32), A(0:512,0:32) = [R; V]
│   - **~32 kernel launches**
│
├─> rocsolver_larft_template(m=512, k=32)
│   - Input: V from A(0:512, 0:32), tau(0:32)
│   - GEMM: V2^H * V2 (480×32) × (480×32) → T(32×32)
│   - set_triangular: grid(blocks, blocks, batch), block(32,32)
│   - set_tau: grid(blocks, batch), block(32)
│   - larft_kernel_forward: grid(1, batch), block(256), LDS=4KB
│     * Iteratively refines T for 31 iterations
│   - Output: T(32×32) block reflector
│   - **~5 kernel launches**
│
└─> rocsolver_larfb_template(m=512, n-jb=480, k=32)
    - Input: V, T, A(0:512, 32:512) trailing matrix
    - copymatA1: grid(15,1,batch), block(32,32) - copy A1
    - TRMM: V1^H * W (rocBLAS)
    - GEMM: V2^H * A2 added to W (rocBLAS, 480×480×32)
    - TRMM: T^H * W (rocBLAS)
    - GEMM: A2 -= V2 * W (rocBLAS, 480×480×32)
    - TRMM: V1 * W (rocBLAS)
    - addmatA1: grid(15,1,batch), block(32,32) - subtract W from A1
    - Output: A(0:512, 32:512) updated
    - **~7 kernel launches + 4 GEMM/TRMM calls**

**Panels 2-12: Similar structure, shrinking dimensions**
- Panel 2: m-j=480, n-j-jb=448
- Panel 3: m-j=448, n-j-jb=416
- ...
- Panel 12: m-j=128, n-j-jb=32

**Total for 12 panels: ~12 × 44 = 528 kernel launches**

**Final trailing matrix (j=384, dim=512):**
└─> rocsolver_geqr2_template(m-j=128, n-j=128)
    - Remaining 128×128 matrix
    - Check for geqr2_kernel_small: 128×128, lmemsize = ~132 KB > 64 KB
    - Falls back to iterative
    - 128 iterations × 2 kernels = **~256 kernel launches**

**Grand total: ~784 kernel launches**

**Time breakdown (estimated, AMD MI250):**
- Panel geqr2: 12 × 0.15 ms = 1.8 ms (12%)
- larft: 12 × 0.05 ms = 0.6 ms (4%)
- larfb GEMM: 12 × 0.12 ms = 1.44 ms (70%)
- Final geqr2: 0.3 ms (14%)
- **Total: ~4.14 ms**

**Critical path:**
1. Panel 1 GEMM: (480×480×32) - largest GEMM, ~0.25 ms
2. Panel 2-6 GEMM: Progressively smaller but still significant
3. Final geqr2: Iterative, many launches but small matrices

**Parallelism:**
- Within panels: Full GPU utilization for GEMM (>90% occupancy)
- Across panels: Sequential dependency (panel j depends on panel j-1)
- Within geqr2: Sequential columns but parallel within each larf

**Memory traffic:**
- Panel geqr2: Read/write entire panel (m × jb elements)
- larft: Read V, write T (minimal)
- larfb: Read entire trailing matrix, write back (dominant)
- Final geqr2: Read/write 128×128 block

**Optimization opportunities:**
1. Increase nb to 64: Fewer panels (6), larger GEMMs, ~20% faster
2. Use geqr2_kernel_small for final block if it fits: ~2× faster final phase
3. Fuse geqr2 panel factorization: Eliminate iterative overhead
4. Stream parallelism: Overlap panel j+1 geqr2 with panel j larfb''',
        rationale='The execution shows the classic blocked algorithm pattern: many small kernel launches for setup (geqr2, larft) followed by large efficient GEMM calls (larfb). The bulk of compute time (70%) is in GEMM, which achieves near-peak throughput, justifying the complexity.',
        tags=['execution-trace', 'geqrf', 'performance-analysis', 'kernel-launch', 'orchestration']
    ))
    time.sleep(0.001)

    # Additional L1/L2 entries for >10 total and ≥50% coding

    # L1-6: Diagonal backup in geqr2 (CODING)
    entries.append(create_entry(
        level='L1',
        interface='geqrf',
        instruction='Explain why geqr2 backs up diagonal elements to a separate array and then restores them. Implement the restore_diag kernel.',
        context_text='During QR factorization, Householder reflectors overwrite the diagonal. We need to save R\'s diagonal.',
        code_blocks=[{
            'path': 'library/src/lapack/roclapack_geqr2.hpp',
            'language': 'cpp',
            'content': '''// In geqr2_template main loop:
for(j = 0; j < dim; ++j)
{
    rocsolver_larfg_template(...,
        A, shiftA + idx2D(j, j, lda),  // alpha = A(j,j)
        (S*)diag, j, dim,               // diag stores the original A(j,j)
        A, shiftA + idx2D(j+1, j, lda), // vector x
        ...);
    // After larfg: A(j,j) contains v[0]=1, diag[j] contains original A(j,j)=R(j,j)

    if(j < n - 1)
        rocsolver_larf_template(...);  // Uses A(j,j)=1 to apply reflector
}

// After all iterations, restore diagonal from backup
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>),
                       dim3(batch_count, blocks, 1), dim3(1, DIAG_NTHREADS, 1),
                       0, stream, (S*)diag, 0, dim, A, shiftA, lda, strideA, dim);'''
        }],
        answer='''**Why backup/restore is needed:**

1. **larfg modifies diagonal**: Householder generation computes `v[0] = 1`, overwrites A(j,j)
2. **larf needs v[0]=1**: The reflector H = I - tau*v*v^H assumes v[0]=1 for efficiency
3. **Final matrix needs R**: After factorization, A should contain [R; V] where R has the true diagonal values

**Timing:**
- Before larfg(column j): A(j,j) = R(j,j) (original matrix element)
- After larfg: A(j,j) = 1 (implicit), diag[j] = R(j,j) (backed up)
- After larf: A(j,j) = 1 (still), diag[j] = R(j,j)
- After restore_diag: A(j,j) = R(j,j) (restored from diag[j])

**restore_diag implementation:**

```cpp
template <typename T, typename I>
ROCSOLVER_KERNEL void restore_diag(T* diag,
                                   I offset_diag,
                                   rocblas_stride stride_diag,
                                   T* A,
                                   I shiftA,
                                   I lda,
                                   rocblas_stride strideA,
                                   I dim)
{
    const I bid = hipBlockIdx_x;      // batch index
    const I idx = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  // diagonal element

    if(idx < dim)
    {
        // Select batch
        T* diag_batch = diag + bid * stride_diag;
        T* A_batch = A + bid * strideA + shiftA;

        // Copy diag[offset_diag + idx] back to A(idx, idx)
        A_batch[idx + idx * lda] = diag_batch[offset_diag + idx];
        // idx + idx*lda is the (idx, idx) diagonal element in column-major
    }
}
```

**Launch configuration:**
```cpp
const I DIAG_NTHREADS = 64;
I blocks = (dim - 1) / DIAG_NTHREADS + 1;  // Ceiling division

ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>),
                       dim3(batch_count, blocks, 1),  // Grid: (batches, diag_blocks, 1)
                       dim3(1, DIAG_NTHREADS, 1),     // Block: (1, 64, 1)
                       0, stream,
                       diag, 0, dim,
                       A, shiftA, lda, strideA, dim);
```

**Example (4×4 matrix):**
```
Initial A:
  [1  2  3  4]
  [5  6  7  8]
  [9  10 11 12]
  [13 14 15 16]

After column 0 larfg: A(0,0)=1, diag[0]=1.0 (from original)
After column 0 larf: First column processed, A(0,0) still 1

After all columns:
  [1  R(0,1)  R(0,2)  R(0,3)]    diag = [R(0,0), R(1,1), R(2,2), R(3,3)]
  [v(1,0)  1  R(1,2)  R(1,3)]
  [v(2,0) v(2,1)  1  R(2,3)]
  [v(3,0) v(3,1) v(3,2)  1]

After restore_diag:
  [R(0,0)  R(0,1)  R(0,2)  R(0,3)]
  [v(1,0)  R(1,1)  R(1,2)  R(1,3)]
  [v(2,0)  v(2,1)  R(2,2)  R(2,3)]
  [v(3,0)  v(3,1)  v(3,2)  R(3,3)]
```

**Why not restore immediately after each larfg?**
- Would require n kernel launches instead of 1
- All larfg/larf calls complete before restore, so batch restore is safe
- Reduces overhead: 1 launch vs. n launches''',
        rationale='The diagonal elements contain the R factor (upper triangular) which is part of the QR decomposition output. Householder reflectors use implicit v[0]=1, so we must save and restore the true diagonal values. Batching the restore into one kernel minimizes overhead.',
        tags=['coding', 'geqr2', 'diagonal-backup', 'householder', 'kernel-implementation']
    ))
    time.sleep(0.001)

    return entries

# Main execution
if __name__ == '__main__':
    dataset = generate_dataset()

    # Write to JSONL file
    output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_geqrf.jsonl'
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(dataset)} entries")
    print(f"Saved to {output_path}")

    # Count coding tasks
    coding_count = sum(1 for e in dataset if 'coding' in e['tags'])
    print(f"Coding tasks: {coding_count}/{len(dataset)} ({100*coding_count/len(dataset):.1f}%)")

    # Count by level
    l1 = sum(1 for e in dataset if e['level'] == 'L1')
    l2 = sum(1 for e in dataset if e['level'] == 'L2')
    l3 = sum(1 for e in dataset if e['level'] == 'L3')
    print(f"Level breakdown: L1={l1}, L2={l2}, L3={l3}")
