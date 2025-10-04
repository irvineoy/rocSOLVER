#!/usr/bin/env python3
import json
import time

# Get base timestamp
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYTRD/HETRD blocked algorithm
entries.append({
    "id": str(base_ts + 1),
    "level": "L3",
    "interface": "sytrd_hetrd",
    "query": "Explain the complete blocked tridiagonalization algorithm used in SYTRD/HETRD. How does it achieve better performance than the unblocked SYTD2/HETD2?",
    "answer": "SYTRD/HETRD uses a blocked algorithm that processes k columns at a time (where k = xxTRD_BLOCKSIZE) to leverage Level 3 BLAS operations. The algorithm has three main phases:\n\n**1. Matrix preparation**: Convert triangular A to full by copying the unreferenced triangle, enabling Level 3 BLAS\n\n**2. Main blocked loop** (for lower triangular, j=0 to n-kk by k):\n   - Use LATRD to reduce k columns (panel factorization)\n   - Update trailing matrix: A := A - V*W' - W*V' using GEMM (Level 3)\n   - This is much faster than SYTD2's Level 2 operations (SYR2)\n\n**3. Tail with SYTD2**: When remaining size < kk (switchsize), use unblocked SYTD2\n\n**Performance advantages over SYTD2**:\n- Level 3 BLAS (GEMM): O(n³) with better cache utilization vs Level 2 BLAS (SYR2)\n- Blocked updates amortize memory traffic\n- LATRD processes k columns together, improving vectorization\n- For large matrices (n > 128), this provides 2-5x speedup\n\nThe algorithm uses xxTRD_xxTD2_SWITCHSIZE to determine when unblocked is more efficient.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_lower)
{
    // reduce the lower part of A
    // main loop running forwards (for each block of columns)
    // when the unreduced part is not large enough, switch to unblocked algorithm
    j = 0;
    while(j < n - kk)
    {
        // reduce columns j:j+k-1
        rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A,
                                             shiftA + idx2D(j, j, lda), lda, strideA, (E + j),
                                             strideE, (tau + j), strideP, tmptau_W, 0, ldw,
                                             strideW, batch_count, scalars, work, norms, workArr);

        // update trailing matrix
        // A = A - V*W' - W*V'
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                       n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
                       strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
                       shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                       n - j - k, n - j - k, k, &minone, tmptau_W, idx2D(k, 0, ldw), ldw,
                       strideW, A, shiftA + idx2D(j + k, j, lda), lda, strideA, &one, A,
                       shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);

        j += k;
    }

    // reduce last columns of A
    rocsolver_sytd2_hetd2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                      strideA, (D + j), strideD, (E + j), strideE, (tau + j),
                                      strideP, batch_count, scalars, work, norms, tmptau_W,
                                      workArr);
}"""
    }]
})

# Entry 2: L2 - LATRD panel factorization
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "sytrd_hetrd",
    "query": "What is LATRD and how does it perform panel factorization in the blocked SYTRD algorithm?",
    "answer": "LATRD (Latent Tridiagonal Reduction) reduces k columns of a symmetric/Hermitian matrix while computing the workspace W needed for delayed updates. For each column i in the panel (i=0 to k-1):\n\n1. **Generate Householder reflector**: Annihilate elements below/above diagonal using LARFG\n2. **Compute intermediate w**: w = A*v using SYMV/HEMV\n3. **Update W and A within panel**:\n   - W := W - 0.5*τ*(v'*w)*v (correction for symmetry)\n   - Apply reflector to remaining panel columns: A := A - v*w' - w*v'\n4. **Store v and τ**: Householder vectors in A, scalars in tau\n\nThe key innovation is that LATRD only updates within the panel. The trailing matrix update is deferred and done once using Level 3 GEMM:\n\nA_trailing := A_trailing - V*W' - W*V'\n\nwhere V contains all k Householder vectors and W contains all k workspace vectors. This amortizes the O(n²k) trailing update cost and enables Level 3 BLAS.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """// reduce columns j:j+k-1
rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A,
                                     shiftA + idx2D(j, j, lda), lda, strideA, (E + j),
                                     strideE, (tau + j), strideP, tmptau_W, 0, ldw,
                                     strideW, batch_count, scalars, work, norms, workArr);

// update trailing matrix (deferred from LATRD)
// A = A - V*W' - W*V'
rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
               n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
               strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
               shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);"""
    }]
})

# Entry 3: L1 - Block size parameters
entries.append({
    "id": str(base_ts + 3),
    "level": "L1",
    "interface": "sytrd_hetrd",
    "query": "What are xxTRD_BLOCKSIZE and xxTRD_xxTD2_SWITCHSIZE? How do they control the blocked algorithm?",
    "answer": "**xxTRD_BLOCKSIZE (k)**: Number of columns processed together in each panel factorization. Typical value is 32 or 64. Larger k means:\n- More Level 3 BLAS work (good)\n- More panel factorization overhead (bad)\n- More workspace (k*n elements for W)\n\n**xxTRD_xxTD2_SWITCHSIZE (kk)**: Threshold to switch from blocked to unblocked algorithm. When the remaining unreduced matrix size < kk, SYTD2 is called instead of continuing with LATRD+GEMM. Typical value is 128.\n\nRationale: For small matrices, the overhead of blocked algorithms (workspace allocation, multiple kernel launches) outweighs the Level 3 BLAS benefit. SYTD2's small kernel optimization is faster for the tail.\n\nThe algorithm checks: if(n <= kk) immediately use SYTD2; else use blocked SYTRD until size < kk.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """rocblas_int k = xxTRD_BLOCKSIZE;
rocblas_int kk = xxTRD_xxTD2_SWITCHSIZE;

// if the matrix is too small, use the unblocked variant of the algorithm
if(n <= kk)
    return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                          E, strideE, tau, strideP, batch_count, scalars,
                                          work_Acpy, norms, tmptau_W, workArr);

// ... blocked algorithm ...
j = 0;
while(j < n - kk)
{
    // blocked panel factorization
    rocsolver_latrd_forsytrd_template<T>(...);
    // Level 3 BLAS update
    rocsolver_gemm(...);
    j += k;
}

// reduce last columns with unblocked algorithm
rocsolver_sytd2_hetd2_template<T>(handle, uplo, n - j, ...);"""
    }]
})

# Entry 4: L2 - Matrix preparation and recovery
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "sytrd_hetrd",
    "query": "Why does SYTRD copy the unreferenced triangle of A, and what is the recover_A parameter?",
    "answer": "SYTRD makes A temporarily non-symmetric/non-Hermitian to enable Level 3 BLAS operations (GEMM instead of SYMM). The algorithm:\n\n1. **Copy unreferenced triangle**: `copy_trans_mat` copies upper to lower (or vice versa) with conjugate transpose\n2. **Perform blocked factorization**: Now both triangles are available for GEMM\n3. **Optionally recover**: If recover_A=true, restore the unreferenced triangle from the saved copy\n\n**Why this works**: The blocked updates (A - V*W' - W*V') are symmetric, so both triangles remain mathematically consistent. However, GPU GEMM doesn't exploit symmetry like SYMM, so we use the full matrix.\n\n**recover_A parameter**: \n- true (default): Save and restore unreferenced triangle (extra n²/2 storage)\n- false: Leave unreferenced triangle modified (saves memory, used when caller doesn't need it)\n\nThis is a classic space-time tradeoff. The copy costs O(n²) memory but enables faster O(n³) GEMM operations.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """rocblas_fill uplo2 = (uplo == rocblas_fill_upper) ? rocblas_fill_lower : rocblas_fill_upper;

if(recover_A)
{
    Acpy = work_Acpy;
    work = Acpy + n * n * batch_count;

    // Save unreferenced triangle to Acpy
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                            0, stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                            no_mask{}, uplo2, rocblas_diagonal_unit);
}
else
    work = work_Acpy;

// Copy unreferenced triangle to make A full (for GEMM)
ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                        dim3(BS2, BS2, 1), 0, stream, rocblas_operation_conjugate_transpose, n,
                        n, A, shiftA, lda, strideA, A, shiftA, lda, strideA, no_mask{}, uplo,
                        rocblas_diagonal_unit);

// ... perform blocked factorization ...

// recover non-referenced part of A if necessary
if(recover_A)
{
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                            0, stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                            no_mask{}, uplo2, rocblas_diagonal_unit);
}"""
    }]
})

# Entry 5: L1 - Upper vs lower processing
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "sytrd_hetrd",
    "query": "How does the upper triangular case differ from the lower triangular case in SYTRD?",
    "answer": "The main differences are:\n\n**Lower triangular (uplo = lower)**:\n- Loop runs forwards: j=0 to n-kk by k\n- Reduces columns j:j+k-1 at each iteration\n- Updates trailing matrix A(j+k:n, j+k:n)\n- Tail factorization at end: SYTD2 on A(j:n, j:n)\n\n**Upper triangular (uplo = upper)**:\n- Loop runs backwards: j=n-k to upkk by -k\n- Reduces columns j:j+k-1 at each iteration\n- Updates leading matrix A(0:j, 0:j)\n- Tail factorization at beginning: SYTD2 on A(0:upkk, 0:upkk)\n- upkk calculation ensures alignment: upkk = n - ((n-kk+k-1)/k)*k\n\nBoth cases use the same LATRD and GEMM operations, just with different matrix regions. The backward iteration for upper triangular ensures Householder vectors are stored correctly above the diagonal.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """else  // uplo == rocblas_fill_upper
{
    // reduce the upper part of A
    // main loop running backwards (for each block of columns)
    // when the unreduced part is not large enough, switch to unblocked algorithm
    j = n - k;
    rocblas_int upkk = n - ((n - kk + k - 1) / k) * k;
    while(j >= upkk)
    {
        // reduce columns j:j+k-1
        rocsolver_latrd_forsytrd_template<T>(handle, uplo, j + k, k, A, shiftA, lda, strideA, E,
                                             strideE, tau, strideP, tmptau_W, 0, ldw, strideW,
                                             batch_count, scalars, work, norms, workArr);

        // update trailing matrix
        // A = A - V*W' - W*V'
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                       j, k, &minone, A, shiftA + idx2D(0, j, lda), lda, strideA, tmptau_W, 0,
                       ldw, strideW, &one, A, shiftA, lda, strideA, batch_count, workArr);
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                       j, k, &minone, tmptau_W, 0, ldw, strideW, A, shiftA + idx2D(0, j, lda),
                       lda, strideA, &one, A, shiftA, lda, strideA, batch_count, workArr);

        j -= k;
    }

    // reduce first columns of A
    rocsolver_sytd2_hetd2_template<T>(handle, uplo, upkk, A, shiftA, lda, strideA, D, strideD,
                                      E, strideE, tau, strideP, batch_count, scalars, work,
                                      norms, tmptau_W, workArr);
}"""
    }]
})

# Entry 6: L2 - Workspace calculation (coding)
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "sytrd_hetrd",
    "query": "Implement the workspace size calculation for SYTRD/HETRD. How does it differ from SYTD2?",
    "answer": "SYTRD requires additional workspace for the blocked algorithm:\n\n1. **s1 (tmptau_W)**: Matrix W for delayed updates: n × xxTRD_BLOCKSIZE × sizeof(T) × batch_count\n2. **w1, n1**: LATRD workspace (includes LARFG, SYMV requirements)\n3. **s2, w2, n2**: SYTD2 workspace (for tail factorization)\n4. **recover_A buffer**: n × n × sizeof(T) × batch_count (if recover_A=true)\n\nThe key difference from SYTD2:\n- SYTRD needs O(n*k) workspace for W matrix\n- SYTRD needs O(n²) for A copy (if recovering)\n- Final workspace is max of blocked and unblocked requirements\n\nTotal additional memory: O(n*k + n²) vs SYTD2's O(n)",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """template <bool BATCHED, typename T>
void rocsolver_sytrd_hetrd_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau_W,
                                         size_t* size_workArr,
                                         bool recover_A = true)
{
    *size_scalars = 0;
    *size_work = 0;
    *size_norms = 0;
    *size_tmptau_W = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
        return;

    size_t s1 = 0, s2 = 0;
    size_t w1 = 0, w2 = 0;
    size_t n1 = 0, n2 = 0;

    // extra requirements to call SYTD2/HETD2
    rocsolver_sytd2_hetd2_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w2, &n2, &s2,
                                                    size_workArr);

    if(n > xxTRD_xxTD2_SWITCHSIZE)
    {
        // size required to store temporary matrix W
        s1 = n * xxTRD_BLOCKSIZE;
        s1 *= sizeof(T) * batch_count;

        // extra requirements to call latrd_forsytrd
        rocsolver_latrd_forsytrd_getMemorySize<BATCHED, T>(n, xxTRD_BLOCKSIZE, batch_count,
                                                           size_scalars, &w1, &n1, size_workArr);
    }

    *size_tmptau_W = std::max(s1, s2);
    *size_work = std::max(w1, w2);
    *size_norms = std::max(n1, n2);

    // when recovering the non-referenced part of A is necessary,
    // add the required buffer size to hold a copy
    if(recover_A)
        *size_work += sizeof(T) * n * n * batch_count;
}"""
    }]
})

# Entry 7: L3 - GEMM-based trailing update (coding)
entries.append({
    "id": str(base_ts + 7),
    "level": "L3",
    "interface": "sytrd_hetrd",
    "query": "Show how the trailing matrix update A := A - V*W' - W*V' is implemented using GEMM. Why are two GEMM calls needed?",
    "answer": "The symmetric rank-2k update A := A - V*W' - W*V' is split into two GEMM calls because:\n\n1. **First GEMM**: A := A - V*W^H (W conjugate transpose)\n   - Computes V*W^H and subtracts from A\n   - V is n-j-k × k (Householder vectors)\n   - W is n-j-k × k (workspace vectors)\n   \n2. **Second GEMM**: A := A - W*V^H\n   - Computes W*V^H and subtracts from A\n   - This is the symmetric counterpart\n\nTwo calls are needed because GEMM doesn't have a symmetric rank-2k update primitive. Each GEMM is O(n²k) with Level 3 cache behavior.\n\nThe key parameters:\n- m = n-j-k, n = n-j-k, k = xxTRD_BLOCKSIZE\n- alpha = -1, beta = 1 (subtract from existing A)\n- V starts at A(j+k, j), W at tmptau_W(k, 0)\n- Result updates A(j+k, j+k)\n\nThis is much faster than k separate SYR2 calls (Level 2 BLAS).",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """// update trailing matrix
// A = A - V*W' - W*V'
const T minone = T(-1);
const T one = T(1);

// First GEMM: A := A - V*W^H
rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
               n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
               strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
               shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);

// Second GEMM: A := A - W*V^H  
rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
               n - j - k, n - j - k, k, &minone, tmptau_W, idx2D(k, 0, ldw), ldw,
               strideW, A, shiftA + idx2D(j + k, j, lda), lda, strideA, &one, A,
               shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);"""
    }]
})

# Entry 8: L1 - latrd_dot_scale_axpy kernel
entries.append({
    "id": str(base_ts + 8),
    "level": "L1",
    "interface": "sytrd_hetrd",
    "query": "What is the latrd_dot_scale_axpy kernel used for in LATRD panel factorization?",
    "answer": "latrd_dot_scale_axpy computes the critical correction: W := W - (1/2)*τ*(v'*w)*v. This kernel:\n\n1. **Dot product**: Computes alpha = v'*w using warp-level reduction\n2. **Scale**: Computes beta = -0.5 * tau * alpha\n3. **AXPY**: W := W + beta*v (which is W - 0.5*τ*v'*w*v)\n\nThis correction ensures the rank-2 update maintains symmetry. Without it, A - v*w' - w*v' would not be symmetric/Hermitian.\n\nOptimizations:\n- Warp shuffle reductions for dot product\n- Shared memory caching of A and W for small vectors (< MAX_THDS)\n- Single kernel fuses dot, scale, and axpy operations\n\nThis is the same kernel used in SYTD2, reused in LATRD.",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_latrd.hpp",
        "language": "cpp",
        "content": """template <int MAX_THDS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) latrd_dot_scale_axpy(const I n,
                                                                       U AA, ...)
{
    // ... setup ...
    
    // dot
    T norm2 = 0;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        T tempA = A[i];
        T tempW = W[i];
        if(i < MAX_THDS)
        {
            sh_A[i] = tempA;
            sh_W[i] = tempW;
        }
        norm2 += tempA * conj(tempW);
    }

    // reduce squared entries to find squared norm of x
    norm2 += shift_left(norm2, 1);
    // ... more reductions ...
    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm2 += sval[k];
        sval[0] = -0.5 * tau[0] * norm2;
    }
    __syncthreads();

    // axpy
    for(I i = tid; i < n; i += MAX_THDS)
    {
        if(i < MAX_THDS)
            W[i] = sh_W[i] + sval[0] * sh_A[i];
        else
            W[i] = W[i] + sval[0] * A[i];
    }
}"""
    }]
})

# Entry 9: L2 - LATRD workspace layout
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "sytrd_hetrd",
    "query": "How is the W workspace organized in LATRD? What does each column represent?",
    "answer": "The W workspace is an n × k matrix where:\n\n- **Dimensions**: n rows (matrix size), k columns (block size)\n- **Column i** (i=0 to k-1): Contains the w vector for the i-th Householder reflector in the panel\n- **Storage**: Column-major with leading dimension ldw = n\n- **Stride**: strideW = n*k for batched cases\n\nAfter LATRD completes:\n- W(0:n-1, 0:k-1) contains all k workspace vectors\n- These are used in the trailing GEMM update: A := A - V*W^H - W*V^H\n- The layout enables efficient GEMM access patterns\n\n**Important detail**: For lower triangular, W starts at row k (W(k:n-1, 0:k-1) is used). For upper triangular, W starts at row 0. This aligns with where the Householder vectors are stored in V (which is A).\n\nThe W matrix is the key to the blocked algorithm's efficiency.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """rocblas_int ldw = n;
rocblas_stride strideW = n * k;

// ... in blocked loop for lower triangular ...

// LATRD computes W for columns j:j+k-1
rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A,
                                     shiftA + idx2D(j, j, lda), lda, strideA, (E + j),
                                     strideE, (tau + j), strideP, tmptau_W, 0, ldw,
                                     strideW, batch_count, scalars, work, norms, workArr);

// Use W in GEMM updates
// V is at A(j+k:n-1, j:j+k-1)
// W is at tmptau_W(k:n-j-1, 0:k-1) - starting at row k, offset idx2D(k, 0, ldw)
rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
               n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
               strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
               shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);"""
    }]
})

# Entry 10: L1 - Pointer mode requirement
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "sytrd_hetrd",
    "query": "Why does SYTRD set pointer mode to host?",
    "answer": "SYTRD sets pointer mode to host because it uses scalar constants (minone = -1, one = 1) that are stored on the host stack. The code:\n\n```cpp\nrocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);\nconst T minone = T(-1);\nconst T one = T(1);\n```\n\nThese scalars are passed to rocBLAS GEMM calls. With host pointer mode, rocBLAS reads the scalar values from host memory (the stack) rather than expecting device pointers.\n\nThis is more convenient than allocating device memory for two constants. The mode is restored at the end: `rocblas_set_pointer_mode(handle, old_mode)`.\n\nNote: SYTD2/HETD2 uses device pointer mode because it has scalars pre-allocated in device memory (the 'scalars' array).",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """// everything must be executed with scalars on the device
rocblas_pointer_mode old_mode;
rocblas_get_pointer_mode(handle, &old_mode);
rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

const T minone = T(-1);
const T one = T(1);

// ... use minone and one in GEMM calls ...

rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
               n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
               strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
               shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);

// ... restore mode ...
rocblas_set_pointer_mode(handle, old_mode);"""
    }]
})

# Entry 11: L3 - Performance analysis
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "sytrd_hetrd",
    "query": "Analyze the computational complexity and performance characteristics of SYTRD vs SYTD2. When does blocked algorithm win?",
    "answer": "**Computational complexity (both O(n³) but different constants):**\n\nSYTD2 (unblocked):\n- n-1 iterations, each with SYMV O(n²) + SYR2 O(n²)\n- Total: ~(2/3)n³ Level 2 BLAS operations\n- Memory traffic: O(n³) due to poor cache reuse\n\nSYTRD (blocked):\n- (n/k) blocks, each with:\n  - LATRD: k × (SYMV O(n²) + updates) = O(n²k) Level 2 BLAS\n  - GEMM: 2 × O(n²k) Level 3 BLAS\n- Total: ~(2/3)n³ operations, but most in Level 3 BLAS\n- Memory traffic: O(n³/√B) due to cache blocking in GEMM (B = cache size)\n\n**Performance crossover:**\n- n < 64: SYTD2 faster (small kernel optimization)\n- 64 ≤ n < 128: Comparable (xxTRD_xxTD2_SWITCHSIZE)\n- n ≥ 128: SYTRD faster (2-5x speedup)\n- n ≥ 1024: SYTRD much faster (5-10x speedup)\n\n**Key factors:**\n- GEMM achieves near-peak FLOPS on GPUs (Level 3 BLAS)\n- Level 2 BLAS (SYMV, SYR2) limited by memory bandwidth\n- Blocking amortizes data movement: load V and W once, reuse in GEMM",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.hpp",
        "language": "cpp",
        "content": """// Blocked algorithm: (n/k) iterations
j = 0;
while(j < n - kk)
{
    // LATRD: O(n²k) Level 2 BLAS
    rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A, ...);

    // GEMM updates: 2 × O(n²k) Level 3 BLAS (fast!)
    // These dominate the computational cost and achieve high throughput
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   n - j - k, n - j - k, k, &minone, A, ..., tmptau_W, ..., &one, A, ...);
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                   n - j - k, n - j - k, k, &minone, tmptau_W, ..., A, ..., &one, A, ...);

    j += k;
}

// Tail: Use SYTD2 when remaining size < kk (unblocked is faster)
rocsolver_sytd2_hetd2_template<T>(handle, uplo, n - j, A, ...);

// Compare with SYTD2 unblocked:
// for(j = 0; j < n - 1; ++j)
// {
//     larfg(...);           // O(n)
//     symv_hemv(...);       // O(n²) Level 2 BLAS (memory bound)
//     syr2_her2(...);       // O(n²) Level 2 BLAS (memory bound)
// }"""
    }]
})

# Entry 12: L1 - C API functions
entries.append({
    "id": str(base_ts + 12),
    "level": "L1",
    "interface": "sytrd_hetrd",
    "query": "Show the C API functions for SYTRD/HETRD. How do they relate to SYTD2/HETD2?",
    "answer": "SYTRD/HETRD has the same four precision variants as SYTD2/HETD2:\n- **ssytrd, dsytrd**: Real symmetric (SYTRD)\n- **chetrd, zhetrd**: Complex Hermitian (HETRD)\n\nThe API is identical to SYTD2/HETD2 - same parameters, same outputs. The difference is purely internal:\n- SYTRD uses blocked algorithm (fast for large matrices)\n- SYTD2 uses unblocked algorithm (simple, good for small matrices)\n\nIn fact, SYTRD calls SYTD2 internally for:\n1. Small matrices (n <= xxTRD_xxTD2_SWITCHSIZE)\n2. Tail blocks in blocked algorithm\n\nFrom the user's perspective, SYTRD is a drop-in replacement for SYTD2 with better performance on large matrices.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytrd_hetrd.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_ssytrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tau)
{
    return rocsolver::rocsolver_sytrd_hetrd_impl<float>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_dsytrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tau)
{
    return rocsolver::rocsolver_sytrd_hetrd_impl<double>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_chetrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tau)
{
    return rocsolver::rocsolver_sytrd_hetrd_impl<rocblas_float_complex>(handle, uplo, n, A, lda, D,
                                                                        E, tau);
}

rocblas_status rocsolver_zhetrd(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tau)
{
    return rocsolver::rocsolver_sytrd_hetrd_impl<rocblas_double_complex>(handle, uplo, n, A, lda, D,
                                                                         E, tau);
}"""
    }]
})

# Write to JSONL file
output_file = "roclapack_sytrd_hetrd.jsonl"
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_file}")

# Verify the output
with open(output_file, 'r') as f:
    lines = f.readlines()
    print(f"Verified {len(lines)} lines in output file")
    
    # Check schema compliance
    for i, line in enumerate(lines):
        entry = json.loads(line)
        assert 'id' in entry, f"Entry {i} missing 'id'"
        assert 'level' in entry, f"Entry {i} missing 'level'"
        assert 'interface' in entry, f"Entry {i} missing 'interface'"
        assert entry['interface'] == 'sytrd_hetrd', f"Entry {i} has wrong interface"
        assert 'code_blocks' in entry, f"Entry {i} missing 'code_blocks'"
        for cb in entry['code_blocks']:
            assert 'content' in cb, f"Entry {i} has code_block without 'content'"
            assert 'path' in cb, f"Entry {i} has code_block without 'path'"
    
    print("Schema validation passed!")

# Count levels and coding tasks
l1_count = sum(1 for line in lines if json.loads(line)['level'] == 'L1')
l2_count = sum(1 for line in lines if json.loads(line)['level'] == 'L2')
l3_count = sum(1 for line in lines if json.loads(line)['level'] == 'L3')
coding_count = sum(1 for line in lines if 'implement' in json.loads(line).get('query', '').lower() or 'show' in json.loads(line).get('query', '').lower())

print(f"\nLevel distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
print(f"Coding-oriented tasks: {coding_count}")
