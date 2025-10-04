#!/usr/bin/env python3
import json
import time

# Get base timestamp
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete GETRF algorithm
entries.append({
    "id": str(base_ts + 1),
    "level": "L3",
    "interface": "getrf",
    "query": "Explain the complete LU factorization with partial pivoting algorithm (GETRF). How does it work and what is the output?",
    "answer": "GETRF computes the LU factorization with partial pivoting of an m×n matrix A: P*A = L*U, where:\n- **P** is a permutation matrix (stored as ipiv vector)\n- **L** is m×min(m,n) lower triangular with unit diagonal\n- **U** is min(m,n)×n upper triangular\n\nThe algorithm uses **blocked right-looking** approach:\n\n**Main loop** (for j=0 to min(m,n) by blk):\n1. **Panel factorization**: Factor panel A(j:m, j:j+blk-1) using getrf_panelLU\n   - This calls GETF2 (unblocked LU) iteratively on inner blocks\n   - Performs partial pivoting (find max element, swap rows)\n   - Updates trailing sub-panel with TRSM and GEMM\n2. **Trailing matrix update**: Update A(j:m, j+blk:n) using TRSM (solve L) and GEMM (subtract L*U)\n\n**Output**:\n- A is overwritten with L (below diagonal) and U (on and above diagonal)\n- ipiv[i] contains the row index that was swapped with row i (1-based)\n- info indicates singularity: 0 if successful, i if U(i,i)==0\n\nThe blocking allows use of Level 3 BLAS (GEMM) for O(n³) operations, achieving high performance on GPUs.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// MAIN LOOP
for(I j = 0; j < dim; j += blk)
{
    jb = std::min(dim - j, blk);

    if(pivot || panel)
    {
        // factorize outer block panel
        getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j * inca, inca,
                                           lda, strideA, ipiv, shiftP + j, strideP, info,
                                           batch_count, pivot, scalars, work1, work2, work3,
                                           work4, optim_mem, pivotval, pivotidx, j, iipiv, m);
    }

    // update trailing matrix
    nextpiv = j + jb; //position for the matrix update
    mm = m - nextpiv; //size for the matrix update
    nn = n - nextpiv; //size for the matrix update
    if(nextpiv < n)
    {
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
            shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
            shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA, batch_count, optim_mem,
            work1, work2, work3, work4);

        if(nextpiv < m)
        {
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb,
                           &minone, A, shiftA + idx2D(nextpiv, j, inca, lda), inca, lda,
                           strideA, A, shiftA + idx2D(j, nextpiv, inca, lda), inca, lda,
                           strideA, &one, A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca,
                           lda, strideA, batch_count, (T**)nullptr);
        }
    }
}"""
    }]
})

# Entry 2: L2 - Panel factorization (getrf_panelLU)
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "getrf",
    "query": "What is getrf_panelLU and how does it factorize a panel in the blocked GETRF algorithm?",
    "answer": "getrf_panelLU factorizes a panel block (m×jb submatrix) using an inner blocking strategy. It divides the panel into smaller inner blocks of size innerBlk and processes them iteratively:\n\n**For each inner block k** (k=0 to jb by innerBlk):\n1. **Factor inner panel**: Call GETF2 to factor the k-th inner block (unblocked LU with pivoting)\n2. **Apply row swaps**: Use getrf_row_permutate kernel to apply pivots to remaining columns in parallel\n3. **Solve triangular system**: Use TRSM to solve L*U = B for remaining columns in the inner block\n4. **Update trailing sub-panel**: Use GEMM to apply rank-k update to remaining panel\n\nThe inner blocking is tuned based on matrix dimensions (getrf_get_innerBlkSize) and balances:\n- Small blocks: Less GEMM work but better cache reuse\n- Large blocks: More GEMM work (Level 3 BLAS) but more memory traffic\n\nTypical inner block sizes are 8, 16, 24, or 32 depending on m and n.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// Main loop
for(I k = 0; k < nn; k += blk)
{
    jb = std::min(nn - k, blk); // number of columns/pivots in the inner block

    // factorize inner panel block
    rocsolver_getf2_template<ISBATCHED, T>(handle, mm - k, jb, A, shiftA + idx2D(k, k, inca, lda),
                                           inca, lda, strideA, ipiv, shiftP + k, strideP, info,
                                           batch_count, scalars, pivotval, pivotidx, pivot,
                                           offset + k, permut_idx, stridePI);
    if(pivot)
    {
        dimx = jb;
        dimy = I(1024) / dimx;
        blocks = (n - jb - 1) / dimy + 1;
        grid = dim3(1, blocks, batch_count);
        threads = dim3(dimx, dimy, 1);
        lmemsize = dimx * dimy * sizeof(T);

        // swap rows
        ROCSOLVER_LAUNCH_KERNEL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n,
                                offset + k, jb, A, r_shiftA + k * inca, inca, lda, strideA,
                                permut_idx, stridePI);
    }

    // update trailing sub-block
    if(k + jb < nn)
    {
        rocsolver_trsm_lower<BATCHED, STRIDED, T>(
            handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb,
            nn - k - jb, A, shiftA + idx2D(k, k, inca, lda), inca, lda, strideA, A,
            shiftA + idx2D(k, k + jb, inca, lda), inca, lda, strideA, batch_count, optim_mem,
            work1, work2, work3, work4);

        if(k + jb < mm)
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm - k - jb,
                           nn - k - jb, jb, &minone, A, shiftA + idx2D(k + jb, k, inca, lda),
                           inca, lda, strideA, A, shiftA + idx2D(k, k + jb, inca, lda), inca,
                           lda, strideA, &one, A, shiftA + idx2D(k + jb, k + jb, inca, lda),
                           inca, lda, strideA, batch_count, (T**)nullptr);
    }
}"""
    }]
})

# Entry 3: L1 - Block size selection
entries.append({
    "id": str(base_ts + 3),
    "level": "L1",
    "interface": "getrf",
    "query": "How does GETRF determine the outer and inner block sizes? What factors influence the choice?",
    "answer": "GETRF uses two levels of blocking:\n\n**Outer block size (blk)**: Determined by getrf_get_blksize() based on:\n- Matrix dimension: min(m, n)\n- Batched vs non-batched\n- Pivoting vs non-pivoting (npvt)\n- Lookup tables tuned for real vs complex types\n- Special values: blk==1 means use full matrix, blk==-1 means panel mode\n\n**Inner block size (innerBlk)**: Determined by getrf_get_innerBlkSize() based on:\n- Panel dimensions (m, n)\n- 2D lookup table indexed by row and column intervals\n- Typical values: 1, 8, 16, 24, 32, 40\n- innerBlk==1 means use full panel width\n\nThese sizes are empirically tuned to balance:\n- Level 3 BLAS efficiency (larger blocks better)\n- Cache reuse (smaller blocks better)\n- GPU occupancy (moderate blocks better)\n\nFor example, GETRF_BLKSIZES_REAL might give blk=256 for n=2048 (real), while GETRF_INNBLKSIZES_REAL[10][5] might give innerBlk=8 for specific (m,n).",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// size of outer blocks
I blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

// ... in getrf_panelLU ...
I blk = getrf_get_innerBlkSize<ISBATCHED, T>(mm, nn, pivot);

// Example lookup for real types:
template <bool ISBATCHED, typename T, typename I, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
I getrf_get_innerBlkSize(I m, I n, const bool pivot)
{
    I blk;
    if(ISBATCHED)
    {
        if(pivot)
        {
            I M = GETRF_BATCH_NUMROWS_REAL - 1;
            I N = GETRF_BATCH_NUMCOLS_REAL - 1;
            I intervalsM[] = {GETRF_BATCH_INTERVALSROW_REAL};
            I intervalsN[] = {GETRF_BATCH_INTERVALSCOL_REAL};
            I size[][GETRF_BATCH_NUMCOLS_REAL] = {GETRF_BATCH_INNBLKSIZES_REAL};
            blk = size[get_index(intervalsM, M, m)][get_index(intervalsN, N, n)];
        }
        // ...
    }
    if(blk == 1)
        blk = n;
    return blk;
}"""
    }]
})

# Entry 4: L2 - Partial pivoting and row swaps
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "getrf",
    "query": "How does GETRF perform partial pivoting and parallel row swaps?",
    "answer": "Partial pivoting finds the maximum magnitude element in each column and swaps rows to place it on the diagonal. GETRF uses a two-stage approach:\n\n**Stage 1: Find pivot (in GETF2)**\n- getf2_iamax kernel: Parallel reduction to find max|A(j:m, j)| and its index\n- Uses warp-level reductions in shared memory\n- 1024 threads cooperate to find maximum\n\n**Stage 2: Apply pivot**\n- getf2_check_singularity kernel:\n  - Swaps rows j and pivot_idx for column j\n  - Stores ipiv[j] = pivot_idx + offset\n  - Updates permutation tracking array\n  - Checks singularity (A[j,j] == 0)\n  - Computes reciprocal 1/A[j,j] for SCAL operation\n\n**Stage 3: Parallel row swaps for panel** \n- getrf_row_permutate kernel:\n  - Applies all panel pivots to remaining columns in parallel\n  - Each thread block handles jb pivots for a subset of columns\n  - Uses shared memory to stage row swap data\n  - This allows GEMM to work on properly permuted data\n\nThe permutation is implicit: ipiv stores the swap sequence, final permutation matrix P can be reconstructed by applying swaps in order.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getf2.hpp",
        "language": "cpp",
        "content": """/** This kernel executes an optimized reduction to find the index of the
    maximum element of a given vector (iamax) **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(IAMAX_THDS) getf2_iamax(const I m,
                                                                U xx, ...)
{
    // batch instance
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    T* x = load_ptr_batch<T>(xx, bid, shiftx, stridex);

    // shared memory setup
    __shared__ S sval[IAMAX_THDS];
    __shared__ I sidx[IAMAX_THDS];

    iamax<IAMAX_THDS>(tid, m, x, incx, sval, sidx);

    // write results back to global memory
    if(tid == 0)
        pivotidx[bid] = sidx[0];
}

/** Execute all permutations dictated by the panel factorization
    in parallel (concurrency by rows and columns) **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void getrf_row_permutate(const I n, const I offset, const I blk,
                                          U AA, ...)
{
    // shared mem for temporary values
    extern __shared__ double lmem[];
    T* temp = reinterpret_cast<T*>(lmem);

    // do permutations in parallel (each tx perform a row swap)
    I idx1 = piv[tx];
    I idx2 = piv[idx1];
    temp[tx + ty * bdx] = A[idx1 * inca + j * lda];
    A[idx1 * inca + j * lda] = A[idx2 * inca + j * lda];
    __syncthreads();

    // copy temp results back to A
    A[tx * inca + j * lda] = temp[tx + ty * bdx];
}"""
    }]
})

# Entry 5: L1 - GETRF vs GETF2
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "getrf",
    "query": "What is the difference between GETRF and GETF2? When is each used?",
    "answer": "**GETRF (blocked)**:\n- Uses panel factorization with inner/outer blocking\n- Processes blk columns at a time\n- Leverages Level 3 BLAS (GEMM) for trailing matrix updates\n- Fast for large matrices (n > 64)\n- More complex, higher setup overhead\n\n**GETF2 (unblocked)**:\n- Processes one column at a time\n- Uses Level 2 BLAS (GER for rank-1 updates)\n- Simple implementation\n- Fast for small matrices (n < 64)\n- Used as building block in GETRF for panel factorization\n\n**When used**:\n- GETRF: Main entry point for users, delegates to GETF2 if blk==0 (small matrices)\n- GETF2: Called by GETRF for inner panel blocks, or directly for small matrices\n\nThe blocked GETRF achieves 2-10x speedup over GETF2 for large matrices by using GEMM which saturates GPU compute.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// size of outer blocks
I blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

if(blk == 0)
    return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, inca, lda, strideA,
                                                  ipiv, shiftP, strideP, info, batch_count,
                                                  scalars, pivotval, pivotidx, pivot);

// ... otherwise use blocked GETRF with outer loop ...
// ... which calls GETF2 for panel factorization:
rocsolver_getf2_template<ISBATCHED, T>(handle, mm - k, jb, A, shiftA + idx2D(k, k, inca, lda),
                                       inca, lda, strideA, ipiv, shiftP + k, strideP, info,
                                       batch_count, scalars, pivotval, pivotidx, pivot,
                                       offset + k, permut_idx, stridePI);"""
    }]
})

# Entry 6: L2 - GETRF_NPVT (no pivoting variant)
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "getrf",
    "query": "What is GETRF_NPVT and how does it differ from standard GETRF?",
    "answer": "GETRF_NPVT is the no-pivoting variant of LU factorization: A = L*U (without permutation matrix P). Key differences:\n\n**GETRF (with pivoting)**:\n- Computes P*A = L*U with partial pivoting\n- Requires ipiv array to store row swaps\n- More stable numerically (pivot prevents division by small numbers)\n- Slightly slower due to pivoting overhead\n- Always safe to use\n\n**GETRF_NPVT (no pivoting)**:\n- Computes A = L*U without row swaps\n- ipiv is nullptr, no pivot tracking\n- Faster (no iamax, no row swaps)\n- Can be unstable if A has small diagonal elements\n- Use only when: (1) A is diagonally dominant, or (2) stability is not critical\n\n**Implementation differences**:\n- pivot parameter controls the code path\n- NPVT skips: getf2_iamax, getf2_check_singularity (pivot part), getrf_row_permutate\n- NPVT may use different block sizes (separate tuning tables)\n- Panel mode: NPVT can factor only diagonal blocks instead of full panels",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_sgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, false);
}"""
    }]
})

# Entry 7: L3 - Workspace and memory management (coding)
entries.append({
    "id": str(base_ts + 7),
    "level": "L3",
    "interface": "getrf",
    "query": "Show the workspace calculation for GETRF. What are the different workspace components and how are they sized?",
    "answer": "GETRF requires multiple workspace arrays:\n\n**1. scalars**: Constants for rocBLAS calls (3 * sizeof(T))\n**2. pivotval, pivotidx**: For pivot search in GETF2\n**3. iipiv**: Permutation tracking for panel swaps (m * sizeof(I) * batch_count)\n**4. iinfo**: Singularity info for sub-blocks (sizeof(I) * batch_count)\n**5. work1-4**: For TRSM calls (memory optimization)\n\nThe calculation depends on block size:\n- If blk==0: Only GETF2 workspace needed\n- If blk>0: Max of GETF2 workspace (for panels) and TRSM workspace (for updates)\n\nTRSM workspace is calculated for largest possible panel (min(dim, 512) × n) to avoid reallocations. The optim_mem parameter controls TRSM memory optimization strategy.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_getrf_getMemorySize(const I m, const I n, const bool pivot,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1, size_t* size_work2,
                                   size_t* size_work3, size_t* size_work4,
                                   size_t* size_pivotval, size_t* size_pivotidx,
                                   size_t* size_iipiv, size_t* size_iinfo,
                                   bool* optim_mem,
                                   const I lda = 1, const I inca = 1)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    I dim = std::min(m, n);
    I blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

    if(blk == 0)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, n, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx, false, inca);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }
    else
    {
        // largest block panel dimension is 512
        dim = min(dim, I(512));

        // requirements for largest possible GETF2 for the sub blocks
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, dim, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx, true, inca);

        // extra workspace to store info about singularity and pivots of sub blocks
        *size_iinfo = sizeof(I) * batch_count;
        *size_iipiv = pivot ? m * sizeof(I) * batch_count : 0;

        // extra workspace for calling largest possible TRSM
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none, dim, n,
                                                batch_count, size_work1, size_work2, size_work3,
                                                size_work4, optim_mem, true, lda, lda, inca, inca);
        // ...
    }
}"""
    }]
})

# Entry 8: L1 - Singularity checking
entries.append({
    "id": str(base_ts + 8),
    "level": "L1",
    "interface": "getrf",
    "query": "How does GETRF check for singularity and what does the info parameter indicate?",
    "answer": "Singularity is checked in getf2_check_singularity (or getf2_npvt_check_singularity) kernel:\n\n**Check**: After pivoting (or for diagonal element if npvt), if A[j,j] == 0:\n1. Set pivot_val[id] = 1 (flag for error)\n2. If info[id] == 0, set info[id] = j + 1 + offset (Fortran 1-based index)\n\n**info values**:\n- info = 0: Success, matrix is non-singular\n- info = i (i > 0): U(i,i) is exactly zero, matrix is singular\n- The first zero diagonal is reported\n\n**What happens on singularity**:\n- Factorization continues (doesn't abort)\n- Division by zero is avoided (pivot_val = 1)\n- Result is still L*U = P*A, but U is singular\n- User must check info and decide how to proceed\n\nThis matches LAPACK behavior: factorization completes but warns about singularity via info parameter.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getf2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void getf2_check_singularity(...)
{
    if(tid == j)
    {
        // update pivot index
        I* ipiv = ipivA + id * strideP + shiftP;
        ipiv[j] = pivot_idx + offset;

        // update row order of final permutated matrix
        if(permut_idx)
        {
            I* permut = permut_idx + id * stridePI;
            if(exch != j)
                swap(permut[j], permut[exch]);
        }

        // update info (check singularity)
        if(A[j * inca + j * lda] == 0)
        {
            pivot_val[id] = 1;
            if(info[id] == 0)
                info[id] = static_cast<INFO>(j + 1 + offset); // use Fortran 1-based indexing
        }
        else
            pivot_val[id] = S(1) / A[j * inca + j * lda];
    }
}"""
    }]
})

# Entry 9: L2 - 64-bit integer support
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "getrf",
    "query": "How does GETRF support 64-bit integers for large matrices?",
    "answer": "GETRF provides _64 variants (sgetrf_64, dgetrf_64, etc.) that use int64_t instead of rocblas_int (int32_t) for matrix dimensions and indices. This enables:\n\n- Matrices larger than 2^31-1 elements\n- Leading dimensions > 2^31-1\n- Batch counts > 2^31-1\n\n**Implementation**:\n- Template parameter I can be int32_t or int64_t\n- All dimension variables (m, n, lda, inca, etc.) use type I\n- All index calculations use type I\n- ipiv array also uses type I (int64_t for _64 variants)\n\n**Availability**:\n- Controlled by HAVE_ROCBLAS_64 preprocessor macro\n- If not compiled with 64-bit support, returns rocblas_status_not_implemented\n- Requires rocBLAS built with 64-bit integer support\n\nThis is essential for large-scale scientific computing where matrices can exceed 32-bit integer limits.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgetrf_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

// Template handles both 32-bit and 64-bit:
template <typename T, typename I, typename U>
rocblas_status rocsolver_getrf_impl(rocblas_handle handle,
                                    const I m,  // I can be int32_t or int64_t
                                    const I n,
                                    U A,
                                    const I lda,
                                    I* ipiv,
                                    I* info,
                                    const bool pivot)
{
    // ...
}"""
    }]
})

# Entry 10: L1 - Panel vs full factorization mode
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "getrf",
    "query": "What is panel mode in GETRF_NPVT and when is it used?",
    "answer": "Panel mode is an optimization for GETRF_NPVT (no-pivoting variant) controlled by the sign of the block size:\n\n**Panel mode (blk < 0)**:\n- Factor the entire panel block (m×blk submatrix)\n- blk is negated: blk = -blk\n- Updates all rows in the panel\n\n**Diagonal-only mode (blk > 0)**:\n- Factor only the diagonal block (blk×blk submatrix)  \n- Update remaining panel rows with TRSM\n- More work deferred to Level 3 BLAS\n\n**When used**:\n- Panel mode: For NPVT with block sizes < 0 in tuning tables\n- Diagonal-only: For NPVT with block sizes > 0\n- With pivoting: Always uses panel mode\n\nPanel mode is determined by: `panel = (blk < 0)` then `blk = -blk`. This allows the tuning tables to encode the factorization strategy in the block size sign.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// in the npvt cases, panel determines whether the whole block-panel or only the
// diagonal block is factorized
bool panel = false;
if(blk < 0)
{
    panel = true;
    blk = -blk;
}

// MAIN LOOP
for(I j = 0; j < dim; j += blk)
{
    jb = std::min(dim - j, blk);

    if(pivot || panel)
    {
        // factorize outer block panel (all m-j rows)
        getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j * inca, ...);
    }
    else
    {
        // factorize only outer diagonal block (jb×jb)
        getrf_panelLU<BATCHED, STRIDED, T>(handle, jb, jb, n, A, shiftA + j * inca, ...);

        // update remaining rows in outer panel with TRSM
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
            m - j - jb, jb, A, shiftA + idx2D(j, j, inca, lda), ...);
    }
}"""
    }]
})

# Entry 11: L3 - TRSM and GEMM updates
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "getrf",
    "query": "Explain the TRSM and GEMM updates in GETRF. Why are they critical for performance?",
    "answer": "GETRF achieves high performance by using Level 3 BLAS (TRSM and GEMM) for the bulk of computation:\n\n**After panel factorization A(j:m, j:j+blk-1) = L*U:**\n\n**1. TRSM (Triangular Solve)**: Solve L*X = B\n- Updates: A(j:j+blk-1, j+blk:n) ← L^(-1) * A(j:j+blk-1, j+blk:n)\n- This computes the U part for columns j+blk:n\n- Uses left-side, lower triangular, unit diagonal TRSM\n- O(blk² * (n-j-blk)) operations\n\n**2. GEMM (Matrix Multiply)**: Rank-blk update\n- Updates: A(j+blk:m, j+blk:n) ← A(j+blk:m, j+blk:n) - L*U\n- where L = A(j+blk:m, j:j+blk-1), U = A(j:j+blk-1, j+blk:n)\n- O((m-j-blk) * (n-j-blk) * blk) operations - this dominates!\n- GEMM achieves near-peak FLOPS on GPUs\n\n**Why critical**:\n- ~90% of FLOPS are in GEMM for large matrices\n- GEMM uses Level 3 BLAS with excellent cache reuse\n- Blocked algorithm transforms O(n³) Level 2 operations into Level 3\n- TRSM prepares data for GEMM (solves triangular system)\n\nWithout blocking, we'd use GER (rank-1 update) which is memory-bound.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.hpp",
        "language": "cpp",
        "content": """// update trailing matrix
nextpiv = j + jb; //position for the matrix update
mm = m - nextpiv; //size for the matrix update
nn = n - nextpiv; //size for the matrix update
if(nextpiv < n)
{
    // TRSM: Solve L*U = B for the right part of U
    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
        handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
        shiftA + idx2D(j, j, inca, lda), inca, lda, strideA, A,
        shiftA + idx2D(j, nextpiv, inca, lda), inca, lda, strideA, batch_count, optim_mem,
        work1, work2, work3, work4);

    if(nextpiv < m)
    {
        // GEMM: Apply rank-blk update to trailing submatrix
        // A(nextpiv:m, nextpiv:n) -= A(nextpiv:m, j:j+blk) * A(j:j+blk, nextpiv:n)
        //                          -= L * U
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb,
                       &minone, A, shiftA + idx2D(nextpiv, j, inca, lda), inca, lda,
                       strideA, A, shiftA + idx2D(j, nextpiv, inca, lda), inca, lda,
                       strideA, &one, A, shiftA + idx2D(nextpiv, nextpiv, inca, lda), inca,
                       lda, strideA, batch_count, (T**)nullptr);
    }
}"""
    }]
})

# Entry 12: L1 - C API functions
entries.append({
    "id": str(base_ts + 12),
    "level": "L1",
    "interface": "getrf",
    "query": "Show the C API functions for GETRF. What variants are available?",
    "answer": "GETRF has multiple variants:\n\n**Standard precision** (32-bit indices):\n- sgetrf, dgetrf, cgetrf, zgetrf: With pivoting\n- sgetrf_npvt, dgetrf_npvt, cgetrf_npvt, zgetrf_npvt: No pivoting\n\n**64-bit integer** (for large matrices):\n- sgetrf_64, dgetrf_64, cgetrf_64, zgetrf_64: With pivoting\n- sgetrf_npvt_64, dgetrf_npvt_64, cgetrf_npvt_64, zgetrf_npvt_64: No pivoting\n\n**Parameters**:\n- m, n: Matrix dimensions\n- A: Input/output matrix (overwritten with L and U)\n- lda: Leading dimension\n- ipiv: Pivot indices (NULL for npvt)\n- info: Singularity indicator\n\nAll variants call the same template rocsolver_getrf_impl with appropriate type T and integer type I, with pivot flag controlling behavior.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_getrf.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_sgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, true);
}

rocblas_status rocsolver_zgetrf(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_getrf_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv, info,
                                                                   true);
}

rocblas_status rocsolver_sgetrf_npvt(rocblas_handle handle,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     float* A,
                                     const rocblas_int lda,
                                     rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_impl<float>(handle, m, n, A, lda, ipiv, info, false);
}

rocblas_status rocsolver_dgetrf_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_impl<double>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}"""
    }]
})

# Write to JSONL file
output_file = "roclapack_getrf.jsonl"
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
        assert entry['interface'] == 'getrf', f"Entry {i} has wrong interface"
        assert 'code_blocks' in entry, f"Entry {i} missing 'code_blocks'"
        for cb in entry['code_blocks']:
            assert 'content' in cb, f"Entry {i} has code_block without 'content'"
            assert 'path' in cb, f"Entry {i} has code_block without 'path'"
    
    print("Schema validation passed!")

# Count levels and coding tasks
l1_count = sum(1 for line in lines if json.loads(line)['level'] == 'L1')
l2_count = sum(1 for line in lines if json.loads(line)['level'] == 'L2')
l3_count = sum(1 for line in lines if json.loads(line)['level'] == 'L3')
coding_count = sum(1 for line in lines if 'show' in json.loads(line).get('query', '').lower() or 'implement' in json.loads(line).get('query', '').lower())

print(f"\nLevel distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
print(f"Coding-oriented tasks: {coding_count}")
