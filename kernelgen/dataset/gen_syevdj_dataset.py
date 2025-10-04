#!/usr/bin/env python3
import json
import time

entries = []

# Entry 1: SYEVDJ vs SYEV comparison (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 0),
    "level": "L3",
    "interface": "syevdj_heevdj",
    "query": "What is the difference between SYEVDJ and SYEV for computing eigenvalues of symmetric/Hermitian matrices, and when should SYEVDJ be used instead of SYEV?",
    "answer": """SYEVDJ and SYEV both compute eigenvalues and eigenvectors of symmetric/Hermitian matrices, but they use fundamentally different algorithms:

SYEV Algorithm (QR-based):
- Reduces matrix to tridiagonal form using SYTRD
- Solves tridiagonal eigenvalue problem using STEQR (implicit QR)
- Back-transforms eigenvectors using ORMTR
- Reliable for all matrix sizes

SYEVDJ Algorithm (Hybrid Divide-and-Conquer with Jacobi):
- Reduces matrix to tridiagonal form using SYTRD
- Solves tridiagonal eigenvalue problem using STEDCJ (divide-and-conquer with Jacobi solver)
- Back-transforms eigenvectors using ORMTR
- More efficient for medium to large matrices when eigenvectors are needed

The key difference is in the tridiagonal solver. SYEVDJ uses a hybrid approach that combines divide-and-conquer parallelism with Jacobi iterations. This provides better GPU utilization than QR for larger matrices.

When to use SYEVDJ:
1. n >= SYEVDJ_MIN_DC_SIZE (typically 25) and evect == rocblas_evect_original
2. Computing eigenvectors is required (for eigenvalues only, both are comparable)
3. Matrix is large enough to benefit from parallel divide-and-conquer

When to use SYEV:
1. Small matrices (n < 25)
2. Only eigenvalues needed (evect == rocblas_evect_none)
3. Memory is constrained (SYEVDJ requires more workspace)

The implementation automatically falls back to SYEVJ (pure Jacobi) for small matrices or when only eigenvalues are needed.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevdj_heevdj.hpp",
            "language": "cpp",
            "content": """if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
{
    // **** do not use D&C approach ****

    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(
        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, (S)0, workE,
        20, workSplits, D, strideD, info, batch_count, workVec, workTau, (S*)work1,
        (rocblas_int*)work2, (rocblas_int*)work3, (rocblas_int*)work4);
}
else
{
    // **** Use D&C approach ****

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D,
                                            strideD, workE, n, workTau, n, batch_count, scalars,
                                            (T*)work1, (T*)work2, (T*)work3, (T**)workArr, false);

    // solve with Jacobi solver
    rocsolver_stedcj_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, strideD, workE, n, workVec, 0, ldv, strideV,
        info, batch_count, work1, (S*)work2, (S*)work3, (S*)work4, workSplits, (S**)workArr);

    // update vectors
    rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
        handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda, strideA,
        workTau, n, workVec, 0, ldv, strideV, batch_count, scalars, (T*)work1, (T*)work2,
        (T*)work3, (T**)workArr);
}"""
        }
    ]
})

# Entry 2: STEDCJ divide-and-conquer algorithm (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 1),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "How does the STEDCJ divide-and-conquer algorithm work for solving tridiagonal eigenvalue problems in SYEVDJ?",
    "answer": """STEDCJ implements a parallel divide-and-conquer algorithm with four main phases:

1. SPLIT Phase (stedcj_split kernel):
   - Identifies independent blocks in the tridiagonal matrix
   - Split occurs when E[j] <= tol = eps * sqrt(|D[j]|) * sqrt(|D[j+1]|)
   - Independent blocks can be solved in parallel
   - Stores split positions in splits array

2. DIVIDE Phase (stedcj_divide_kernel):
   - Each split-block is recursively divided into sub-blocks
   - Number of levels determined by stedcj_num_levels(n)
   - Sub-blocks are created by artificially "cutting" the matrix
   - D elements at cut points are modified: D[k] -= E[k-1], D[k-1] -= E[k-1]
   - Creates rank-1 modification structure for later merging

3. SOLVE Phase (stedcj_solve_kernel):
   - Solves each sub-block independently using Jacobi iteration
   - Converts D,E vectors to full tridiagonal matrix
   - Calls run_syevj device function
   - Computes eigenvalues and eigenvectors for each sub-block
   - Maximum MAXSWEEPS (20) Jacobi sweeps per sub-block

4. MERGE Phase (multiple kernels for each level):
   - stedcj_mergePrepare: Deflation and secular equation setup
   - stedcj_mergeValues: Solve secular equations for eigenvalues
   - stedcj_mergeVectors: Compute eigenvectors from secular equation
   - stedcj_mergeUpdate: Update D and C with merged results
   - Processes levels from k=0 to maxlevs-1

The divide-and-conquer structure allows massive parallelism across split-blocks and sub-blocks, making it much faster than sequential QR for large matrices on GPUs.""",
    "code_blocks": [
        {
            "path": "library/src/auxiliary/rocauxiliary_stedcj.hpp",
            "language": "cpp",
            "content": """// 1. divide phase
ROCSOLVER_LAUNCH_KERNEL((stedcj_divide_kernel<S>), dim3(batch_count), dim3(STEDCJ_BDIM), 0,
                        stream, n, D, strideD, E, strideE, splits_map);

// 2. solve phase
ROCSOLVER_LAUNCH_KERNEL((stedcj_solve_kernel<S>),
                        dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDCJ_BDIM),
                        lmemsize, stream, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT,
                        info, static_cast<S*>(work_stack), splits_map, eps, ssfmin, ssfmax);

// 3. merge phase
for(rocblas_int k = 0; k < maxlevs; ++k)
{
    // a. prepare secular equations
    rocblas_int numgrps2 = 1 << (maxlevs - 1 - k);
    ROCSOLVER_LAUNCH_KERNEL((stedcj_mergePrepare_kernel<S>), ...);

    // b. solve to find merged eigen values
    ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeValues_kernel<S>), ...);

    // c. find merged eigen vectors
    ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeVectors_kernel<STEDCJ_EXTERNAL_GEMM, S>), ...);

    // c. update level
    ROCSOLVER_LAUNCH_KERNEL((stedcj_mergeUpdate_kernel<S>), ...);
}

// 4. update and sort
local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                static_cast<S*>(work_stack), 0, ldt, strideT, batch_count,
                                workArr);

ROCSOLVER_LAUNCH_KERNEL((stedcj_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n, D,
                        strideD, C, shiftC, ldc, strideC, batch_count, splits_map);"""
        }
    ]
})

# Entry 3: Jacobi rotation algorithm (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 2),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "How does the Jacobi rotation algorithm work for computing eigenvalues in SYEVDJ, and why is it effective for small sub-blocks?",
    "answer": """The Jacobi eigenvalue algorithm iteratively applies rotation matrices to zero out off-diagonal elements until the matrix becomes diagonal (containing eigenvalues):

Algorithm Overview:
1. For each off-diagonal element A[i,j], compute a rotation J such that (J'AJ)[i,j] = 0
2. J only affects rows/columns i and j, so ceil(n/2) rotations can be applied in parallel
3. Use top/bottom pairing scheme to cycle through all off-diagonal indices
4. Repeat (sweep) until off-diagonal norm < tolerance or max_sweeps reached

Rotation Calculation:
For element A[i,j], compute rotation J that zeros it:
- aij = A[i,j], magnitude mag = |aij|
- f = real(A[j,j] - A[i,i])
- g = 2*mag
- Compute c (cosine) and s (sine) using LARTG
- s is scaled by phase: s1 = s * aij / mag

Application:
- Right multiply: A = A * J (affects columns i,j)
- Left multiply: A = J' * A (affects rows i,j)
- For eigenvectors: V = V * J (accumulate rotations)

Why Effective for Sub-blocks:
1. Massive parallelism: ceil(n/2) rotations computed/applied simultaneously per sweep
2. GPU-friendly: Each rotation is independent within a sweep
3. Numerical stability: Tridiagonal structure ensures quick convergence
4. Small matrices: Convergence typically occurs in 5-10 sweeps for n < 100
5. No auxiliary storage: Works in-place on the matrix

The tridiagonal structure from SYTRD ensures that most off-diagonal elements are already zero, so Jacobi converges very quickly (often 1-2 sweeps after D&C).""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevj_heevj.hpp",
            "language": "cpp",
            "content": """// calculate current rotation J
if(tiy == 0 && i < n && j < n)
{
    aij = Acpy[i + j * n];
    mag = std::abs(aij);

    if(mag * mag < small_num)
    {
        c = 1;
        s1 = 0;
    }
    else
    {
        g = 2 * mag;
        f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
        f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
        lartg(f, g, c, s, r);
        s1 = s * aij / mag;
    }
    cosines_res[tix] = c;
    sines_diag[tix] = s1;
}
__syncthreads();

// apply J from the right and update vectors
if(i < n && j < n)
{
    c = cosines_res[tix];
    s1 = sines_diag[tix];
    s2 = conj(s1);

    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
    {
        rocblas_int y1 = ky * 2;
        rocblas_int y2 = y1 + 1;

        temp1 = Acpy[y1 + i * n];
        temp2 = Acpy[y1 + j * n];
        Acpy[y1 + i * n] = c * temp1 + s2 * temp2;
        Acpy[y1 + j * n] = -s1 * temp1 + c * temp2;
    }
}"""
        }
    ]
})

# Entry 4: Secular equation solving in merge phase (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 3),
    "level": "L3",
    "interface": "syevdj_heevdj",
    "query": "How are secular equations solved during the merge phase of STEDCJ, and what is the role of deflation?",
    "answer": """During the merge phase, STEDCJ combines eigenvalues/eigenvectors from two sub-blocks by solving a rank-1 modification problem, which reduces to finding roots of a secular equation.

Rank-1 Modification Structure:
When merging sub-blocks, the original tridiagonal matrix T can be written as:
T = D + p*z*z' (rank-1 modification)
where D is diagonal, p is the cut off-diagonal element, and z is the rank-1 modification vector.

Secular Equation:
The eigenvalues λ of T satisfy:
1 + p * Σ(z[i]² / (D[i] - λ)) = 0

This is a rational function with poles at D[i] and roots (the eigenvalues) between the poles.

Deflation Process (stedcj_mergePrepare):
Before solving, deflation eliminates trivial eigenvalues:
1. Zero component deflation: If |p*z[i]| <= tol, then D[i] is already an eigenvalue
2. Repeated value deflation: If |D[i] - D[j]| <= tol, apply Givens rotation to eliminate one z component
3. Tolerance: tol = 8 * eps * max(|D|, |z|)

Solving (stedcj_mergeValues):
1. Sort remaining D values and z components by permutation array
2. For each non-deflated eigenvalue position k:
   - Binary search to find interval containing the root
   - Use seq_solve or seq_solve_ext to find λ by Newton-Raphson iteration
   - Update distances D[i] - λ to prevent collapse
3. Re-scale z to avoid numerical issues

Vector Reconstruction (stedcj_mergeVectors):
1. Compute eigenvector components: v[i] = z[i] / (D[i] - λ)
2. Normalize: v = v / ||v||
3. Transform back: C = C_old * v (multiply by sub-block eigenvectors)

Deflation is crucial for:
- Avoiding division by zero when λ ≈ D[i]
- Reducing computational cost (fewer secular equations to solve)
- Improving numerical stability""",
    "code_blocks": [
        {
            "path": "library/src/auxiliary/rocauxiliary_stedcj.hpp",
            "language": "cpp",
            "content": """// 3c. deflate eigenvalues
// first deflate zero components
S f, g, c, s, rr;
for(int i = tidb; i < sz; i += hipBlockDim_x)
{
    tx = in + i;
    g = z[tx];
    if(abs(p * g) <= tol)
        // deflated ev because component in z is zero
        idd[tx] = 0;
    else
        idd[tx] = 1;
}
__syncthreads();

// now deflate repeated values
for(int r = 0; r < sz_even - 1; ++r)
{
    for(int i = tidb; i < sz_half; i += hipBlockDim_x)
    {
        // determine pair of values (base, top)
        // ... pairing logic ...

        // compare values and deflate if needed
        if(idd[base] == 1 && idd[top] == 1 && top < sz + in)
        {
            if(abs(D[base] - D[top]) <= tol)
            {
                // deflated ev because it is repeated
                idd[top] = 0;
                // rotation to eliminate component in z
                g = z[top];
                f = z[base];
                lartg(f, g, c, s, rr);
                z[base] = rr;
                z[top] = 0;
                // update C with the rotation
                for(int ii = 0; ii < n; ++ii)
                {
                    valf = C[ii + base * ldc];
                    valg = C[ii + top * ldc];
                    C[ii + base * ldc] = valf * c - valg * s;
                    C[ii + top * ldc] = valf * s + valg * c;
                }
            }
        }
    }
}

// 3e. Solve secular eqns
for(int j = iam; j < sz; j += bdm)
{
    if(mask[j] == 1)
    {
        // find position in the ordered array
        valf = p < 0 ? -ev[j] : ev[j];
        int count = dd, cc = 0;
        while(count > 0)
        {
            auto step = count / 2;
            auto it = cc + step;
            if(tmpd[it + j * n] < valf)
            {
                cc = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }

        rocblas_int linfo;
        linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps,
                          ssfmin, ssfmax);
    }
}"""
        }
    ]
})

# Entry 5: workSplits array usage (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 4),
    "level": "L1",
    "interface": "syevdj_heevdj",
    "query": "What is the purpose of the workSplits array in SYEVDJ, and what information does it contain?",
    "answer": """The workSplits array is a critical data structure that stores the divide-and-conquer tree structure for the STEDCJ algorithm.

Array Layout (per batch instance):
Size: (5*n + 2) integers
- splits[0..n]: Starting positions of split-blocks
- splits[n+1]: Number of split-blocks (nb)
- nsA[0..n-1] = splits[n+2..2n+1]: Sub-block sizes
- psA[0..n-1] = splits[2n+2..3n+1]: Sub-block starting positions
- idd[0..n-1] = splits[3n+2..4n+1]: Deflation mask (0=deflated, 1=active)
- pers[0..n-1] = splits[4n+2..5n+1]: Permutation array for secular equation

Usage in Different Phases:

1. Split Phase: Populates splits[0..nb] with independent block boundaries
2. Divide Phase: Computes nsA and psA for all sub-blocks
3. Solve Phase: Uses psA and nsA to identify which sub-block to solve
4. Merge Phase: Uses idd for deflation tracking, pers for ordering

The array allows all kernels to share the same tree structure without recomputation.""",
    "code_blocks": [
        {
            "path": "library/src/auxiliary/rocauxiliary_stedcj.hpp",
            "language": "cpp",
            "content": """// temporary arrays in global memory
// contains the beginning of split blocks
rocblas_int* splits = splitsA + bid * (5 * n + 2);
// the sub-blocks sizes
rocblas_int* nsA = splits + n + 2;
// the sub-blocks initial positions
rocblas_int* psA = nsA + n;
// if idd[i] = 0, the value in position i has been deflated
rocblas_int* idd = psA + n;
// container of permutations when solving the secular eqns
rocblas_int* pers = idd + n;

// total number of split blocks
rocblas_int nb = splits[n + 1];

// Select current split block
p1 = splits[kb];
p2 = splits[kb + 1];
bs = p2 - p1;
ns = nsA + p1;
ps = psA + p1;"""
        }
    ]
})

# Entry 6: Workspace allocation strategy (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 5),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "How does SYEVDJ allocate and reuse workspace memory for different phases of the algorithm?",
    "answer": """SYEVDJ uses a carefully designed workspace strategy that reuses memory across different phases:

Workspace Arrays:
1. size_scalars: Constants for rocBLAS calls (TRSM, GEMM, etc.)
2. size_workE: Superdiagonal of tridiagonal form (n*batch_count elements)
3. size_workTau: Householder scalars from SYTRD (n*batch_count elements)
4. size_workVec: Temporary eigenvector storage (n²*batch_count elements)
5. size_workSplits: Split/sub-block tree structure (5*n+2)*batch_count integers
6. size_work1, work2, work3, work4: Reusable workspace buffers
7. size_workArr: Array of pointers (batched mode only)

Memory Reuse Strategy:

For small matrices (n < SYEVDJ_MIN_DC_SIZE or evect != original):
- Falls back to pure SYEVJ (Jacobi without D&C)
- Only needs: workE (residual), workSplits (sweep counter), work1-4 (Jacobi workspace)
- Minimal memory footprint

For large matrices using D&C:
- Phase 1 (SYTRD): Uses scalars, work1, work2, work3, workArr
- Phase 2 (STEDCJ): Uses work1, work2, work3, work4, workSplits (D&C tree), workVec (eigenvectors)
- Phase 3 (ORMTR): Uses scalars, work1, work2, work3, workArr

The work1-4 buffers are sized to accommodate the maximum requirement across all phases:
*size_work1 = std::max({w11, w12, w13});  // SYTRD, STEDCJ, ORMTR
*size_work2 = std::max({w21, w22, w23});
*size_work3 = std::max({w31, w32, w33});

This approach minimizes total memory while ensuring each phase has sufficient workspace.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevdj_heevdj.hpp",
            "language": "cpp",
            "content": """// space for the superdiagonal of tridiag form
*size_workE = sizeof(S) * n * batch_count;

// space for the householder scalars
*size_workTau = sizeof(T) * n * batch_count;

// temp space for eigenvectors
*size_workVec = sizeof(T) * n * n * batch_count;

// requirements for tridiagonalization (sytrd/hetrd)
rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w11, &w21, &w31,
                                                &unused, false);

// extra requirements for computing eigenvalues and vectors (stedcj)
rocsolver_stedcj_getMemorySize<BATCHED, T, S>(rocblas_evect_tridiagonal, n, batch_count, &w12,
                                              &w22, &w32, size_work4, size_workSplits, &unused);

// extra requirements for ormtr/unmtr
rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                &unused, &w13, &w23, &w33, &unused);

// get max values
*size_work1 = std::max({w11, w12, w13});
*size_work2 = std::max({w21, w22, w23});
*size_work3 = std::max({w31, w32, w33});"""
        }
    ]
})

# Entry 7: SYEVDJ_MIN_DC_SIZE threshold (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 6),
    "level": "L1",
    "interface": "syevdj_heevdj",
    "query": "What is the SYEVDJ_MIN_DC_SIZE constant and why does SYEVDJ fall back to pure Jacobi for smaller matrices?",
    "answer": """SYEVDJ_MIN_DC_SIZE (typically 25) is the minimum matrix size for using divide-and-conquer. Below this threshold, SYEVDJ uses pure Jacobi iteration (SYEVJ) instead.

Reasons for the threshold:

1. Overhead vs Benefit:
   - D&C has overhead from split detection, divide phase, merge phase kernels
   - For small matrices, this overhead exceeds the parallelism benefit
   - Pure Jacobi completes in fewer kernel launches for n < 25

2. Memory Efficiency:
   - D&C requires workspace: O(n²) for tempvect, (5n+2) integers for splits
   - Pure Jacobi needs minimal workspace
   - For small n, memory savings are significant

3. Convergence Speed:
   - Jacobi converges very quickly on small matrices (typically 3-5 sweeps)
   - The iterative nature is not a bottleneck for small n
   - D&C splitting doesn't provide advantage when the whole matrix fits in fast memory

4. GPU Occupancy:
   - Small matrices don't generate enough parallelism to saturate GPU
   - Single Jacobi kernel can handle n=25 efficiently with proper thread organization
   - D&C parallelism is wasted when n is small

The fallback ensures SYEVDJ is always optimal regardless of input size.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevdj_heevdj.hpp",
            "language": "cpp",
            "content": """// if size too small or no vectors required
if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
{
    // space to store the residual
    *size_workE = sizeof(S) * batch_count;

    // space to store the number of sweeps
    *size_workSplits = sizeof(rocblas_int) * batch_count;

    // requirements for jacobi
    rocsolver_syevj_heevj_getMemorySize<BATCHED, T, S>(evect, uplo, n, batch_count,
                                                       size_workVec, size_workTau, size_work1,
                                                       size_work2, size_work3, size_work4);

    *size_scalars = 0;
    *size_workArr = 0;

    return;
}

// In template:
if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
{
    // **** do not use D&C approach ****

    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(
        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, (S)0, workE,
        20, workSplits, D, strideD, info, batch_count, workVec, workTau, (S*)work1,
        (rocblas_int*)work2, (rocblas_int*)work3, (rocblas_int*)work4);
}"""
        }
    ]
})

# Entry 8: Copy vectors operation (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 7),
    "level": "L1",
    "interface": "syevdj_heevdj",
    "query": "Why does SYEVDJ copy eigenvectors from workVec to A at the end, and what kernel is used?",
    "answer": """SYEVDJ computes eigenvectors in a temporary workspace array (workVec) and then copies them back to the original matrix A. This is done for two reasons:

1. Algorithm Structure:
   - STEDCJ works on a temporary identity matrix in workVec
   - The divide-and-conquer algorithm builds eigenvectors incrementally in workVec
   - Tridiagonal eigenvectors are stored in workVec, not A
   - ORMTR back-transforms using workVec as input

2. Preserve Input:
   - Matrix A contains the tridiagonal form after SYTRD
   - STEDCJ needs the original structure preserved until ORMTR
   - Only after ORMTR completes can A be overwritten with final eigenvectors

The copy operation uses the copy_mat kernel:
- Launched with (n × n) thread blocks arranged in a 2D grid
- Each block handles BS2 × BS2 elements (typically 32×32)
- Simple element-wise copy: A[i,j] = workVec[i,j]
- Parallelized across batch dimension in z

This final copy is very fast (memory-bandwidth limited) compared to the eigenvalue computation, so the overhead is negligible.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevdj_heevdj.hpp",
            "language": "cpp",
            "content": """// solve with Jacobi solver
rocsolver_stedcj_template<false, ISBATCHED, T>(
    handle, rocblas_evect_tridiagonal, n, D, strideD, workE, n, workVec, 0, ldv, strideV,
    info, batch_count, work1, (S*)work2, (S*)work3, (S*)work4, workSplits, (S**)workArr);

// update vectors
rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
    handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda, strideA,
    workTau, n, workVec, 0, ldv, strideV, batch_count, scalars, (T*)work1, (T*)work2,
    (T*)work3, (T**)workArr);

// copy vectors into A
const rocblas_int copyblocks = (n - 1) / BS2 + 1;
ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                        dim3(BS2, BS2), 0, stream, n, n, workVec, 0, ldv, strideV, A,
                        shiftA, lda, strideA);"""
        }
    ]
})

# Entry 9: evect parameter behavior (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 8),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "How does the evect parameter control eigenvector computation in SYEVDJ, and what are the differences between rocblas_evect_none and rocblas_evect_original?",
    "answer": """The evect parameter determines whether and how eigenvectors are computed:

rocblas_evect_none:
- Only eigenvalues are computed (stored in D)
- Matrix A is overwritten but does not contain eigenvectors
- Falls back to pure Jacobi (SYEVJ) regardless of matrix size
- Minimal workspace requirements
- Fastest mode when only eigenvalues are needed
- STEDCJ is NOT used (only SYEVJ)

rocblas_evect_original:
- Both eigenvalues (D) and eigenvectors (A) are computed
- For n >= SYEVDJ_MIN_DC_SIZE: Uses full D&C algorithm (SYTRD + STEDCJ + ORMTR)
- For n < SYEVDJ_MIN_DC_SIZE: Falls back to SYEVJ
- Eigenvectors are the eigenvectors of the original matrix
- Requires significantly more workspace (n² elements for workVec)
- Uses back-transformation (ORMTR) to convert tridiagonal eigenvectors to original eigenvectors

The evect parameter directly controls the algorithm choice:
- evect=none → Pure Jacobi only (simple, minimal memory)
- evect=original + large n → Divide-and-conquer (fast, more memory)

SYEVDJ does NOT support rocblas_evect_tridiagonal (this is only used internally by STEDCJ).""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevdj_heevdj.hpp",
            "language": "cpp",
            "content": """// if size too small or no vectors required
if(evect != rocblas_evect_original || n < SYEVDJ_MIN_DC_SIZE)
{
    // **** do not use D&C approach ****

    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(
        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, (S)0, workE,
        20, workSplits, D, strideD, info, batch_count, workVec, workTau, (S*)work1,
        (rocblas_int*)work2, (rocblas_int*)work3, (rocblas_int*)work4);
}
else
{
    // **** Use D&C approach ****

    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D,
                                            strideD, workE, n, workTau, n, batch_count, scalars,
                                            (T*)work1, (T*)work2, (T*)work3, (T**)workArr, false);

    const rocblas_int ldv = n;
    const rocblas_stride strideV = n * n;

    // solve with Jacobi solver
    rocsolver_stedcj_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, strideD, workE, n, workVec, 0, ldv, strideV,
        info, batch_count, work1, (S*)work2, (S*)work3, (S*)work4, workSplits, (S**)workArr);

    // update vectors
    rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
        handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda, strideA,
        workTau, n, workVec, 0, ldv, strideV, batch_count, scalars, (T*)work1, (T*)work2,
        (T*)work3, (T**)workArr);
}"""
        }
    ]
})

# Entry 10: Complete usage example (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 9),
    "level": "L3",
    "interface": "syevdj_heevdj",
    "query": "Provide a complete example showing how to use SYEVDJ to compute eigenvalues and eigenvectors of a symmetric matrix, including workspace allocation.",
    "answer": """Here's a complete example using SYEVDJ to compute eigenvalues and eigenvectors:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

int main() {
    const rocblas_int n = 512;
    const rocblas_int lda = n;

    // Create handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate host matrix (symmetric)
    std::vector<double> h_A(n * n);
    // ... initialize h_A with symmetric matrix data ...

    // Allocate device memory
    double *d_A, *d_D, *d_E;
    rocblas_int *d_info;
    hipMalloc(&d_A, sizeof(double) * n * n);
    hipMalloc(&d_D, sizeof(double) * n);        // eigenvalues
    hipMalloc(&d_E, sizeof(double) * n);        // unused, but required
    hipMalloc(&d_info, sizeof(rocblas_int));

    // Copy matrix to device
    hipMemcpy(d_A, h_A.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);

    // Query workspace size
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
    size_t size_scalars, size_workE, size_workTau, size_workVec;
    size_t size_workSplits, size_work1, size_work2, size_work3, size_work4, size_workArr;

    rocsolver_dsyevdj(handle, rocblas_evect_original, rocblas_fill_upper,
                      n, nullptr, lda, nullptr, nullptr);
    rocblas_get_device_memory_size(handle, &size_scalars, &size_workE, &size_workTau,
                                   &size_workVec, &size_workSplits, &size_work1,
                                   &size_work2, &size_work3, &size_work4, &size_workArr);

    // Allocate workspace
    void *d_scalars, *d_workE, *d_workTau, *d_workVec;
    void *d_workSplits, *d_work1, *d_work2, *d_work3, *d_work4, *d_workArr;
    hipMalloc(&d_scalars, size_scalars);
    hipMalloc(&d_workE, size_workE);
    hipMalloc(&d_workTau, size_workTau);
    hipMalloc(&d_workVec, size_workVec);
    hipMalloc(&d_workSplits, size_workSplits);
    hipMalloc(&d_work1, size_work1);
    hipMalloc(&d_work2, size_work2);
    hipMalloc(&d_work3, size_work3);
    hipMalloc(&d_work4, size_work4);
    hipMalloc(&d_workArr, size_workArr);

    rocblas_set_workspace(handle, d_scalars, size_scalars);
    rocblas_set_workspace(handle, d_workE, size_workE);
    rocblas_set_workspace(handle, d_workTau, size_workTau);
    rocblas_set_workspace(handle, d_workVec, size_workVec);
    rocblas_set_workspace(handle, d_workSplits, size_workSplits);
    rocblas_set_workspace(handle, d_work1, size_work1);
    rocblas_set_workspace(handle, d_work2, size_work2);
    rocblas_set_workspace(handle, d_work3, size_work3);
    rocblas_set_workspace(handle, d_work4, size_work4);
    rocblas_set_workspace(handle, d_workArr, size_workArr);

    // Compute eigenvalues and eigenvectors
    rocsolver_dsyevdj(handle, rocblas_evect_original, rocblas_fill_upper,
                      n, d_A, lda, d_D, d_info);

    // Copy results back
    std::vector<double> h_D(n);
    std::vector<double> h_evecs(n * n);
    rocblas_int h_info;
    hipMemcpy(h_D.data(), d_D, sizeof(double) * n, hipMemcpyDeviceToHost);
    hipMemcpy(h_evecs.data(), d_A, sizeof(double) * n * n, hipMemcpyDeviceToHost);
    hipMemcpy(&h_info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

    // Check convergence
    if (h_info == 0) {
        std::cout << "Eigenvalues computed successfully\\n";
        std::cout << "First eigenvalue: " << h_D[0] << "\\n";
        std::cout << "Last eigenvalue: " << h_D[n-1] << "\\n";
    }

    // Cleanup
    hipFree(d_A); hipFree(d_D); hipFree(d_E); hipFree(d_info);
    hipFree(d_scalars); hipFree(d_workE); hipFree(d_workTau);
    hipFree(d_workVec); hipFree(d_workSplits);
    hipFree(d_work1); hipFree(d_work2); hipFree(d_work3);
    hipFree(d_work4); hipFree(d_workArr);
    rocblas_destroy_handle(handle);

    return 0;
}
```

Key points:
- For n=512 >= SYEVDJ_MIN_DC_SIZE, uses full D&C algorithm
- Eigenvalues returned in D (sorted ascending)
- Eigenvectors returned in A (column i contains eigenvector for D[i])
- E array is not used by SYEVDJ but required by API
- info=0 indicates successful convergence""",
    "code_blocks": []
})

# Entry 11: Performance characteristics (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 10),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "What are the performance characteristics of SYEVDJ compared to other eigenvalue solvers (SYEV, SYEVD, SYEVJ)?",
    "answer": """SYEVDJ performance characteristics compared to other eigenvalue solvers:

SYEVDJ (Divide-and-Conquer with Jacobi):
- Best for: Medium to large matrices (n >= 100) with eigenvectors
- Complexity: O(n³) but with better GPU parallelism than QR
- Memory: O(n²) workspace for eigenvectors
- Convergence: Guaranteed (combines D&C structure with Jacobi convergence)
- GPU utilization: Excellent (high parallelism in divide, solve, merge phases)
- Typical speedup vs SYEV: 2-3× for n > 500

SYEV (QR-based):
- Best for: General purpose, all matrix sizes
- Complexity: O(n³) with limited parallelism in implicit QR
- Memory: O(n) workspace
- Convergence: Guaranteed
- GPU utilization: Moderate (sequential QR sweeps)
- Most portable, well-tested algorithm

SYEVD (Divide-and-Conquer with QR):
- Best for: Large matrices (n >= 500) with eigenvectors
- Complexity: O(n³) but highly parallel
- Memory: O(n²) workspace
- Convergence: Guaranteed
- GPU utilization: Very good (D&C + STEDC)
- Typically fastest for very large matrices (n > 2000)

SYEVJ (Pure Jacobi):
- Best for: Small matrices (n < 100) or when memory is limited
- Complexity: O(n³) but with excellent GPU parallelism for small n
- Memory: O(n) workspace (minimal)
- Convergence: Iterative (may not converge, but typically does in 5-20 sweeps)
- GPU utilization: Good for small n, poor for large n
- Can fail to converge (info > 0)

Recommended choice:
- n < 25: SYEVJ (simple, minimal memory)
- 25 <= n < 500: SYEVDJ (good balance)
- n >= 500 with vectors: SYEVD (fastest)
- n >= 500 without vectors: SYEV (less memory)
- Memory constrained: SYEV or SYEVJ""",
    "code_blocks": []
})

# Entry 12: Coding task - Jacobi convergence analysis (L2, coding)
entries.append({
    "id": str(int(time.time() * 1000) + 11),
    "level": "L2",
    "interface": "syevdj_heevdj",
    "query": "Write a HIP kernel to compute the off-diagonal Frobenius norm of a matrix stored in workVec after each Jacobi sweep in STEDCJ, for convergence analysis.",
    "answer": """Here's a HIP kernel to compute the off-diagonal Frobenius norm for convergence analysis:

```cpp
template <typename T, typename S>
__global__ void compute_offdiag_norm(const rocblas_int n,
                                     T* A,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     S* norms,
                                     const rocblas_int batch_count)
{
    // Batch instance
    rocblas_int bid = hipBlockIdx_z;

    // Thread indices
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bdim = hipBlockDim_x;

    // Shared memory for partial sums
    extern __shared__ S shared_sums[];

    if(bid < batch_count)
    {
        T* matrix = A + bid * strideA;
        S local_sum = 0;

        // Each thread computes partial sum of off-diagonal elements
        // Upper triangular traversal to avoid double-counting
        for(rocblas_int j = 0; j < n; ++j)
        {
            for(rocblas_int i = tid; i < j; i += bdim)
            {
                T aij = matrix[i + j * lda];
                // Frobenius norm: sum of squared absolute values
                // Count both A[i,j] and A[j,i] (symmetric)
                local_sum += 2 * std::norm(aij);
            }
        }

        // Store partial sum in shared memory
        shared_sums[tid] = local_sum;
        __syncthreads();

        // Parallel reduction in shared memory
        for(rocblas_int s = bdim / 2; s > 0; s >>= 1)
        {
            if(tid < s)
            {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }

        // Thread 0 writes the final norm
        if(tid == 0)
        {
            // Square root to get actual Frobenius norm
            norms[bid] = sqrt(shared_sums[0]);
        }
    }
}

// Host function to launch kernel
template <typename T, typename S>
void compute_jacobi_convergence(rocblas_handle handle,
                                const rocblas_int n,
                                T* A,
                                const rocblas_int lda,
                                const rocblas_stride strideA,
                                S* norms,
                                const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Use 256 threads per block for good occupancy
    constexpr rocblas_int threads = 256;
    size_t smem_size = threads * sizeof(S);

    dim3 grid(1, 1, batch_count);
    dim3 block(threads, 1, 1);

    hipLaunchKernelGGL(compute_offdiag_norm<T, S>,
                       grid, block, smem_size, stream,
                       n, A, lda, strideA, norms, batch_count);
}

// Usage in STEDCJ after each merge level:
// S* convergence_norms;
// hipMalloc(&convergence_norms, sizeof(S) * batch_count * maxlevs);
//
// for(rocblas_int k = 0; k < maxlevs; ++k) {
//     // ... merge kernels ...
//
//     // Check convergence
//     compute_jacobi_convergence(handle, n, tempvect, ldt, strideT,
//                               convergence_norms + k * batch_count, batch_count);
// }
```

This kernel:
1. Computes ||A - diag(A)||_F for each batch instance
2. Uses parallel reduction for efficiency
3. Stores per-batch norms for analysis
4. Can be used to track convergence across D&C levels
5. Helps determine if MAXSWEEPS should be adjusted

The norm should decrease geometrically with each sweep. If norm > tolerance after max_sweeps, convergence failed.""",
    "code_blocks": []
})

# Write to file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_syevdj_heevdj.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_path}")

# Verify the schema
import jsonschema

schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "level": {"enum": ["L1", "L2", "L3"]},
        "interface": {"type": "string"},
        "query": {"type": "string"},
        "answer": {"type": "string"},
        "code_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "language": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "language", "content"]
            }
        }
    },
    "required": ["id", "level", "interface", "query", "answer", "code_blocks"]
}

try:
    with open(output_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            jsonschema.validate(entry, schema)
    print("Schema validation passed!")

    # Count level distribution
    levels = {"L1": 0, "L2": 0, "L3": 0}
    for entry in entries:
        levels[entry["level"]] += 1
    print(f"Level distribution: L1={levels['L1']}, L2={levels['L2']}, L3={levels['L3']}")
except Exception as e:
    print(f"Schema validation failed: {e}")
