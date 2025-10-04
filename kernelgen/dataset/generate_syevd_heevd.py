#!/usr/bin/env python3
import json
import time

# Get base timestamp
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYEVD/HEEVD algorithm flow
entries.append({
    "id": str(base_ts + 1),
    "level": "L3",
    "interface": "syevd_heevd",
    "query": "Explain the complete algorithm flow of SYEVD/HEEVD (divide-and-conquer eigenvalue solver) and how it differs from the standard SYEV/HEEV QR-based solver.",
    "answer": "SYEVD/HEEVD uses a divide-and-conquer algorithm which is generally faster than the QR iteration used in SYEV/HEEV, especially for large matrices. The algorithm has three main phases:\n\n1. **Reduce to tridiagonal form** using SYTRD/HETRD (same as SYEV/HEEV)\n2. **Compute eigenvalues/eigenvectors of tridiagonal matrix** using STEDC (divide-and-conquer) instead of STEQR (QR iteration)\n3. **Transform eigenvectors back** using ORMTR/UNMTR (if evect == original)\n\nThe key difference is in phase 2. STEDC divides the tridiagonal matrix into smaller sub-blocks, solves them independently using STEQR, then merges the solutions by solving rank-1 secular equations. This provides O(n^2.5) complexity vs O(n^3) for SYEV/HEEV.\n\nThe algorithm can also operate in hybrid mode: if evect != original, STERF (eigenvalues only, no vectors) is used instead of STEDC for better performance.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_syevd_heevd.hpp",
        "language": "cpp",
        "content": """// reduce A to tridiagonal form
rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                        strideE, tau, n, batch_count, scalars, (T*)work1,
                                        (T*)work2, tmptau_W, workArr, false);

if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
{
    // only in hybrid mode, compute eigenvalues using sterf
    rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                (rocblas_int*)work1);
}
else
{
    // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
    constexpr bool ISBATCHED = BATCHED || STRIDED;
    const rocblas_int ldw = n;
    const rocblas_stride strideW = n * n;

    rocsolver_stedc_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0, ldw,
        strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

    // update the eigenvectors (if applicable)
    if(evect == rocblas_evect_original)
    {
        rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
            handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda,
            strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count, scalars, (T*)work2, tmpz,
            splits, work4, (T*)work1, (T*)work3, workArr, optim_mem);

        // copy matrix product into A
        const rocblas_int copyblocks = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW, A,
                                shiftA, lda, strideA);
    }
}"""
    }]
})

# Entry 2: L2 - STEDC divide-and-conquer algorithm
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "syevd_heevd",
    "query": "How does the STEDC divide-and-conquer algorithm work? Explain the divide, solve, and merge phases.",
    "answer": "STEDC implements a divide-and-conquer algorithm for tridiagonal eigenvalue problems with three main phases:\n\n**1. Divide Phase** (stedc_divide_kernel):\n- Recursively splits the n×n tridiagonal matrix into 2^levels sub-blocks\n- Number of levels = ceil(log2(n)) - 4 for n > 16\n- At each split point p, subtracts off-diagonal element E[p-1] from both D[p] and D[p-1]\n- This decouples the matrix into independent sub-problems\n\n**2. Solve Phase** (stedc_solve_kernel):\n- Solves each sub-block independently in parallel using classical STEQR (QR iteration)\n- All sub-blocks are solved simultaneously using different thread blocks\n- Each sub-block is small enough for efficient QR iteration\n\n**3. Merge Phase** (stedc_merge* kernels):\n- Iteratively merges pairs of sub-blocks at each level\n- For each merge, solves a rank-1 secular equation to find new eigenvalues\n- Performs deflation to reduce computational cost\n- Updates eigenvectors by solving the secular equation and applying rotations\n\nThis approach achieves O(n^2.5) complexity vs O(n^3) for pure QR iteration.",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """// 1. divide phase
//-----------------------------
rocblas_int groups = (batch_count - 1) / STEDC_BDIM + 1;
ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<S>),
                        dim3(groups), dim3(STEDC_BDIM), 0, stream, levs, blks, n, D + shiftD,
                        strideD, E + shiftE, strideE, batch_count, splits);

// 2. solve phase
//-----------------------------
ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>),
                        dim3(blks, batch_count), dim3(64), 0, stream, levs, blks, 
                        n, D + shiftD, strideD, E + shiftE, strideE, 
                        V, 0, ldv, strideV, info, (S*)work_stack, splits, 
                        eps, ssfmin, ssfmax);

// 3. merge phase
//----------------
size_t lmemsize1 = sizeof(S) * 2 * STEDC_BDIM;
size_t lmemsize3 = sizeof(S) * STEDC_BDIM;
rocblas_int numgrps3 = ((n - 1) / blks + 1) * blks;

// launch merge for level k
for(rocblas_int k = 0; k < levs; ++k)
{
    // a. prepare secular equations
    rocblas_int numgrps2 = 1 << (levs - 1 - k);
    ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<S>),
                            dim3(numgrps2, batch_count), dim3(STEDC_BDIM), lmemsize1, stream, 
                            levs, blks, k, n, D + shiftD, strideD,
                            E + shiftE, strideE, V, 0, ldv, strideV, tmpz, tempgemm, splits,
                            eps);

    // b. solve secular eq to find merged eigenvalues
    ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<S>),
                            dim3(numgrps2, batch_count), dim3(STEDC_BDIM), 0, stream, 
                            levs, blks, k, n, D + shiftD, strideD,
                            E + shiftE, strideE, tmpz, tempgemm, splits, eps, ssfmin, ssfmax);

    // c. find merged eigenvectors
    ROCSOLVER_LAUNCH_KERNEL(
        (stedc_mergeVectors_kernel<STEDC_EXTERNAL_GEMM, S>),
        dim3(numgrps3, batch_count), dim3(STEDC_BDIM), lmemsize3, stream, 
        levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE, V, 0, ldv, strideV, 
        tmpz, tempgemm, splits);

    if(STEDC_EXTERNAL_GEMM)
    {
        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                       &one, V, 0, ldv, strideV, tempgemm, n * n, n, 2 * n * n, &zero,
                       tempgemm, 0, n, 2 * n * n, batch_count, workArr);
    }

    // d. update level
    ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<S>),
                            dim3(numgrps3, batch_count), dim3(STEDC_BDIM), 0, stream, 
                            levs, blks, k, n, D + shiftD, strideD,
                            V, 0, ldv, strideV, tmpz, tempgemm, splits);
}"""
    }]
})

# Entry 3: L1 - Hybrid mode algorithm selection
entries.append({
    "id": str(base_ts + 3),
    "level": "L1",
    "interface": "syevd_heevd",
    "query": "What is the hybrid mode in SYEVD and when is it used?",
    "answer": "Hybrid mode is an optimization where STERF (eigenvalues-only QR iteration) is used instead of STEDC (divide-and-conquer) when only eigenvalues are needed (evect != original). STERF is simpler and faster when eigenvectors aren't required. The mode is controlled by rocsolver_alg_mode setting for rocsolver_function_sterf. When evect == original, STEDC is always used because eigenvectors are needed for the final transformation step.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_syevd_heevd.hpp",
        "language": "cpp",
        "content": """rocsolver_alg_mode sterf_mode;
ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &sterf_mode));

// ... later ...

if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
{
    // only in hybrid mode, compute eigenvalues using sterf
    rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                (rocblas_int*)work1);
}
else
{
    // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
    constexpr bool ISBATCHED = BATCHED || STRIDED;
    const rocblas_int ldw = n;
    const rocblas_stride strideW = n * n;

    rocsolver_stedc_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0, ldw,
        strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);"""
    }]
})

# Entry 4: L2 - Workspace calculation complexity
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "syevd_heevd",
    "query": "How is the workspace calculated for SYEVD/HEEVD? What are the main workspace components?",
    "answer": "SYEVD/HEEVD requires multiple workspace allocations for different algorithm stages:\n\n1. **scalars** - constants for rocBLAS calls\n2. **work1, work2, work3, work4** - reusable workspaces shared between SYTRD, STEDC, and ORMTR\n3. **tmpz** - temporary diagonal and rank-1 modification vector (2*n per batch) for STEDC secular equations\n4. **splits** - sub-block positions and metadata (5*n+2 per batch) for divide-and-conquer\n5. **tmptau_W** - larger of: workspace for SYTRD (Householder reflectors) or n×n matrix for ORMTR updates\n6. **tau** - Householder scalars (n per batch)\n7. **workArr** - array of pointers (batched cases only)\n\nThe work1-4 sizes are computed as maximum of requirements from SYTRD, STEDC, and ORMTR calls. In hybrid mode without eigenvectors, tmpz and splits are not needed.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_syevd_heevd.hpp",
        "language": "cpp",
        "content": """// requirements for tridiagonalization (sytrd/hetrd)
rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w11, &w21, &t1,
                                                &unused, false);

if(alg_mode != rocsolver_alg_mode_hybrid || evect == rocblas_evect_original)
{
    // extra requirements for computing eigenvalues and vectors (stedc)
    rocsolver_stedc_getMemorySize<BATCHED, T, S>(rocblas_evect_tridiagonal, n, batch_count,
                                                 &w31, &w22, &w12, &z1, &s1, &unused);
}

if(evect == rocblas_evect_original)
{
    // extra requirements for ormtr/unmtr
    rocsolver_ormtr_unmtr_getMemorySize<BATCHED, STRIDED, T>(
        rocblas_side_left, uplo, rocblas_operation_none, n, n, batch_count, &unused, &w23, &z2,
        &s2, size_work4, &w13, &w32, &unused, optim_mem);
}

// size of array for temporary matrix products
t2 = sizeof(T) * n * n * batch_count;

// get max values
*size_work1 = std::max({w11, w12, w13});
*size_work2 = std::max({w21, w22, w23});
*size_work3 = std::max(w31, w32);
*size_tmptau_W = std::max(t1, t2);
*size_splits = std::max(s1, s2);
*size_tmpz = std::max(z1, z2);"""
    }]
})

# Entry 5: L1 - Number of levels calculation
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "syevd_heevd",
    "query": "How many levels are used in the divide-and-conquer algorithm for STEDC?",
    "answer": "The number of levels is ceil(log2(n)) - 4 for n > 16, and 0 for n <= 16. This means:\n- n=16: 0 levels, use STEQR directly\n- n=32: 1 level, 2 sub-blocks\n- n=64: 2 levels, 4 sub-blocks\n- n=128: 3 levels, 8 sub-blocks\n- n=256: 4 levels, 16 sub-blocks\n- n=1024: 6 levels, 64 sub-blocks\n\nThe formula ensures sub-blocks are at least ~16 elements, which is the sweet spot for STEQR efficiency.",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
    will be divided during the divide phase of divide & conquer algorithm 
    i.e. number of sub-blocks = 2^levels **/
inline rocblas_int stedc_num_levels(const rocblas_int n)
{
    rocblas_int levels;

    if(n <= 16)
        levels = 0;
    else
        levels = std::ceil(std::log2(n)) - 4;

    return levels;
}"""
    }]
})

# Entry 6: L2 - Deflation in secular equation
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "syevd_heevd",
    "query": "What is deflation in the STEDC merge phase and how is it performed?",
    "answer": "Deflation is an optimization that reduces the degree of the secular equation by identifying and removing eigenvalues that can be computed trivially. Two types of deflation occur in stedc_mergePrepare_kernel:\n\n**1. Zero component deflation**: If |p*z[i]| <= tol, then z[i] is effectively zero. The corresponding diagonal element D[i] is already an eigenvalue and doesn't participate in the secular equation.\n\n**2. Repeated value deflation**: If |D[i] - D[j]| <= tol and both are non-deflated, they represent the same eigenvalue. One is deflated and a Givens rotation eliminates its z component.\n\nThe tolerance is: tol = 8 * eps * max(|D|, |z|)\n\nDeflation can significantly reduce the secular equation degree from n to dd << n, improving performance. Non-deflated values are compacted and sorted to prepare for the secular equation solver.",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """// tol should be  8 * eps * (max diagonal or z element participating in merge)
maxd = inrmsd[0];
maxz = inrmsz[0];
maxd = maxz > maxd ? maxz : maxd;
S tol = 8 * eps * maxd;

// 3. deflate eigenvalues
// ----------------------------------------------------------------
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
// ... (parallel odd-even comparison network) ...
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
}"""
    }]
})

# Entry 7: L3 - Secular equation solving (coding)
entries.append({
    "id": str(base_ts + 7),
    "level": "L3",
    "interface": "syevd_heevd",
    "query": "Implement the core logic for solving a rank-1 secular equation in the STEDC merge phase. Show how to find the eigenvalue given the poles (tmpd), z vector, and rho parameter.",
    "answer": "The secular equation is: 1 + rho * sum(z[i]^2 / (d[i] - lambda)) = 0, where d[i] are the poles (diagonal elements) and lambda is the eigenvalue to find. The solver:\n\n1. Locates the eigenvalue's interval using binary search on the sorted poles\n2. Calls seq_solve or seq_solve_ext to find the root\n3. Updates tmpd with distances (D - lambda) to prevent numerical issues\n4. Re-scales z to avoid bad numerics when eigenvalue is near a pole\n\nThis code shows the parallel solving where each thread finds one eigenvalue independently:",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """// 2. Solve secular eqns, i.e. find the dd zeros
// corresponding to non-deflated new eigenvalues of the merged block
// ----------------------------------------------------------------- 
// each thread will find a different zero in parallel
S a, b;
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

        // computed zero will overwrite 'ev' at the corresponding position.
        // 'tmpd' will be updated with the distances D - lambda_i.
        // deflated values are not changed.
        rocblas_int linfo;

#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
        linfo = slaed4(dd, cc, tmpd + j * n, zz, std::abs(p), ev[j]);
#else
        if(cc == dd - 1)
            linfo = seq_solve_ext(dd, tmpd + j * n, zz, (p < 0 ? -p : p), ev + j, eps,
                                  ssfmin, ssfmax);
        else
            linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps,
                              ssfmin, ssfmax);
#endif
        if(p < 0)
            ev[j] *= -1;
    }
}
__syncthreads();

// Re-scale vector Z to avoid bad numerics when an eigenvalue
// is too close to a pole
for(int i = iam; i < dd; i += bdm)
{
    valf = 1;
    for(int j = 0; j < sz; ++j)
    {
        if(mask[j] == 1)
        {
            valg = tmpd[i + j * n];
            valf *= (per[i] == j) ? valg : valg / (diag[per[i]] - diag[j]);
        }
    }
    valf = sqrt(std::abs(valf));
    zz[i] = zz[i] < 0 ? -valf : valf;
}"""
    }]
})

# Entry 8: L1 - STEDC_BDIM thread block size
entries.append({
    "id": str(base_ts + 8),
    "level": "L1",
    "interface": "syevd_heevd",
    "query": "What is STEDC_BDIM and why is it set to 512?",
    "answer": "STEDC_BDIM = 512 is the number of threads per thread block used in the main STEDC kernels. This size balances:\n- Occupancy: 512 threads allows multiple blocks per CU on AMD GPUs\n- Shared memory: Used for reductions (e.g., 2*512*sizeof(S) for deflation tolerance)\n- Parallelism: Enough threads to cover secular equation solving and vector updates in parallel\n- Register pressure: Not so large that register usage limits occupancy",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """#define STEDC_BDIM 512 // Number of threads per thread-block used in main stedc kernels

// ... later in kernel launch ...

size_t lmemsize1 = sizeof(S) * 2 * STEDC_BDIM;  // For deflation reductions
size_t lmemsize3 = sizeof(S) * STEDC_BDIM;      // For vector norm reductions

ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<S>),
                        dim3(numgrps2, batch_count), dim3(STEDC_BDIM), lmemsize1, stream, 
                        levs, blks, k, n, D + shiftD, strideD,
                        E + shiftE, strideE, V, 0, ldv, strideV, tmpz, tempgemm, splits,
                        eps);"""
    }]
})

# Entry 9: L2 - STEDC_EXTERNAL_GEMM vector update strategy
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "syevd_heevd",
    "query": "What is the STEDC_EXTERNAL_GEMM optimization and how does it work?",
    "answer": "STEDC_EXTERNAL_GEMM is a compile-time flag (currently set to true) that chooses between two strategies for updating eigenvectors after solving the secular equation:\n\n**Internal GEMM (false)**: Each thread block computes V*temp row-by-row using custom reduction code\n**External GEMM (true)**: Uses a single rocBLAS GEMM call to compute the entire n×n×n matrix product\n\nWith external GEMM:\n1. stedc_mergeVectors_kernel computes the secular equation solution vectors and stores them in a padded matrix 'temps'\n2. A single GEMM call multiplies V by temps: vecs = V * temps\n3. stedc_mergeUpdate_kernel copies the result back to V\n\nExternal GEMM is faster because it leverages highly optimized rocBLAS GEMM, but requires additional temporary storage (2*n*n*batch_count).",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """if(STEDC_EXTERNAL_GEMM)
{
    // using external gemms with padded matrices to do the vector update
    // One single full gemm of size n x n x n merges all the blocks in the level
    // TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
    // STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
    // external gemm based updates.
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                   &one, V, 0, ldv, strideV, tempgemm, n * n, n, 2 * n * n, &zero,
                   tempgemm, 0, n, 2 * n * n, batch_count, workArr);
}

// ... in stedc_mergeVectors_kernel ...

if(USEGEMM)
{
    // when using external gemms for the update, we need to
    // put vectors in padded matrix 'temps'
    // (this is to compute 'vecs = C * temps' using external gemm call)
    for(int i = tidb; i < in + sz; i += dim)
    {
        if(i >= in && idd[p2 + j] == 1 && idd[i] == 1)
        {
            dd = 0;
            for(int k = in; k < i; ++k)
            {
                if(idd[k] == 0)
                    dd++;
            }
            temps[pers[i - dd] + in + (p2 + j) * n]
                = vecs[i - dd - in + (p2 + j) * n] / nrm;
        }
        else
            temps[i + (p2 + j) * n] = 0;
    }
}"""
    }]
})

# Entry 10: L2 - Divide kernel sub-block calculation
entries.append({
    "id": str(base_ts + 10),
    "level": "L2",
    "interface": "syevd_heevd",
    "query": "How does stedc_divide_kernel compute the sizes and positions of sub-blocks during the divide phase?",
    "answer": "The divide kernel recursively splits the n×n matrix into 2^levels sub-blocks using a tree structure:\n\n1. **Compute sub-block sizes** (ns array):\n   - Start with ns[0] = n (root)\n   - For each level i from 0 to levs-1:\n     - For each node j at level i:\n       - Split node into two children: ns[2j] = t/2, ns[2j+1] = (t+1)/2 if t is odd, else t/2\n   - This creates a balanced binary tree of sizes\n\n2. **Compute sub-block positions** (ps array):\n   - ps[0] = 0 (first sub-block starts at 0)\n   - For each subsequent sub-block i: ps[i] = ps[i-1] + ns[i-1]\n\n3. **Decouple sub-blocks**:\n   - At each split point p: D[p] -= E[p-1], D[p-1] -= E[p-1]\n   - This removes the coupling between adjacent sub-blocks",
    "code_blocks": [{
        "path": "library/src/auxiliary/rocauxiliary_stedc.hpp",
        "language": "cpp",
        "content": """// temporary arrays in global memory
rocblas_int* splits = splitsA + bid * (5 * n + 2);
// the sub-blocks sizes
rocblas_int* ns = splits + n + 2;
// the sub-blocks initial positions
rocblas_int* ps = ns + n;

// find sizes of sub-blocks
ns[0] = n;
rocblas_int t, t2;
for(int i = 0; i < levs; ++i)
{
    for(int j = (1 << i); j > 0; --j)
    {
        t = ns[j - 1];
        t2 = t / 2;
        ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
        ns[j * 2 - 2] = t2;
    }
}

// find beginning of sub-blocks and update elements in D
rocblas_int p2 = 0;
ps[0] = p2;
for(int i = 1; i < blks; ++i)
{
    p2 += ns[i - 1];
    ps[i] = p2;

    // perform sub-block division
    S p = E[p2 - 1];
    D[p2] -= p;
    D[p2 - 1] -= p;
}"""
    }]
})

# Entry 11: L3 - SYEVD vs SYEV comparison
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "syevd_heevd",
    "query": "Compare SYEVD/HEEVD with SYEV/HEEV. When should each be used and what are the trade-offs?",
    "answer": "**SYEVD/HEEVD (Divide-and-Conquer)**:\n- Complexity: O(n^2.5) for eigenvalues and vectors\n- Memory: Higher workspace requirement (n^2 for vector storage + secular equation workspace)\n- Performance: Faster for large matrices (n > 128), especially when eigenvectors are needed\n- Accuracy: Slightly less accurate due to deflation and secular equation solving\n- Best for: Large dense matrices where performance matters\n\n**SYEV/HEEV (QR Iteration)**:\n- Complexity: O(n^3) for eigenvalues and vectors\n- Memory: Lower workspace requirement (no n^2 storage needed)\n- Performance: Faster for small matrices (n < 128), competitive for eigenvalues-only\n- Accuracy: More accurate, numerically stable QR iteration\n- Best for: Small matrices, when accuracy is critical, or when memory is limited\n\nBoth use SYTRD/HETRD for tridiagonalization (same first step). The difference is in solving the tridiagonal eigenvalue problem: STEDC vs STEQR.\n\nHybrid mode in SYEVD can use STERF (eigenvalues-only) for better performance when vectors aren't needed.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_syevd_heevd.hpp",
        "language": "cpp",
        "content": """// SYEVD/HEEVD: reduce A to tridiagonal form (same as SYEV/HEEV)
rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                        strideE, tau, n, batch_count, scalars, (T*)work1,
                                        (T*)work2, tmptau_W, workArr, false);

if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
{
    // Hybrid mode: use STERF for eigenvalues only (O(n^2))
    rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                (rocblas_int*)work1);
}
else
{
    // Use divide-and-conquer STEDC (O(n^2.5) with vectors)
    rocsolver_stedc_template<false, ISBATCHED, T>(
        handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0, ldw,
        strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

    // Update eigenvectors if needed
    if(evect == rocblas_evect_original)
    {
        rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
            handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda,
            strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count, scalars, (T*)work2, tmpz,
            splits, work4, (T*)work1, (T*)work3, workArr, optim_mem);
    }
}

// Compare with SYEV/HEEV which uses STEQR (QR iteration, O(n^3)):
// rocsolver_steqr_template<T>(handle, evect, n, D, 0, strideD, E, 0, strideE, C,
//                              0, ldc, strideC, info, batch_count, work);"""
    }]
})

# Entry 12: L1 - C API functions (coding)
entries.append({
    "id": str(base_ts + 12),
    "level": "L1",
    "interface": "syevd_heevd",
    "query": "Show the C API functions for SYEVD/HEEVD. What are the precision variants?",
    "answer": "SYEVD/HEEVD has four precision variants: single/double real (ssyevd, dsyevd) and single/double complex (cheevd, zheevd). All follow the same pattern: call rocsolver_syevd_heevd_impl template with appropriate type T. The key parameters are:\n- evect: rocblas_evect_none (values only), rocblas_evect_tridiagonal, or rocblas_evect_original\n- uplo: rocblas_fill_upper or rocblas_fill_lower\n- A: input matrix, overwritten with eigenvectors if evect == original\n- D: output eigenvalues (real, even for complex input)\n- E: workspace for off-diagonal elements\n- info: convergence status",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_syevd_heevd.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_ssyevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<float>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_dsyevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<double>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_cheevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<rocblas_float_complex>(handle, evect, uplo, n, A,
                                                                        lda, D, E, info);
}

rocblas_status rocsolver_zheevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<rocblas_double_complex>(handle, evect, uplo, n, A,
                                                                         lda, D, E, info);
}"""
    }]
})

# Write to JSONL file
output_file = "roclapack_syevd_heevd.jsonl"
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
        assert entry['interface'] == 'syevd_heevd', f"Entry {i} has wrong interface"
        assert 'code_blocks' in entry, f"Entry {i} missing 'code_blocks'"
        for cb in entry['code_blocks']:
            assert 'content' in cb, f"Entry {i} has code_block without 'content'"
            assert 'path' in cb, f"Entry {i} has code_block without 'path'"
    
    print("Schema validation passed!")

# Count levels and coding tasks
l1_count = sum(1 for line in lines if json.loads(line)['level'] == 'L1')
l2_count = sum(1 for line in lines if json.loads(line)['level'] == 'L2')
l3_count = sum(1 for line in lines if json.loads(line)['level'] == 'L3')
coding_count = sum(1 for line in lines if 'coding' in json.loads(line).get('query', '').lower() or 'implement' in json.loads(line).get('query', '').lower())

print(f"\nLevel distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
print(f"Coding tasks: {coding_count}")
