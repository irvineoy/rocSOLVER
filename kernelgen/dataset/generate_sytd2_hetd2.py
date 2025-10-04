#!/usr/bin/env python3
import json
import time

# Get base timestamp
base_ts = int(time.time() * 1000)

entries = []

# Entry 1: L3 - Complete SYTD2/HETD2 algorithm
entries.append({
    "id": str(base_ts + 1),
    "level": "L3",
    "interface": "sytd2_hetd2",
    "query": "Explain the complete unblocked tridiagonalization algorithm used in SYTD2/HETD2. How does it reduce a symmetric/Hermitian matrix to tridiagonal form?",
    "answer": "SYTD2/HETD2 reduces a symmetric (real) or Hermitian (complex) matrix A to tridiagonal form using Householder reflections. The algorithm proceeds column-by-column:\n\n**For lower triangular (uplo = lower)**, running forwards j=0 to n-2:\n1. Generate Householder reflector H to annihilate A(j+2:n-1, j)\n2. Compute w = τ*A*v - (1/2)*τ²*(v'*A*v)*v where v is the Householder vector\n3. Apply rank-2 update: A := A - v*w' - w*v'\n4. Store τ in tau[j] and the first off-diagonal element in E[j]\n\n**For upper triangular (uplo = upper)**, running backwards j=n-1 to 1:\n1. Generate Householder reflector H to annihilate A(0:j-2, j)\n2. Compute w = τ*A*v - (1/2)*τ²*(v'*A*v)*v\n3. Apply rank-2 update: A := A - v*w' - w*v'\n4. Store τ in tau[j-1] and the first off-diagonal element in E[j-1]\n\nThe result is a tridiagonal matrix with diagonal in D, off-diagonal in E, and Householder vectors stored in the unused triangle of A. This is the unblocked version (O(n³)), while SYTRD/HETRD is blocked (faster for large matrices).",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_lower)
{
    // reduce the lower part of A
    // main loop running forwards (for each column)
    for(rocblas_int j = 0; j < n - 1; ++j)
    {
        // 1. generate Householder reflector to annihilate A(j+2:n-1,j) and copy off-diagonal element to E[j]
        rocsolver_larfg_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                    strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda),
                                    1, strideA, tmptau, stridet, batch_count, work, norms);

        // 2. overwrite tau with w = tmptau*A*v - 1/2*tmptau*(tmptau*v'*A*v)*v
        rocblasCall_symv_hemv<T>(handle, uplo, n - 1 - j, tmptau, stridet, A,
                                 shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                                 shiftA + idx2D(j + 1, j, lda), 1, strideA, scalars + 1, 0, tau,
                                 j, 1, strideP, batch_count, work, workArr);

        ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<64, T>), dim3(1, 1, batch_count),
                                dim3(64, 1, 1), 0, stream, n - 1 - j, A,
                                shiftA + idx2D(j + 1, j, lda), strideA, tau, j, strideP, tmptau,
                                stridet);

        // 3. apply the Householder reflector to A as a rank-2 update:
        // A = A - v*w' - w*v'
        rocblasCall_syr2_her2<T>(handle, uplo, n - 1 - j, scalars, A,
                                 shiftA + idx2D(j + 1, j, lda), 1, strideA, tau, j, 1, strideP,
                                 A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                 batch_count, workArr);

        // 4. Save the used householder scalar
        ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                tau + j, strideP);
    }
}"""
    }]
})

# Entry 2: L2 - Small kernel optimization
entries.append({
    "id": str(base_ts + 2),
    "level": "L2",
    "interface": "sytd2_hetd2",
    "query": "What is the small kernel optimization in SYTD2/HETD2 and when is it used?",
    "answer": "SYTD2/HETD2 has a specialized kernel (sytd2_lower_kernel_small / sytd2_upper_kernel_small) for small remaining matrices. The optimization activates when:\n1. Shared memory requirement fits: lmemsize <= sharedMemPerBlock\n2. Matrix size is small enough: nn <= xxTD2_SSKER_MAX_N\n\nWhen active, the kernel:\n- Loads the entire remaining submatrix into shared memory (LDS)\n- Performs all remaining tridiagonalization steps in a single kernel launch\n- Uses 256 threads with warp-level reductions for LARFG, SYMV, DOT operations\n- Writes results back to global memory\n\nThis eliminates kernel launch overhead and global memory traffic for the tail of the algorithm. The LDS requirement is: ((256/warpSize) + 2*nn + 1 + nn²) * sizeof(T), dominated by the nn² storage for matrix A.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """for(rocblas_int j = 0; j < n - 1; ++j)
{
    const rocblas_int nn = n - j;
    const size_t lmemsize = ((256 / props.warpSize) + 2 * nn + 1 + nn * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn <= xxTD2_SSKER_MAX_N)
    {
        ROCSOLVER_LAUNCH_KERNEL((sytd2_lower_kernel_small<256, T>), dim3(1, 1, batch_count),
                                dim3(256), lmemsize, stream, nn, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, D + j, strideD,
                                E + j, strideE, tau + j, strideP);

        break;
    }
    // ... regular path with multiple kernel launches ...
}"""
    }]
})

# Entry 3: L1 - Rank-2 update computation
entries.append({
    "id": str(base_ts + 3),
    "level": "L1",
    "interface": "sytd2_hetd2",
    "query": "Why is the rank-2 update in SYTD2 computed as A := A - v*w' - w*v' instead of A := H*A*H'?",
    "answer": "The rank-2 update A := A - v*w' - w*v' is mathematically equivalent to H*A*H' where H = I - τ*v*v', but is much more efficient:\n\n1. Direct H*A*H' would be O(n³) with poor cache behavior\n2. The rank-2 form uses SYR2/HER2 (symmetric rank-2 update) which is O(n²) for each column\n3. w is carefully computed as w = τ*A*v - (1/2)*τ²*(v'*A*v)*v to ensure the update maintains symmetry\n4. Only the lower/upper triangle needs to be updated (symmetric property)\n\nThis is a classic LAPACK optimization that reduces work and improves numerical stability.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """// ----- 2. compute w = tau*A*v - 1/2*tau*tau*(v'*A*v)*v -----
// symv
for(I i = tid; i < nn; i += MAX_THDS)
{
    T temp = 0;
    T* Atmp = a + (j + 1) + (j + 1) * n;
    for(I jj = 0; jj < nn; jj++)
        temp += Atmp[i + jj * n] * x[jj];
    w[i] = tmptau[0] * temp;
}

// dot: compute v'*A*v
norm2 = 0;
for(I i = tid; i < nn; i += MAX_THDS)
    norm2 += x[i] * conj(w[i]);
// ... reduction ...
if(tid == 0)
{
    for(I k = 1; k < MAX_THDS / warpSize; k++)
        norm2 += sval[k];
    sval[0] = -0.5 * tmptau[0] * norm2;
}
__syncthreads();

// axpy: w := w - (1/2*tau²*v'*A*v)*v
for(I i = tid; i < nn; i += MAX_THDS)
    w[i] += sval[0] * x[i];
__syncthreads();

// ----- 3. apply the Householder reflector to A as a rank-2 update: A = A - v*w' - w*v' -----
// syr2
for(I i = tid; i < nn; i += MAX_THDS)
{
    for(I jj = 0; jj < nn; jj++)
    {
        T* Atmp = a + (j + 1) + (j + 1) * n;
        Atmp[i + jj * n] = Atmp[i + jj * n] - x[i] * conj(w[jj]) - w[i] * conj(x[jj]);
    }
}"""
    }]
})

# Entry 4: L2 - Hermitian diagonal handling
entries.append({
    "id": str(base_ts + 4),
    "level": "L2",
    "interface": "sytd2_hetd2",
    "query": "How does HETD2 handle the diagonal elements of Hermitian matrices differently from SYTD2?",
    "answer": "Hermitian matrices have real diagonal elements by definition (imaginary part must be zero). HETD2 handles this in two ways:\n\n1. **In the small kernel**: Before tridiagonalization, diagonal elements are forced to real: `a[i + j*n] = std::real(a[i + j*n])`\n\n2. **In set_tridiag kernel**: When copying diagonal to D, the real part is extracted: `tmp = a[i + i*lda].real(); d[i] = tmp; a[i + i*lda] = T(tmp);`\n\nNote: If the input matrix is not truly Hermitian (has imaginary diagonal), the imaginary part is simply ignored. The comment warns that the resulting tridiagonal form won't have the same eigenvalues as the original non-Hermitian matrix.\n\nFor SYTD2 (real symmetric), no special handling is needed since diagonal elements are already real.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D,
                                  const rocblas_stride strideD,
                                  S* E,
                                  const rocblas_stride strideE)
{
    // ... 
    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        /* (Hermitian matrices have real terms in the diagonal. In the case where the input
            matrix is not Hermitian, we are simply ignoring the imaginary part so that outputs
            in A and E coincide with the outputs generated by other similar libraries.
            -- Note that the resulting tridiagonal form is not really similar to the original A;
            they will not have the same eigenvalues) */
        tmp = a[i + i * lda].real();
        d[i] = tmp;
        a[i + i * lda] = T(tmp);

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}"""
    }]
})

# Entry 5: L1 - Upper vs lower triangular direction
entries.append({
    "id": str(base_ts + 5),
    "level": "L1",
    "interface": "sytd2_hetd2",
    "query": "Why does SYTD2 process columns forwards (j=0 to n-2) for lower triangular but backwards (j=n-1 to 1) for upper triangular?",
    "answer": "The direction matches the storage pattern:\n\n**Lower triangular**: Householder vectors are stored below the diagonal. Processing column j annihilates A(j+2:n-1, j) and stores the vector in A(j+1:n-1, j). Forward iteration naturally builds from left to right.\n\n**Upper triangular**: Householder vectors are stored above the diagonal. Processing column j annihilates A(0:j-2, j) and stores the vector in A(0:j-1, j). Backward iteration naturally builds from right to left.\n\nThis ensures each iteration only modifies unprocessed parts of the matrix, avoiding conflicts with stored Householder vectors from previous iterations.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """if(uplo == rocblas_fill_lower)
{
    // reduce the lower part of A
    // main loop running forwards (for each column)
    for(rocblas_int j = 0; j < n - 1; ++j)
    {
        // 1. generate Householder reflector to annihilate A(j+2:n-1,j)
        rocsolver_larfg_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                    strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda),
                                    1, strideA, tmptau, stridet, batch_count, work, norms);
        // ...
    }
}
else
{
    // reduce the upper part of A
    // main loop running backwards (for each column)
    for(rocblas_int j = n - 1; j > 0; --j)
    {
        // 1. generate Householder reflector to annihilate A(0:j-2,j)
        rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                    strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                    1, batch_count, work, norms);
        // ...
    }
}"""
    }]
})

# Entry 6: L2 - Workspace calculation (coding)
entries.append({
    "id": str(base_ts + 6),
    "level": "L2",
    "interface": "sytd2_hetd2",
    "query": "Implement the workspace size calculation for SYTD2/HETD2. What are the different workspace components and how are they sized?",
    "answer": "The workspace calculation determines memory requirements for different operations:\n\n1. **size_scalars**: Constants for rocBLAS calls (3 * sizeof(T))\n2. **size_tmptau**: Temporary Householder scalars (batch_count * sizeof(T))\n3. **size_workArr**: Array of pointers (batched only: batch_count * sizeof(T*))\n4. **size_norms**: For LARFG norm storage, max of:\n   - n1: LARFG requirements\n   - n2: sizeof(T) * batch_count\n5. **size_work**: Reusable workspace, max of:\n   - w1: LARFG workspace\n   - w2: SYMV/HEMV workspace\n   - w3: DOT workspace = ((n-2)/512 + 2) * sizeof(T) * batch_count\n\nThe workspace is reused across iterations, taking the maximum of all requirements.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """template <bool BATCHED, typename T>
void rocsolver_sytd2_hetd2_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_tmptau = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0;

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of array to store temporary householder scalars
    *size_tmptau = sizeof(T) * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements to call larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    *size_norms = std::max(n1, n2);
    *size_work = std::max({w1, w2, w3});
}"""
    }]
})

# Entry 7: L3 - Small kernel shared memory layout (coding)
entries.append({
    "id": str(base_ts + 7),
    "level": "L3",
    "interface": "sytd2_hetd2",
    "query": "Show the shared memory layout and loading strategy in sytd2_lower_kernel_small. How is the matrix loaded and why?",
    "answer": "The small kernel uses shared memory (LDS) to hold the entire working submatrix and temporary vectors. The layout is:\n\n```\nlmem: [tmptau(1) | a(nn×nn) | x(nn) | w(nn) | sval(warp_count)]\n```\n\n**Loading strategy**:\n- Uses 256 threads split into 2 groups (MAX_THDS/2 = 128 each)\n- Each thread loads multiple elements: `i = tid % 128; tidy = tid / 128`\n- Loop over rows i with stride 128, loop over columns j with stride 2\n- This coalesced access pattern maximizes memory bandwidth\n\n**Triangle handling**:\n- After loading, copies lower triangle to upper (for symmetric access in computations)\n- For Hermitian, forces diagonal to real: `a[i+j*n] = std::real(a[i+j*n])`\n\nThe entire submatrix fits in LDS, eliminating global memory traffic during tridiagonalization.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    sytd2_lower_kernel_small(const I n, ...)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    
    // shared variables
    extern __shared__ double lmem[];
    T* tmptau = reinterpret_cast<T*>(lmem);
    T* a = reinterpret_cast<T*>(tmptau + 1);
    T* x = reinterpret_cast<T*>(a + n * n);
    T* w = reinterpret_cast<T*>(x + n);
    T* sval = reinterpret_cast<T*>(w + n);

    // load A to lds
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            a[i + j * n] = A[i + j * lda];
        }
    }

    __syncthreads();

    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            // ignore imaginary part of the diagonal
            if(i == j)
                a[i + j * n] = std::real(a[i + j * n]);
            // copy lower triangle to upper triangle
            if(i < j)
                a[i + j * n] = conj(a[j + i * n]);
        }
    }

    __syncthreads();
    
    // ... perform tridiagonalization in LDS ...
}"""
    }]
})

# Entry 8: L1 - Warp-level reduction for LARFG
entries.append({
    "id": str(base_ts + 8),
    "level": "L1",
    "interface": "sytd2_hetd2",
    "query": "How does the small kernel perform warp-level reductions for computing the Householder reflector norm?",
    "answer": "The small kernel uses warp shuffle instructions (shift_left) for fast intra-warp reductions:\n\n1. Each thread computes partial sum: `norm2 += x[i] * conj(x[i])` for its assigned elements\n2. Warp-level reduction using shift_left by powers of 2: 1, 2, 4, 8, 16, 32 (for 64-thread warps)\n3. Lane 0 of each warp writes its result to shared memory: `sval[tid/warpSize]`\n4. Thread 0 sums across warps: `for(k=1; k < MAX_THDS/warpSize; k++) norm2 += sval[k]`\n5. Thread 0 computes tau and scaling factor, broadcasts via shared memory\n\nThis avoids atomic operations and minimizes shared memory bank conflicts. The same pattern is used for DOT products in the w computation.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """// larfg
T norm2 = 0;
for(I i = tid; i < nn - 1; i += MAX_THDS)
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
    run_set_taubeta<T>(tmptau, &norm2, x, E + j);

    tau[j] = tmptau[0];
    sval[0] = norm2;
}
__syncthreads();"""
    }]
})

# Entry 9: L2 - latrd_dot_scale_axpy kernel
entries.append({
    "id": str(base_ts + 9),
    "level": "L2",
    "interface": "sytd2_hetd2",
    "query": "What does the latrd_dot_scale_axpy kernel do in the SYTD2 algorithm?",
    "answer": "The latrd_dot_scale_axpy kernel performs the critical step: w := w - (1/2)*τ²*(v'*A*v)*v\n\nThis modifies the intermediate vector w (stored in tau) to the final form needed for the rank-2 update. The kernel:\n1. Computes dot product: alpha = v'*w = sum(v[i]*conj(w[i]))\n2. Computes scaling factor: beta = -0.5 * tmptau * alpha\n3. Performs axpy: w[i] := w[i] + beta*v[i]\n\nThis ensures that the rank-2 update A - v*w' - w*v' maintains symmetry/Hermitian property. Without this correction term, numerical errors would accumulate and break symmetry.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """// 2. overwrite tau with w = tmptau*A*v - 1/2*tmptau*(tmptau*v'*A*v)*v
rocblasCall_symv_hemv<T>(handle, uplo, n - 1 - j, tmptau, stridet, A,
                         shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                         shiftA + idx2D(j + 1, j, lda), 1, strideA, scalars + 1, 0, tau,
                         j, 1, strideP, batch_count, work, workArr);

ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<64, T>), dim3(1, 1, batch_count),
                        dim3(64, 1, 1), 0, stream, n - 1 - j, A,
                        shiftA + idx2D(j + 1, j, lda), strideA, tau, j, strideP, tmptau,
                        stridet);"""
    }]
})

# Entry 10: L1 - set_tridiag kernel
entries.append({
    "id": str(base_ts + 10),
    "level": "L1",
    "interface": "sytd2_hetd2",
    "query": "What does the set_tridiag kernel do and when is it called?",
    "answer": "The set_tridiag kernel is called at the end of SYTD2/HETD2 to finalize the tridiagonal form. It:\n1. Copies diagonal elements from A to D: `d[i] = a[i + i*lda]`\n2. Copies off-diagonal elements from E back to A:\n   - Lower: `a[i+1 + i*lda] = e[i]`\n   - Upper: `a[i + (i+1)*lda] = e[i]`\n3. For Hermitian (complex), extracts real part of diagonal: `tmp = a[i+i*lda].real(); d[i] = tmp; a[i+i*lda] = T(tmp)`\n\nAfter this, A contains the tridiagonal matrix in the appropriate triangle with Householder vectors stored in the unused triangle, D contains diagonal, and E contains off-diagonal.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """template <typename T, typename S, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A, const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D, const rocblas_stride strideD,
                                  S* E, const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    bool lower = (uplo == rocblas_fill_lower);

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        d[i] = a[i + i * lda];

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}"""
    }]
})

# Entry 11: L3 - SYTD2 vs SYTRD comparison
entries.append({
    "id": str(base_ts + 11),
    "level": "L3",
    "interface": "sytd2_hetd2",
    "query": "Compare SYTD2/HETD2 (unblocked) with SYTRD/HETRD (blocked). When should each be used?",
    "answer": "**SYTD2/HETD2 (Unblocked)**:\n- Complexity: O(n³) with poor cache behavior\n- Memory: Minimal workspace (no blocking)\n- Performance: Good for small matrices (n < 64) due to small kernel optimization\n- Implementation: Simple, processes one column at a time\n- Best for: Small matrices, or as subroutine in blocked algorithms\n\n**SYTRD/HETRD (Blocked)**:\n- Complexity: O(n³) but with Level 3 BLAS (better cache utilization)\n- Memory: Higher workspace for blocking\n- Performance: Much faster for large matrices (n > 128)\n- Implementation: Uses LATRD for panel factorization, SYTD2 for tails\n- Best for: Large matrices where performance matters\n\nSYTRD achieves better performance by:\n1. Processing nb columns at once (panel factorization with LATRD)\n2. Using Level 3 BLAS (GEMM/SYMM) for trailing matrix updates\n3. Falling back to SYTD2 for small tail blocks\n\nThe small kernel optimization in SYTD2 makes it competitive for the tail, avoiding the overhead of blocked algorithms on small matrices.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.hpp",
        "language": "cpp",
        "content": """// SYTD2: Unblocked algorithm, one column at a time
for(rocblas_int j = 0; j < n - 1; ++j)
{
    // Check if small enough for optimized kernel
    const rocblas_int nn = n - j;
    const size_t lmemsize = ((256 / props.warpSize) + 2 * nn + 1 + nn * nn) * sizeof(T);
    if(lmemsize <= props.sharedMemPerBlock && nn <= xxTD2_SSKER_MAX_N)
    {
        // Use single-kernel optimization for small tail
        ROCSOLVER_LAUNCH_KERNEL((sytd2_lower_kernel_small<256, T>), dim3(1, 1, batch_count),
                                dim3(256), lmemsize, stream, nn, A,
                                shiftA + idx2D(j, j, lda), lda, strideA, D + j, strideD,
                                E + j, strideE, tau + j, strideP);
        break;
    }

    // Regular unblocked path: LARFG + SYMV + SYR2 for each column
    rocsolver_larfg_template<T>(...);
    rocblasCall_symv_hemv<T>(...);
    ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<64, T>), ...);
    rocblasCall_syr2_her2<T>(...);
}

// Compare with SYTRD blocked algorithm:
// for(rocblas_int j = 0; j < n - nb; j += nb)
// {
//     rocsolver_latrd_template(...);  // Panel factorization (nb columns)
//     rocblasCall_symm(...);           // Level 3 BLAS update
// }
// rocsolver_sytd2_template(...);      // Unblocked for tail"""
    }]
})

# Entry 12: L1 - C API functions
entries.append({
    "id": str(base_ts + 12),
    "level": "L1",
    "interface": "sytd2_hetd2",
    "query": "Show the C API functions for SYTD2/HETD2. What are the differences between symmetric and Hermitian versions?",
    "answer": "SYTD2/HETD2 has four variants:\n- **ssytd2, dsytd2**: Real symmetric (SYTD2)\n- **chetd2, zhetd2**: Complex Hermitian (HETD2)\n\nKey differences:\n1. **Input matrix A**: float/double for SY*, rocblas_float_complex/rocblas_double_complex for HE*\n2. **Output D, E**: Always real (float/double) even for complex input\n3. **tau**: Same type as A (complex for Hermitian)\n\nThe Hermitian versions must handle conjugate transpose operations and ensure diagonal elements are real. The algorithm is otherwise identical.",
    "code_blocks": [{
        "path": "library/src/lapack/roclapack_sytd2_hetd2.cpp",
        "language": "cpp",
        "content": """rocblas_status rocsolver_ssytd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<float>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_dsytd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<double>(handle, uplo, n, A, lda, D, E, tau);
}

rocblas_status rocsolver_chetd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<rocblas_float_complex>(handle, uplo, n, A, lda, D,
                                                                        E, tau);
}

rocblas_status rocsolver_zhetd2(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tau)
{
    return rocsolver::rocsolver_sytd2_hetd2_impl<rocblas_double_complex>(handle, uplo, n, A, lda, D,
                                                                         E, tau);
}"""
    }]
})

# Write to JSONL file
output_file = "roclapack_sytd2_hetd2.jsonl"
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
        assert entry['interface'] == 'sytd2_hetd2', f"Entry {i} has wrong interface"
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
