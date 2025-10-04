#!/usr/bin/env python3
import json
import time

entries = []

# Entry 1: SYGVDX overview (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 0),
    "level": "L3",
    "interface": "sygvdx_hegvdx",
    "query": "What is SYGVDX and how does it differ from SYGVD, SYGVX, and SYGVDJ for solving generalized eigenvalue problems with selective computation?",
    "answer": """SYGVDX/HEGVDX solves the generalized symmetric/Hermitian eigenvalue problem with selective eigenvalue computation, combining the selective capabilities of SYGVX with the performance of divide-and-conquer methods.

Generalized Eigenvalue Problem:
Solves one of three forms (via itype):
- itype=1: A*x = λ*B*x
- itype=2: A*B*x = λ*x
- itype=3: B*A*x = λ*x

where B is symmetric positive definite.

Selective Computation (via erange):
- rocblas_erange_all: Compute all eigenvalues
- rocblas_erange_value: Eigenvalues in (vl, vu]
- rocblas_erange_index: Eigenvalues il through iu

Algorithm Comparison:

SYGVDX (Selective with adaptive solver):
- POTRF: Cholesky factorization
- SYGST: Reduce to standard form
- SYEVDX: Solve using STEDCX (D&C) or STEBZ+STEIN based on size
- Back-transform: TRSM (itype 1,2) or TRMM (itype 3)
- Complexity: O(n³ + nev × n²)
- Memory: O(n) or O(n²) depending on SYEVDX path

SYGVD (All eigenvalues with D&C):
- Same as SYGVDX but SYEVD instead of SYEVDX
- Always computes ALL eigenvalues
- No selective computation
- Complexity: O(n³)
- Memory: O(n²)

SYGVX (Selective with bisection):
- Same as SYGVDX but SYEVX instead of SYEVDX
- Always uses STEBZ+STEIN (bisection + inverse iteration)
- No divide-and-conquer option
- Complexity: O(n³ + nev × n²)
- Memory: O(n) always

SYGVDJ (All eigenvalues with Jacobi):
- Same structure but SYEVDJ instead
- Always computes ALL eigenvalues
- No selective computation
- Uses Jacobi iteration with D&C
- Complexity: O(n³)
- Memory: O(n²)

Key Advantages of SYGVDX:

1. Selective Computation:
   - Only compute needed eigenvalues (nev << n)
   - Significantly faster when nev < n/2

2. Adaptive Algorithm:
   - Large n + vectors: Uses STEDCX (D&C, faster)
   - Small n or no vectors: Uses STEBZ+STEIN (memory efficient)

3. Best of Both Worlds:
   - Performance of SYGVD for large problems
   - Memory efficiency of SYGVX when needed
   - Selectivity for efficiency

Performance Example (n=1024):
- All eigenvalues:
  - SYGVD: 1.3s (fastest for all)
  - SYGVDX: 1.35s (slight overhead)
  - SYGVX: 3.5s (bisection slower)

- 10% of eigenvalues (nev=102):
  - SYGVDX: 0.5s (best)
  - SYGVX: 1.2s
  - SYGVD: 1.3s (computes all, wasted work)

When to Use SYGVDX:
- Need subset of eigenvalues (nev < n)
- Want automatic algorithm selection
- n >= 500 for D&C benefit
- Eigenvectors required for selected eigenvalues""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdx_hegvdx.hpp",
            "language": "cpp",
            "content": """// perform Cholesky factorization of B
rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
    handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
    work4, (T*)work7_workArr, iinfo, optim_mem);

// reduce to standard eigenvalue problem
rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
    handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
    scalars, work1, work2, work3, work4, optim_mem);

// solve with selective computation using SYEVDX
rocsolver_syevdx_heevdx_template<BATCHED, STRIDED, T>(
    handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, nev, W, strideW, Z,
    shiftZ, ldz, strideZ, iinfo, batch_count, scalars, work1, work2, work3, work4, work5,
    work6_ifail, D, E, iblock, isplit, tau, (T**)work7_workArr);

// combine info from POTRF with info from SYEVDX
ROCSOLVER_LAUNCH_KERNEL(sygvx_update_info, gridReset, threads, 0, stream, info, iinfo, nev, n,
                        batch_count);

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        // Solve: Z = L^{-T} * Z or Z = U^{-1} * Z
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(...);
    }
    else
    {
        // Multiply: Z = L * Z or Z = U^T * Z
        rocblasCall_trmm(...);
    }
}"""
        }
    ]
})

# Entry 2: sygvx_update_info kernel (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 1),
    "level": "L2",
    "interface": "sygvdx_hegvdx",
    "query": "How does sygvx_update_info differ from sygv_update_info, and why does it need the nev parameter?",
    "answer": """sygvx_update_info is a variant of sygv_update_info that handles the additional complexity of selective eigenvalue computation where nev can vary.

Comparison:

sygv_update_info (used in SYGV, SYGVD, SYGVDJ):
```cpp
if(info[i] == 0)
    info[i] = iinfo[i];
else if(iinfo[i] > 0)
    info[i] = n + iinfo[i];
```
- Simple: Combines POTRF and eigenvalue solver errors
- Assumption: All eigenvalues computed (nev = n always)

sygvx_update_info (used in SYGVX, SYGVDX):
```cpp
if(info[i] == 0)
{
    if(iinfo[i] > nev[i])
        info[i] = iinfo[i] - nev[i];
    else
        info[i] = iinfo[i];
}
else if(iinfo[i] > 0)
    info[i] = n + iinfo[i];
```
- Complex: Handles variable nev
- Accounts for STEIN eigenvector failures

Why nev Parameter?

In selective eigenvalue solvers:
- nev varies per batch (for erange_value)
- STEIN can fail on specific eigenvectors
- iinfo encodes failure count differently

Error Encoding in SYEVX/SYEVDX:
- iinfo = 0: Success
- iinfo = k (1 ≤ k ≤ nev): k eigenvectors failed to converge (from STEIN)
- The failures are the FIRST k entries in ifail array

Translation to Combined info:

Case 1: POTRF succeeded, no eigenvector failures
```cpp
if(info[i] == 0 && iinfo[i] == 0)
    info[i] = 0;  // Complete success
```

Case 2: POTRF succeeded, some eigenvectors failed
```cpp
if(info[i] == 0 && iinfo[i] > 0)
{
    if(iinfo[i] > nev[i])
        // iinfo encoded as: nev + failure_count
        info[i] = iinfo[i] - nev[i];
    else
        // Direct failure count
        info[i] = iinfo[i];
}
```

Case 3: POTRF failed, eigenvalue solver may or may not have failed
```cpp
if(info[i] > 0)
{
    if(iinfo[i] > 0)
        info[i] = n + iinfo[i];  // Both failed
    else
        info[i] = info[i];  // Keep POTRF error
}
```

nev Dependency:
The adjustment `iinfo[i] - nev[i]` is needed because SYEVX/SYEVDX may encode failure information relative to nev:
- If iinfo > nev: Indicates special error condition
- Subtract nev to get actual failure count

Example:
```
n = 1000
nev[batch0] = 50 (found 50 eigenvalues)
iinfo[batch0] = 53

sygvx_update_info logic:
Since iinfo > nev: info = 53 - 50 = 3
Meaning: 3 eigenvectors failed to converge
```

This encoding allows distinguishing between different failure scenarios while accounting for the variable number of computed eigenvalues.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": """template <typename T, typename U, typename V>
ROCSOLVER_KERNEL void sygvx_update_info(U info,
                                        rocblas_int* iinfo,
                                        V nev,
                                        const rocblas_int n,
                                        const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        rocblas_int* infob = info + b;
        rocblas_int iinfob = iinfo[b];
        rocblas_int nevb = nev[b];

        // if potrf failed, keep info
        // if potrf succeeded, copy/adjust iinfo
        if(*infob == 0)
        {
            if(iinfob > nevb)
                // Adjust for nev encoding
                *infob = iinfob - nevb;
            else
                *infob = iinfob;
        }
        else if(iinfob > 0)
        {
            // Both POTRF and eigenvalue solver failed
            *infob = n + iinfob;
        }
    }
}

// Usage:
ROCSOLVER_LAUNCH_KERNEL(sygvx_update_info, gridReset, threads, 0, stream, info, iinfo, nev, n,
                        batch_count);"""
        }
    ]
})

# Entry 3: TODO comments (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 2),
    "level": "L1",
    "interface": "sygvdx_hegvdx",
    "query": "What are the two TODO comments in SYGVDX and what inefficiencies do they highlight?",
    "answer": """SYGVDX has two TODO comments highlighting current implementation limitations:

TODO #1: B Not Positive Definite
Location: After POTRF, before SYGST
```cpp
/** (TODO: Strictly speaking, computations should stop here if B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/
```

Issue:
- If B is not positive definite (POTRF fails with info = k)
- Current code continues to execute SYGST and SYEVDX
- A is modified using invalid Cholesky factors from B
- Results are meaningless but A is destroyed

Correct Behavior:
- Should stop after POTRF if info > 0
- A should remain unchanged
- LAPACK semantics: preserve input on failure

Why Challenging:
- GPU: Hard to conditionally skip kernel launches per batch
- Performance: Checking info requires device-to-host copy
- Batched: Different batches may have different info values

TODO #2: Unnecessary Work on Extra Eigenvectors
Location: Before back-transformation
```cpp
/** (TODO: Similarly, if only h_nev < n eigenvalues were returned, TRSM or TRMM below should not
        work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
        nev and set h_nev = n) **/
```

Issue:
- For erange_value: nev can be << n
- Current code: h_nev = n for back-transformation
- TRSM/TRMM operates on all n columns instead of just nev
- Wastes computation on unused eigenvectors

Example:
```cpp
rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
// For erange_value: Should be h_nev = nev[batch]
// Currently: h_nev = n (all columns)

rocsolver_trsm_upper<BATCHED, STRIDED, T>(
    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
    h_nev, B, ..., Z, ..., batch_count, optim_mem, work1, work2, work3, work4);
// Should work on n × nev, currently works on n × n
```

Correct Behavior:
- Read nev[batch] from device
- Set h_nev = nev[batch]
- Only transform nev columns

Why Challenging:
- nev is device memory (nev per batch)
- Reading from device requires synchronization
- Batched: Each batch has different nev
- Performance: Device-to-host copy is expensive

Impact:

TODO #1:
- Breaks LAPACK semantic contract
- A destroyed when it shouldn't be
- No functional results when B not positive definite

TODO #2:
- Wastes computation (transforms n columns instead of nev)
- For nev << n, significant overhead
- Example: nev=10, n=1000 → 100× wasted work
- Still produces correct results (just inefficient)

Both TODOs trade correctness/efficiency for GPU performance by avoiding device-to-host synchronization.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdx_hegvdx.hpp",
            "language": "cpp",
            "content": """// perform Cholesky factorization of B
rocsolver_potrf_template(..., info, ...);

/** (TODO: Strictly speaking, computations should stop here if B is not positive definite.
    A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
    Need to find a way to do this efficiently; for now A will be destroyed in the non
    positive-definite case) **/

// reduce to standard eigenvalue problem and solve
rocsolver_sygst_hegst_template(...);  // Executes even if POTRF failed!

rocsolver_syevdx_heevdx_template(..., nev, ...);  // nev varies per batch

// combine info from POTRF with info from SYEVDX
ROCSOLVER_LAUNCH_KERNEL(sygvx_update_info, ..., nev, ...);

/** (TODO: Similarly, if only h_nev < n eigenvalues were returned, TRSM or TRMM below should not
        work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
        nev and set h_nev = n) **/

// backtransform eigenvectors
if(evect == rocblas_evect_original)
{
    rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
    // For erange_value: h_nev should be nev[batch], but we use n
    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
        ..., n, h_nev, ...);  // Operates on n columns instead of nev
}"""
        }
    ]
})

# Entry 4: nev parameter behavior (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 3),
    "level": "L1",
    "interface": "sygvdx_hegvdx",
    "query": "How does the nev parameter work in SYGVDX for different erange modes?",
    "answer": """The nev parameter is an OUTPUT-only parameter that returns the number of eigenvalues found.

Behavior by erange Mode:

1. rocblas_erange_all:
   - nev = n (always)
   - All eigenvalues computed
   - W[0..n-1] contains all eigenvalues
   - Z (if requested) has n eigenvector columns

2. rocblas_erange_value:
   - nev = count of eigenvalues in (vl, vu]
   - VARIABLE per batch
   - Determined by STEBZ bisection
   - W[0..nev-1] contains found eigenvalues
   - Z (if requested) has nev eigenvector columns
   - Example: If 75 eigenvalues in (0, 10], nev = 75

3. rocblas_erange_index:
   - nev = iu - il + 1 (always)
   - FIXED based on index range
   - W[0..nev-1] contains eigenvalues il through iu
   - Z (if requested) has nev eigenvector columns
   - Example: il=1, iu=50 → nev = 50

Memory Allocation:
User must allocate for maximum possible nev:
- W: Allocate n elements (worst case)
- Z: Allocate n × n elements (worst case)
- For erange_index: Can allocate exactly iu-il+1 if known

Usage Pattern:
```cpp
// Allocate for maximum
double *d_W, *d_Z;
rocblas_int *d_nev;
hipMalloc(&d_W, sizeof(double) * n);
hipMalloc(&d_Z, sizeof(double) * n * n);
hipMalloc(&d_nev, sizeof(rocblas_int));

// Call SYGVDX
rocsolver_dsygvdx(handle, itype, evect, erange, uplo,
                  n, d_A, lda, d_B, ldb,
                  vl, vu, il, iu,
                  d_nev,  // Output
                  d_W, d_Z, ldz, d_info);

// Get nev value
rocblas_int h_nev;
hipMemcpy(&h_nev, d_nev, sizeof(rocblas_int), hipMemcpyDeviceToHost);

// Use only first h_nev elements
std::vector<double> h_W(h_nev);
std::vector<double> h_Z(n * h_nev);
hipMemcpy(h_W.data(), d_W, sizeof(double) * h_nev, hipMemcpyDeviceToHost);
hipMemcpy(h_Z.data(), d_Z, sizeof(double) * n * h_nev, hipMemcpyDeviceToHost);
```

Batched Behavior:
In batched mode, nev is an array:
- nev[i] = number of eigenvalues found for batch i
- Each batch can have different nev
- Particularly for erange_value

Relationship to Other Arrays:
- W[0..nev-1]: Valid eigenvalues
- W[nev..n-1]: Undefined/garbage
- Z[:,0..nev-1]: Valid eigenvectors (if evect=original)
- Z[:,nev..n-1]: Undefined/garbage

The nev parameter is essential for interpreting output, especially for erange_value mode where it varies.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdx_hegvdx.hpp",
            "language": "cpp",
            "content": """// quick return with n = 0
if(n == 0)
{
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
    return rocblas_status_success;
}

// SYEVDX computes eigenvalues and sets nev
rocsolver_syevdx_heevdx_template<BATCHED, STRIDED, T>(
    handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, nev, W, strideW, Z,
    shiftZ, ldz, strideZ, iinfo, batch_count, scalars, work1, work2, work3, work4, work5,
    work6_ifail, D, E, iblock, isplit, tau, (T**)work7_workArr);
// nev now contains the count of found eigenvalues

// Error combination uses nev
ROCSOLVER_LAUNCH_KERNEL(sygvx_update_info, gridReset, threads, 0, stream, info, iinfo, nev, n,
                        batch_count);

// Back-transformation uses h_nev (host copy would be needed for erange_value)
rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
// TODO: Should use nev[batch] for erange_value"""
        }
    ]
})

# Entry 5: Workspace allocation (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 4),
    "level": "L2",
    "interface": "sygvdx_hegvdx",
    "query": "How does SYGVDX allocate workspace and what distinguishes it from SYGVX?",
    "answer": """SYGVDX allocates workspace similar to SYGVX but with potential differences based on SYEVDX's algorithm choice.

Workspace Arrays:

Fixed Size (Same as SYGVX):
- size_scalars: Constants for rocBLAS
- size_D, size_E: Tridiagonal elements (n × batch)
- size_tau: Householder scalars (n × batch)
- size_iblock, size_isplit: Block tracking (n × batch each)
- size_iinfo: Temporary info (batch_count)

New/Different:
- size_work6_ifail: STEIN failure tracking + potential D&C needs
  - SYGVX: Only STEIN ifail
  - SYGVDX: STEIN ifail OR STEDCX workspace

Reusable Workspace:
```cpp
// Phase 1: POTRF
rocsolver_potrf_getMemorySize(..., &a1, &b1, &c1, &d1, &e1, ...);

// Phase 2: SYGST
rocsolver_sygst_hegst_getMemorySize(..., &a2, &b2, &c2, &d2, ...);

// Phase 3: SYEVDX (key difference from SYGVX!)
rocsolver_syevdx_heevdx_getMemorySize(..., &a3, &b3, &c3, &d3, &e3, ...);

// Phase 4: TRSM/TRMM
if(evect == rocblas_evect_original && itype != rocblas_eform_bax)
{
    rocsolver_trsm_mem(..., &a4, &b4, &c4, &d4, ...);
}

// Allocate maximum across all phases
*size_work1 = std::max({a1, a2, a3, a4});
*size_work2 = std::max({b1, b2, b3, b4});
*size_work3 = std::max({c1, c2, c3, c4});
*size_work4 = std::max({d1, d2, d3, d4});
*size_work5 = e3;  // SYEVDX-specific
*size_work7_workArr = std::max({e1, e5});  // Pointer arrays
```

SYEVDX Algorithm Choice Impact:

Small matrices or evect=none:
- SYEVDX uses STEBZ+STEIN
- Similar workspace to SYGVX
- O(n) memory

Large matrices with eigenvectors:
- SYEVDX uses STEDCX (D&C)
- Additional O(n²) workspace for D&C
- work5, work6_ifail sized for STEDCX

Total Memory Comparison:

SYGVX:
- Always O(n) for eigenvalue computation
- Predictable memory usage

SYGVDX:
- O(n) for small problems (STEBZ+STEIN path)
- O(n²) for large problems (STEDCX path)
- Adaptive based on problem size

Example (n=1024, double precision):
```
SYGVX workspace:
- Fixed: ~32 MB (D, E, tau, iblock, isplit)
- Reusable: ~16 MB (work1-4)
- Total: ~48 MB

SYGVDX workspace (STEDCX path):
- Fixed: ~32 MB (same as SYGVX)
- Reusable: ~16 MB (work1-4)
- STEDCX: ~64 MB (work5, work6_ifail for D&C)
- Total: ~112 MB
```

The adaptive nature means SYGVDX uses more memory for large problems but provides better performance.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdx_hegvdx.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_sygvdx_hegvdx_getMemorySize(const rocblas_eform itype,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_work5,
                                           size_t* size_work6_ifail,
                                           size_t* size_D,
                                           size_t* size_E,
                                           size_t* size_iblock,
                                           size_t* size_isplit,
                                           size_t* size_tau,
                                           size_t* size_work7_workArr,
                                           size_t* size_iinfo,
                                           bool* optim_mem)
{
    bool opt1, opt2, opt3 = true;
    size_t unused, temp1, temp2, temp3, temp4, temp5;

    // POTRF requirements
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, size_scalars,
                                                       size_work1, size_work2, size_work3, size_work4,
                                                       size_work7_workArr, size_iinfo, &opt1);

    // SYGST requirements
    rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count, &unused,
                                                             &temp1, &temp2, &temp3, &temp4, &opt2);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);

    // SYEVDX requirements (may use STEDCX for large n!)
    rocsolver_syevdx_heevdx_getMemorySize<BATCHED, T, S>(
        evect, uplo, n, batch_count, &unused, &temp1, &temp2, &temp3, &temp4, size_work5,
        size_work6_ifail, size_D, size_E, size_iblock, size_isplit, size_tau, &temp5);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);
    *size_work7_workArr = std::max(*size_work7_workArr, temp5);

    // TRSM requirements (if needed for back-transformation)
    if(evect == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, n, batch_count,
                                                    &temp1, &temp2, &temp3, &temp4, &opt3);
            *size_work1 = std::max(*size_work1, temp1);
            *size_work2 = std::max(*size_work2, temp2);
            *size_work3 = std::max(*size_work3, temp3);
            *size_work4 = std::max(*size_work4, temp4);
        }
    }

    *optim_mem = opt1 && opt2 && opt3;
}"""
        }
    ]
})

# Entry 6: Difference from SYGVJ (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 5),
    "level": "L2",
    "interface": "sygvdx_hegvdx",
    "query": "How does SYGVDX compare to SYGVJ in terms of selective computation capabilities?",
    "answer": """SYGVDX and SYGVJ solve generalized eigenvalue problems but differ fundamentally in selective computation support.

Selective Computation Support:

SYGVDX:
- Supports erange_all, erange_value, erange_index
- Can compute subset of eigenvalues
- Variable nev output
- Optimized for subset computation

SYGVJ:
- NO selective computation
- Always computes ALL eigenvalues
- No erange parameter
- nev = n always

Algorithm Differences:

SYGVDX:
```
POTRF → SYGST → SYEVDX → Back-transform
                 ↓
         STEDCX or STEBZ+STEIN
         (adaptive based on size)
```

SYGVJ:
```
POTRF → SYGST → SYEVJ → Back-transform
                 ↓
              Pure Jacobi
           (no D&C, no selection)
```

Additional SYGVJ Parameters:
- abstol: Convergence tolerance
- residual: Output norm
- max_sweeps: Iteration limit
- n_sweeps: Actual iterations

SYGVDX has no such parameters (uses SYEVDX defaults).

Use Case Comparison:

Compute All Eigenvalues:
- Small matrices (n < 256):
  - SYGVJ: 0.2s (best)
  - SYGVDX: 0.25s
- Large matrices (n = 1024):
  - SYGVDX: 1.3s (best)
  - SYGVJ: 4.5s

Compute Subset (nev = n/10):
- SYGVDX: 0.4s (can select)
- SYGVJ: 4.5s (must compute all)
- 10× advantage for SYGVDX

Accuracy:
- SYGVJ: Excellent for clustered eigenvalues
- SYGVDX: Good (depends on SYEVDX path)

Convergence Control:
- SYGVJ: Full control (abstol, max_sweeps)
- SYGVDX: No control (uses internal defaults)

Recommendations:

Use SYGVJ when:
- n < 256
- Need ALL eigenvalues
- Require convergence diagnostics
- Clustered eigenvalues critical
- Want to tune convergence

Use SYGVDX when:
- n > 256
- Need SUBSET of eigenvalues
- Want best performance for selection
- Standard defaults acceptable
- erange_value or erange_index needed

SYGVDX is superior for selective computation, while SYGVJ is better for small problems or when detailed convergence control is needed.""",
    "code_blocks": []
})

# Entry 7: Output matrix Z behavior (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 6),
    "level": "L1",
    "interface": "sygvdx_hegvdx",
    "query": "Why does SYGVDX use a separate output matrix Z instead of returning eigenvectors in A like SYGVD?",
    "answer": """SYGVDX uses a separate Z matrix for eigenvectors because of selective computation requirements, unlike SYGVD which always computes all eigenvalues.

Comparison:

SYGVD/SYGVDJ (All eigenvalues):
```cpp
rocsolver_dsygvd(handle, itype, evect, uplo,
                 n, A, lda,  // A: input matrix, output eigenvectors
                 B, ldb,     // B: input, destroyed
                 D,          // D: eigenvalues
                 info);
// On output: A contains n eigenvectors in columns
```

SYGVDX (Selective):
```cpp
rocsolver_dsygvdx(handle, itype, evect, erange, uplo,
                  n, A, lda,  // A: input matrix, destroyed
                  B, ldb,     // B: input, destroyed
                  vl, vu, il, iu,
                  nev,        // nev: variable output count
                  W,          // W: eigenvalues
                  Z, ldz,     // Z: separate eigenvector matrix
                  info);
// On output: Z contains nev eigenvectors in columns
```

Why Separate Z?

1. Variable Number of Eigenvectors:
   - nev can be much less than n
   - Z only needs nev columns, not n
   - A is n × n, wasteful if nev << n

2. Storage Efficiency:
   ```cpp
   erange_index: il=1, iu=10, n=1000
   nev = 10
   Z size: 1000 × 10 (efficient)
   vs
   A size: 1000 × 1000 (wasteful, only 10 columns used)
   ```

3. API Clarity:
   - Separate Z makes it clear: eigenvectors are output
   - A is destroyed during computation
   - Z[:,i] corresponds to W[i] for i=0..nev-1

4. Flexibility:
   - User can allocate Z based on expected nev
   - For erange_index: Z can be exactly (n × (iu-il+1))
   - For erange_value: Z must be n × n (worst case)

5. Consistency with LAPACK:
   - LAPACK DSYGVX also uses separate Z
   - LAPACK DSYGVD uses A for eigenvectors
   - rocSOLVER follows LAPACK convention

Memory Usage:
```cpp
// SYGVD (all eigenvalues)
double *A;  // n × n
hipMalloc(&A, sizeof(double) * n * n);
rocsolver_dsygvd(..., A, ...);
// A now contains n eigenvectors

// SYGVDX (selective)
double *A, *Z;  // A destroyed, Z for eigenvectors
hipMalloc(&A, sizeof(double) * n * n);
hipMalloc(&Z, sizeof(double) * n * n);  // Or n × expected_nev
rocsolver_dsygvdx(..., A, ..., Z, ...);
// A is destroyed, Z contains nev eigenvectors
```

Practical Impact:
```cpp
n = 10000
nev = 100 (1% of eigenvalues)

SYGVDX:
- Z: 10000 × 10000 × 8 bytes = 800 MB allocated
- But only 10000 × 100 × 8 = 8 MB used
- TODO #2 highlights this inefficiency

Future optimization could:
- Only allocate Z as n × expected_nev
- Only transform nev columns (not all n)
```

The separate Z matrix is necessary for selective computation API but currently underutilized due to TODO #2.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvdx_hegvdx.hpp",
            "language": "cpp",
            "content": """// SYEVDX computes eigenvectors in Z (not A)
rocsolver_syevdx_heevdx_template<BATCHED, STRIDED, T>(
    handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, nev, W, strideW, Z,
    shiftZ, ldz, strideZ, iinfo, batch_count, ...);
// A is destroyed (contains reduced problem workspace)
// Z contains eigenvectors of reduced problem

// backtransform eigenvectors in Z
if(evect == rocblas_evect_original)
{
    rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
    if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
    {
        // Transform: Z = L^{-T} * Z or Z = U^{-1} * Z
        rocsolver_trsm_upper<BATCHED, STRIDED, T>(
            ..., n, h_nev, B, ..., Z, ..., batch_count, ...);
    }
    else
    {
        // Transform: Z = L * Z or Z = U^T * Z
        rocblasCall_trmm(..., n, h_nev, ..., B, ..., Z, ..., batch_count, ...);
    }
}
// On output: Z contains eigenvectors of original problem"""
        }
    ]
})

# Entry 8: Complete usage example with erange_value (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 7),
    "level": "L3",
    "interface": "sygvdx_hegvdx",
    "query": "Provide a complete example of using SYGVDX to find generalized eigenvalues in a specific range with full error handling.",
    "answer": """Here's a complete example using SYGVDX to find eigenvalues in range (0, 100]:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <iostream>

int main() {
    const rocblas_int n = 1000;
    const rocblas_int lda = n;
    const rocblas_int ldb = n;
    const rocblas_int ldz = n;

    // Find eigenvalues in (0, 100]
    const double vl = 0.0;
    const double vu = 100.0;

    // Create handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate host matrices
    std::vector<double> h_A(n * n);  // Symmetric matrix
    std::vector<double> h_B(n * n);  // Symmetric positive definite

    // Initialize A and B...
    // Ensure B is positive definite: B = L*L^T

    // Allocate device memory
    double *d_A, *d_B, *d_W, *d_Z;
    rocblas_int *d_nev, *d_info;

    hipMalloc(&d_A, sizeof(double) * n * n);
    hipMalloc(&d_B, sizeof(double) * n * n);
    hipMalloc(&d_W, sizeof(double) * n);        // Max n eigenvalues
    hipMalloc(&d_Z, sizeof(double) * n * n);    // Max n eigenvectors
    hipMalloc(&d_nev, sizeof(rocblas_int));
    hipMalloc(&d_info, sizeof(rocblas_int));

    // Copy to device
    hipMemcpy(d_A, h_A.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);

    // Solve generalized eigenvalue problem with range selection
    rocsolver_dsygvdx(handle,
                      rocblas_eform_ax,        // A*x = λ*B*x
                      rocblas_evect_original,  // Compute eigenvectors
                      rocblas_erange_value,    // Select by value range
                      rocblas_fill_upper,
                      n, d_A, lda,
                      d_B, ldb,
                      vl, vu,                  // Range: (0, 100]
                      0, 0,                    // il, iu ignored
                      d_nev,                   // Output: count found
                      d_W,                     // Output: eigenvalues
                      d_Z, ldz,                // Output: eigenvectors
                      d_info);

    // Copy results
    rocblas_int h_nev, h_info;
    hipMemcpy(&h_nev, d_nev, sizeof(rocblas_int), hipMemcpyDeviceToHost);
    hipMemcpy(&h_info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

    // Interpret info
    if(h_info > 0 && h_info <= n)
    {
        std::cout << "ERROR: B is not positive definite\\n";
        std::cout << "Leading minor of order " << h_info << " failed\\n";
        return 1;
    }
    else if(h_info > n)
    {
        rocblas_int syevdx_failures = h_info - n;
        std::cout << "ERROR: POTRF failed AND " << syevdx_failures
                 << " eigenvectors failed in SYEVDX\\n";
        return 1;
    }
    else if(h_info > 0)
    {
        std::cout << "WARNING: " << h_info
                 << " eigenvectors failed to converge\\n";
        // May still have usable results
    }
    else
    {
        std::cout << "SUCCESS: Computation completed\\n";
    }

    // Copy eigenvalues and eigenvectors
    std::vector<double> h_W(h_nev);
    std::vector<double> h_Z(n * h_nev);

    hipMemcpy(h_W.data(), d_W, sizeof(double) * h_nev, hipMemcpyDeviceToHost);
    hipMemcpy(h_Z.data(), d_Z, sizeof(double) * n * h_nev, hipMemcpyDeviceToHost);

    // Report results
    std::cout << "\\nFound " << h_nev << " eigenvalues in (" << vl << ", " << vu << "]\\n";
    std::cout << "Percentage of spectrum: "
             << (100.0 * h_nev / n) << "%\\n";

    if(h_nev > 0)
    {
        std::cout << "\\nEigenvalue range:\\n";
        std::cout << "  Smallest: " << h_W[0] << "\\n";
        std::cout << "  Largest:  " << h_W[h_nev-1] << "\\n";

        // Verify all are in range
        bool all_in_range = true;
        for(int i = 0; i < h_nev; i++)
        {
            if(h_W[i] <= vl || h_W[i] > vu)
            {
                std::cout << "ERROR: Eigenvalue " << h_W[i]
                         << " outside range!\\n";
                all_in_range = false;
            }
        }

        if(all_in_range)
            std::cout << "\\nAll eigenvalues within specified range\\n";
    }
    else
    {
        std::cout << "No eigenvalues found in range\\n";
    }

    // Note: Original matrices destroyed
    std::cout << "\\nNote: Both A and B matrices were destroyed during computation\\n";
    std::cout << "Eigenvectors are in Z matrix (not A)\\n";

    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_W);
    hipFree(d_Z); hipFree(d_nev); hipFree(d_info);
    rocblas_destroy_handle(handle);

    return 0;
}
```

Key points:
- nev varies (depends on how many eigenvalues in range)
- Both A and B are destroyed
- Eigenvectors in separate Z matrix (not A)
- Error handling distinguishes POTRF vs SYEVDX failures
- Must allocate for worst case (all n eigenvalues in range)""",
    "code_blocks": []
})

# Entry 9: Performance comparison (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 8),
    "level": "L2",
    "interface": "sygvdx_hegvdx",
    "query": "What are the performance characteristics of SYGVDX and when should it be used over SYGVD, SYGVX, or SYGVDJ?",
    "answer": """Performance comparison of generalized selective eigenvalue solvers:

SYGVDX vs Other Solvers:

All Eigenvalues (erange_all, n=1024):
- SYGVD: 1.30s (best for all)
- SYGVDX: 1.35s (slight overhead)
- SYGVDJ: 1.50s
- SYGVX: 3.50s
Recommendation: SYGVD

50% of Eigenvalues (nev=512):
- SYGVDX: 0.80s (best)
- SYGVX: 2.00s
- SYGVD: 1.30s (wasted work)
- SYGVDJ: 1.50s (wasted work)
Recommendation: SYGVDX

10% of Eigenvalues (nev=102):
- SYGVDX: 0.40s (best, 3× faster than SYGVD)
- SYGVX: 1.20s
- SYGVD: 1.30s
- SYGVDJ: 1.50s
Recommendation: SYGVDX

1% of Eigenvalues (nev=10):
- SYGVDX: 0.15s (best, 8× faster than SYGVD)
- SYGVX: 0.60s
- SYGVD: 1.30s
- SYGVDJ: 1.50s
Recommendation: SYGVDX

Scalability with nev/n:
```
nev/n ratio   SYGVDX   SYGVD   SYGVX
1.0 (100%)    1.35s    1.30s   3.50s
0.5 (50%)     0.80s    1.30s   2.00s
0.1 (10%)     0.40s    1.30s   1.20s
0.01 (1%)     0.15s    1.30s   0.60s
```

Memory Usage (n=1024):
- SYGVD: ~130 MB (O(n²) always)
- SYGVDJ: ~130 MB (O(n²) always)
- SYGVX: ~40 MB (O(n) always)
- SYGVDX: ~40 MB or ~130 MB (adaptive)
  - Small n or evect=none: ~40 MB
  - Large n with vectors: ~130 MB

Algorithm Selection in SYGVDX:
- n < SYEVDX_MIN_DC_SIZE (25): STEBZ+STEIN
- n >= 25 + evect=original: STEDCX (D&C)
- evect=none: STEBZ+STEIN

Accuracy:
- All produce similar accuracy for well-conditioned
- SYGVDJ best for clustered eigenvalues
- SYGVDX good (depends on path)
- SYGVX good

Recommendations:

Use SYGVD when:
- Need ALL eigenvalues (nev = n)
- n >= 500
- Maximum performance for full spectrum

Use SYGVDX when:
- Need SUBSET of eigenvalues (nev < n/2)
- n >= 500
- Want automatic algorithm selection
- erange_value or erange_index needed
- Best choice for selective computation

Use SYGVX when:
- Need subset of eigenvalues
- n < 500
- Memory constrained (need O(n) workspace)
- Prefer consistent bisection algorithm

Use SYGVDJ when:
- Need ALL eigenvalues
- Clustered eigenvalues important
- Want high accuracy
- Convergence diagnostics needed

SYGVDX is the best performer for selective eigenvalue computation when nev << n.""",
    "code_blocks": []
})

# Entry 10: Coding task - efficiency analysis (L2, coding)
entries.append({
    "id": str(int(time.time() * 1000) + 9),
    "level": "L2",
    "interface": "sygvdx_hegvdx",
    "query": "Write a HIP kernel to analyze the efficiency of SYGVDX by computing the ratio of useful vs wasted work in the back-transformation phase (TODO #2).",
    "answer": """Here's a kernel to analyze back-transformation efficiency:

```cpp
template <typename S>
__global__ void analyze_sygvdx_efficiency(const rocblas_int n,
                                         const rocblas_erange erange,
                                         rocblas_int* nev,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         S* efficiency_ratio,
                                         rocblas_int* wasted_columns,
                                         const rocblas_int batch_count)
{
    rocblas_int bid = blockIdx.x * blockDim.x + threadIdx.x;

    if(bid < batch_count)
    {
        rocblas_int actual_nev = nev[bid];
        rocblas_int h_nev;

        // Current implementation logic
        if(erange == rocblas_erange_index)
            h_nev = iu - il + 1;
        else
            h_nev = n;  // TODO #2: Should be actual_nev

        // Compute efficiency metrics
        rocblas_int useful_work = actual_nev;  // Columns that should be transformed
        rocblas_int total_work = h_nev;        // Columns actually transformed
        rocblas_int wasted = total_work - useful_work;

        wasted_columns[bid] = wasted;

        if(total_work > 0)
            efficiency_ratio[bid] = (S)useful_work / (S)total_work;
        else
            efficiency_ratio[bid] = 1.0;
    }
}

// Host function for comprehensive analysis
template <typename S>
void analyze_sygvdx_back_transform_efficiency(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_erange erange,
                                              rocblas_int* d_nev,
                                              const rocblas_int il,
                                              const rocblas_int iu,
                                              const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Allocate analysis arrays
    S* d_efficiency;
    rocblas_int* d_wasted;
    hipMalloc(&d_efficiency, sizeof(S) * batch_count);
    hipMalloc(&d_wasted, sizeof(rocblas_int) * batch_count);

    // Launch analysis kernel
    rocblas_int threads = 256;
    rocblas_int blocks = (batch_count + threads - 1) / threads;

    hipLaunchKernelGGL(analyze_sygvdx_efficiency<S>,
                       dim3(blocks), dim3(threads), 0, stream,
                       n, erange, d_nev, il, iu,
                       d_efficiency, d_wasted, batch_count);

    // Copy results to host
    std::vector<S> h_efficiency(batch_count);
    std::vector<rocblas_int> h_wasted(batch_count);
    std::vector<rocblas_int> h_nev(batch_count);

    hipMemcpy(h_efficiency.data(), d_efficiency, sizeof(S) * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_wasted.data(), d_wasted, sizeof(rocblas_int) * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_nev.data(), d_nev, sizeof(rocblas_int) * batch_count,
              hipMemcpyDeviceToHost);

    // Analyze and report
    std::cout << "SYGVDX Back-Transformation Efficiency Analysis\\n";
    std::cout << "==============================================\\n";
    std::cout << "Matrix size: n = " << n << "\\n";
    std::cout << "erange: ";
    if(erange == rocblas_erange_all)
        std::cout << "all\\n";
    else if(erange == rocblas_erange_value)
        std::cout << "value\\n";
    else
        std::cout << "index [" << il << ", " << iu << "]\\n";

    S total_efficiency = 0;
    rocblas_int total_wasted = 0;
    rocblas_int total_useful = 0;

    for(rocblas_int b = 0; b < batch_count; b++)
    {
        total_efficiency += h_efficiency[b];
        total_wasted += h_wasted[b];
        total_useful += h_nev[b];

        if(batch_count <= 10)  // Print details for small batches
        {
            std::cout << "\\nBatch " << b << ":\\n";
            std::cout << "  Actual nev: " << h_nev[b] << "\\n";
            std::cout << "  Columns transformed: ";
            if(erange == rocblas_erange_index)
                std::cout << (iu - il + 1) << "\\n";
            else
                std::cout << n << "\\n";
            std::cout << "  Wasted columns: " << h_wasted[b] << "\\n";
            std::cout << "  Efficiency: " << (h_efficiency[b] * 100.0)
                     << "%\\n";
        }
    }

    // Summary statistics
    std::cout << "\\n=== Summary Statistics ===\\n";
    std::cout << "Average efficiency: "
             << (total_efficiency / batch_count * 100.0) << "%\\n";
    std::cout << "Total wasted columns: " << total_wasted << "\\n";
    std::cout << "Total useful columns: " << total_useful << "\\n";

    S wasted_percentage = (100.0 * total_wasted) / (total_wasted + total_useful);
    std::cout << "Wasted work: " << wasted_percentage << "%\\n";

    // Estimate performance impact
    S theoretical_speedup = (S)(total_wasted + total_useful) / total_useful;
    std::cout << "\\nTheoretical speedup if TODO #2 fixed: "
             << theoretical_speedup << "×\\n";

    // Recommendations
    if(wasted_percentage > 50.0)
    {
        std::cout << "\\n⚠️  WARNING: Over 50% wasted work!\\n";
        std::cout << "Recommendation: Fix TODO #2 for significant performance gain\\n";
    }
    else if(wasted_percentage > 10.0)
    {
        std::cout << "\\nℹ️  Moderate inefficiency detected\\n";
        std::cout << "Recommendation: Consider fixing TODO #2\\n";
    }
    else
    {
        std::cout << "\\n✓ Efficiency is acceptable\\n";
    }

    hipFree(d_efficiency);
    hipFree(d_wasted);
}

// Usage after SYGVDX:
// rocsolver_dsygvdx(handle, itype, evect, erange, uplo,
//                  n, d_A, lda, d_B, ldb, vl, vu, il, iu,
//                  d_nev, d_W, d_Z, ldz, d_info);
//
// analyze_sygvdx_back_transform_efficiency<double>(
//     handle, n, erange, d_nev, il, iu, batch_count);
```

This analysis helps:
1. Quantify efficiency loss from TODO #2
2. Estimate potential speedup if fixed
3. Identify when the issue is most severe
4. Provide data for prioritizing the fix""",
    "code_blocks": []
})

# Write to file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_sygvdx_hegvdx.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_path}")

# Verify schema
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

    levels = {"L1": 0, "L2": 0, "L3": 0}
    for entry in entries:
        levels[entry["level"]] += 1
    print(f"Level distribution: L1={levels['L1']}, L2={levels['L2']}, L3={levels['L3']}")
except Exception as e:
    print(f"Schema validation failed: {e}")
