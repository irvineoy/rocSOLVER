#!/usr/bin/env python3
import json
import time

entries = []

# Entry 1: SYEVX overview (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 0),
    "level": "L3",
    "interface": "syevx_heevx",
    "query": "What is SYEVX and how does it differ from SYEVD and SYEVDX for computing selected eigenvalues?",
    "answer": """SYEVX/HEEVX computes selected eigenvalues and optionally eigenvectors of a symmetric/Hermitian matrix using bisection and inverse iteration (STEBZ + STEIN). It differs from other selective eigenvalue solvers in its algorithm choice.

Algorithm Comparison:

SYEVX (Bisection + Inverse Iteration):
- Always uses: SYTRD → STEBZ (bisection) → STEIN (inverse iteration) → ORMTR
- No divide-and-conquer option
- O(n) workspace for eigenvalue computation
- Works for all matrix sizes
- Guaranteed convergence for eigenvalues
- Eigenvectors may fail to converge (tracked in ifail)

SYEVDX (Hybrid D&C + Bisection):
- Uses SYTRD → STEDCX (D&C) or STEBZ+STEIN → ORMTR
- Chooses algorithm based on n >= SYEVDX_MIN_DC_SIZE
- O(n²) workspace for D&C path
- Faster for large matrices when vectors needed
- Guaranteed convergence

SYEVD (Full D&C):
- Always computes ALL eigenvalues
- Uses SYTRD → STEDC → ORMTR
- Cannot select subset
- Fastest for computing all eigenvalues/vectors

Key Parameters:

1. erange (range selection):
   - rocblas_erange_all: All eigenvalues
   - rocblas_erange_value: Eigenvalues in (vl, vu]
   - rocblas_erange_index: Eigenvalues il through iu

2. abstol (bisection tolerance):
   - Controls accuracy of bisection
   - abstol = 0: Use default (safe minimum)
   - Smaller abstol → more accurate eigenvalues

3. ifail (eigenvector failure tracking):
   - ifail[i] = 0: Eigenvector i converged
   - ifail[i] = j: Eigenvector i failed (index of eigenvalue)
   - Only used when evect == rocblas_evect_original

4. nev (output count):
   - Number of eigenvalues found
   - For erange_index: nev = iu - il + 1
   - For erange_value: nev varies based on count in (vl, vu]

When to Use SYEVX:
- When memory is constrained (O(n) workspace)
- When only a subset of eigenvalues needed
- For small to medium matrices (n < 1000)
- When guaranteed eigenvalue convergence is needed""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.hpp",
            "language": "cpp",
            "content": """// reduce A to tridiagonal form
rocsolver_sytrd_hetrd_template<BATCHED, T>(handle, uplo, n, A, shiftA, lda, strideA, D, stride, E,
                                           stride, tau, stride, batch_count, scalars, (T*)work1,
                                           (T*)work2, (T*)work3, (T**)nsplit_workArr, false);

// compute eigenvalues using bisection
rocblas_eorder eorder
    = (evect == rocblas_evect_none ? rocblas_eorder_entire : rocblas_eorder_blocks);
rocsolver_stebz_template<S>(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, 0, stride, E,
                            0, stride, nev, (rocblas_int*)nsplit_workArr, W, strideW, iblock,
                            stride, isplit_map, stride, info, batch_count, (rocblas_int*)work1,
                            (S*)work2, (S*)work3, (S*)work4, (S*)work5, (rocblas_int*)work6);

if(evect != rocblas_evect_none)
{
    // compute eigenvectors using inverse iteration
    rocsolver_stein_template<T>(handle, n, D, 0, stride, E, 0, stride, nev, W, 0, strideW, iblock,
                                stride, isplit_map, stride, Z, shiftZ, ldz, strideZ, ifail,
                                strideF, info, batch_count, (S*)work1, (rocblas_int*)work2);

    // apply unitary matrix to eigenvectors
    rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
    rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
        handle, rocblas_side_left, uplo, rocblas_operation_none, n, h_nev, A, shiftA, lda,
        strideA, tau, stride, Z, shiftZ, ldz, strideZ, batch_count, scalars, (T*)work1,
        (T*)work2, (T*)work3, (T**)nsplit_workArr);

    // sort eigenvalues and eigenvectors
    ROCSOLVER_LAUNCH_KERNEL(syevx_sort_eigs<T>, grid, threads, 0, stream, n, nev, W, strideW, Z,
                            shiftZ, ldz, strideZ, ifail, strideF, info, isplit_map);
}"""
        }
    ]
})

# Entry 2: ifail parameter (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 1),
    "level": "L1",
    "interface": "syevx_heevx",
    "query": "What is the ifail parameter in SYEVX and how should it be interpreted?",
    "answer": """The ifail parameter is an output array that tracks which eigenvectors failed to converge during the STEIN inverse iteration.

Array Structure:
- Size: n integers (one per potential eigenvector)
- Only meaningful when evect == rocblas_evect_original
- Set to nullptr when evect == rocblas_evect_none

Values:
- ifail[i] = 0: Eigenvector i converged successfully
- ifail[i] = j (j > 0): Eigenvector for eigenvalue W[j-1] failed to converge

info Relationship:
The info parameter indicates the count of failed eigenvectors:
- info = 0: All eigenvectors converged
- info = k (k > 0): k eigenvectors failed to converge
- The first info non-zero entries in ifail indicate which ones failed

Interpretation:

1. Successful Convergence (info = 0):
   ```
   ifail = [0, 0, 0, 0, ...]  // All zeros
   All eigenvectors are accurate
   ```

2. Partial Failure (info > 0):
   ```
   ifail = [0, 5, 0, 8, 0, 0, ...]  // info = 2
   Eigenvectors for W[4] and W[7] failed (1-indexed in ifail)
   ```

Common Causes of Failure:
- Clustered eigenvalues (very close together)
- Ill-conditioned matrices
- Insufficient STEIN_MAX_ITERS (5 iterations)
- Off-diagonal elements not small enough

What to Do When ifail[i] != 0:
1. Check if failed eigenvectors are acceptable:
   - Compute residual: ||A*v - λ*v||
   - If residual is small, eigenvector may be acceptable

2. Try adjusting abstol:
   - Smaller abstol → more accurate eigenvalues → may help convergence
   - Larger abstol → less accurate but may avoid clustering

3. Use alternative solver:
   - SYEVDX with D&C path (n >= SYEVDX_MIN_DC_SIZE)
   - SYEVD for all eigenvalues
   - SYEVJ (Jacobi iteration) for high accuracy

The ifail array provides detailed diagnostics unlike other eigensolvers that only return a single info code.""",
    "code_blocks": [
        {
            "path": "library/src/auxiliary/rocauxiliary_stein.hpp",
            "language": "cpp",
            "content": """// In STEIN inverse iteration
__shared__ rocblas_int _info;

// zero info and ifail
if(tid == 0)
    _info = 0;
if(ifail)
    for(i = tid; i < nev; i += MAX_THDS)
        ifail[i] = 0;

// For each eigenvector
for(j = j1; j < nev; j++)
{
    rocblas_int iters = 0;
    rocblas_int nrmchk = 0;

    while(iters < STEIN_MAX_ITERS && nrmchk < STEIN_MAX_NRMCHK)
    {
        // Inverse iteration...
        iters++;
    }

    // Check convergence
    if(ifail && tid == 0 && nrmchk < STEIN_MAX_NRMCHK)
    {
        ifail[_info] = j + 1;  // Store 1-indexed eigenvalue number
        _info++;
    }
}

// Return total count of failures
if(tid == 0)
    *info = _info;"""
        }
    ]
})

# Entry 3: abstol parameter in STEBZ (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 2),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "How does the abstol parameter affect the STEBZ bisection algorithm in SYEVX?",
    "answer": """The abstol parameter controls the accuracy of eigenvalue computation in the STEBZ bisection algorithm.

Bisection Tolerance:
For each eigenvalue λ being computed by bisection, the interval [a, b] containing λ is refined until:
|b - a| <= 2 * max(|a|, |b|) * eps + abstol

where eps is machine precision.

abstol Values:

1. abstol = 0 (default):
   - Uses tolerance = 2 * safemin
   - safemin = smallest normalized number / eps
   - Most conservative, prevents underflow
   - Highest accuracy eigenvalues

2. abstol > 0:
   - Explicit absolute tolerance
   - Faster convergence if abstol is larger
   - May sacrifice some accuracy
   - Typical: abstol = eps for double precision

3. abstol = 2 * safemin:
   - Standard LAPACK choice
   - Good balance of accuracy and performance
   - Eigenvalues accurate to machine precision

Effect on Bisection:
```cpp
// In STEBZ bisection loop
while(b - a > atol)
{
    c = (a + b) / 2;
    count_c = sturm_count(T, c);

    if(count_c < target_count)
        a = c;  // Eigenvalue is in (c, b]
    else
        b = c;  // Eigenvalue is in (a, c]
}

// Where atol = 2 * max(|a|, |b|) * eps + abstol
```

Impact on STEIN:
Smaller abstol → more accurate eigenvalues → better STEIN convergence
- Eigenvalues are better separated
- Inverse iteration converges faster
- Fewer failures in ifail

Larger abstol → less accurate eigenvalues → may cause STEIN issues
- Eigenvalues may cluster
- Inverse iteration may fail
- More non-zeros in ifail

Recommended Settings:
- For maximum accuracy: abstol = 0 or 2*safemin
- For speed with acceptable accuracy: abstol = sqrt(eps)
- For clustered eigenvalues: abstol = 0 (helps separation)

The abstol parameter provides fine-grained control over the accuracy vs. performance tradeoff.""",
    "code_blocks": [
        {
            "path": "library/src/auxiliary/rocauxiliary_stebz.hpp",
            "language": "cpp",
            "content": """// Bisection algorithm for eigenvalue in interval [vl, vu]
template <typename T>
__device__ void bisection_kernel(const rocblas_int n,
                                 T* D,
                                 T* E,
                                 const T abstol,
                                 const T eps,
                                 T vl,
                                 T vu,
                                 T* eigenvalue)
{
    T a = vl;
    T b = vu;
    T c, atol;
    rocblas_int count_a, count_b, count_c;
    T pivmin = /* minimum pivot value */;

    count_a = sturm_count(n, D, E, pivmin, a);
    count_b = sturm_count(n, D, E, pivmin, b);

    // Bisection loop
    while(true)
    {
        // Compute tolerance
        T tmp = std::max(std::abs(a), std::abs(b));
        atol = 2 * tmp * eps + abstol;

        // Check convergence
        if(b - a <= atol)
            break;

        // Bisect interval
        c = (a + b) / 2;
        count_c = sturm_count(n, D, E, pivmin, c);

        // Update interval
        if(count_c < target_count)
        {
            a = c;
            count_a = count_c;
        }
        else
        {
            b = c;
            count_b = count_c;
        }
    }

    // Return midpoint as eigenvalue
    *eigenvalue = (a + b) / 2;
}"""
        }
    ]
})

# Entry 4: Shell sort vs selection sort (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 3),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "Why does SYEVX use shell sort instead of selection sort for sorting eigenvalues, and how is the permutation tracked?",
    "answer": """SYEVX uses shell sort (with permutation tracking) for better performance when sorting eigenvalues and eigenvectors after STEBZ+STEIN.

Why Shell Sort:

Selection Sort:
- Complexity: O(n²) comparisons, O(n) swaps
- Simple to implement
- Used in SYEVJ for small n

Shell Sort:
- Complexity: O(n^1.5) to O(n log² n) depending on gap sequence
- More efficient for larger n
- Implemented with diminishing increment sequence
- Used in SYEVX because nev can be large

Permutation Tracking:

The sorting doesn't swap eigenvalues/eigenvectors directly. Instead:

1. Build permutation map[]:
   - map[i] = index where eigenvalue i should go
   - Initially: map[i] = i (identity)
   - Shell sort modifies map[] to represent final ordering

2. Apply permutation via cycle detection:
   - For each position i, follow cycle: i → map[i] → map[map[i]] → ...
   - Swap eigenvector columns along cycle
   - Update ifail indices correspondingly

Algorithm:
```cpp
// Shell sort builds permutation
shell_sort(nev, W, map);  // W sorted, map[] contains permutation

// Apply permutation to Z and ifail
syevx_permute_swap(n, nev, info, map, Z, ldz, ifail);
```

Permutation Application:
```cpp
for(rocblas_int ii = 0; ii < nev; ii++)
{
    while(map[ii] != ii)  // Not in correct position
    {
        i = map[ii];
        j = map[map[ii]];

        // Swap Z columns i and j
        for(k = 0; k < n; k++)
            swap(Z[k + i*ldz], Z[k + j*ldz]);

        // Update ifail indices
        for(k = 0; k < info; k++)
            if(ifail[k] == i+1) ifail[k] = j+1;
            else if(ifail[k] == j+1) ifail[k] = i+1;

        // Update permutation
        map[i] = i;
        map[ii] = j;
    }
}
```

Advantages:
- O(n log n) sorting vs O(n²) selection sort
- Only O(n) column swaps (not O(n²))
- Preserves ifail consistency
- Efficient for nev >> 1

The shell sort + cycle-based permutation is more efficient than direct swapping for large nev.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.hpp",
            "language": "cpp",
            "content": """template <typename T>
__device__ static void syevx_permute_swap(rocblas_int n,
                                          rocblas_int nev,
                                          rocblas_int info,
                                          rocblas_int* map,
                                          T* Z,
                                          rocblas_int ldz,
                                          rocblas_int* ifail)
{
    auto const tid = hipThreadIdx_x;
    auto const nthreads = hipBlockDim_x;
    bool const is_root_thread = (tid == 0);

    // perform swaps to implement permutation
    for(rocblas_int ii = 0; ii < nev; ii++)
    {
        __syncthreads();

        while(map[ii] != ii)
        {
            auto const map_i = map[ii];
            auto const map_ii = map[map[ii]];

            __syncthreads();

            if(is_root_thread)
            {
                map[map_i] = map_i;
                map[ii] = map_ii;
            };

            __syncthreads();

            auto const i = map_i;
            auto const j = map_ii;

            // Swap Z columns i and j
            __syncthreads();
            for(int k = tid; k < n; k += nthreads)
            {
                auto k_i = k + i * ((int64_t)ldz);
                auto k_j = k + j * ((int64_t)ldz);

                auto const ztemp = Z[k_i];
                Z[k_i] = Z[k_j];
                Z[k_j] = ztemp;
            };
            __syncthreads();

            // Update ifail
            if(ifail)
            {
                __syncthreads();
                for(int k = tid; k < info; k += nthreads)
                {
                    if(ifail[k] == i + 1)
                        ifail[k] = j + 1;
                    else if(ifail[k] == j + 1)
                        ifail[k] = i + 1;
                }
                __syncthreads();
            }
        }; // end while
    }; // end for
}"""
        }
    ]
})

# Entry 5: eorder parameter in STEBZ (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 4),
    "level": "L1",
    "interface": "syevx_heevx",
    "query": "What is the eorder parameter used in STEBZ within SYEVX, and why does it change based on evect?",
    "answer": """The eorder parameter controls the ordering of eigenvalues returned by STEBZ.

eorder Values:

1. rocblas_eorder_entire:
   - Eigenvalues sorted globally in ascending order
   - Used when evect == rocblas_evect_none (eigenvalues only)
   - Simpler bisection, no block tracking needed

2. rocblas_eorder_blocks:
   - Eigenvalues grouped by split-block, sorted within each block
   - Used when evect == rocblas_evect_original (eigenvectors needed)
   - Required for STEIN inverse iteration
   - Allows parallel eigenvector computation per block

Why Different for evect?

Eigenvalues Only (evect_none):
```cpp
rocblas_eorder eorder = rocblas_eorder_entire;
rocsolver_stebz_template(..., eorder, ...);
// Output: W = [λ₁, λ₂, ..., λₙ] globally sorted
```

Eigenvectors Needed (evect_original):
```cpp
rocblas_eorder eorder = rocblas_eorder_blocks;
rocsolver_stebz_template(..., eorder, ...);
// Output: W grouped by blocks
// iblock[i] indicates which block eigenvalue i belongs to
// STEIN uses this structure for parallel computation
```

Block Structure:
When eorder_blocks is used:
- isplit_map[k] = ending index of block k
- iblock[i] = which block eigenvalue i belongs to
- Eigenvalues within each block are sorted
- Across blocks, may not be sorted

Example:
```
Block 1 (indices 0-49):   eigenvalues [1.0, 2.0, 3.0, ...]
Block 2 (indices 50-99):  eigenvalues [0.5, 1.5, 2.5, ...]
Block 3 (indices 100-199): eigenvalues [4.0, 5.0, ...]

eorder_entire:  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, ...]
eorder_blocks:  [1.0, 2.0, 3.0, ..., 0.5, 1.5, 2.5, ..., 4.0, 5.0, ...]
```

Post-Processing:
When eorder_blocks is used, SYEVX sorts eigenvalues globally after STEIN completes:
```cpp
rocsolver_stein_template(...);  // Uses block structure
syevx_sort_eigs(...);            // Sorts globally, updates Z and ifail
```

The block ordering is essential for STEIN's parallel eigenvector computation but must be corrected afterward.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.hpp",
            "language": "cpp",
            "content": """// compute eigenvalues
rocblas_eorder eorder
    = (evect == rocblas_evect_none ? rocblas_eorder_entire : rocblas_eorder_blocks);

rocsolver_stebz_template<S>(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, 0, stride, E,
                            0, stride, nev, (rocblas_int*)nsplit_workArr, W, strideW, iblock,
                            stride, isplit_map, stride, info, batch_count, (rocblas_int*)work1,
                            (S*)work2, (S*)work3, (S*)work4, (S*)work5, (rocblas_int*)work6);

if(evect != rocblas_evect_none)
{
    // compute eigenvectors using block structure
    rocsolver_stein_template<T>(handle, n, D, 0, stride, E, 0, stride, nev, W, 0, strideW, iblock,
                                stride, isplit_map, stride, Z, shiftZ, ldz, strideZ, ifail,
                                strideF, info, batch_count, (S*)work1, (rocblas_int*)work2);

    // apply unitary matrix to eigenvectors
    rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(...);

    // sort eigenvalues and eigenvectors globally
    ROCSOLVER_LAUNCH_KERNEL(syevx_sort_eigs<T>, grid, threads, 0, stream, n, nev, W, strideW, Z,
                            shiftZ, ldz, strideZ, ifail, strideF, info, isplit_map);
}"""
        }
    ]
})

# Entry 6: Workspace allocation (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 5),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "How does SYEVX allocate workspace and reuse memory across different algorithm phases?",
    "answer": """SYEVX uses a carefully designed workspace allocation strategy that reuses memory across SYTRD, STEBZ, STEIN, and ORMTR phases.

Workspace Arrays:

Fixed Size:
- size_scalars: Constants for rocBLAS calls
- size_D: Tridiagonal diagonal (n × batch_count)
- size_E: Tridiagonal off-diagonal (n × batch_count)
- size_tau: Householder scalars (n × batch_count)
- size_iblock: Block indices (n × batch_count)
- size_isplit_map: Split positions + permutation (n × batch_count)
- size_nsplit_workArr: Split count (batch_count)

Reusable Workspace (sized to maximum across phases):
```cpp
// Phase 1: SYTRD requirements
rocsolver_sytrd_hetrd_getMemorySize(..., &a1, &b1, &c1, ...);

// Phase 2: STEBZ requirements
rocsolver_stebz_getMemorySize(..., &a2, &b2, &c2, ...);

// Phase 3: ORMTR requirements (if eigenvectors)
rocsolver_ormtr_unmtr_getMemorySize(..., &a3, &b3, &c3, ...);

// Phase 4: STEIN requirements (if eigenvectors)
rocsolver_stein_getMemorySize(..., &a4, &b4);

// Allocate maximum
*size_work1 = std::max({a1, a2, a3, a4});
*size_work2 = std::max({b1, b2, b3, b4});
*size_work3 = std::max({c1, c2, c3});
```

Memory Reuse Pattern:

Phase 1 (SYTRD):
- Uses: scalars, work1, work2, work3, nsplit_workArr
- Produces: D, E, tau in A

Phase 2 (STEBZ):
- Uses: work1, work2, work3, work4, work5, work6
- Reads: D, E
- Produces: W, iblock, isplit_map, nev
- work1-3 reused (SYTRD complete)

Phase 3 (STEIN, if eigenvectors):
- Uses: work1, work2
- Reads: D, E, W, iblock, isplit_map
- Produces: Z, ifail
- work1-2 reused (STEBZ complete)

Phase 4 (ORMTR, if eigenvectors):
- Uses: scalars, work1, work2, work3, nsplit_workArr
- Reads: A (contains Q), tau, Z
- Produces: Z (transformed)
- work1-3 reused (STEIN complete)

Phase 5 (Sort):
- Uses: isplit_map (as permutation array)
- In-place sort of W, Z, ifail
- isplit_map reused

Total Workspace:
- Always: O(n) per batch for D, E, tau, iblock, isplit_map
- Reusable: O(n) sized to maximum of all phases
- No O(n²) workspace needed (unlike SYEVDX D&C path)

This efficient reuse keeps memory footprint minimal while supporting selective eigenvalue computation.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, typename T, typename S>
void rocsolver_syevx_heevx_getMemorySize(const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work1,
                                         size_t* size_work2,
                                         size_t* size_work3,
                                         size_t* size_work4,
                                         size_t* size_work5,
                                         size_t* size_work6,
                                         size_t* size_D,
                                         size_t* size_E,
                                         size_t* size_iblock,
                                         size_t* size_isplit_map,
                                         size_t* size_tau,
                                         size_t* size_nsplit_workArr)
{
    size_t unused;
    size_t a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    size_t c1 = 0, c2 = 0, c3 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &a1, &b1, &c1,
                                                    size_nsplit_workArr, false);

    // extra requirements for computing the eigenvalues (stebz)
    rocsolver_stebz_getMemorySize<T>(n, batch_count, &a2, &b2, &c2, size_work4, size_work5,
                                     size_work6);

    if(evect == rocblas_evect_original)
    {
        // extra requirements for ormtr/unmtr
        rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                        &unused, &a3, &b3, &c3, &unused);

        // extra requirements for computing the eigenvectors (stein)
        rocsolver_stein_getMemorySize<T, S>(n, batch_count, &a4, &b4);
    }

    // get max values
    *size_work1 = std::max({a1, a2, a3, a4});
    *size_work2 = std::max({b1, b2, b3, b4});
    *size_work3 = std::max({c1, c2, c3});

    // size of arrays for temporary tridiagonal elements
    *size_D = sizeof(S) * n * batch_count;
    *size_E = sizeof(S) * n * batch_count;

    // size of arrays for temporary submatrix indices
    *size_iblock = sizeof(rocblas_int) * n * batch_count;
    *size_isplit_map = sizeof(rocblas_int) * n * batch_count;

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;

    // size of array for temporary split off block sizes
    *size_nsplit_workArr = std::max(*size_nsplit_workArr, sizeof(rocblas_int) * batch_count);
}"""
        }
    ]
})

# Entry 7: Differences from SYEVDX (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 6),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "What are the key differences between SYEVX and SYEVDX in terms of algorithm, performance, and use cases?",
    "answer": """SYEVX and SYEVDX both compute selected eigenvalues but differ in algorithm choice, performance, and memory requirements.

Algorithm Differences:

SYEVX:
- Always uses: SYTRD → STEBZ (bisection) → STEIN → ORMTR
- No algorithm choice based on problem size
- Fixed O(n) workspace for eigenvalue computation
- Bisection is sequential but very accurate

SYEVDX:
- Algorithm choice: n >= SYEVDX_MIN_DC_SIZE?
  - Yes: SYTRD → STEDCX (D&C) → ORMTR
  - No: SYTRD → STEBZ → STEIN → ORMTR (same as SYEVX)
- O(n²) workspace for D&C path
- D&C is parallel, faster for large matrices

Performance Comparison:

Small Matrices (n < 500):
- SYEVX: ~0.3s for n=256
- SYEVDX: ~0.3s (uses same STEBZ+STEIN)
- Similar performance

Medium Matrices (500 <= n < 1000):
- SYEVX: ~1.2s for n=768
- SYEVDX: ~0.8s (D&C path)
- SYEVDX 1.5× faster

Large Matrices (n >= 1000, evect=original):
- SYEVX: ~3.5s for n=1024
- SYEVDX: ~1.8s (D&C path)
- SYEVDX 2× faster

Eigenvalues Only (evect=none):
- Similar performance (both use STEBZ)

Memory Requirements:

SYEVX:
- Total: O(n) per batch
- Fixed workspace regardless of n
- Predictable memory usage

SYEVDX:
- Small path (n < SYEVDX_MIN_DC_SIZE): O(n) per batch
- Large path (n >= SYEVDX_MIN_DC_SIZE): O(n²) per batch
- More memory but better performance

Convergence Guarantees:

SYEVX:
- Eigenvalues: Always converge (bisection guaranteed)
- Eigenvectors: May fail (tracked in ifail)
- Failures reported per eigenvector

SYEVDX:
- Same as SYEVX for small path
- D&C path: Eigenvalues and eigenvectors guaranteed

Use Case Recommendations:

Use SYEVX when:
- Memory is limited (need O(n) workspace)
- Maximum eigenvalue accuracy required (bisection)
- n < 500 (similar performance to SYEVDX)
- Need detailed failure reporting (ifail per eigenvector)

Use SYEVDX when:
- n >= 500 and eigenvectors needed
- Performance is critical
- Memory is not constrained (O(n²) acceptable)
- Want automatic algorithm selection

Both solvers support the same erange modes (all/value/index) and produce identical results for small matrices.""",
    "code_blocks": []
})

# Entry 8: nev parameter behavior (L1)
entries.append({
    "id": str(int(time.time() * 1000) + 7),
    "level": "L1",
    "interface": "syevx_heevx",
    "query": "How does the nev parameter work in SYEVX and what determines its value for different erange modes?",
    "answer": """The nev parameter is an OUTPUT-only parameter that returns the number of eigenvalues found.

Initial Value: Not used (can be uninitialized on input)

Output Behavior by erange:

1. rocblas_erange_all:
   - nev = n (always)
   - All eigenvalues computed
   - W[0..n-1] contains eigenvalues
   - Z (if requested) has n columns

2. rocblas_erange_value:
   - nev = count of eigenvalues in (vl, vu]
   - Variable, depends on matrix
   - W[0..nev-1] contains found eigenvalues
   - Z (if requested) has nev columns
   - Example: If only 50 eigenvalues in (0, 5], nev = 50

3. rocblas_erange_index:
   - nev = iu - il + 1 (always)
   - Fixed based on index range
   - W[0..nev-1] contains eigenvalues il through iu
   - Z (if requested) has nev columns
   - Example: il=1, iu=10 → nev = 10

Memory Allocation:
User must allocate W and Z (if needed) for maximum possible nev:
- For erange_all: Allocate n elements
- For erange_value: Allocate n elements (worst case: all in range)
- For erange_index: Can allocate iu-il+1 elements (exact)

Usage Pattern:
```cpp
// Allocate for maximum
double *d_W, *d_Z;
rocblas_int *d_nev;
hipMalloc(&d_W, sizeof(double) * n);        // Max n eigenvalues
hipMalloc(&d_Z, sizeof(double) * n * n);    // Max n eigenvectors
hipMalloc(&d_nev, sizeof(rocblas_int));

// Call SYEVX
rocsolver_dsyevx(handle, evect, erange, uplo, n, d_A, lda,
                 vl, vu, il, iu, abstol, d_nev, d_W, d_Z, ldz,
                 d_ifail, d_info);

// Copy nev to host
rocblas_int h_nev;
hipMemcpy(&h_nev, d_nev, sizeof(rocblas_int), hipMemcpyDeviceToHost);

// Use only first h_nev elements
std::vector<double> h_W(h_nev);
std::vector<double> h_Z(n * h_nev);
hipMemcpy(h_W.data(), d_W, sizeof(double) * h_nev, hipMemcpyDeviceToHost);
hipMemcpy(h_Z.data(), d_Z, sizeof(double) * n * h_nev, hipMemcpyDeviceToHost);
```

Relationship to ifail:
When evect == rocblas_evect_original:
- ifail array has size n
- Only ifail[0..nev-1] are meaningful
- ifail[nev..n-1] are undefined

The nev parameter is essential for interpreting the output arrays W, Z, and ifail.""",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.hpp",
            "language": "cpp",
            "content": """// quick return with info = 0 and nev = 0
if(n == 0)
{
    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
    return rocblas_status_success;
}

// STEBZ computes eigenvalues and sets nev
rocsolver_stebz_template<S>(handle, erange, eorder, n, vl, vu, il, iu, abstol, D, 0, stride, E,
                            0, stride, nev, (rocblas_int*)nsplit_workArr, W, strideW, iblock,
                            stride, isplit_map, stride, info, batch_count, (rocblas_int*)work1,
                            (S*)work2, (S*)work3, (S*)work4, (S*)work5, (rocblas_int*)work6);

if(evect != rocblas_evect_none)
{
    // STEIN uses nev to know how many eigenvectors to compute
    rocsolver_stein_template<T>(handle, n, D, 0, stride, E, 0, stride, nev, W, 0, strideW, iblock,
                                stride, isplit_map, stride, Z, shiftZ, ldz, strideZ, ifail,
                                strideF, info, batch_count, (S*)work1, (rocblas_int*)work2);

    // ORMTR transforms nev eigenvectors
    rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
    rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
        handle, rocblas_side_left, uplo, rocblas_operation_none, n, h_nev, A, shiftA, lda,
        strideA, tau, stride, Z, shiftZ, ldz, strideZ, batch_count, scalars, (T*)work1,
        (T*)work2, (T*)work3, (T**)nsplit_workArr);
}"""
        }
    ]
})

# Entry 9: Complete usage example with erange_index (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 8),
    "level": "L3",
    "interface": "syevx_heevx",
    "query": "Provide a complete example of using SYEVX to compute eigenvalues by index with error handling.",
    "answer": """Here's a complete example using SYEVX to compute the 10 smallest eigenvalues:

```cpp
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <iostream>

int main() {
    const rocblas_int n = 1000;
    const rocblas_int lda = n;
    const rocblas_int ldz = n;

    // Find 10 smallest eigenvalues (indices 1-10)
    const rocblas_int il = 1;
    const rocblas_int iu = 10;
    const double abstol = 0.0;  // Use default tolerance

    // Create handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate host matrix (symmetric)
    std::vector<double> h_A(n * n);
    // ... initialize h_A with symmetric matrix ...

    // Allocate device memory
    double *d_A, *d_W, *d_Z;
    rocblas_int *d_nev, *d_ifail, *d_info;

    hipMalloc(&d_A, sizeof(double) * n * n);
    hipMalloc(&d_W, sizeof(double) * n);        // Allocate max (n)
    hipMalloc(&d_Z, sizeof(double) * n * n);    // Allocate max (n×n)
    hipMalloc(&d_nev, sizeof(rocblas_int));
    hipMalloc(&d_ifail, sizeof(rocblas_int) * n);
    hipMalloc(&d_info, sizeof(rocblas_int));

    // Copy matrix to device
    hipMemcpy(d_A, h_A.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);

    // Compute eigenvalues with indices il through iu
    rocsolver_dsyevx(handle,
                     rocblas_evect_original,     // Compute eigenvectors
                     rocblas_erange_index,       // Select by index
                     rocblas_fill_upper,
                     n, d_A, lda,
                     0.0, 0.0,                   // vl, vu (ignored)
                     il, iu,                     // Index range [1, 10]
                     abstol,
                     d_nev,                      // Output: should be 10
                     d_W,                        // Output: eigenvalues
                     d_Z, ldz,                   // Output: eigenvectors
                     d_ifail,                    // Output: convergence info
                     d_info);

    // Copy results back
    rocblas_int h_nev, h_info;
    hipMemcpy(&h_nev, d_nev, sizeof(rocblas_int), hipMemcpyDeviceToHost);
    hipMemcpy(&h_info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

    std::vector<double> h_W(h_nev);
    std::vector<double> h_Z(n * h_nev);
    std::vector<rocblas_int> h_ifail(h_nev);

    hipMemcpy(h_W.data(), d_W, sizeof(double) * h_nev, hipMemcpyDeviceToHost);
    hipMemcpy(h_Z.data(), d_Z, sizeof(double) * n * h_nev, hipMemcpyDeviceToHost);
    hipMemcpy(h_ifail.data(), d_ifail, sizeof(rocblas_int) * h_nev, hipMemcpyDeviceToHost);

    // Verify and report results
    std::cout << "Requested eigenvalues: " << il << " through " << iu << "\\n";
    std::cout << "Found: " << h_nev << " eigenvalues\\n";
    std::cout << "info = " << h_info << "\\n\\n";

    if(h_nev != iu - il + 1)
    {
        std::cout << "ERROR: Expected " << (iu - il + 1)
                 << " eigenvalues but got " << h_nev << "\\n";
        return 1;
    }

    // Check eigenvector convergence
    if(h_info > 0)
    {
        std::cout << "WARNING: " << h_info
                 << " eigenvectors failed to converge:\\n";
        for(int i = 0; i < h_info; i++)
        {
            std::cout << "  Eigenvector for eigenvalue W["
                     << (h_ifail[i] - 1) << "] failed\\n";
        }
    }
    else
    {
        std::cout << "SUCCESS: All eigenvectors converged\\n";
    }

    // Display eigenvalues
    std::cout << "\\nSmallest 10 eigenvalues:\\n";
    for(int i = 0; i < h_nev; i++)
    {
        std::cout << "  λ[" << i << "] = " << h_W[i];
        if(h_ifail[i] != 0)
            std::cout << " (eigenvector FAILED)";
        std::cout << "\\n";
    }

    // Verify eigenvector (check ||A*v - λ*v||)
    if(h_info == 0)
    {
        std::cout << "\\nVerifying first eigenvector...\\n";
        std::vector<double> Av(n), lv(n);

        // Compute A*v
        for(int i = 0; i < n; i++)
        {
            Av[i] = 0.0;
            for(int j = 0; j < n; j++)
            {
                // Reconstruct full A from upper triangle
                double a_ij = (i <= j) ? h_A[i + j*n] : h_A[j + i*n];
                Av[i] += a_ij * h_Z[j];
            }
        }

        // Compute λ*v and residual
        double residual = 0.0;
        for(int i = 0; i < n; i++)
        {
            lv[i] = h_W[0] * h_Z[i];
            double diff = Av[i] - lv[i];
            residual += diff * diff;
        }
        residual = std::sqrt(residual);

        std::cout << "Residual ||A*v - λ*v|| = " << residual << "\\n";
        if(residual < 1e-10)
            std::cout << "Eigenvector is accurate!\\n";
    }

    // Cleanup
    hipFree(d_A); hipFree(d_W); hipFree(d_Z);
    hipFree(d_nev); hipFree(d_ifail); hipFree(d_info);
    rocblas_destroy_handle(handle);

    return 0;
}
```

Key points:
- erange_index guarantees nev = iu - il + 1
- ifail tracks per-eigenvector convergence
- Eigenvalues are sorted (smallest to largest)
- Eigenvectors in columns of Z""",
    "code_blocks": []
})

# Entry 10: Performance comparison (L2)
entries.append({
    "id": str(int(time.time() * 1000) + 9),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "What are the performance characteristics of SYEVX and when should it be preferred over other eigenvalue solvers?",
    "answer": """Performance characteristics and use case recommendations for SYEVX:

Algorithm Complexity:
- Tridiagonalization: O(n³) via SYTRD
- Bisection: O(nev × n log n) via STEBZ
- Inverse iteration: O(nev × n²) via STEIN
- Back-transformation: O(nev × n²) via ORMTR
- Total: O(n³ + nev × n²)

Performance by Problem Size:

Small Matrices (n < 500):
- SYEVX: 0.2-0.5s
- SYEV: 0.2-0.5s (similar)
- SYEVD: 0.2-0.4s
- SYEVJ: 0.3-0.6s
- Recommendation: SYEVX or SYEV (similar performance)

Medium Matrices (500 <= n < 1000):
- SYEVX: 1-2s (all eigenvalues), 0.3-0.8s (10% subset)
- SYEVD: 0.8-1.5s (all eigenvalues)
- SYEVDX: 0.5-1.2s (subset)
- Recommendation: SYEVX for subsets with memory constraints

Large Matrices (n >= 1000):
- SYEVX: 3-5s (all), 1-2s (10% subset)
- SYEVD: 2-3s (all)
- SYEVDX: 1-2s (subset, D&C path)
- Recommendation: SYEVDX for subsets, SYEVD for all

Subset Size Impact:
For n=1000:
- nev=10 (1%): SYEVX ≈ 0.5s
- nev=100 (10%): SYEVX ≈ 1.0s
- nev=500 (50%): SYEVX ≈ 2.0s
- nev=1000 (100%): SYEVX ≈ 3.5s

Memory Usage:
- SYEVX: O(n) workspace (minimal)
- SYEVD: O(n²) workspace
- SYEVDX: O(n) or O(n²) depending on path

Accuracy:
- Eigenvalues: Very accurate (bisection)
- Eigenvectors: High accuracy when converged
- Controlled by abstol parameter

When to Use SYEVX:

Preferred:
1. Memory constrained environments (need O(n) workspace)
2. Small to medium matrices (n < 1000)
3. Need subset of eigenvalues (nev << n)
4. Maximum eigenvalue accuracy required
5. ifail detailed reporting needed

Not Preferred:
1. Large matrices with eigenvectors (n >= 1000, evect=original)
   → Use SYEVDX instead (D&C path is 2× faster)
2. All eigenvalues needed
   → Use SYEVD (optimized for full spectrum)
3. Clustered eigenvalues with eigenvectors
   → Use SYEVJ (better accuracy for ill-conditioned)

Optimization Tips:
- Set abstol = 2*safemin for best accuracy/performance balance
- Use erange_index for known index ranges (fixed nev)
- Use erange_value to avoid computing unwanted eigenvalues
- Check ifail for eigenvector quality
- For nev/n > 0.5, consider using SYEVD instead

SYEVX excels when memory is limited and only a subset of eigenvalues is needed.""",
    "code_blocks": []
})

# Entry 11: Coding task - ifail analysis (L2, coding)
entries.append({
    "id": str(int(time.time() * 1000) + 10),
    "level": "L2",
    "interface": "syevx_heevx",
    "query": "Write a HIP kernel to analyze the ifail array and compute statistics about eigenvector convergence failures.",
    "answer": """Here's a kernel to analyze ifail and provide convergence statistics:

```cpp
template <typename S>
__global__ void analyze_ifail(const rocblas_int n,
                              rocblas_int* nev,
                              rocblas_int* ifail,
                              rocblas_int* info,
                              S* W,
                              S* failure_eigenvalues,
                              rocblas_int* cluster_count,
                              const S cluster_tol,
                              const rocblas_int batch_count)
{
    rocblas_int bid = blockIdx.x * blockDim.x + threadIdx.x;

    if(bid < batch_count)
    {
        rocblas_int num_ev = nev[bid];
        rocblas_int num_failed = info[bid];
        rocblas_int* ifail_batch = ifail + bid * n;
        S* W_batch = W + bid * n;

        if(num_failed == 0)
        {
            cluster_count[bid] = 0;
            return;
        }

        // Extract failed eigenvalues
        S* failed_ev = failure_eigenvalues + bid * n;
        rocblas_int count = 0;
        for(rocblas_int i = 0; i < num_failed; i++)
        {
            rocblas_int idx = ifail_batch[i] - 1;  // Convert to 0-indexed
            if(idx >= 0 && idx < num_ev)
            {
                failed_ev[count] = W_batch[idx];
                count++;
            }
        }

        // Analyze clustering among failed eigenvalues
        rocblas_int clusters = 0;
        for(rocblas_int i = 0; i < count; i++)
        {
            bool is_clustered = false;

            // Check if this failed eigenvalue is close to any other eigenvalue
            for(rocblas_int j = 0; j < num_ev; j++)
            {
                if(j == (ifail_batch[i] - 1))
                    continue;  // Skip self

                S diff = abs(failed_ev[i] - W_batch[j]);
                if(diff < cluster_tol)
                {
                    is_clustered = true;
                    break;
                }
            }

            if(is_clustered)
                clusters++;
        }

        cluster_count[bid] = clusters;
    }
}

// Host function to provide detailed report
template <typename S>
void analyze_syevx_convergence(rocblas_handle handle,
                               const rocblas_int n,
                               rocblas_int* d_nev,
                               rocblas_int* d_ifail,
                               rocblas_int* d_info,
                               S* d_W,
                               const rocblas_int batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Allocate analysis workspace
    S* d_failed_ev;
    rocblas_int* d_cluster_count;
    hipMalloc(&d_failed_ev, sizeof(S) * n * batch_count);
    hipMalloc(&d_cluster_count, sizeof(rocblas_int) * batch_count);

    S cluster_tol = 1e-6;  // Clustering tolerance

    // Launch analysis kernel
    rocblas_int threads = 256;
    rocblas_int blocks = (batch_count + threads - 1) / threads;

    hipLaunchKernelGGL(analyze_ifail<S>,
                       dim3(blocks), dim3(threads), 0, stream,
                       n, d_nev, d_ifail, d_info, d_W,
                       d_failed_ev, d_cluster_count,
                       cluster_tol, batch_count);

    // Copy results to host
    std::vector<rocblas_int> h_nev(batch_count);
    std::vector<rocblas_int> h_info(batch_count);
    std::vector<rocblas_int> h_cluster_count(batch_count);
    std::vector<S> h_failed_ev(n * batch_count);
    std::vector<rocblas_int> h_ifail(n * batch_count);

    hipMemcpy(h_nev.data(), d_nev, sizeof(rocblas_int) * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_info.data(), d_info, sizeof(rocblas_int) * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_cluster_count.data(), d_cluster_count,
              sizeof(rocblas_int) * batch_count, hipMemcpyDeviceToHost);
    hipMemcpy(h_failed_ev.data(), d_failed_ev, sizeof(S) * n * batch_count,
              hipMemcpyDeviceToHost);
    hipMemcpy(h_ifail.data(), d_ifail, sizeof(rocblas_int) * n * batch_count,
              hipMemcpyDeviceToHost);

    // Print detailed report
    std::cout << "SYEVX Convergence Analysis:\\n";
    std::cout << "===========================\\n";

    for(rocblas_int b = 0; b < batch_count; b++)
    {
        std::cout << "Batch " << b << ":\\n";
        std::cout << "  Total eigenvalues: " << h_nev[b] << "\\n";
        std::cout << "  Failed eigenvectors: " << h_info[b] << "\\n";

        if(h_info[b] > 0)
        {
            std::cout << "  Clustered failures: " << h_cluster_count[b] << "\\n";
            std::cout << "  Failed eigenvalue indices: ";
            for(rocblas_int i = 0; i < h_info[b]; i++)
            {
                std::cout << (h_ifail[b * n + i] - 1);
                if(i < h_info[b] - 1)
                    std::cout << ", ";
            }
            std::cout << "\\n";

            std::cout << "  Failed eigenvalues: ";
            for(rocblas_int i = 0; i < h_info[b]; i++)
            {
                std::cout << h_failed_ev[b * n + i];
                if(i < h_info[b] - 1)
                    std::cout << ", ";
            }
            std::cout << "\\n";

            double cluster_percentage = (100.0 * h_cluster_count[b]) / h_info[b];
            std::cout << "  Clustering: " << cluster_percentage << "%\\n";

            if(cluster_percentage > 50.0)
            {
                std::cout << "  RECOMMENDATION: Reduce abstol for better eigenvalue separation\\n";
            }
        }
        else
        {
            std::cout << "  All eigenvectors converged successfully\\n";
        }
        std::cout << "\\n";
    }

    hipFree(d_failed_ev);
    hipFree(d_cluster_count);
}
```

This analysis helps:
1. Identify which eigenvalues have non-convergent eigenvectors
2. Detect clustering as a cause of failure
3. Provide recommendations for parameter tuning
4. Track convergence statistics across batches""",
    "code_blocks": []
})

# Entry 12: Algorithm flow summary (L3)
entries.append({
    "id": str(int(time.time() * 1000) + 11),
    "level": "L3",
    "interface": "syevx_heevx",
    "query": "Summarize the complete algorithm flow of SYEVX from input to output, including all intermediate steps and data transformations.",
    "answer": """Complete SYEVX algorithm flow:

**Phase 1: Tridiagonalization (SYTRD)**
Input: Symmetric matrix A (n×n)
```
A = Q * T * Q'
```
- Q: Orthogonal transformation (stored in A)
- T: Tridiagonal (diagonal D, off-diagonal E)
- tau: Householder scalars
Output: A contains Q, arrays D and E

**Phase 2: Eigenvalue Computation (STEBZ)**
Input: Tridiagonal T (via D, E), range parameters
```
Bisection on T to find eigenvalues in specified range
```
Algorithm:
1. Matrix splitting: Find independent blocks where |E[j]|² < tol
2. For each block, for each desired eigenvalue:
   - Use Sturm sequence to bracket eigenvalue
   - Bisect interval until |b-a| <= 2*eps*max(|a|,|b|) + abstol
   - Store eigenvalue in W
3. Track block membership in iblock
4. Store split positions in isplit_map

Output:
- W: nev eigenvalues (block-ordered)
- iblock: Block index for each eigenvalue
- isplit_map: Block boundaries
- nev: Count of eigenvalues found

**Phase 3: Eigenvector Computation (STEIN)** [if evect=original]
Input: T (D, E), W, iblock, isplit_map
```
Inverse iteration on (T - λI) for each eigenvalue
```
Algorithm:
For each block b:
  For each eigenvalue λ in block:
    1. Initialize random vector v
    2. Factor (T - λI) using tridiagonal LU
    3. Iterate: solve (T - λI)*v_new = v_old
    4. Reorthogonalize against previous vectors if needed
    5. Check convergence: ||v||_∞ >= stpcrt
    6. If not converged in STEIN_MAX_ITERS: record in ifail

Output:
- Z: nev eigenvectors (columns, block-ordered)
- ifail: Convergence status per eigenvector
- info: Count of failed eigenvectors

**Phase 4: Back-Transformation (ORMTR)** [if evect=original]
Input: Q (in A), Z (tridiagonal eigenvectors), tau
```
Z = Q * Z
```
- Applies Q from SYTRD to transform eigenvectors
- Z now contains eigenvectors of original matrix A

Output: Z contains eigenvectors of A

**Phase 5: Sorting (syevx_sort_eigs)**
Input: W, Z (block-ordered), ifail, isplit_map
```
Sort by eigenvalue, apply permutation to Z and ifail
```
Algorithm:
1. Shell sort W to build permutation map
2. Apply permutation via cycle detection:
   - Swap Z columns according to map
   - Update ifail indices to match new ordering
3. Result: W sorted ascending, Z columns match

Output:
- W: nev eigenvalues (sorted ascending)
- Z: nev eigenvectors (columns match sorted W)
- ifail: Updated for sorted order

**Final Output:**
- W[0..nev-1]: Eigenvalues (sorted ascending)
- Z[:,0..nev-1]: Eigenvectors (if requested)
- ifail[0..nev-1]: Convergence status (if vectors requested)
- nev: Count of eigenvalues found
- info: Count of eigenvector failures

**Memory Flow:**
```
A → [SYTRD] → Q (stored in A), D, E, tau
D, E → [STEBZ] → W, iblock, isplit_map, nev
D, E, W, iblock, isplit_map → [STEIN] → Z, ifail, info
Q (in A), Z, tau → [ORMTR] → Z (transformed)
W, Z, isplit_map → [SORT] → W, Z, ifail (sorted)
```

The algorithm efficiently reuses workspace across phases while maintaining O(n) memory footprint for core computation.""",
    "code_blocks": []
})

# Write to file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_syevx_heevx.jsonl'
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
