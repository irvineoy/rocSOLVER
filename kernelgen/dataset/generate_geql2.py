#!/usr/bin/env python3
"""
Generator for GEQL2 (unblocked QL factorization) dataset following DATASET.md schema.
"""
import json
import time
from pathlib import Path

def generate_entries():
    entries = []
    base_timestamp = int(time.time() * 1000)
    
    # L1-1: LARFG for Bottom-Up Householder Generation (CODING)
    entries.append({
        "id": str(base_timestamp + 1),
        "level": "L1",
        "interface": "geql2",
        "instruction": "Implement the LARFG call for GEQL2 that generates a Householder reflector for column (n-j-1) starting from row (m-j-1) down to row 0. Explain why QL factorization processes columns right-to-left and rows bottom-to-top.",
        "context_text": "GEQL2 computes the QL factorization A = Q*L where Q is orthogonal and L is lower triangular. Unlike QR which processes left-to-right, QL processes right-to-left, building reflectors from the bottom-up of each column.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// Main loop: j = 0 to min(m,n)-1
for(rocblas_int j = 0; j < dim; j++)
{
    // generate Householder reflector to work on column n - j - 1
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);
    // ...
}"""
            }
        ],
        "answer": """The LARFG call generates a Householder reflector for column (n-j-1):

```cpp
rocsolver_larfg_template(
    handle, 
    m - j,                                      // length: rows 0 to (m-j-1)
    A, shiftA + idx2D(m - j - 1, n - j - 1, lda),  // alpha = A[m-j-1, n-j-1]
    A, shiftA + idx2D(0, n - j - 1, lda),          // x = A[0:m-j-2, n-j-1]
    1,                                             // incx = 1
    strideA, 
    (ipiv + dim - j - 1),                          // tau stored at ipiv[dim-j-1]
    strideP, 
    batch_count, 
    (T*)work_workArr, 
    Abyx_norms
);
```

Key points:
1. **Right-to-left**: Process column (n-j-1), starting from rightmost (j=0, column n-1)
2. **Bottom-to-top**: Element (m-j-1) is the pivot, annihilates elements 0 to (m-j-2) above it
3. **QL vs QR**: QR processes top-to-bottom, left-to-right. QL reverses both directions.
4. **Tau storage**: tau stored at (dim-j-1) so final tau is at index 0""",
        "rationale": "QL factorization builds L (lower triangular) from bottom-right corner. Each iteration j processes column (n-j-1) and uses element (m-j-1) as the pivot to annihilate elements above it, forming the lower triangular structure.",
        "tags": ["coding", "householder", "ql-factorization", "larfg"]
    })

    # L1-2: Diagonal Manipulation for LARF (CODING)
    entries.append({
        "id": str(base_timestamp + 2),
        "level": "L1",
        "interface": "geql2",
        "instruction": "Explain and implement the diagonal save/restore pattern used before and after LARF application in GEQL2. Why must A[m-j-1, n-j-1] be temporarily set to 1?",
        "context_text": "Before applying the Householder reflector with LARF, GEQL2 saves the diagonal element, sets it to 1, applies the reflector, then restores the original value.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// insert one in A(m-j-1,n-j-1) to build/apply the householder matrix
ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                        0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                        lda, strideA, 1, true);

// Apply Householder reflector to the rest of matrix from the left
rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                        shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                        strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                        (T**)work_workArr);

// restore original value of A(m-j-1,n-j-1)
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                        dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                        shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);"""
            }
        ],
        "answer": """The diagonal manipulation ensures correct Householder reflector application:

```cpp
// Step 1: Save diagonal to temporary array
// set_diag with last param=true: diag[0] = A[m-j-1, n-j-1], then A[m-j-1, n-j-1] = 1
ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), 
    dim3(batch_count, 1, 1), dim3(1, 1, 1),
    0, stream, 
    diag, 0, 1,                              // save to diag buffer
    A, shiftA + idx2D(m - j - 1, n - j - 1, lda),  // matrix element
    lda, strideA, 1, 
    true);                                   // save=true

// Step 2: Apply LARF with implicit v[m-j-1] = 1
rocsolver_larf_template(...);

// Step 3: Restore original diagonal value
// restore_diag: A[m-j-1, n-j-1] = diag[0]
ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), 
    dim3(batch_count, 1, 1), dim3(1, 1, 1),
    0, stream, 
    diag, 0, 1,
    A, shiftA + idx2D(m - j - 1, n - j - 1, lda), 
    lda, strideA, 1);
```

**Why set to 1?**
- Householder vector v has implicit first element v[0] = 1 after LARFG
- LARF expects v stored in matrix with v[0] = 1
- Original A[m-j-1, n-j-1] contains β (beta from LARFG), not 1
- Setting to 1 allows LARF to use the column directly as v
- Must restore afterward to preserve L factor""",
        "rationale": "LARFG computes v with v[0]=1 implicitly and stores beta separately. LARF needs v with v[0]=1 explicitly in the matrix column. The save/restore preserves the L factor while allowing in-place reflector application.",
        "tags": ["coding", "larf", "diagonal-manipulation", "householder-storage"]
    })

    # L1-3: Complex Tau Conjugation (CODING)
    entries.append({
        "id": str(base_timestamp + 3),
        "level": "L1",
        "interface": "geql2",
        "instruction": "Implement the tau conjugation logic for complex matrices in GEQL2. Why must tau be conjugated before LARF and restored after?",
        "context_text": "For complex Hermitian matrices, the Householder reflector H = I - tau*v*v' requires proper conjugation of tau when applying from the left.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// conjugate tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

// Apply Householder reflector to the rest of matrix from the left
rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                        shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                        strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                        (T**)work_workArr);

// restore tau
if(COMPLEX)
    rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);"""
            }
        ],
        "answer": """For complex matrices, tau conjugation ensures correct Hermitian reflector application:

```cpp
template <bool COMPLEX = rocblas_is_complex<T>>
void apply_reflector_ql(...)
{
    // Conjugate tau before LARF
    if(COMPLEX)
        rocsolver_lacgv_template<T>(
            handle, 
            1,                      // conjugate 1 element
            ipiv,                   // tau array
            dim - j - 1,            // offset to tau[dim-j-1]
            1,                      // increment
            strideP, 
            batch_count
        );

    // Apply reflector: H = I - conj(tau) * v * v'
    rocsolver_larf_template(
        handle, 
        rocblas_side_left,          // apply from left
        m - j,                      // reflector size
        n - j - 1,                  // columns to update
        A, shiftA + idx2D(0, n - j - 1, lda),  // v vector
        1, strideA,
        (ipiv + dim - j - 1),       // tau (now conjugated)
        strideP,
        A, shiftA,                  // matrix to update
        lda, strideA,
        batch_count, scalars, Abyx_norms, (T**)work_workArr
    );

    // Restore original tau
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);
}
```

**Why conjugate?**
1. Hermitian reflector: H = I - tau*v*v* (v* = conjugate transpose)
2. LARF computes: C := C - tau * v * (v' * C)
3. For left-side: need conj(tau) to match Hermitian form
4. Restore tau for storage (GEQLF needs original tau values)""",
        "rationale": "Complex Householder reflectors for Hermitian matrices require H = I - tau*v*v*. LARF uses tau*v*(v'*C), so tau must be conjugated for left-side application to produce the correct Hermitian update.",
        "tags": ["coding", "complex-arithmetic", "hermitian", "conjugation", "lacgv"]
    })

    # L2-1: LARFG-LARF Pipeline (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 4),
        "level": "L2",
        "interface": "geql2",
        "instruction": "Trace the complete LARFG-LARF pipeline for one iteration of GEQL2 on a 5×3 matrix at j=0. Show the matrix state before and after, including v storage and tau computation.",
        "context_text": "Each GEQL2 iteration generates a Householder reflector (LARFG) and applies it to the left portion of the matrix (LARF).",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """for(rocblas_int j = 0; j < dim; j++)
{
    // generate Householder reflector to work on column n - j - 1
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(m-j-1,n-j-1) to build/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                            lda, strideA, 1, true);

    // Apply Householder reflector to the rest of matrix from the left
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                            shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                            strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                            (T**)work_workArr);

    // restore original value of A(m-j-1,n-j-1)
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                            shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);
}"""
            }
        ],
        "answer": """Iteration j=0 on 5×3 matrix (dim=3, process column 2):

**Initial A:**
```
[ a00  a01  a02 ]
[ a10  a11  a12 ]
[ a20  a21  a22 ]
[ a30  a31  a32 ]
[ a40  a41  a42 ]
```

**Step 1: LARFG on column 2, m-j=5 elements**
- Alpha: A[4,2] (bottom element)
- x: A[0:3,2] (elements above)
- Computes: norm = sqrt(|a02|² + |a12|² + |a22|² + |a32|² + |a42|²)
- tau[2] = 2/(1 + |norm/a42|)
- beta[2] = ±norm (sign to avoid cancellation)
- Scaling: A[0:3,2] *= 1/(a42 - beta)

**After LARFG:**
```
[ v02  a01  a02 ]     tau[2] computed
[ v12  a11  a12 ]     ipiv[2] = tau
[ v22  a21  a22 ]     A[4,2] = beta
[ v32  a31  a32 ]     
[ v42  a41  beta]     
```

**Step 2: set_diag saves beta, sets A[4,2]=1**
- diag[0] = beta
- A[4,2] = 1

**Step 3: LARF applies H to columns 0:1**
- w = (v' * A[:,0:1]) with v = [v02, v12, v22, v32, 1]'
- A[:,0:1] -= tau * v * w

**After LARF:**
```
[ a00' a01' v02 ]
[ a10' a11' v12 ]
[ a20' a21' v22 ]
[ a30' a31' v32 ]
[ a40' a41'  1  ]
```

**Step 4: restore_diag**
- A[4,2] = beta (restore L factor diagonal)

**Final after j=0:**
```
[ a00' a01' v02 ]
[ a10' a11' v12 ]
[ a20' a21' v22 ]
[ a30' a31' v32 ]
[ a40' a41' l22 ]  where l22=beta, v stored above
```

Next iterations j=1,2 process columns 1,0 similarly.""",
        "rationale": "The LARFG-LARF pipeline is the core of QL factorization: LARFG builds the reflector from bottom-up, LARF applies it to remaining columns. The diagonal manipulation allows in-place storage of both L factor and Q reflectors.",
        "tags": ["analysis", "larfg", "larf", "pipeline", "matrix-update", "ql-factorization"]
    })

    # L2-2: Memory Access Patterns (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 5),
        "level": "L2",
        "interface": "geql2",
        "instruction": "Analyze the memory access pattern for GEQL2 vs GEQR2. Why is QL factorization potentially less cache-friendly than QR on column-major matrices?",
        "context_text": "GEQL2 processes columns right-to-left and rows bottom-to-top, while GEQR2 processes left-to-right and top-to-bottom.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// GEQL2: j=0 to dim-1, processing column (n-j-1)
for(rocblas_int j = 0; j < dim; j++)
{
    rocsolver_larfg_template(handle, m - j, 
                             A, shiftA + idx2D(m - j - 1, n - j - 1, lda),  // bottom of column (n-j-1)
                             A, shiftA + idx2D(0, n - j - 1, lda),          // top of column (n-j-1)
                             1, strideA, ...);
    
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1,
                            A, shiftA + idx2D(0, n - j - 1, lda), ...      // update columns 0 to (n-j-2)
                            A, shiftA, ...);                                // starting from column 0
}"""
            },
            {
                "path": "library/src/lapack/roclapack_geqr2.hpp",
                "language": "cpp",
                "content": """// GEQR2: j=0 to dim-1, processing column j
for(rocblas_int j = 0; j < dim; ++j)
{
    rocsolver_larfg_template(handle, m - j, 
                             A, shiftA + idx2D(j, j, lda),                  // diagonal of column j
                             A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda),  // below diagonal
                             (I)1, strideA, ...);

    if(j < n - 1)
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), ...             // update columns (j+1) to (n-1)
                                A, shiftA + idx2D(j, j + 1, lda), ...);
}"""
            }
        ],
        "answer": """Memory access comparison (column-major storage):

**GEQR2 (QR factorization):**
- Iteration 0: Access column 0 (contiguous), update columns 1..n-1
- Iteration 1: Access column 1 (contiguous), update columns 2..n-1
- Access pattern: left-to-right, each column accessed once sequentially
- Cache-friendly: columns accessed in storage order

**GEQL2 (QL factorization):**
- Iteration 0: Access column (n-1) (rightmost), update columns 0..(n-2)
- Iteration 1: Access column (n-2), update columns 0..(n-3)
- Access pattern: right-to-left, columns accessed in reverse order
- Less cache-friendly: jumping backward through memory

**Cache analysis for m=1024, n=512:**

QR: 
- j=0: touch column 0, stream through columns 1-511 → cache lines loaded sequentially
- j=1: touch column 1 (likely in cache from j=0), stream through 2-511
- Temporal locality: column (j+1) often in cache when needed for iteration j

QL:
- j=0: touch column 511, stream through columns 0-510 → load entire width
- j=1: touch column 510, stream through 0-509 → reload columns 0-509
- Poor temporal locality: columns 0..(n-j-2) reloaded each iteration

**Why QL is slower:**
1. Right-to-left column access fights prefetcher (expects forward stride)
2. Each iteration reloads left columns (0 to n-j-2)
3. Cache thrashing: left columns evicted and reloaded
4. QR benefits from forward streaming and column reuse

**Mitigation:**
- Blocked algorithm (GEQLF) with panel factorization
- Larger working set fits in L2/L3 cache
- For small n, difference is minimal""",
        "rationale": "Column-major storage and left-to-right caches favor QR's sequential access. QL's reverse access pattern and repeated left-column updates cause cache misses. This motivates blocked GEQLF for large matrices.",
        "tags": ["analysis", "memory-access", "cache", "performance", "qr-vs-ql"]
    })

    # L3-1: Complete Algorithm (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 6),
        "level": "L3",
        "interface": "geql2",
        "instruction": "Explain the complete GEQL2 algorithm including workspace management, diagonal storage, and the relationship between Q and L factors. How are they stored in-place?",
        "context_text": "GEQL2 computes A = Q*L where Q is m×m orthogonal and L is m×n lower triangular. Both factors are stored in-place in the original matrix A.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_geql2_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms,
                                        T* diag)
{
    ROCSOLVER_ENTER("geql2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int dim = std::min(m, n);

    for(rocblas_int j = 0; j < dim; j++)
    {
        // generate Householder reflector to work on column n - j - 1
        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                                 shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                                 strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // insert one in A(m-j-1,n-j-1) to build/apply the householder matrix
        ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                                0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                                lda, strideA, 1, true);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);

        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(0, n - j - 1, lda), 1, strideA, (ipiv + dim - j - 1),
                                strideP, A, shiftA, lda, strideA, batch_count, scalars, Abyx_norms,
                                (T**)work_workArr);

        ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                                dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                                shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);

        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, dim - j - 1, 1, strideP, batch_count);
    }

    return rocblas_status_success;
}"""
            }
        ],
        "answer": """Complete GEQL2 algorithm for A = Q*L factorization:

**Input:** m×n matrix A (m >= n typically)
**Output:** 
- L factor: lower triangle of A
- Q representation: Householder vectors in strict upper triangle + tau in ipiv

**Algorithm (dim = min(m,n)):**

```
for j = 0 to dim-1:
    col = n - j - 1                    // Process columns right-to-left
    pivot_row = m - j - 1              // Pivot at bottom
    
    1. LARFG(A[0:pivot_row, col]):
       - Annihilate A[0:pivot_row-1, col] using A[pivot_row, col] as pivot
       - Compute tau[dim-j-1], beta[dim-j-1]
       - Store v[0:pivot_row-1] in A[0:pivot_row-1, col]
       - Store beta in A[pivot_row, col]
    
    2. Save diagonal: diag[0] = A[pivot_row, col]
    
    3. Set A[pivot_row, col] = 1 (for LARF)
    
    4. If complex: conjugate tau[dim-j-1]
    
    5. LARF(A[0:m-j-1, 0:col-1]):
       - H = I - tau * v * v'
       - A[:,0:col-1] = H * A[:,0:col-1]
    
    6. Restore diagonal: A[pivot_row, col] = diag[0]
    
    7. If complex: restore tau[dim-j-1]
```

**Storage layout after completion (5×3 example):**
```
A = [ v02  v01  v00 ]     L factor: lower triangle
    [ v12  v11  v10 ]     Q vectors: strict upper triangle + unit diagonal implied
    [ v22  v21  l22 ]     ipiv = [tau0, tau1, tau2]
    [ v32  l32  l33 ]
    [ l42  l43  l44 ]

L = [  0    0   l22 ]     Q = (I - tau2*v2*v2') * (I - tau1*v1*v1') * (I - tau0*v0*v0')
    [  0   l32  l33 ]     where v0 = [v00, v10, v20, v30, 1]'
    [ l42  l43  l44 ]           v1 = [v01, v11, v21, 1, 0]'
    [l52  l53  l54 ]           v2 = [v02, v12, 1, 0, 0]'
    [l62  l63  l64 ]
```

**Workspace:**
1. scalars: constants for rocBLAS (-1, 0, 1)
2. work_workArr: max(LARF workspace, LARFG workspace)
3. Abyx_norms: max(LARF norms, LARFG norms)
4. diag: temporary diagonal storage (1 element)

**Key properties:**
- Unblocked: O(mn²) complexity
- In-place: only O(1) extra storage
- Numerically stable: uses Householder reflectors
- Foundation for blocked GEQLF""",
        "rationale": "GEQL2 builds QL factorization by applying Householder reflectors from right-to-left, bottom-to-top. In-place storage is achieved by storing L in lower triangle and Q's Householder vectors in strict upper triangle with tau in separate array.",
        "tags": ["analysis", "algorithm", "ql-factorization", "in-place", "workspace"]
    })

    # L3-2: Workspace Calculation (CODING)
    entries.append({
        "id": str(base_timestamp + 7),
        "level": "L3",
        "interface": "geql2",
        "instruction": "Implement the workspace calculation for GEQL2 explaining why it takes the maximum of LARF and LARFG requirements.",
        "context_text": "GEQL2 reuses workspace between LARF and LARFG calls since they never execute concurrently.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """template <bool BATCHED, typename T>
void rocsolver_geql2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_left, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(m, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}"""
            }
        ],
        "answer": """Complete workspace calculation implementation:

```cpp
template <bool BATCHED, typename T>
void rocsolver_geql2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // Quick return: no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    size_t s1, s2, w1, w2;
    
    // LARF requirements (largest iteration: m rows, n columns)
    rocsolver_larf_getMemorySize<BATCHED, T>(
        rocblas_side_left,          // left-side application
        m,                          // max reflector size
        n,                          // max columns to update
        batch_count, 
        size_scalars,               // constants: -1, 0, 1
        &s1,                        // norms workspace for DOT
        &w1                         // work for GEMV
    );
    
    // LARFG requirements (largest iteration: m elements)
    rocsolver_larfg_getMemorySize<T>(
        m,                          // max vector size
        batch_count, 
        &w2,                        // work for DOT (norm computation)
        &s2                         // norms workspace
    );
    
    // Take maximum since LARF and LARFG never execute concurrently
    *size_work_workArr = std::max(w1, w2);      // Reusable workspace
    *size_Abyx_norms = std::max(s1, s2);        // Norms array
    
    // Diagonal storage: 1 element per batch instance
    *size_diag = sizeof(T) * batch_count;
}
```

**Why maximum?**
1. Sequential execution: LARFG completes before LARF starts
2. Same memory can be reused for both operations
3. Reduces total allocation vs. sum of requirements

**Workspace breakdown for m=1024, n=512, batch=16, float:**

LARF:
- s1 (norms): DOT workspace ≈ ((m-1)/512 + 2) * 16 * 4 = 256 bytes
- w1 (GEMV work): vector of size m*16 = 64KB

LARFG:
- s2 (norms): DOT workspace ≈ ((m-1)/512 + 2) * 16 * 4 = 256 bytes
- w2 (DOT work): same as s2

Result:
- size_scalars: 3 * 4 = 12 bytes (constants)
- size_work_workArr: max(64KB, 256) = 64KB
- size_Abyx_norms: max(256, 256) = 256 bytes
- size_diag: 16 * 4 = 64 bytes
- **Total: ~64KB (vs 128KB if summed)**

**Memory reuse is safe because:**
- LARFG writes to Abyx_norms, completes
- LARF reads from Abyx_norms (different data)
- No overlap in read/write phases""",
        "rationale": "Workspace reuse between sequential operations (LARFG then LARF) minimizes memory allocation. Taking the maximum size handles the largest requirement from either operation.",
        "tags": ["coding", "memory-management", "workspace", "optimization"]
    })

    # Additional entries to reach >10 total
    
    # L1-4: Right-to-Left Column Processing (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 8),
        "level": "L1",
        "interface": "geql2",
        "instruction": "Explain why QL factorization processes columns from right to left (n-j-1) and how this builds a lower triangular L factor.",
        "context_text": "The column ordering in QL factorization is opposite to QR factorization.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """rocblas_int dim = std::min(m, n);

for(rocblas_int j = 0; j < dim; j++)
{
    // generate Householder reflector to work on column n - j - 1
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), ...);
    // ...
}"""
            }
        ],
        "answer": """Right-to-left processing builds L from bottom-right corner:

**Iteration sequence:**
- j=0: column n-1 (rightmost), uses rows 0 to m-1
- j=1: column n-2, uses rows 0 to m-2
- j=2: column n-3, uses rows 0 to m-3
- ...
- j=dim-1: column 0 (if m>=n) or column n-dim (if m<n)

**Why right-to-left produces lower triangular L:**

For 5×3 matrix:
```
After j=0 (col 2): [  *    *   l22 ]    Diagonal at (4,2), zeros above
                   [  *    *   l32 ]
                   [  *    *   l42 ]
                   [  *    *   l52 ]
                   [  *    *   l62 ]

After j=1 (col 1): [  *   l11  l22 ]    Diagonal at (3,1), zeros above
                   [  *   l21  l32 ]
                   [  *   l31  l42 ]
                   [  *   l41  l52 ]
                   [  *   l51  l62 ]

After j=2 (col 0): [ l00  l11  l22 ]    Diagonal at (2,0), final L
                   [ l10  l21  l32 ]
                   [ l20  l31  l42 ]
                   [ l30  l41  l52 ]
                   [ l40  l51  l62 ]
```

**Key insight:** Each iteration j creates zeros in column (n-j-1) from rows 0 to (m-j-2), leaving the diagonal at row (m-j-1) intact. This naturally produces lower triangular structure from right to left.

Compare to QR: processes left-to-right, creating zeros below diagonal, producing upper triangular R.""",
        "rationale": "Right-to-left column processing with bottom-to-top pivots naturally produces lower triangular L. Each iteration annihilates elements above the diagonal in that column.",
        "tags": ["analysis", "ql-factorization", "column-ordering", "lower-triangular"]
    })

    # L2-3: Tau Storage Order (CODING)
    entries.append({
        "id": str(base_timestamp + 9),
        "level": "L2",
        "interface": "geql2",
        "instruction": "Implement the tau indexing scheme for GEQL2 and explain why tau[dim-j-1] is used instead of tau[j].",
        "context_text": "GEQL2 stores tau values in reverse order compared to GEQR2.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """for(rocblas_int j = 0; j < dim; j++)
{
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(0, n - j - 1, lda), 1, strideA, 
                             (ipiv + dim - j - 1),  // tau stored at index (dim-j-1)
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);
    // ...
}"""
            }
        ],
        "answer": """Tau indexing implementation:

```cpp
rocblas_int dim = std::min(m, n);

for(rocblas_int j = 0; j < dim; j++)
{
    rocblas_int col = n - j - 1;           // Column being processed
    rocblas_int tau_index = dim - j - 1;   // Where to store tau
    
    rocsolver_larfg_template(
        handle, 
        m - j,
        A, shiftA + idx2D(m - j - 1, col, lda),    // alpha
        A, shiftA + idx2D(0, col, lda),            // x vector
        1, strideA,
        (ipiv + tau_index),                         // tau output location
        strideP, batch_count, work, norms
    );
}
```

**Why ipiv[dim-j-1] instead of ipiv[j]?**

Iteration mapping:
| j | Column | Row range | Tau index | Tau location |
|---|--------|-----------|-----------|--------------|
| 0 | n-1    | 0 to m-1  | dim-1     | ipiv[dim-1]  |
| 1 | n-2    | 0 to m-2  | dim-2     | ipiv[dim-2]  |
| 2 | n-3    | 0 to m-3  | dim-3     | ipiv[dim-3]  |
|...| ...    | ...       | ...       | ...          |
|dim-1| n-dim | 0 to m-dim+1 | 0     | ipiv[0]      |

**Rationale:**
1. **Natural correspondence**: ipiv[i] corresponds to column i of the matrix
2. **LAPACK compatibility**: DGEQL2/SGEQL2 use same ordering
3. **Q reconstruction**: Q = H(0) * H(1) * ... * H(dim-1) processes tau left-to-right
4. **Consistency**: tau ordering matches column ordering after completion

**Storage after completion (5×3):**
```
A columns:    0     1     2
ipiv:      tau0  tau1  tau2
           
tau0 for rightmost column (originally col 2, j=2, now at index 0)
tau1 for middle column (originally col 1, j=1, now at index 1)  
tau2 for leftmost column (originally col 0, j=0, now at index 2)
```

This reverse indexing ensures tau[i] corresponds to column i in the final result.""",
        "rationale": "Reverse tau indexing (dim-j-1) ensures that after factorization, ipiv[i] contains the tau value for column i, maintaining consistency with LAPACK conventions and Q reconstruction order.",
        "tags": ["coding", "indexing", "tau-storage", "lapack-compatibility"]
    })

    # L3-3: Production API (CODING)
    entries.append({
        "id": str(base_timestamp + 10),
        "level": "L3",
        "interface": "geql2",
        "instruction": "Implement the complete C API wrapper for GEQL2 for all four precisions (S/D/C/Z) with proper memory allocation and error handling.",
        "context_text": "The production API must handle memory queries, allocation, initialization, and execution.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.cpp",
                "language": "cpp",
                "content": """template <typename T, typename U>
rocblas_status rocsolver_geql2_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("geql2", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes
    size_t size_scalars;
    size_t size_work_workArr;
    size_t size_Abyx_norms;
    size_t size_diag;
    rocsolver_geql2_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_diag);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_diag);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA, ipiv, stridep,
                                       batch_count, (T*)scalars, work_workArr, (T*)Abyx_norms,
                                       (T*)diag);
}

extern "C" {

rocblas_status rocsolver_sgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geql2_impl<float>(handle, m, n, A, lda, ipiv);
}

// ... similar for d/c/z

}"""
            }
        ],
        "answer": """Complete production API implementation with all precisions:

```cpp
ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_geql2_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    ROCSOLVER_ENTER_TOP("geql2", "-m", m, "-n", n, "--lda", lda);

    // 1. Validate handle
    if(!handle)
        return rocblas_status_invalid_handle;

    // 2. Argument validation
    rocblas_status st = rocsolver_geql2_geqlf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // 3. Setup for non-batched execution
    rocblas_int shiftA = 0;
    rocblas_stride strideA = 0;
    rocblas_stride stridep = 0;
    rocblas_int batch_count = 1;

    // 4. Calculate workspace requirements
    size_t size_scalars, size_work_workArr, size_Abyx_norms, size_diag;
    rocsolver_geql2_getMemorySize<false, T>(m, n, batch_count, 
                                            &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_diag);

    // 5. Handle memory size query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, 
                                                      size_work_workArr,
                                                      size_Abyx_norms, size_diag);

    // 6. Allocate device memory
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, 
                              size_Abyx_norms, size_diag);
    
    if(!mem)
        return rocblas_status_memory_error;

    // 7. Extract allocated pointers
    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];
    
    // 8. Initialize scalar constants
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);  // Sets -1, 0, 1

    // 9. Execute algorithm
    return rocsolver_geql2_template<T>(handle, m, n, A, shiftA, lda, strideA, 
                                       ipiv, stridep, batch_count, 
                                       (T*)scalars, work_workArr, 
                                       (T*)Abyx_norms, (T*)diag);
}

ROCSOLVER_END_NAMESPACE

// C API Wrappers
extern "C" {

rocblas_status rocsolver_sgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_geql2_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_geql2_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_geql2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgeql2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_geql2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
}

} // extern "C"
```

**Error handling flow:**
1. Invalid handle → return immediately
2. Invalid arguments → early return from argCheck
3. Memory query → return sizes without allocation
4. Allocation failure → return memory_error
5. Execution → return status from template

**Memory management:**
- rocblas_device_malloc: RAII-style allocation
- Automatic cleanup on scope exit
- Supports query mode for workspace planning""",
        "rationale": "Production API requires proper validation, memory management, and error handling. Template implementation enables code reuse across precisions while C wrappers provide standard LAPACK-compatible interface.",
        "tags": ["coding", "api-design", "memory-management", "error-handling", "production-code"]
    })

    # L2-4: GEQL2 vs GEQR2 Comparison (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 11),
        "level": "L2",
        "interface": "geql2",
        "instruction": "Compare GEQL2 and GEQR2 implementations side-by-side. What code changes are needed to convert QR to QL factorization?",
        "context_text": "GEQL2 and GEQR2 are algorithmic duals: one produces Q*L, the other produces Q*R.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// GEQL2 main loop
rocblas_int dim = std::min(m, n);
for(rocblas_int j = 0; j < dim; j++)
{
    rocsolver_larfg_template(handle, m - j, 
                             A, shiftA + idx2D(m - j - 1, n - j - 1, lda),  // alpha: bottom of column
                             A, shiftA + idx2D(0, n - j - 1, lda),          // x: above alpha
                             1, strideA, (ipiv + dim - j - 1), ...);        // tau index

    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ..., A, shiftA + idx2D(m - j - 1, n - j - 1, lda), ...);

    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, 
                            A, shiftA + idx2D(0, n - j - 1, lda), ...,
                            A, shiftA, ...);  // update columns 0 to (n-j-2)
    
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), ..., A, shiftA + idx2D(m - j - 1, n - j - 1, lda), ...);
}"""
            },
            {
                "path": "library/src/lapack/roclapack_geqr2.hpp",
                "language": "cpp",
                "content": """// GEQR2 main loop  
rocblas_int dim = std::min(m, n);
for(rocblas_int j = 0; j < dim; ++j)
{
    rocsolver_larfg_template(handle, m - j, 
                             A, shiftA + idx2D(j, j, lda),                      // alpha: diagonal
                             A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), // x: below diagonal
                             1, strideA, (ipiv + j), ...);                      // tau index

    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ..., A, shiftA + idx2D(j, j, lda), ...);

    if(j < n - 1)
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1,
                                A, shiftA + idx2D(j, j, lda), ...,
                                A, shiftA + idx2D(j, j + 1, lda), ...);  // update columns (j+1) to (n-1)
    
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), ..., A, shiftA + idx2D(j, j, lda), ...);
}"""
            }
        ],
        "answer": """Side-by-side comparison:

| Aspect | GEQR2 (Q*R) | GEQL2 (Q*L) |
|--------|-------------|-------------|
| **Column order** | j (left→right) | n-j-1 (right→left) |
| **Pivot row** | j (diagonal) | m-j-1 (bottom) |
| **LARFG alpha** | A[j,j] | A[m-j-1, n-j-1] |
| **LARFG x** | A[j+1:m-1, j] | A[0:m-j-2, n-j-1] |
| **Tau index** | j | dim-j-1 |
| **set_diag loc** | (j, j) | (m-j-1, n-j-1) |
| **LARF size** | (m-j) × (n-j-1) | (m-j) × (n-j-1) |
| **LARF update** | A[j:m-1, j+1:n-1] | A[0:m-j-1, 0:n-j-2] |

**Code transformation QR → QL:**

```cpp
// QR to QL changes:
1. Column index:
   QR: j                  → QL: n - j - 1
   
2. Row index:  
   QR: j                  → QL: m - j - 1
   
3. Tau storage:
   QR: ipiv + j           → QL: ipiv + dim - j - 1
   
4. LARF offset:
   QR: idx2D(j, j+1, lda) → QL: idx2D(0, 0, lda)
```

**Minimal diff:**
```cpp
- for(rocblas_int j = 0; j < dim; ++j)
+ for(rocblas_int j = 0; j < dim; j++)
  {
+     rocblas_int col = n - j - 1;
+     rocblas_int row = m - j - 1;
+     
-     rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda),
-                              A, shiftA + idx2D(j+1, j, lda), ...);
+     rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(row, col, lda),
+                              A, shiftA + idx2D(0, col, lda), ...);
      
-     ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ..., shiftA + idx2D(j, j, lda), ...);
+     ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ..., shiftA + idx2D(row, col, lda), ...);
      
-     rocsolver_larf_template(..., shiftA + idx2D(j, j+1, lda), ...);
+     rocsolver_larf_template(..., shiftA + idx2D(0, 0, lda), ...);
  }
```

**Mathematical duality:**
- QR: A = Q*R where R is upper triangular
- QL: A = Q*L where L is lower triangular
- Both use Householder reflectors, just from opposite corners""",
        "rationale": "GEQL2 is algorithmically symmetric to GEQR2 with reversed indexing. The core operations (LARFG, LARF) are identical, only the matrix regions change. This duality reflects the mathematical equivalence of QR and QL factorizations.",
        "tags": ["analysis", "comparison", "qr-vs-ql", "algorithm-design"]
    })

    # L3-4: Blocked GEQLF Relationship (ANALYSIS)
    entries.append({
        "id": str(base_timestamp + 12),
        "level": "L3",
        "interface": "geql2",
        "instruction": "Explain how GEQL2 serves as the foundation for blocked GEQLF algorithm. What are the key differences and why does blocking improve performance?",
        "context_text": "GEQL2 is the unblocked variant used within blocked GEQLF for panel factorization.",
        "code_blocks": [
            {
                "path": "library/src/lapack/roclapack_geql2.hpp",
                "language": "cpp",
                "content": """// GEQL2: Unblocked algorithm
for(rocblas_int j = 0; j < dim; j++)
{
    // Factor column (n-j-1)
    rocsolver_larfg_template(handle, m - j, ...);
    
    // Apply reflector to columns 0 to (n-j-2)
    rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, ...);
}
// Complexity: O(mn²) with BLAS-2 operations (GEMV)"""
            }
        ],
        "answer": """GEQL2 as foundation for blocked GEQLF:

**GEQL2 (Unblocked):**
- Processes one column at a time
- Uses BLAS-2 (LARF with GEMV)
- Complexity: ~2mn² - (2/3)n³ FLOPs
- Memory-bound: arithmetic intensity ~0.25 FLOP/byte
- Used for small matrices (n < 128) or final panel

**GEQLF (Blocked):**
```
blk = getBlockSize(n)
for j = 0, blk, 2*blk, ..., (n - n%blk):
    jb = min(blk, remaining columns)
    
    // 1. Panel factorization (right jb columns)
    GEQL2(A[0:m-1, n-j-jb:n-j-1])              // Use unblocked GEQL2
    
    // 2. Build triangular factor T for block reflector
    LARFT(V, tau, T)                            // T is jb×jb
    
    // 3. Apply block reflector to left part
    LARFB(V, T, A[0:m-1, 0:n-j-jb-1])          // Uses GEMM
```

**Key differences:**

| Aspect | GEQL2 | GEQLF |
|--------|-------|-------|
| Granularity | Column-by-column | Block of columns |
| Main operation | LARF (GEMV) | LARFB (GEMM) |
| Arithmetic intensity | ~0.25 FLOP/byte | ~10 FLOP/byte |
| Parallelism | Limited | High (GEMM) |
| Cache efficiency | Poor | Good |
| Complexity class | BLAS-2 | BLAS-3 |

**Performance improvement for n=1024, m=1024:**

GEQL2:
- ~2*1024*1024² ≈ 2.1B FLOPs
- Memory traffic: ~8.4GB (assuming 4-byte float)
- Peak: 250 GFLOP/s on memory-bound GPU = 8.4 seconds
- Actual: much slower due to kernel launch overhead

GEQLF (blk=64):
- Same FLOPs: ~2.1B
- Memory traffic: reduced due to cache reuse in GEMM
- GEMM blocks achieve ~5000 GFLOP/s
- Time: ~0.4 seconds (20× faster)

**Why blocking helps:**
1. **GEMM efficiency**: 64×64 blocks fit in cache, high FLOP/byte ratio
2. **Kernel launch amortization**: ~16 GEMM calls vs 1024 LARF calls
3. **Cache reuse**: V block loaded once, used for entire left matrix update
4. **Parallelism**: GEMM saturates GPU compute units

**Usage hierarchy:**
- n < 32: Direct GEQL2 (unblocked)
- 32 ≤ n < 128: GEQL2 with small-matrix kernel
- n ≥ 128: GEQLF with GEQL2 for panels

**Panel factorization:**
GEQLF calls GEQL2 for the jb columns, which is still unblocked but on a smaller problem. The blocked structure comes from using LARFB (GEMM-based) for the trailing matrix update.""",
        "rationale": "GEQL2 provides the unblocked core algorithm that GEQLF uses for panel factorization. Blocking with LARFB/GEMM converts memory-bound BLAS-2 into compute-bound BLAS-3, achieving order-of-magnitude speedups on modern GPUs.",
        "tags": ["analysis", "blocked-algorithm", "performance", "blas-levels", "geqlf"]
    })

    return entries

def main():
    entries = generate_entries()
    output_file = Path(__file__).parent / "roclapack_geql2.jsonl"
    
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    # Count coding tasks
    coding_count = sum(1 for e in entries if "coding" in e.get("tags", []))
    
    print(f"Generated {len(entries)} entries")
    print(f"Output: {output_file}")
    print(f"Coding tasks: {coding_count}/{len(entries)} ({coding_count/len(entries)*100:.1f}%)")
    print(f"L1 entries: {sum(1 for e in entries if e['level'] == 'L1')}")
    print(f"L2 entries: {sum(1 for e in entries if e['level'] == 'L2')}")
    print(f"L3 entries: {sum(1 for e in entries if e['level'] == 'L3')}")

if __name__ == "__main__":
    main()
