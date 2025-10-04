#!/usr/bin/env python3
"""
Generator for GERQ2 dataset - RQ factorization (unblocked)

GERQ2 computes an RQ factorization of an m-by-n matrix A using Householder reflectors.
The factorization has the form A = R * Q where R is upper trapezoidal and Q is orthogonal.
Processes rows from BOTTOM to TOP (j=0..dim-1 → row m-j-1), applying reflectors from RIGHT.
"""

import json
import time
from pathlib import Path

def create_entry(level, interface, instruction, context_text, code_blocks, answer, rationale, tags):
    """Helper to create a dataset entry"""
    return {
        "id": str(int(time.time() * 1000000)),
        "level": level,
        "interface": interface,
        "instruction": instruction,
        "context_text": context_text,
        "code_blocks": code_blocks,
        "answer": answer,
        "rationale": rationale,
        "tags": tags
    }

entries = []

# L1-1: CODING - Bottom-to-top row processing
entries.append(create_entry(
    level="L1",
    interface="gerq2",
    instruction="Implement the row indexing logic for GERQ2 that processes rows from bottom to top. For iteration j, which row is processed, where is the Householder vector stored, and where is tau stored?",
    context_text="GERQ2 processes rows in reverse order: j=0 processes row m-1, j=1 processes row m-2, etc. This creates an RQ factorization where R appears in the bottom-right corner.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """rocblas_int dim = std::min(m, n); // total number of pivots

for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(m - j - 1, 0, lda), lda,
                                    strideA, batch_count);

    // generate Householder reflector to work on row m - j - 1
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                                (ipiv + dim - j - 1), strideP, A, shiftA, lda, strideA,
                                batch_count, scalars, Abyx_norms, (T**)work_workArr);
    }
}"""
        }
    ],
    answer="""Here's the complete row indexing for bottom-to-top GERQ2:

```cpp
rocblas_int dim = std::min(m, n);  // Total pivots: min(5,4)=4 for 5x4 matrix

for(rocblas_int j = 0; j < dim; ++j)
{
    // Current row index: m - j - 1 (bottom to top)
    rocblas_int row = m - j - 1;

    // Tau storage index: dim - j - 1 (reverse order)
    rocblas_int tau_idx = dim - j - 1;

    // Vector length: n - j (decreases as we go up)
    rocblas_int vec_len = n - j;

    // Diagonal position: A(row, n-j-1) = A(m-j-1, n-j-1)
    rocblas_int diag_col = n - j - 1;

    // Example for 5x4 matrix:
    // j=0: row=4, tau_idx=3, vec_len=4, diagonal=A(4,3) ← Bottom row
    // j=1: row=3, tau_idx=2, vec_len=3, diagonal=A(3,2)
    // j=2: row=2, tau_idx=1, vec_len=2, diagonal=A(2,1)
    // j=3: row=1, tau_idx=0, vec_len=1, diagonal=A(1,0) ← Stop (row 0 unchanged)

    // Conjugate row (for complex)
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            vec_len,                              // n-j elements
            A, shiftA + idx2D(row, 0, lda),      // Start: A(m-j-1, 0)
            lda,                                  // Stride: lda (row elements)
            strideA, batch_count);
    }

    // Generate Householder reflector
    rocsolver_larfg_template(
        handle,
        vec_len,                                  // Vector length: n-j
        A, shiftA + idx2D(row, diag_col, lda),   // Alpha: A(m-j-1, n-j-1)
        A, shiftA + idx2D(row, 0, lda),          // x: A(m-j-1, 0:n-j-2)
        lda, strideA,                             // incx=lda (row stride)
        (ipiv + tau_idx), strideP,               // tau[dim-j-1]
        batch_count, work_workArr, Abyx_norms);

    // Save diagonal and replace with 1.0
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>),
                            dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            diag, 0, 1,
                            A, shiftA + idx2D(row, diag_col, lda),
                            lda, strideA, 1, true);

    // Apply reflector from right to rows above
    if(j < m - 1)  // If not the top row
    {
        rocsolver_larf_template(
            handle,
            rocblas_side_right,                   // A * H
            m - j - 1,                            // Rows to update: 0 to m-j-2
            vec_len,                              // Reflector size: n-j
            A, shiftA + idx2D(row, 0, lda),      // Householder vector (row m-j-1)
            lda, strideA,                         // incx=lda
            (ipiv + tau_idx), strideP,           // tau[dim-j-1]
            A, shiftA,                            // Submatrix: A(0:m-j-2, 0:n-j-1)
            lda, strideA,
            batch_count, scalars, Abyx_norms, work_workArr);
    }

    // Restore diagonal
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>),
                            dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            diag, 0, 1,
                            A, shiftA + idx2D(row, diag_col, lda),
                            lda, strideA, 1);

    // Unconjugate (only first n-j-1 elements, diagonal is real)
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            vec_len - 1,                          // n-j-1 elements (not diagonal)
            A, shiftA + idx2D(row, 0, lda),
            lda, strideA, batch_count);
    }
}
```

**Indexing Summary for 5×4 matrix:**

| j | Row (m-j-1) | Tau index (dim-j-1) | Vec length (n-j) | Diagonal | Updates rows |
|---|-------------|---------------------|------------------|----------|--------------|
| 0 | 4 (bottom)  | 3                   | 4                | A(4,3)   | 0-3          |
| 1 | 3           | 2                   | 3                | A(3,2)   | 0-2          |
| 2 | 2           | 1                   | 2                | A(2,1)   | 0-1          |
| 3 | 1           | 0                   | 1                | A(1,0)   | none         |

**Key Points:**
- Row index: m-j-1 (descends from m-1 to m-dim)
- Tau index: dim-j-1 (reverse order in tau array)
- Vector starts at column 0, ends at column n-j-1
- LARF updates rows 0 to m-j-2 (all rows above current)
- Conjugation: full row before LARFG, only first n-j-1 elements after (diagonal is real)""",
    rationale="GERQ2 processes rows bottom-to-top (m-j-1) to build R factor in bottom-right. Tau is stored in reverse order (dim-j-1) to match row processing. The Householder vector for row k is stored in A(k, 0:n-j-2), with diagonal A(k, n-j-1) holding the R factor. LARF updates all rows above (0 to m-j-2) from the right.",
    tags=["coding", "bottom-to-top-processing", "row-indexing", "rq-factorization", "reverse-order", "householder-storage"]
))

# L1-2: ANALYSIS - RQ vs LQ factorization structure
entries.append(create_entry(
    level="L1",
    interface="gerq2",
    instruction="Explain why GERQ2 processes rows from bottom to top while GELQ2 processes top to bottom. How does this affect where the R factor appears vs the L factor?",
    context_text="GERQ2 creates RQ factorization (A = R*Q) with R upper trapezoidal, while GELQ2 creates LQ factorization (A = L*Q) with L lower trapezoidal. The row processing order determines factor placement.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // generate Householder reflector to work on row m - j - 1
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);

    // Apply Householder reflector to the rest of matrix from the right
    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                                (ipiv + dim - j - 1), strideP, A, shiftA, lda, strideA,
                                batch_count, scalars, Abyx_norms, (T**)work_workArr);
    }
}"""
        }
    ],
    answer="""Bottom-to-top processing creates RQ factorization structure:

**GERQ2 (RQ factorization):**
- **Processing**: Rows m-1 → m-dim (BOTTOM to TOP)
- **Reflector H_j**: Zeros row m-j-1 to the LEFT of diagonal
- **Application**: A * H_j (RIGHT multiplication)
- **Factor R**: UPPER trapezoidal in BOTTOM-RIGHT corner
- **Result**: A = R * Q

**Example: 5×4 matrix**

Iteration j=0 (row 4, bottom):
```
Before: [ *  *  *  * ]    After: [ *  *  *  * ]
        [ *  *  *  * ]           [ *  *  *  * ]
        [ *  *  *  * ]    H₀ →   [ *  *  *  * ]
        [ *  *  *  * ]           [ *  *  *  * ]
        [ *  *  *  * ]           [ 0  0  0  R₄₃]

H₀ zeros A(4, 0:2) by acting on columns 0:3
```

Iteration j=1 (row 3):
```
Before: [ *  *  *  * ]    After: [ *  *  *  * ]
        [ *  *  *  * ]           [ *  *  *  * ]
        [ *  *  *  * ]    H₁ →   [ *  *  *  * ]
        [ *  *  *  * ]           [ 0  0  R₃₂ R₃₃]
        [ 0  0  0  R₄₃]          [ 0  0  0   R₄₃]

H₁ zeros A(3, 0:1), preserves column 3 (already factored)
```

Final R factor (upper trapezoidal, bottom-right):
```
A = [ 0   0   0   0  ]      R = [ 0   0   0   0  ]
    [ 0   0   0   0  ]          [ 0   0   0   0  ]
    [ 0   0   R₂₁ R₂₂]          [ 0   0   R₂₁ R₂₂]
    [ 0   0   R₃₂ R₃₃]          [ 0   0   R₃₂ R₃₃]
    [ 0   0   0   R₄₃]          [ 0   0   0   R₄₃]
```

**GELQ2 (LQ factorization) - Comparison:**
- **Processing**: Rows 0 → dim-1 (TOP to BOTTOM)
- **Reflector H_j**: Zeros row j to the RIGHT of diagonal
- **Application**: A * H_j (RIGHT multiplication)
- **Factor L**: LOWER trapezoidal in TOP-LEFT corner
- **Result**: A = L * Q

**Why Bottom-to-Top for RQ?**

1. **R factor placement**: Upper trapezoidal must be in bottom-right
   - Bottom row (m-1) has diagonal at rightmost position
   - Each row above has diagonal one column to the left
   - Creates upper triangular structure aligned bottom-right

2. **Preservation of factored rows**:
   - j=0 processes row m-1, creates R(m-1, n-1)
   - j=1 processes row m-2, must preserve row m-1
   - RIGHT multiplication A*H_j affects all rows, but H_j only modifies columns 0:n-j-1
   - Column n-j:n-1 already factored, preserved

3. **Contrast with other factorizations**:

| Factorization | Order | Direction | Factor | Location |
|---------------|-------|-----------|--------|----------|
| GEQR2 (QR) | Col 0→n-1 | Left→Right | R upper | Top-left |
| GEQL2 (QL) | Col n-1→0 | Right→Left | L lower | Bottom-right |
| GELQ2 (LQ) | Row 0→m-1 | Top→Bottom | L lower | Top-left |
| GERQ2 (RQ) | Row m-1→0 | Bottom→Top | R upper | Bottom-right |

**Mathematical Insight:**

For RQ: A = R * Q where R is m×n upper trapezoidal

Row m-1: [ 0  0  ...  0  r_{m-1,n-1} ] = r_{m-1,n-1} * q_{n-1,:}

Building from bottom ensures each r_{i,j} is computed before being used as a pivot for row i-1.

**Storage Pattern:**
- GERQ2 stores Householder vectors in A(k, 0:j-1) for row k
- Diagonal A(k, j) holds R factor
- Tau stored in reverse: tau[dim-j-1] for row m-j-1""",
    rationale="GERQ2 processes bottom-to-top to place the upper trapezoidal R factor in the bottom-right corner, matching the natural structure of RQ factorization. Each iteration zeros elements to the left of the diagonal, preserving already-factored columns on the right. This is the row-based counterpart to GEQL2's right-to-left column processing.",
    tags=["analysis", "rq-factorization", "bottom-to-top-processing", "factor-placement", "matrix-structure", "lq-vs-rq"]
))

# L1-3: CODING - Asymmetric conjugation for GERQ2
entries.append(create_entry(
    level="L1",
    interface="gerq2",
    instruction="Implement the asymmetric conjugation pattern in GERQ2: conjugate n-j elements before LARFG, but only unconjugate n-j-1 elements after LARF. Why is this different from GELQ2's symmetric conjugation?",
    context_text="GERQ2 conjugates the full row (n-j elements) before LARFG, but only unconjugates n-j-1 elements after LARF, excluding the diagonal. This differs from GELQ2 which conjugates/unconjugates the same number of elements.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // conjugate the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(m - j - 1, 0, lda), lda,
                                    strideA, batch_count);

    // generate Householder reflector to work on row m - j - 1
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);

    // ... set_diag, larf, restore_diag ...

    // restore the jth row of A
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(m - j - 1, 0, lda),
                                    lda, strideA, batch_count);
}"""
        }
    ],
    answer="""Here's the asymmetric conjugation pattern for GERQ2:

```cpp
for(rocblas_int j = 0; j < dim; ++j)
{
    rocblas_int row = m - j - 1;
    rocblas_int vec_len = n - j;
    rocblas_int diag_col = n - j - 1;

    // Step 1: Conjugate FULL row (n-j elements: columns 0 to n-j-1)
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            vec_len,                          // n-j elements (INCLUDES diagonal)
            A, shiftA + idx2D(row, 0, lda),  // A(m-j-1, 0:n-j-1)
            lda,                              // Row stride
            strideA, batch_count);

        // Before: [a₀+ib₀, a₁+ib₁, ..., a_{n-j-1}+ib_{n-j-1}]
        // After:  [a₀-ib₀, a₁-ib₁, ..., a_{n-j-1}-ib_{n-j-1}]
    }

    // Step 2: LARFG generates reflector
    // Input:  Conjugated row (including conjugated diagonal)
    // Output: R diagonal (REAL) in A(row, diag_col)
    //         Householder vector in A(row, 0:n-j-2)
    rocsolver_larfg_template(
        handle,
        vec_len,                              // n-j
        A, shiftA + idx2D(row, diag_col, lda),  // Alpha (diagonal, conjugated)
        A, shiftA + idx2D(row, 0, lda),        // x (columns 0:n-j-2, conjugated)
        lda, strideA,
        (ipiv + dim - j - 1), strideP,
        batch_count, work_workArr, Abyx_norms);

    // After LARFG:
    // A(row, diag_col) = R factor diagonal (REAL, even for complex!)
    // A(row, 0:n-j-2) = Householder vector (still CONJUGATED)

    // Step 3: set_diag, LARF, restore_diag
    ROCSOLVER_LAUNCH_KERNEL((set_diag<T>), ...,
                            A, shiftA + idx2D(row, diag_col, lda), ...);

    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, vec_len,
                               A, shiftA + idx2D(row, 0, lda), lda, strideA,
                               (ipiv + dim - j - 1), strideP,
                               A, shiftA, lda, strideA,
                               batch_count, scalars, Abyx_norms, work_workArr);
    }

    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T>), ...,
                            A, shiftA + idx2D(row, diag_col, lda), ...);

    // Step 4: Unconjugate ONLY Householder vector (n-j-1 elements: columns 0 to n-j-2)
    // EXCLUDE diagonal!
    if(COMPLEX)
    {
        rocsolver_lacgv_template<T>(
            handle,
            vec_len - 1,                      // n-j-1 elements (EXCLUDES diagonal)
            A, shiftA + idx2D(row, 0, lda),  // A(m-j-1, 0:n-j-2)
            lda, strideA, batch_count);

        // Before: [v₀, v₁, ..., v_{n-j-2}] (conjugated) | R_{diag} (real)
        // After:  [v₀', v₁', ..., v_{n-j-2}'] (unconjugated) | R_{diag} (unchanged)
    }
}
```

**Why Asymmetric (n-j vs n-j-1)?**

1. **LARFG transforms diagonal to REAL**:
   - Input: Complex diagonal a_{n-j-1} + ib_{n-j-1} (conjugated)
   - Output: Real R factor diagonal (magnitude)
   - No imaginary part to unconjugate!

2. **Before conjugation (row with n-j complex elements)**:
   ```
   [a₀+ib₀, a₁+ib₁, ..., a_{n-j-2}+ib_{n-j-2}, a_{n-j-1}+ib_{n-j-1}]
   ```

3. **After LARFG (diagonal is now REAL)**:
   ```
   [v₀-iv₀', v₁-iv₁', ..., v_{n-j-2}-iv'_{n-j-2}, R_{real}]
    ← Householder vector (conjugated) →  ← R factor (real) →
   ```

4. **After unconjugation (only n-j-1 elements)**:
   ```
   [v₀+iv₀', v₁+iv₁', ..., v_{n-j-2}+iv'_{n-j-2}, R_{real}]
    ← Householder vector (restored) →    ← Unchanged →
   ```

**Contrast with GELQ2 (Symmetric conjugation):**

GELQ2 conjugates and unconjugates the SAME count because:
- Diagonal stays at same position throughout
- Diagonal is saved/restored as complex value
- After restore, diagonal needs unconjugation like other elements

GERQ2 is asymmetric because:
- LARFG REPLACES complex diagonal with REAL R factor
- Restore_diag writes back REAL value
- No need to unconjugate a real number!

**Memory Access Pattern:**
```
Conjugate:     A(row, 0), A(row, lda), ..., A(row, (n-j-1)*lda)     [n-j elements]
Unconjugate:   A(row, 0), A(row, lda), ..., A(row, (n-j-2)*lda)     [n-j-1 elements]
                                             ↑ Stop one element early
```""",
    rationale="GERQ2's asymmetric conjugation (n-j before, n-j-1 after) occurs because LARFG transforms the complex diagonal into a REAL R factor diagonal. After restore_diag, the diagonal is real and doesn't need unconjugation. Only the Householder vector (n-j-1 elements) needs to be unconjugated. GELQ2 is symmetric because it saves/restores the diagonal as complex.",
    tags=["coding", "complex-arithmetic", "asymmetric-conjugation", "real-diagonal", "rq-factorization", "larfg-transformation"]
))

# L2-1: ANALYSIS - Workspace comparison RQ vs LQ
entries.append(create_entry(
    level="L2",
    interface="gerq2",
    instruction="Compare the workspace requirements of GERQ2 and GELQ2. Both use RIGHT-side LARF and process rows, but do they need the same workspace size? Analyze for a 1024×512 matrix.",
    context_text="Both GERQ2 and GELQ2 apply reflectors from the right and process rows, but the matrix dimensions and iteration patterns differ.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """template <bool BATCHED, typename T>
void rocsolver_gerq2_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    // if quick return no workspace needed
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
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_right, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(T) * batch_count;
}"""
        }
    ],
    answer="""Workspace comparison for 1024×512 matrix (double precision):

**GERQ2 (m=1024, n=512):**
```cpp
// LARF workspace (RIGHT-side)
rocsolver_larf_getMemorySize<BATCHED, T>(
    rocblas_side_right,  // A * H
    m=1024,              // Matrix rows
    n=512,               // Reflector size
    batch_count,
    &size_scalars,       // 3 * sizeof(double) = 24 bytes
    &s1,                 // Temp for GEMV: m * sizeof(T) = 1024 * 8 = 8 KB
    &w1);                // Work/pointer array

// LARFG workspace
rocsolver_larfg_getMemorySize<T>(
    n=512,               // Max row length (j=0)
    batch_count,
    &w2,                 // DOT reduction: ((512-2)/512 + 2) * 8 = 16 bytes
    &s2);                // Norm storage: 8 bytes

// Final sizes
size_scalars:      24 bytes
size_work_workArr: max(w1, w2) ≈ 4 KB (pointer array)
size_Abyx_norms:   max(8 KB, 8 bytes) = 8 KB
size_diag:         8 bytes (single element per batch)

Total: ~12 KB
```

**GELQ2 (m=1024, n=512) - IDENTICAL workspace!**
```cpp
rocsolver_larf_getMemorySize<BATCHED, T>(
    rocblas_side_right,  // A * H (same as GERQ2!)
    m=1024,
    n=512,
    batch_count, ...);

rocsolver_larfg_getMemorySize<T>(
    n=512,               // Same max row length
    batch_count, ...);

Total: ~12 KB (identical)
```

**Why Identical Despite Different Processing Order?**

1. **Same LARF parameters**:
   - Both: rocblas_side_right
   - Both: Matrix dimensions m×n
   - RIGHT-side LARF temp = m * sizeof(T) (same for both)

2. **Same LARFG parameters**:
   - GERQ2: max row length = n (at j=0, row m-1)
   - GELQ2: max row length = n (at j=0, row 0)
   - Same DOT workspace

3. **Same diagonal storage**:
   - Both: Per-iteration backup (sizeof(T) * batch)
   - Not dim-dependent like GEQR2

**Workspace Formula (both RQ and LQ):**
```
size_scalars:      3 * sizeof(T)
size_work_workArr: max(LARF_work, LARFG_work)
                 ≈ sizeof(T*) * batch (for batched)
size_Abyx_norms:   max(m * sizeof(T), sizeof(T))  ← RIGHT-side uses m!
                 = m * sizeof(T)
size_diag:         sizeof(T) * batch

Total ≈ m * sizeof(T) * batch  (dominated by Abyx_norms)
```

**Contrast with Column-based (QR/QL):**

**GEQR2 (m=512, n=1024):**
```cpp
rocsolver_larf_getMemorySize<BATCHED, T>(
    rocblas_side_left,   // H * A (LEFT!)
    m=512,
    n=1024,
    ...);

// LEFT-side LARF temp = n * sizeof(T) = 1024 * 8 = 8 KB
// LARFG: m = 512 (column length)

size_Abyx_norms: max(8 KB, 8 bytes) = 8 KB
Total: ~12 KB
```

**Key Insight: Workspace depends on LARF side, not factorization type!**

| Factorization | LARF side | LARF temp size | Max row/col | Workspace |
|---------------|-----------|----------------|-------------|-----------|
| GEQR2 (m×n) | LEFT | n * sizeof(T) | m (col len) | max(m,n)*sizeof(T) |
| GEQL2 (m×n) | LEFT | n * sizeof(T) | m (col len) | max(m,n)*sizeof(T) |
| GELQ2 (m×n) | RIGHT | m * sizeof(T) | n (row len) | max(m,n)*sizeof(T) |
| GERQ2 (m×n) | RIGHT | m * sizeof(T) | n (row len) | max(m,n)*sizeof(T) |

**For 1024×512:**
- QR/QL: max(1024, 512) * 8 = 8 KB
- LQ/RQ: max(1024, 512) * 8 = 8 KB

**For 512×1024:**
- QR/QL: max(512, 1024) * 8 = 8 KB
- LQ/RQ: max(512, 1024) * 8 = 8 KB

Workspace is ALWAYS max(m,n) * sizeof(T) regardless of factorization!""",
    rationale="GERQ2 and GELQ2 have identical workspace because both use RIGHT-side LARF with the same m,n parameters. LARF temp size depends on side: LEFT uses n, RIGHT uses m. LARFG depends on max vector length (n for row-based, m for column-based). Bottom-to-top vs top-to-bottom processing doesn't affect workspace, only the iteration order.",
    tags=["analysis", "workspace-management", "rq-vs-lq", "larf-side-effects", "memory-requirements", "gerq2-vs-gelq2"]
))

# L2-2: CODING - Reverse tau storage pattern
entries.append(create_entry(
    level="L2",
    interface="gerq2",
    instruction="Implement the reverse tau storage pattern used by GERQ2 and explain why tau[0] corresponds to the TOP row (row m-dim) while tau[dim-1] corresponds to the BOTTOM row (row m-1). How does this differ from GELQ2?",
    context_text="GERQ2 stores tau in reverse order relative to row processing: the first processed row (bottom, m-1) gets tau[dim-1], the last processed row gets tau[0]. This is intentional for compatibility with blocked RQ.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """for(rocblas_int j = 0; j < dim; ++j)
{
    // generate Householder reflector to work on row m - j - 1
    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);
}"""
        }
    ],
    answer="""Here's the reverse tau storage pattern for GERQ2:

```cpp
template <typename T, typename U>
void gerq2_with_reverse_tau_storage(rocblas_handle handle,
                                    rocblas_int m, rocblas_int n,
                                    U A, rocblas_int lda, rocblas_stride strideA,
                                    T* ipiv, rocblas_stride strideP,
                                    rocblas_int batch_count, ...)
{
    rocblas_int dim = std::min(m, n);

    // Tau storage mapping:
    // Row m-1 (j=0)   → tau[dim-1]
    // Row m-2 (j=1)   → tau[dim-2]
    // ...
    // Row m-dim (j=dim-1) → tau[0]

    for(rocblas_int j = 0; j < dim; ++j)
    {
        rocblas_int row = m - j - 1;          // Bottom to top: m-1, m-2, ..., m-dim
        rocblas_int tau_idx = dim - j - 1;    // Reverse index: dim-1, dim-2, ..., 0

        // Generate reflector for row (m-j-1)
        rocsolver_larfg_template(
            handle,
            n - j,                                    // Vector length
            A, shiftA + idx2D(row, n - j - 1, lda),  // Alpha
            A, shiftA + idx2D(row, 0, lda),          // x
            lda, strideA,
            (ipiv + tau_idx),                         // tau[dim-j-1]
            strideP,
            batch_count, work_workArr, Abyx_norms);

        // Apply with same tau index
        rocsolver_larf_template(handle, rocblas_side_right,
                               m - j - 1, n - j,
                               A, shiftA + idx2D(row, 0, lda), lda, strideA,
                               (ipiv + tau_idx),                 // tau[dim-j-1]
                               strideP, A, shiftA, lda, strideA,
                               batch_count, ...);
    }
}
```

**Tau Mapping Example (5×4 matrix, dim=4):**

| Iteration j | Row (m-j-1) | Tau index (dim-j-1) | Tau location | Physical row |
|-------------|-------------|---------------------|--------------|--------------|
| 0           | 4 (bottom)  | 3                   | tau[3]       | Row 4        |
| 1           | 3           | 2                   | tau[2]       | Row 3        |
| 2           | 2           | 1                   | tau[1]       | Row 2        |
| 3           | 1           | 0                   | tau[0]       | Row 1        |

**Resulting Tau Array:**
```
tau[0] = τ for row 1 (TOP of factored block)
tau[1] = τ for row 2
tau[2] = τ for row 3
tau[3] = τ for row 4 (BOTTOM of factored block)
```

**Why Reverse Order?**

1. **Compatibility with blocked GERQF**:
   - Blocked RQ processes panels bottom-to-top
   - Needs tau in top-to-bottom order for LARFB application
   - tau[0] for topmost row of panel, tau[dim-1] for bottom row

2. **Q construction order**:
   - Q = H_{m-dim} * H_{m-dim+1} * ... * H_{m-2} * H_{m-1}
   - tau[0] → H_{m-dim} (first in product, topmost row)
   - tau[dim-1] → H_{m-1} (last in product, bottom row)

3. **Natural indexing for blocked algorithms**:
   ```cpp
   // In GERQF, after unblocked tail:
   // tau[0..nb-1] corresponds to rows panel_start to panel_start+nb-1
   // Sequential access in Q construction
   ```

**Contrast with GELQ2 (Forward tau storage):**

```cpp
// GELQ2: Top to bottom processing
for(rocblas_int j = 0; j < dim; ++j)
{
    rocblas_int row = j;              // Top to bottom: 0, 1, 2, ...
    rocblas_int tau_idx = j;          // Forward index: 0, 1, 2, ...

    rocsolver_larfg_template(..., (ipiv + tau_idx), ...);
}

// Tau mapping:
// Row 0 → tau[0]
// Row 1 → tau[1]
// ...
// Row dim-1 → tau[dim-1]
```

GELQ2 uses forward storage because:
- Processes top-to-bottom
- tau[j] naturally maps to row j
- Q = H_0 * H_1 * ... * H_{dim-1} (forward order)

**Implementation Pattern:**
```cpp
// General formula for row-based factorizations:
if(bottom_to_top)  // RQ, QL
{
    row = m - j - 1;
    tau_idx = dim - j - 1;  // Reverse
}
else  // LQ, QR
{
    row = j;
    tau_idx = j;  // Forward
}
```

**Accessing Tau for Q Construction:**
```cpp
// GERQ2: Build Q from stored factors
// Q = H_{m-dim} * H_{m-dim+1} * ... * H_{m-1}
for(rocblas_int i = 0; i < dim; ++i)
{
    rocblas_int row = m - dim + i;    // Top to bottom of factored block
    T tau = ipiv[i];                   // tau[0], tau[1], ..., tau[dim-1]
    // Apply H_row using tau
}
```""",
    rationale="GERQ2 uses reverse tau storage (tau[dim-j-1] for row m-j-1) to provide natural sequential access during Q construction and compatibility with blocked GERQF. The first processed row (bottom, m-1) gets tau[dim-1], while the last processed row (m-dim) gets tau[0]. This differs from GELQ2's forward storage where tau[j] maps to row j.",
    tags=["coding", "tau-storage", "reverse-indexing", "bottom-to-top-processing", "rq-factorization", "blocked-compatibility"]
))

# L3-1: ANALYSIS - Complete algorithm trace
entries.append(create_entry(
    level="L3",
    interface="gerq2",
    instruction="Trace the complete GERQ2 algorithm for a 512×1024 double-precision complex matrix, including all kernel launches, tau storage, and conjugation operations. Compare computational cost with GELQ2 for the same matrix.",
    context_text="GERQ2 on 512×1024 processes 512 rows bottom-to-top with complex conjugation. Analyze the complete execution including reverse tau storage pattern.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """rocblas_int dim = std::min(m, n);

for(rocblas_int j = 0; j < dim; ++j)
{
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j, A, shiftA + idx2D(m - j - 1, 0, lda), lda,
                                    strideA, batch_count);

    rocsolver_larfg_template(handle, n - j, A, shiftA + idx2D(m - j - 1, n - j - 1, lda), A,
                             shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                             (ipiv + dim - j - 1), strideP, batch_count, (T*)work_workArr,
                             Abyx_norms);

    ROCSOLVER_LAUNCH_KERNEL((set_diag<T, rocblas_int>), dim3(batch_count, 1, 1), dim3(1, 1, 1),
                            0, stream, diag, 0, 1, A, shiftA + idx2D(m - j - 1, n - j - 1, lda),
                            lda, strideA, 1, true);

    if(j < m - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_right, m - j - 1, n - j, A,
                                shiftA + idx2D(m - j - 1, 0, lda), lda, strideA,
                                (ipiv + dim - j - 1), strideP, A, shiftA, lda, strideA,
                                batch_count, scalars, Abyx_norms, (T**)work_workArr);
    }

    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, rocblas_int>), dim3(batch_count, 1, 1),
                            dim3(1, 1, 1), 0, stream, diag, 0, 1, A,
                            shiftA + idx2D(m - j - 1, n - j - 1, lda), lda, strideA, 1);

    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, n - j - 1, A, shiftA + idx2D(m - j - 1, 0, lda),
                                    lda, strideA, batch_count);
}"""
        }
    ],
    answer="""Complete GERQ2 trace for 512×1024 complex FP64:

**Setup:**
- m=512, n=1024, dim=min(512,1024)=512
- Type: rocblas_double_complex (16 bytes)
- Processing: Rows 511→0 (bottom to top)
- Tau: Reverse storage tau[511]→tau[0]

**Per Iteration j (0 to 511):**

| j | Row (m-j-1) | Tau[dim-j-1] | Vec len (n-j) | Diagonal | LACGV (before) | LACGV (after) |
|---|-------------|--------------|---------------|----------|----------------|---------------|
| 0 | 511 (bottom)| tau[511]     | 1024          | A(511,1023) | 1024 elem   | 1023 elem     |
| 1 | 510         | tau[510]     | 1023          | A(510,1022) | 1023 elem   | 1022 elem     |
| ... | ...       | ...          | ...           | ...      | ...            | ...           |
| 511 | 0 (top)   | tau[0]       | 513           | A(0,512) | 513 elem    | 512 elem      |

**Kernel Launches Per Iteration:**
1. LACGV (conjugate): 1 kernel (n-j elements)
2. LARFG: 1 kernel
3. SET_DIAG: 1 kernel
4. LARF: 1 kernel (if j < 511)
5. RESTORE_DIAG: 1 kernel
6. LACGV (unconjugate): 1 kernel (n-j-1 elements)

Total per iter: 6 kernels
Total: 512 * 6 = 3,072 kernel launches

**FLOPs Per Iteration j:**

LARFG: ~10 * (1024-j) FLOPs (complex)
```
j=0:   10,240
j=1:   10,230
...
j=511: 5,130
Sum: ~3.9 MFLOPs
```

LARF: 16 * (511-j) * (1024-j) FLOPs (complex GEMV+GER)
```
j=0:   16 * 511 * 1024 = 8,372,224
j=1:   16 * 510 * 1023 = 8,346,480
...
j=510: 16 * 1 * 514 = 8,224
Sum: ≈ 2.14 GFLOPs
```

**Total FLOPs: ~2.14 GFLOPs**

**Memory Traffic:**

LACGV (×2 per iter, asymmetric):
```
Before: Σ(j=0..511) [2*(1024-j)*16] ≈ 8.4 MB
After:  Σ(j=0..511) [2*(1023-j)*16] ≈ 8.4 MB
Total LACGV: 16.8 MB
```

LARFG:
```
Per iter: 2*(1024-j)*16 bytes
Sum: ≈ 8.4 MB
```

LARF:
```
Per iter: ~4*(511-j)*(1024-j)*16 bytes
Sum: ≈ 6.85 GB
```

**Total Memory: ~6.88 GB**

**Comparison with GELQ2 (512×1024 complex):**

| Metric | GERQ2 (512×1024) | GELQ2 (512×1024) | Ratio |
|--------|------------------|------------------|-------|
| Iterations | 512 | 512 | 1.0x |
| Kernel launches | 3,072 | 3,072 | 1.0x |
| FLOPs | 2.14 GFLOP | 2.14 GFLOP | 1.0x |
| Memory | 6.88 GB | 6.88 GB | 1.0x |
| Tau order | Reverse (511→0) | Forward (0→511) | Different |
| Rows processed | 511→0 (bottom-up) | 0→511 (top-down) | Opposite |

**They are IDENTICAL in cost!**

Both process same m×n matrix with:
- Same dim iterations
- Same RIGHT-side LARF (m-j-1 rows × n-j cols)
- Same LARFG (n-j length)
- Same LACGV overhead (complex)

**Only differences:**
1. Row order: GERQ2 bottom-up, GELQ2 top-down
2. Tau indexing: GERQ2 reverse, GELQ2 forward
3. Factor location: GERQ2 R in bottom-right, GELQ2 L in top-left

**Performance Estimate (MI250X):**
```
Kernel launches: 3,072 * 5μs = 15.4 ms
Compute: 2.14 GFLOP @ 200 GFLOPS = 10.7 ms
Memory: 6.88 GB @ 1.6 TB/s = 4.3 ms
Total: ~30 ms (launch-bound)
```

**Arithmetic Intensity:**
```
2.14 GFLOP / 6.88 GB = 0.31 FLOP/byte (memory-bound)
```

**Tau Storage Final State:**
```
tau[0]   = τ for row 0 (top of factored region)
tau[1]   = τ for row 1
...
tau[510] = τ for row 510
tau[511] = τ for row 511 (bottom, first processed)
```

**R Factor Structure (bottom-right):**
```
A = [ 0   0   0   ...  0   0   0  ]
    [ 0   0   0   ...  0   0   0  ]
    ...
    [ 0   0   0   ... R₅₁₁,₅₁₂ R₅₁₁,₅₁₃ ... R₅₁₁,₁₀₂₃ ]
```""",
    rationale="GERQ2 and GELQ2 have identical computational cost for the same m×n matrix: same iterations, kernel launches, FLOPs, and memory traffic. The only differences are row processing order (bottom-up vs top-down), tau indexing (reverse vs forward), and factor placement (R bottom-right vs L top-left). Both are launch-bound with ~3000 kernel launches dominating runtime.",
    tags=["analysis", "algorithm-trace", "gerq2-vs-gelq2", "complex-arithmetic", "performance-analysis", "reverse-tau-storage", "flop-counting"]
))

# L3-2: CODING - Production wrapper
entries.append(create_entry(
    level="L3",
    interface="gerq2",
    instruction="Implement a complete production GERQ2 wrapper including proper argument validation, workspace management, and support for all precision types. Include the shared argument checker used by GERQ2 and GERQF.",
    context_text="GERQ2 must validate arguments, allocate workspace for RIGHT-side LARF, and handle complex types correctly with asymmetric conjugation.",
    code_blocks=[
        {
            "path": "library/src/lapack/roclapack_gerq2.hpp",
            "language": "cpp",
            "content": """template <typename T, typename U>
rocblas_status rocsolver_gerq2_gerqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}"""
        }
    ],
    answer="""Complete production GERQ2 wrapper:

```cpp
// Shared argument checker (used by GERQ2 and GERQF)
template <typename T, typename U>
rocblas_status rocsolver_gerq2_gerqf_argCheck(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              U ipiv,
                                              const rocblas_int batch_count = 1)
{
    // Validation order critical for unit test compatibility

    // 1. Invalid/non-supported values (none for GERQ2/GERQF)
    // N/A

    // 2. Invalid sizes
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // Skip pointer check during workspace query
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. Invalid pointers (only when m,n > 0)
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// Main implementation template
template <typename T, typename U>
rocblas_status rocsolver_gerq2_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    T* ipiv)
{
    // Logging
    ROCSOLVER_ENTER_TOP("gerq2", "-m", m, "-n", n, "--lda", lda);

    // Step 1: Validate handle
    if(!handle)
        return rocblas_status_invalid_handle;

    // Step 2: Argument validation
    rocblas_status st = rocsolver_gerq2_gerqf_argCheck(handle, m, n, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // Step 3: Setup for non-batched execution
    const rocblas_int shiftA = 0;
    const rocblas_stride strideA = 0;
    const rocblas_stride stridep = 0;
    const rocblas_int batch_count = 1;

    // Step 4: Query workspace requirements
    size_t size_scalars, size_work_workArr, size_Abyx_norms, size_diag;
    rocsolver_gerq2_getMemorySize<false, T>(m, n, batch_count,
                                           &size_scalars,
                                           &size_work_workArr,
                                           &size_Abyx_norms,
                                           &size_diag);

    // Step 5: Workspace query mode
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle,
                                                     size_scalars,
                                                     size_work_workArr,
                                                     size_Abyx_norms,
                                                     size_diag);

    // Step 6: Allocate device workspace
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle,
                             size_scalars,
                             size_work_workArr,
                             size_Abyx_norms,
                             size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    // Step 7: Assign workspace pointers
    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];

    // Step 8: Initialize scalar constants
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // Step 9: Execute algorithm
    return rocsolver_gerq2_template<T>(handle,
                                      m, n,
                                      A, shiftA, lda, strideA,
                                      ipiv, stridep,
                                      batch_count,
                                      (T*)scalars,
                                      work_workArr,
                                      (T*)Abyx_norms,
                                      (T*)diag);
}

// C API wrappers
extern "C" {

rocblas_status rocsolver_sgerq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_gerq2_impl<float>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_dgerq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_gerq2_impl<double>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_cgerq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_gerq2_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv);
}

rocblas_status rocsolver_zgerq2(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_gerq2_impl<rocblas_double_complex>(handle, m, n, A, lda, ipiv);
}

} // extern "C"
```

**Key Features:**

1. **Shared validation**: `rocsolver_gerq2_gerqf_argCheck` used by both GERQ2 and GERQF
2. **Type-generic**: Single template for float, double, complex float, complex double
3. **Workspace management**:
   - RIGHT-side LARF (m * sizeof(T) temp)
   - Per-iteration diagonal backup
4. **Complex handling**:
   - COMPLEX template parameter auto-detected
   - Asymmetric conjugation handled in template (n-j before, n-j-1 after)
5. **Error handling order**: invalid_handle → invalid_size → invalid_pointer → memory_error

**Usage Example:**
```cpp
// Setup
rocblas_handle handle;
const int m = 512, n = 1024;
rocblas_double_complex* A;  // device, m×n
rocblas_double_complex* tau;  // device, min(m,n)

// Query workspace
rocsolver_zgerq2(handle, m, n, nullptr, m, nullptr);
size_t ws_size;
rocblas_get_device_memory_size(handle, &ws_size);

// Allocate workspace
rocblas_set_device_memory_size(handle, ws_size);

// Execute RQ factorization
rocsolver_zgerq2(handle, m, n, A, m, tau);

// Result: A contains R (bottom-right) and Householder vectors
//         tau[0..511] in reverse order (tau[0] for row m-dim)
```

**Workspace Breakdown (512×1024 complex FP64):**
```
size_scalars:      24 bytes
size_work_workArr: ~4 KB (pointer array)
size_Abyx_norms:   512 * 16 = 8 KB (m elements for RIGHT-side)
size_diag:         16 bytes (per-iteration)
Total: ~12 KB
```""",
    rationale="Production GERQ2 wrapper handles all precisions via templates, shares validation with GERQF, manages RIGHT-side LARF workspace (m * sizeof(T)), and automatically handles asymmetric complex conjugation. The design separates concerns: shared argument checking, type-aware workspace calculation, and template-based complex handling. Reverse tau storage is handled transparently in the template.",
    tags=["coding", "production-wrapper", "shared-validation", "workspace-management", "complex-handling", "rq-factorization", "api-design"]
))

# Write to JSONL
output_file = Path("kernelgen/dataset/roclapack_gerq2.jsonl")
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Calculate statistics
total = len(entries)
l1 = sum(1 for e in entries if e['level'] == 'L1')
l2 = sum(1 for e in entries if e['level'] == 'L2')
l3 = sum(1 for e in entries if e['level'] == 'L3')
coding = sum(1 for e in entries if 'coding' in e['tags'])
analysis = sum(1 for e in entries if 'analysis' in e['tags'])

print(f"Generated {total} entries")
print(f"Written to: {output_file.absolute()}")
print(f"\nDistribution:")
print(f"  L1: {l1} entries")
print(f"  L2: {l2} entries")
print(f"  L3: {l3} entries")
print(f"  Coding: {coding} ({coding*100//total}%)")
print(f"  Analysis: {analysis} ({analysis*100//total}%)")
