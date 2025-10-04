#!/usr/bin/env python3
import json
import time

# TRTRI: Triangular matrix inverse
entries = []

# Entry 1: Algorithm overview
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L3",
    "interface": "trtri",
    "instruction": "Explain the complete TRTRI algorithm for computing the inverse of a triangular matrix. How does the blocked algorithm work for upper vs lower triangles?",
    "context_text": "TRTRI computes the inverse of a triangular matrix using a blocked recursive algorithm that leverages BLAS-3 operations for efficiency.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "if(uplo == rocblas_fill_upper)\n{\n    for(rocblas_int j = 0; j < n; j += blk)\n    {\n        jb = std::min(n - j, blk);\n\n        // update current block column\n        rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none, diag, j,\n                         jb, &one, 0, A, shiftA, lda, strideA, A, shiftA + idx2D(0, j, lda),\n                         lda, strideA, batch_count);\n\n        rocblasCall_trsm(handle, rocblas_side_right, uplo, rocblas_operation_none, diag, j,\n                         jb, &minone, A, shiftA + idx2D(j, j, lda), lda, strideA, A,\n                         shiftA + idx2D(0, j, lda), lda, strideA, batch_count, optim_mem,\n                         work1, work2, work3, work4);\n\n        trti2<T>(handle, uplo, diag, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,\n                 batch_count, (T*)work1, (T*)work3);\n    }\n}"
        }
    ],
    "answer": "TRTRI triangular matrix inverse algorithm:\n\nGoal: Compute U^(-1) from upper triangular U (or L^(-1) from lower L)\n\nBlocked Algorithm (upper triangle, block size blk):\n\nFor j = 0 to n-1 by blk:\n  Block [j:j+blk, j:j+blk] is current diagonal block\n  \n  Step 1: TRMM - Update off-diagonal block column\n  Compute: A[0:j, j:j+blk] = U[0:j, 0:j] * A[0:j, j:j+blk]\n  - Uses previously inverted blocks U^(-1)[0:j, 0:j]\n  - side=left, op=none, m=j, n=blk\n  \n  Step 2: TRSM - Solve triangular system\n  Solve: U[j:j+blk, j:j+blk] * X = -A[0:j, j:j+blk]\n  Result: A[0:j, j:j+blk] = -U^(-1)[j:j+blk, j:j+blk] * A[0:j, j:j+blk]\n  - side=right, op=none, alpha=-1\n  - Updates off-diagonal using current diagonal block\n  \n  Step 3: TRTI2 - Invert diagonal block\n  Compute: U[j:j+blk, j:j+blk]^(-1) in place\n  - Unblocked algorithm for small diagonal block\n  - Uses TRMV and SCAL operations\n\nLower triangle (process bottom-up):\nFor j = n-1 down to 0 by blk:\n  - TRMM: Update L[j+blk:n, j:j+blk]\n  - TRSM: Solve with L[j:j+blk, j:j+blk]\n  - TRTI2: Invert L[j:j+blk, j:j+blk]\n\nUnblocked TRTI2 (upper, called for diagonal blocks):\nFor j = n-1 down to 0:\n  Step 1: TRMV - Multiply U^(-1)[0:j, 0:j] * U[0:j, j]\n  Step 2: SCAL - Scale by -1/U[j,j]\n  Result: U^(-1)[0:j, j] computed\n  Step 3: Diagonal inversion: U[j,j] = 1/U[j,j]\n\nWhy blocked?\n- BLAS-3 operations (TRMM, TRSM) >> BLAS-2 (TRMV)\n- Better cache reuse and GPU utilization\n- Block size tuned per architecture (typically 64-128)\n\nComplexity: O(n^3/3) flops\n- Same as unblocked but much faster in practice\n- TRMM: O(j*blk^2) per iteration\n- TRSM: O(j*blk^2) per iteration\n- TRTI2: O(blk^3/3) per iteration\n- Total across all blocks: O(n^3/3)",
    "rationale": "TRTRI uses a blocked algorithm that processes the triangular matrix in blocks from top-left to bottom-right (upper) or bottom-right to top-left (lower). Each iteration uses TRMM to update the off-diagonal block column, TRSM to solve for updates, and TRTI2 to invert the diagonal block. This leverages BLAS-3 for efficiency.",
    "tags": ["algorithm", "trtri", "triangular-inverse", "blocked", "trmm", "trsm"]
})
time.sleep(0.001)

# Entry 2: diag parameter
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L1",
    "interface": "trtri",
    "instruction": "What is the diag parameter in TRTRI? How does unit vs non-unit diagonal affect the computation?",
    "context_text": "TRTRI accepts a diag parameter that specifies whether the triangular matrix has unit or non-unit diagonal.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "// check for singularities if non-unit diagonal\nif(diag == rocblas_diagonal_non_unit)\n{\n    ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, 64, 1), 0,\n                            stream, n, A, shiftA, lda, strideA, info);\n}"
        }
    ],
    "answer": "TRTRI diag parameter:\n\nrocblas_diagonal_unit:\n- Matrix has 1's on diagonal (not stored explicitly)\n- Diagonal elements are implicitly 1.0\n- Inverse also has unit diagonal\n- No singularity check needed (always invertible)\n- Diagonal reciprocals: 1/1 = 1 (trivial)\n\nrocblas_diagonal_non_unit:\n- Matrix has arbitrary diagonal values\n- Diagonal elements stored in A[i,i]\n- Must check for zero diagonals (singularity)\n- Requires diagonal inversion: A[i,i] = 1/A[i,i]\n\nDifferences in computation:\n\n1. Singularity checking:\nUnit diagonal: Skip check (always invertible)\nNon-unit: Run check_singularity kernel\n  - Scans A[i,i] for i=0..n-1\n  - Sets info=i+1 if A[i,i]==0\n  - Aborts if singular\n\n2. Diagonal inversion (invdiag kernel):\nUnit diagonal:\n  alphas[i] = -1.0 (for scaling)\n  A[i,i] unchanged (stays 1)\n\nNon-unit diagonal:\n  if A[i,i] != 0:\n    A[i,i] = 1/A[i,i] (invert)\n    alphas[i] = -1/A[i,i] (for scaling)\n\n3. BLAS operations:\nBoth pass diag to TRMV, TRSM, TRMM\n- Tells BLAS whether to use diagonal\n- Unit: ignore stored diagonal, use 1\n- Non-unit: use stored diagonal values\n\nExample:\nUnit diagonal:\nU = [1  2  3]    U^(-1) = [1  -2   1]\n    [0  1  4]              [0   1  -4]\n    [0  0  1]              [0   0   1]\nDiagonal stays 1, only off-diag changes\n\nNon-unit diagonal:\nU = [2  1  0]    U^(-1) = [0.5  -0.25  0]\n    [0  4  2]              [0    0.25  -0.125]\n    [0  0  8]              [0    0     0.125]\nDiagonal: [2,4,8] -> [0.5, 0.25, 0.125]\n\nWhen to use:\nUnit: LU factorization (L has unit diagonal)\nNon-unit: Cholesky (L has non-unit diagonal), general triangular",
    "rationale": "The diag parameter specifies whether the triangular matrix has unit (all 1's) or non-unit diagonal. Unit diagonal skips singularity checks and diagonal inversion since 1/1=1. Non-unit diagonal requires checking for zeros and computing reciprocals A[i,i]=1/A[i,i].",
    "tags": ["trtri", "diag", "unit-diagonal", "non-unit", "parameters"]
})
time.sleep(0.001)

# Entry 3: Workspace calculation
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L2",
    "interface": "trtri",
    "instruction": "How does TRTRI calculate workspace requirements? Explain the dependency on block size and diag parameter.",
    "context_text": "TRTRI workspace depends on block size selection and whether the diagonal is unit or non-unit.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "// get block size\nrocblas_int blk = trtri_get_blksize<ISBATCHED>(n);\n\n// size of temporary array required for copies\nif(diag == rocblas_diagonal_unit && blk > 0)\n    *size_tmpcopy = 0;\nelse\n    *size_tmpcopy = n * n * sizeof(T) * batch_count;\n\nif(blk == 0)\n{\n    // requirements for calling rocBLAS TRTRI\n    rocblasCall_trtri_mem<BATCHED, T>(n, batch_count, size_work1, size_work2);\n}\nelse if(blk == 1)\n{\n    *size_work1 = w1a;  // TRMV workspace\n    *size_work3 = w3a;  // alphas array\n}\nelse\n{\n    rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, nn, blk, 1, 1,\n                                     batch_count, &w1b, size_work2, &w3b, size_work4);\n    *size_work1 = std::max(w1a, w1b);\n    *size_work3 = std::max(w3a, w3b);\n}"
        }
    ],
    "answer": "TRTRI workspace calculation:\n\n1. Block size selection:\nblk = trtri_get_blksize(n)\n- blk=0: Use rocBLAS TRTRI (vendor-optimized)\n- blk=1: Use unblocked TRTI2\n- blk>1: Use blocked algorithm\n- Size intervals tuned per architecture\n\n2. tmpcopy buffer:\nUnit diagonal AND blk>0: size_tmpcopy = 0 (not needed)\nNon-unit OR blk==0: size_tmpcopy = n*n*batch_count\n\nWhy?\n- Non-unit: Need to save A before inversion (if singular)\n- Unit: No singularity check, no save needed\n- blk==0: rocBLAS TRTRI needs temp buffer\n\n3. Work buffers by block size:\n\nblk==0 (rocBLAS TRTRI):\n  work1, work2: From rocblasCall_trtri_mem\n  work3, work4: 0 (not used)\n  Uses vendor TRTRI implementation\n\nblk==1 (unblocked TRTI2):\n  work1: n*batch_count (TRMV workspace)\n  work2: 0 (not used)\n  work3: n*batch_count (alphas array for -1/diag)\n  work4: 0 (not used)\n\nblk>1 (blocked algorithm):\n  Query TRTI2 requirements:\n    w1a: blk*batch_count (TRMV)\n    w3a: blk*batch_count (alphas)\n  \n  Query TRSM requirements:\n    w1b, w2, w3b, w4: From rocblasCall_trsm_mem\n    - side=right, m=n, n=blk\n  \n  Final sizes:\n    work1 = max(w1a, w1b)\n    work2 = w2\n    work3 = max(w3a, w3b)\n    work4 = w4\n\nWhy max()?\n- TRTI2 (diagonal block) and TRSM (off-diagonal) use same buffers\n- Sequential execution allows reuse\n- Must accommodate larger requirement\n\nExample (n=1024, blk=64, non-unit, float, batch=1):\n\nTRTI2 needs:\n  w1a = 64*4 = 256 bytes (TRMV for 64x64 block)\n  w3a = 64*4 = 256 bytes (alphas)\n\nTRSM needs:\n  w1b = 1024*64*4 = 256KB (temp for TRSM)\n  w2 = ...\n  w3b = ...\n  w4 = ...\n\nFinal:\n  work1 = max(256B, 256KB) = 256KB (TRSM dominates)\n  work3 = max(256B, w3b) = larger of two\n\ntmpcopy = 1024*1024*4 = 4MB (save A if singular)",
    "rationale": "TRTRI workspace depends on block size: blk=0 uses rocBLAS (vendor-specific), blk=1 uses unblocked (TRMV+alphas), blk>1 uses blocked (max of TRTI2 and TRSM). The tmpcopy buffer is only needed for non-unit diagonal to save A before inversion in case of singularity.",
    "tags": ["memory", "workspace", "trtri", "block-size", "trsm"]
})
time.sleep(0.001)

# Entry 4: Argument validation
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L1",
    "interface": "trtri",
    "instruction": "What arguments does TRTRI validate? Why must both uplo and diag be validated?",
    "context_text": "TRTRI validates uplo and diag parameters along with sizes and pointers.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "// 1. invalid/non-supported values\nif(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)\n    return rocblas_status_invalid_value;\nif(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)\n    return rocblas_status_invalid_value;\n\n// 2. invalid size\nif(n < 0 || lda < n || batch_count < 0)\n    return rocblas_status_invalid_size;\n\n// 3. invalid pointers\nif((n && !A) || (batch_count && !info))\n    return rocblas_status_invalid_pointer;"
        }
    ],
    "answer": "TRTRI argument validation:\n\n1. uplo validation (triangle specification):\nif(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)\n    return rocblas_status_invalid_value;\n\nValid values:\n- rocblas_fill_upper: Upper triangular matrix\n- rocblas_fill_lower: Lower triangular matrix\n\nInvalid:\n- rocblas_fill_full: Not meaningful for triangular\n- Other values: Undefined behavior\n\n2. diag validation (diagonal type):\nif(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)\n    return rocblas_status_invalid_value;\n\nValid values:\n- rocblas_diagonal_unit: Diagonal is all 1's\n- rocblas_diagonal_non_unit: Diagonal stored in matrix\n\nInvalid: Other values undefined\n\n3. Size validation:\n- n >= 0: Matrix dimension\n- lda >= n: Leading dimension must accommodate matrix\n- batch_count >= 0: Number of matrices\n\n4. Pointer validation:\n- (n && !A): A required when n > 0\n- (batch_count && !info): info required when batch_count > 0\n\nWhy both uplo and diag required?\n\nuplo needed:\n- Determines which triangle to process\n- Algorithm differs for upper vs lower\n- Affects BLAS operation ordering\n\ndiag needed:\n- Controls singularity checking\n- Affects diagonal inversion logic\n- Passed to BLAS operations\n\nCombinations:\nuplo=upper, diag=unit:\n  U = [1  2  3]  Invert upper with unit diagonal\n      [0  1  4]\n      [0  0  1]\n\nuplo=upper, diag=non_unit:\n  U = [2  1  0]  Invert upper with general diagonal\n      [0  3  1]\n      [0  0  4]\n\nuplo=lower, diag=unit:\n  L = [1  0  0]  Invert lower with unit diagonal\n      [2  1  0]\n      [3  4  1]\n\nuplo=lower, diag=non_unit:\n  L = [2  0  0]  Invert lower with general diagonal\n      [1  3  0]\n      [0  1  4]\n\nCommon mistakes:\n- Passing uplo=full for triangular matrix\n- Forgetting diag parameter (required, no default)\n- Mismatching diag with actual matrix structure",
    "rationale": "TRTRI validates uplo (upper/lower triangle) and diag (unit/non-unit diagonal) because both are required to determine the correct algorithm. uplo specifies which triangle to process, while diag controls singularity checking and diagonal inversion. Both must be valid enum values.",
    "tags": ["validation", "trtri", "uplo", "diag", "arguments"]
})
time.sleep(0.001)

# Entry 5: C API
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L3",
    "interface": "trtri",
    "instruction": "Write the C API signatures for all 4 TRTRI precision variants. How do they differ from POTRI's API?",
    "context_text": "TRTRI provides C functions for each precision with an additional diag parameter compared to POTRI.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.cpp",
            "language": "cpp",
            "content": "rocblas_status rocsolver_strtri(rocblas_handle handle,\n                                const rocblas_fill uplo,\n                                const rocblas_diagonal diag,\n                                const rocblas_int n,\n                                float* A,\n                                const rocblas_int lda,\n                                rocblas_int* info)\n{\n    return rocsolver::rocsolver_trtri_impl<float>(handle, uplo, diag, n, A, lda, info);\n}"
        }
    ],
    "answer": "TRTRI C API for all precisions:\n\n1. rocsolver_strtri (float)\nrocblas_status rocsolver_strtri(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    float* A,\n    const rocblas_int lda,\n    rocblas_int* info);\n\n2. rocsolver_dtrtri (double)\nrocblas_status rocsolver_dtrtri(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    double* A,\n    const rocblas_int lda,\n    rocblas_int* info);\n\n3. rocsolver_ctrtri (complex float)\nrocblas_status rocsolver_ctrtri(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    rocblas_float_complex* A,\n    const rocblas_int lda,\n    rocblas_int* info);\n\n4. rocsolver_ztrtri (complex double)\nrocblas_status rocsolver_ztrtri(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    rocblas_double_complex* A,\n    const rocblas_int lda,\n    rocblas_int* info);\n\nComparison with POTRI:\n\nTRTRI parameters:\n  handle, uplo, diag, n, A, lda, info\n\nPOTRI parameters:\n  handle, uplo, n, A, lda, info\n\nKey difference: diag parameter\n- TRTRI: Has diag (unit vs non-unit diagonal)\n- POTRI: No diag (always non-unit from Cholesky)\n\nWhy POTRI doesn't need diag?\n- Cholesky factor always has non-unit diagonal\n- L from A=L*L^T has L[i,i] = sqrt(A[i,i])\n- Never unit diagonal in Cholesky context\n- diag implicitly rocblas_diagonal_non_unit\n\nWhy TRTRI needs diag?\n- General triangular matrices can have either\n- LU factorization: L has unit diagonal\n- Cholesky: L has non-unit diagonal\n- QR factorization: R has non-unit diagonal\n- User must specify which case\n\nCommon parameters:\n- uplo: Both need triangle specification\n- n, lda: Matrix dimensions\n- A: Input/output matrix\n- info: Singularity indicator\n\nUsage examples:\n\nTRTRI (LU factorization L):\nrocsolver_sgetrf(handle, n, n, A, lda, ipiv, info);\n// A now contains L (unit diag) and U\nrocsolver_strtri(handle, rocblas_fill_lower,\n                 rocblas_diagonal_unit, n, A, lda, info);\n// L inverted in place\n\nPOTRI (Cholesky):\nrocsolver_spotrf(handle, rocblas_fill_lower, n, A, lda, info);\n// A now contains L (non-unit diag)\nrocsolver_spotri(handle, rocblas_fill_lower, n, A, lda, info);\n// A now contains A^(-1)",
    "rationale": "TRTRI provides 4 precision variants (strtri, dtrtri, ctrtri, ztrtri) with an additional diag parameter compared to POTRI. The diag parameter specifies unit vs non-unit diagonal, which POTRI doesn't need because Cholesky factors always have non-unit diagonals.",
    "tags": ["api", "trtri", "potri", "diag", "precision"]
})
time.sleep(0.001)

# Entry 6: invdiag kernel details
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L2",
    "interface": "trtri",
    "instruction": "Explain the invdiag kernel in detail. Why does it compute both A[i,i]=1/A[i,i] and alphas[i]=-1/A[i,i]?",
    "context_text": "The invdiag kernel preprocesses diagonal elements before the main TRTRI algorithm.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "if(a[i + i * lda] != 0 && diag == rocblas_diagonal_non_unit)\n{\n    a[i + i * lda] = 1.0 / a[i + i * lda];\n    d[i] = -a[i + i * lda];\n}\nelse\n    d[i] = -1.0;"
        }
    ],
    "answer": "invdiag kernel detailed explanation:\n\nPurpose: Precompute diagonal reciprocals for TRTI2\n\nKernel execution:\n- Parallel across diagonal indices i=0..n-1\n- Per-batch processing (batch dimension in blockIdx.y)\n- Thread i processes diagonal element A[i,i]\n\nFor non-unit diagonal:\nif(A[i,i] != 0):\n  1. A[i,i] = 1 / A[i,i]  (store reciprocal in place)\n  2. alphas[i] = -1 / A[i,i] = -A[i,i]  (store negated reciprocal)\n\nFor unit diagonal:\n  1. A[i,i] stays 1 (unchanged)\n  2. alphas[i] = -1\n\nWhy two values?\n\n1. A[i,i] = 1/A[i,i] (positive reciprocal):\nUsed in: TRMV operations within TRTI2\nFormula: U^(-1)[0:i,0:i] * U[0:i,i]\n- Uses inverted diagonal U^(-1)[i,i] = 1/U[i,i]\n- Positive value needed for matrix multiply\n\n2. alphas[i] = -1/A[i,i] (negative reciprocal):\nUsed in: SCAL operations within TRTI2\nFormula: U^(-1)[0:i,i] = -(1/U[i,i]) * U[0:i,0:i] * U[0:i,i]\n- Negative sign from inversion formula\n- Precomputed to avoid repeated negation\n- Stored in separate array (alphas)\n\nTRTI2 usage (upper triangle, column j):\nStep 1: work = U[0:j,0:j] * U[0:j,j]  (TRMV)\nStep 2: U[0:j,j] = alphas[j] * work   (SCAL with -1/U[j,j])\nStep 3: U[j,j] already contains 1/U[j,j] (from invdiag)\n\nWhy precompute?\n1. Division is expensive (especially on GPU)\n2. Each diagonal used n-i times (i-th diagonal in n-i columns)\n3. Parallel computation across all diagonals\n4. Separate arrays avoid read-after-write hazards\n\nMemory access pattern:\nalphas[batch*n + i] - sequential per batch\n- Coalesced reads in SCAL\n- Good cache locality\n\nExample (n=3, upper):\nInput: U = [2  1  3]\n           [0  4  2]\n           [0  0  5]\n\nAfter invdiag:\nA = [0.5  1    3  ]  (diagonals: 1/2, 1/4, 1/5)\n    [0    0.25 2  ]\n    [0    0    0.2]\n\nalphas = [-0.5, -0.25, -0.2]  (negative reciprocals)\n\nTRTI2 for column 2 (j=2):\n  work = [0.5 0; 0 0.25] * [3; 2] = [1.5; 0.5]\n  U[0:2,2] = -0.2 * [1.5; 0.5] = [-0.3; -0.1]\n  U[2,2] = 0.2 (already set by invdiag)\n\nFinal: U^(-1) = [0.5  -0.3  ...]",
    "rationale": "invdiag computes both positive reciprocals (stored in A[i,i]) for use in TRMV operations and negative reciprocals (stored in alphas[i]) for use in SCAL operations. Precomputing both avoids repeated expensive divisions and provides the correct signs needed in the TRTI2 column update formula.",
    "tags": ["trtri", "invdiag", "kernel", "diagonal", "reciprocal", "trti2"]
})
time.sleep(0.001)

# Entry 7: Block size selection
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L2",
    "interface": "trtri",
    "instruction": "How does TRTRI select block size? What are the three cases (blk=0, blk=1, blk>1) and when is each used?",
    "context_text": "TRTRI uses different algorithms based on the selected block size, which is tuned per matrix dimension and batch configuration.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "rocblas_int blk = trtri_get_blksize<ISBATCHED>(n);\n\nif(blk == 0)\n{\n    // simply use rocblas_trtri\n    rocblasCall_trtri(handle, uplo, diag, n, A, shiftA, lda, strideA, tmpcopy, 0, ldw, strideW,\n                      batch_count, (T*)work1, (T**)work2, workArr);\n}\nelse if(blk == 1)\n{\n    // use the unblocked algorithm\n    trti2<T>(handle, uplo, diag, n, A, shiftA, lda, strideA, batch_count, (T*)work1, (T*)work3);\n}\nelse\n{\n    // use blocked algorithm with block size blk\n    // ... TRMM, TRSM, TRTI2 calls ...\n}"
        }
    ],
    "answer": "TRTRI block size selection:\n\nBlock size determination:\nblk = trtri_get_blksize<ISBATCHED>(n)\n\nLookup tables (architecture-specific):\n- TRTRI_INTERVALS: [n0, n1, n2, ...]\n- TRTRI_BLKSIZES: [blk0, blk1, blk2, ...]\n- TRTRI_BATCH_INTERVALS, TRTRI_BATCH_BLKSIZES for batched\n\nExample intervals:\nIf n <= 64: blk = 0\nIf 64 < n <= 256: blk = 32\nIf 256 < n <= 1024: blk = 64\nIf n > 1024: blk = 128\n\nThree cases:\n\n1. blk = 0 (Use rocBLAS TRTRI):\nWhen: Very small matrices (typically n <= 64)\nAlgorithm: Vendor-optimized rocBLAS routine\nWhy: Vendor libraries highly tuned for small sizes\nWorkspace: From rocblasCall_trtri_mem\nPerformance: Best for tiny matrices\n\n2. blk = 1 (Unblocked TRTI2):\nWhen: Small matrices (64 < n <= 128, architecture-dependent)\nAlgorithm: Column-by-column TRMV + SCAL\nWhy: Blocked overhead not worth it for small n\nWorkspace: TRMV buffer + alphas array\nComplexity: O(n^3/3) with BLAS-2 operations\nPerformance: Simple, good for moderate sizes\n\n3. blk > 1 (Blocked algorithm):\nWhen: Large matrices (n > threshold)\nAlgorithm: Block-recursive with TRMM, TRSM, TRTI2\nWhy: BLAS-3 operations much faster for large n\nWorkspace: Max of TRTI2 and TRSM requirements\nComplexity: O(n^3/3) with BLAS-3 operations\nPerformance: Best for large matrices\n\nSelection rationale:\n\nSmall n (blk=0):\n- rocBLAS TRTRI highly optimized\n- Overhead of our algorithm too high\n- Example: n=32, use vendor routine\n\nMedium n (blk=1):\n- Blocked overhead not justified\n- Unblocked simpler and sufficient\n- BLAS-2 still efficient at this scale\n- Example: n=100, use TRTI2\n\nLarge n (blk>1):\n- Blocked algorithm essential\n- BLAS-3 dominates performance\n- Block size tuned to cache (64-128 typical)\n- Example: n=1024, blk=64\n\nBatch considerations:\n- Batched: Different intervals (favor smaller blocks)\n- Non-batched: Larger blocks for better cache use\n- GPU: Prefer larger blocks (more parallelism)\n\nPerformance (n=1024, float):\nblk=0 (rocBLAS): ~3ms (if supported)\nblk=1 (unblocked): ~50ms (BLAS-2 slow)\nblk=64 (blocked): ~8ms (BLAS-3 fast)\n\nOptimal block size:\n- Balances BLAS-3 efficiency vs overhead\n- Fits in cache (L1: 32-64, L2: 128-256)\n- Tuned via autotuning or heuristics\n- Typically 64 for modern GPUs",
    "rationale": "TRTRI selects block size based on matrix dimension using lookup tables. blk=0 uses vendor rocBLAS (small n), blk=1 uses unblocked TRTI2 (medium n), blk>1 uses blocked algorithm with BLAS-3 (large n). The choice optimizes for the best algorithm at each scale.",
    "tags": ["trtri", "block-size", "algorithm-selection", "performance", "rocblas"]
})
time.sleep(0.001)

# Entry 8: Singularity detection and recovery
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L2",
    "interface": "trtri",
    "instruction": "How does TRTRI detect and handle singular matrices? Explain the save/restore mechanism using tmpcopy.",
    "context_text": "TRTRI must detect zero diagonals and preserve the original matrix when singularity is detected.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "// check for singularities if non-unit diagonal\nif(diag == rocblas_diagonal_non_unit)\n{\n    ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1), dim3(1, 64, 1), 0,\n                            stream, n, A, shiftA, lda, strideA, info);\n}\n\nif(diag == rocblas_diagonal_non_unit && blk > 0)\n{\n    // save copy of A to restore it in cases where info is nonzero\n    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,\n                            stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,\n                            info_mask(info));\n}\n\n// ... inversion algorithm ...\n\nif(diag == rocblas_diagonal_non_unit && blk > 0)\n{\n    // restore A in cases where info is nonzero\n    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(32, 32), 0,\n                            stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, tmpcopy,\n                            info_mask(info));\n}"
        }
    ],
    "answer": "TRTRI singularity detection and recovery:\n\nStep 1: Singularity detection (non-unit diagonal only)\ncheck_singularity kernel:\n- Parallel across batches\n- Each batch scans diagonal A[i,i] for i=0..n-1\n- If any A[i,i] == 0: set info[batch] = i+1 (1-based)\n- First zero diagonal found determines info value\n\nStep 2: Conditional save (before inversion)\nif(diag==non_unit AND blk>0):\n  copy_mat(copymat_to_buffer, A -> tmpcopy,\n           info_mask(info), ...)\n\ninfo_mask(info) creates mask:\n  mask[b] = (info[b] != 0) ? 1 : 0\n  Only copy batches where singular detected\n\nWhy save?\n- Inversion will corrupt A (division by zero, NaN)\n- User expects original A preserved on error\n- LAPACK convention: outputs unchanged on failure\n\nStep 3: Inversion attempt\n- Runs regardless of singularity\n- Singular batches produce garbage (NaN, Inf)\n- Non-singular batches produce valid inverse\n\nStep 4: Conditional restore (after inversion)\nif(diag==non_unit AND blk>0):\n  copy_mat(copymat_from_buffer, tmpcopy -> A,\n           info_mask(info), ...)\n\nSame mask: Only restore singular batches\n\nFinal state:\n- info[b]=0: A[b] contains valid inverse\n- info[b]>0: A[b] contains original matrix (restored)\n\nWhy only for blk>0?\n- blk=0 (rocBLAS): vendor handles singularity internally\n- blk=1 or blk>1: our code must handle it\n\nExample (batch=2, n=3):\n\nBatch 0: Non-singular\nA = [2  1  0]    info[0] = 0\n    [0  3  1]\n    [0  0  4]\n\nBatch 1: Singular (A[1,1]=0)\nA = [2  1  0]    info[1] = 2\n    [0  0  1]\n    [0  0  4]\n\nAfter check_singularity:\n  info = [0, 2]\n\nBefore inversion (save):\n  mask = [0, 1] (only batch 1)\n  tmpcopy[1] = A[1] (save batch 1)\n  A[0] not saved (info[0]=0)\n\nAfter inversion:\n  A[0] = inv(A[0]) (valid)\n  A[1] = garbage (NaN from 1/0)\n\nAfter restore:\n  A[0] unchanged (inverse)\n  A[1] = tmpcopy[1] (original restored)\n\nUser sees:\n  A[0]: Inverse (success)\n  A[1]: Original (failure)\n  info[1]=2: Indicates which diagonal failed\n\nMemory cost:\n- Worst case: All singular -> copy all\n- Best case: All non-singular -> no copy (mask=0)\n- Typical: Few singular -> minimal copy overhead",
    "rationale": "TRTRI detects singularity via check_singularity kernel (scans for zero diagonals), then conditionally saves singular batches to tmpcopy before inversion. After inversion, it conditionally restores only the singular batches from tmpcopy, preserving original matrices where info>0 while keeping valid inverses where info=0.",
    "tags": ["trtri", "singularity", "error-handling", "info-mask", "save-restore"]
})
time.sleep(0.001)

# Entry 9: TRTI2 unblocked algorithm
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L3",
    "interface": "trtri",
    "instruction": "Explain the TRTI2 unblocked algorithm in detail. How does it compute triangular inverse column by column?",
    "context_text": "TRTI2 is the unblocked algorithm used for small matrices or diagonal blocks in the blocked algorithm.",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_trtri.hpp",
            "language": "cpp",
            "content": "if(uplo == rocblas_fill_upper)\n{\n    for(rocblas_int j = 1; j < n; ++j)\n    {\n        rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, j, A, shiftA, lda,\n                            strideA, A, shiftA + idx2D(0, j, lda), 1, strideA, work, stdw,\n                            batch_count);\n\n        rocblasCall_scal<T>(handle, j, alphas + j, stdw, A, shiftA + idx2D(0, j, lda), 1,\n                            strideA, batch_count);\n    }\n}\nelse //rocblas_fill_lower\n{\n    for(rocblas_int j = n - 2; j >= 0; --j)\n    {\n        rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, diag, n - j - 1, A,\n                            shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,\n                            shiftA + idx2D(j + 1, j, lda), 1, strideA, work, stdw, batch_count);\n\n        rocblasCall_scal<T>(handle, n - j - 1, alphas + j, stdw, A,\n                            shiftA + idx2D(j + 1, j, lda), 1, strideA, batch_count);\n    }\n}"
        }
    ],
    "answer": "TRTI2 unblocked triangular inverse algorithm:\n\nGoal: Compute U^(-1) column by column (or L^(-1) for lower)\n\nPrerequisite: invdiag kernel already ran\n- A[i,i] contains 1/original_A[i,i]\n- alphas[i] contains -1/original_A[i,i]\n\nUpper triangle algorithm:\nFor j = 1 to n-1:\n  Compute column j of U^(-1)\n  \n  Step 1: TRMV (triangular matrix-vector multiply)\n  work = U^(-1)[0:j, 0:j] * U[0:j, j]\n  \n  Breakdown:\n  - U^(-1)[0:j, 0:j]: Already inverted (columns 0..j-1)\n  - U[0:j, j]: Original column j (above diagonal)\n  - work: Temporary result vector\n  \n  Step 2: SCAL (vector scaling)\n  U^(-1)[0:j, j] = alphas[j] * work\n                 = (-1/U[j,j]) * work\n  \n  Result: U^(-1)[0:j, j] computed\n  Diagonal: U[j,j] already contains 1/original_U[j,j]\n\nLower triangle algorithm:\nFor j = n-2 down to 0:\n  Compute column j of L^(-1)\n  \n  Step 1: TRMV\n  work = L^(-1)[j+1:n, j+1:n] * L[j+1:n, j]\n  \n  Step 2: SCAL\n  L^(-1)[j+1:n, j] = alphas[j] * work\n                    = (-1/L[j,j]) * work\n\nMathematical derivation:\n\nFor upper triangle, column j:\nU * U^(-1) = I\nU[0:j+1, 0:j+1] * U^(-1)[0:j+1, j] = e_j (j-th unit vector)\n\nExpanding:\n[U[0:j,0:j]  U[0:j,j]  ] * [U^(-1)[0:j,j]] = [0]\n[   0        U[j,j]    ]   [U^(-1)[j,j]  ]   [1]\n\nFrom second equation:\nU[j,j] * U^(-1)[j,j] = 1\nU^(-1)[j,j] = 1/U[j,j]  (already set by invdiag)\n\nFrom first equation:\nU[0:j,0:j] * U^(-1)[0:j,j] + U[0:j,j] * U^(-1)[j,j] = 0\nU^(-1)[0:j,0:j] * U^(-1)[0:j,j] + U[0:j,j] * (1/U[j,j]) = 0\nU^(-1)[0:j,j] = -(1/U[j,j]) * U^(-1)[0:j,0:j] * U[0:j,j]\n               = alphas[j] * (U^(-1)[0:j,0:j] * U[0:j,j])\n\nThis is exactly SCAL(alphas[j], TRMV(...))\n\nExample (n=3, upper):\nInput:\nU = [2  1  3]    (after invdiag: diag = [0.5, 0.25, 0.2])\n    [0  4  2]     alphas = [-0.5, -0.25, -0.2]\n    [0  0  5]\n\nColumn 0: U^(-1)[0,0] = 0.5 (already set)\n\nColumn 1 (j=1):\n  TRMV: work[0] = U^(-1)[0,0] * U[0,1]\n                = 0.5 * 1 = 0.5\n  SCAL: U^(-1)[0,1] = alphas[1] * work[0]\n                    = -0.25 * 0.5 = -0.125\n  Diagonal: U^(-1)[1,1] = 0.25 (already set)\n\nColumn 2 (j=2):\n  TRMV: work[0:2] = [U^(-1)[0,0]  U^(-1)[0,1]] * [U[0,2]]\n                                 [0  U^(-1)[1,1]]   [U[1,2]]\n                  = [0.5  -0.125] * [3]\n                    [0    0.25  ]   [2]\n                  = [0.5*3 + -0.125*2]\n                    [0.25*2         ]\n                  = [1.25]\n                    [0.5 ]\n  SCAL: U^(-1)[0:2,2] = alphas[2] * work\n                      = -0.2 * [1.25, 0.5]\n                      = [-0.25, -0.1]\n  Diagonal: U^(-1)[2,2] = 0.2\n\nFinal:\nU^(-1) = [0.5  -0.125  -0.25]\n         [0    0.25   -0.1 ]\n         [0    0      0.2  ]\n\nComplexity:\n- Column j: O(j^2) for TRMV + O(j) for SCAL\n- Total: sum_{j=1}^{n-1} O(j^2) = O(n^3/3)\n- BLAS-2 operations (TRMV, SCAL) not as efficient as BLAS-3",
    "rationale": "TRTI2 computes triangular inverse column by column using TRMV to multiply previously inverted columns by the original column, then SCAL to apply -1/diagonal. The formula U^(-1)[0:j,j] = -(1/U[j,j]) * U^(-1)[0:j,0:j] * U[0:j,j] is implemented via TRMV+SCAL with precomputed alphas.",
    "tags": ["algorithm", "trti2", "trtri", "unblocked", "trmv", "scal"]
})
time.sleep(0.001)

# Entry 10: Upper vs lower differences
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L1",
    "interface": "trtri",
    "instruction": "How does the TRTRI algorithm differ for upper vs lower triangles? Why does upper process forward and lower backward?",
    "context_text": "TRTRI processes upper triangles from top-left to bottom-right and lower triangles from bottom-right to top-left.",
    "code_blocks": [],
    "answer": "TRTRI upper vs lower triangle processing:\n\nUpper triangle (process forward, j=0 to n-1):\n- Start at top-left corner\n- Process columns left to right\n- Each column depends on columns to its left\n- Column j uses columns 0..j-1 (already computed)\n\nLower triangle (process backward, j=n-1 to 0):\n- Start at bottom-right corner\n- Process columns right to left\n- Each column depends on columns to its right\n- Column j uses columns j+1..n-1 (already computed)\n\nWhy different directions?\n\nUpper triangle dependency:\nU^(-1)[0:j, j] = -(1/U[j,j]) * U^(-1)[0:j, 0:j] * U[0:j, j]\n- Needs U^(-1)[0:j, 0:j] (columns left of j)\n- Must compute left columns first\n- Forward sweep: j=0, 1, 2, ..., n-1\n\nLower triangle dependency:\nL^(-1)[j+1:n, j] = -(1/L[j,j]) * L^(-1)[j+1:n, j+1:n] * L[j+1:n, j]\n- Needs L^(-1)[j+1:n, j+1:n] (columns right of j)\n- Must compute right columns first\n- Backward sweep: j=n-1, n-2, ..., 0\n\nVisual example (n=4):\n\nUpper triangle order:\nStep 1: [x  ?  ?  ?]    Compute column 0 (diagonal only)\n        [0  x  ?  ?]\n        [0  0  x  ?]\n        [0  0  0  x]\n\nStep 2: [x  x  ?  ?]    Compute column 1 (uses column 0)\n        [0  x  ?  ?]\n        [0  0  x  ?]\n        [0  0  0  x]\n\nStep 3: [x  x  x  ?]    Compute column 2 (uses columns 0,1)\n        [0  x  x  ?]\n        [0  0  x  ?]\n        [0  0  0  x]\n\nStep 4: [x  x  x  x]    Compute column 3 (uses columns 0,1,2)\n        [0  x  x  x]\n        [0  0  x  x]\n        [0  0  0  x]\n\nLower triangle order:\nStep 1: [x  0  0  0]    Compute column 3 (diagonal only)\n        [?  x  0  0]\n        [?  ?  x  0]\n        [?  ?  ?  x]\n\nStep 2: [x  0  0  0]    Compute column 2 (uses column 3)\n        [?  x  0  0]\n        [?  x  x  0]\n        [?  ?  x  x]\n\nStep 3: [x  0  0  0]    Compute column 1 (uses columns 2,3)\n        [x  x  0  0]\n        [x  x  x  0]\n        [x  ?  x  x]\n\nStep 4: [x  0  0  0]    Compute column 0 (uses columns 1,2,3)\n        [x  x  0  0]\n        [x  x  x  0]\n        [x  x  x  x]\n\nBlocked algorithm follows same pattern:\nUpper: Process block columns left to right\nLower: Process block columns right to left\n\nCode structure:\nUpper: for(j=0; j<n; j+=blk) {...}\nLower: for(j=n-blk; j>=0; j-=blk) {...}\n\nSummary:\n- Upper: Forward (left to right) - columns depend on left\n- Lower: Backward (right to left) - columns depend on right\n- Both achieve same dependency resolution\n- Direction determined by matrix structure",
    "rationale": "TRTRI processes upper triangles forward (left to right) because each column depends on previously computed columns to its left. Lower triangles are processed backward (right to left) because each column depends on columns to its right. The direction ensures dependencies are satisfied in both cases.",
    "tags": ["trtri", "upper-lower", "algorithm", "dependencies"]
})
time.sleep(0.001)

# Entry 11: Comparison with POTRI usage
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L2",
    "interface": "trtri",
    "instruction": "How is TRTRI used by POTRI? What parameters does POTRI pass to TRTRI?",
    "context_text": "POTRI calls TRTRI as its first step to invert the Cholesky factor.",
    "code_blocks": [],
    "answer": "TRTRI usage by POTRI:\n\nPOTRI algorithm:\nStep 1: TRTRI - Invert Cholesky factor\nStep 2: TRMM - Compute symmetric product\n\nTRTRI call from POTRI:\nrocsolver_trtri_template<BATCHED, STRIDED, T>(\n    handle,\n    uplo,                          // Same as POTRI uplo\n    rocblas_diagonal_non_unit,     // Always non-unit for Cholesky\n    n, A, shiftA, lda, strideA,\n    info, batch_count,\n    work1, work2, work3, work4,\n    tmpcopy, workArr, optim_mem);\n\nKey parameters:\n\n1. uplo: Passed from POTRI\n- rocblas_fill_lower: L from A=L*L^T\n- rocblas_fill_upper: U from A=U^T*U\n- POTRI and TRTRI use same triangle\n\n2. diag: Always rocblas_diagonal_non_unit\n- Cholesky factors have non-unit diagonal\n- L[i,i] = sqrt(A[i,i]) != 1\n- Never unit diagonal in Cholesky context\n- POTRI hardcodes this value\n\n3. Shared workspace:\n- work1-4: Reused by POTRI for TRMM\n- tmpcopy: Reused for TRMM B matrix\n- optim_mem: Optimization flag\n\n4. info: Singularity detection\n- TRTRI sets info if diagonal zero\n- POTRI checks and aborts if info > 0\n- Preserves A when singular\n\nWorkflow:\n\nInput to POTRI:\nA contains L (lower) or U (upper) from POTRF\n\nAfter TRTRI:\nA contains L^(-1) (lower) or U^(-1) (upper)\ninfo indicates if singular\n\nPOTRI check:\nif(info > 0):\n  // Singular Cholesky factor\n  // POTRI aborts, A preserved\n  return\n\nPOTRI continues:\nTRMM computes L^(-1)^T * L^(-1) or U^(-1) * U^(-1)^T\nResult: A^(-1)\n\nExample (lower triangle):\n\nInput to POTRI:\nA = [2  0  0]    (L from Cholesky)\n    [1  1  0]\n    [3  2  1]\n\nAfter TRTRI (inverts L):\nA = [0.5    0     0  ]    (L^(-1))\n    [-0.5   1     0  ]\n    [-1    -2     1  ]\n\nAfter TRMM (L^(-1)^T * L^(-1)):\nA = [0.5   -0.5  -1  ]^T * [0.5    0     0  ]\n    [0      1    -2  ]     [-0.5   1     0  ]\n    [0      0     1  ]     [-1    -2     1  ]\n\n  = [... final A^(-1) ...]\n\nWhy POTRI doesn't have diag parameter?\n- Always uses non-unit diagonal\n- Cholesky context fixes this\n- No user choice needed\n- Simpler API\n\nTRTRI flexibility:\n- Can handle unit or non-unit\n- Used by POTRI (non-unit)\n- Used by GETRI (unit for L, non-unit for U)\n- General-purpose triangular inverse",
    "rationale": "POTRI calls TRTRI with diag=rocblas_diagonal_non_unit because Cholesky factors always have non-unit diagonals (L[i,i]=sqrt(A[i,i])). POTRI passes its uplo parameter to TRTRI and reuses the same workspace. After TRTRI inverts the factor, POTRI uses TRMM to compute the symmetric product for the final inverse.",
    "tags": ["trtri", "potri", "cholesky", "usage", "integration"]
})
time.sleep(0.001)

# Entry 12: Implement simplified TRTRI (coding)
entries.append({
    "id": str(int(time.time() * 1000)),
    "level": "L3",
    "interface": "trtri",
    "instruction": "Write a simplified TRTRI implementation using the unblocked TRTI2 algorithm. Include invdiag kernel and singularity checking.",
    "context_text": "Implement a basic TRTRI that uses only the unblocked column-by-column algorithm.",
    "code_blocks": [],
    "answer": "Simplified TRTRI implementation (unblocked):\n\n```cpp\n// File: rocsolver_trtri_simple.hpp\n#pragma once\n\n#include \"rocblas.hpp\"\n\ntemplate <typename T>\nROCSOLVER_KERNEL void invdiag_simple(\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    T* A,\n    const rocblas_int lda,\n    T* alphas)\n{\n    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    \n    if(i < n)\n    {\n        if(A[i + i*lda] != 0 && diag == rocblas_diagonal_non_unit)\n        {\n            A[i + i*lda] = 1.0 / A[i + i*lda];\n            alphas[i] = -A[i + i*lda];\n        }\n        else\n            alphas[i] = -1.0;\n    }\n}\n\ntemplate <typename T>\nROCSOLVER_KERNEL void check_singularity_simple(\n    const rocblas_int n,\n    T* A,\n    const rocblas_int lda,\n    rocblas_int* info)\n{\n    for(int i = 0; i < n; i++)\n    {\n        if(A[i + i*lda] == 0)\n        {\n            *info = i + 1;  // 1-based index\n            return;\n        }\n    }\n    *info = 0;\n}\n\ntemplate <typename T>\nrocblas_status rocsolver_trtri_simple(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    T* A,\n    const rocblas_int lda,\n    rocblas_int* info)\n{\n    // 1. Argument validation\n    if(!handle)\n        return rocblas_status_invalid_handle;\n    \n    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)\n        return rocblas_status_invalid_value;\n    if(diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit)\n        return rocblas_status_invalid_value;\n    \n    if(n < 0 || lda < n)\n        return rocblas_status_invalid_size;\n    \n    if((n && !A) || !info)\n        return rocblas_status_invalid_pointer;\n    \n    if(n == 0) {\n        *info = 0;\n        return rocblas_status_success;\n    }\n    \n    hipStream_t stream;\n    rocblas_get_stream(handle, &stream);\n    \n    // 2. Allocate workspace\n    T* work;  // TRMV workspace\n    T* alphas;  // Diagonal reciprocals\n    \n    hipMalloc(&work, n * sizeof(T));\n    hipMalloc(&alphas, n * sizeof(T));\n    \n    // 3. Check singularity (non-unit diagonal only)\n    if(diag == rocblas_diagonal_non_unit)\n    {\n        ROCSOLVER_LAUNCH_KERNEL(\n            (check_singularity_simple<T>),\n            dim3(1), dim3(1), 0, stream,\n            n, A, lda, info);\n        \n        // Check if singular\n        rocblas_int h_info;\n        hipMemcpy(&h_info, info, sizeof(rocblas_int),\n                  hipMemcpyDeviceToHost);\n        if(h_info > 0) {\n            hipFree(work);\n            hipFree(alphas);\n            return rocblas_status_success;  // Singular, abort\n        }\n    }\n    else {\n        hipMemset(info, 0, sizeof(rocblas_int));\n    }\n    \n    // 4. Invert diagonal and compute alphas\n    rocblas_int blocks = (n - 1) / 32 + 1;\n    ROCSOLVER_LAUNCH_KERNEL(\n        (invdiag_simple<T>),\n        dim3(blocks), dim3(32), 0, stream,\n        diag, n, A, lda, alphas);\n    \n    // 5. Set pointer mode for BLAS calls\n    rocblas_pointer_mode old_mode;\n    rocblas_get_pointer_mode(handle, &old_mode);\n    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);\n    \n    // 6. Unblocked inversion (TRTI2)\n    if(uplo == rocblas_fill_upper)\n    {\n        // Forward sweep: j = 1 to n-1\n        for(rocblas_int j = 1; j < n; j++)\n        {\n            // work = U^(-1)[0:j, 0:j] * U[0:j, j]\n            rocblasCall_trmv<T>(\n                handle, uplo, rocblas_operation_none, diag,\n                j,  // Size of upper-left block\n                A, 0, lda, 0,  // Matrix U^(-1)\n                A, idx2D(0, j, lda), 1, 0,  // Column j\n                work, 0, 1);\n            \n            // U^(-1)[0:j, j] = alphas[j] * work\n            rocblasCall_scal<T>(\n                handle, j,\n                alphas + j, 0,  // Scale by -1/U[j,j]\n                A, idx2D(0, j, lda), 1, 0, 1);\n        }\n    }\n    else  // rocblas_fill_lower\n    {\n        // Backward sweep: j = n-2 down to 0\n        for(rocblas_int j = n - 2; j >= 0; j--)\n        {\n            rocblas_int m = n - j - 1;  // Submatrix size\n            \n            // work = L^(-1)[j+1:n, j+1:n] * L[j+1:n, j]\n            rocblasCall_trmv<T>(\n                handle, uplo, rocblas_operation_none, diag,\n                m,\n                A, idx2D(j+1, j+1, lda), lda, 0,\n                A, idx2D(j+1, j, lda), 1, 0,\n                work, 0, 1);\n            \n            // L^(-1)[j+1:n, j] = alphas[j] * work\n            rocblasCall_scal<T>(\n                handle, m,\n                alphas + j, 0,\n                A, idx2D(j+1, j, lda), 1, 0, 1);\n        }\n    }\n    \n    rocblas_set_pointer_mode(handle, old_mode);\n    \n    // 7. Cleanup\n    hipFree(work);\n    hipFree(alphas);\n    \n    return rocblas_status_success;\n}\n\n// C API wrapper\nextern \"C\" rocblas_status rocsolver_strtri_simple(\n    rocblas_handle handle,\n    const rocblas_fill uplo,\n    const rocblas_diagonal diag,\n    const rocblas_int n,\n    float* A,\n    const rocblas_int lda,\n    rocblas_int* info)\n{\n    return rocsolver_trtri_simple<float>(\n        handle, uplo, diag, n, A, lda, info);\n}\n```\n\nUsage:\n```cpp\nfloat U[9] = {2,0,0, 1,3,0, 4,2,5};  // Upper triangular\nrocblas_int info;\n\nrocsolver_strtri_simple(handle,\n    rocblas_fill_upper,\n    rocblas_diagonal_non_unit,\n    3, U, 3, &info);\n\nif(info == 0)\n    printf(\"U inverted successfully\\n\");\nelse\n    printf(\"Singular at position %d\\n\", info);\n```\n\nLimitations:\n- Unblocked only (BLAS-2, slower for large n)\n- No save/restore (A corrupted if singular detected late)\n- Simple memory management (no optimal workspace)\n- Non-batched only\n\nBut demonstrates core algorithm clearly!",
    "rationale": "Simplified TRTRI implements the unblocked TRTI2 algorithm with invdiag preprocessing and singularity checking. It uses TRMV to multiply inverted columns by original columns, then SCAL to apply -1/diagonal. The implementation is simpler than production code but demonstrates the core triangular inversion logic.",
    "tags": ["coding", "trtri", "trti2", "implementation", "unblocked"]
})
time.sleep(0.001)

# Output
print(f"Generated {len(entries)} entries")
with open('/root/rocSOLVER/kernelgen/dataset/roclapack_trtri.jsonl', 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

coding_count = sum(1 for e in entries if 'coding' in e['tags'] or 'implementation' in e['tags'])
l1 = sum(1 for e in entries if e['level'] == 'L1')
l2 = sum(1 for e in entries if e['level'] == 'L2')
l3 = sum(1 for e in entries if e['level'] == 'L3')

print(f"Output: /root/rocSOLVER/kernelgen/dataset/roclapack_trtri.jsonl")
print(f"Coding tasks: {coding_count}/{len(entries)} ({coding_count/len(entries)*100:.1f}%)")
print(f"L1: {l1}, L2: {l2}, L3: {l3}")
