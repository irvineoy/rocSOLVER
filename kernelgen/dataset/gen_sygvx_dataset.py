#!/usr/bin/env python3
import json
import time

def generate_timestamp_id():
    return str(int(time.time() * 1000000))

entries = []

# Entry 1 (L2): SYGVX overview - selective eigenvalue computation
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What is rocsolver_sygvx and how does the erange parameter enable selective eigenvalue computation?",
    "answer": "rocsolver_sygvx solves the generalized symmetric eigenvalue problem A*x = λ*B*x with selective computation of eigenvalues and eigenvectors. The erange parameter controls which eigenvalues to compute: rocblas_erange_all computes all n eigenvalues, rocblas_erange_value computes eigenvalues in the interval (vl, vu], and rocblas_erange_index computes eigenvalues with indices il through iu (1-indexed). The algorithm performs POTRF on B, SYGST to reduce to standard form, then SYEVX (which uses STEBZ for bisection and STEIN for eigenvectors) to selectively compute eigenvalues, and finally back-transforms eigenvectors. The nev output parameter reports how many eigenvalues were found, which can vary for erange_value. Unlike SYGVD which always computes all eigenvalues, SYGVX is more efficient when only a subset is needed.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvx_hegvx_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevx_heevx_template<BATCHED, STRIDED, T>(\n        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, abstol, nev, W,\n        strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, iinfo, batch_count, scalars, work1, work2,\n        work3, work4, work5, work6, D, E, iblock, isplit, tau, (T**)work7_workArr);"
        }
    ]
})

time.sleep(0.001)

# Entry 2 (L1): erange parameter validation
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What validation checks are performed on the erange, vl, vu, il, and iu parameters in rocsolver_sygvx_hegvx_argCheck?",
    "answer": "The argCheck function validates: (1) erange must be rocblas_erange_all, rocblas_erange_value, or rocblas_erange_index. (2) For erange_value: vl < vu is required, otherwise returns rocblas_status_invalid_size. (3) For erange_index: il >= 1 and iu >= 0, and if n > 0 then il <= iu and iu <= n, otherwise returns rocblas_status_invalid_size. These checks ensure the interval (vl,vu] is valid for value-based selection, and the index range [il,iu] is valid and within [1,n] for index-based selection. The validation is ordered specifically for unit tests and occurs before pointer checks.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvx_hegvx_argCheck",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    if(erange == rocblas_erange_value && vl >= vu)\n        return rocblas_status_invalid_size;\n    if(erange == rocblas_erange_index && (il < 1 || iu < 0))\n        return rocblas_status_invalid_size;\n    if(erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))\n        return rocblas_status_invalid_size;"
        }
    ]
})

time.sleep(0.001)

# Entry 3 (L2): sygvx_update_info kernel - nev handling
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How does the sygvx_update_info kernel differ from sygv_update_info, and why does it take the nev parameter?",
    "answer": "The sygvx_update_info kernel extends sygv_update_info by handling the variable number of eigenvalues returned by SYEVX. It takes three arrays: info (POTRF errors), iinfo (SYEVX errors), and nev (number of eigenvalues found). If POTRF failed (info[b] != 0), it sets info[b] += n and nev[b] = 0 to indicate no eigenvalues are available. If POTRF succeeded, it copies iinfo[b] to info[b]. The nev parameter is crucial because with erange_value, SYEVX may find fewer eigenvalues than n, and when POTRF fails, nev must be zeroed to prevent the caller from using uninitialized eigenvalue data. This differs from sygv_update_info which doesn't modify nev since SYGVD always attempts all n eigenvalues.",
    "difficulty": "L2",
    "interface": "sygvx_update_info",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "template <typename T>\nROCSOLVER_KERNEL void\n    sygvx_update_info(T* info, T* iinfo, T* nev, const rocblas_int n, const rocblas_int bc)\n{\n    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n\n    if(b < bc)\n    {\n        if(info[b] != 0)\n        {\n            info[b] += n;\n            nev[b] = 0;\n        }\n        else\n            info[b] = iinfo[b];\n    }\n}"
        }
    ]
})

time.sleep(0.001)

# Entry 4 (L1): Z matrix vs A matrix for eigenvectors
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "Why does rocsolver_sygvx use a separate Z matrix for eigenvectors instead of overwriting A like rocsolver_sygvd?",
    "answer": "SYGVX uses a separate Z matrix because selective eigenvalue computation may return fewer than n eigenvectors. When erange is rocblas_erange_value or rocblas_erange_index, only nev eigenvectors are computed (where nev <= n). Storing these in the n×n matrix A would waste space and complicate indexing. The Z matrix has dimensions n×ldz but only the first nev columns contain valid eigenvectors. Additionally, SYEVX (the underlying standard eigenvalue solver) naturally produces eigenvectors in a separate output matrix rather than in-place. The back-transformation (TRSM or TRMM) operates on Z, transforming the selected eigenvectors from the standard form back to the original generalized problem.",
    "difficulty": "L1",
    "interface": "rocsolver_ssygvx",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.cpp",
            "language": "cpp",
            "content": "rocblas_status rocsolver_ssyevx(rocblas_handle handle,\n                                const rocblas_evect evect,\n                                const rocblas_erange erange,\n                                const rocblas_fill uplo,\n                                const rocblas_int n,\n                                float* A,\n                                const rocblas_int lda,\n                                const float vl,\n                                const float vu,\n                                const rocblas_int il,\n                                const rocblas_int iu,\n                                const float abstol,\n                                rocblas_int* nev,\n                                float* W,\n                                float* Z,\n                                const rocblas_int ldz,\n                                rocblas_int* ifail,\n                                rocblas_int* info)"
        }
    ]
})

time.sleep(0.001)

# Entry 5 (L2): h_nev calculation for back-transformation
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How is h_nev calculated for the back-transformation in rocsolver_sygvx, and what does it represent?",
    "answer": "In the back-transformation phase (line 322), h_nev is calculated as: h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n). For erange_index, h_nev equals the count of requested eigenvalues (iu - il + 1) since the exact number is known in advance. For erange_all and erange_value, h_nev defaults to n because the implementation currently cannot efficiently handle variable nev values (see TODO comment). The h_nev value determines how many columns of Z are processed during TRSM or TRMM back-transformation. Ideally for erange_value, h_nev should equal the actual nev found, but the TODO at line 315 indicates this optimization is not yet implemented.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvx_hegvx_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    // backtransform eigenvectors\n    if(evect == rocblas_evect_original)\n    {\n        rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);"
        }
    ]
})

time.sleep(0.001)

# Entry 6 (L1): TODO about B not positive definite
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What does the TODO comment at line 296 in roclapack_sygvx_hegvx.hpp indicate about the current behavior when B is not positive definite?",
    "answer": "The TODO comment states that when POTRF fails because B is not positive definite, the algorithm should immediately stop and not modify matrix A, since no valid eigenvalues or eigenvectors can be computed from the generalized problem. However, the current implementation continues processing and destroys matrix A. The comment acknowledges 'Need to find a way to do this efficiently' - implementing early termination in a batched GPU setting is non-trivial because different batches may succeed or fail independently. The workaround is that sygvx_update_info sets nev=0 when POTRF fails, signaling to the caller that no eigenvalues are available, but A remains corrupted.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvx_hegvx_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    /** (TODO: Strictly speaking, computations should stop here if B is not positive definite.\n        A should not be modified in this case as no eigenvalues or eigenvectors can be computed.\n        Need to find a way to do this efficiently; for now A will be destroyed in the non\n        positive-definite case) **/"
        }
    ]
})

time.sleep(0.001)

# Entry 7 (L2): TODO about h_nev inefficiency
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What inefficiency is described in the TODO comment at line 315 regarding h_nev in the back-transformation phase?",
    "answer": "The TODO comment points out that when fewer than n eigenvalues are returned (particularly for erange_value where nev is variable), the back-transformation should only process those nev columns of Z rather than all n columns. Currently, for erange_value and erange_all, h_nev is set to n, causing TRSM or TRMM to unnecessarily process n columns even if nev < n. The comment states 'TRSM or TRMM below should not work with the entire matrix. Need to find a way to do this efficiently; for now we ignore nev and set h_nev = n.' This wastes computation and memory bandwidth. For erange_index, h_nev is correctly set to iu-il+1, but this doesn't help erange_value where the count is unknown until SYEVX completes.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvx_hegvx_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    /** (TODO: Similarly, if only h_nev < n eigenvalues were returned, TRSM or TRMM below should not\n            work with the entire matrix. Need to find a way to do this efficiently; for now we ignore\n            nev and set h_nev = n) **/\n\n    // backtransform eigenvectors\n    if(evect == rocblas_evect_original)\n    {\n        rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);"
        }
    ]
})

time.sleep(0.001)

# Entry 8 (L3): Complete usage example with erange_value
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "Write a complete HIP code example demonstrating rocsolver_dsygvx with erange_value to compute eigenvalues in a specific interval, including proper handling of the variable nev output and ifail array.",
    "answer": "Here is a complete example showing SYGVX with value-based selection:\n\n```cpp\n#include <hip/hip_runtime.h>\n#include <rocsolver/rocsolver.h>\n#include <iostream>\n#include <vector>\n\nint main() {\n    rocblas_handle handle;\n    rocblas_create_handle(&handle);\n    \n    const rocblas_int n = 1024;\n    const rocblas_int lda = n, ldb = n, ldz = n;\n    \n    // Allocate host memory\n    std::vector<double> hA(n * n);\n    std::vector<double> hB(n * n);\n    std::vector<double> hW(n);  // Allocate for all n, but only nev will be filled\n    std::vector<double> hZ(n * n);\n    std::vector<rocblas_int> hIfail(n);\n    \n    // Initialize symmetric A and positive definite B\n    // ... (initialization code)\n    \n    // Allocate device memory\n    double *dA, *dB, *dW, *dZ;\n    rocblas_int *dIfail;\n    hipMalloc(&dA, sizeof(double) * n * n);\n    hipMalloc(&dB, sizeof(double) * n * n);\n    hipMalloc(&dW, sizeof(double) * n);\n    hipMalloc(&dZ, sizeof(double) * n * n);\n    hipMalloc(&dIfail, sizeof(rocblas_int) * n);\n    \n    hipMemcpy(dA, hA.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);\n    hipMemcpy(dB, hB.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);\n    \n    // Select eigenvalues in interval (0.5, 2.5]\n    const double vl = 0.5;\n    const double vu = 2.5;\n    const double abstol = 1e-10;  // Tolerance for STEBZ bisection\n    rocblas_int nev;  // Output: number of eigenvalues found\n    rocblas_int info;\n    \n    // Solve generalized eigenvalue problem with value-based selection\n    rocsolver_dsygvx(handle,\n                     rocblas_eform_ax,           // itype=1: A*x = λ*B*x\n                     rocblas_evect_original,     // compute eigenvectors\n                     rocblas_erange_value,       // select by value range\n                     rocblas_fill_upper,         // use upper triangle\n                     n,\n                     dA, lda,                    // A matrix (overwritten with reduction)\n                     dB, ldb,                    // B matrix (overwritten with Cholesky)\n                     vl, vu,                     // eigenvalue interval (vl, vu]\n                     0, 0,                       // il, iu unused for erange_value\n                     abstol,                     // convergence tolerance for bisection\n                     &nev,                       // output: count of eigenvalues found\n                     dW,                         // eigenvalues (first nev entries valid)\n                     dZ, ldz,                    // eigenvectors (first nev columns valid)\n                     dIfail,                     // indices of eigenvectors that failed to converge\n                     &info);                     // error code\n    \n    // Check results\n    if (info != 0) {\n        if (info <= n) {\n            std::cout << \"POTRF failed: B not positive definite at element \" << info << std::endl;\n            std::cout << \"nev was set to 0, no eigenvalues computed.\" << std::endl;\n        } else {\n            std::cout << \"SYEVX reported \" << info - n << \" eigenvectors failed to converge\" << std::endl;\n        }\n    } else {\n        std::cout << \"Success: Found \" << nev << \" eigenvalues in interval (\" \n                  << vl << \", \" << vu << \"]\" << std::endl;\n    }\n    \n    // Copy results back (only first nev eigenvalues/eigenvectors are valid)\n    hipMemcpy(hW.data(), dW, sizeof(double) * nev, hipMemcpyDeviceToHost);\n    hipMemcpy(hZ.data(), dZ, sizeof(double) * n * nev, hipMemcpyDeviceToHost);\n    hipMemcpy(hIfail.data(), dIfail, sizeof(rocblas_int) * nev, hipMemcpyDeviceToHost);\n    \n    // Check ifail for convergence failures\n    for (rocblas_int i = 0; i < nev; i++) {\n        if (hIfail[i] != 0) {\n            std::cout << \"Warning: Eigenvector \" << i << \" did not converge (ifail=\" \n                      << hIfail[i] << \")\" << std::endl;\n        }\n    }\n    \n    // Print eigenvalues found\n    std::cout << \"Eigenvalues in (\" << vl << \", \" << vu << \"]:\" << std::endl;\n    for (rocblas_int i = 0; i < std::min(nev, (rocblas_int)10); i++) {\n        std::cout << \"  λ[\" << i << \"] = \" << hW[i] << std::endl;\n    }\n    \n    // Cleanup\n    hipFree(dA);\n    hipFree(dB);\n    hipFree(dW);\n    hipFree(dZ);\n    hipFree(dIfail);\n    rocblas_destroy_handle(handle);\n    \n    return 0;\n}\n```\n\nKey points: nev is an output that can be less than n for erange_value, only the first nev entries of W and columns of Z are valid, and ifail reports per-eigenvector convergence status from STEIN.",
    "difficulty": "L3",
    "interface": "rocsolver_dsygvx",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_syevx_heevx.cpp",
            "language": "cpp",
            "content": "rocblas_status rocsolver_dsyevx(rocblas_handle handle,\n                                const rocblas_evect evect,\n                                const rocblas_erange erange,\n                                const rocblas_fill uplo,\n                                const rocblas_int n,\n                                double* A,\n                                const rocblas_int lda,\n                                const double vl,\n                                const double vu,\n                                const rocblas_int il,\n                                const rocblas_int iu,\n                                const double abstol,\n                                rocblas_int* nev,\n                                double* W,\n                                double* Z,\n                                const rocblas_int ldz,\n                                rocblas_int* ifail,\n                                rocblas_int* info)"
        }
    ]
})

time.sleep(0.001)

# Entry 9 (L2): Comparison with SYGVD and SYGVJ
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "Compare rocsolver_sygvx with rocsolver_sygvd and rocsolver_sygvj in terms of when each should be used and their algorithmic approaches.",
    "answer": "SYGVD (divide-and-conquer) is fastest for computing all eigenvalues, using SYEVD which has O(n³) complexity with good constants. Use it when all eigenvalues are needed and default accuracy is acceptable. SYGVJ (Jacobi iteration) provides iterative control via abstol, max_sweeps, residual, and n_sweeps parameters, offering better stability for ill-conditioned problems and user-controlled accuracy/convergence. Use it when specific accuracy is required or for difficult numerical cases. SYGVX (bisection+inverse iteration) enables selective computation via erange, computing only eigenvalues in a value interval or index range using STEBZ+STEIN. Use it when only a subset of eigenvalues is needed, as it can be much faster than computing all n eigenvalues. SYGVX has additional outputs (nev for variable count, ifail for per-vector convergence, separate Z matrix). All three follow POTRF→SYGST→eigenvalue solver→back-transform, differing only in the eigenvalue solver phase.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvx_hegvx_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevx_heevx_template<BATCHED, STRIDED, T>(\n        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, abstol, nev, W,\n        strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, iinfo, batch_count, scalars, work1, work2,\n        work3, work4, work5, work6, D, E, iblock, isplit, tau, (T**)work7_workArr);"
        }
    ]
})

time.sleep(0.001)

# Entry 10 (L2): Workspace allocation for SYEVX components
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What additional workspace arrays does rocsolver_sygvx_hegvx_getMemorySize allocate compared to rocsolver_sygvd, and what are they used for?",
    "answer": "SYGVX allocates several additional arrays needed by SYEVX's STEBZ (bisection) and STEIN (inverse iteration) algorithms: (1) size_D and size_E for the diagonal and off-diagonal of the tridiagonal matrix from SYTRD, (2) size_iblock for block indices used by STEBZ to track eigenvalue intervals, (3) size_isplit for split points in the tridiagonal matrix where off-diagonals are negligible, (4) size_tau for Householder reflector scalars from SYTRD, and (5) size_work7_workArr for additional workspace specific to SYEVX. These are obtained from rocsolver_syevx_heevx_getMemorySize at line 181. In contrast, SYGVD only needs work1-6 plus iinfo since SYEVD (divide-and-conquer) has different workspace requirements. The getMemorySize function takes the max across POTRF, SYGST, SYEVX, and TRSM requirements for reusable workspaces (work1-4).",
    "difficulty": "L2",
    "interface": "rocsolver_sygvx_hegvx_getMemorySize",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    // requirements for calling SYEVX/HEEVX\n    rocsolver_syevx_heevx_getMemorySize<BATCHED, T, S>(\n        evect, uplo, n, batch_count, &unused, &temp1, &temp2, &temp3, &temp4, size_work5,\n        size_work6, size_D, size_E, size_iblock, size_isplit, size_tau, &temp5);\n    *size_work1 = std::max(*size_work1, temp1);\n    *size_work2 = std::max(*size_work2, temp2);\n    *size_work3 = std::max(*size_work3, temp3);\n    *size_work4 = std::max(*size_work4, temp4);\n    *size_work7_workArr = std::max(*size_work7_workArr, temp5);"
        }
    ]
})

time.sleep(0.001)

# Entry 11 (L1): ifail array purpose
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What is the purpose of the ifail array in rocsolver_sygvx, and when is it required?",
    "answer": "The ifail array reports convergence status for individual eigenvectors computed by STEIN (inverse iteration). It has length n, but only the first nev entries are meaningful. If ifail[i] = 0, the i-th eigenvector converged successfully. If ifail[i] > 0, that eigenvector did not converge within STEIN's iteration limit. The ifail parameter is required (checked in argCheck at line 114) only when evect = rocblas_evect_original (computing eigenvectors). When evect = rocblas_evect_none (eigenvalues only), ifail can be NULL. This differs from the info parameter which reports global errors: info > 0 indicates either POTRF failure (info <= n) or the count of non-converged eigenvectors (info > n, specifically info - n failed). The ifail array provides finer-grained per-vector diagnostics.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvx_hegvx_argCheck",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    // 3. invalid pointers\n    if((n && !A) || (n && !B) || (n && !W) || (batch_count && !nev) || (batch_count && !info))\n        return rocblas_status_invalid_pointer;\n    if(evect != rocblas_evect_none && ((n && !Z) || (n && !ifail)))\n        return rocblas_status_invalid_pointer;"
        }
    ]
})

time.sleep(0.001)

# Entry 12 (L3, coding): Implement kernel to extract eigenvalue subrange post-computation
entries.append({
    "id": generate_timestamp_id(),
    "type": "coding",
    "question": "Implement a HIP device kernel `extract_eigenvalue_range` that takes the full eigenvalue array W (sorted ascending) and extracts eigenvalues in a specified value range [vl_new, vu_new], compacting them to the beginning of the array and returning the new count. This could be useful for post-filtering SYGVX results. Include both the kernel and a host wrapper.",
    "answer": "Here is the implementation:\n\n```cpp\n#include <hip/hip_runtime.h>\n\n// Kernel to count eigenvalues in range (per batch)\n__global__ void count_eigenvalues_in_range(const double* W,\n                                           const rocblas_int n,\n                                           const double vl,\n                                           const double vu,\n                                           rocblas_int* count,\n                                           const rocblas_stride strideW,\n                                           const rocblas_int batch_count)\n{\n    rocblas_int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    \n    if(bid < batch_count)\n    {\n        const double* W_batch = W + bid * strideW;\n        rocblas_int local_count = 0;\n        \n        // W is sorted ascending, use binary search for efficiency\n        // Find first eigenvalue > vl\n        rocblas_int left = 0, right = n;\n        while(left < right)\n        {\n            rocblas_int mid = (left + right) / 2;\n            if(W_batch[mid] <= vl)\n                left = mid + 1;\n            else\n                right = mid;\n        }\n        rocblas_int start = left;\n        \n        // Find first eigenvalue > vu\n        left = start;\n        right = n;\n        while(left < right)\n        {\n            rocblas_int mid = (left + right) / 2;\n            if(W_batch[mid] <= vu)\n                left = mid + 1;\n            else\n                right = mid;\n        }\n        rocblas_int end = left;\n        \n        count[bid] = end - start;\n    }\n}\n\n// Kernel to compact eigenvalues in range to beginning of array\n__global__ void compact_eigenvalues_in_range(double* W,\n                                             const rocblas_int n,\n                                             const double vl,\n                                             const double vu,\n                                             const rocblas_stride strideW,\n                                             const rocblas_int batch_count)\n{\n    rocblas_int bid = hipBlockIdx_x;\n    \n    if(bid < batch_count)\n    {\n        double* W_batch = W + bid * strideW;\n        \n        // Find range using binary search (same as count kernel)\n        rocblas_int left = 0, right = n;\n        while(left < right)\n        {\n            rocblas_int mid = (left + right) / 2;\n            if(W_batch[mid] <= vl)\n                left = mid + 1;\n            else\n                right = mid;\n        }\n        rocblas_int start = left;\n        \n        left = start;\n        right = n;\n        while(left < right)\n        {\n            rocblas_int mid = (left + right) / 2;\n            if(W_batch[mid] <= vu)\n                left = mid + 1;\n            else\n                right = mid;\n        }\n        rocblas_int end = left;\n        \n        // Compact: shift eigenvalues to beginning\n        rocblas_int tid = hipThreadIdx_x;\n        rocblas_int range_size = end - start;\n        \n        for(rocblas_int i = tid; i < range_size; i += hipBlockDim_x)\n        {\n            W_batch[i] = W_batch[start + i];\n        }\n    }\n}\n\n// Host wrapper function\nrocblas_status rocsolver_extract_eigenvalue_range(rocblas_handle handle,\n                                                  double* dW,\n                                                  const rocblas_int n,\n                                                  const double vl,\n                                                  const double vu,\n                                                  rocblas_int* d_nev_new,\n                                                  const rocblas_stride strideW,\n                                                  const rocblas_int batch_count)\n{\n    if(n <= 0 || batch_count <= 0 || vl >= vu)\n        return rocblas_status_invalid_size;\n    \n    if(!dW || !d_nev_new)\n        return rocblas_status_invalid_pointer;\n    \n    hipStream_t stream;\n    rocblas_get_stream(handle, &stream);\n    \n    // Step 1: Count eigenvalues in range for each batch\n    rocblas_int blocks = (batch_count - 1) / 256 + 1;\n    dim3 grid_count(blocks, 1, 1);\n    dim3 threads_count(256, 1, 1);\n    \n    hipLaunchKernelGGL(count_eigenvalues_in_range,\n                       grid_count, threads_count, 0, stream,\n                       dW, n, vl, vu, d_nev_new, strideW, batch_count);\n    \n    // Step 2: Compact eigenvalues to beginning of array\n    dim3 grid_compact(batch_count, 1, 1);\n    dim3 threads_compact(256, 1, 1);\n    \n    hipLaunchKernelGGL(compact_eigenvalues_in_range,\n                       grid_compact, threads_compact, 0, stream,\n                       dW, n, vl, vu, strideW, batch_count);\n    \n    return rocblas_status_success;\n}\n\n// Example usage after SYGVX call\nvoid example_usage()\n{\n    rocblas_handle handle;\n    rocblas_create_handle(&handle);\n    \n    const rocblas_int n = 1024;\n    double *dW;\n    rocblas_int *d_nev_new;\n    hipMalloc(&dW, sizeof(double) * n);\n    hipMalloc(&d_nev_new, sizeof(rocblas_int));\n    \n    // After calling rocsolver_dsygvx with erange_all to get all eigenvalues...\n    // Now extract only eigenvalues in [1.0, 5.0]\n    \n    rocsolver_extract_eigenvalue_range(handle, dW, n, 1.0, 5.0, d_nev_new, 0, 1);\n    \n    rocblas_int h_nev_new;\n    hipMemcpy(&h_nev_new, d_nev_new, sizeof(rocblas_int), hipMemcpyDeviceToHost);\n    \n    std::cout << \"Extracted \" << h_nev_new << \" eigenvalues in [1.0, 5.0]\" << std::endl;\n    // First h_nev_new entries of dW now contain the filtered eigenvalues\n    \n    hipFree(dW);\n    hipFree(d_nev_new);\n    rocblas_destroy_handle(handle);\n}\n```\n\nThe implementation uses binary search to efficiently locate the range boundaries in the sorted eigenvalue array, then compacts the selected values to the array start. This is useful for post-filtering or combining results from multiple SYGVX calls.",
    "difficulty": "L3",
    "interface": "extract_eigenvalue_range",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvx_hegvx.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevx_heevx_template<BATCHED, STRIDED, T>(\n        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, abstol, nev, W,\n        strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, iinfo, batch_count, scalars, work1, work2,\n        work3, work4, work5, work6, D, E, iblock, isplit, tau, (T**)work7_workArr);"
        }
    ]
})

# Write to JSONL file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_sygvx_hegvx.jsonl'
with open(output_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(entries)} entries in {output_path}")

# Validate schema
def validate_entry(entry):
    assert 'id' in entry
    assert 'type' in entry
    assert entry['type'] in ['question-answer', 'coding']
    assert 'question' in entry
    assert 'answer' in entry
    assert 'difficulty' in entry
    assert entry['difficulty'] in ['L1', 'L2', 'L3']
    assert 'interface' in entry
    assert 'code_blocks' in entry
    for block in entry['code_blocks']:
        assert 'path' in block
        assert 'language' in block
        assert 'content' in block

for entry in entries:
    validate_entry(entry)

print("Schema validation passed!")

# Print level distribution
l1_count = sum(1 for e in entries if e['difficulty'] == 'L1')
l2_count = sum(1 for e in entries if e['difficulty'] == 'L2')
l3_count = sum(1 for e in entries if e['difficulty'] == 'L3')
print(f"Level distribution: L1={l1_count}, L2={l2_count}, L3={l3_count}")
