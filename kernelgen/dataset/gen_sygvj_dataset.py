#!/usr/bin/env python3
import json
import time

def generate_timestamp_id():
    return str(int(time.time() * 1000000))

entries = []

# Entry 1 (L2): SYGVJ overview - Jacobi iteration with max_sweeps
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What is rocsolver_sygvj and how does it differ from rocsolver_sygvd in terms of convergence control?",
    "answer": "rocsolver_sygvj solves the generalized symmetric eigenvalue problem A*x = λ*B*x using the Jacobi iteration method combined with divide-and-conquer. Unlike SYGVD which runs to completion, SYGVJ provides iterative convergence control through three key parameters: abstol (absolute tolerance for convergence), max_sweeps (maximum number of Jacobi sweeps), and n_sweeps (output parameter reporting actual sweeps performed). The algorithm first performs POTRF on B, then SYGST to reduce to standard form, then SYEVJ (Jacobi eigenvalue solver) with the convergence parameters, and finally back-transforms eigenvectors using TRSM or TRMM depending on itype. The residual output parameter reports the final convergence metric, allowing users to verify if the solution meets their accuracy requirements.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(\n        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual,\n        max_sweeps, n_sweeps, W, strideW, iinfo, batch_count, (T*)work1, (T*)work2, (S*)work3,\n        (rocblas_int*)work4, (rocblas_int*)work5, (rocblas_int*)work6);"
        }
    ]
})

time.sleep(0.001)

# Entry 2 (L1): abstol and residual parameters
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What is the purpose of the abstol and residual parameters in rocsolver_sygvj, and how do they relate to convergence?",
    "answer": "The abstol parameter sets the absolute tolerance threshold for convergence of the Jacobi iteration. The algorithm iterates until the off-diagonal elements are reduced below this tolerance or max_sweeps is reached. The residual parameter is an output that reports the final residual value (typically the maximum off-diagonal element) after the algorithm completes. By comparing residual against abstol, users can verify whether the solution converged to the desired accuracy. If residual > abstol and n_sweeps == max_sweeps, the algorithm stopped due to iteration limit rather than convergence, indicating the user should either increase max_sweeps or relax abstol.",
    "difficulty": "L1",
    "interface": "rocsolver_ssygvj",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.cpp",
            "language": "cpp",
            "content": "rocblas_status rocsolver_ssygvj(rocblas_handle handle,\n                                const rocblas_eform itype,\n                                const rocblas_evect evect,\n                                const rocblas_fill uplo,\n                                const rocblas_int n,\n                                float* A,\n                                const rocblas_int lda,\n                                float* B,\n                                const rocblas_int ldb,\n                                const float abstol,\n                                float* residual,\n                                const rocblas_int max_sweeps,\n                                rocblas_int* n_sweeps,\n                                float* W,\n                                rocblas_int* info)"
        }
    ]
})

time.sleep(0.001)

# Entry 3 (L2): sygv_update_info kernel usage
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How does the sygv_update_info kernel combine error information from POTRF and SYEVJ in rocsolver_sygvj?",
    "answer": "The sygv_update_info kernel merges error codes from two phases: POTRF (Cholesky factorization stored in info) and SYEVJ (eigenvalue solver stored in iinfo). The kernel is launched with batch_count threads and passes the matrix dimension n. If POTRF failed (info > 0), that error is preserved since it indicates B is not positive definite and no eigenvalues can be computed. If POTRF succeeded but SYEVJ failed to converge for some eigenvalues (iinfo > 0), the error code is adjusted by adding n to distinguish it from POTRF errors. This allows the user to determine whether the failure occurred during factorization or eigenvalue computation.",
    "difficulty": "L2",
    "interface": "sygv_update_info",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    // combine info from POTRF with info from SYEV/HEEV\n    ROCSOLVER_LAUNCH_KERNEL(sygv_update_info, gridReset, threadsReset, 0, stream, info, iinfo, n,\n                            batch_count);"
        }
    ]
})

time.sleep(0.001)

# Entry 4 (L1): TODO about B not positive definite
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What does the TODO comment at line 245 of roclapack_sygvj_hegvj.hpp indicate about the current behavior when B is not positive definite?",
    "answer": "The TODO comment indicates that in the current implementation, if POTRF fails because B is not positive definite, the algorithm continues to process matrix A even though no valid eigenvalues can be computed. This is inefficient and mathematically incorrect - the comment states 'A should not be modified in this case as no eigenvalues or eigenvectors can be computed.' The developers acknowledge the need to stop computation immediately after POTRF failure, but note 'Need to find a way to do this efficiently.' As a workaround, the current implementation will destroy matrix A in the non-positive-definite case, which is undesirable behavior.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    /** (TODO: Strictly speaking, computations should stop here is B is not positive definite.\n        A should not be modified in this case as no eigenvalues or eigenvectors can be computed.\n        Need to find a way to do this efficiently; for now A will be destroyed in the non\n        positive-definite case) **/"
        }
    ]
})

time.sleep(0.001)

# Entry 5 (L2): Workspace memory allocation strategy
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How does rocsolver_sygvj_hegvj_getMemorySize calculate workspace requirements across the multiple algorithm phases (POTRF, SYGST, SYEVJ, back-transform)?",
    "answer": "The function calculates workspace requirements by taking the maximum across all phases rather than summing them, since workspaces are reused. It calls getMemorySize functions for each phase: POTRF (work1-5, iinfo), SYGST (work1-4), SYEVJ (work1-5, work6), and TRSM/TRMM for back-transformation (work1-4). For each work array, it uses std::max to ensure sufficient space for the largest requirement. For example, *size_work1 = std::max(*size_work1, temp1) is called after each phase query. The function also computes an optim_mem flag by AND-ing the optimization flags from all phases (opt1 && opt2 && opt3), which indicates whether optimized batched memory mode can be used across the entire computation.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvj_hegvj_getMemorySize",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    // requirements for calling SYGST/HEGST\n    rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count, &unused,\n                                                             &temp1, &temp2, &temp3, &temp4, &opt2);\n    *size_work1 = std::max(*size_work1, temp1);\n    *size_work2 = std::max(*size_work2, temp2);\n    *size_work3 = std::max(*size_work3, temp3);\n    *size_work4 = std::max(*size_work4, temp4);"
        }
    ]
})

time.sleep(0.001)

# Entry 6 (L2): Back-transformation based on itype
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How does the back-transformation of eigenvectors in rocsolver_sygvj differ for different itype values, and why?",
    "answer": "For itype=rocblas_eform_ax or rocblas_eform_abx (types 1 and 2), the back-transformation uses TRSM (triangular solve) with B: Z = L^(-T)*Y or Z = U^(-1)*Y depending on uplo. For itype=rocblas_eform_bax (type 3), it uses TRMM (triangular multiply): Z = L*Y or Z = U^T*Y. This difference arises from the mathematical form of the problem reduction. Types 1 and 2 transform to C = L^T*A*L or C = U*A*U^T, requiring inverse operations (TRSM) to recover eigenvectors. Type 3 transforms to C = L^(-1)*A*L^(-T), requiring forward multiplication (TRMM). The operation and uplo parameters are carefully selected to match the mathematical requirements of each problem type.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)\n        {\n            if(uplo == rocblas_fill_upper)\n                rocsolver_trsm_upper<BATCHED, STRIDED, T>(\n                    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,\n                    n, B, shiftB, ldb, strideB, A, shiftA, lda, strideA, batch_count, optim_mem,\n                    work1, work2, work3, work4);\n        }\n        else\n        {\n            rocblas_operation trans\n                = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose\n                                              : rocblas_operation_none);\n            rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,\n                             neig, &one, 0, B, shiftB, ldb, strideB, A, shiftA, lda, strideA,\n                             batch_count, (T**)work5);\n        }"
        }
    ]
})

time.sleep(0.001)

# Entry 7 (L1): TODO about neig < n efficiency
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "What inefficiency is described in the TODO comment at line 264 regarding partial convergence in SYGVJ?",
    "answer": "The TODO comment points out that if SYEVJ only converges for neig < n eigenvalues (stored in iinfo), the back-transformation (TRSM or TRMM) should only operate on the neig converged eigenvectors rather than the entire n×n matrix. However, the current implementation ignores iinfo and sets neig = n, forcing the back-transformation to process all n columns even when some haven't converged. This wastes computational resources and may produce invalid results for the non-converged eigenvectors. The comment states 'Need to find a way to do this efficiently,' indicating that selective column processing in TRSM/TRMM is not straightforward in the current batched GPU implementation.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    /** (TODO: Similarly, if only neig < n eigenvalues converged, TRSM or TRMM below should not\n        work with the entire matrix. Need to find a way to do this efficiently; for now we ignore\n        iinfo and set neig = n) **/\n\n    rocblas_int neig = n; //number of converged eigenvalues"
        }
    ]
})

time.sleep(0.001)

# Entry 8 (L3): Complete usage example with convergence monitoring
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "Write a complete HIP code example demonstrating rocsolver_dsygvj with itype=1, including convergence monitoring through abstol, residual, max_sweeps, and n_sweeps parameters.",
    "answer": "Here is a complete example showing SYGVJ usage with convergence control:\n\n```cpp\n#include <hip/hip_runtime.h>\n#include <rocsolver/rocsolver.h>\n#include <iostream>\n#include <vector>\n\nint main() {\n    rocblas_handle handle;\n    rocblas_create_handle(&handle);\n    \n    const rocblas_int n = 512;\n    const rocblas_int lda = n, ldb = n;\n    \n    // Allocate host memory\n    std::vector<double> hA(n * n);\n    std::vector<double> hB(n * n);\n    std::vector<double> hW(n);\n    \n    // Initialize symmetric A and positive definite B\n    // ... (initialization code)\n    \n    // Allocate device memory\n    double *dA, *dB, *dW;\n    hipMalloc(&dA, sizeof(double) * n * n);\n    hipMalloc(&dB, sizeof(double) * n * n);\n    hipMalloc(&dW, sizeof(double) * n);\n    \n    hipMemcpy(dA, hA.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);\n    hipMemcpy(dB, hB.data(), sizeof(double) * n * n, hipMemcpyHostToDevice);\n    \n    // Convergence parameters\n    const double abstol = 1e-8;\n    const rocblas_int max_sweeps = 100;\n    double residual;\n    rocblas_int n_sweeps;\n    rocblas_int info;\n    \n    // Solve generalized eigenvalue problem with Jacobi iteration\n    rocsolver_dsygvj(handle,\n                     rocblas_eform_ax,           // itype=1: A*x = λ*B*x\n                     rocblas_evect_original,     // compute eigenvectors\n                     rocblas_fill_upper,         // use upper triangle\n                     n,\n                     dA, lda,                    // A matrix (overwritten with eigenvectors)\n                     dB, ldb,                    // B matrix (overwritten with Cholesky factor)\n                     abstol,                     // convergence tolerance\n                     &residual,                  // output: final residual\n                     max_sweeps,                 // maximum iterations\n                     &n_sweeps,                  // output: actual iterations performed\n                     dW,                         // eigenvalues output\n                     &info);                     // error code\n    \n    // Check convergence\n    if (info != 0) {\n        if (info <= n) {\n            std::cout << \"POTRF failed: B not positive definite at element \" << info << std::endl;\n        } else {\n            std::cout << \"SYEVJ failed: \" << (info - n) << \" eigenvalues did not converge\" << std::endl;\n        }\n    } else if (residual > abstol) {\n        std::cout << \"Warning: Did not converge to abstol=\" << abstol \n                  << \", residual=\" << residual \n                  << \" after n_sweeps=\" << n_sweeps << \" (max=\" << max_sweeps << \")\" << std::endl;\n    } else {\n        std::cout << \"Converged in \" << n_sweeps << \" sweeps, residual=\" << residual << std::endl;\n    }\n    \n    // Copy results back\n    hipMemcpy(hW.data(), dW, sizeof(double) * n, hipMemcpyDeviceToHost);\n    hipMemcpy(hA.data(), dA, sizeof(double) * n * n, hipMemcpyDeviceToHost);\n    \n    // Cleanup\n    hipFree(dA);\n    hipFree(dB);\n    hipFree(dW);\n    rocblas_destroy_handle(handle);\n    \n    return 0;\n}\n```\n\nThis example demonstrates proper convergence monitoring by checking info, comparing residual against abstol, and examining n_sweeps to determine if the iteration limit was reached.",
    "difficulty": "L3",
    "interface": "rocsolver_dsygvj",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.cpp",
            "language": "cpp",
            "content": "rocblas_status rocsolver_dsygvj(rocblas_handle handle,\n                                const rocblas_eform itype,\n                                const rocblas_evect evect,\n                                const rocblas_fill uplo,\n                                const rocblas_int n,\n                                double* A,\n                                const rocblas_int lda,\n                                double* B,\n                                const rocblas_int ldb,\n                                const double abstol,\n                                double* residual,\n                                const rocblas_int max_sweeps,\n                                rocblas_int* n_sweeps,\n                                double* W,\n                                rocblas_int* info)\n{\n    return rocsolver::rocsolver_sygvj_hegvj_impl<double>(handle, itype, evect, uplo, n, A, lda, B,\n                                                         ldb, abstol, residual, max_sweeps,\n                                                         n_sweeps, W, info);\n}"
        }
    ]
})

time.sleep(0.001)

# Entry 9 (L2): Comparison with SYGVD
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "When should a developer choose rocsolver_sygvj over rocsolver_sygvd, and what are the key algorithmic differences?",
    "answer": "SYGVJ should be chosen when iterative convergence control is needed or when working with ill-conditioned matrices where Jacobi iteration may be more stable. SYGVD uses SYEVD (divide-and-conquer) which is generally faster but runs to completion without user control over iterations. SYGVJ uses SYEVJ (Jacobi iteration) which allows users to set abstol for accuracy requirements and max_sweeps to limit computation time, with residual and n_sweeps providing convergence feedback. Both follow the same reduction strategy (POTRF→SYGST→eigenvalue solver→back-transform), differing only in the eigenvalue solver phase. SYGVJ is preferable for applications requiring guaranteed accuracy (via abstol), early stopping for approximate solutions, or better numerical stability on difficult problems. SYGVD is better for general cases where maximum performance is needed and default accuracy is acceptable.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(\n        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual,\n        max_sweeps, n_sweeps, W, strideW, iinfo, batch_count, (T*)work1, (T*)work2, (S*)work3,\n        (rocblas_int*)work4, (rocblas_int*)work5, (rocblas_int*)work6);"
        }
    ]
})

time.sleep(0.001)

# Entry 10 (L1): rocblas_esort_ascending parameter
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "Why is rocblas_esort_ascending hardcoded in the call to rocsolver_syevj_heevj_template within SYGVJ, and can users control eigenvalue ordering?",
    "answer": "The rocblas_esort_ascending parameter is hardcoded in line 256 of the template implementation, meaning eigenvalues are always returned in ascending order for SYGVJ, regardless of user preference. This differs from SYEVJ which accepts esort as a user parameter. Users cannot control the ordering in SYGVJ - eigenvalues in W will always be sorted from smallest to largest, with corresponding eigenvectors in the columns of A. If descending order is needed, users must manually reverse the W array and reorder the columns of A after the computation completes. This design choice simplifies the SYGVJ interface at the cost of flexibility.",
    "difficulty": "L1",
    "interface": "rocsolver_sygvj_hegvj_template",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    rocsolver_syevj_heevj_template<BATCHED, STRIDED, T>(\n        handle, rocblas_esort_ascending, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual,\n        max_sweeps, n_sweeps, W, strideW, iinfo, batch_count, (T*)work1, (T*)work2, (S*)work3,\n        (rocblas_int*)work4, (rocblas_int*)work5, (rocblas_int*)work6);"
        }
    ]
})

time.sleep(0.001)

# Entry 11 (L2): optim_mem flag propagation
entries.append({
    "id": generate_timestamp_id(),
    "type": "question-answer",
    "question": "How is the optim_mem flag computed and used in rocsolver_sygvj_hegvj, and what does it optimize?",
    "answer": "The optim_mem flag is computed by AND-ing the optimization flags from all algorithm phases: opt1 (from POTRF), opt2 (from SYGST), and opt3 (from TRSM/TRMM back-transformation). It's calculated in the getMemorySize function as *optim_mem = opt1 && opt2 && opt3. The flag is true only if all phases can use optimized batched memory layouts. This flag is then passed to the template function and forwarded to POTRF, SYGST, TRSM, and TRMM calls to enable optimized memory access patterns for batched operations. When optim_mem is false, the implementation falls back to less efficient but more general memory layouts. This optimization is particularly important for batched SYGVJ where multiple problems are solved simultaneously, as it can significantly improve memory bandwidth utilization.",
    "difficulty": "L2",
    "interface": "rocsolver_sygvj_hegvj_getMemorySize",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.hpp",
            "language": "cpp",
            "content": "    *optim_mem = opt1 && opt2 && opt3;"
        },
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.cpp",
            "language": "cpp",
            "content": "    return rocsolver_sygvj_hegvj_template<false, false, T>(\n        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, abstol,\n        residual, max_sweeps, n_sweeps, W, strideW, info, batch_count, (T*)scalars, work1, work2,\n        work3, work4, work5, work6, (rocblas_int*)iinfo, optim_mem);"
        }
    ]
})

time.sleep(0.001)

# Entry 12 (L3, coding): Implement adaptive max_sweeps calculator
entries.append({
    "id": generate_timestamp_id(),
    "type": "coding",
    "question": "Implement a HIP device kernel `adaptive_max_sweeps_kernel` that calculates an appropriate max_sweeps value for SYGVJ based on matrix size n and desired abstol. The kernel should use the heuristic: max_sweeps = max(30, min(100, 10 + 2*log2(n) + 5*log10(1/abstol))). Write both the device kernel and a host wrapper function that can be called before rocsolver_sygvj.",
    "answer": "Here is the implementation:\n\n```cpp\n#include <hip/hip_runtime.h>\n#include <cmath>\n\n// Device kernel to compute adaptive max_sweeps\n__global__ void adaptive_max_sweeps_kernel(const rocblas_int n,\n                                           const double abstol,\n                                           rocblas_int* max_sweeps,\n                                           const rocblas_int batch_count)\n{\n    rocblas_int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    \n    if(bid < batch_count)\n    {\n        // Compute heuristic: base iterations + size scaling + tolerance scaling\n        double log2_n = log2((double)n);\n        double log10_invtol = -log10(abstol);  // log10(1/abstol)\n        \n        double sweeps_estimate = 10.0 + 2.0 * log2_n + 5.0 * log10_invtol;\n        \n        // Clamp to reasonable range [30, 100]\n        rocblas_int sweeps = (rocblas_int)sweeps_estimate;\n        sweeps = max(30, min(100, sweeps));\n        \n        max_sweeps[bid] = sweeps;\n    }\n}\n\n// Host wrapper function\nrocblas_status rocsolver_compute_adaptive_max_sweeps(rocblas_handle handle,\n                                                     const rocblas_int n,\n                                                     const double abstol,\n                                                     rocblas_int* d_max_sweeps,\n                                                     const rocblas_int batch_count)\n{\n    if(n <= 0 || abstol <= 0.0 || batch_count <= 0)\n        return rocblas_status_invalid_size;\n    \n    if(!d_max_sweeps)\n        return rocblas_status_invalid_pointer;\n    \n    hipStream_t stream;\n    rocblas_get_stream(handle, &stream);\n    \n    // Launch with one thread per batch\n    rocblas_int blocks = (batch_count - 1) / 256 + 1;\n    dim3 grid(blocks, 1, 1);\n    dim3 threads(256, 1, 1);\n    \n    hipLaunchKernelGGL(adaptive_max_sweeps_kernel,\n                       grid, threads, 0, stream,\n                       n, abstol, d_max_sweeps, batch_count);\n    \n    return rocblas_status_success;\n}\n\n// Example usage before SYGVJ call\nvoid example_usage()\n{\n    rocblas_handle handle;\n    rocblas_create_handle(&handle);\n    \n    const rocblas_int n = 1024;\n    const double abstol = 1e-10;\n    \n    // Compute adaptive max_sweeps on device\n    rocblas_int *d_max_sweeps;\n    hipMalloc(&d_max_sweeps, sizeof(rocblas_int));\n    \n    rocsolver_compute_adaptive_max_sweeps(handle, n, abstol, d_max_sweeps, 1);\n    \n    // Copy back to host\n    rocblas_int h_max_sweeps;\n    hipMemcpy(&h_max_sweeps, d_max_sweeps, sizeof(rocblas_int), hipMemcpyDeviceToHost);\n    \n    std::cout << \"Adaptive max_sweeps for n=\" << n \n              << \", abstol=\" << abstol \n              << \": \" << h_max_sweeps << std::endl;\n    // Expected: 10 + 2*log2(1024) + 5*log10(1e10) = 10 + 20 + 50 = 80\n    \n    // Now use h_max_sweeps in rocsolver_dsygvj call\n    // ...\n    \n    hipFree(d_max_sweeps);\n    rocblas_destroy_handle(handle);\n}\n```\n\nThe kernel uses logarithmic scaling for both matrix size and tolerance to provide sensible iteration counts. Larger matrices and stricter tolerances require more sweeps. The [30, 100] bounds prevent extreme values.",
    "difficulty": "L3",
    "interface": "adaptive_max_sweeps_kernel",
    "code_blocks": [
        {
            "path": "library/src/lapack/roclapack_sygvj_hegvj.cpp",
            "language": "cpp",
            "content": "    ROCSOLVER_ENTER_TOP(name, \"--itype\", itype, \"--evect\", evect, \"--uplo\", uplo, \"-n\", n, \"--lda\",\n                        lda, \"--ldb\", ldb, \"--abstol\", abstol, \"--max_sweeps\", max_sweeps);"
        }
    ]
})

# Write to JSONL file
output_path = '/root/rocSOLVER/kernelgen/dataset/roclapack_sygvj_hegvj.jsonl'
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
