#!/usr/bin/env python3
"""
Generator for POTF2 (unblocked Cholesky factorization) SFT dataset entries.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def generate_entries():
    entries = []

    # L1-1: Diagonal Update with Positive Definiteness Check (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L1_sqrt_diag",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the sqrtDiagOnward kernel that updates the diagonal element during Cholesky factorization and checks for positive definiteness. This is the critical operation that computes the square root of the updated diagonal.

Requirements:
- Handle both real and complex types (complex uses only real part of diagonal)
- Compute t = A[j,j] - sum(A[0:j-1,j]' * A[0:j-1,j]) where result is stored in res[id]
- Check if t <= 0: matrix is not positive definite, set info[id] = j+1 (Fortran 1-based)
- If t > 0: compute A[j,j] = sqrt(t) and res[id] = 1/A[j,j] for subsequent SCAL
- Thread launch: dim3(batch_count), dim3(1) - one thread per batch instance

Provide complete implementation for both real and complex types, explaining why only the real part is used for complex Hermitian matrices."""
            }]
        }
    })

    # L1-2: Upper Triangular Cholesky Iteration (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L1_upper_iteration",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement a single iteration (j-th column) of the upper triangular Cholesky factorization algorithm. The iteration computes A = U'*U where U is upper triangular.

For iteration j:
1. Compute A[j,j] = sqrt(A[j,j] - A[0:j-1,j]' * A[0:j-1,j])
2. Compute A[j,j+1:n-1] = (A[j,j+1:n-1] - A[0:j-1,j]' * A[0:j-1,j+1:n-1]) / A[j,j]

Requirements:
- Use DOT to compute A[0:j-1,j]' * A[0:j-1,j] (stored in pivots array)
- Call sqrtDiagOnward to update A[j,j] and compute 1/A[j,j]
- For complex types, conjugate A[0:j-1,j] before and after GEMV
- Use GEMV (transpose) to compute A[0:j-1,j]' * A[0:j-1,j+1:n-1]
- Use SCAL to multiply A[j,j+1:n-1] by 1/A[j,j]

Implement the single iteration logic with detailed comments explaining each step."""
            }]
        }
    })

    # L1-3: Lower Triangular Cholesky Iteration (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L1_lower_iteration",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement a single iteration (j-th column) of the lower triangular Cholesky factorization algorithm. The iteration computes A = L*L' where L is lower triangular.

For iteration j:
1. Compute A[j,j] = sqrt(A[j,j] - A[j,0:j-1] * A[j,0:j-1]')
2. Compute A[j+1:n-1,j] = (A[j+1:n-1,j] - A[j+1:n-1,0:j-1] * A[j,0:j-1]') / A[j,j]

Requirements:
- Use DOT to compute A[j,0:j-1] * A[j,0:j-1]' (with stride=lda for row access)
- Call sqrtDiagOnward to update A[j,j] and compute 1/A[j,j]
- For complex types, conjugate A[j,0:j-1] before and after GEMV
- Use GEMV (no transpose) to compute A[j+1:n-1,0:j-1] * conj(A[j,0:j-1])
- Use SCAL to multiply A[j+1:n-1,j] by 1/A[j,j]

Explain the difference in memory access patterns between upper and lower variants (stride 1 vs. lda)."""
            }]
        }
    })

    # L1-4: Conjugation Requirement (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_potf2_L1_complex_conjugation",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Explain why conjugation of row/column vectors is required before GEMV operations in complex Hermitian Cholesky factorization, and why it must be undone afterward.

Questions to address:
1. Why does Hermitian symmetry (A = A*) require special handling in Cholesky?
2. What would happen if conjugation was omitted for complex matrices?
3. Why is the conjugation applied to the vector being dot-producted, not the matrix columns/rows?
4. Explain the mathematical relationship: (U'*U)[i,j] = sum(conj(U[k,i]) * U[k,j]) for k < min(i,j)
5. Why must the original vector be restored after GEMV?
6. Compare computational cost: in-place conjugation vs. creating conjugated copy"""
            }]
        }
    })

    # L2-1: Small Matrix Specialized Kernel (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_potf2_L2_small_kernel",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the potf2_run_small specialized kernel that performs Cholesky factorization in shared memory for small matrices. Explain the performance benefits and implementation strategy.

Questions to address:
1. What is POTRF_BLOCKSIZE(T) and POTF2_MAX_SMALL_SIZE(T) for different precisions?
2. Why is no workspace needed for small matrices?
3. What operations are performed entirely in shared memory?
4. Calculate shared memory requirements for n=64 matrix with double precision
5. Compare performance: small kernel vs. iterative POTF2 for n=32, n=64, n=128
6. How does batching improve GPU utilization for small matrices?
7. What is the maximum matrix size limited by shared memory (48KB)?"""
            }]
        }
    })

    # L2-2: DOT Product for Diagonal Update (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L2_dot_diagonal",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the DOT product computation used to update diagonal elements in Cholesky factorization. This computes the inner product of a column/row with itself.

For upper triangular (A = U'*U):
- Compute: sum = A[0:j-1,j]' * A[0:j-1,j]
- Access pattern: stride=1 for both vectors (column access)

For lower triangular (A = L*L'):
- Compute: sum = A[j,0:j-1] * A[j,0:j-1]'
- Access pattern: stride=lda for both vectors (row access)

Requirements:
- For real types: DOT computes x' * y
- For complex types: DOT computes conj(x)' * y (DOTC operation)
- Result stored in pivots array for each batch instance
- Use work array for reduction workspace (size: (n-1)/512 + 2 per batch)

Explain how the DOT result is used in sqrtDiagOnward: t = A[j,j] - pivots[id]."""
            }]
        }
    })

    # L2-3: GEMV for Off-Diagonal Updates (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L2_gemv_update",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the GEMV operation used to update off-diagonal elements in Cholesky factorization. This computes rank-j updates to the trailing matrix.

For upper triangular (A = U'*U), update row j:
A[j,j+1:n-1] = A[j,j+1:n-1] - conj(A[0:j-1,j])' * A[0:j-1,j+1:n-1]

For lower triangular (A = L*L'), update column j:
A[j+1:n-1,j] = A[j+1:n-1,j] - A[j+1:n-1,0:j-1] * conj(A[j,0:j-1])'

Requirements:
- Use rocblas_operation_transpose for upper case
- Use rocblas_operation_none for lower case
- Scalars on device: scalars[0]=-1 (alpha), scalars[2]=1 (beta)
- For complex types, vector x must be conjugated before GEMV

Explain why the output stride differs: lda for upper (row storage) vs. 1 for lower (column storage)."""
            }]
        }
    })

    # L2-4: SCAL for Column/Row Scaling (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L2_scal_operation",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the SCAL operation that scales the computed off-diagonal elements by 1/sqrt(diagonal) in Cholesky factorization.

After computing:
- Upper: A[j,j+1:n-1] -= A[0:j-1,j]' * A[0:j-1,j+1:n-1]
- Lower: A[j+1:n-1,j] -= A[j+1:n-1,0:j-1] * A[j,0:j-1]'

Scale by alpha = 1/A[j,j]:
- Upper: A[j,j+1:n-1] *= alpha  (stride = lda, row elements)
- Lower: A[j+1:n-1,j] *= alpha  (stride = 1, column elements)

Requirements:
- Alpha value stored in pivots[id] = 1/A[j,j] by sqrtDiagOnward kernel
- For upper: n-j-1 elements with stride lda
- For lower: n-j-1 elements with stride 1
- Handle batched operations with strideA

Explain the numerical stability consideration: why compute 1/sqrt(t) once vs. dividing each element by sqrt(t)."""
            }]
        }
    })

    # L3-1: Complete Algorithm Trace (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_potf2_L3_algorithm_trace",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Trace the complete execution of POTF2 for a 4×4 symmetric positive definite matrix with lower triangular storage. Show all intermediate states and kernel launches.

Initial matrix A (symmetric, only lower triangle used):
[  4   *   *   * ]     [  4   .   .   . ]
[ -2   5   *   * ]     [ -2   5   .   . ]
[  1  -3   6   * ]  =  [  1  -3   6   . ]
[  3   1   2   7 ]     [  3   1   2   7 ]

For each iteration j=0,1,2,3 show:
1. DOT computation
2. sqrtDiagOnward: diagonal update
3. GEMV: off-diagonal updates
4. SCAL: scaling
5. Matrix state after iteration

Also show:
- Kernel launch configurations
- Memory access patterns (stride calculations)
- FLOPs count per iteration
- Total algorithm complexity O(n³/3)"""
            }]
        }
    })

    # L3-2: Error Handling (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_potf2_L3_error_handling",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the error handling mechanism in POTF2 for detecting non-positive definite matrices. Explain the detection strategy, error reporting, and algorithm behavior after error.

Questions to address:
1. Why is t = A[j,j] - sum(L[j,0:j-1]²) checked for t <= 0?
2. What does t < 0 mathematically indicate about the matrix?
3. Why use Fortran 1-based indexing (j+1) for error reporting?
4. Why check if(info[id] == 0) before setting info[id]?
5. What happens to subsequent diagonal elements if error occurs at j=5?
6. Why is res[id] set to 0 when error occurs?
7. What effect does SCAL with alpha=0 have on remaining elements?
8. Should the check be t <= epsilon instead of t <= 0 for numerical stability?
9. Show a matrix that becomes non-positive definite at iteration 2
10. Trace info array values for batch_count=3 with errors at different iterations"""
            }]
        }
    })

    # L3-3: Memory Workspace Management (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L3_memory_workspace",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the memory workspace calculation and allocation strategy for POTF2, including the conditional logic for small matrix optimization.

Requirements:
- Calculate 3 workspace regions: scalars, work (DOT workspace), pivots (diagonal scaling)
- Optimize for small matrices: no workspace if n <= POTF2_MAX_SMALL_SIZE(T)
- DOT workspace size: ((n-1)/512 + 2) * batch_count * sizeof(T)
- Pivots array size: batch_count * sizeof(T)
- Scalars size: 3 * sizeof(T) for alpha, unused, beta constants

Questions to address:
1. Calculate exact workspace for n=1024, batch_count=32, double precision
2. What is POTF2_MAX_SMALL_SIZE for float vs. double?
3. Why is DOT workspace size (n-1)/512 + 2 instead of just (n-1)/512?
4. How does memory usage scale: O(1) vs O(n) vs O(n²)?
5. Why can small matrices avoid allocating work and pivots?

Provide complete implementation of rocsolver_potf2_getMemorySize and allocation logic."""
            }]
        }
    })

    # L3-4: Production API (CODING)
    entries.append({
        "custom_id": "roclapack_potf2_L3_production_api",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Implement the complete production C API for POTF2 including all four precisions (S/D/C/Z), 64-bit integer variants, and proper error handling.

Requirements:
- Implement rocsolver_potf2_impl template with memory allocation
- C wrappers: spotf2, dpotf2, cpotf2, zpotf2
- 64-bit variants: spotf2_64, dpotf2_64, cpotf2_64, zpotf2_64
- Argument validation with rocblas_fill check
- Device memory size queries
- 3-region memory allocation (scalars, work, pivots)
- Scalar initialization on device

Argument checking must validate:
1. uplo == rocblas_fill_upper or rocblas_fill_lower
2. n >= 0, lda >= n, batch_count >= 0
3. Pointers: A and info

Provide complete implementation for all 8 C API functions with proper error handling and type safety."""
            }]
        }
    })

    # L3-5: Performance Analysis (ANALYSIS)
    entries.append({
        "custom_id": "roclapack_potf2_L3_performance",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": """Analyze the performance characteristics of POTF2 and compare the unblocked algorithm with blocked POTRF variants. Discuss optimization strategies for different matrix sizes and architectures.

Algorithm complexity:
- DOT operations: j elements per iteration (j=0 to n-1) → O(n²) FLOPs
- GEMV operations: j × (n-j) per iteration → O(n³) FLOPs
- SCAL operations: (n-j) per iteration → O(n²) FLOPs
- Total: ~n³/3 FLOPs

Questions to address:

1. Calculate performance metrics for n=1024, gfx942 (MI300):
   - Peak FLOP/s for POTF2 (assuming memory bandwidth limited)
   - Expected speedup with blocked POTRF
   - Memory bandwidth utilization

2. Derive the crossover point where blocked POTRF outperforms POTF2

3. Explain why lower triangular is often faster than upper on GPUs

4. Analyze batched performance: how does batch_count affect kernel launch amortization?

5. Compare complex vs. real performance: overhead of conjugation operations

6. Arithmetic intensity analysis:
   - DOT: 2j FLOPs / memory bytes ≈ 0.25 FLOP/byte
   - GEMV: 2j(n-j) FLOPs / memory bytes ≈ 0.25 FLOP/byte
   - SCAL: (n-j) FLOPs / memory bytes = 0.125 FLOP/byte

7. Blocked POTRF pipeline comparison for n=1024, nb=64:
   - Kernel launch count: POTF2 vs POTRF
   - Arithmetic intensity improvement with GEMM-based updates"""
            }]
        }
    })

    return entries

def main():
    entries = generate_entries()
    output_file = Path(__file__).parent / "roclapack_potf2.jsonl"
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    coding_tasks = sum(1 for e in entries
                      if 'implement' in e['body']['messages'][0]['content'].lower()[:200]
                      or 'write' in e['body']['messages'][0]['content'].lower()[:200]
                      or 'provide the complete' in e['body']['messages'][0]['content'].lower()[:300]
                      or 'provide complete' in e['body']['messages'][0]['content'].lower()[:300])

    print(f"Generated {len(entries)} entries")
    print(f"Output: {output_file}")
    print(f"Coding tasks: {coding_tasks}/{len(entries)} ({coding_tasks/len(entries)*100:.1f}%)")

if __name__ == "__main__":
    main()
