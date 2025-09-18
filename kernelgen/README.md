

# KernelGen for rocSOLVER Agentic Optimization 

This directory contains the necessary tools and configurations to integrate the rocSOLVER library into an LLM-based **agentic optimization workflow**. The primary goal is to enable the targeted optimization of individual HIP kernels within the complex rocSOLVER project, providing high-quality performance feedback to a code-generating agent.

## The Challenge

The rocSOLVER library, like many large HPC projects, presents several challenges for kernel-level optimization:

  * **Monolithic Build:** It compiles into a single shared library (`librocsolver.so`) and a few large binaries (`rocsolver-bench`, `rocsolver-test`), hiding internal implementation details.
  * **Complex Dependencies:** Internal functions have intricate dependencies, making it extremely difficult to isolate a single kernel and its dependencies for a standalone build.
  * **Slow Compilation:** A full build from scratch can take dozens of minutes, which is too slow for a rapid, iterative optimization loop.

## Solution: Incremental Compilation & Targeted Execution

To overcome these challenges, we adopt a strategy that avoids kernel isolation and instead focuses on rapidly recompiling the entire project.

1.  **Incremental Compilation with `ccache`**: We use `ccache` to drastically speed up subsequent builds. After the first full compilation, changes to a single kernel file will only trigger a rebuild of that file and its dependents. This reduces compilation time from **minutes to seconds**, making an iterative workflow feasible. 

2.  **Configuration-Driven Workflow**: The entire process for a given kernel is defined by a `.yaml` configuration file. This file tells the optimization agent everything it needs to know: where the source code is, what functions to optimize, and how to compile and test the new version.

3.  **Targeted Execution**: Although we rebuild the entire `rocsolver-test` and `rocsolver-bench` binaries, the configuration file specifies the exact commands needed to run **only the tests and benchmarks relevant to the optimized kernel**. This allows us to gather precise correctness and performance data for the specific kernel being tuned.


## Directory Structure

  * `*.yaml`: Configuration files for each kernel optimization task. Each file defines a self-contained optimization target. There are `48` kernels in total. '_stride' and '_batched' kernels are ignored since their performance depends on the base kernel.
  * `patch_rocsolver.py`: A Python script that modifies the project's CMake configuration to use `ccache` for faster builds.
  * `setup.sh`: A shell script to handle dependencies, patch rocsolver with ccache, and conduct the initial compilation
  * `tools/`: Necessary scripts to build and verify the kernel configs. **Please run those scripts from the root directory.**




## Configuration File Explained

Each `.yaml` file defines a complete optimization task. Here is a breakdown of the fields:

  * `source_file_path: List[str]`
    The path(s) to the original source file(s) containing the kernel to be optimized. This is the code that will be fed to the LLM.

  * `gen_file_path: List[str]`
    The path(s) where the LLM-generated code should be saved. In most of cases, this will be the same as `source_file_path`, overwriting the original.

  * `target_kernel_functions: List[str]`
    The specific function(s) within the source files that the LLM should focus on optimizing.

  * `compile_command: List[str]`
    The command(s) required to compile the entire project after the source code has been modified.

  * `correctness_command: List[str]`
    The command(s) to run the relevant correctness tests for the modified kernel. We use filters (e.g., `--gtest_filter`) to isolate the specific tests.

  * `performance_command: List[str]`
    The command(s) to benchmark the performance of the modified kernel. This typically involves wrapping a call to `rocsolver-bench` with a profiling tool like `rocprof`.


## Version: V1



 ⚠️ Major Issues Found

  1. Over-Broad Dependency Resolution

  Problem: Complex functions like GESDD, GESVDX are pulling in unrelated eigenvalue kernels
  - GESDD (SVD) has 95 kernels including 21 eigenvalue kernels (stedc_*, syevj_*)
  - This happens because recursive dependency search includes too many headers

  2. Additional Generic Kernels Need Blacklisting

  Problem: Many very generic kernels still appear in most files:
  - conj_in_place (34 files) - conjugation helper
  - rocsolver_lacgv_template (34 files) - vector conjugation
  - copy_trans_mat (27 files) - matrix transpose copy
  - larf_left_kernel/larf_right_kernel (27 files) - generic Householder reflectors
  - set_taubeta, set_tau (29/23 files) - generic tau setters

  📋 Recommendations for Improvement

  High Priority Fixes:

  1. Expand Blacklist - Add these generic kernels:
  # Add to GENERIC_KERNEL_BLACKLIST:
  'conj_in_place',       # Conjugation helper
  'copy_trans_mat',      # Transpose copy (too generic)
  'larf_left_kernel',    # Generic Householder left
  'larf_right_kernel',   # Generic Householder right  
  'set_tau',             # Generic tau setter
  'set_taubeta',         # Generic tau/beta setter
  'rocsolver_lacgv_template',  # Vector conjugation template
  'rocsolver_larf_template',   # Generic LARF template
  'rocsolver_larfg_template',  # Generic Householder generator

  2. Improve Dependency Filtering - Modify script to:
    - Stop recursive dependency search at certain function boundaries
    - Don't include eigenvalue headers for pure SVD/factorization functions
    - Add function-category awareness to dependency resolution
  3. Function-Specific Kernel Filtering - Keep only kernels relevant to the main operation:
    - SVD functions: Keep bdsqr_*, exclude stedc_*/syevj_*
    - Factorization: Keep getf2_*, potf2_*, exclude eigenvalue kernels
    - Eigenvalue: Keep stedc_*, syevj_*, exclude SVD kernels

  Medium Priority Improvements:

  4. Source File Optimization - Some overly common includes:
    - rocsolver_run_specialized_kernels.hpp appears in all 46 files (might be too broad)
    - Consider if some auxiliary headers are needed for all functions
  5. Performance Command Optimization - Standardize matrix sizes:
    - Most use -n 4096 or -m 4096 -n 4096
    - Some use --lda 3000 while others don't specify
    - Consider consistent parameters for fair comparisons

  🎯 Expected Impact of Fixes

  After implementing the expanded blacklist and dependency filtering:
  - Kernel count reduction: Functions like GESDD should drop from 95 → ~40-50 kernels
  - Better LLM context: More focused on performance-critical, function-specific kernels
  - Reduced noise: Remove generic operations that appear everywhere
  - Cleaner grouping: Clear separation between SVD, eigenvalue, and factorization kernels