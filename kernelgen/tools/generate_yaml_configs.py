#!/usr/bin/env python3
"""
Script to generate YAML configuration files for all roclapack functions
based on the example roclapack_syevd_heevd.yaml
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Maximum source files to prevent LLM context overload
MAX_SOURCE_FILES = 30

# Blacklist of generic kernels that won't help with performance optimization
# These are too generic and appear in almost all functions
GENERIC_KERNEL_BLACKLIST = {
    # Logging functions - not relevant for performance optimization
    'rocsolver_log_begin_impl',
    'rocsolver_log_end_impl',
    'rocsolver_log_flush_profile_impl',
    'rocsolver_log_restore_defaults_impl',
    'rocsolver_log_set_layer_mode_impl',
    'rocsolver_log_set_max_levels_impl',
    'rocsolver_log_write_profile_impl',

    # Very generic utility kernels that appear everywhere
    'copy_mat',  # Basic matrix copy
    'reset_info',  # Info reset
    'reset_batch_info',  # Batch info reset
    'get_array',  # Array getter
    'shift_array',  # Array shifter
    'init_ident',  # Identity matrix init
    'check_singularity',  # Generic singularity check

    # Basic arithmetic kernels too generic for specific optimization
    'axpy_kernel',  # Basic axpy operation
    'scal_kernel',  # Basic scaling
    'rot_kernel',  # Basic rotation
    'swap_kernel',  # Basic swap
    'scale_axpy',  # Combined scale and axpy

    # Generic memory/data movement operations
    'copyshift_down',
    'copyshift_left',
    'copyshift_right',
    'restore_diag',  # Diagonal restoration
    'set_diag',  # Set diagonal
    'set_offdiag',  # Set off-diagonal
    'set_zero',  # Zero out arrays

    # Other generic helpers
    'restau',  # Tau restoration
    'subtract_tau',  # Tau subtraction

    # Additional kernels identified as too generic (Code analysis results):
    'conj_in_place',  # Trivial element-wise conjugation - no optimization potential
    'copy_trans_mat',  # Basic matrix copy/transpose - very generic utility
    'set_tau',  # Trivial negation: tp[i] = -tp[i] - no optimization potential

    # Generic host-side template functions (orchestrators, not GPU kernels)
    'rocsolver_lacgv_template',  # Host function orchestrating conjugation
    'rocsolver_larf_template',   # Host function orchestrating LARF operations
    'rocsolver_larfg_template',  # Host function orchestrating LARFG operations

    # Note: We keep performance-critical kernels like:
    # - larf_left_kernel/larf_right_kernel (complex GEMV+GER with shared memory)
    # - set_taubeta/run_set_taubeta (numerical algorithms with optimization potential)
    # - Function-specific kernels with prefixes like stedc_*, bdsqr_*, getf2_*, etc.
    # - set_tridiag, set_triangular (specific to algorithms, not generic setters)
}

# Algorithm categorization for cross-contamination detection
ALGORITHM_CATEGORIES = {
    'eigenvalue': ['syev', 'heev', 'syevd', 'heevd', 'syevx', 'heevx', 'syevj', 'heevj',
                   'sytrd', 'hetrd', 'sytd2', 'hetd2', 'orgtr', 'ungtr', 'ormtr', 'unmtr',
                   'sterf', 'stedc', 'steqr', 'stedcj'],
    'svd': ['gesvd', 'gesvdx', 'gesdd', 'gesvdj', 'gebrd', 'gebd2', 'orgbr', 'ungbr', 'ormbr', 'unmbr',
            'bdsqr', 'bdsvdx'],
    'qr_decomp': ['geqrf', 'geqr2', 'orgqr', 'ungqr', 'ormqr', 'unmqr'],
    'lq_decomp': ['gelqf', 'gelq2', 'orglq', 'unglq', 'ormlq', 'unmlq'],
    'ql_decomp': ['geqlf', 'geql2', 'orgql', 'ungql', 'ormql', 'unmql'],
    'rq_decomp': ['gerqf', 'gerq2', 'orgrq', 'ungrq', 'ormrq', 'unmrq'],
    'lu_decomp': ['getrf', 'getf2', 'getri', 'getrs'],
    'cholesky': ['potrf', 'potf2', 'potri', 'potrs', 'pstrf', 'pstf2', 'posv'],
    'ldl': ['sytrf', 'sytf2', 'sytri', 'sytrs'],
    'generalized_eigen': ['sygv', 'hegv', 'sygvd', 'hegvd', 'sygvx', 'hegvx', 'sygvj', 'hegvj',
                         'sygs2', 'hegs2', 'sygst', 'hegst'],
    'triangular': ['trtri', 'trsm', 'trmm'],
    'auxiliary': ['lacgv', 'larf', 'larfb', 'larfg', 'larft', 'lasr', 'latrd', 'labrd',
                  'laswp', 'lacpy', 'laset'],
    'solvers': ['gesv', 'gels', 'posv'],
    'banded': ['geblttrf', 'geblttrs']
}

# File importance levels for prioritization
FILE_IMPORTANCE_LEVELS = {
    'core_algorithm': {
        'patterns': [r'roclapack_\w+\.cpp$', r'roclapack_\w+\.hpp$'],
        'priority': 1
    },
    'auxiliary_algorithm': {
        'patterns': [r'rocauxiliary_\w+\.hpp$'],
        'priority': 2
    },
    'specialized_kernels': {
        'patterns': [r'specialized/.*\.hpp$'],
        'priority': 3
    },
    'common_utilities': {
        'patterns': [r'lib_\w+\.hpp$', r'libcommon\.hpp$'],
        'priority': 4
    },
    'infrastructure': {
        'patterns': [r'ideal_sizes\.hpp$', r'rocsolver_logger\.hpp$', r'rocsolver_logvalue\.hpp$'],
        'priority': 5
    },
    'device_helpers': {
        'patterns': [r'device_functions\.hpp$', r'device_helpers\.hpp$'],
        'priority': 6
    }
}

def detect_algorithm_category(function_name: str) -> str:
    """
    Detect the algorithm category of a function based on its name.
    Returns the category or 'unknown' if no match found.
    """
    function_name = function_name.lower()

    for category, patterns in ALGORITHM_CATEGORIES.items():
        for pattern in patterns:
            if pattern in function_name:
                return category

    return 'unknown'

def is_cross_algorithm_contamination(current_function: str, dependency_function: str) -> bool:
    """
    Detect if a dependency function belongs to a different algorithm category
    than the current function, indicating potential cross-contamination.

    Args:
        current_function: The main function being analyzed
        dependency_function: A function found in dependencies

    Returns:
        True if cross-contamination is detected, False otherwise
    """
    current_category = detect_algorithm_category(current_function)
    dependency_category = detect_algorithm_category(dependency_function)

    # Allow unknown or auxiliary functions
    if current_category == 'unknown' or dependency_category == 'unknown':
        return False

    # Allow auxiliary functions in any algorithm
    if dependency_category == 'auxiliary':
        return False

    # Cross-contamination if different non-auxiliary categories
    return current_category != dependency_category

def filter_kernels_by_algorithm_relevance(kernels: List[str], target_function: str) -> List[str]:
    """
    Filter kernel functions to only include those relevant to the target algorithm.
    Removes kernels from different algorithm categories to prevent cross-contamination.

    Args:
        kernels: List of kernel function names
        target_function: The main function being analyzed

    Returns:
        Filtered list of relevant kernels
    """
    target_category = detect_algorithm_category(target_function)
    filtered_kernels = []

    for kernel in kernels:
        # Skip kernels that are in blacklist
        if kernel in GENERIC_KERNEL_BLACKLIST:
            continue

        # Check for cross-contamination
        if not is_cross_algorithm_contamination(target_function, kernel):
            filtered_kernels.append(kernel)
        else:
            # Log removal for debugging
            kernel_category = detect_algorithm_category(kernel)
            print(f"    Removing cross-contaminated kernel: {kernel} ({kernel_category}) from {target_function} ({target_category})")

    return filtered_kernels

def prioritize_source_files(file_paths: List[str], target_kernels: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Prioritize source files based on their importance and relevance to target kernels.
    Returns essential files first, then supporting files based on priority levels.

    Args:
        file_paths: List of source file paths
        target_kernels: List of target kernel functions

    Returns:
        Tuple of (prioritized_files, file_priorities)
    """
    file_priorities = {}
    essential_files = []
    supporting_files = []

    for file_path in file_paths:
        # Normalize path for consistent processing
        normalized_path = normalize_path(file_path)

        # Determine file importance level
        priority = 10  # Default low priority

        for _, level_info in FILE_IMPORTANCE_LEVELS.items():
            for pattern in level_info['patterns']:
                if re.search(pattern, normalized_path):
                    priority = level_info['priority']
                    break
            if priority != 10:
                break

        file_priorities[normalized_path] = priority

        # Check if file contains target kernels (essential) or is high priority
        is_essential = False
        if target_kernels and priority <= 3:  # Core, auxiliary, or specialized files
            try:
                if os.path.exists(normalized_path):
                    with open(normalized_path, 'r') as f:
                        content = f.read()

                    # Check if any target kernel is defined in this file
                    for kernel in target_kernels:
                        # Look for function definitions, not just mentions
                        patterns = [
                            rf'\b{re.escape(kernel)}\s*\(',
                            rf'ROCSOLVER_KERNEL\s+.*\b{re.escape(kernel)}\s*\(',
                            rf'__global__\s+.*\b{re.escape(kernel)}\s*\('
                        ]
                        for pattern in patterns:
                            if re.search(pattern, content):
                                is_essential = True
                                break
                        if is_essential:
                            break
            except Exception as e:
                print(f"    Warning: Could not analyze file {normalized_path}: {e}")

        # Include essential files (contains target kernels) or high-priority files (priority <= 3)
        # Priority 1: core algorithm, 2: auxiliary algorithm, 3: specialized kernels
        if is_essential or priority <= 3:  # Essential or high-priority algorithm files
            essential_files.append((normalized_path, priority))
        else:
            supporting_files.append((normalized_path, priority))

    # Sort essential files by priority (lower number = higher priority)
    essential_files.sort(key=lambda x: x[1])
    supporting_files.sort(key=lambda x: x[1])

    # Combine: essential files first, then supporting files
    prioritized_files = [f[0] for f in essential_files] + [f[0] for f in supporting_files]

    print(f"    Essential files: {len(essential_files)}, Supporting files: {len(supporting_files)}")

    return prioritized_files, file_priorities

def get_base_function_names(lapack_dir: str) -> Dict[str, List[str]]:
    """
    Get all unique base function names from the lapack directory.
    Groups related files (base, batched, etc.) together.
    Ignores files with _strided and _strided_batched suffixes.
    """
    function_groups = {}
    
    for file_path in Path(lapack_dir).glob("*.cpp"):
        filename = file_path.stem  # filename without extension
        
        # Remove the roclapack_ prefix
        if filename.startswith("roclapack_"):
            func_name = filename[10:]  # Remove "roclapack_"
            
            # Skip files with _strided or _strided_batched suffixes
            if "_strided_batched" in func_name or func_name.endswith("_strided") or func_name.endswith("_batched"):
                continue
            
            # Identify the base function name (without _batched, etc.)
            base_name = func_name
            for suffix in ["_batched", "_ptr_batched", 
                          "_interleaved_batched", "_outofplace", "_info32", 
                          "_notransv", "_inplace"]:
                if func_name.endswith(suffix):
                    base_name = func_name[:func_name.rfind(suffix)]
                    break
            
            if base_name not in function_groups:
                function_groups[base_name] = []
            
            function_groups[base_name].append(str(file_path))
    
    return function_groups

def normalize_path(path: str) -> str:
    """
    Normalize a path to remove '..' and resolve to relative form from project root.
    """
    # Convert to Path object
    path_obj = Path(path)

    # If path exists, resolve it completely first
    if path_obj.exists():
        resolved_path = path_obj.resolve()

        # Get the current working directory (project root)
        project_root = Path.cwd()

        # Try to make the path relative to project root
        try:
            relative_path = resolved_path.relative_to(project_root)
            return str(relative_path)
        except ValueError:
            # Path is not under project root, return as is but resolved
            return str(resolved_path)

    # Otherwise, normalize it as much as possible
    # This handles cases like "library/src/include/../auxiliary/file.hpp"
    parts = path.split('/')
    result = []
    for part in parts:
        if part == '..':
            if result:
                result.pop()
        elif part and part != '.':
            result.append(part)

    normalized = '/'.join(result)

    # If the path starts with /root/rocSOLVER/, make it relative
    if normalized.startswith('/root/rocSOLVER/'):
        normalized = normalized[len('/root/rocSOLVER/'):]

    return normalized

def get_include_search_paths() -> List[str]:
    """
    Get the list of directories to search for header files.
    This can be extended to read from CMake configuration or environment variables.
    """
    # Base paths relative to project root
    base_paths = [
        "library/src/include",
        "library/src/lapack",
        "library/src/auxiliary",
        "library/src/common",
        "library/src/specialized",
        "library/src/refact",
        "library/src"
    ]

    # Could be extended to read from CMakeLists.txt or environment
    # For example: parse CMAKE_INCLUDE_PATH or read from a config file

    return base_paths

def find_header_dependencies(file_paths: List[str], ignore_patterns: List[str] = None, visited: Set[str] = None) -> Set[str]:
    """
    Find all header file dependencies from the given files recursively.
    Ignores rocblas.hpp and rocsolver/rocsolver.h
    """
    if ignore_patterns is None:
        ignore_patterns = ['rocblas.hpp', 'rocsolver/rocsolver.h', 'rocblas.h', '<', '"hip/', '"rocblas/']

    if visited is None:
        visited = set()

    dependencies = set()
    include_pattern = r'#include\s+["<]([^">\n]+)[">]'

    # Get configurable include search paths
    include_search_paths = get_include_search_paths()

    for file_path in file_paths:
        # Normalize the path first
        file_path = normalize_path(file_path)

        # Skip if already visited (prevent infinite recursion)
        if file_path in visited:
            continue
        visited.add(file_path)
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Find all include statements
            includes = re.findall(include_pattern, content)

            for include in includes:
                # Check if this include should be ignored
                should_ignore = False
                for pattern in ignore_patterns:
                    if pattern in include:
                        should_ignore = True
                        break

                if not should_ignore and include.endswith('.hpp'):
                    # Try to find the full path of the header
                    if not include.startswith('/'):
                        # It's a relative include, try to find it using configured search paths
                        possible_paths = [f"{search_path}/{include}" for search_path in include_search_paths]
                        
                        for possible_path in possible_paths:
                            if os.path.exists(possible_path):
                                # Normalize the path before adding
                                normalized_path = normalize_path(possible_path)
                                dependencies.add(normalized_path)
                                # Recursively find dependencies of this header (pass visited set)
                                sub_deps = find_header_dependencies([normalized_path], ignore_patterns, visited)
                                dependencies.update(sub_deps)
                                break
        except Exception as e:
            print(f"Error analyzing dependencies in {file_path}: {e}")
    
    return dependencies

def find_kernel_functions(file_paths: List[str]) -> List[str]:
    """
    Find kernel functions including:
    1. Host-side template functions returning rocblas_status (using exclusion approach)
    2. GPU kernel functions (marked with ROCSOLVER_KERNEL, __global__, or __launch_bounds__)
    Filters out generic kernels that won't help with optimization.
    """
    kernel_functions = set()

    # Host function patterns to EXCLUDE (utility/infrastructure functions)
    HOST_FUNCTION_EXCLUSIONS = {
        'getMemorySize', 'argCheck', 'getBlockSize', 'get_blksize', 'get_innerBlkSize',
        'get_index', 'calcSweeps', 'is_', 'should_', 'can_', 'has_'
    }

    for file_path in file_paths:
        # Normalize path first
        file_path = normalize_path(file_path)

        # Only scan .hpp files for host functions (they're defined in headers)
        is_hpp_file = file_path.endswith('.hpp')

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Find ALL host-side template functions returning rocblas_status in .hpp files
            # Pattern matches: template<...> rocblas_status function_name(
            if is_hpp_file:
                host_pattern = r'template\s*<[^>]+>\s*rocblas_status\s+(\w+)\s*\('
                host_matches = re.findall(host_pattern, content, re.DOTALL)

                for match in host_matches:
                    # Exclude based on exclusion patterns
                    should_exclude = False
                    for exclusion in HOST_FUNCTION_EXCLUSIONS:
                        if exclusion in match:
                            should_exclude = True
                            break

                    # Also check against generic blacklist
                    if not should_exclude and match not in GENERIC_KERNEL_BLACKLIST:
                        kernel_functions.add(match)

            # Find GPU kernel functions with various patterns
            # Pattern 1: ROCSOLVER_KERNEL [void] function_name(
            kernel_pattern1 = r'ROCSOLVER_KERNEL\s+(?:void|__device__|__host__)*\s*(?:__launch_bounds__\([^)]+\)\s*)?(\w+)\s*\('
            kernel_matches1 = re.findall(kernel_pattern1, content)

            # Pattern 2: __global__ [void] function_name(
            kernel_pattern2 = r'__global__\s+(?:void|__device__|__host__)*\s*(?:__launch_bounds__\([^)]+\)\s*)?(\w+)\s*\('
            kernel_matches2 = re.findall(kernel_pattern2, content)

            # Pattern 3: __launch_bounds__(N) function_name(
            kernel_pattern3 = r'__launch_bounds__\s*\([^)]+\)\s*(\w+)\s*\('
            kernel_matches3 = re.findall(kernel_pattern3, content)

            # Pattern 4: template<...> ROCSOLVER_KERNEL [void] function_name(
            # Handle multi-line templates with re.DOTALL
            kernel_pattern4 = r'template\s*<[^>]+>\s*ROCSOLVER_KERNEL\s+(?:void|__device__|__host__)*\s*(?:__launch_bounds__\([^)]+\)\s*)?(\w+)\s*\('
            kernel_matches4 = re.findall(kernel_pattern4, content, re.DOTALL)

            # Pattern 5: template<...> __launch_bounds__(N) function_name(
            kernel_pattern5 = r'template\s*<[^>]+>\s*__launch_bounds__\s*\([^)]+\)\s*(\w+)\s*\('
            kernel_matches5 = re.findall(kernel_pattern5, content, re.DOTALL)

            # Pattern 6: static ROCSOLVER_KERNEL [void] function_name(
            kernel_pattern6 = r'static\s+ROCSOLVER_KERNEL\s+(?:void|__device__|__host__)*\s*(?:__launch_bounds__\([^)]+\)\s*)?(\w+)\s*\('
            kernel_matches6 = re.findall(kernel_pattern6, content)

            # Pattern 7: inline __device__ function_name(
            kernel_pattern7 = r'inline\s+__device__\s+(?:\w+\s+)*(\w+)\s*\('
            kernel_matches7 = re.findall(kernel_pattern7, content)

            # Combine all matches
            all_matches = (kernel_matches1 + kernel_matches2 + kernel_matches3 +
                          kernel_matches4 + kernel_matches5 + kernel_matches6 + kernel_matches7)

            # Add all GPU kernel functions (filter against blacklist)
            for match in all_matches:
                # Skip common utility names and type keywords that might be captured
                if (match not in ['void', 'static', '__device__', '__host__', 'inline',
                                 'const', 'unsigned', 'int', 'float', 'double'] and
                    match not in GENERIC_KERNEL_BLACKLIST):
                    kernel_functions.add(match)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Sort for consistent ordering
    return sorted(list(kernel_functions))

def get_test_filter(base_name: str) -> str:
    """
    Generate test filter pattern based on the base function name.
    """
    # Convert function name to uppercase for test filter
    # Handle special cases where function names have multiple parts
    parts = base_name.upper().split('_')
    
    # For functions like syevd_heevd, create pattern like *SYEVD.*:*HEEVD.*
    if len(parts) == 2 and parts[0] != parts[1]:
        return f'"*{parts[0]}.*:*{parts[1]}.*"'
    elif '_' in base_name:
        # For other multi-part names, use the full name
        return f'"*{base_name.upper()}.*"'
    else:
        return f'"*{base_name.upper()}.*"'

def get_bench_function_name(base_name: str) -> str:
    """
    Get the benchmark function name (usually the first part before underscore).
    """
    # For functions like syevd_heevd, use just syevd
    # For functions like getrf, use getrf
    parts = base_name.split('_')
    return parts[0]

def generate_yaml_config(base_name: str, files: List[str], existing_yaml_path: str = None) -> Dict:
    """
    Generate YAML configuration for a specific function group.
    If existing_yaml_path is provided and exists, preserve its commands.
    """
    # Check if YAML file already exists and load it
    existing_config = None
    if existing_yaml_path and os.path.exists(existing_yaml_path):
        try:
            with open(existing_yaml_path, 'r') as f:
                existing_config = yaml.safe_load(f)
            print(f"  Loading existing config from {existing_yaml_path}")
        except Exception as e:
            print(f"  Warning: Could not load existing YAML: {e}")
            existing_config = None
    
    # Determine source and gen file paths
    source_files = []
    gen_files = []

    # Find the main cpp and hpp files
    base_cpp = f"library/src/lapack/roclapack_{base_name}.cpp"
    base_hpp = f"library/src/lapack/roclapack_{base_name}.hpp"

    if os.path.exists(base_cpp):
        source_files.append(normalize_path(base_cpp))
        gen_files.append(normalize_path(base_cpp))

    if os.path.exists(base_hpp):
        source_files.append(normalize_path(base_hpp))
        gen_files.append(normalize_path(base_hpp))

    # If no base files found, use the first available file
    if not source_files and files:
        for file in files:
            if file.endswith('.cpp'):
                source_files.append(normalize_path(file))
                gen_files.append(normalize_path(file))
                # Check for corresponding hpp
                hpp_file = file.replace('.cpp', '.hpp')
                if os.path.exists(hpp_file):
                    source_files.append(normalize_path(hpp_file))
                    gen_files.append(normalize_path(hpp_file))
                break

    # Find all header dependencies (includes recursive search)
    dependencies = find_header_dependencies(source_files)

    # Look for auxiliary function dependencies in all source files
    # This is more generic than checking specific base_names
    auxiliary_dir = "library/src/auxiliary"
    for src_file in source_files:
        if os.path.exists(src_file):
            try:
                with open(src_file, 'r') as f:
                    content = f.read()
                # Look for references to auxiliary functions in includes
                aux_pattern = r'rocauxiliary_(\w+)\.hpp'
                aux_matches = re.findall(aux_pattern, content)
                for match in aux_matches:
                    aux_file = f"{auxiliary_dir}/rocauxiliary_{match}.hpp"
                    if os.path.exists(aux_file):
                        dependencies.add(normalize_path(aux_file))
            except:
                pass

    # Normalize all dependency paths and remove duplicates
    normalized_deps = {normalize_path(dep) for dep in dependencies}

    # Add dependencies to source files (for analysis) but not to gen files
    # Use set to remove duplicates, then convert back to list
    all_source_files_set = set(normalize_path(f) for f in source_files)
    all_source_files_set.update(normalized_deps)

    # Remove any paths with '..' that couldn't be resolved
    all_source_files_set = {f for f in all_source_files_set if '..' not in f}

    all_source_files = list(all_source_files_set)
    all_source_files.sort()  # Sort for consistent ordering
    
    # Find kernel functions in all source files including dependencies
    # We search in all_source_files to include GPU kernels from dependencies
    all_kernels = find_kernel_functions(all_source_files)

    # Apply intelligent filtering
    print(f"  Found {len(all_kernels)} total kernels before filtering")

    # Filter kernels by algorithm relevance to prevent cross-contamination
    target_functions = filter_kernels_by_algorithm_relevance(all_kernels, base_name)
    print(f"  Filtered to {len(target_functions)} relevant kernels")

    # Prioritize source files based on importance and kernel relevance
    prioritized_source_files, _ = prioritize_source_files(all_source_files, target_functions)

    # Use prioritized files for source_file_path
    all_source_files = prioritized_source_files

    # Limit source files for LLM context optimization with smarter selection
    if len(all_source_files) > MAX_SOURCE_FILES:
        print(f"  Limiting source files from {len(all_source_files)} to {MAX_SOURCE_FILES} for LLM optimization")

        # Smarter file selection: prioritize files containing target kernels
        kernel_containing_files = set()
        for kernel in target_functions:
            for source_file in all_source_files:
                try:
                    if os.path.exists(source_file):
                        with open(source_file, 'r') as f:
                            content = f.read()
                            # Check if this file contains the kernel definition
                            if re.search(rf'\b{re.escape(kernel)}\s*\(', content):
                                kernel_containing_files.add(source_file)
                except:
                    pass

        # Keep all kernel-containing files plus top priority files up to MAX_SOURCE_FILES
        essential_files = []
        supporting_files = []

        for f in all_source_files:
            if f in kernel_containing_files:
                essential_files.append(f)
            else:
                supporting_files.append(f)

        # Calculate how many supporting files we can include
        remaining_slots = MAX_SOURCE_FILES - len(essential_files)
        if remaining_slots > 0:
            all_source_files = essential_files + supporting_files[:remaining_slots]
        else:
            # If too many kernel-containing files, keep only the most essential ones
            all_source_files = essential_files[:MAX_SOURCE_FILES]

        print(f"  Selected {len(essential_files)} kernel-containing files and {min(remaining_slots, len(supporting_files))} supporting files")
    
    # If existing config exists, preserve commands and only update paths/functions
    if existing_config:
        config = existing_config.copy()
        # Only override these three fields
        config['source_file_path'] = all_source_files
        config['gen_file_path'] = all_source_files
        config['target_kernel_functions'] = target_functions
        print(f"  Updated paths and functions, preserved existing commands")
    else:
        # Create new config with default values
        # Generate test filter
        test_filter = get_test_filter(base_name)
        
        # Generate bench function name
        bench_func = get_bench_function_name(base_name)
        
        config = {
            'source_file_path': all_source_files,  # Include dependencies for analysis
            'gen_file_path': all_source_files, 
            'target_kernel_functions': target_functions,
            'compile_command': [
                './install.sh --architecture gfx942 --clients --relwithdebinfo'
            ],
            'correctness_command': [
                f'build/release-debug/clients/staging/rocsolver-test --gtest_filter={test_filter}'
            ],
            'performance_command': [
                f'rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -b SQ -b TCP -b TCC -- build/release-debug/clients/staging/rocsolver-bench -f {bench_func} -r s -m 3000 -n 3000 --lda 3000 --iters 2',
                'rocprof-compute analyze --path rocprof_compute_profile -b 2'
            ]
        }
        print(f"  Created new config with default commands")
    
    return config

def main():
    lapack_dir = "library/src/lapack"
    # Output YAML files to parent kernelgen directory (not tools/)
    output_dir = "kernelgen"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all function groups
    function_groups = get_base_function_names(lapack_dir)
    
    print(f"Found {len(function_groups)} unique function groups")
    
    # Generate YAML config for each function group
    generated_count = 0
    updated_count = 0
    for base_name, files in function_groups.items():
        output_file = os.path.join(output_dir, f"roclapack_{base_name}.yaml")
        
        # Check if file already exists
        file_exists = os.path.exists(output_file)
        
        if file_exists:
            print(f"Updating {base_name} - preserving existing commands")
            config = generate_yaml_config(base_name, files, output_file)
            updated_count += 1
        else:
            print(f"Creating new config for {base_name}")
            config = generate_yaml_config(base_name, files)
            generated_count += 1
        
        # Only write if we have valid source files
        if config['source_file_path']:
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                         allow_unicode=True, width=1000)
            if file_exists:
                print(f"Updated: {output_file}")
            else:
                print(f"Generated: {output_file}")
        else:
            print(f"Skipping {base_name} - no source files found")
    
    print(f"\nSummary:")
    print(f"  Generated {generated_count} new YAML configuration files")
    print(f"  Updated {updated_count} existing YAML configuration files")

if __name__ == "__main__":
    main()