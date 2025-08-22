#!/usr/bin/env python3
"""
Script to generate YAML configuration files for all roclapack functions
based on the example roclapack_syevd_heevd.yaml
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Set

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

def find_header_dependencies(file_paths: List[str], ignore_patterns: List[str] = None) -> Set[str]:
    """
    Find all header file dependencies from the given files.
    Ignores rocblas.hpp and rocsolver/rocsolver.h
    """
    if ignore_patterns is None:
        ignore_patterns = ['rocblas.hpp', 'rocsolver/rocsolver.h', 'rocblas.h', '<', '"hip/', '"rocblas/']
    
    dependencies = set()
    include_pattern = r'#include\s+["<]([^">\n]+)[">]'
    
    for file_path in file_paths:
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
                        # It's a relative include, try to find it
                        possible_paths = [
                            f"library/src/include/{include}",
                            f"library/src/lapack/{include}",
                            f"library/src/auxiliary/{include}",
                            f"library/src/common/{include}",
                            f"library/src/specialized/{include}",
                            f"library/src/refact/{include}",
                            f"library/src/{include}"
                        ]
                        
                        for possible_path in possible_paths:
                            if os.path.exists(possible_path):
                                dependencies.add(possible_path)
                                # Recursively find dependencies of this header # TODO: no recursion for now
                                # sub_deps = find_header_dependencies([possible_path], ignore_patterns)
                                # dependencies.update(sub_deps)
                                break
        except Exception as e:
            print(f"Error analyzing dependencies in {file_path}: {e}")
    
    return dependencies

def find_kernel_functions(file_paths: List[str]) -> List[str]:
    """
    Find kernel functions (impl and template functions) in cpp and hpp files.
    """
    kernel_functions = set()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Find all rocsolver_*_impl functions
            impl_pattern = r'rocsolver_(\w+)_impl'
            impl_matches = re.findall(impl_pattern, content)
            
            # Find all rocsolver_*_template functions
            template_pattern = r'rocsolver_(\w+)_template'
            template_matches = re.findall(template_pattern, content)
            
            # Add found functions (without _impl suffix for function fields)
            for match in impl_matches:
                kernel_functions.add(f"rocsolver_{match}_impl")
            for match in template_matches:
                kernel_functions.add(f"rocsolver_{match}_template")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return list(kernel_functions)

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
        source_files.append(base_cpp)
        gen_files.append(base_cpp)
    
    if os.path.exists(base_hpp):
        source_files.append(base_hpp)
        gen_files.append(base_hpp)
    
    # If no base files found, use the first available file
    if not source_files and files:
        for file in files:
            if file.endswith('.cpp'):
                source_files.append(file)
                gen_files.append(file)
                # Check for corresponding hpp
                hpp_file = file.replace('.cpp', '.hpp')
                if os.path.exists(hpp_file):
                    source_files.append(hpp_file)
                    gen_files.append(hpp_file)
                break
    
    # Find all header dependencies
    dependencies = find_header_dependencies(source_files)
    
    # Add dependencies to source files (for analysis) but not to gen files
    # Use set to remove duplicates, then convert back to list
    all_source_files_set = set(source_files)
    all_source_files_set.update(dependencies)
    all_source_files = list(all_source_files_set)
    all_source_files.sort()  # Sort for consistent ordering
    
    # Find kernel functions in all source files including dependencies
    target_functions = find_kernel_functions(source_files) # TODO: could also try all_source_files
    
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
                f'rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -- build/release-debug/clients/staging/rocsolver-bench -f {bench_func} -r s -m 3000 -n 3000 --lda 3000 --iters 2',
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