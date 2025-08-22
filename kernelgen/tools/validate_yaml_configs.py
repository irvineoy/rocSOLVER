#!/usr/bin/env python3
"""
Script to validate YAML configuration files in kernelgen directory.
Checks that source files exist and target kernel functions are present in those files.
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return os.path.exists(file_path)

def find_functions_in_file(file_path: str) -> List[str]:
    """Find all function definitions in a C++ file."""
    functions = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find all rocsolver_*_impl and rocsolver_*_template functions
        # These are the main kernel functions we're interested in
        
        # Pattern for impl functions
        impl_pattern = r'rocsolver_(\w+)_impl'
        impl_matches = re.findall(impl_pattern, content)
        for match in impl_matches:
            functions.add(f"rocsolver_{match}_impl")
        
        # Pattern for template functions
        template_pattern = r'rocsolver_(\w+)_template'
        template_matches = re.findall(template_pattern, content)
        for match in template_matches:
            functions.add(f"rocsolver_{match}_template")
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return list(functions)

def validate_yaml_file(yaml_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a single YAML configuration file.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load YAML: {e}"]
    
    # Check source_file_path
    source_files = config.get('source_file_path', [])
    if not source_files:
        errors.append("No source_file_path specified")
    else:
        for source_file in source_files:
            if not check_file_exists(source_file):
                errors.append(f"Source file does not exist: {source_file}")
    
    # Check target_kernel_functions
    target_functions = config.get('target_kernel_functions', [])
    if not target_functions:
        errors.append("No target_kernel_functions specified")
    else:
        # Check if functions exist in source files (both cpp and hpp)
        all_functions = []
        for source_file in source_files:
            if check_file_exists(source_file):
                found_functions = find_functions_in_file(source_file)
                all_functions.extend(found_functions)
        
        # Check each target function
        for target_func in target_functions:
            if target_func not in all_functions:
                # Try to find partial matches
                partial_matches = [f for f in all_functions if target_func in f or f in target_func]
                if partial_matches:
                    errors.append(f"Function '{target_func}' not found exactly, but found similar: {partial_matches}")
                else:
                    errors.append(f"Target function not found in source files: {target_func}")
    
    return len(errors) == 0, errors

def main():
    # YAML files are in parent kernelgen directory
    kernelgen_dir = "kernelgen"
    
    # Get all YAML files
    yaml_files = list(Path(kernelgen_dir).glob("roclapack_*.yaml"))
    
    print(f"Validating {len(yaml_files)} YAML configuration files...\n")
    
    valid_count = 0
    invalid_count = 0
    all_issues = {}
    
    for yaml_file in sorted(yaml_files):
        is_valid, errors = validate_yaml_file(str(yaml_file))
        
        if is_valid:
            valid_count += 1
            print(f"✓ {yaml_file.name}")
        else:
            invalid_count += 1
            print(f"✗ {yaml_file.name}")
            all_issues[yaml_file.name] = errors
            for error in errors:
                print(f"  - {error}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Validation Summary:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {invalid_count}")
    print(f"  Total files: {len(yaml_files)}")
    
    if invalid_count > 0:
        print(f"\nFiles with issues:")
        for filename in sorted(all_issues.keys()):
            print(f"  - {filename}")
    
    return invalid_count == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)