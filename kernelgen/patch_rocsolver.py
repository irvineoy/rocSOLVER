#!/usr/bin/env python3
"""
Script to patch CMakeLists.txt with ccache configuration
"""

import sys
from pathlib import Path

def patch_cmake_with_ccache():
    """Insert ccache configuration into CMakeLists.txt if not already present"""
    
    # Define the ccache configuration block
    ccache_config = '''# Enable ccache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Found ccache: ${CCACHE_FOUND}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_FOUND}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_FOUND}")
    
    # Enable ccache for HIP compilation
    set(CMAKE_HIP_COMPILER_LAUNCHER "${CCACHE_FOUND}")
    
    # Also set for hipcc when used as CXX compiler
    if(CMAKE_CXX_COMPILER MATCHES ".*hipcc.*")
        message(STATUS "Detected hipcc as CXX compiler - ccache enabled")
    endif()
    
    message(STATUS "ccache enabled for C/C++/HIP compilation")
else()
    message(STATUS "ccache not available - compilation will not be cached")
endif()'''
    
    # Get the path to the root CMakeLists.txt
    script_dir = Path(__file__).parent
    cmake_file = script_dir.parent / "CMakeLists.txt"
    
    if not cmake_file.exists():
        print(f"Error: CMakeLists.txt not found at {cmake_file}")
        return False
    
    # Read the current content
    with open(cmake_file, 'r') as f:
        content = f.read()
    
    # Check if ccache config already exists
    if "Enable ccache if available" in content:
        print("✓ ccache configuration already exists in CMakeLists.txt")
        return True
    
    # Find the insertion point - after the CMAKE_VERSION message
    insertion_marker = 'message(STATUS "Using CMake ${CMAKE_VERSION}")'
    
    if insertion_marker not in content:
        print(f"Warning: Could not find insertion marker: {insertion_marker}")
        print("Looking for alternative insertion point...")
        
        # Alternative: insert before project() command
        if "project(" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("project("):
                    # Insert before project line
                    lines.insert(i, ccache_config)
                    lines.insert(i+1, "")  # Add blank line after
                    content = '\n'.join(lines)
                    break
        else:
            print("Error: Could not find suitable insertion point")
            return False
    else:
        # Insert after the CMAKE_VERSION message
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if insertion_marker in line:
                # Insert after this line (with a blank line)
                lines.insert(i+1, "")
                lines.insert(i+2, ccache_config)
                content = '\n'.join(lines)
                break
    
    # Write the modified content back
    with open(cmake_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Successfully added ccache configuration to {cmake_file}")
    return True

def main():
    """Main entry point - patches CMakeLists.txt with ccache configuration"""
    success = patch_cmake_with_ccache()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()