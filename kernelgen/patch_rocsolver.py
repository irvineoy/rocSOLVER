#!/usr/bin/env python3
"""
Script to add ALL implemented functions from library/src to rocsolver-functions.h
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict, OrderedDict

def is_function_in_extern_c_block(content, func_pos):
    """Check if function at given position is inside an extern "C" block"""
    # Find all extern "C" { blocks and their corresponding closing braces
    extern_c_pattern = r'extern\s+"C"\s*\{'
    
    # Find all extern "C" blocks
    extern_blocks = []
    for match in re.finditer(extern_c_pattern, content):
        extern_start = match.end() - 1  # Position of opening brace
        
        # Find the corresponding closing brace
        brace_count = 1
        pos = extern_start + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:  # Found matching closing brace
            extern_end = pos - 1  # Position of closing brace
            extern_blocks.append((extern_start, extern_end))
    
    # Check if function position is within any extern "C" block
    for start, end in extern_blocks:
        if start < func_pos < end:
            return True
    
    return False

def extract_template_params(content, func_start_pos):
    """Extract template parameters by looking backwards from function"""
    lines = content[:func_start_pos].split('\n')
    
    # Look backwards for template declaration
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith('template'):
            # Use simple bracket counting for template params
            template_line = line
            bracket_count = 0
            start_found = False
            template_content = ""
            
            for char in template_line:
                if char == '<':
                    if not start_found:
                        start_found = True
                    bracket_count += 1
                elif char == '>':
                    bracket_count -= 1
                
                if start_found:
                    template_content += char
                    
                if start_found and bracket_count == 0:
                    # Remove the < and > brackets
                    return template_content[1:-1].strip()
            break
    
    return None

def extract_function_signatures(src_dir):
    """Extract all function signatures from library/src"""
    functions = {}
    
    for root, dirs, files in os.walk(src_dir):
        if 'CMakeFiles' in root:
            continue
            
        for file in files:
            if file.endswith(('.cpp', '.cc', '.c')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, src_dir)
                
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                    # Simple search for rocsolver function implementations
                    # Look for: rocblas_status rocsolver_functionname(params) {
                    pattern = r'rocblas_status\s+(rocsolver_\w+)\s*(\([^{]*?\))\s*\{'
                    
                    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                        func_name = match.group(1)
                        params = match.group(2)
                        
                        # Clean up parameters
                        params = re.sub(r'\s+', ' ', params).strip()
                        
                        # Check if this function has template parameters
                        template_params = extract_template_params(content, match.start())
                        
                        # Check if function is inside extern "C" block more accurately
                        is_in_extern_c = is_function_in_extern_c_block(content, match.start())
                        
                        # Determine function type
                        if template_params is not None:
                            func_type = 'template'
                        elif is_in_extern_c:
                            func_type = 'extern_c'
                        else:
                            func_type = 'standard'
                        
                        # Store the function info
                        if func_name not in functions:
                            functions[func_name] = {
                                'return_type': 'rocblas_status',
                                'params': params,
                                'file': relative_path,
                                'type': func_type,
                                'template_params': template_params
                            }
                            
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    return functions

def extract_existing_declarations(header_file):
    """Extract existing function declarations from header"""
    existing = set()
    
    with open(header_file, 'r') as f:
        content = f.read()
    
    # Match ROCSOLVER_EXPORT declarations
    pattern = r'ROCSOLVER_EXPORT\s+\w+\s+(\w+)\s*\('
    matches = re.findall(pattern, content)
    
    for func_name in matches:
        existing.add(func_name)
    
    return existing, content

def categorize_functions(functions):
    """Categorize functions by type"""
    categories = {
        'auxiliary': [],
        'lapack': [],
        'refact': [],
        'specialized': [],
        'common': [],
        'impl': [],
        'other': []
    }
    
    for func_name, info in functions.items():
        file_path = info['file']
        
        # Categorize _impl functions separately but don't skip them
        if '_impl' in func_name:
            categories['impl'].append(func_name)
        elif 'auxiliary/' in file_path:
            categories['auxiliary'].append(func_name)
        elif 'lapack/' in file_path:
            categories['lapack'].append(func_name)
        elif 'refact/' in file_path:
            categories['refact'].append(func_name)
        elif 'specialized/' in file_path:
            categories['specialized'].append(func_name)
        elif 'common/' in file_path:
            categories['common'].append(func_name)
        else:
            categories['other'].append(func_name)
    
    return categories

def generate_declaration(func_name, info):
    """Generate a proper function declaration"""
    params = info['params']
    return_type = info['return_type']
    
    # Clean up the parameters for declaration
    # Remove default values
    params = re.sub(r'=\s*[^,)]+', '', params)
    
    # For template functions, use the actual template parameters from source
    if info['type'] == 'template' and info.get('template_params'):
        template_params_str = info['template_params'].strip()
        template_decl = f"template<{template_params_str}>\n"
        declaration = f"{template_decl}ROCSOLVER_EXPORT {return_type} {func_name}{params};"
    else:
        # Regular function declaration
        declaration = f"ROCSOLVER_EXPORT {return_type} {func_name}{params};"
    
    return declaration

def update_header_file(header_file, functions_to_add):
    """Update the header file with new function declarations"""
    
    # Read existing content
    with open(header_file, 'r') as f:
        lines = f.readlines()
    
    # Separate template functions from regular C functions
    template_functions = {}
    c_functions = {}
    
    for func_name, info in functions_to_add.items():
        if info['type'] == 'template':
            template_functions[func_name] = info
        else:
            c_functions[func_name] = info
    
    # Find insertion points
    # 1. C functions go before the closing #ifdef __cplusplus (inside extern "C")
    c_insert_index = -1
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line == '#ifdef __cplusplus' and i > 100:  # Second occurrence
            c_insert_index = i
            break
    
    # 2. Template functions go after the extern "C" block (after #endif)
    template_insert_index = -1
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith('#endif') and 'ROCSOLVER_FUNCTIONS_H' in line:
            template_insert_index = i
            break
    
    if c_insert_index == -1 or template_insert_index == -1:
        print("Error: Could not find insertion points in header file")
        return False
    
    # Prepare C function declarations (go inside extern "C")
    if c_functions:
        c_declarations = []
        c_declarations.append("\n")
        c_declarations.append("// ========== ADDITIONAL C FUNCTIONS ==========\n")
        
        for func_name in sorted(c_functions.keys()):
            info = c_functions[func_name]
            declaration = generate_declaration(func_name, info)
            c_declarations.append(f"{declaration}\n")
        
        c_declarations.append("\n")
        
        # Insert C functions
        for i, line in enumerate(c_declarations):
            lines.insert(c_insert_index + i, line)
        
        # Adjust template insert index
        template_insert_index += len(c_declarations)
    
    # Prepare template function declarations (go outside extern "C")
    if template_functions:
        template_declarations = []
        template_declarations.append("\n")
        template_declarations.append("// ========== TEMPLATE FUNCTIONS ==========\n")
        template_declarations.append("// Note: Template functions must be outside extern \"C\" block\n")
        template_declarations.append("// and only available for C++ compilation\n")
        template_declarations.append("\n")
        template_declarations.append("#ifdef __cplusplus\n")
        template_declarations.append("#include <rocblas/internal/rocblas-complex-types.h>\n")
        template_declarations.append("\n")
        
        for func_name in sorted(template_functions.keys()):
            info = template_functions[func_name]
            declaration = generate_declaration(func_name, info)
            template_declarations.append(f"{declaration}\n")
        
        template_declarations.append("\n")
        template_declarations.append("#endif /* __cplusplus */\n")
        template_declarations.append("\n")
        
        # Insert template functions before final #endif
        for i, line in enumerate(template_declarations):
            lines.insert(template_insert_index + i, line)
    
    # Write back to the original header file in-place
    with open(header_file, 'w') as f:
        f.writelines(lines)
    
    return True

def main():
    src_dir = "library/src"
    header_file = "library/include/rocsolver/rocsolver-functions.h"
    
    print("=" * 80)
    print("Adding ALL implemented functions to rocsolver-functions.h")
    print("=" * 80)
    print()
    
    # Extract all implemented functions
    print("📖 Scanning source files...")
    all_functions = extract_function_signatures(src_dir)
    print(f"   Found {len(all_functions)} total functions")
    
    # Extract existing declarations
    print("\n📖 Reading existing header...")
    existing, header_content = extract_existing_declarations(header_file)
    print(f"   Found {len(existing)} existing declarations")
    
    # Find functions to add
    functions_to_add = {}
    for func_name, info in all_functions.items():
        if func_name not in existing:
            functions_to_add[func_name] = info
    
    print(f"\n📝 Functions to add: {len(functions_to_add)}")
    
    # Categorize and display
    categories = categorize_functions(functions_to_add)
    
    print("\nBreakdown by category:")
    for category, funcs in categories.items():
        if funcs:
            print(f"  • {category:15s}: {len(funcs):4d} functions")
    
    # Include ALL functions now (including _impl functions)
    impl_count = len(categories['impl'])
    functions_to_add_filtered = functions_to_add  # Don't filter out anything
    
    print(f"\n✅ Including {impl_count} _impl functions")
    print(f"✅ Will add {len(functions_to_add_filtered)} total functions")
    
    if functions_to_add_filtered:
        # Backup original header and update in place
        print("\n📝 Backing up and updating header in place...")
        backup_file = f"{header_file}.bak"
        try:
            shutil.copy2(header_file, backup_file)
            print(f"✅ Backup created: {backup_file}")
        except Exception as e:
            print(f"❌ Failed to create backup '{backup_file}': {e}")
            return 1

        updated_ok = update_header_file(header_file, functions_to_add_filtered)
        
        if updated_ok:
            print(f"✅ Updated header saved in-place: {header_file}")

            print("\nSample of added functions:")
            
            # Show examples from different categories
            categories_sample = categorize_functions(functions_to_add_filtered)
            for category, funcs in categories_sample.items():
                if funcs:
                    print(f"\n  {category.upper()} functions ({len(funcs)}):")
                    for func_name in sorted(funcs)[:3]:  # Show up to 3 per category
                        info = functions_to_add_filtered[func_name]
                        print(f"    • {func_name}")
                    if len(funcs) > 3:
                        print(f"    ... and {len(funcs) - 3} more")
            
            print("\n" + "=" * 80)
            print("NEXT STEPS:")
            print("=" * 80)
            print(f"1. Review the updated header: {header_file}")
            print(f"2. A backup was saved at: {backup_file}")
            print("3. Rebuild to ensure everything compiles correctly")
        else:
            print("❌ Failed to update header file")
    else:
        print("\n✅ No functions need to be added!")
    
    return 0

if __name__ == "__main__":
    exit(main())