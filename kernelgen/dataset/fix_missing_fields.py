#!/usr/bin/env python3
"""
Fix JSONL files that are missing required fields:
- Rename 'query' to 'instruction'
- Add 'context_text' based on code_blocks and interface
- Add 'rationale' based on answer
- Add 'tags' based on level, interface, and content analysis
"""
import json
import sys
import re

def infer_tags(entry):
    """Infer tags from entry content"""
    tags = []

    interface = entry.get('interface', '').lower()
    level = entry.get('level', '')
    answer = entry.get('answer', '').lower()
    instruction = entry.get('instruction', '').lower()

    # Add interface tag
    if interface:
        tags.append(interface)

    # Check for coding-related content
    if any(word in instruction for word in ['implement', 'write', 'code', 'function', 'kernel']):
        tags.append('coding')
    if any(word in answer for word in ['```cpp', '```c', '```hip', 'implementation:']):
        tags.append('coding')

    # Algorithm-related tags
    if any(word in answer for word in ['algorithm', 'approach', 'method', 'procedure']):
        tags.append('algorithm')

    # Performance tags
    if any(word in answer for word in ['performance', 'optimize', 'efficient', 'speedup', 'flop']):
        tags.append('performance')

    # Memory tags
    if any(word in answer for word in ['workspace', 'memory', 'allocation', 'buffer']):
        tags.append('workspace')
    if 'memory' in answer and 'management' in answer:
        tags.append('memory-management')

    # Error handling
    if any(word in answer for word in ['error', 'validation', 'invalid', 'check']):
        tags.append('error-handling')

    # API-related
    if any(word in answer for word in ['api', 'interface', 'parameter', 'argument']):
        tags.append('api')

    # Batched execution
    if any(word in answer for word in ['batch', 'batched', 'strided']):
        tags.append('batched-execution')

    # Level-specific tags
    if level == 'L1':
        tags.append('single-function')
    elif level == 'L2':
        tags.append('subsystem')
    elif level == 'L3':
        tags.append('interface-level')

    # Analysis vs implementation
    if 'explain' in instruction or 'describe' in instruction or 'what' in instruction:
        tags.append('analysis')

    # Remove duplicates and return
    return sorted(list(set(tags)))

def generate_context_text(entry):
    """Generate context_text from code_blocks and interface"""
    interface = entry.get('interface', 'this routine')
    code_blocks = entry.get('code_blocks', [])

    if not code_blocks:
        return f"This question relates to the {interface.upper()} routine in rocSOLVER."

    # Extract file paths and create context
    files = [block.get('path', '').split('/')[-1] for block in code_blocks if block.get('path')]
    if files:
        file_list = ', '.join(set(files))
        return f"The {interface.upper()} routine uses code from {file_list}. Understanding this implementation is essential for working with {interface.upper()}."

    return f"This question examines the {interface.upper()} implementation in rocSOLVER."

def generate_rationale(entry):
    """Generate rationale from answer"""
    answer = entry.get('answer', '')
    instruction = entry.get('instruction', '')

    # Extract key points from answer
    answer_lines = [line.strip() for line in answer.split('\n') if line.strip() and not line.strip().startswith('```')]

    if not answer_lines:
        return "The answer is derived from the implementation details shown in the code blocks."

    # Take first substantial sentence/paragraph as rationale
    for line in answer_lines[:3]:
        if len(line) > 50 and not line.startswith('-'):
            # Clean up markdown formatting
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            return clean_line[:300] + ('...' if len(clean_line) > 300 else '')

    return "The answer is based on the actual implementation in the rocSOLVER codebase as shown in the code excerpts."

def fix_entry(entry):
    """Fix a single entry by adding missing fields"""
    fixed = entry.copy()

    # Rename query to instruction
    if 'query' in fixed:
        fixed['instruction'] = fixed.pop('query')

    # Add missing fields if not present
    if 'context_text' not in fixed or not fixed.get('context_text'):
        fixed['context_text'] = generate_context_text(fixed)

    if 'rationale' not in fixed or not fixed.get('rationale'):
        fixed['rationale'] = generate_rationale(fixed)

    if 'tags' not in fixed or not fixed.get('tags'):
        fixed['tags'] = infer_tags(fixed)

    return fixed

def fix_jsonl_file(input_path, output_path=None):
    """Fix a JSONL file"""
    if output_path is None:
        output_path = input_path

    entries = []
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                fixed_entry = fix_entry(entry)
                entries.append(fixed_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue

    # Write back
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    return len(entries)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_missing_fields.py <file1.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        try:
            count = fix_jsonl_file(filepath)
            print(f"✓ Fixed {filepath}: {count} entries")
        except Exception as e:
            print(f"✗ Error fixing {filepath}: {e}", file=sys.stderr)
