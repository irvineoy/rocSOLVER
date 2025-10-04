#!/usr/bin/env python3
import json
import os
import glob
from collections import defaultdict

def validate_entry(entry, filename):
    """Validate a single entry against the correct DATASET.md schema"""
    errors = []

    # Check required fields (correct schema)
    required_fields = ['id', 'level', 'interface', 'instruction', 'context_text',
                      'code_blocks', 'answer', 'rationale', 'tags']
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    # Validate level
    if 'level' in entry and entry['level'] not in ['L1', 'L2', 'L3']:
        errors.append(f"Invalid level: {entry['level']}")

    # Validate tags
    if 'tags' in entry:
        if not isinstance(entry['tags'], list):
            errors.append("tags must be a list")

    # Validate code_blocks
    if 'code_blocks' in entry:
        if not isinstance(entry['code_blocks'], list):
            errors.append("code_blocks must be a list")
        else:
            for i, block in enumerate(entry['code_blocks']):
                if not isinstance(block, dict):
                    errors.append(f"code_blocks[{i}] must be a dict")
                    continue

                required_block_fields = ['path', 'language', 'content']
                for field in required_block_fields:
                    if field not in block:
                        errors.append(f"code_blocks[{i}] missing field: {field}")

                # Check that we're using 'content', not 'start_line'/'end_line'
                if 'start_line' in block or 'end_line' in block:
                    errors.append(f"code_blocks[{i}] uses deprecated start_line/end_line instead of content")

    return errors

def count_chars(text):
    """Count characters in text"""
    if isinstance(text, str):
        return len(text)
    return 0

def count_tokens(text):
    """Estimate tokens (characters / 4)"""
    return count_chars(text) / 4

def analyze_entry(entry):
    """Analyze an entry and return statistics"""
    stats = {
        'chars_instruction': count_chars(entry.get('instruction', '')),
        'chars_context': count_chars(entry.get('context_text', '')),
        'chars_answer': count_chars(entry.get('answer', '')),
        'chars_rationale': count_chars(entry.get('rationale', '')),
        'chars_code': 0,
        'num_code_blocks': len(entry.get('code_blocks', []))
    }

    for block in entry.get('code_blocks', []):
        stats['chars_code'] += count_chars(block.get('content', ''))

    stats['chars_total'] = (stats['chars_instruction'] + stats['chars_context'] +
                           stats['chars_answer'] + stats['chars_rationale'] + stats['chars_code'])
    stats['tokens_total'] = stats['chars_total'] / 4

    return stats

def main():
    dataset_dir = '/root/rocSOLVER/kernelgen/dataset'
    output_file = '/root/rocSOLVER/kernelgen/dataset.jsonl'

    # Find all JSONL files
    jsonl_files = sorted(glob.glob(os.path.join(dataset_dir, '*.jsonl')))

    print(f"Found {len(jsonl_files)} JSONL files\n")

    # Statistics
    total_entries = 0
    total_chars = 0
    total_tokens = 0
    level_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    interface_counts = defaultdict(int)
    file_stats = []
    all_entries = []
    validation_errors = []

    # Process each file
    for jsonl_file in jsonl_files:
        filename = os.path.basename(jsonl_file)
        print(f"Processing {filename}...")

        file_entry_count = 0
        file_chars = 0
        file_level = defaultdict(int)
        file_tags = defaultdict(int)

        try:
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Validate entry
                        errors = validate_entry(entry, filename)
                        if errors:
                            validation_errors.append({
                                'file': filename,
                                'line': line_num,
                                'errors': errors
                            })

                        # Analyze entry
                        stats = analyze_entry(entry)

                        # Update statistics
                        total_entries += 1
                        file_entry_count += 1
                        total_chars += stats['chars_total']
                        total_tokens += stats['tokens_total']

                        level = entry.get('level', 'UNKNOWN')
                        level_counts[level] += 1
                        file_level[level] += 1

                        interface = entry.get('interface', 'UNKNOWN')
                        interface_counts[interface] += 1

                        # Count tags
                        for tag in entry.get('tags', []):
                            tag_counts[tag] += 1
                            file_tags[tag] += 1

                        file_chars += stats['chars_total']

                        # Add to combined list
                        all_entries.append(entry)

                    except json.JSONDecodeError as e:
                        validation_errors.append({
                            'file': filename,
                            'line': line_num,
                            'errors': [f"JSON decode error: {e}"]
                        })

        except Exception as e:
            print(f"  ERROR reading file: {e}")
            continue

        file_stats.append({
            'filename': filename,
            'entries': file_entry_count,
            'chars': file_chars,
            'tokens': file_chars / 4,
            'level': dict(file_level),
            'tags': dict(file_tags)
        })

        print(f"  Entries: {file_entry_count}, Chars: {file_chars:,}, Tokens: {file_chars/4:,.0f}")

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    if validation_errors:
        print(f"\n⚠️  Found {len(validation_errors)} validation errors:\n")
        for error_info in validation_errors[:10]:  # Show first 10
            print(f"  {error_info['file']}:{error_info['line']}")
            for error in error_info['errors']:
                print(f"    - {error}")
            print()

        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors")
    else:
        print("\n✓ All entries passed schema validation!")

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    print(f"\nTotal files processed: {len(jsonl_files)}")
    print(f"Total entries: {total_entries:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens (estimated): {total_tokens:,.0f}")
    print(f"Average chars per entry: {total_chars/total_entries if total_entries > 0 else 0:,.0f}")
    print(f"Average tokens per entry: {total_tokens/total_entries if total_entries > 0 else 0:,.0f}")

    print("\n" + "-"*80)
    print("LEVEL DISTRIBUTION")
    print("-"*80)
    for level in ['L1', 'L2', 'L3', 'UNKNOWN']:
        count = level_counts[level]
        if count > 0:
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            print(f"  {level}: {count:4d} ({percentage:5.1f}%)")

    print("\n" + "-"*80)
    print("TOP 20 TAGS")
    print("-"*80)
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (tag, count) in enumerate(sorted_tags, 1):
        percentage = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"  {i:2d}. {tag:30s} {count:4d} ({percentage:5.1f}%)")

    # Count coding tasks
    coding_count = tag_counts.get('coding', 0)
    print(f"\n  Coding tasks: {coding_count}/{total_entries} ({coding_count*100/total_entries if total_entries > 0 else 0:.1f}%)")

    print("\n" + "-"*80)
    print("TOP 10 INTERFACES")
    print("-"*80)
    sorted_interfaces = sorted(interface_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (interface, count) in enumerate(sorted_interfaces, 1):
        print(f"  {i:2d}. {interface:30s} {count:4d} entries")

    print("\n" + "-"*80)
    print("TOP 10 FILES BY ENTRY COUNT")
    print("-"*80)
    sorted_files = sorted(file_stats, key=lambda x: x['entries'], reverse=True)[:10]
    for i, fstat in enumerate(sorted_files, 1):
        print(f"  {i:2d}. {fstat['filename']:50s} {fstat['entries']:3d} entries")

    print("\n" + "-"*80)
    print("TOP 10 FILES BY TOKEN COUNT")
    print("-"*80)
    sorted_files = sorted(file_stats, key=lambda x: x['tokens'], reverse=True)[:10]
    for i, fstat in enumerate(sorted_files, 1):
        print(f"  {i:2d}. {fstat['filename']:50s} {fstat['tokens']:8,.0f} tokens")

    # Write combined dataset
    print("\n" + "="*80)
    print(f"Writing combined dataset to {output_file}")
    print("="*80)

    with open(output_file, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✓ Successfully wrote {len(all_entries)} entries to {output_file}")

    # Write detailed report
    report_file = '/root/rocSOLVER/kernelgen/dataset_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROCSOLVER DATASET REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total files: {len(jsonl_files)}\n")
        f.write(f"Total entries: {total_entries:,}\n")
        f.write(f"Total characters: {total_chars:,}\n")
        f.write(f"Total tokens (estimated): {total_tokens:,.0f}\n")
        f.write(f"Average chars per entry: {total_chars/total_entries if total_entries > 0 else 0:,.0f}\n")
        f.write(f"Average tokens per entry: {total_tokens/total_entries if total_entries > 0 else 0:,.0f}\n\n")

        f.write("-"*80 + "\n")
        f.write("LEVEL DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for level in ['L1', 'L2', 'L3']:
            count = level_counts[level]
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            f.write(f"{level}: {count:4d} ({percentage:5.1f}%)\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("TAG DISTRIBUTION (Top 30)\n")
        f.write("-"*80 + "\n")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        for tag, count in sorted_tags:
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            f.write(f"{tag:30s} {count:4d} ({percentage:5.1f}%)\n")

        coding_count = tag_counts.get('coding', 0)
        f.write(f"\nCoding tasks: {coding_count}/{total_entries} ({coding_count*100/total_entries if total_entries > 0 else 0:.1f}%)\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("INTERFACE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        sorted_interfaces = sorted(interface_counts.items(), key=lambda x: x[1], reverse=True)
        for interface, count in sorted_interfaces:
            f.write(f"{interface:30s} {count:4d} entries\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("PER-FILE STATISTICS\n")
        f.write("-"*80 + "\n")
        sorted_files = sorted(file_stats, key=lambda x: x['filename'])
        for fstat in sorted_files:
            f.write(f"\n{fstat['filename']}\n")
            f.write(f"  Entries: {fstat['entries']}\n")
            f.write(f"  Characters: {fstat['chars']:,}\n")
            f.write(f"  Tokens: {fstat['tokens']:,.0f}\n")
            f.write(f"  Level: ")
            for lvl in ['L1', 'L2', 'L3']:
                count = fstat['level'].get(lvl, 0)
                if count > 0:
                    f.write(f"{lvl}={count} ")
            f.write("\n")

            # Show top 5 tags for this file
            if fstat['tags']:
                f.write(f"  Top tags: ")
                top_tags = sorted(fstat['tags'].items(), key=lambda x: x[1], reverse=True)[:5]
                f.write(", ".join([f"{tag}({count})" for tag, count in top_tags]))
                f.write("\n")

    print(f"✓ Wrote detailed report to {report_file}")

    if validation_errors:
        print(f"\n⚠️  WARNING: {len(validation_errors)} validation errors found!")
        print("Please review the validation errors above.")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
