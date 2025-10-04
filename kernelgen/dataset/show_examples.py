#!/usr/bin/env python3
"""
Display examples from the gebd2 dataset for inspection.
"""

import json
import sys
from pathlib import Path

def truncate(text, max_len=150):
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

def show_entry(entry, index):
    """Display a single dataset entry in a readable format."""
    level_emoji = {"L1": "🔹", "L2": "🔸", "L3": "🔶"}
    task_emoji = "💻" if "coding" in entry['tags'] else "📊"

    print(f"\n{'='*80}")
    print(f"{level_emoji[entry['level']]} {task_emoji} Entry #{index}: {entry['level']} - {entry['id']}")
    print(f"{'='*80}")

    print(f"\n📝 INSTRUCTION:")
    print(f"   {entry['instruction']}")

    print(f"\n📋 CONTEXT:")
    print(f"   {entry['context_text']}")

    print(f"\n💾 CODE BLOCKS ({len(entry['code_blocks'])}):")
    for i, block in enumerate(entry['code_blocks'], 1):
        print(f"   [{i}] {block['path']} ({block['language']})")
        print(f"       {len(block['content'])} chars, ~{len(block['content'].splitlines())} lines")
        # Show first 3 lines
        lines = block['content'].strip().split('\n')[:3]
        for line in lines:
            print(f"       | {line[:75]}")
        if len(block['content'].splitlines()) > 3:
            print(f"       | ... ({len(block['content'].splitlines()) - 3} more lines)")

    print(f"\n✅ ANSWER:")
    # Show answer in chunks for readability
    answer_lines = entry['answer'].split('\n')
    for line in answer_lines[:10]:  # First 10 lines
        print(f"   {line}")
    if len(answer_lines) > 10:
        print(f"   ... ({len(answer_lines) - 10} more lines)")

    print(f"\n🔍 RATIONALE:")
    # Show rationale wrapped
    rationale = entry['rationale']
    print(f"   {truncate(rationale, 300)}")
    if len(rationale) > 300:
        print(f"   ... ({len(rationale) - 300} more chars)")

    print(f"\n🏷️  TAGS:")
    print(f"   {', '.join(entry['tags'])}")

    print(f"\n📊 STATS:")
    print(f"   Interface: {entry['interface']}")
    print(f"   Level: {entry['level']}")
    print(f"   Instruction: {len(entry['instruction'])} chars")
    print(f"   Answer: {len(entry['answer'])} chars")
    print(f"   Rationale: {len(entry['rationale'])} chars")
    print(f"   Code blocks: {len(entry['code_blocks'])}")
    total_code = sum(len(b['content']) for b in entry['code_blocks'])
    print(f"   Total code: {total_code} chars")

def show_summary(entries):
    """Show dataset summary statistics."""
    print(f"\n{'='*80}")
    print(f"📊 DATASET SUMMARY: {len(entries)} entries")
    print(f"{'='*80}\n")

    # Level distribution
    levels = {}
    for entry in entries:
        levels[entry['level']] = levels.get(entry['level'], 0) + 1

    print("📈 Level Distribution:")
    for level in sorted(levels.keys()):
        pct = 100 * levels[level] / len(entries)
        bar = '█' * int(pct / 5)
        print(f"   {level}: {levels[level]:2d} ({pct:5.1f}%) {bar}")

    # Task type distribution
    coding = sum(1 for e in entries if 'coding' in e['tags'])
    analysis = sum(1 for e in entries if 'analysis' in e['tags'])

    print(f"\n🔨 Task Type Distribution:")
    coding_pct = 100 * coding / len(entries)
    analysis_pct = 100 * analysis / len(entries)
    print(f"   Coding:   {coding:2d} ({coding_pct:5.1f}%) {'█' * int(coding_pct / 5)}")
    print(f"   Analysis: {analysis:2d} ({analysis_pct:5.1f}%) {'█' * int(analysis_pct / 5)}")

    # Tag frequency
    from collections import Counter
    all_tags = []
    for entry in entries:
        all_tags.extend(entry['tags'])
    tag_counts = Counter(all_tags)

    print(f"\n🏷️  Top Tags:")
    for tag, count in tag_counts.most_common(8):
        pct = 100 * count / len(entries)
        print(f"   {tag:30s}: {count:2d} ({pct:5.1f}%)")

    # Content statistics
    print(f"\n📝 Content Statistics:")
    avg_instruction = sum(len(e['instruction']) for e in entries) / len(entries)
    avg_answer = sum(len(e['answer']) for e in entries) / len(entries)
    avg_rationale = sum(len(e['rationale']) for e in entries) / len(entries)
    avg_code_blocks = sum(len(e['code_blocks']) for e in entries) / len(entries)

    print(f"   Avg instruction length: {avg_instruction:.0f} chars")
    print(f"   Avg answer length:      {avg_answer:.0f} chars")
    print(f"   Avg rationale length:   {avg_rationale:.0f} chars")
    print(f"   Avg code blocks:        {avg_code_blocks:.1f} per entry")

def show_compact_list(entries):
    """Show compact list of all entries."""
    print(f"\n{'='*80}")
    print(f"📋 ALL ENTRIES ({len(entries)} total)")
    print(f"{'='*80}\n")

    for i, entry in enumerate(entries, 1):
        level_emoji = {"L1": "🔹", "L2": "🔸", "L3": "🔶"}
        task_emoji = "💻" if "coding" in entry['tags'] else "📊"

        # Truncate instruction for compact view
        instr = truncate(entry['instruction'], 65)
        print(f"{i:2d}. {level_emoji[entry['level']]} {task_emoji} [{entry['level']}] {instr}")

def main():
    # Check if first argument is a file path
    dataset_file = None
    arg_offset = 1  # Which argument contains the command (N, all, coding, etc.)

    if len(sys.argv) >= 2 and (sys.argv[1].endswith('.jsonl') or '/' in sys.argv[1]):
        # First argument is a file path
        dataset_file = Path(sys.argv[1])
        arg_offset = 2
    else:
        # Use default file
        dataset_file = Path("kernelgen/dataset/roclapack_gebd2.jsonl")
        arg_offset = 1

    if not dataset_file.exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        sys.exit(1)

    with open(dataset_file, 'r') as f:
        entries = [json.loads(line.strip()) for line in f]

    if len(sys.argv) < arg_offset + 1:
        # Default: show summary and compact list
        show_summary(entries)
        show_compact_list(entries)
        print(f"\n💡 Usage:")
        print(f"   python3 {sys.argv[0]} [file.jsonl]           # Show summary and list")
        print(f"   python3 {sys.argv[0]} [file.jsonl] <N>       # Show detailed entry N (1-{len(entries)})")
        print(f"   python3 {sys.argv[0]} [file.jsonl] all       # Show all entries in detail")
        print(f"   python3 {sys.argv[0]} [file.jsonl] coding    # Show only coding tasks")
        print(f"   python3 {sys.argv[0]} [file.jsonl] analysis  # Show only analysis tasks")
        print(f"   python3 {sys.argv[0]} [file.jsonl] L1/L2/L3  # Show only specific level")

    elif sys.argv[arg_offset] == 'all':
        # Show all entries in detail
        for i, entry in enumerate(entries, 1):
            show_entry(entry, i)

    elif sys.argv[arg_offset] in ['coding', 'analysis']:
        # Filter by task type
        task_type = sys.argv[arg_offset]
        filtered = [e for e in entries if task_type in e['tags']]
        print(f"\n{'='*80}")
        print(f"Showing {len(filtered)} {task_type} tasks")
        print(f"{'='*80}")
        for i, entry in enumerate(filtered, 1):
            show_entry(entry, i)

    elif sys.argv[arg_offset] in ['L1', 'L2', 'L3']:
        # Filter by level
        level = sys.argv[arg_offset]
        filtered = [e for e in entries if e['level'] == level]
        print(f"\n{'='*80}")
        print(f"Showing {len(filtered)} {level} entries")
        print(f"{'='*80}")
        for i, entry in enumerate(filtered, 1):
            show_entry(entry, i)

    else:
        # Show specific entry by number
        try:
            index = int(sys.argv[arg_offset])
            if 1 <= index <= len(entries):
                show_entry(entries[index - 1], index)
            else:
                print(f"Error: Entry number must be between 1 and {len(entries)}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid argument '{sys.argv[arg_offset]}'")
            print(f"Usage: {sys.argv[0]} [file.jsonl] [N|all|coding|analysis|L1|L2|L3]")
            sys.exit(1)

if __name__ == '__main__':
    main()
