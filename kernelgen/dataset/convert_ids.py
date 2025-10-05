#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime

def convert_timestamp_to_id(timestamp_str):
    """Convert timestamp string to 13-digit numeric ID (milliseconds since epoch)"""
    try:
        # Parse the timestamp
        dt = datetime.fromisoformat(timestamp_str)
        # Convert to milliseconds since epoch (13 digits)
        epoch = datetime(1970, 1, 1)
        milliseconds = int((dt - epoch).total_seconds() * 1000)
        return str(milliseconds)
    except:
        # If it's already a numeric ID
        if timestamp_str.isdigit():
            # If it's 16 digits (microseconds), convert to 13 digits (milliseconds)
            if len(timestamp_str) == 16:
                return timestamp_str[:13]
            # If it's already 13 digits, keep it
            elif len(timestamp_str) == 13:
                return timestamp_str
            # If it's shorter, pad to 13 digits
            elif len(timestamp_str) < 13:
                return timestamp_str.ljust(13, '0')
            # If it's longer than 16, truncate to 13
            else:
                return timestamp_str[:13]
        # Otherwise, try to extract numeric value
        return timestamp_str.replace('T', '').replace(':', '').replace('-', '').replace('.', '')[:13]

def process_jsonl_file(filepath):
    """Process a single JSONL file and convert all IDs"""
    print(f"Processing {filepath}...")

    # Read all lines
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Convert each line
    converted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            # Convert the ID
            if 'id' in data:
                old_id = data['id']
                data['id'] = convert_timestamp_to_id(old_id)
                print(f"  Converted: {old_id} -> {data['id']}")

            converted_lines.append(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"  Error processing line: {e}")
            print(f"  Line: {line[:100]}...")
            continue

    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')

    print(f"  Completed. Converted {len(converted_lines)} entries.\n")

def main():
    # Get the dataset directory
    dataset_dir = Path(__file__).parent

    # Find all .jsonl files
    jsonl_files = list(dataset_dir.glob('*.jsonl'))

    print(f"Found {len(jsonl_files)} JSONL files to process.\n")

    for filepath in sorted(jsonl_files):
        process_jsonl_file(filepath)

    print("All files processed successfully!")

if __name__ == '__main__':
    main()
