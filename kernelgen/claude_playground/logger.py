#!/usr/bin/env python3
import sys
from datetime import datetime

def log_message(message):
    """Append message to claudecode.log with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('claudecode.log', 'a') as f:
        f.write(f"\n{message}\n")
        f.write(f"{timestamp}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        log_message(message)
    else:
        log_message("=== LOG ENTRY ===")