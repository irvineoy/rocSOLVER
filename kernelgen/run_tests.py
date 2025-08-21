#!/usr/bin/env python3
"""
Script to execute correctness_command and performance_command for all yaml config files
Tests can be run in parallel with configurable concurrency
Each test execution is logged to separate files
"""

import os
import yaml
import subprocess
import sys
import threading
import time
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MAX_PARALLEL_TESTS = 8  # Number of tests to run in parallel
LOG_DIR = "kernelgen/logs"

def setup_logging():
    """Create log directory if it doesn't exist"""
    os.makedirs(LOG_DIR, exist_ok=True)

def find_yaml_files(directory="kernelgen"):
    """Find all yaml files in the specified directory"""
    yaml_files = []
    for file in Path(directory).glob("*.yaml"):
        yaml_files.append(str(file))
    return sorted(yaml_files)

def load_yaml_config(file_path):
    """Load yaml configuration file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Unable to load config file {file_path}: {e}")
        return None

def extract_benchmark_commands(performance_commands):
    """Extract only rocsolver-bench commands from performance_command list and modify iters to 1"""
    bench_commands = []
    
    for cmd in performance_commands:
        # Look for rocsolver-bench commands
        if 'rocsolver-bench' in cmd:
            # Modify --iters parameter to 1
            modified_cmd = re.sub(r'--iters\s+\d+', '--iters 1', cmd)
            bench_commands.append(modified_cmd)
    
    return bench_commands

def execute_command(command, log_file, test_type=""):
    """Execute command and write results to log file"""
    start_time = datetime.now()
    
    with open(log_file, 'a') as log:
        log.write(f"{test_type}Command: {command}\n")
        log.write(f"Start time: {start_time}\n")
        log.write("-" * 80 + "\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = datetime.now()
        duration = end_time - start_time
        
        with open(log_file, 'a') as log:
            log.write(f"Return code: {result.returncode}\n")
            log.write(f"Duration: {duration}\n")
            log.write(f"End time: {end_time}\n\n")
            
            if result.stdout:
                log.write("Standard output:\n")
                log.write(result.stdout)
                log.write("\n")
            
            if result.stderr:
                log.write("Error output:\n")
                log.write(result.stderr)
                log.write("\n")
            
            log.write("=" * 80 + "\n\n")
            
        return result.returncode == 0
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        with open(log_file, 'a') as log:
            log.write(f"Exception during execution: {e}\n")
            log.write(f"Duration: {duration}\n")
            log.write(f"End time: {end_time}\n")
            log.write("=" * 80 + "\n\n")
        
        return False

def run_correctness_test(yaml_file, config):
    """Run correctness test for a single yaml file"""
    config_name = Path(yaml_file).stem
    log_file = os.path.join(LOG_DIR, f"{config_name}_correctness.log")
    
    # Initialize log file
    with open(log_file, 'w') as log:
        log.write(f"Correctness test log for: {yaml_file}\n")
        log.write(f"Test started at: {datetime.now()}\n")
        log.write("=" * 80 + "\n\n")
    
    correctness_commands = config.get('correctness_command', [])
    if not correctness_commands:
        with open(log_file, 'a') as log:
            log.write(f"No correctness_command found in {yaml_file}\n")
        return False, "No correctness_command"
    
    with open(log_file, 'a') as log:
        log.write(f"Found {len(correctness_commands)} correctness command(s):\n")
        for i, cmd in enumerate(correctness_commands, 1):
            log.write(f"  {i}. {cmd}\n")
        log.write("\n")
    
    # Execute all correctness commands
    all_success = True
    for i, command in enumerate(correctness_commands, 1):
        with open(log_file, 'a') as log:
            log.write(f"Executing correctness command {i}/{len(correctness_commands)}:\n")
        
        success = execute_command(command, log_file, "Correctness ")
        if not success:
            all_success = False
            with open(log_file, 'a') as log:
                log.write(f"Correctness command {i} failed!\n\n")
    
    # Write final result
    with open(log_file, 'a') as log:
        if all_success:
            log.write(f"✅ All correctness tests passed for {yaml_file}\n")
        else:
            log.write(f"❌ Correctness tests failed for {yaml_file}\n")
        log.write(f"Test completed at: {datetime.now()}\n")
    
    return all_success, "Correctness test completed"

def run_performance_test(yaml_file, config):
    """Run performance test for a single yaml file"""
    config_name = Path(yaml_file).stem
    log_file = os.path.join(LOG_DIR, f"{config_name}_performance.log")
    
    # Initialize log file
    with open(log_file, 'w') as log:
        log.write(f"Performance test log for: {yaml_file}\n")
        log.write(f"Test started at: {datetime.now()}\n")
        log.write("=" * 80 + "\n\n")
    
    performance_commands = config.get('performance_command', [])
    if not performance_commands:
        with open(log_file, 'a') as log:
            log.write(f"No performance_command found in {yaml_file}\n")
        return False, "No performance_command"
    
    # Extract only benchmark commands and modify iters to 1
    bench_commands = extract_benchmark_commands(performance_commands)
    
    if not bench_commands:
        with open(log_file, 'a') as log:
            log.write(f"No rocsolver-bench commands found in performance_command for {yaml_file}\n")
        return False, "No rocsolver-bench commands"
    
    with open(log_file, 'a') as log:
        log.write(f"Found {len(bench_commands)} benchmark command(s):\n")
        for i, cmd in enumerate(bench_commands, 1):
            log.write(f"  {i}. {cmd}\n")
        log.write("\n")
    
    # Execute all benchmark commands
    all_success = True
    for i, command in enumerate(bench_commands, 1):
        with open(log_file, 'a') as log:
            log.write(f"Executing performance command {i}/{len(bench_commands)}:\n")
        
        success = execute_command(command, log_file, "Performance ")
        if not success:
            all_success = False
            with open(log_file, 'a') as log:
                log.write(f"Performance command {i} failed!\n\n")
    
    # Write final result
    with open(log_file, 'a') as log:
        if all_success:
            log.write(f"✅ All performance tests passed for {yaml_file}\n")
        else:
            log.write(f"❌ Performance tests failed for {yaml_file}\n")
        log.write(f"Test completed at: {datetime.now()}\n")
    
    return all_success, "Performance test completed"

def run_single_test(yaml_file, test_type="both"):
    """Run correctness and/or performance test for a single yaml file"""
    config = load_yaml_config(yaml_file)
    if not config:
        return yaml_file, False, False, "Configuration load failed"
    
    correctness_success = True
    performance_success = True
    messages = []
    
    if test_type in ["both", "correctness"]:
        correctness_success, msg = run_correctness_test(yaml_file, config)
        messages.append(f"Correctness: {msg}")
    
    if test_type in ["both", "performance"]:
        performance_success, msg = run_performance_test(yaml_file, config)
        messages.append(f"Performance: {msg}")
    
    return yaml_file, correctness_success, performance_success, " | ".join(messages)

def run_tests(test_type="both"):
    """Main function: execute all tests in parallel"""
    setup_logging()
    
    yaml_files = find_yaml_files()
    
    if not yaml_files:
        print("No yaml configuration files found")
        return
    
    print(f"Found {len(yaml_files)} yaml configuration files:")
    for i, file in enumerate(yaml_files, 1):
        print(f"  {i}. {file}")
    print(f"\nRunning {test_type} tests with max {MAX_PARALLEL_TESTS} parallel processes")
    print(f"Logs will be written to: {LOG_DIR}/")
    print()
    
    successful_correctness = []
    failed_correctness = []
    successful_performance = []
    failed_performance = []
    
    # Run tests in parallel
    start_time = datetime.now()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TESTS) as executor:
        # Submit all tasks
        future_to_yaml = {executor.submit(run_single_test, yaml_file, test_type): yaml_file 
                         for yaml_file in yaml_files}
        
        # Process completed tasks
        for future in as_completed(future_to_yaml):
            yaml_file = future_to_yaml[future]
            try:
                yaml_path, correctness_success, performance_success, message = future.result()
                
                # Handle correctness results
                if test_type in ["both", "correctness"]:
                    if correctness_success:
                        successful_correctness.append(yaml_path)
                        print(f"✅ {yaml_path} - Correctness passed")
                    else:
                        failed_correctness.append((yaml_path, message))
                        print(f"❌ {yaml_path} - Correctness failed")
                
                # Handle performance results
                if test_type in ["both", "performance"]:
                    if performance_success:
                        successful_performance.append(yaml_path)
                        print(f"🚀 {yaml_path} - Performance passed")
                    else:
                        failed_performance.append((yaml_path, message))
                        print(f"⚠️  {yaml_path} - Performance failed")
                        
            except Exception as exc:
                if test_type in ["both", "correctness"]:
                    failed_correctness.append((yaml_file, f"Exception: {exc}"))
                if test_type in ["both", "performance"]:
                    failed_performance.append((yaml_file, f"Exception: {exc}"))
                print(f"❌ {yaml_file} - Exception: {exc}")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    print(f"Total configs: {len(yaml_files)}")
    
    if test_type in ["both", "correctness"]:
        print(f"Correctness - Successful: {len(successful_correctness)}, Failed: {len(failed_correctness)}")
    
    if test_type in ["both", "performance"]:
        print(f"Performance - Successful: {len(successful_performance)}, Failed: {len(failed_performance)}")
    
    print(f"Total duration: {total_duration}")
    print(f"Logs directory: {LOG_DIR}/")
    
    if test_type in ["both", "correctness"] and successful_correctness:
        print(f"\nSuccessful correctness tests:")
        for test in successful_correctness:
            print(f"  ✅ {test}")
    
    if test_type in ["both", "correctness"] and failed_correctness:
        print(f"\nFailed correctness tests:")
        for test, reason in failed_correctness:
            print(f"  ❌ {test} - {reason}")
    
    if test_type in ["both", "performance"] and successful_performance:
        print(f"\nSuccessful performance tests:")
        for test in successful_performance:
            print(f"  🚀 {test}")
    
    if test_type in ["both", "performance"] and failed_performance:
        print(f"\nFailed performance tests:")
        for test, reason in failed_performance:
            print(f"  ⚠️  {test} - {reason}")
    
    # Write summary log
    summary_log = os.path.join(LOG_DIR, f"test_summary_{test_type}.log")
    with open(summary_log, 'w') as log:
        log.write(f"{test_type.title()} Test Summary\n")
        log.write(f"Test run started: {start_time}\n")
        log.write(f"Test run completed: {end_time}\n")
        log.write(f"Total duration: {total_duration}\n")
        log.write(f"Max parallel processes: {MAX_PARALLEL_TESTS}\n")
        log.write("=" * 80 + "\n\n")
        
        log.write(f"Total configs: {len(yaml_files)}\n")
        
        if test_type in ["both", "correctness"]:
            log.write(f"Correctness - Successful: {len(successful_correctness)}, Failed: {len(failed_correctness)}\n")
            
            if successful_correctness:
                log.write("\nSuccessful correctness tests:\n")
                for test in successful_correctness:
                    log.write(f"  ✅ {test}\n")
            
            if failed_correctness:
                log.write("\nFailed correctness tests:\n")
                for test, reason in failed_correctness:
                    log.write(f"  ❌ {test} - {reason}\n")
        
        if test_type in ["both", "performance"]:
            log.write(f"Performance - Successful: {len(successful_performance)}, Failed: {len(failed_performance)}\n")
            
            if successful_performance:
                log.write("\nSuccessful performance tests:\n")
                for test in successful_performance:
                    log.write(f"  🚀 {test}\n")
            
            if failed_performance:
                log.write("\nFailed performance tests:\n")
                for test, reason in failed_performance:
                    log.write(f"  ⚠️  {test} - {reason}\n")
    
    print(f"\nSummary written to: {summary_log}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run correctness and/or performance tests")
    parser.add_argument("--type", choices=["correctness", "performance", "both"], 
                       default="both", help="Type of tests to run")
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL_TESTS,
                       help="Number of parallel processes")
    
    args = parser.parse_args()
    
    # Update global config
    MAX_PARALLEL_TESTS = args.parallel
    
    run_tests(args.type)