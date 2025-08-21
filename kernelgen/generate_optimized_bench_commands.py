#!/usr/bin/env python3
"""
Script to generate optimized rocsolver-bench commands using Claude AI Opus 4.1 API
For each config file, it calls Claude AI to generate better benchmark parameters
"""

import os
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List
import anthropic

# Configuration
API_KEY_FILE = "claude_api_key.txt"  # File containing the API key
MODEL = "claude-3-5-sonnet-20241022"  # Using Claude 3.5 Sonnet as it's more widely available

def load_api_key():
    """Load Claude API key from file"""
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, 'r') as f:
                return f.read().strip()
        else:
            # Try environment variable
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                return api_key
            else:
                print(f"Please create {API_KEY_FILE} with your Claude API key or set ANTHROPIC_API_KEY environment variable")
                return None
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

def load_rocsolver_bench_context():
    """Load the prompt template from rocsolver-bench_context.py"""
    try:
        with open("kernelgen/rocsolver-bench_context.py", 'r') as f:
            content = f.read()
        
        # Extract the template string
        template_match = re.search(r"template = f'''(.*?)'''", content, re.DOTALL)
        if template_match:
            return template_match.group(1)
        else:
            print("Error: Could not find template in rocsolver-bench_context.py")
            return None
    except Exception as e:
        print(f"Error loading rocsolver-bench context: {e}")
        return None

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

def extract_function_name(yaml_file):
    """Extract function name from yaml filename"""
    filename = Path(yaml_file).stem
    if filename.startswith("roclapack_"):
        return filename[10:]  # Remove "roclapack_" prefix
    return filename

def load_cpp_file_content(cpp_files):
    """Load content from all cpp files in the config"""
    content = ""
    for cpp_file in cpp_files:
        if cpp_file.endswith('.cpp') and os.path.exists(cpp_file):
            try:
                with open(cpp_file, 'r') as f:
                    file_content = f.read()
                content += f"\\n\\n// Content from {cpp_file}:\\n"
                content += file_content
            except Exception as e:
                print(f"Warning: Could not read {cpp_file}: {e}")
    return content

def extract_current_bench_command(performance_commands):
    """Extract the current rocsolver-bench command from performance_command list"""
    for cmd in performance_commands:
        if 'rocsolver-bench' in cmd:
            # Extract just the rocsolver-bench part
            bench_match = re.search(r'build/release/clients/staging/rocsolver-bench[^\\n]*', cmd)
            if bench_match:
                return bench_match.group(0)
    return None

def call_claude_api(client, prompt):
    """Call Claude API to generate optimized bench command"""
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        response_text = message.content[0].text
        
        # Extract the rocsolver-bench command from the response
        bench_match = re.search(r'build/release/clients/staging/rocsolver-bench[^\\n]*', response_text)
        if bench_match:
            return bench_match.group(0).strip()
        else:
            # If no exact match, try to find any line that looks like a command
            lines = response_text.strip().split('\\n')
            for line in lines:
                line = line.strip()
                if 'rocsolver-bench' in line and line.startswith('build/'):
                    return line
            
            print(f"Warning: Could not extract bench command from response: {response_text}")
            return None
            
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None

def update_config_file(yaml_file, config, new_bench_command):
    """Update the config file with the new bench command"""
    try:
        # Update performance_command
        updated_commands = []
        bench_command_updated = False
        
        for cmd in config.get('performance_command', []):
            if 'rocsolver-bench' in cmd:
                # Replace the rocsolver-bench part with the new command
                if 'rocprof-compute-rocsolver' in cmd:
                    # For rocprof commands, replace just the bench part
                    updated_cmd = re.sub(
                        r'build/release/clients/staging/rocsolver-bench[^\\\\]*',
                        new_bench_command,
                        cmd
                    )
                    updated_commands.append(updated_cmd)
                else:
                    # Direct bench command
                    updated_commands.append(new_bench_command)
                bench_command_updated = True
            else:
                updated_commands.append(cmd)
        
        if bench_command_updated:
            config['performance_command'] = updated_commands
            
            # Write back to file
            with open(yaml_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                         allow_unicode=True, width=120)
            
            return True
        else:
            print(f"Warning: No rocsolver-bench command found in {yaml_file}")
            return False
            
    except Exception as e:
        print(f"Error updating config file {yaml_file}: {e}")
        return False

def process_config_file(client, yaml_file, template):
    """Process a single config file"""
    print(f"\\nProcessing: {yaml_file}")
    
    config = load_yaml_config(yaml_file)
    if not config:
        print(f"  ❌ Failed to load config")
        return False
    
    # Extract function name
    function_name = extract_function_name(yaml_file)
    print(f"  Function: {function_name}")
    
    # Load cpp file content
    source_files = config.get('source_file_path', [])
    cpp_files = [f for f in source_files if f.endswith('.cpp')]
    
    if not cpp_files:
        print(f"  ❌ No cpp files found in config")
        return False
    
    print(f"  Loading content from: {cpp_files}")
    implementation_context = load_cpp_file_content(cpp_files)
    
    if not implementation_context.strip():
        print(f"  ❌ No implementation content found")
        return False
    
    # Get current bench command
    current_bench_command = extract_current_bench_command(config.get('performance_command', []))
    if current_bench_command:
        print(f"  Current command: {current_bench_command}")
    
    # Prepare prompt
    prompt = template.replace('FUNCTION_NAME', function_name)
    prompt = prompt.replace('FUNCTION_IMPLEMENTATION_CONTEXT', implementation_context)
    
    print(f"  Calling Claude API...")
    
    # Call Claude API
    new_bench_command = call_claude_api(client, prompt)
    
    if not new_bench_command:
        print(f"  ❌ Failed to get response from Claude API")
        return False
    
    print(f"  New command: {new_bench_command}")
    
    # Update config file
    success = update_config_file(yaml_file, config, new_bench_command)
    
    if success:
        print(f"  ✅ Updated successfully")
        return True
    else:
        print(f"  ❌ Failed to update config file")
        return False

def main():
    """Main function"""
    print("Claude AI Bench Command Optimizer")
    print("=" * 50)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        return
    
    # Initialize Claude client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(f"✅ Claude API client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Claude API client: {e}")
        return
    
    # Load prompt template
    template = load_rocsolver_bench_context()
    if not template:
        return
    
    print(f"✅ Loaded prompt template")
    
    # Find yaml files
    yaml_files = find_yaml_files()
    
    if not yaml_files:
        print("❌ No yaml configuration files found")
        return
    
    print(f"✅ Found {len(yaml_files)} yaml configuration files")
    
    # Process each file
    successful = 0
    failed = 0
    
    for yaml_file in yaml_files:
        try:
            success = process_config_file(client, yaml_file, template)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Exception processing {yaml_file}: {e}")
            failed += 1
    
    # Print summary
    print(f"\\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total files: {len(yaml_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\\n✅ Successfully updated {successful} config files")
    if failed > 0:
        print(f"❌ Failed to update {failed} config files")

if __name__ == "__main__":
    main()