# Claude AI Bench Command Optimizer

This script uses Claude AI to generate optimized `rocsolver-bench` commands for each YAML configuration file.

## Setup

### Option 1: API Key File
Create a file named `claude_api_key.txt` in the kernelgen directory with your Claude API key:

```bash
echo "your-claude-api-key-here" > kernelgen/claude_api_key.txt
```

### Option 2: Environment Variable
Set the API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-claude-api-key-here"
```

## Usage

```bash
cd /root/rocSOLVER
python3 kernelgen/generate_optimized_bench_commands.py
```

## What it does

1. **Reads each YAML config file** in the kernelgen directory
2. **Extracts function name** from the filename (removes `roclapack_` prefix)
3. **Loads implementation context** from the cpp files listed in the config
4. **Calls Claude AI** with the prompt template and context
5. **Updates the performance_command** in the YAML file with the optimized command

## Features

- **Smart command extraction**: Only replaces `rocsolver-bench` commands, preserves other performance commands
- **Context-aware**: Provides Claude AI with the actual function implementation
- **Parameter optimization**: Claude AI suggests better matrix dimensions and parameters
- **Batch processing**: Processes all config files automatically
- **Error handling**: Graceful handling of API failures and file errors

## Example

**Before:**
```yaml
performance_command:
- rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -- build/release-debug/clients/staging/rocsolver-bench -f getrf -r s -m 3000 -n 3000 --lda 3000 --iters 2
```

**After:**
```yaml
performance_command:
- rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -- build/release-debug/clients/staging/rocsolver-bench -f getrf -r s -m 4096 -n 4096 --lda 4096 --iters 1
```

## Requirements

- `anthropic` Python package: `pip install anthropic`
- Valid Claude API key
- Access to Claude 3.5 Sonnet model