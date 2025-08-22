#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Change to the kernelgen directory (parent of tools/)
KERNELGEN_DIR="$SCRIPT_DIR/.."
cd "$KERNELGEN_DIR"

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not installed, installing now..."

    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv installation failed. Please install uv manually."
        exit 1
    fi
fi
echo "[INFO] uv installed, version: $(uv --version)"

# Sets up the development environment for the kernel-agentic project
# .venv will be created in kernelgen/ directory
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment in $KERNELGEN_DIR/.venv"
    uv venv .venv --python 3.12
fi

source .venv/bin/activate

# Get the absolute path to the kernelgen directory
KERNELGEN_ABS_PATH=$(pwd)

# Find the python path inside the .venv
VENV_PYTHON="$KERNELGEN_ABS_PATH/.venv/bin/python"

# Check if rocprof-compute exists before setting up
if [ -f "/opt/rocm/bin/rocprof-compute" ]; then
    echo "[INFO] Installing rocprof-compute requirements..."
    uv pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt --no-cache-dir

    # Set the alias for current session
    alias rocprof-compute="$VENV_PYTHON /opt/rocm/bin/rocprof-compute"

    # Prepare the alias command with the venv python
    ALIAS_CMD="alias rocprof-compute=\"$VENV_PYTHON /opt/rocm/bin/rocprof-compute\""

    echo ""
    echo "========================================="
    echo "[INFO] rocprof-compute setup complete!"
    echo ""
    echo "To use rocprof-compute with the correct Python environment,"
    echo "you can add the following alias to your ~/.bashrc:"
    echo ""
    echo "  $ALIAS_CMD"
    echo ""
    echo "Or run this command to add it automatically:"
    echo "  echo '$ALIAS_CMD' >> ~/.bashrc"
    echo "========================================="
else
    echo "[ERROR] /opt/rocm/bin/rocprof-compute not found, please install rocprofiler-compute first"
    exit 1
fi

echo ""
echo "Run 'source $KERNELGEN_ABS_PATH/.venv/bin/activate' to activate the virtual environment."