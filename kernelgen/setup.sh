#!/bin/bash

cd "$(dirname "$0")"

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not install, install now..."

    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv installation failed. Please install uv manually."
        exit 1
    fi
fi
echo "[INFO] uv installed, version: $(uv --version)"


# Sets up the development environment for the kernel-agentic project.
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.12
fi


source .venv/bin/activate


# Get the current working directory
CUR_DIR=$(pwd)

# Find the python path inside the .venv
VENV_PYTHON="$CUR_DIR/.venv/bin/python"

# Check if rocprof-compute exists before creating alias
if [ -f "/opt/rocm/bin/rocprof-compute" ]; then
    uv pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt --no-cache-dir

    alias rocprof-compute-rocsolver="$VENV_PYTHON /opt/rocm/bin/rocprof-compute"

    # Prepare the alias command with the venv python
    ALIAS_CMD="alias rocprof-compute-rocsolver=\"$VENV_PYTHON /opt/rocm/bin/rocprof-compute\""

    # Check if the alias already exists in .bashrc to avoid duplicates
    if ! grep -Fxq "$ALIAS_CMD" ~/.bashrc; then
        echo "$ALIAS_CMD" >> ~/.bashrc
        echo "[INFO] Added rocprof-compute alias to ~/.bashrc"
    else
        echo "[INFO] rocprof-compute alias already exists in ~/.bashrc"
    fi
else
    echo "[INFO] /opt/rocm/bin/rocprof-compute not found, please install rocprofiler-compute first"

fi

echo "Run 'source .venv/bin/activate' to activate the virtual environment."

# Start to patch up the rocsolver
sudo apt-get install -y ccache

$VENV_PYTHON ./patch_rocsolver.py


../install.sh -a gfx942 -c -g -d


# ./install.sh -a gfx942
# ./install.sh -a gfx942 -c -g -d


