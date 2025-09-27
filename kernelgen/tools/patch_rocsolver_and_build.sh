#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Change to the root directory of the project (two levels up from kernelgen/tools/)
cd "$SCRIPT_DIR/../.."

# Save the root directory path
ROOT_DIR=$(pwd)

echo "Running from root directory: $ROOT_DIR"

# Install ccache if needed
sudo apt-get install -y ccache

# Run patch_rocsolver.py from the root directory
python3 kernelgen/tools/patch_rocsolver.py

# Run install.sh from the root directory
./install.sh --architecture gfx942 --clients --relwithdebinfo --dependencies


