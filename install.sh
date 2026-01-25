#!/bin/bash
# Readtube installer
# Usage: curl -sSL https://raw.githubusercontent.com/unbalancedparentheses/readtube/main/install.sh | bash

set -e

echo "Installing Readtube..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.8+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Install with pipx if available, otherwise pip
if command -v pipx &> /dev/null; then
    echo "Installing with pipx..."
    pipx install git+https://github.com/unbalancedparentheses/readtube.git
    echo ""
    echo "Done! Run 'readtube --help' to get started."
else
    echo "Installing with pip..."
    pip install --user git+https://github.com/unbalancedparentheses/readtube.git
    echo ""
    echo "Done! Run 'python -m fetch_transcript --help' to get started."
    echo ""
    echo "Tip: Install pipx for a cleaner installation:"
    echo "  brew install pipx && pipx ensurepath"
fi
