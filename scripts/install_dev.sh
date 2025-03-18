#!/bin/bash

# Exit on error
set -e

echo "Installing package in development mode..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode with all dependencies
echo "Installing package and dependencies..."
pip install -e ".[dev]"

echo "Installation completed successfully!" 