#!/bin/bash

# Exit on error
set -e

echo "Running demonstration..."

# Activate virtual environment
source venv/bin/activate

# Run the demonstration script
python examples/demo.py

echo "Demonstration completed successfully!" 