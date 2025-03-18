#!/bin/bash

# Exit on error
set -e

echo "Running drift detection..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the drift detection script
python scripts/detect_drift.py

echo "Drift detection completed successfully!"