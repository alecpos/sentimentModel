#!/bin/bash

# Exit on error
set -e

echo "Running SHAP analysis..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the SHAP analysis script
python scripts/analyze_shap.py

echo "SHAP analysis completed successfully!" 