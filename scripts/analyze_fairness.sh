#!/bin/bash

# Exit on error
set -e

echo "Running fairness analysis..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the fairness analysis script
python scripts/analyze_fairness.py

echo "Fairness analysis completed successfully!" 