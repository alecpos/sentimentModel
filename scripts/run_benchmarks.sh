#!/bin/bash

# Exit on error
set -e

echo "Running performance benchmarks..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the benchmarks script
python scripts/run_benchmarks.py

echo "Benchmarks completed successfully!" 