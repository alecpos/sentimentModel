#!/bin/bash

# Exit on error
set -e

echo "Creating benchmark visualizations..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the visualization script
python scripts/visualize_benchmarks.py

echo "Visualizations completed successfully!" 