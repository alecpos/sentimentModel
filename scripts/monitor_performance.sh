#!/bin/bash

# Exit on error
set -e

echo "Running performance monitoring..."

# Create results directory if it doesn't exist
mkdir -p results

# Activate virtual environment
source venv/bin/activate

# Run the monitoring script
python scripts/monitor_performance.py

echo "Performance monitoring completed successfully!" 