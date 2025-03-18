#!/bin/bash

# Exit on error
set -e

echo "Running CLI tool..."

# Activate virtual environment
source venv/bin/activate

# Run bagging ensemble
echo "\nRunning bagging ensemble..."
enhanced-ensemble --mode bagging \
    --data data/sample_data.csv \
    --output results/bagging_results.csv \
    --n-estimators 10

# Run stacking ensemble
echo "\nRunning stacking ensemble..."
enhanced-ensemble --mode stacking \
    --data data/sample_data.csv \
    --output results/stacking_results.csv \
    --n-splits 5

# Run weight optimization
echo "\nRunning weight optimization..."
enhanced-ensemble --mode optimize \
    --data data/sample_data.csv \
    --output results/optimized_weights.csv

# Run visualization
echo "\nGenerating visualizations..."
enhanced-ensemble --mode visualize \
    --data data/sample_data.csv \
    --output results/ensemble_visualization.png

echo "\nCLI demonstration completed successfully!" 