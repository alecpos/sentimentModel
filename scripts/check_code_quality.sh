#!/bin/bash

# Exit on error
set -e

echo "Running code quality checks..."

# Run black for code formatting
echo "\nChecking code formatting with black..."
black --check app/ tests/

# Run flake8 for style checking
echo "\nRunning flake8..."
flake8 app/ tests/

# Run pylint for code analysis
echo "\nRunning pylint..."
pylint app/ tests/

# Run mypy for type checking
echo "\nRunning mypy..."
mypy app/ tests/

# Run pytest with coverage
echo "\nRunning tests with coverage..."
pytest tests/ --cov=app --cov-report=term-missing

echo "\nAll checks completed successfully!" 