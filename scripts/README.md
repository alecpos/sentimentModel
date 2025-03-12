# WITHIN ML Prediction System Scripts

This directory contains utility scripts for the WITHIN ML Prediction System. These scripts support various development, testing, and maintenance tasks.

## Available Scripts

### Documentation Validation

**File**: `validate_documentation.py`

This script runs the Documentation Reference Validator to ensure all documentation references are accurate and that documentation is complete and up-to-date relative to the actual code.

**Usage**:
```bash
# Run with default settings
python scripts/validate_documentation.py

# Specify a custom starting point
python scripts/validate_documentation.py --index-file app/models/ml/README.md

# Output to a file in JSON format
python scripts/validate_documentation.py --format json --output validation_report.json

# Exclude specific directories
python scripts/validate_documentation.py --exclude-dirs "venv,test_data,examples"

# Focus on high-priority documentation issues
python scripts/validate_documentation.py --priority high

# Get detailed logs
python scripts/validate_documentation.py --verbose
```

**Options**:
- `--index-file PATH`: Starting point for validation (default: app/README.md)
- `--exclude-dirs LIST`: Comma-separated list of directories to exclude
- `--exclude-files LIST`: Comma-separated list of files to exclude
- `--format FORMAT`: Output format: markdown or json (default: markdown)
- `--output PATH`: File to write the report to (default: stdout)
- `--verbose`: Enable detailed logging
- `--priority LEVEL`: Filter issues by priority: 'high' or 'all' (default: all)

### Standalone Documentation Structure Validator

**File**: `standalone_doc_validator.py`

This script checks the documentation structure across the codebase without requiring imports from the application code. It's useful when there are import errors in the codebase that prevent the main validator from running.

**Usage**:
```bash
# Run with default settings
python scripts/standalone_doc_validator.py

# Specify a custom root directory
python scripts/standalone_doc_validator.py --root-dir app/models

# Output to a file in JSON format
python scripts/standalone_doc_validator.py --format json --output doc_structure.json

# Output in markdown format
python scripts/standalone_doc_validator.py --format markdown --output doc_structure.md

# Exclude specific directories
python scripts/standalone_doc_validator.py --exclude-dirs "venv,test_data,examples"

# Get detailed logs
python scripts/standalone_doc_validator.py --verbose
```

**Options**:
- `--root-dir PATH`: Root directory to start validation (default: app/)
- `--exclude-dirs LIST`: Comma-separated list of directories to exclude
- `--exclude-files LIST`: Comma-separated list of files to exclude
- `--format FORMAT`: Output format: text, json, or markdown (default: text)
- `--output PATH`: File to write the report to (default: stdout)
- `--verbose`: Enable detailed logging

## Adding New Scripts

When adding new scripts to this directory, please follow these guidelines:

1. Include a detailed docstring at the top of the script explaining its purpose and usage
2. Add command-line argument parsing with helpful descriptions
3. Include error handling and appropriate logging
4. Document the script in this README.md file
5. Make the script executable (`chmod +x script_name.py`)
6. Use the shebang line `#!/usr/bin/env python` at the top of the file

## Script Design Principles

1. **Self-contained**: Scripts should be runnable with minimal setup
2. **Well-documented**: Include clear usage instructions and examples
3. **Robust**: Handle errors gracefully and provide helpful error messages
4. **Configurable**: Use command-line arguments for configuration
5. **Consistent**: Follow the same coding style and patterns as the rest of the project

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines to automate various tasks. For example, the documentation validation script can be run as part of the CI pipeline to ensure documentation remains up-to-date. 