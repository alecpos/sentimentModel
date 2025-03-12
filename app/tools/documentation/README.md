# Documentation Validation Tools

This package provides tools for validating the alignment between code and documentation in the WITHIN ML project. The main tools include:

1. **Documentation Reference Validator**: Recursively analyzes the codebase to ensure that all referenced documentation exists and accurately reflects the actual code implementation.

2. **Documentation Structure Checker**: Scans the directory structure to identify missing README.md and __init__.py files, and can generate template documentation files.

## Documentation Reference Validator

### Key Features

- **Reference Validation**: Verifies that all documentation references in code point to existing documentation files
- **Content Alignment**: Ensures documentation accurately describes the code it references
- **Code Structure Analysis**: Analyzes code structures (classes, functions) to identify missing documentation
- **Documentation Completeness**: Checks that documentation covers all important code components
- **Remediation Suggestions**: Generates placeholder documentation for missing or incomplete documentation

### Usage

#### Command-line Interface

The simplest way to use the validator is through the provided command-line script:

```bash
# Run from project root with default settings
python scripts/validate_documentation.py

# Specify a custom starting point
python scripts/validate_documentation.py --index-file docs/implementation/ml/index.md

# Output to a file in JSON format
python scripts/validate_documentation.py --format json --output validation_report.json

# Exclude specific directories
python scripts/validate_documentation.py --exclude-dirs "venv,test_data,examples"

# Focus on high-priority documentation issues
python scripts/validate_documentation.py --priority high

# Get detailed logs
python scripts/validate_documentation.py --verbose
```

#### Python API

You can also use the validator programmatically in your Python code:

```python
from app.tools.documentation import DocReferenceValidator, ValidationSeverity

# Initialize the validator
validator = DocReferenceValidator(
    index_file="docs/index.md",
    workspace_root="/path/to/workspace",
    exclude_dirs=[".git", "__pycache__", "venv"],
    exclude_files=[".DS_Store"]
)

# Run the validation
report = validator.validate()

# Process results
print(f"Found {len(report.issues)} issues")
print(f"Errors: {report.error_count}")
print(f"Warnings: {report.warning_count}")

# Get critical issues only
critical_issues = [
    issue for issue in report.issues 
    if issue.severity == ValidationSeverity.CRITICAL
]

# Generate report
markdown_report = report.to_markdown()
json_report = report.to_json()
```

### Main Components

- **DocReferenceValidator**: Core validation engine that analyzes code and documentation
- **ValidationReport**: Contains validation results, including issues and suggestions
- **ValidationIssue**: Represents a specific issue found during validation
- **CodeStructure**: Represents a structure in the code (class, function, etc.)
- **DocumentationReference**: Represents a reference to a documentation file

### Validation Process

1. Start from specified index file and extract all documentation references
2. Build a graph of documentation dependencies
3. Analyze code structure to identify important components
4. Validate that referenced documentation exists
5. Check that documentation content aligns with code implementation
6. Identify code components lacking proper documentation
7. Generate a comprehensive validation report with remediation suggestions

### Configuration Options

- **index_file**: Starting point for validation
- **workspace_root**: Root directory of the codebase
- **exclude_dirs**: Directories to exclude from analysis
- **exclude_files**: Files to exclude from analysis
- **format**: Output format (markdown or JSON)
- **output**: File to write the report to
- **verbose**: Enable detailed logging
- **priority**: Filter issues by priority ('high' or 'all')
- **doc-tracker**: Path to documentation tracker file

### Integration with CI/CD

The validator can be integrated into CI/CD pipelines to ensure documentation remains up-to-date:

```yaml
documentation-validation:
  stage: test
  script:
    - python scripts/validate_documentation.py --format json --output doc_validation.json
    - python scripts/parse_validation_results.py doc_validation.json
  artifacts:
    paths:
      - doc_validation.json
    reports:
      junit: doc_validation_junit.xml
```

## Documentation Structure Checker

The Documentation Structure Checker helps ensure consistent documentation structure across the codebase by identifying directories that are missing README.md or __init__.py files, and can generate template documentation files.

### Key Features

- **Structure Validation**: Scans the directory structure to identify missing documentation files
- **Template Generation**: Creates template README.md and __init__.py files for missing documentation
- **Documentation Status**: Marks documentation as complete or incomplete for tracking purposes
- **Report Generation**: Generates reports on the status of documentation across the codebase

### Usage

#### Command-line Interface

The Documentation Structure Checker can be run from the command line:

```bash
# Check for missing documentation files
python -m app.tools.documentation.documentation_checker --root-dir app

# Generate a report file
python -m app.tools.documentation.documentation_checker --root-dir app --report-file docs/documentation_report.txt

# Create template files for missing documentation (dry run)
python -m app.tools.documentation.documentation_checker --root-dir app --create-templates --dry-run

# Actually create the template files
python -m app.tools.documentation.documentation_checker --root-dir app --create-templates
```

#### Python API

You can also use the checker programmatically:

```python
from app.tools.documentation.documentation_checker import DocumentationChecker

# Initialize the checker
checker = DocumentationChecker(
    root_dir="app",
    ignore_dirs=["__pycache__", ".git", "venv"]
)

# Check all directories
checker.check_all()

# Generate a report
report = checker.generate_report()
print(report)

# Create template files for missing documentation
checker.create_template_files(dry_run=True)  # Set to False to actually create files
```

### Template Files

The checker creates template files with the following structure:

#### README.md Template

```markdown
# [Directory Name] Components

This directory contains [directory name] components for the WITHIN ML Prediction System.

DOCUMENTATION STATUS: INCOMPLETE - This is an auto-generated template that needs completion.

## Purpose

The [directory name] system provides capabilities for:
- [Describe main purpose 1]
- [Describe main purpose 2]
...
```

#### __init__.py Template

```python
"""
[Directory Name] components for the WITHIN ML Prediction System.

This module provides [brief description of what this module does].
It includes components for [key functionality areas].

Key functionality includes:
- [Key functionality 1]
- [Key functionality 2]
...

DOCUMENTATION STATUS: INCOMPLETE - This is an auto-generated template that needs completion.
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
# [Add your constants here]

# When implementations are added, they will be imported and exported here
```

### Integration

- Can be run as part of the development workflow to ensure documentation coverage
- Can be integrated into CI/CD pipelines to track documentation status
- Helps maintain consistent documentation structure across the codebase
- Supports the onboarding process by ensuring developers understand documentation requirements 