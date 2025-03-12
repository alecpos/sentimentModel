#!/usr/bin/env python
"""
Documentation Validation Script

This script runs the Documentation Reference Validator to ensure all documentation
references are accurate and that documentation is complete and up-to-date relative
to the actual code.

Usage:
    python scripts/validate_documentation.py [options]

Options:
    --index-file PATH         Starting point for validation (default: app/README.md)
    --exclude-dirs LIST       Comma-separated list of directories to exclude
    --exclude-files LIST      Comma-separated list of files to exclude
    --format FORMAT           Output format: markdown or json (default: markdown)
    --output PATH             File to write the report to (default: stdout)
    --verbose                 Enable detailed logging
    --priority LEVEL          Filter issues by priority: 'high' or 'all' (default: all)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.tools.documentation import (
    DocReferenceValidator,
    ValidationSeverity
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate documentation references")
    parser.add_argument(
        "--index-file",
        default="app/README.md",
        help="Starting point for validation"
    )
    parser.add_argument(
        "--exclude-dirs",
        default=".git,__pycache__,venv",
        help="Comma-separated list of directories to exclude"
    )
    parser.add_argument(
        "--exclude-files",
        default=".DS_Store",
        help="Comma-separated list of files to exclude"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        help="File to write the report to (default: stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    parser.add_argument(
        "--priority",
        choices=["high", "all"],
        default="all",
        help="Filter issues by priority"
    )
    return parser.parse_args()


def main():
    """Run the documentation validator."""
    args = parse_args()
    
    # Configure logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse exclude lists
    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(",") if d.strip()]
    exclude_files = [f.strip() for f in args.exclude_files.split(",") if f.strip()]
    
    # Initialize the validator
    validator = DocReferenceValidator(
        index_file=args.index_file,
        workspace_root=str(project_root),
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )
    
    # Run the validation
    print(f"Starting documentation validation from {args.index_file}...")
    report = validator.validate()
    
    # Filter issues by priority if needed
    if args.priority == "high":
        report.issues = [
            issue for issue in report.issues
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ]
    
    # Generate the report
    if args.format == "markdown":
        output = report.to_markdown()
    else:  # json
        output = report.to_json()
    
    # Write the report
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print("\n" + "=" * 80)
        print("DOCUMENTATION VALIDATION REPORT")
        print("=" * 80 + "\n")
        print(output)
    
    # Print summary
    print("\nSummary:")
    print(f"- Total issues: {len(report.issues)}")
    print(f"- Errors: {report.error_count}")
    print(f"- Warnings: {report.warning_count}")
    print(f"- Missing documentation: {len(report.missing_documentation)}")
    print(f"- Incomplete documentation: {len(report.incomplete_documentation)}")
    
    # Return non-zero exit code if there are errors
    return 1 if report.error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main()) 