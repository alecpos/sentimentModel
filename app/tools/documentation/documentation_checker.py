#!/usr/bin/env python3
"""
Documentation Structure Checker

This script checks the directory structure of the app to identify directories
that are missing README.md or __init__.py files. It generates reports and can
optionally create template files for missing documentation.

DOCUMENTATION STATUS: COMPLETE
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional


README_TEMPLATE = """# {dir_name} Components

This directory contains {dir_name_lower} components for the WITHIN ML Prediction System.

DOCUMENTATION STATUS: INCOMPLETE - This is an auto-generated template that needs completion.

## Purpose

The {dir_name_lower} system provides capabilities for:
- [Describe main purpose 1]
- [Describe main purpose 2]
- [Describe main purpose 3]
- [Describe main purpose 4]
- [Describe main purpose 5]

## Key Components

### [Component Category 1]

Components for [category description]:
- [Component 1]
- [Component 2]
- [Component 3]
- [Component 4]
- [Component 5]

### [Component Category 2]

Components for [category description]:
- [Component 1]
- [Component 2]
- [Component 3]
- [Component 4]
- [Component 5]

## Usage Example

```python
# Add a usage example for the {dir_name_lower} components
from app.{relative_path} import [Component]

# Initialize component
component = [Component](param1="value1", param2="value2")

# Use the component
result = component.process_something(input_data)
```

## Integration Points

- **[System 1]**: [Describe integration]
- **[System 2]**: [Describe integration]
- **[System 3]**: [Describe integration]
- **[System 4]**: [Describe integration]

## Dependencies

- [Dependency 1] for [purpose]
- [Dependency 2] for [purpose]
- [Dependency 3] for [purpose]
- [Dependency 4] for [purpose]
"""

INIT_TEMPLATE = """\"\"\"
{dir_name} components for the WITHIN ML Prediction System.

This module provides [brief description of what this module does].
It includes components for [key functionality areas].

Key functionality includes:
- [Key functionality 1]
- [Key functionality 2]
- [Key functionality 3]
- [Key functionality 4]
- [Key functionality 5]

DOCUMENTATION STATUS: INCOMPLETE - This is an auto-generated template that needs completion.
\"\"\"

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
# [Add your constants here]

# When implementations are added, they will be imported and exported here
"""


class DocumentationChecker:
    """Checks the directory structure for missing documentation files."""
    
    def __init__(
        self, 
        root_dir: str = "app", 
        ignore_dirs: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ):
        """Initialize the documentation checker.
        
        Args:
            root_dir: The root directory to check
            ignore_dirs: List of directory names to ignore
            ignore_patterns: List of patterns to ignore
        """
        self.root_dir = Path(root_dir)
        self.ignore_dirs = set(ignore_dirs or ["__pycache__", ".git", "venv", "env", ".vscode", "node_modules"])
        self.ignore_patterns = ignore_patterns or [r"\..*"]  # Ignore hidden directories
        self.missing_readme: List[Path] = []
        self.missing_init: List[Path] = []
        self.incomplete_readme: List[Path] = []
        self.incomplete_init: List[Path] = []
        
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored.
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if the path should be ignored
        """
        if path.name in self.ignore_dirs:
            return True
        
        for pattern in self.ignore_patterns:
            if pattern.startswith(r"\.") and path.name.startswith("."):
                return True
                
        return False
    
    def check_directory(self, directory: Path) -> None:
        """Check a directory for README.md and __init__.py files.
        
        Args:
            directory: The directory to check
        """
        has_readme = (directory / "README.md").exists()
        has_init = (directory / "__init__.py").exists()
        
        if not has_readme:
            self.missing_readme.append(directory)
        else:
            # Check if README is marked as incomplete
            readme_content = (directory / "README.md").read_text()
            if "DOCUMENTATION STATUS: INCOMPLETE" in readme_content:
                self.incomplete_readme.append(directory)
        
        if not has_init and directory != self.root_dir:
            # Only check for __init__.py in Python package directories
            # Determine if this is likely a Python package by checking for .py files
            has_py_files = any(f.suffix == '.py' for f in directory.iterdir() if f.is_file())
            if has_py_files:
                self.missing_init.append(directory)
        elif has_init:
            # Check if __init__.py is marked as incomplete
            init_content = (directory / "__init__.py").read_text()
            if "DOCUMENTATION STATUS: INCOMPLETE" in init_content:
                self.incomplete_init.append(directory)
    
    def check_all(self) -> None:
        """Check all directories recursively."""
        for root, dirs, _ in os.walk(self.root_dir):
            root_path = Path(root)
            
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(root_path / d)]
            
            self.check_directory(root_path)
    
    def generate_report(self) -> str:
        """Generate a report of missing documentation files.
        
        Returns:
            str: The report as a string
        """
        report = []
        report.append(f"Documentation Check Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        report.append("=" * 80)
        report.append("")
        
        report.append("Directories Missing README.md:")
        for path in sorted(self.missing_readme):
            report.append(f"  - {path}")
        report.append("")
        
        report.append("Directories Missing __init__.py:")
        for path in sorted(self.missing_init):
            report.append(f"  - {path}")
        report.append("")
        
        report.append("Directories with Incomplete README.md:")
        for path in sorted(self.incomplete_readme):
            report.append(f"  - {path}")
        report.append("")
        
        report.append("Directories with Incomplete __init__.py:")
        for path in sorted(self.incomplete_init):
            report.append(f"  - {path}")
        report.append("")
        
        report.append("Summary:")
        report.append(f"  - Total directories checked: {len(set(self.missing_readme) | set(self.missing_init) | set(self.incomplete_readme) | set(self.incomplete_init))}")
        report.append(f"  - Missing README.md: {len(self.missing_readme)}")
        report.append(f"  - Missing __init__.py: {len(self.missing_init)}")
        report.append(f"  - Incomplete README.md: {len(self.incomplete_readme)}")
        report.append(f"  - Incomplete __init__.py: {len(self.incomplete_init)}")
        
        return "\n".join(report)
    
    def create_template_files(self, dry_run: bool = False) -> None:
        """Create template files for missing documentation.
        
        Args:
            dry_run: If True, don't actually create the files
        """
        for directory in self.missing_readme:
            readme_path = directory / "README.md"
            dir_name = directory.name.replace("_", " ").title()
            dir_name_lower = directory.name.replace("_", " ").lower()
            relative_path = str(directory.relative_to(self.root_dir)).replace("\\", "/")
            
            content = README_TEMPLATE.format(
                dir_name=dir_name,
                dir_name_lower=dir_name_lower,
                relative_path=relative_path
            )
            
            if not dry_run:
                readme_path.write_text(content)
                print(f"Created template README.md in {directory}")
            else:
                print(f"Would create template README.md in {directory}")
        
        for directory in self.missing_init:
            init_path = directory / "__init__.py"
            dir_name = directory.name.replace("_", " ").title()
            
            content = INIT_TEMPLATE.format(dir_name=dir_name)
            
            if not dry_run:
                init_path.write_text(content)
                print(f"Created template __init__.py in {directory}")
            else:
                print(f"Would create template __init__.py in {directory}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check for missing documentation files")
    parser.add_argument("--root-dir", default="app", help="Root directory to check")
    parser.add_argument("--report-file", help="File to write the report to")
    parser.add_argument("--create-templates", action="store_true", help="Create template files for missing documentation")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually create any files")
    
    args = parser.parse_args()
    
    checker = DocumentationChecker(root_dir=args.root_dir)
    checker.check_all()
    
    report = checker.generate_report()
    print(report)
    
    if args.report_file:
        with open(args.report_file, "w") as f:
            f.write(report)
    
    if args.create_templates:
        checker.create_template_files(dry_run=args.dry_run)


if __name__ == "__main__":
    main() 