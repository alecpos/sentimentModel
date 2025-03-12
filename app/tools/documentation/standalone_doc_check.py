#!/usr/bin/env python3
"""
Standalone documentation checker script.

This script is completely standalone and does not import any app modules.
It scans the directory structure and reports missing README.md and __init__.py files.

DOCUMENTATION STATUS: COMPLETE
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional


class StandaloneDocChecker:
    """Simple standalone documentation checker."""
    
    def __init__(
        self, 
        root_dir: str = "app", 
        ignore_dirs: Optional[List[str]] = None
    ):
        """Initialize the documentation checker.
        
        Args:
            root_dir: The root directory to check
            ignore_dirs: List of directory names to ignore
        """
        self.root_dir = Path(root_dir)
        self.ignore_dirs = set(ignore_dirs or ["__pycache__", ".git", "venv", "env", ".vscode", "node_modules"])
        self.missing_readme: List[Path] = []
        self.missing_init: List[Path] = []
        
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored.
        
        Args:
            path: The path to check
            
        Returns:
            bool: True if the path should be ignored
        """
        if path.name in self.ignore_dirs or path.name.startswith("."):
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
        
        if not has_init and directory != self.root_dir:
            # Only check for __init__.py in Python package directories
            # Determine if this is likely a Python package by checking for .py files
            has_py_files = any(f.suffix == '.py' for f in directory.iterdir() if f.is_file())
            if has_py_files:
                self.missing_init.append(directory)
    
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
        
        report.append("Summary:")
        report.append(f"  - Total directories checked: {len(set(self.missing_readme) | set(self.missing_init))}")
        report.append(f"  - Missing README.md: {len(self.missing_readme)}")
        report.append(f"  - Missing __init__.py: {len(self.missing_init)}")
        
        return "\n".join(report)


def main():
    """Main function."""
    print("Running standalone documentation check...")
    
    checker = StandaloneDocChecker(root_dir='app')
    checker.check_all()
    
    report = checker.generate_report()
    print(report)
    
    # Print top 10 directories missing README.md
    if checker.missing_readme:
        print("\nTop 10 Directories Missing README.md:")
        for i, path in enumerate(sorted(checker.missing_readme)[:10]):
            print(f"  {i+1}. {path}")
    
    # Print top 10 directories missing __init__.py
    if checker.missing_init:
        print("\nTop 10 Directories Missing __init__.py:")
        for i, path in enumerate(sorted(checker.missing_init)[:10]):
            print(f"  {i+1}. {path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 