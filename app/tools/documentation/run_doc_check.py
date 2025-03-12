#!/usr/bin/env python3
"""
Simple script to run the documentation checker without importing the entire app.

DOCUMENTATION STATUS: COMPLETE
"""

import os
import sys
from pathlib import Path

# Add the root directory to sys.path if not already there
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import the documentation checker
from app.tools.documentation.documentation_checker import DocumentationChecker

def main():
    """Run the documentation checker."""
    print("Running documentation check...")
    checker = DocumentationChecker(root_dir='app')
    checker.check_all()
    
    # Generate report
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