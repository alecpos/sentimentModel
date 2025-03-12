#!/usr/bin/env python
"""
Documentation Coverage Checker

This script analyzes the project's Python modules to check if all exported classes/functions
in __init__.py files are properly documented in corresponding README.md files.

Usage:
    python check_documentation_coverage.py [directory]

If no directory is specified, the script will check the entire app directory.
"""

import os
import sys
import re
import ast
from typing import Dict, List, Set, Tuple, Optional

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
END = "\033[0m"

def extract_all_list(init_file: str) -> Set[str]:
    """
    Extract the __all__ list from a Python __init__.py file.
    
    Args:
        init_file: Path to the __init__.py file
        
    Returns:
        Set of strings representing the exported symbols
    """
    try:
        with open(init_file, 'r') as f:
            file_content = f.read()
            
        # Parse the Python file
        tree = ast.parse(file_content)
        
        # Find the __all__ assignment
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            # Extract string literals from the list
                            return {
                                element.value for element in node.value.elts 
                                if isinstance(element, ast.Str)
                            }
                        
    except Exception as e:
        print(f"Error parsing {init_file}: {str(e)}")
    
    return set()

def extract_documented_classes(readme_file: str) -> Set[str]:
    """
    Extract class names documented in a README.md file.
    
    Args:
        readme_file: Path to the README.md file
        
    Returns:
        Set of strings representing documented class names
    """
    documented_classes = set()
    
    if not os.path.exists(readme_file):
        return documented_classes
        
    with open(readme_file, 'r') as f:
        content = f.read()
    
    # Regular expressions to find documented classes
    
    # Find classes documented with ### ClassName header
    header_pattern = re.compile(r'###\s+`?(\w+)`?')
    documented_classes.update(header_pattern.findall(content))
    
    # Find classes documented with `ClassName` in text
    backtick_pattern = re.compile(r'`([A-Z]\w+)`')
    documented_classes.update(backtick_pattern.findall(content))
    
    # Find classes in code blocks: from module import ClassName
    import_pattern = re.compile(r'from\s+[\w\.]+\s+import\s+([A-Z]\w+)')
    documented_classes.update(import_pattern.findall(content))
    
    # Find class instantiations in code blocks: instance = ClassName()
    instantiation_pattern = re.compile(r'=\s+([A-Z]\w+)\(')
    documented_classes.update(instantiation_pattern.findall(content))
    
    return documented_classes

def analyze_directory(directory: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Analyze a directory to find __init__.py and README.md pairs and check documentation.
    
    Args:
        directory: Root directory to analyze
        
    Returns:
        Dictionary with documentation status for each module
    """
    results = {}
    
    for root, dirs, files in os.walk(directory):
        if '__init__.py' in files:
            init_file = os.path.join(root, '__init__.py')
            readme_file = os.path.join(root, 'README.md')
            
            # Skip if no README.md exists (we can't validate coverage)
            if not os.path.exists(readme_file):
                continue
                
            module_name = os.path.relpath(root, directory)
            exported_symbols = extract_all_list(init_file)
            documented_classes = extract_documented_classes(readme_file)
            
            # Check which exported symbols are documented
            documented = exported_symbols.intersection(documented_classes)
            missing = exported_symbols - documented_classes
            
            results[module_name] = {
                'exported': sorted(list(exported_symbols)),
                'documented': sorted(list(documented)),
                'missing': sorted(list(missing))
            }
    
    return results

def calculate_coverage(results: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Calculate documentation coverage for each module.
    
    Args:
        results: Analysis results from analyze_directory
        
    Returns:
        Dictionary mapping module names to coverage percentages
    """
    coverage = {}
    
    for module, data in results.items():
        exported_count = len(data['exported'])
        documented_count = len(data['documented'])
        
        if exported_count > 0:
            coverage[module] = (documented_count / exported_count) * 100
        else:
            coverage[module] = 100.0  # No exports means 100% coverage
            
    return coverage

def print_results(results: Dict[str, Dict[str, List[str]]], coverage: Dict[str, float]) -> None:
    """
    Print the documentation coverage results in a readable format.
    
    Args:
        results: Analysis results from analyze_directory
        coverage: Coverage percentages from calculate_coverage
    """
    print(f"\n{BOLD}DOCUMENTATION COVERAGE ANALYSIS{END}")
    print("-" * 80)
    
    # Sort modules by coverage (ascending)
    sorted_modules = sorted(coverage.items(), key=lambda x: x[1])
    
    total_exported = sum(len(data['exported']) for data in results.values())
    total_documented = sum(len(data['documented']) for data in results.values())
    overall_coverage = (total_documented / total_exported * 100) if total_exported > 0 else 100
    
    for module, cov in sorted_modules:
        # Determine color based on coverage
        if cov < 50:
            color = RED
        elif cov < 80:
            color = YELLOW
        else:
            color = GREEN
            
        exported_count = len(results[module]['exported'])
        documented_count = len(results[module]['documented'])
        
        print(f"{BOLD}{module}{END}: {color}{cov:.1f}%{END} coverage ({documented_count}/{exported_count} symbols documented)")
        
        if results[module]['missing']:
            print(f"  {YELLOW}Missing documentation for:{END}")
            for missing in results[module]['missing']:
                print(f"    - {missing}")
    
    print("-" * 80)
    print(f"{BOLD}Overall coverage:{END} {overall_coverage:.1f}% ({total_documented}/{total_exported} symbols documented)")
    
    # Recommendation based on overall coverage
    if overall_coverage < 60:
        print(f"\n{RED}URGENT:{END} Documentation coverage is very low. Consider prioritizing documentation improvements.")
    elif overall_coverage < 80:
        print(f"\n{YELLOW}ATTENTION:{END} Documentation coverage needs improvement in several modules.")
    else:
        print(f"\n{GREEN}GOOD:{END} Documentation coverage is reasonable, but there's still room for improvement.")

def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'app'
        
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
        
    print(f"Analyzing documentation coverage in '{directory}'...")
    results = analyze_directory(directory)
    coverage = calculate_coverage(results)
    print_results(results, coverage)

if __name__ == "__main__":
    main() 