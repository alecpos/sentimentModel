#!/usr/bin/env python3
"""
Tool for verifying alignment between __init__.py exports and README documentation.

This script checks for proper alignment between exported symbols in __init__.py
files and their documentation in corresponding README.md files. It helps ensure
that all public APIs are properly documented and that terminology is consistent.

Features:
- Extracts __all__ list from __init__.py files
- Parses README.md files to detect documented classes/functions
- Identifies misalignments between exports and documentation
- Reports undocumented exports and documented items not in exports
- Checks for terminology consistency between code and documentation

Example usage:
    python verify_documentation_alignment.py app/models/ml

Output format:
    Module: path/to/module
    - Missing documentation: [list of undocumented exports]
    - Extra documentation: [list of documented items not in exports]
    - Terminology mismatches: [list of terms with inconsistent naming]
"""

import os
import re
import ast
import argparse
import logging
from typing import Dict, List, Set, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

class AlignmentChecker:
    """
    Checks alignment between __init__.py exports and README.md documentation.
    
    This class analyzes Python modules to ensure proper documentation coverage
    by comparing exported symbols with documentation in README files.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the alignment checker.
        
        Args:
            base_dir: Base directory to start checking from
        """
        self.base_dir = os.path.abspath(base_dir)
        self.results = {}
        
    def find_modules(self) -> List[str]:
        """
        Find all modules that have both __init__.py and README.md files.
        
        Returns:
            List of paths to directories containing both files
        """
        modules = []
        
        for root, _, files in os.walk(self.base_dir):
            if "__init__.py" in files and "README.md" in files:
                rel_path = os.path.relpath(root, self.base_dir)
                modules.append(rel_path if rel_path != "." else "")
                
        return modules
    
    def extract_all_list(self, init_file: str) -> Set[str]:
        """
        Extract the __all__ list from an __init__.py file.
        
        Args:
            init_file: Path to the __init__.py file
            
        Returns:
            Set of exported symbol names
        """
        with open(init_file, 'r') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            
            # Try to find __all__ assignment
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, ast.List):
                                return {
                                    element.value
                                    for element in node.value.elts
                                    if isinstance(element, ast.Str)
                                }
            
            # If no __all__ list, collect all exported symbols (classes and functions)
            exports = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    # Skip private members (starting with _)
                    if not node.name.startswith('_'):
                        exports.add(node.name)
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        if not name.name.startswith('_'):
                            exports.add(name.name if name.asname is None else name.asname)
                elif isinstance(node, ast.ImportFrom):
                    if node.module is not None:
                        for name in node.names:
                            if not name.name.startswith('_'):
                                exports.add(name.name if name.asname is None else name.asname)
            
            return exports
            
        except SyntaxError:
            logger.error(f"Syntax error in {init_file}")
            return set()
            
    def extract_documented_items(self, readme_file: str) -> Set[str]:
        """
        Extract documented class and function names from a README.md file.
        
        Args:
            readme_file: Path to the README.md file
            
        Returns:
            Set of documented item names
        """
        with open(readme_file, 'r') as f:
            content = f.read()
            
        # Patterns to match documented items
        patterns = [
            # Match class/function definitions with backticks and parameters
            r'`([A-Za-z0-9_]+)`(?:\s*\(|\s*:)',
            # Match class/function names in headers
            r'#{1,6}\s+([A-Za-z0-9_]+)\s*(?:\(|\n)',
            # Match class/function names in bold
            r'\*\*([A-Za-z0-9_]+)\*\*(?:\s*\(|\s*:)',
            # Match items with explicit class/function keywords
            r'(?:class|function)\s+`?([A-Za-z0-9_]+)`?',
            # Match code blocks with class/function definitions
            r'```python\s*(?:.*\n)*?(?:class|def)\s+([A-Za-z0-9_]+)\s*\(',
        ]
        
        documented_items = set()
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                item_name = match.group(1)
                documented_items.add(item_name)
                
        return documented_items
    
    def check_terminology_consistency(self, 
                                     init_file: str, 
                                     readme_file: str, 
                                     exports: Set[str]) -> List[str]:
        """
        Check for terminology consistency between code and documentation.
        
        Args:
            init_file: Path to the __init__.py file
            readme_file: Path to the README.md file
            exports: Set of exported symbol names
            
        Returns:
            List of inconsistencies found
        """
        inconsistencies = []
        
        # Extract docstrings from __init__.py
        with open(init_file, 'r') as f:
            init_content = f.read()
            
        with open(readme_file, 'r') as f:
            readme_content = f.read()
            
        try:
            tree = ast.parse(init_content)
            
            # Check class and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in exports:
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Extract first sentence as summary
                        summary = docstring.split('.')[0].strip()
                        
                        # Check if the class/function is documented in README with
                        # a different description
                        class_pattern = rf'(?:`|#{1,6}\s+|class\s+)({node.name})(?:`|\s*\(|\s*:|\n)'
                        readme_match = re.search(class_pattern, readme_content)
                        
                        if readme_match:
                            # Find the surrounding text (up to 200 chars)
                            pos = readme_match.start()
                            context_start = max(0, pos - 100)
                            context_end = min(len(readme_content), pos + 200)
                            context = readme_content[context_start:context_end]
                            
                            # Check if summary appears in context
                            if summary and len(summary) > 10 and summary not in context:
                                inconsistencies.append(f"{node.name}: Different descriptions in code vs README")
        except SyntaxError:
            logger.error(f"Syntax error in {init_file}")
            
        return inconsistencies
                                    
    def check_alignment(self) -> Dict[str, Dict]:
        """
        Check alignment for all modules.
        
        Returns:
            Dictionary with alignment results for each module
        """
        modules = self.find_modules()
        results = {}
        
        for module in modules:
            module_path = os.path.join(self.base_dir, module)
            init_file = os.path.join(module_path, "__init__.py")
            readme_file = os.path.join(module_path, "README.md")
            
            # Extract exports and documented items
            exports = self.extract_all_list(init_file)
            documented_items = self.extract_documented_items(readme_file)
            
            # Check for misalignments
            missing_docs = exports - documented_items
            extra_docs = documented_items - exports
            
            # Check terminology consistency
            inconsistencies = self.check_terminology_consistency(init_file, readme_file, exports)
            
            # Store results
            results[module or '.'] = {
                'exports': exports,
                'documented': documented_items,
                'missing_docs': missing_docs,
                'extra_docs': extra_docs,
                'inconsistencies': inconsistencies,
                'coverage': len(documented_items.intersection(exports)) / len(exports) if exports else 1.0
            }
            
        self.results = results
        return results
    
    def print_results(self) -> None:
        """
        Print the alignment check results in a readable format.
        """
        print(f"\n{BOLD}DOCUMENTATION ALIGNMENT ANALYSIS{RESET}")
        print("-" * 80)
        
        # Sort modules by coverage (ascending)
        sorted_modules = sorted(
            self.results.items(), 
            key=lambda x: x[1]['coverage']
        )
        
        for module, result in sorted_modules:
            coverage_pct = result['coverage'] * 100
            
            # Determine color based on coverage
            if coverage_pct < 70:
                color = RED
            elif coverage_pct < 90:
                color = YELLOW
            else:
                color = GREEN
                
            print(f"{color}{module}{RESET}: {color}{coverage_pct:.1f}%{RESET} alignment "
                  f"({len(result['documented'].intersection(result['exports']))}/{len(result['exports'])} "
                  f"symbols aligned)")
            
            if result['missing_docs']:
                print(f"  {YELLOW}Missing documentation for:{RESET}")
                for item in sorted(result['missing_docs']):
                    print(f"    - {item}")
                    
            if result['extra_docs']:
                print(f"  {YELLOW}Documentation for non-exported symbols:{RESET}")
                for item in sorted(result['extra_docs']):
                    print(f"    - {item}")
                    
            if result['inconsistencies']:
                print(f"  {YELLOW}Terminology inconsistencies:{RESET}")
                for item in result['inconsistencies']:
                    print(f"    - {item}")
                    
        print("-" * 80)
        
        # Calculate overall statistics
        total_exports = sum(len(r['exports']) for r in self.results.values())
        total_aligned = sum(len(r['exports'].intersection(r['documented'])) for r in self.results.values())
        overall_coverage = total_aligned / total_exports if total_exports else 1.0
        
        # Overall summary
        color = GREEN if overall_coverage >= 0.9 else (YELLOW if overall_coverage >= 0.7 else RED)
        print(f"{BOLD}Overall alignment:{RESET} {color}{overall_coverage * 100:.1f}%{RESET} "
              f"({total_aligned}/{total_exports} symbols aligned)")
        
        # Recommendations
        if overall_coverage < 1.0:
            print(f"\n{BOLD}RECOMMENDATIONS:{RESET}")
            if overall_coverage < 0.7:
                print(f"{RED}CRITICAL: Documentation alignment is severely lacking. Focus on documenting "
                      f"exported symbols in READMEs.{RESET}")
            elif overall_coverage < 0.9:
                print(f"{YELLOW}WARNING: Documentation alignment needs improvement. Address missing "
                      f"documentation in high-priority modules.{RESET}")
            else:
                print(f"{GREEN}GOOD: Documentation alignment is reasonable, but there's still room for "
                      f"improvement.{RESET}")
                
def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Verify alignment between __init__.py exports and README.md documentation."
    )
    parser.add_argument(
        "directory",
        help="Directory to analyze for documentation alignment"
    )
    args = parser.parse_args()
    
    logger.info(f"Analyzing documentation alignment in '{args.directory}'...")
    
    checker = AlignmentChecker(args.directory)
    checker.check_alignment()
    checker.print_results()

if __name__ == "__main__":
    main() 