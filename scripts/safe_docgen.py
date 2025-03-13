#!/usr/bin/env python3
"""
Safe Docstring Generator Wrapper

This script ensures that docstring generation only adds missing docstrings
and never replaces existing ones, regardless of their quality.

It works by:
1. Parsing Python files to identify elements with and without docstrings
2. Creating a temporary "protected elements" file to protect all existing docstrings
3. Running the docstring generator only on elements that need docstrings

Usage:
    python safe_docgen.py [--apply] [--verbose] path/to/file_or_directory
"""

import os
import sys
import ast
import argparse
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class DocstringScanner:
    """Scanner for finding elements with and without docstrings in Python files."""
    
    def __init__(self, file_path: str):
        """Initialize the scanner.
        
        Args:
            file_path: Path to the Python file to scan
        """
        self.file_path = file_path
        self.elements_with_docstrings = set()
        self.elements_without_docstrings = set()
        
    def scan(self) -> Tuple[Set[str], Set[str]]:
        """Scan the file for elements with and without docstrings.
        
        Returns:
            Tuple of (elements_with_docstrings, elements_without_docstrings)
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Check module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                self.elements_with_docstrings.add(f"{self.file_path}:module")
            else:
                self.elements_without_docstrings.add(f"{self.file_path}:module")
            
            # Visit nodes for functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_docstring = ast.get_docstring(node)
                    
                    element_id = f"{self.file_path}:{class_name}"
                    
                    if class_docstring:
                        self.elements_with_docstrings.add(element_id)
                    else:
                        self.elements_without_docstrings.add(element_id)
                    
                    # Check methods within class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            method_docstring = ast.get_docstring(item)
                            
                            element_id = f"{self.file_path}:{class_name}.{method_name}"
                            
                            if method_docstring:
                                self.elements_with_docstrings.add(element_id)
                            else:
                                self.elements_without_docstrings.add(element_id)
                                
                elif isinstance(node, ast.FunctionDef) and node.parent_field != 'body':
                    # Only include top-level functions, not methods
                    function_name = node.name
                    function_docstring = ast.get_docstring(node)
                    
                    element_id = f"{self.file_path}:{function_name}"
                    
                    if function_docstring:
                        self.elements_with_docstrings.add(element_id)
                    else:
                        self.elements_without_docstrings.add(element_id)
            
            return self.elements_with_docstrings, self.elements_without_docstrings
            
        except Exception as e:
            logger.error(f"Error scanning {self.file_path}: {str(e)}")
            return set(), set()

def create_protected_elements_file(elements_to_protect: Set[str], protected_file_path: Optional[str] = None) -> str:
    """Create a file listing protected elements that should not be modified.
    
    Args:
        elements_to_protect: Set of elements to protect in format 'file_path:element_name'
        protected_file_path: Optional path to write the protected elements file
        
    Returns:
        Path to the created file
    """
    if not protected_file_path:
        fd, protected_file_path = tempfile.mkstemp(suffix='.txt', prefix='protected_docstrings_')
        os.close(fd)
    
    with open(protected_file_path, 'w', encoding='utf-8') as f:
        f.write("# Protected Docstrings - DO NOT MODIFY THESE ELEMENTS\n")
        f.write("# Generated automatically by safe_docgen.py\n\n")
        
        for element in sorted(elements_to_protect):
            f.write(f"{element}\n")
    
    logger.info(f"Created protected elements file with {len(elements_to_protect)} entries: {protected_file_path}")
    return protected_file_path

def run_docstring_generator(file_path: str, protected_file_path: str, apply: bool = False, verbose: bool = False) -> bool:
    """Run the docstring generator with protection for existing docstrings.
    
    Args:
        file_path: Path to the Python file to process
        protected_file_path: Path to the file listing protected elements
        apply: Whether to apply changes or just preview them
        verbose: Whether to show detailed logs
        
    Returns:
        True if successful, False otherwise
    """
    cmd = ["./scripts/mac_docstring_generator.py", file_path, "--protected", protected_file_path]
    
    if apply:
        cmd.append("--apply")
        
    if verbose:
        cmd.append("--verbose")
        
    try:
        logger.info(f"Running docstring generator for {file_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if verbose:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"Error generating docstrings for {file_path}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run docstring generator: {str(e)}")
        return False

def process_file(file_path: str, apply: bool = False, verbose: bool = False) -> bool:
    """Process a single Python file to add missing docstrings while preserving existing ones.
    
    Args:
        file_path: Path to the Python file to process
        apply: Whether to apply changes or just preview them
        verbose: Whether to show detailed logs
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Scan for elements with docstrings
        scanner = DocstringScanner(file_path)
        with_docstrings, without_docstrings = scanner.scan()
        
        if not with_docstrings and not without_docstrings:
            logger.warning(f"Could not analyze {file_path} for docstrings")
            return False
            
        logger.info(f"Found {len(with_docstrings)} elements with docstrings and {len(without_docstrings)} without")
        
        if not without_docstrings:
            logger.info(f"No missing docstrings in {file_path}, skipping")
            return True
            
        # Create protected elements file
        protected_file_path = create_protected_elements_file(with_docstrings)
        
        # Run docstring generator
        result = run_docstring_generator(file_path, protected_file_path, apply, verbose)
        
        # Clean up temporary file
        os.remove(protected_file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def find_python_files(path: str, recursive: bool = True) -> List[str]:
    """Find Python files to process.
    
    Args:
        path: Path to a file or directory
        recursive: Whether to search recursively in directories
        
    Returns:
        List of Python file paths
    """
    if os.path.isfile(path):
        return [path] if path.endswith('.py') else []
        
    if recursive:
        return [str(p) for p in Path(path).rglob('*.py')]
    else:
        return [str(p) for p in Path(path).glob('*.py')]

def process_priority_list(priority_file: str, apply: bool = False, verbose: bool = False) -> Dict:
    """Process files according to priority list.
    
    Args:
        priority_file: Path to the priority configuration file
        apply: Whether to apply changes or just preview them
        verbose: Whether to show detailed logs
        
    Returns:
        Stats dictionary
    """
    if not os.path.exists(priority_file):
        logger.error(f"Priority file not found: {priority_file}")
        return {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0
        }
    
    # Read priority list
    with open(priority_file, 'r', encoding='utf-8') as f:
        priority_entries = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                priority_entries.append(line)
    
    logger.info(f"Processing {len(priority_entries)} priority entries")
    
    # Track stats
    stats = {
        'files_processed': 0,
        'files_successful': 0,
        'files_failed': 0
    }
    
    # Process each priority entry
    processed_files = set()
    
    for entry in priority_entries:
        if os.path.isfile(entry) and entry.endswith('.py'):
            # Single file
            if entry in processed_files:
                logger.info(f"Skipping already processed file: {entry}")
                continue
                
            stats['files_processed'] += 1
            if process_file(entry, apply, verbose):
                stats['files_successful'] += 1
            else:
                stats['files_failed'] += 1
                
            processed_files.add(entry)
            
        elif os.path.isdir(entry):
            # Directory of files
            py_files = find_python_files(entry)
            for py_file in py_files:
                if py_file in processed_files:
                    logger.info(f"Skipping already processed file: {py_file}")
                    continue
                    
                stats['files_processed'] += 1
                if process_file(py_file, apply, verbose):
                    stats['files_successful'] += 1
                else:
                    stats['files_failed'] += 1
                    
                processed_files.add(py_file)
                
        else:
            logger.warning(f"Invalid entry in priority list: {entry}")
    
    return stats

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate docstrings while preserving existing ones",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "target",
        help="Python file or directory to process"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is to preview only)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--priority-file",
        default="configs/docstring_priority.txt",
        help="Path to priority configuration file"
    )
    
    parser.add_argument(
        "--use-priority",
        action="store_true",
        help="Process files according to priority list"
    )
    
    args = parser.parse_args()
    
    if args.use_priority and os.path.exists(args.priority_file):
        logger.info(f"Processing according to priority list: {args.priority_file}")
        stats = process_priority_list(args.priority_file, args.apply, args.verbose)
    else:
        # Process target path
        if os.path.isfile(args.target):
            logger.info(f"Processing single file: {args.target}")
            success = process_file(args.target, args.apply, args.verbose)
            
            stats = {
                'files_processed': 1,
                'files_successful': 1 if success else 0,
                'files_failed': 0 if success else 1
            }
        else:
            logger.info(f"Processing directory: {args.target}")
            py_files = find_python_files(args.target, args.recursive)
            logger.info(f"Found {len(py_files)} Python files")
            
            stats = {
                'files_processed': 0,
                'files_successful': 0,
                'files_failed': 0
            }
            
            for py_file in py_files:
                stats['files_processed'] += 1
                if process_file(py_file, args.apply, args.verbose):
                    stats['files_successful'] += 1
                else:
                    stats['files_failed'] += 1
    
    # Print summary
    logger.info("=== Summary ===")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Files successful: {stats['files_successful']}")
    logger.info(f"Files failed: {stats['files_failed']}")
    
    if not args.apply:
        logger.info("This was a preview only. Run with --apply to make changes.")
        
    return 0 if stats['files_failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 