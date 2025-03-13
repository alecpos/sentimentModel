#!/usr/bin/env python3
"""
Safe Docstring Indentation Fixer

This script automatically fixes indentation issues with docstrings in Python files.
It includes multiple safety features to prevent damage to your codebase:
1. Creates backups of all files before modifying them
2. Validates Python syntax after changes
3. Offers a dry-run mode to preview changes without applying them
4. Only fixes docstring indentation issues, not other code
5. Provides detailed logs of all changes
6. Includes ability to revert changes if issues are found

Usage:
    python fix_docstring_indentation.py [--dry-run] [--no-backup] [--verbose] path/to/file_or_directory

Options:
    --dry-run       Preview changes without modifying files
    --no-backup     Skip creating backup files (not recommended)
    --verbose       Show detailed information about each change
    --recursive     Process directories recursively
"""

import os
import re
import sys
import ast
import argparse
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def backup_file(file_path: str, backup_dir: Optional[str] = None) -> str:
    """Create a backup of a file before modifying it.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store backups (defaults to same directory as file)
    
    Returns:
        Path to the backup file
    """
    backup_path = file_path + ".bak"
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(file_path) + ".bak")
    
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    return backup_path

def validate_python_syntax(code: str, file_path: str) -> bool:
    """Validate that a Python file has correct syntax.
    
    Args:
        code: Python code to validate
        file_path: Path to the file (for error reporting)
    
    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path} at line {e.lineno}, col {e.offset}: {e.msg}")
        return False

def find_class_definitions(code: str) -> List[Tuple[int, int, str]]:
    """Find all class definitions in the code and their indentation levels.
    
    Args:
        code: Python code to analyze
    
    Returns:
        List of tuples (line_number, indentation_level, class_name)
    """
    class_pattern = re.compile(r'^(\s*)class\s+(\w+)')
    
    classes = []
    for i, line in enumerate(code.splitlines(), 1):
        match = class_pattern.match(line)
        if match:
            indent = len(match.group(1))
            class_name = match.group(2)
            classes.append((i, indent, class_name))
    
    return classes

def find_docstring_issues(code: str) -> List[Dict]:
    """Find docstrings with incorrect indentation.
    
    Args:
        code: Python code to analyze
    
    Returns:
        List of dictionaries with information about identified issues
    """
    lines = code.splitlines()
    class_defs = find_class_definitions(code)
    issues = []
    
    for line_num, indent, class_name in class_defs:
        if line_num >= len(lines):
            continue
            
        # Check if there's a docstring on the next line
        if line_num < len(lines) and '"""' in lines[line_num]:
            docstring_line = lines[line_num]
            docstring_indent = len(docstring_line) - len(docstring_line.lstrip())
            
            # Check if docstring is improperly indented (should be class indent + 4)
            expected_indent = indent + 4
            
            if docstring_indent != expected_indent:
                issues.append({
                    'line': line_num + 1,  # +1 because line_num is the class definition
                    'class_name': class_name,
                    'current_indent': docstring_indent,
                    'expected_indent': expected_indent,
                    'docstring_start': lines[line_num]
                })
    
    return issues

def fix_docstring_indentation(file_path: str, dry_run: bool = False, 
                             create_backup: bool = True, backup_dir: Optional[str] = None,
                             verbose: bool = False) -> Tuple[bool, List[Dict]]:
    """Fix docstring indentation issues in a Python file.
    
    Args:
        file_path: Path to the Python file to fix
        dry_run: Don't make actual changes, just report what would be changed
        create_backup: Create a backup of the file before modifying
        backup_dir: Directory to store backups
        verbose: Show detailed information about each change
    
    Returns:
        Tuple (success, changes) where success is a boolean and changes is a list of changes made
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find indentation issues
        issues = find_docstring_issues(content)
        
        if not issues:
            if verbose:
                logger.info(f"No docstring indentation issues found in {file_path}")
            return True, []
            
        # Create backup if requested
        if create_backup and not dry_run:
            backup_file(file_path, backup_dir)
        
        # Apply fixes
        lines = content.splitlines()
        modified_lines = lines.copy()
        changes = []
        
        for issue in issues:
            line_idx = issue['line'] - 1  # Convert to 0-indexed
            if line_idx >= len(lines):
                continue
                
            old_line = lines[line_idx]
            new_line = ' ' * issue['expected_indent'] + old_line.lstrip()
            modified_lines[line_idx] = new_line
            
            # Find the end of the docstring block
            in_docstring = True
            current_idx = line_idx
            
            while in_docstring and current_idx < len(lines) - 1:
                current_idx += 1
                current_line = lines[current_idx]
                
                # Check if this line has the closing triple quotes
                if '"""' in current_line and current_line.strip() != '"""':
                    # Closing quotes are on the same line as other content
                    in_docstring = False
                    break
                elif current_line.strip() == '"""':
                    # Standalone closing quotes
                    modified_lines[current_idx] = ' ' * issue['expected_indent'] + current_line.lstrip()
                    in_docstring = False
                    break
                else:
                    # This is a content line in the docstring
                    modified_lines[current_idx] = ' ' * issue['expected_indent'] + current_line.lstrip()
            
            changes.append({
                'file': file_path,
                'line': issue['line'],
                'class': issue['class_name'],
                'old_indent': issue['current_indent'],
                'new_indent': issue['expected_indent']
            })
            
            if verbose:
                logger.info(f"Line {issue['line']}: Fixed indentation for docstring in class {issue['class_name']}")
                
        modified_content = '\n'.join(modified_lines)
        
        # Validate that we haven't broken syntax
        if not validate_python_syntax(modified_content, file_path):
            logger.error(f"Fixed version of {file_path} contains syntax errors. Aborting.")
            return False, changes
            
        # Write changes if not in dry run mode
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
                
            logger.info(f"Fixed {len(changes)} docstring indentation issues in {file_path}")
        else:
            logger.info(f"Would fix {len(changes)} docstring indentation issues in {file_path} (dry run)")
            
        return True, changes
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False, []

def process_files(target_path: str, dry_run: bool = False, 
                 create_backup: bool = True, backup_dir: Optional[str] = None,
                 verbose: bool = False, recursive: bool = False) -> Dict:
    """Process Python files to fix docstring indentation issues.
    
    Args:
        target_path: Path to a file or directory
        dry_run: Don't make actual changes, just report what would be changed
        create_backup: Create a backup of files before modifying
        backup_dir: Directory to store backups
        verbose: Show detailed information about each change
        recursive: Process directories recursively
    
    Returns:
        Dictionary with statistics about the changes made
    """
    path = Path(target_path)
    
    if path.is_file():
        if path.suffix != '.py':
            logger.warning(f"Skipping non-Python file: {path}")
            return {
                'files_processed': 0,
                'files_modified': 0,
                'total_changes': 0,
                'errors': 0
            }
        
        files = [path]
    else:
        if recursive:
            files = list(path.glob('**/*.py'))
        else:
            files = list(path.glob('*.py'))
    
    stats = {
        'files_processed': 0,
        'files_modified': 0,
        'total_changes': 0,
        'errors': 0
    }
    
    for file_path in files:
        stats['files_processed'] += 1
        success, changes = fix_docstring_indentation(
            str(file_path), 
            dry_run=dry_run, 
            create_backup=create_backup,
            backup_dir=backup_dir,
            verbose=verbose
        )
        
        if not success:
            stats['errors'] += 1
        
        if changes:
            stats['files_modified'] += 1
            stats['total_changes'] += len(changes)
    
    return stats

def restore_from_backup(backup_path: str) -> bool:
    """Restore a file from its backup.
    
    Args:
        backup_path: Path to the backup file
    
    Returns:
        True if restoration was successful, False otherwise
    """
    original_path = backup_path[:-4]  # Remove .bak extension
    try:
        shutil.copy2(backup_path, original_path)
        logger.info(f"Restored {original_path} from backup")
        return True
    except Exception as e:
        logger.error(f"Failed to restore from backup {backup_path}: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Safely fix docstring indentation issues in Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "target",
        help="Python file or directory to process"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files (not recommended)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Show detailed information about each change"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--backup-dir",
        help="Directory to store backups"
    )
    
    args = parser.parse_args()
    
    # Validate target path
    if not os.path.exists(args.target):
        parser.error(f"Target path does not exist: {args.target}")
    
    logger.info(f"Processing {'recursively ' if args.recursive else ''}{'(DRY RUN)' if args.dry_run else ''}: {args.target}")
    
    stats = process_files(
        args.target,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
        backup_dir=args.backup_dir,
        verbose=args.verbose,
        recursive=args.recursive
    )
    
    # Print summary
    logger.info("=== Summary ===")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Files with fixes: {stats['files_modified']}")
    logger.info(f"Total indentation issues fixed: {stats['total_changes']}")
    
    if stats['errors'] > 0:
        logger.warning(f"Errors encountered: {stats['errors']}")
        return 1
    
    if args.dry_run and stats['total_changes'] > 0:
        logger.info("Run without --dry-run to apply these changes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 