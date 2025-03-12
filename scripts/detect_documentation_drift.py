#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Documentation Drift Detector

This script analyzes git commits to detect potential documentation drift - when code changes 
but corresponding documentation doesn't change. It helps identify documentation that might 
be outdated due to code evolution.

Usage:
    python detect_documentation_drift.py --since="2 weeks ago" --path=app/models/ml
    python detect_documentation_drift.py --commits=10 --path=app/models --report=drift_report.json
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('doc_drift_detector')

def run_git_command(command: List[str]) -> str:
    """Run a git command and return its output.
    
    Args:
        command: List of command components
        
    Returns:
        Command output as string
        
    Raises:
        RuntimeError: If the command fails
    """
    try:
        result = subprocess.run(
            ["git"] + command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        logger.error(f"Command output: {e.stderr}")
        raise RuntimeError(f"Git command failed: {e}")


def get_recent_commits(since: Optional[str] = None, count: Optional[int] = None, path: Optional[str] = None) -> List[str]:
    """Get list of recent commit hashes.
    
    Args:
        since: Get commits since this time (e.g., "2 weeks ago")
        count: Number of commits to retrieve
        path: Path to limit commits to
        
    Returns:
        List of commit hashes
    """
    command = ["log", "--pretty=format:%H"]
    
    if since:
        command.extend(["--since", since])
    if count:
        command.extend(["-n", str(count)])
    if path:
        command.append("--")
        command.append(path)
    
    output = run_git_command(command)
    if not output:
        return []
    
    return output.split("\n")


def get_changed_files(commit: str) -> Dict[str, str]:
    """Get files changed in a commit with their change type.
    
    Args:
        commit: Commit hash
        
    Returns:
        Dict mapping filenames to change types (A/M/D)
    """
    command = ["show", "--name-status", "--format=", commit]
    output = run_git_command(command)
    
    changed_files = {}
    for line in output.split("\n"):
        if not line.strip():
            continue
        
        parts = line.split("\t")
        if len(parts) >= 2:
            change_type = parts[0]  # A (added), M (modified), D (deleted), etc.
            file_path = parts[1]
            if file_path.endswith(".py"):
                changed_files[file_path] = change_type
    
    return changed_files


def get_changed_functions(commit: str, file_path: str) -> Set[str]:
    """Get names of functions changed in a commit.
    
    Args:
        commit: Commit hash
        file_path: Path to the file
        
    Returns:
        Set of function names that were modified
    """
    # Get the diff for this specific file in this commit
    command = ["show", "--format=", "--unified=0", f"{commit}:{file_path}"]
    try:
        diff_output = run_git_command(command)
    except RuntimeError:
        # File might not exist in this commit
        return set()
    
    # Extract function definitions from the file
    function_names = set()
    
    # Pattern to match function definitions (both regular and class methods)
    func_pattern = r'^\s*def\s+([a-zA-Z0-9_]+)\s*\('
    class_method_pattern = r'^\s+def\s+([a-zA-Z0-9_]+)\s*\('
    
    for line in diff_output.split("\n"):
        # Check for function definitions
        match = re.search(func_pattern, line) or re.search(class_method_pattern, line)
        if match:
            function_names.add(match.group(1))
    
    return function_names


def is_docstring_updated(commit: str, file_path: str) -> bool:
    """Check if docstrings were updated in this commit.
    
    Args:
        commit: Commit hash
        file_path: Path to the file
        
    Returns:
        True if docstrings were updated, False otherwise
    """
    command = ["show", commit, "--", file_path]
    diff_output = run_git_command(command)
    
    # Check for docstring patterns in the diff
    docstring_patterns = [
        r'^\+\s*"""',          # Adding triple-quoted string
        r'^\+\s*\'\'\'',        # Adding triple-quoted string (single quotes)
        r'^\+\s*Args:',         # Adding Args section
        r'^\+\s*Returns:',      # Adding Returns section
        r'^\+\s*Raises:',       # Adding Raises section
        r'^\+\s*Examples:',     # Adding Examples section
    ]
    
    for pattern in docstring_patterns:
        if re.search(pattern, diff_output, re.MULTILINE):
            return True
    
    return False


def analyze_drift(
    commits: List[str],
    max_drift_score: float = 1.0,
    ignore_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze documentation drift across specified commits.
    
    Args:
        commits: List of commit hashes to analyze
        max_drift_score: Maximum drift score (higher means more drift)
        ignore_paths: List of path patterns to ignore
        
    Returns:
        Dictionary with drift analysis results
    """
    ignore_paths = ignore_paths or []
    
    # Track files and their drift score
    drift_data = {}
    total_changes = 0
    
    for commit in commits:
        # Get commit info
        commit_info = run_git_command(["show", "-s", "--format=%an|%at|%s", commit])
        author, timestamp, subject = commit_info.split("|", 2)
        commit_time = datetime.fromtimestamp(int(timestamp))
        
        # Get changed files
        changed_files = get_changed_files(commit)
        
        for file_path, change_type in changed_files.items():
            # Skip deleted files and non-Python files
            if change_type == "D" or not file_path.endswith(".py"):
                continue
                
            # Skip ignored paths
            if any(re.search(pattern, file_path) for pattern in ignore_paths):
                continue
            
            # Initialize file data if needed
            if file_path not in drift_data:
                drift_data[file_path] = {
                    "changes": 0,
                    "docstring_updates": 0,
                    "last_code_change": None,
                    "last_doc_change": None,
                    "drift_score": 0.0,
                    "changed_functions": set()
                }
            
            # Update file data
            file_data = drift_data[file_path]
            file_data["changes"] += 1
            total_changes += 1
            
            # Check if docstrings were updated
            doc_updated = is_docstring_updated(commit, file_path)
            if doc_updated:
                file_data["docstring_updates"] += 1
                file_data["last_doc_change"] = commit_time.isoformat()
            
            # Update last code change timestamp
            file_data["last_code_change"] = commit_time.isoformat()
            
            # Get changed functions
            changed_funcs = get_changed_functions(commit, file_path)
            file_data["changed_functions"].update(changed_funcs)
    
    # Calculate drift scores
    for file_path, file_data in drift_data.items():
        if file_data["changes"] > 0:
            # Calculate drift score (0 = no drift, 1 = complete drift)
            if file_data["docstring_updates"] == 0:
                # No docstring updates at all
                file_data["drift_score"] = max_drift_score
            else:
                # Some docstring updates
                file_data["drift_score"] = 1.0 - (file_data["docstring_updates"] / file_data["changes"])
        
        # Convert changed_functions set to list for JSON serialization
        file_data["changed_functions"] = list(file_data["changed_functions"])
    
    # Overall stats
    results = {
        "analyzed_commits": len(commits),
        "total_changes": total_changes,
        "files_with_drift": sum(1 for data in drift_data.values() if data["drift_score"] > 0.5),
        "average_drift_score": sum(data["drift_score"] for data in drift_data.values()) / len(drift_data) if drift_data else 0,
        "timestamp": datetime.now().isoformat(),
        "file_data": drift_data
    }
    
    # Sort files by drift score
    results["high_drift_files"] = sorted(
        [(path, data["drift_score"]) for path, data in drift_data.items() if data["drift_score"] > 0.5],
        key=lambda x: x[1],
        reverse=True
    )
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect documentation drift when code changes but docs don't update"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--since", type=str, help="Analyze commits since this date (e.g., '2 weeks ago')")
    group.add_argument("--commits", type=int, help="Number of recent commits to analyze")
    
    parser.add_argument("--path", type=str, default="app", help="Path to analyze")
    parser.add_argument("--ignore", type=str, nargs="*", default=["test_", "tests/"], help="Patterns to ignore")
    parser.add_argument("--report", type=str, help="Path to save drift report (JSON)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Drift score threshold for warnings")
    
    args = parser.parse_args()
    
    try:
        # Get commits to analyze
        if args.since:
            commits = get_recent_commits(since=args.since, path=args.path)
        else:
            commits = get_recent_commits(count=args.commits, path=args.path)
        
        if not commits:
            logger.warning("No commits found to analyze")
            return 0
        
        logger.info(f"Analyzing {len(commits)} commits for documentation drift")
        
        # Analyze drift
        results = analyze_drift(commits, ignore_paths=args.ignore)
        
        # Print summary
        print(f"\nDocumentation Drift Analysis")
        print(f"=============================")
        print(f"Analyzed {results['analyzed_commits']} commits with {results['total_changes']} changes")
        print(f"Average drift score: {results['average_drift_score']:.2f}")
        print(f"Files with significant drift: {results['files_with_drift']}\n")
        
        # Print high drift files
        if results['high_drift_files']:
            print("Files with high documentation drift:")
            for file_path, drift_score in results['high_drift_files']:
                print(f"  {file_path}: {drift_score:.2f}")
        else:
            print("No files with significant documentation drift found.")
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Drift report saved to {args.report}")
        
        # Exit with error if drift exceeds threshold
        if results['average_drift_score'] > args.threshold:
            logger.warning(f"Documentation drift ({results['average_drift_score']:.2f}) exceeds threshold ({args.threshold})")
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Error detecting documentation drift: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 