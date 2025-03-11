#!/usr/bin/env python3
"""
Documentation Filter Tool

This script helps filter and find documentation files based on multiple criteria:
- Implementation status (NOT_IMPLEMENTED, PARTIALLY_IMPLEMENTED, IMPLEMENTED)
- File type (markdown, images, PDFs)
- Content pattern matching
- Last modified date
- Directory/path filtering

Usage:
    python filter_docs.py --status NOT_IMPLEMENTED --type md --days 30
    python filter_docs.py --status PARTIALLY_IMPLEMENTED --content "TODO"
    python filter_docs.py --path "technical" --status NOT_IMPLEMENTED
"""

import argparse
import os
import re
import sys
import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DocFilter:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        # Implementation status patterns
        self.status_pattern = re.compile(r'\*\*IMPLEMENTATION STATUS: ([\w_]+)\*\*')
        # Extensions to consider
        self.extensions = {
            'md': ['.md'],
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'pdf': ['.pdf'],
            'all': ['.md', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.placeholder', '.note']
        }

    def get_file_status(self, file_path: str) -> Optional[str]:
        """Extract implementation status from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = self.status_pattern.search(content)
                if match:
                    return match.group(1)
        except (UnicodeDecodeError, IsADirectoryError):
            pass  # Not a text file or is a directory
        return None

    def filter_files(self, 
                    status: Optional[str] = None, 
                    file_type: str = 'all', 
                    content_pattern: Optional[str] = None, 
                    days: Optional[int] = None,
                    path_filter: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Filter files based on multiple criteria.
        
        Args:
            status: Implementation status to filter by (NOT_IMPLEMENTED, PARTIALLY_IMPLEMENTED, IMPLEMENTED)
            file_type: Type of files to filter (md, image, pdf, all)
            content_pattern: Regex pattern to search in file content
            days: Only include files modified in the last X days
            path_filter: Only include files whose path contains this string
            
        Returns:
            Dictionary mapping status to list of matching files
        """
        result = defaultdict(list)
        extensions = self.extensions.get(file_type.lower(), self.extensions['all'])
        
        # Compile content pattern if provided
        content_re = re.compile(content_pattern, re.IGNORECASE) if content_pattern else None
        
        # Calculate cutoff date if days is provided
        cutoff_date = None
        if days is not None:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.base_dir)
                
                # Apply path filter if specified
                if path_filter and path_filter not in rel_path:
                    continue
                
                # Apply file extension filter
                if not any(file.endswith(ext) for ext in extensions):
                    continue
                
                # Apply days filter if specified
                if cutoff_date:
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mod_time < cutoff_date:
                        continue
                
                # Check file content if pattern provided
                if content_re and file.endswith('.md'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if not content_re.search(content):
                                continue
                    except UnicodeDecodeError:
                        continue  # Skip binary files
                
                # Get implementation status
                file_status = self.get_file_status(file_path)
                
                # Apply status filter if specified
                if status and file_status != status:
                    continue
                
                # Add to results
                result[file_status or "UNKNOWN"].append(rel_path)
        
        return result

    def generate_report(self, filter_results: Dict[str, List[str]]) -> str:
        """Generate a formatted report from filter results."""
        total_files = sum(len(files) for files in filter_results.values())
        
        report = [
            f"{Colors.HEADER}{Colors.BOLD}DOCUMENTATION FILTER REPORT{Colors.ENDC}",
            f"{Colors.BOLD}Total matching files: {total_files}{Colors.ENDC}",
            ""
        ]
        
        # Add status sections
        for status, files in sorted(filter_results.items()):
            if not files:
                continue
                
            # Choose color based on status
            color = Colors.RED if status == "NOT_IMPLEMENTED" else \
                   Colors.WARNING if status == "PARTIALLY_IMPLEMENTED" else \
                   Colors.GREEN if status == "IMPLEMENTED" else \
                   Colors.BLUE
            
            report.append(f"{color}{Colors.BOLD}Status: {status} ({len(files)} files){Colors.ENDC}")
            for file in sorted(files):
                report.append(f"  {file}")
            report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Filter documentation files based on various criteria')
    parser.add_argument('--base-dir', default='/Users/alecposner/WITHIN/docs', 
                        help='Base directory to search (default: /Users/alecposner/WITHIN/docs)')
    parser.add_argument('--status', choices=['NOT_IMPLEMENTED', 'PARTIALLY_IMPLEMENTED', 'IMPLEMENTED'],
                        help='Filter by implementation status')
    parser.add_argument('--type', choices=['md', 'image', 'pdf', 'all'], default='all',
                        help='Filter by file type')
    parser.add_argument('--content', help='Filter by content pattern (regex)')
    parser.add_argument('--days', type=int, help='Only include files modified in the last X days')
    parser.add_argument('--path', help='Filter by path containing this string')
    parser.add_argument('--csv', action='store_true', help='Output in CSV format')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics only')
    
    args = parser.parse_args()
    
    doc_filter = DocFilter(args.base_dir)
    results = doc_filter.filter_files(
        status=args.status,
        file_type=args.type,
        content_pattern=args.content,
        days=args.days,
        path_filter=args.path
    )
    
    if args.csv:
        print("status,file_path")
        for status, files in results.items():
            for file in sorted(files):
                print(f"{status},{file}")
    elif args.summary:
        print(f"{'Status':<25} {'Count':<10}")
        print("-" * 35)
        total = 0
        for status, files in sorted(results.items()):
            count = len(files)
            total += count
            print(f"{status:<25} {count:<10}")
        print("-" * 35)
        print(f"{'TOTAL':<25} {total:<10}")
    else:
        print(doc_filter.generate_report(results))

if __name__ == "__main__":
    main() 