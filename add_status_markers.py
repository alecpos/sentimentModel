#!/usr/bin/env python3
"""
Add Implementation Status Markers

This script analyzes documentation files with UNKNOWN status
and adds appropriate implementation status markers based on content analysis.

Usage:
    python add_status_markers.py [--dry-run] [--path PATH] [--status STATUS]
"""

import os
import re
import sys
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple, Set

def get_unknown_files(docs_dir: str, path_filter: Optional[str] = None) -> List[str]:
    """Get all files with UNKNOWN status."""
    cmd = ['python', 'filter_docs.py', '--csv']
    if path_filter:
        cmd.extend(['--path', path_filter])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running filter_docs.py: {result.stderr}")
        return []
    
    unknown_files = []
    for line in result.stdout.strip().split('\n'):
        if line.startswith('UNKNOWN,'):
            file_path = line.split(',', 1)[1].strip()
            unknown_files.append(os.path.join(docs_dir, file_path))
    
    return unknown_files

def extract_sections(content: str) -> Dict[str, str]:
    """
    Extract sections from the document based on Markdown headings.
    Returns a dictionary mapping section names to their content.
    """
    # Find all level 2 headings (##)
    headings = re.findall(r'^## (.+)$', content, re.MULTILINE)
    sections = {}
    
    # Extract content for each section
    for i, heading in enumerate(headings):
        # Find the start position of this section
        section_start = content.find(f"## {heading}")
        
        # Find the end position (either the next section or the end of the document)
        if i < len(headings) - 1:
            next_section = f"## {headings[i+1]}"
            section_end = content.find(next_section, section_start)
        else:
            section_end = len(content)
        
        # Extract the section content
        section_content = content[section_start:section_end]
        sections[heading] = section_content
    
    # If no sections were found, consider the whole document as one section
    if not sections:
        sections["Main"] = content
    
    return sections

def is_reference_section(section_name: str) -> bool:
    """
    Determine if a section is a reference or related section that might
    contain references to other documents.
    """
    reference_section_patterns = [
        r"related\s*documentation",
        r"references",
        r"see\s*also",
        r"resources",
        r"external\s*links",
        r"further\s*reading"
    ]
    
    return any(re.search(pattern, section_name, re.IGNORECASE) 
              for pattern in reference_section_patterns)

def detect_placeholder_patterns(text: str) -> bool:
    """
    Check for placeholder patterns in text.
    Returns True if any patterns are found.
    """
    placeholder_patterns = [
        r"placeholder",
        r"todo",
        r"to\s*be\s*implemented",
        r"to\s*be\s*filled",
        r"coming\s*soon",
        r"under\s*development",
        r"work\s*in\s*progress",
        r"<[^>]*>"  # HTML-like tags often indicate placeholders
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) 
              for pattern in placeholder_patterns)

def calculate_completeness_score(content: str) -> int:
    """
    Calculate a completeness score based on various indicators.
    Returns a score where higher values indicate more complete documentation.
    """
    score = 0
    
    # Length-based indicators
    if len(content.strip()) > 5000:
        score += 3  # Very long documents are likely complete
    elif len(content.strip()) > 2000:
        score += 2  # Substantial content
    elif len(content.strip()) > 1000:
        score += 1  # Moderate content
    
    # Structure indicators
    heading_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
    score += min(heading_count // 3, 2)  # More structured documents get higher scores
    
    paragraph_count = content.count('\n\n')
    score += min(paragraph_count // 10, 2)  # More paragraphs suggest more content
    
    # Content indicators
    code_block_count = len(re.findall(r'```', content)) // 2  # Each block has start and end markers
    score += min(code_block_count, 2)  # Code examples suggest implementation details
    
    link_count = len(re.findall(r'\[\w+\]', content))
    score += min(link_count // 2, 1)  # References to other content
    
    # Special content indicators
    if re.search(r'table|---|:-+:|', content, re.IGNORECASE):
        score += 1  # Tables suggest organized data
    
    if re.search(r'!\[.*\]\(.*\)', content):
        score += 1  # Images suggest diagrams or screenshots
    
    return score

def analyze_file(file_path: str) -> str:
    """
    Analyze a file and suggest an implementation status with
    context-aware placeholder detection and section-specific analysis.
    
    This uses enhanced heuristics to determine if a file is:
    - NOT_IMPLEMENTED: Very little content, clear placeholder text
    - PARTIALLY_IMPLEMENTED: Some content but not complete
    - IMPLEMENTED: Appears to be complete
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Short or empty files are NOT_IMPLEMENTED
        if len(content.strip()) < 200:
            return "NOT_IMPLEMENTED"
        
        # Extract the document title
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else os.path.basename(file_path)
        
        # Extract sections from the document
        sections = extract_sections(content)
        
        # Analyze sections separately
        section_scores = {}
        placeholder_count = 0
        
        for section_name, section_content in sections.items():
            # Check if this is a reference section
            is_reference = is_reference_section(section_name)
            
            # Only count placeholders in non-reference sections
            if not is_reference and detect_placeholder_patterns(section_content):
                placeholder_count += 1
            
            # Calculate completeness score for this section
            section_scores[section_name] = calculate_completeness_score(section_content)
        
        # Calculate overall completeness score
        completeness_score = sum(section_scores.values())
        
        # More sections generally indicate more complete documentation
        completeness_score += len(sections)
        
        # Very comprehensive documents should be considered IMPLEMENTED
        # even with some placeholder markers (they might be references to future work)
        if completeness_score > 100:
            return "IMPLEMENTED"
        
        # Make the final decision based on scores
        if placeholder_count > 2 and completeness_score < 20:
            return "NOT_IMPLEMENTED"
        elif placeholder_count == 0 and completeness_score >= 6:
            return "IMPLEMENTED"
        elif placeholder_count <= 3 and completeness_score >= 30:
            return "IMPLEMENTED"  # More lenient with placeholders for large documents
        elif completeness_score >= 3:
            return "PARTIALLY_IMPLEMENTED"
        else:
            return "NOT_IMPLEMENTED"
        
    except (UnicodeDecodeError, IsADirectoryError):
        # Binary or other non-text files
        return "NOT_IMPLEMENTED"

def add_status_marker(file_path: str, status: str, dry_run: bool = False) -> bool:
    """
    Add implementation status marker to the file.
    
    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has a status marker
        if re.search(r'\*\*IMPLEMENTATION STATUS: [\w_]+\*\*', content):
            print(f"   Already has status marker: {file_path}")
            return False
        
        # Find the title (a line starting with # )
        title_match = re.search(r'^# .+$', content, re.MULTILINE)
        if not title_match:
            print(f"   No title found, skipping: {file_path}")
            return False
        
        title_end = title_match.end()
        
        # Insert status marker after the title
        new_content = (
            content[:title_end] + 
            f"\n\n**IMPLEMENTATION STATUS: {status}**\n" + 
            content[title_end:]
        )
        
        if dry_run:
            print(f"   Would add '{status}' status to: {file_path}")
            return True
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"   Added '{status}' status to: {file_path}")
        return True
    
    except (UnicodeDecodeError, IsADirectoryError):
        print(f"   Error: Cannot process file: {file_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Add implementation status markers to documentation files')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without making changes')
    parser.add_argument('--path', help='Only process files in this path')
    parser.add_argument('--status', choices=['NOT_IMPLEMENTED', 'PARTIALLY_IMPLEMENTED', 'IMPLEMENTED'],
                       help='Force this status for all files (default: analyze each file)')
    parser.add_argument('--docs-dir', default='/Users/alecposner/WITHIN/docs',
                       help='Base directory containing documentation files')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Show detailed diagnostic information about file analysis')
    
    args = parser.parse_args()
    
    print("Finding files with UNKNOWN implementation status...")
    unknown_files = get_unknown_files(args.docs_dir, args.path)
    print(f"Found {len(unknown_files)} files with UNKNOWN status.")
    
    if not unknown_files:
        print("No files to process.")
        return
    
    modified_count = 0
    
    for file_path in unknown_files:
        rel_path = os.path.relpath(file_path, args.docs_dir)
        print(f"Processing: {rel_path}")
        
        # Use specified status or analyze the file
        if args.status:
            status = args.status
        else:
            # Run diagnostic analysis if requested
            if args.diagnostic:
                status = analyze_file_with_diagnostics(file_path)
            else:
                status = analyze_file(file_path)
        
        if add_status_marker(file_path, status, args.dry_run):
            modified_count += 1
    
    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n{action} {modified_count} of {len(unknown_files)} files.")
    
    if not args.dry_run and modified_count > 0:
        print("\nTaking a new documentation snapshot...")
        subprocess.run(['python', 'track_doc_progress.py', '--record'], 
                      check=False, capture_output=True)
        print("Done.")

def analyze_file_with_diagnostics(file_path: str) -> str:
    """
    Analyze a file with detailed diagnostic output.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Short or empty files are NOT_IMPLEMENTED
        if len(content.strip()) < 200:
            print("   DIAGNOSTIC: File is too short, marking as NOT_IMPLEMENTED")
            return "NOT_IMPLEMENTED"
        
        # Extract the document title
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else os.path.basename(file_path)
        print(f"   DIAGNOSTIC: Document title: {title}")
        
        # Extract sections from the document
        sections = extract_sections(content)
        print(f"   DIAGNOSTIC: Found {len(sections)} sections: {', '.join(sections.keys())}")
        
        # Analyze sections separately
        section_scores = {}
        placeholder_count = 0
        
        for section_name, section_content in sections.items():
            # Check if this is a reference section
            is_reference = is_reference_section(section_name)
            print(f"   DIAGNOSTIC: Section '{section_name}' is {'a reference' if is_reference else 'a regular'} section")
            
            # Only count placeholders in non-reference sections
            if not is_reference:
                has_placeholder = detect_placeholder_patterns(section_content)
                if has_placeholder:
                    placeholder_count += 1
                    print(f"   DIAGNOSTIC: Found placeholder in section '{section_name}'")
            
            # Calculate completeness score for this section
            section_score = calculate_completeness_score(section_content)
            section_scores[section_name] = section_score
            print(f"   DIAGNOSTIC: Section '{section_name}' has completeness score {section_score}")
        
        # Calculate overall completeness score
        completeness_score = sum(section_scores.values())
        print(f"   DIAGNOSTIC: Sum of section scores: {completeness_score}")
        
        # More sections generally indicate more complete documentation
        completeness_score += len(sections)
        print(f"   DIAGNOSTIC: Final completeness score (with section bonus): {completeness_score}")
        print(f"   DIAGNOSTIC: Placeholder count: {placeholder_count}")
        
        # Very comprehensive documents should be considered IMPLEMENTED
        # even with some placeholder markers (they might be references to future work)
        if completeness_score > 100:
            print("   DIAGNOSTIC: Very high completeness score overrides placeholders")
            return "IMPLEMENTED"
        
        # Make the final decision based on scores
        if placeholder_count > 2 and completeness_score < 20:
            print("   DIAGNOSTIC: Too many placeholders and low completeness")
            return "NOT_IMPLEMENTED"
        elif placeholder_count == 0 and completeness_score >= 6:
            print("   DIAGNOSTIC: No placeholders and high completeness")
            return "IMPLEMENTED"
        elif placeholder_count <= 3 and completeness_score >= 30:
            print("   DIAGNOSTIC: Low placeholders (relative to size) and good completeness")
            return "IMPLEMENTED"  # More lenient with placeholders for large documents
        elif completeness_score >= 3:
            print("   DIAGNOSTIC: Moderate completeness")
            return "PARTIALLY_IMPLEMENTED"
        else:
            print("   DIAGNOSTIC: Low completeness")
            return "NOT_IMPLEMENTED"
        
    except (UnicodeDecodeError, IsADirectoryError):
        print("   DIAGNOSTIC: File could not be processed")
        return "NOT_IMPLEMENTED"

if __name__ == "__main__":
    main() 