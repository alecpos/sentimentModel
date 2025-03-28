#!/usr/bin/env python3
"""
Script to automate Git commits while preserving file timestamps.

This script helps maintain accurate development timelines when uploading projects
to GitHub by creating commits based on file modification dates.
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FileTimestamp:
    """Data class to store file timestamp information."""
    path: str
    mtime: float
    ctime: float
    size: int

def clean_file_path(path: str) -> str:
    """Clean and normalize a file path.
    
    Args:
        path: The file path to clean
        
    Returns:
        Cleaned and normalized file path
    """
    # Remove quotes and newlines
    path = path.strip('"').replace('\\n', '')
    
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Remove any double slashes
    path = re.sub(r'/+', '/', path)
    
    # Remove any leading/trailing whitespace
    path = path.strip()
    
    # Remove any duplicate directory names
    parts = path.split('/')
    cleaned_parts = []
    for part in parts:
        if part not in cleaned_parts:
            cleaned_parts.append(part)
    path = '/'.join(cleaned_parts)
    
    # Remove any non-printable characters
    path = ''.join(char for char in path if char.isprintable())
    
    return path

def validate_file_path(path: str) -> bool:
    """Validate if a file path is correct and the file exists.
    
    Args:
        path: The file path to validate
        
    Returns:
        bool: True if the path is valid and file exists, False otherwise
    """
    try:
        # Clean the path first
        clean_path = clean_file_path(path)
        
        # Check if file exists
        if not os.path.exists(clean_path):
            logger.warning(f"File does not exist: {clean_path}")
            return False
            
        # Check if it's a regular file (not a directory)
        if not os.path.isfile(clean_path):
            logger.warning(f"Path is not a regular file: {clean_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        return False

def get_file_timestamps(file_path: str) -> Optional[FileTimestamp]:
    """Get modification and creation timestamps for a file."""
    try:
        stat = os.stat(file_path)
        return FileTimestamp(
            path=file_path,
            mtime=stat.st_mtime,
            ctime=stat.st_ctime,
            size=stat.st_size
        )
    except OSError as e:
        logger.error(f"Error getting timestamps for {file_path}: {e}")
        return None

def get_git_tracked_files() -> List[str]:
    """Get list of files that Git would track (not ignored)."""
    try:
        # First, get all files that Git would track
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True,
            check=True
        )
        tracked_files = result.stdout.splitlines()
        
        # Then get untracked files that aren't ignored
        result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            capture_output=True,
            text=True,
            check=True
        )
        untracked_files = result.stdout.splitlines()
        
        # Clean up and validate file paths
        all_files = []
        seen_paths = set()  # To prevent duplicates
        
        for file in tracked_files + untracked_files:
            clean_path = clean_file_path(file)
            if clean_path not in seen_paths and validate_file_path(clean_path):
                all_files.append(clean_path)
                seen_paths.add(clean_path)
        
        return all_files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting Git tracked files: {e}")
        return []

def group_files_by_date(files: List[FileTimestamp]) -> Dict[str, List[FileTimestamp]]:
    """Group files by their modification date."""
    date_groups = defaultdict(list)
    for file in files:
        date = datetime.fromtimestamp(file.mtime).strftime('%Y-%m-%d')
        date_groups[date].append(file)
    return dict(date_groups)

def handle_untracked_files() -> List[str]:
    """Handle untracked files and return list of successfully staged files.
    
    Returns:
        List of successfully staged file paths
    """
    staged_files = []
    try:
        # Get list of untracked files
        result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            capture_output=True,
            text=True,
            check=True
        )
        untracked_files = result.stdout.splitlines()
        
        logger.info(f"Found {len(untracked_files)} untracked files")
        
        for file in untracked_files:
            try:
                # Clean and validate the path
                clean_path = clean_file_path(file)
                logger.debug(f"Original path: {file}")
                logger.debug(f"Cleaned path: {clean_path}")
                
                if not validate_file_path(clean_path):
                    logger.warning(f"Invalid file path: {clean_path}")
                    continue
                    
                # Force add the file
                subprocess.run(['git', 'add', '-f', clean_path], check=True)
                staged_files.append(clean_path)
                logger.info(f"Successfully staged: {clean_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to stage {clean_path}: {e}")
                continue
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting untracked files: {e}")
        
    return staged_files

def stage_files_for_commit(files: List[FileTimestamp]) -> List[str]:
    """Stage files for commit and return list of successfully staged files.
    
    Args:
        files: List of FileTimestamp objects to stage
        
    Returns:
        List of successfully staged file paths
    """
    staged_files = []
    
    # First handle any untracked files
    staged_files.extend(handle_untracked_files())
    
    # Then handle the files from the timestamp list
    for file in files:
        try:
            # Clean and validate the path
            clean_path = clean_file_path(file.path)
            if not validate_file_path(clean_path):
                continue
                
            # Force add app/models directory
            if clean_path.startswith('app/models/'):
                subprocess.run(['git', 'add', '-f', clean_path], check=True)
            else:
                subprocess.run(['git', 'add', clean_path], check=True)
                
            staged_files.append(clean_path)
            logger.info(f"Successfully staged: {clean_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to stage {clean_path}: {e}")
            continue
            
    # Verify files are staged
    if staged_files:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        if not result.stdout.strip():
            logger.warning("No files were staged, trying to force add all files")
            try:
                subprocess.run(['git', 'add', '-A'], check=True)
                staged_files = [f.path for f in files]
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to force add files: {e}")
                
    return staged_files

def create_commit_for_date(
    date: str,
    files: List[FileTimestamp],
    commit_message: Optional[str] = None
) -> bool:
    """Create a Git commit for files modified on a specific date."""
    try:
        # Set the commit date
        commit_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d 00:00:00')
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = commit_date
        env['GIT_COMMITTER_DATE'] = commit_date

        # Stage files for commit
        staged_files = stage_files_for_commit(files)
        
        if not staged_files:
            logger.info(f"No files to commit for {date}")
            return True

        # Create commit
        message = commit_message or f"Changes from {date}"
        try:
            subprocess.run(
                ['git', 'commit', '-m', message],
                env=env,
                check=True
            )
            logger.info(f"Created commit for {date} with {len(staged_files)} files")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating commit: {e}")
            # Try to commit with --allow-empty
            try:
                subprocess.run(
                    ['git', 'commit', '--allow-empty', '-m', message],
                    env=env,
                    check=True
                )
                logger.info(f"Created empty commit for {date}")
                return True
            except subprocess.CalledProcessError as e2:
                logger.error(f"Error creating empty commit: {e2}")
                return False
    except Exception as e:
        logger.error(f"Unexpected error in create_commit_for_date: {e}")
        return False

def initialize_git_repo() -> bool:
    """Initialize Git repository if not already initialized."""
    if not os.path.exists('.git'):
        try:
            subprocess.run(['git', 'init'], check=True)
            logger.info("Initialized Git repository")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error initializing Git repository: {e}")
            return False
    return True

def setup_remote_repo(remote_url: str) -> bool:
    """Set up remote repository."""
    try:
        # Check if remote already exists
        result = subprocess.run(
            ['git', 'remote', '-v'],
            capture_output=True,
            text=True,
            check=True
        )
        if remote_url not in result.stdout:
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
            logger.info(f"Added remote repository: {remote_url}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up remote repository: {e}")
        return False

def main():
    """Main function to handle Git commit automation."""
    parser = argparse.ArgumentParser(description='Automate Git commits with timestamp preservation')
    parser.add_argument('--remote-url', help='GitHub repository URL')
    parser.add_argument('--branch', default='main', help='Branch name (default: main)')
    parser.add_argument('--commit-message', help='Custom commit message template')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be committed without actually committing')
    args = parser.parse_args()

    # Initialize Git repository
    if not initialize_git_repo():
        return

    # Set up remote repository if URL provided
    if args.remote_url and not setup_remote_repo(args.remote_url):
        return

    # Get list of files that Git would track
    tracked_files = get_git_tracked_files()
    if not tracked_files:
        logger.warning("No files found to commit. Make sure you have files that aren't ignored by .gitignore")
        return

    # Get timestamps for tracked files
    files = []
    for file_path in tracked_files:
        timestamp = get_file_timestamps(file_path)
        if timestamp:
            files.append(timestamp)

    if not files:
        logger.warning("No files found with valid timestamps")
        return

    # Group files by date
    date_groups = group_files_by_date(files)

    # Create commits for each date
    success = True
    for date, date_files in sorted(date_groups.items()):
        if args.dry_run:
            logger.info(f"Would commit {len(date_files)} files from {date}")
            continue

        commit_message = args.commit_message.format(date=date) if args.commit_message else None
        if not create_commit_for_date(date, date_files, commit_message):
            success = False
            break

    if success and not args.dry_run:
        logger.info("Successfully created all commits")
        if args.remote_url:
            try:
                subprocess.run(['git', 'push', '-u', 'origin', args.branch], check=True)
                logger.info(f"Successfully pushed to {args.remote_url}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error pushing to remote repository: {e}")
    elif args.dry_run:
        logger.info("Dry run completed - no commits were made")

if __name__ == '__main__':
    main() 