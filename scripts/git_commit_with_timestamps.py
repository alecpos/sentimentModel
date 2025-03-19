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
        
        return tracked_files + untracked_files
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

        # Add files to staging
        for file in files:
            try:
                subprocess.run(['git', 'add', file.path], check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Skipping {file.path}: {e}")
                continue

        # Create commit
        message = commit_message or f"Changes from {date}"
        subprocess.run(
            ['git', 'commit', '-m', message],
            env=env,
            check=True
        )
        logger.info(f"Created commit for {date} with {len(files)} files")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating commit for {date}: {e}")
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