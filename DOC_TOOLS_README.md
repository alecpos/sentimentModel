# Documentation Management Tools

This directory contains tools for managing, tracking, and filtering documentation files, particularly those with implementation status flags (NOT_IMPLEMENTED, PARTIALLY_IMPLEMENTED, IMPLEMENTED).

## Scripts Overview

1. **filter_docs.py** - Python script for filtering documentation files with multiple criteria
2. **filter_docs.sh** - Shell script with shortcuts for common filtering operations
3. **track_doc_progress.py** - Python script for tracking documentation progress over time
4. **add_status_markers.py** - Python script for analyzing and adding status markers to UNKNOWN files

## Requirements

- Python 3.9+
- No additional packages required (uses standard library only)

## Basic Usage

### Using the Shell Script Shortcuts

For quick filtering operations, use the shell script:

```bash
# Show help
./filter_docs.sh help

# List all NOT_IMPLEMENTED documentation files
./filter_docs.sh not-implemented

# List all files with UNKNOWN implementation status
./filter_docs.sh unknown

# Examine content of specific files
./filter_docs.sh check-file docs/implementation/ml/index.md

# Sample and display content of UNKNOWN files
./filter_docs.sh check-unknown 5

# Show a summary of documentation by implementation status
./filter_docs.sh summary

# List files modified in the last 7 days
./filter_docs.sh recent

# Find files with TODO markers
./filter_docs.sh todo

# List high-priority files that need implementation
./filter_docs.sh priority
```

### Using the Python Filter Script Directly

For more complex filtering, use the Python script directly:

```bash
# Get help
python filter_docs.py --help

# Filter by implementation status
python filter_docs.py --status NOT_IMPLEMENTED

# Filter by file type
python filter_docs.py --type md

# Filter by content
python filter_docs.py --content "TODO|FIXME"

# Filter by modification date
python filter_docs.py --days 14

# Filter by path
python filter_docs.py --path "technical"

# Combine multiple filters
python filter_docs.py --status PARTIALLY_IMPLEMENTED --type md --path "ml" --days 30
```

### Tracking Progress Over Time

To track documentation progress:

```bash
# Record current documentation state
python track_doc_progress.py --record

# Generate progress report
python track_doc_progress.py --report

# Export history to CSV
python track_doc_progress.py --export-csv progress.csv
```

### Handling UNKNOWN Status Files

To analyze and fix files with UNKNOWN status:

```bash
# Check content of UNKNOWN files (samples random files)
./filter_docs.sh check-unknown 5

# List all files with UNKNOWN status
./filter_docs.sh unknown

# Analyze and add status markers (dry run, no changes)
python add_status_markers.py --dry-run

# Analyze files in a specific path
python add_status_markers.py --dry-run --path "api"

# Analyze and add status markers to all UNKNOWN files
python add_status_markers.py

# Force a specific status for all files in a path
python add_status_markers.py --path "user_guides" --status PARTIALLY_IMPLEMENTED
```

## Examples

### Find High-Priority Files to Implement

```bash
# Find NOT_IMPLEMENTED files in the ML technical section
python filter_docs.py --status NOT_IMPLEMENTED --path "ml/technical"

# Find recently created placeholder files
python filter_docs.py --status NOT_IMPLEMENTED --days 7
```

### Identify and Fix UNKNOWN Status Files

```bash
# First check what UNKNOWN files exist
./filter_docs.sh summary

# Examine sample files to understand their content
./filter_docs.sh check-unknown 3

# Run analyzer in dry-run mode to see what it would do
python add_status_markers.py --dry-run

# Process files in specific directories first
python add_status_markers.py --path "implementation/ml"

# Take a new snapshot after updating files
python track_doc_progress.py --record
```

### Track Documentation Progress for Reports

```bash
# Take a snapshot of the current state
python track_doc_progress.py --record

# Generate markdown report showing progress over time
python track_doc_progress.py --report > documentation_progress.md
```

### Generate CSV for Data Analysis

```bash
# Export documentation files by status to CSV
./filter_docs.sh export-csv > documentation_status.csv

# Export progress history to CSV for analysis in Excel/Google Sheets
python track_doc_progress.py --export-csv progress_history.csv
```

## Tips for Documentation Management

1. **Prioritize implementation** based on:
   - Core functionality documentation
   - Public-facing APIs and interfaces
   - Referenced files (that others link to)

2. **Track progress regularly**:
   - Record snapshots weekly to track velocity
   - Use the progress reports for team updates

3. **Use status flags consistently**:
   - `NOT_IMPLEMENTED`: Placeholder only, needs full implementation
   - `PARTIALLY_IMPLEMENTED`: Basic content exists but needs completion
   - `IMPLEMENTED`: Complete documentation

4. **Handle UNKNOWN files systematically**:
   - Run the analyzer to suggest appropriate statuses
   - Review the suggestions and make necessary adjustments
   - Mark files that don't need implementation status (e.g. README files) appropriately

5. **Add TODOs for specific points** that need implementation in partially implemented files

## Maintenance Workflow

1. Run `./filter_docs.sh summary` to get an overview
2. Process UNKNOWN files with `python add_status_markers.py --dry-run`
3. After reviewing, apply the changes with `python add_status_markers.py`
4. Run `./filter_docs.sh priority` to identify high-priority files to implement
5. After implementing some files, run `python track_doc_progress.py --record` to track progress
6. Generate reports with `python track_doc_progress.py --report` for team updates 