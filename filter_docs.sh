#!/bin/bash

# Documentation Filter Helper Script
# This script provides shortcuts for common documentation filtering operations

# Base directory
DOCS_DIR="/Users/alecposner/WITHIN/docs"

# Make sure the Python script is executable
chmod +x filter_docs.py

# Function to print usage instructions
print_usage() {
  echo "Documentation Filter Helper"
  echo ""
  echo "Usage: ./filter_docs.sh [COMMAND]"
  echo ""
  echo "Commands:"
  echo "  not-implemented      List all NOT_IMPLEMENTED documentation files"
  echo "  partially            List all PARTIALLY_IMPLEMENTED documentation files"
  echo "  implemented          List all IMPLEMENTED documentation files"
  echo "  unknown              List all files with UNKNOWN implementation status"
  echo "  check-file <path>    View content and detect status of a specific file"
  echo "  check-unknown [n]    Sample and display content of n UNKNOWN files (default: 5)"
  echo "  summary              Show summary counts of files by implementation status"
  echo "  recent [days]        List files modified in the last N days (default: 7)"
  echo "  todo                 Find files containing TODO markers"
  echo "  images               List all image files"
  echo "  pdfs                 List all PDF files"
  echo "  tech-not-implemented List NOT_IMPLEMENTED files in technical directories"
  echo "  ml-not-implemented   List NOT_IMPLEMENTED files in ML directories"
  echo "  priority             List high-priority files that need implementation"
  echo "  export-csv [status]  Export files with given status to CSV (default: all)"
  echo "  help                 Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./filter_docs.sh not-implemented"
  echo "  ./filter_docs.sh check-file docs/implementation/ml/index.md"
  echo "  ./filter_docs.sh check-unknown 3"
}

# Function to check/show content of a specific file
check_file_content() {
  local file_path="$1"
  
  if [ ! -f "$file_path" ]; then
    echo "Error: File not found: $file_path"
    return 1
  fi
  
  echo "==== FILE INFORMATION ====="
  echo "Path: $file_path"
  echo "Size: $(du -h "$file_path" | cut -f1) ($(wc -l < "$file_path") lines)"
  echo "Modified: $(date -r "$file_path")"
  echo ""
  
  # Detect status
  local status=$(grep -E '\*\*IMPLEMENTATION STATUS: ([A-Z_]+)\*\*' "$file_path" | head -1 | sed -E 's/.*\*\*IMPLEMENTATION STATUS: ([A-Z_]+)\*\*.*/\1/')
  
  if [ -n "$status" ]; then
    echo "Implementation Status: $status"
  else
    echo "Implementation Status: UNKNOWN (no status marker found)"
  fi
  
  echo ""
  echo "==== FILE CONTENT PREVIEW ====="
  
  # For text files, show first 20 lines
  if file "$file_path" | grep -q text; then
    head -20 "$file_path"
    
    # If file longer than 20 lines, show this is truncated
    if [ "$(wc -l < "$file_path")" -gt 20 ]; then
      echo "... [truncated, $(wc -l < "$file_path") lines total] ..."
    fi
  else
    echo "Binary file, cannot display content"
  fi
}

# Function to sample and check content of UNKNOWN files
check_unknown_files() {
  local num_samples="${1:-5}" # Default to 5 samples
  
  # Get list of UNKNOWN files using the Python script
  echo "Finding files with UNKNOWN implementation status..."
  local unknown_files=$(python filter_docs.py --csv | grep "^UNKNOWN," | cut -d',' -f2-)
  
  # Count total files
  local total_files=$(echo "$unknown_files" | wc -l)
  echo "Found $total_files files with UNKNOWN status."
  
  if [ "$total_files" -eq 0 ]; then
    echo "No UNKNOWN files found."
    return
  fi
  
  # Sample at most num_samples files
  local num_to_show=$(( num_samples < total_files ? num_samples : total_files ))
  echo "Showing sample of $num_to_show files:"
  echo ""
  
  # Take random samples using shuf if available, otherwise just take the first n
  if command -v shuf >/dev/null 2>&1; then
    # Get random samples using shuf
    local samples=$(echo "$unknown_files" | shuf -n "$num_to_show")
  else
    # macOS doesn't have shuf, so just take the first n
    local samples=$(echo "$unknown_files" | head -n "$num_to_show")
  fi
  
  # Check each sampled file
  local i=1
  while read -r file; do
    echo "[$i/$num_to_show] Examining: $DOCS_DIR/$file"
    check_file_content "$DOCS_DIR/$file"
    echo "----------------------------------------"
    i=$((i+1))
  done <<< "$samples"
  
  echo ""
  echo "File type analysis for all UNKNOWN files:"
  echo "$unknown_files" | xargs -I{} file "$DOCS_DIR/{}" | cut -d: -f2 | sort | uniq -c | sort -nr
}

# No arguments provided, show help
if [ $# -eq 0 ]; then
  print_usage
  exit 0
fi

# Process commands
case "$1" in
  "not-implemented")
    echo "Listing all NOT_IMPLEMENTED documentation files:"
    python filter_docs.py --status NOT_IMPLEMENTED --type md
    ;;
    
  "partially")
    echo "Listing all PARTIALLY_IMPLEMENTED documentation files:"
    python filter_docs.py --status PARTIALLY_IMPLEMENTED --type md
    ;;
    
  "implemented")
    echo "Listing all IMPLEMENTED documentation files:"
    python filter_docs.py --status IMPLEMENTED --type md
    ;;
    
  "unknown")
    echo "Listing all files with UNKNOWN implementation status:"
    python filter_docs.py --type md | grep -A 100 "Status: UNKNOWN"
    ;;
    
  "check-file")
    if [ -z "$2" ]; then
      echo "Error: Please specify a file path to check"
      echo "Usage: ./filter_docs.sh check-file <file_path>"
      exit 1
    fi
    check_file_content "$2"
    ;;
    
  "check-unknown")
    check_unknown_files "$2"
    ;;
    
  "summary")
    echo "Summary of documentation files by implementation status:"
    python filter_docs.py --summary
    ;;
    
  "recent")
    # Default to 7 days if not specified
    DAYS=${2:-7}
    echo "Listing files modified in the last $DAYS days:"
    python filter_docs.py --days "$DAYS"
    ;;
    
  "todo")
    echo "Finding files with TODO markers:"
    python filter_docs.py --content "TODO|FIXME|XXX"
    ;;
    
  "images")
    echo "Listing all image files:"
    python filter_docs.py --type image
    ;;
    
  "pdfs")
    echo "Listing all PDF files:"
    python filter_docs.py --type pdf
    ;;
    
  "tech-not-implemented")
    echo "Listing NOT_IMPLEMENTED files in technical directories:"
    python filter_docs.py --status NOT_IMPLEMENTED --path "technical"
    ;;
    
  "ml-not-implemented")
    echo "Listing NOT_IMPLEMENTED files in ML directories:"
    python filter_docs.py --status NOT_IMPLEMENTED --path "ml"
    ;;
    
  "priority")
    echo "Listing high-priority files that need implementation:"
    # We consider high priority:
    # 1. Recently created NOT_IMPLEMENTED files
    # 2. Files in core documentation areas (ml, technical)
    python filter_docs.py --status NOT_IMPLEMENTED --days 14 --path "ml/technical"
    ;;
    
  "export-csv")
    # Default to all statuses if not specified
    STATUS=${2:-""}
    if [ -z "$STATUS" ]; then
      echo "Exporting all files to CSV:"
      python filter_docs.py --csv
    else
      echo "Exporting $STATUS files to CSV:"
      python filter_docs.py --status "$STATUS" --csv
    fi
    ;;
    
  "help"|*)
    print_usage
    ;;
esac 