#!/bin/bash
# =============================================================================
#  Comprehensive Docstring Generation Script
# =============================================================================
# This script automates docstring generation throughout the codebase with
# safety measures to protect existing high-quality documentation.
#
# The process follows these steps:
# 1. Fix any docstring indentation issues
# 2. Verify existing docstrings and generate a report
# 3. Generate docstrings, respecting protected elements
# 4. Verify the results
# =============================================================================

set -e  # Exit on error

# Set default options
DRY_RUN=false
VERBOSE=false
TARGET_DIR="app"
QUALITY_THRESHOLD=0.7
VERIFY_ONLY=false
SKIP_INDENTATION=false
CREATE_REPORT=false
REPORT_FORMAT="text"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help message
show_help() {
    echo "Usage: $0 [OPTIONS] [TARGET_DIRECTORY]"
    echo ""
    echo "Options:"
    echo "  --dry-run             Preview changes without modifying files"
    echo "  --verbose             Show detailed information during execution"
    echo "  --verify-only         Only verify docstrings, don't generate any"
    echo "  --skip-indentation    Skip the indentation fixing step"
    echo "  --create-report       Create an HTML report of docstring quality"
    echo "  --report-format TYPE  Report format (html, json, text) [default: text]"
    echo "  --quality-threshold N Set minimum docstring quality threshold [0-1]"
    echo "  --help                Display this help message"
    echo ""
    echo "TARGET_DIRECTORY defaults to 'app' if not provided"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --skip-indentation)
            SKIP_INDENTATION=true
            shift
            ;;
        --create-report)
            CREATE_REPORT=true
            shift
            ;;
        --report-format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        --quality-threshold)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# Validate report format
if [[ "$REPORT_FORMAT" != "text" && "$REPORT_FORMAT" != "html" && "$REPORT_FORMAT" != "json" ]]; then
    echo -e "${RED}Invalid report format: $REPORT_FORMAT. Valid options are text, html, or json.${NC}"
    exit 1
fi

# Determine verbosity flag
if $VERBOSE; then
    VERBOSITY="--verbose"
else
    VERBOSITY=""
fi

# Define output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="docstring_report_${TIMESTAMP}.${REPORT_FORMAT}"
LOG_FILE="docstring_generation_${TIMESTAMP}.log"

# Log function
log() {
    local level="$1"
    local message="$2"
    local color=""
    
    case $level in
        "INFO") color="$BLUE" ;;
        "SUCCESS") color="$GREEN" ;;
        "WARNING") color="$YELLOW" ;;
        "ERROR") color="$RED" ;;
        *) color="$NC" ;;
    esac
    
    echo -e "${color}[$level] $message${NC}"
    echo "[$level] $message" >> "$LOG_FILE"
}

# Initialize log file
echo "Docstring Generation Log - $(date)" > "$LOG_FILE"
echo "=================================" >> "$LOG_FILE"
echo "Target directory: $TARGET_DIR" >> "$LOG_FILE"
echo "Options: DRY_RUN=$DRY_RUN, VERBOSE=$VERBOSE, VERIFY_ONLY=$VERIFY_ONLY" >> "$LOG_FILE"
echo "=================================" >> "$LOG_FILE"

log "INFO" "Starting docstring processing for $TARGET_DIR"

# Step 1: Fix docstring indentation issues if needed
if ! $SKIP_INDENTATION; then
    log "INFO" "Fixing docstring indentation issues..."
    
    if $DRY_RUN; then
        ./scripts/fix_docstring_indentation.py "$TARGET_DIR" --recursive --dry-run $VERBOSITY
    else
        ./scripts/fix_docstring_indentation.py "$TARGET_DIR" --recursive $VERBOSITY
    fi
    
    log "SUCCESS" "Completed indentation fixes"
else
    log "INFO" "Skipping indentation fixing step"
fi

# Step 2: Verify existing docstrings
log "INFO" "Verifying existing docstrings..."

if $CREATE_REPORT; then
    ./scripts/verify_docstrings.py "$TARGET_DIR" --recursive --format "$REPORT_FORMAT" --output "$REPORT_FILE" --min-score "$QUALITY_THRESHOLD"
    log "INFO" "Created verification report: $REPORT_FILE"
else
    ./scripts/verify_docstrings.py "$TARGET_DIR" --recursive --min-score "$QUALITY_THRESHOLD"
fi

# Exit if only verification was requested
if $VERIFY_ONLY; then
    log "INFO" "Verification completed. Exiting as requested."
    exit 0
fi

# Step 3: Find Python files and generate docstrings
log "INFO" "Generating docstrings for Python files in $TARGET_DIR..."

PYTHON_FILES=$(find "$TARGET_DIR" -name "*.py" -type f)
FILE_COUNT=$(echo "$PYTHON_FILES" | wc -l | xargs)
PROCESSED_COUNT=0
SUCCESS_COUNT=0
ERROR_COUNT=0

for py_file in $PYTHON_FILES; do
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    
    if $VERBOSE; then
        log "INFO" "Processing file $PROCESSED_COUNT/$FILE_COUNT: $py_file"
    else
        # Show progress without too much output
        if (( PROCESSED_COUNT % 10 == 0 )) || (( PROCESSED_COUNT == FILE_COUNT )); then
            log "INFO" "Processed $PROCESSED_COUNT/$FILE_COUNT files..."
        fi
    fi
    
    # Check if file is listed in protected docstrings
    if grep -q "^$py_file:" configs/protected_docstrings.txt 2>/dev/null; then
        log "WARNING" "Skipping protected file: $py_file"
        continue
    fi
    
    # Generate docstrings
    if $DRY_RUN; then
        # Preview only
        ./scripts/mac_docstring_generator.py "$py_file" $VERBOSITY
    else
        # Apply changes
        if ./scripts/mac_docstring_generator.py "$py_file" --apply $VERBOSITY; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            ERROR_COUNT=$((ERROR_COUNT + 1))
            log "ERROR" "Failed to generate docstrings for $py_file"
        fi
    fi
done

# Step 4: Final verification if not in dry run mode
if ! $DRY_RUN; then
    log "INFO" "Verifying updated docstrings..."
    if $CREATE_REPORT; then
        ./scripts/verify_docstrings.py "$TARGET_DIR" --recursive --format "$REPORT_FORMAT" --output "updated_$REPORT_FILE" --min-score "$QUALITY_THRESHOLD"
        log "INFO" "Created updated verification report: updated_$REPORT_FILE"
    else
        ./scripts/verify_docstrings.py "$TARGET_DIR" --recursive --min-score "$QUALITY_THRESHOLD"
    fi
fi

# Summary
log "SUCCESS" "Docstring processing completed!"
log "INFO" "Summary:"
log "INFO" "- Files processed: $PROCESSED_COUNT"
log "INFO" "- Successful generations: $SUCCESS_COUNT"
log "INFO" "- Errors encountered: $ERROR_COUNT"

if $DRY_RUN; then
    log "WARNING" "This was a dry run. No changes were actually applied."
    log "INFO" "Run without --dry-run to apply the changes."
fi

log "INFO" "Log saved to: $LOG_FILE"
if $CREATE_REPORT; then
    log "INFO" "Report saved to: $REPORT_FILE"
    if ! $DRY_RUN; then
        log "INFO" "Updated report saved to: updated_$REPORT_FILE"
    fi
fi 