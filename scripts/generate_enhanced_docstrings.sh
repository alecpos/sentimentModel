#!/bin/bash
# Enhanced Docstring Generation Script
# This script provides easy access to the NLP-enhanced docstring generation capabilities

set -e  # Exit on error

# Initialize variables
TARGET=""
STYLE="google"
RECURSIVE=false
APPLY=false
VALIDATE=false
OUTPUT=""

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display script usage
function show_usage {
    echo -e "${BLUE}NLP-Enhanced Docstring Generator Script${NC}"
    echo
    echo "This script helps generate high-quality docstrings using NLP techniques"
    echo "based on the latest research in docstring generation (March 2025)."
    echo
    echo "Usage:"
    echo "  $0 [options] <target-file-or-directory>"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -s, --style STYLE          Docstring style (google, numpy, sphinx)"
    echo "  -r, --recursive            Process directories recursively"
    echo "  -a, --apply                Apply generated docstrings to files"
    echo "  -v, --validate             Validate existing docstrings"
    echo "  -o, --output FILE          Save results to JSON file"
    echo "  --check-deps               Check for dependencies and install if missing"
    echo
    echo "Examples:"
    echo "  $0 app/models/ml/prediction/ad_score_predictor.py"
    echo "  $0 -s numpy -r app/models"
    echo "  $0 -a -v app/core/validation.py"
    echo
}

# Check for Python dependencies
function check_dependencies {
    echo -e "${BLUE}Checking for required Python dependencies...${NC}"
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}Error: pip is not installed or not in PATH${NC}"
        exit 1
    fi
    
    # Check for required packages
    MISSING_DEPS=()
    
    echo "Checking torch..."
    python -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
    
    echo "Checking numpy..."
    python -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")
    
    echo "Checking sentence-transformers..."
    python -c "import sentence_transformers" 2>/dev/null || MISSING_DEPS+=("sentence-transformers")
    
    # Install missing dependencies if requested
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
        
        if [ "$INSTALL_DEPS" = true ]; then
            echo -e "${BLUE}Installing missing dependencies...${NC}"
            pip install ${MISSING_DEPS[@]}
            echo -e "${GREEN}Dependencies installed successfully${NC}"
        else
            echo -e "${YELLOW}To install missing dependencies, run:${NC}"
            echo "pip install ${MISSING_DEPS[*]}"
            echo -e "${YELLOW}Or run this script with the --check-deps flag${NC}"
        fi
    else
        echo -e "${GREEN}All required dependencies are installed${NC}"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -s|--style)
            STYLE="$2"
            shift 2
            ;;
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -a|--apply)
            APPLY=true
            shift
            ;;
        -v|--validate)
            VALIDATE=true
            shift
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        --check-deps)
            INSTALL_DEPS=true
            check_dependencies
            shift
            ;;
        *)
            if [ -z "$TARGET" ]; then
                TARGET="$1"
            else
                echo -e "${RED}Error: Unexpected argument '$1'${NC}"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if target is specified
if [ -z "$TARGET" ]; then
    echo -e "${RED}Error: No target file or directory specified${NC}"
    show_usage
    exit 1
fi

# Check if target exists
if [ ! -e "$TARGET" ]; then
    echo -e "${RED}Error: Target '$TARGET' does not exist${NC}"
    exit 1
fi

# Build command arguments
CMD_ARGS=""

if [ "$RECURSIVE" = true ]; then
    CMD_ARGS="$CMD_ARGS --recursive"
fi

if [ "$APPLY" = true ]; then
    CMD_ARGS="$CMD_ARGS --apply"
fi

if [ "$VALIDATE" = true ]; then
    CMD_ARGS="$CMD_ARGS --validate"
fi

if [ ! -z "$OUTPUT" ]; then
    CMD_ARGS="$CMD_ARGS --output $OUTPUT"
fi

# Run the Python script
echo -e "${BLUE}Running NLP-Enhanced Docstring Generator${NC}"
echo -e "Target: ${GREEN}$TARGET${NC}"
echo -e "Style: ${GREEN}$STYLE${NC}"
echo -e "Options: ${GREEN}$CMD_ARGS${NC}"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/nlp_enhanced_docstring_generator.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script not found at $PYTHON_SCRIPT${NC}"
    exit 1
fi

python "$PYTHON_SCRIPT" --style "$STYLE" $CMD_ARGS "$TARGET"

echo -e "${GREEN}Docstring generation completed successfully${NC}" 