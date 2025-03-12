#!/usr/bin/env python3
"""
Documentation linting tool for the WITHIN ML Prediction System.

This script checks the quality and consistency of docstrings across the codebase,
ensuring they follow the Google docstring format and contain all required sections.

Usage:
    python lint_documentation.py [path]

Example:
    python lint_documentation.py app/models/ml
    python lint_documentation.py app/models/ml/prediction/training.py
"""

import os
import re
import sys
import ast
import logging
import argparse
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required sections in Google docstring format for different types
REQUIRED_SECTIONS = {
    "module": [""],  # Module docstrings need at least a description
    "class": [""],  # Class docstrings need at least a description
    "method": ["Args", "Returns"],  # Methods should document args and returns
    "function": ["Args", "Returns"],  # Functions should document args and returns
    "property": [""],  # Properties need at least a description
}

# Optional but recommended sections
RECOMMENDED_SECTIONS = {
    "module": ["Examples", "Attributes"],
    "class": ["Examples", "Attributes"],
    "method": ["Examples", "Raises"],
    "function": ["Examples", "Raises"],
    "property": ["Returns"],
}


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor that extracts docstrings from Python code."""

    def __init__(self) -> None:
        """Initialize the visitor with empty data structures."""
        self.docstrings: Dict[str, Dict[str, Any]] = {
            "module": {},
            "class": {},
            "method": {},
            "function": {},
            "property": {},
        }
        self.current_class: Optional[str] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        """Extract module docstring."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            self.docstrings["module"][""] = {
                "docstring": node.body[0].value.s,
                "lineno": node.body[0].lineno,
            }
        else:
            self.errors.append(f"Module missing docstring")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class docstring."""
        old_class = self.current_class
        self.current_class = node.name
        
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            self.docstrings["class"][node.name] = {
                "docstring": node.body[0].value.s,
                "lineno": node.body[0].lineno,
            }
        else:
            self.errors.append(f"Class {node.name} missing docstring (line {node.lineno})")
        
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function or method docstring."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            if self.current_class:
                if node.name.startswith("__") and node.name.endswith("__"):
                    # Special methods like __init__ should have docstrings, but we'll just warn
                    if node.name != "__init__":
                        category = "special_method"
                    else:
                        category = "method"
                elif hasattr(node, "decorator_list") and any(
                    d.id == "property" if isinstance(d, ast.Name) else False
                    for d in node.decorator_list
                ):
                    category = "property"
                else:
                    category = "method"
                
                full_name = f"{self.current_class}.{node.name}"
            else:
                category = "function"
                full_name = node.name
            
            self.docstrings[category][full_name] = {
                "docstring": node.body[0].value.s,
                "lineno": node.body[0].lineno,
                "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                "returns": bool(node.returns),
            }
        else:
            if self.current_class:
                full_name = f"{self.current_class}.{node.name}"
                # Skip private methods (starting with _) unless they're __init__
                if node.name.startswith("_") and node.name != "__init__":
                    pass
                else:
                    self.errors.append(f"Method {full_name} missing docstring (line {node.lineno})")
            else:
                # Skip private functions (starting with _)
                if not node.name.startswith("_") or node.name.startswith("__"):
                    self.errors.append(f"Function {node.name} missing docstring (line {node.lineno})")
        
        self.generic_visit(node)


def parse_docstring(docstring: str) -> Dict[str, str]:
    """
    Parse a Google-style docstring into sections.
    
    Args:
        docstring: The docstring to parse
        
    Returns:
        A dictionary mapping section names to their content
    """
    if not docstring:
        return {}
    
    # Clean up the docstring
    lines = docstring.strip().split("\n")
    if len(lines) <= 1:
        return {"": docstring.strip()}
    
    # Extract sections
    sections = {"": []}
    current_section = ""
    
    for line in lines:
        line = line.strip()
        section_match = re.match(r"^([A-Za-z]+):\s*$", line)
        
        if section_match:
            current_section = section_match.group(1)
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)
        else:
            sections[""].append(line)
    
    # Join lines within sections
    for section, lines in sections.items():
        sections[section] = "\n".join(lines).strip()
    
    return sections


def check_docstring_format(
    docstring: str, docstring_type: str, name: str, lineno: int, args: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Check if a docstring follows the Google format and has all required sections.
    
    Args:
        docstring: The docstring to check
        docstring_type: Type of the docstring (module, class, method, function, property)
        name: Name of the entity the docstring belongs to
        lineno: Line number in the source file
        args: List of argument names if docstring_type is method or function
        
    Returns:
        Two lists of error and warning messages
    """
    errors = []
    warnings = []
    
    sections = parse_docstring(docstring)
    
    # Check for required sections
    for required_section in REQUIRED_SECTIONS.get(docstring_type, []):
        if required_section not in sections:
            if required_section:
                errors.append(
                    f"{docstring_type.capitalize()} {name} missing required section '{required_section}' (line {lineno})"
                )
    
    # Check for recommended sections
    for recommended_section in RECOMMENDED_SECTIONS.get(docstring_type, []):
        if recommended_section not in sections:
            warnings.append(
                f"{docstring_type.capitalize()} {name} missing recommended section '{recommended_section}' (line {lineno})"
            )
    
    # For methods and functions, check that all args are documented
    if docstring_type in ("method", "function") and args:
        if "Args" not in sections:
            # Already reported as missing required section
            pass
        else:
            # Extract parameter names from Args section
            args_section = sections.get("Args", "")
            documented_args = []
            for line in args_section.split("\n"):
                arg_match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", line)
                if arg_match:
                    documented_args.append(arg_match.group(1))
            
            # Check for undocumented args
            for arg in args:
                if arg not in documented_args:
                    errors.append(
                        f"{docstring_type.capitalize()} {name} missing documentation for parameter '{arg}' (line {lineno})"
                    )
            
            # Check for documented args that don't exist
            for arg in documented_args:
                if arg not in args:
                    warnings.append(
                        f"{docstring_type.capitalize()} {name} documents non-existent parameter '{arg}' (line {lineno})"
                    )
    
    # Check if the docstring is too short
    description = sections.get("", "")
    if len(description.split()) < 3:
        warnings.append(
            f"{docstring_type.capitalize()} {name} has a very short description (line {lineno})"
        )
    
    return errors, warnings


def lint_file(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Lint the docstrings in a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Two lists of error and warning messages
    """
    errors = []
    warnings = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        try:
            tree = ast.parse(content, file_path)
        except SyntaxError as e:
            errors.append(f"Syntax error in {file_path}: {e}")
            return errors, warnings
        
        visitor = DocstringVisitor()
        visitor.visit(tree)
        
        # Add visitor errors
        errors.extend(visitor.errors)
        warnings.extend(visitor.warnings)
        
        # Check each docstring's format
        for docstring_type, docstrings in visitor.docstrings.items():
            for name, info in docstrings.items():
                doc_errors, doc_warnings = check_docstring_format(
                    info["docstring"],
                    docstring_type,
                    name,
                    info["lineno"],
                    info.get("args")
                )
                errors.extend(doc_errors)
                warnings.extend(doc_warnings)
    
    except Exception as e:
        errors.append(f"Error processing {file_path}: {e}")
    
    return errors, warnings


def lint_path(path: str) -> Tuple[int, int, int]:
    """
    Lint Python files in a directory or a single file.
    
    Args:
        path: Path to a directory or file to lint
        
    Returns:
        A tuple of (total files, error count, warning count)
    """
    total_files = 0
    total_errors = 0
    total_warnings = 0
    
    if os.path.isfile(path) and path.endswith(".py"):
        # Lint a single file
        total_files = 1
        logger.info(f"Linting {path}")
        errors, warnings = lint_file(path)
        
        if errors:
            for error in errors:
                logger.error(f"{path}: {error}")
            total_errors += len(errors)
        
        if warnings:
            for warning in warnings:
                logger.warning(f"{path}: {warning}")
            total_warnings += len(warnings)
        
        if not errors and not warnings:
            logger.info(f"{path}: No issues found")
    
    elif os.path.isdir(path):
        # Lint all Python files in the directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    
                    logger.info(f"Linting {file_path}")
                    errors, warnings = lint_file(file_path)
                    
                    if errors:
                        for error in errors:
                            logger.error(f"{file_path}: {error}")
                        total_errors += len(errors)
                    
                    if warnings:
                        for warning in warnings:
                            logger.warning(f"{file_path}: {warning}")
                        total_warnings += len(warnings)
                    
                    if not errors and not warnings:
                        logger.info(f"{file_path}: No issues found")
    else:
        logger.error(f"{path} is not a valid Python file or directory")
    
    return total_files, total_errors, total_warnings


def main() -> None:
    """Run the documentation linter."""
    parser = argparse.ArgumentParser(description="Documentation linting tool")
    parser.add_argument(
        "path",
        nargs="?",
        default="app",
        help="Path to a file or directory to lint (default: app)",
    )
    args = parser.parse_args()
    
    path = args.path
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist")
        sys.exit(1)
    
    logger.info(f"Linting documentation in {path}")
    total_files, error_count, warning_count = lint_path(path)
    
    logger.info(f"Linted {total_files} files")
    logger.info(f"Found {error_count} errors and {warning_count} warnings")
    
    if error_count > 0:
        sys.exit(1)  # Exit with error code
    else:
        sys.exit(0)  # Exit with success code


if __name__ == "__main__":
    main() 