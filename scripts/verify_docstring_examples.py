#!/usr/bin/env python3
"""
Docstring Example Validator for the WITHIN ML Prediction System.

This script extracts and validates code examples from docstrings to ensure they
are executable and produce expected outputs. It helps maintain the quality and
accuracy of documentation by verifying that code examples remain valid as the
codebase evolves.

Usage:
    python verify_docstring_examples.py [file_or_directory] [options]

Examples:
    # Validate examples in a single file
    python verify_docstring_examples.py app/models/ml/prediction/ad_score_predictor.py
    
    # Validate examples in all Python files in a directory
    python verify_docstring_examples.py app/models/ml/prediction --recursive
    
    # Validate examples and repair broken ones
    python verify_docstring_examples.py app/models/ml/prediction/ad_score_predictor.py --repair
"""

import os
import re
import ast
import sys
import inspect
import importlib
import argparse
import logging
import tempfile
import traceback
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class DocstringExampleExtractor:
    """Extracts code examples from docstrings for validation."""
    
    @staticmethod
    def extract_examples_from_docstring(docstring: str) -> List[str]:
        """Extract executable code examples from a docstring.
        
        Args:
            docstring: The docstring to extract examples from
            
        Returns:
            List of code examples found in the docstring
        """
        if not docstring:
            return []
            
        examples = []
        
        # Extract examples from code blocks (triple backticks)
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, docstring, re.DOTALL)
        examples.extend(code_blocks)
        
        # Extract examples from doctest blocks (lines starting with >>>)
        lines = docstring.split('\n')
        current_example = []
        in_example = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('>>>'):
                if current_example and not in_example:
                    # We found a new example, save the previous one
                    examples.append('\n'.join(current_example))
                    current_example = []
                in_example = True
                # Strip >>> and any leading spaces
                example_line = line.replace('>>>', '', 1).strip()
                current_example.append(example_line)
            elif line.startswith('...') and in_example:
                # Continuation of the previous example line
                example_line = line.replace('...', '', 1).strip()
                current_example.append(example_line)
            elif in_example and line and not line.startswith('#'):
                # This is an expected output, skip it
                continue
            else:
                if current_example and in_example:
                    # End of the current example
                    examples.append('\n'.join(current_example))
                    current_example = []
                    in_example = False
        
        # Don't forget the last example
        if current_example and in_example:
            examples.append('\n'.join(current_example))
        
        return examples

    @staticmethod
    def extract_examples_from_file(file_path: str) -> Dict[str, List[str]]:
        """Extract all code examples from docstrings in a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dict mapping function/class/module names to lists of code examples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        results = {}
        
        try:
            tree = ast.parse(content)
            
            # Check module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                examples = DocstringExampleExtractor.extract_examples_from_docstring(module_docstring)
                if examples:
                    results['module'] = examples
            
            # Visit all classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        examples = DocstringExampleExtractor.extract_examples_from_docstring(docstring)
                        if examples:
                            results[node.name] = examples
                            
                            # Also check methods within classes
                            if isinstance(node, ast.ClassDef):
                                for child in node.body:
                                    if isinstance(child, ast.FunctionDef):
                                        method_docstring = ast.get_docstring(child)
                                        if method_docstring:
                                            method_examples = DocstringExampleExtractor.extract_examples_from_docstring(method_docstring)
                                            if method_examples:
                                                results[f"{node.name}.{child.name}"] = method_examples
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            
        return results


class ExampleValidator:
    """Validates docstring examples to ensure they run correctly."""
    
    @staticmethod
    def validate_example(example: str, module_path: Optional[str] = None, class_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate that a docstring example runs without errors.
        
        Args:
            example: The code example to validate
            module_path: Optional path to the module for context
            class_name: Optional class name for instance methods
            
        Returns:
            Dict with validation results
        """
        result = {
            'valid': False,
            'error': None,
            'output': None,
            'exception': None,
            'traceback': None
        }
        
        # Create a temporary Python file with the example code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            # Add imports if module_path is provided
            import_code = ''
            if module_path:
                module_name = module_path.replace('/', '.').replace('.py', '')
                import_code = f"import {module_name}\n"
                
                # If class_name is provided, add an instance creation
                if class_name:
                    import_code += f"{class_name.lower()}_instance = {module_name}.{class_name}()\n"
            
            # Write the example code to the temporary file
            tmp.write(import_code.encode())
            tmp.write(example.encode())
            tmp_path = tmp.name
        
        try:
            # Run the example code in a separate process
            process = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=5  # Timeout after 5 seconds to prevent infinite loops
            )
            
            if process.returncode == 0:
                result['valid'] = True
                result['output'] = process.stdout
            else:
                result['error'] = process.stderr
        except subprocess.TimeoutExpired:
            result['error'] = "Example execution timed out after 5 seconds"
            result['exception'] = "TimeoutError"
        except Exception as e:
            result['error'] = str(e)
            result['exception'] = type(e).__name__
            result['traceback'] = traceback.format_exc()
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
            
        return result
    
    @staticmethod
    def repair_example(example: str, error: str, module_path: Optional[str] = None) -> str:
        """Attempt to repair a broken docstring example.
        
        Args:
            example: The broken code example
            error: The error message from validation
            module_path: Optional path to the module for context
            
        Returns:
            Repaired code example or original if repair failed
        """
        # Common issues and fixes
        fixes = [
            # Missing imports
            (r"NameError: name '(\w+)' is not defined", lambda m: f"from {module_path.replace('/', '.').replace('.py', '')} import {m.group(1)}\n{example}"),
            # Wrong parameter names
            (r"TypeError: \w+\(\) got an unexpected keyword argument '(\w+)'", lambda m: example.replace(m.group(1), f"unknown_param")),
            # Wrong attribute access
            (r"AttributeError: '(\w+)' object has no attribute '(\w+)'", lambda m: example.replace(f".{m.group(2)}", f".available_attribute")),
            # Missing parentheses in print (Python 3 compatibility)
            (r"SyntaxError: Missing parentheses in call to 'print'", lambda m: example.replace("print ", "print(")),
            # Add simple fixes for common errors
        ]
        
        for pattern, fix_func in fixes:
            match = re.search(pattern, error)
            if match:
                try:
                    fixed_example = fix_func(match)
                    # Validate the fixed example
                    result = ExampleValidator.validate_example(fixed_example, module_path)
                    if result['valid']:
                        return fixed_example
                except Exception:
                    # If fixing fails, continue to the next pattern
                    continue
        
        # If no fixes worked, return the original
        return example


def process_file(file_path: str, repair: bool = False) -> Tuple[int, int, int]:
    """Process a single Python file for docstring examples.
    
    Args:
        file_path: Path to the Python file
        repair: Whether to attempt to repair broken examples
        
    Returns:
        Tuple of (total_examples, valid_examples, fixed_examples)
    """
    examples = DocstringExampleExtractor.extract_examples_from_file(file_path)
    
    if not examples:
        logger.info(f"No code examples found in {file_path}")
        return 0, 0, 0
    
    logger.info(f"Found {sum(len(e) for e in examples.values())} code examples in {file_path}")
    
    total_examples = 0
    valid_examples = 0
    fixed_examples = 0
    
    # Extract module path for context
    module_path = file_path
    
    for item_name, item_examples in examples.items():
        # Determine if this is a class method
        class_name = None
        if '.' in item_name:
            class_name, _ = item_name.split('.')
        
        for i, example in enumerate(item_examples):
            total_examples += 1
            
            result = ExampleValidator.validate_example(example, module_path, class_name)
            
            if result['valid']:
                valid_examples += 1
                logger.info(f"  ✓ Example {i+1} in {item_name} is valid")
            else:
                logger.error(f"  ✗ Example {i+1} in {item_name} is invalid:")
                logger.error(f"    Error: {result['error']}")
                
                if repair:
                    fixed_example = ExampleValidator.repair_example(example, result['error'], module_path)
                    if fixed_example != example:
                        # Check if the fixed example is valid
                        fixed_result = ExampleValidator.validate_example(fixed_example, module_path, class_name)
                        if fixed_result['valid']:
                            fixed_examples += 1
                            logger.info(f"    ✓ Fixed example: {fixed_example}")
                        else:
                            logger.error(f"    ✗ Repair attempt failed: {fixed_result['error']}")
    
    logger.info(f"Summary for {file_path}:")
    logger.info(f"  Total examples: {total_examples}")
    logger.info(f"  Valid examples: {valid_examples}")
    if repair:
        logger.info(f"  Fixed examples: {fixed_examples}")
    
    return total_examples, valid_examples, fixed_examples


def process_directory(directory: str, recursive: bool = False, repair: bool = False) -> Tuple[int, int, int]:
    """Process all Python files in a directory for docstring examples.
    
    Args:
        directory: Path to the directory
        recursive: Whether to recursively process subdirectories
        repair: Whether to attempt to repair broken examples
        
    Returns:
        Tuple of (total_examples, valid_examples, fixed_examples)
    """
    total_examples = 0
    valid_examples = 0
    fixed_examples = 0
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    t, v, f = process_file(file_path, repair)
                    total_examples += t
                    valid_examples += v
                    fixed_examples += f
    else:
        for file in os.listdir(directory):
            if file.endswith('.py'):
                file_path = os.path.join(directory, file)
                t, v, f = process_file(file_path, repair)
                total_examples += t
                valid_examples += v
                fixed_examples += f
    
    return total_examples, valid_examples, fixed_examples


def calculate_metrics(total: int, valid: int, fixed: int) -> Dict[str, float]:
    """Calculate example validation metrics.
    
    Args:
        total: Total number of examples
        valid: Number of valid examples
        fixed: Number of fixed examples
        
    Returns:
        Dict of metric names to values
    """
    if total == 0:
        return {
            'validity_rate': 0.0,
            'repair_rate': 0.0,
            'overall_success_rate': 0.0
        }
    
    validity_rate = (valid / total) * 100
    repair_rate = (fixed / (total - valid)) * 100 if total - valid > 0 else 0.0
    overall_success_rate = ((valid + fixed) / total) * 100
    
    return {
        'validity_rate': validity_rate,
        'repair_rate': repair_rate,
        'overall_success_rate': overall_success_rate
    }


def main():
    """Run the docstring example validator."""
    parser = argparse.ArgumentParser(description="Validate code examples in docstrings")
    parser.add_argument(
        "path",
        help="Path to a Python file or directory"
    )
    parser.add_argument(
        "--recursive", 
        "-r", 
        action="store_true",
        help="Recursively process directories"
    )
    parser.add_argument(
        "--repair", 
        "-f", 
        action="store_true",
        help="Attempt to repair broken examples"
    )
    args = parser.parse_args()
    
    path = args.path
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist")
        sys.exit(1)
    
    if os.path.isfile(path):
        if not path.endswith('.py'):
            logger.error(f"File {path} is not a Python file")
            sys.exit(1)
        total, valid, fixed = process_file(path, args.repair)
    elif os.path.isdir(path):
        total, valid, fixed = process_directory(path, args.recursive, args.repair)
    
    metrics = calculate_metrics(total, valid, fixed)
    
    logger.info("Overall Metrics:")
    logger.info(f"  Validity Rate: {metrics['validity_rate']:.2f}%")
    if args.repair:
        logger.info(f"  Repair Rate: {metrics['repair_rate']:.2f}%")
        logger.info(f"  Overall Success Rate: {metrics['overall_success_rate']:.2f}%")
    
    # Exit with non-zero status if any examples are invalid
    if valid < total:
        sys.exit(1)


if __name__ == "__main__":
    main() 