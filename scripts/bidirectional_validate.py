#!/usr/bin/env python3
"""
Bidirectional Documentation Validator for the WITHIN ML Prediction System.

This script implements bidirectional validation between code and documentation,
validating in both Code → Doc and Doc → Code directions. It identifies
inconsistencies where documented behavior differs from actual implementation
and where implementations do not match their documentation.

Usage:
    python bidirectional_validate.py [file_or_directory] [options]

Examples:
    # Validate bidirectional alignment in a file
    python bidirectional_validate.py app/models/ml/prediction/ad_score_predictor.py
    
    # Validate with high sensitivity
    python bidirectional_validate.py app/models/ml/prediction/ad_score_predictor.py --threshold 0.7
    
    # Generate a detailed report
    python bidirectional_validate.py app/models/ml/prediction --report report.json
"""

import os
import re
import ast
import sys
import json
import inspect
import argparse
import logging
import importlib
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from pathlib import Path

# Try to import embedding libraries, provide instructions if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) 

class CodeAnalyzer:
    """Analyzes Python code to extract actual behavior and implementation details."""
    
    def __init__(self):
        """Initialize the code analyzer."""
        pass
    
    def extract_behavior_from_function(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract behavioral information from a function's implementation.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            Dictionary containing extracted behavioral information
        """
        behavior = {
            'name': func_node.name,
            'parameters': self._extract_parameters(func_node),
            'return_values': self._extract_return_values(func_node),
            'raises': self._extract_exceptions(func_node),
            'imports': self._extract_imports(func_node),
            'calls': self._extract_function_calls(func_node),
            'control_flow': self._extract_control_flow(func_node),
            'summary': self._generate_behavior_summary(func_node)
        }
        
        return behavior
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract information about function parameters.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            List of parameter information dictionaries
        """
        parameters = []
        
        # Process arguments
        for arg in func_node.args.args:
            if arg.arg == 'self':
                continue
                
            param_info = {
                'name': arg.arg,
                'has_default': False,
                'type_hint': None,
                'usage_count': 0,
                'modified': False,
                'required': True
            }
            
            # Extract type hint if available
            if arg.annotation:
                param_info['type_hint'] = ast.unparse(arg.annotation)
            
            parameters.append(param_info)
        
        # Process default values
        defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
        for i, default in enumerate(func_node.args.defaults):
            arg_pos = i + defaults_offset
            if arg_pos >= 0 and arg_pos < len(parameters):
                parameters[arg_pos]['has_default'] = True
                parameters[arg_pos]['required'] = False
        
        # Analyze parameter usage in function body
        for param_info in parameters:
            param_name = param_info['name']
            
            # Count references to the parameter
            for node in ast.walk(func_node):
                if isinstance(node, ast.Name) and node.id == param_name:
                    # Check if this is a reference or an assignment
                    is_store = False
                    if isinstance(node.ctx, ast.Store):
                        is_store = True
                        param_info['modified'] = True
                    
                    if not is_store:
                        param_info['usage_count'] += 1
        
        return parameters 
    
    def _extract_return_values(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract information about function return values.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            Dictionary with return value information
        """
        return_info = {
            'has_return': False,
            'return_count': 0,
            'return_types': set(),
            'return_paths': [],
            'return_none': False,
            'type_hint': None
        }
        
        # Extract return type hint if available
        if func_node.returns:
            return_info['type_hint'] = ast.unparse(func_node.returns)
            return_info['has_return'] = True
        
        # Find all return statements
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return_info['has_return'] = True
                return_info['return_count'] += 1
                
                if node.value is None:
                    return_info['return_none'] = True
                elif isinstance(node.value, ast.Constant):
                    if node.value.value is None:
                        return_info['return_none'] = True
                    else:
                        return_info['return_types'].add(type(node.value.value).__name__)
                elif isinstance(node.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    return_info['return_types'].add(type(node.value).__name__.lower())
                elif isinstance(node.value, ast.Name):
                    return_info['return_types'].add('variable')
                elif isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        return_info['return_types'].add(f'call({node.value.func.id})')
                    elif isinstance(node.value.func, ast.Attribute):
                        return_info['return_types'].add(f'call({ast.unparse(node.value.func)})')
                
                # Get some context for each return path
                return_info['return_paths'].append(ast.unparse(node))
        
        # Convert set to list for JSON serialization
        return_info['return_types'] = list(return_info['return_types'])
        
        return return_info
    
    def _extract_exceptions(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract information about exceptions raised by the function.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            List of exception information dictionaries
        """
        exceptions = []
        
        for node in ast.walk(func_node):
            # Explicit raises
            if isinstance(node, ast.Raise):
                exception_info = {
                    'type': 'explicit',
                    'name': None,
                    'condition': None
                }
                
                if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    exception_info['name'] = node.exc.func.id
                elif isinstance(node.exc, ast.Name):
                    exception_info['name'] = node.exc.id
                
                # Try to extract condition from surrounding if statement
                parent_if = None
                for parent in ast.walk(func_node):
                    if isinstance(parent, ast.If):
                        for child in ast.iter_child_nodes(parent):
                            if isinstance(child, ast.Raise) and child == node:
                                parent_if = parent
                                break
                
                if parent_if:
                    exception_info['condition'] = ast.unparse(parent_if.test)
                
                exceptions.append(exception_info)
            
            # Try-except blocks
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    exception_info = {
                        'type': 'handled',
                        'name': None,
                        'reraises': False
                    }
                    
                    if handler.type:
                        if isinstance(handler.type, ast.Name):
                            exception_info['name'] = handler.type.id
                        elif isinstance(handler.type, ast.Tuple):
                            names = []
                            for elt in handler.type.elts:
                                if isinstance(elt, ast.Name):
                                    names.append(elt.id)
                            exception_info['name'] = tuple(names)
                    
                    # Check if the exception is re-raised
                    for n in ast.walk(handler):
                        if isinstance(n, ast.Raise) and n.exc is None:
                            exception_info['reraises'] = True
                            break
                    
                    exceptions.append(exception_info)
        
        return exceptions 
    
    def _extract_imports(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract import statements within a function.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            List of imported module/function names
        """
        imports = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        
        return imports
    
    def _extract_function_calls(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function calls made within a function.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            List of function call information dictionaries
        """
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                call_info = {
                    'name': None,
                    'args_count': len(node.args),
                    'kwargs_count': len(node.keywords)
                }
                
                if isinstance(node.func, ast.Name):
                    call_info['name'] = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_info['name'] = ast.unparse(node.func)
                
                calls.append(call_info)
        
        return calls
    
    def _extract_control_flow(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract control flow information from a function.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            Dictionary with control flow information
        """
        control_flow = {
            'has_conditionals': False,
            'has_loops': False,
            'has_try_except': False,
            'has_with': False,
            'complexity': 1  # Start with 1 for the function itself
        }
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.IfExp)):
                control_flow['has_conditionals'] = True
                control_flow['complexity'] += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                control_flow['has_loops'] = True
                control_flow['complexity'] += 1
            elif isinstance(node, ast.Try):
                control_flow['has_try_except'] = True
                control_flow['complexity'] += 1
            elif isinstance(node, ast.With):
                control_flow['has_with'] = True
                control_flow['complexity'] += 1
        
        return control_flow
    
    def _generate_behavior_summary(self, func_node: ast.FunctionDef) -> str:
        """Generate a plain text summary of the function's behavior.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            String summarizing the function's behavior
        """
        # This is a simplified approach - in a real system, you might use
        # more sophisticated NLP to generate summaries
        function_str = ast.unparse(func_node)
        
        # Remove docstring if present
        if ast.get_docstring(func_node):
            lines = function_str.split('\n')
            in_docstring = False
            filtered_lines = []
            
            for line in lines:
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    continue
                if not in_docstring:
                    filtered_lines.append(line)
            
            function_str = '\n'.join(filtered_lines)
        
        # Basic summary: extract key parts of the implementation
        summary_parts = []
        
        # Function signature
        signature = f"def {func_node.name}("
        args = []
        for arg in func_node.args.args:
            args.append(arg.arg)
        signature += ", ".join(args) + ")"
        summary_parts.append(signature)
        
        # Look for key operations
        if self._extract_return_values(func_node)['has_return']:
            summary_parts.append("Returns value(s)")
        
        control_flow = self._extract_control_flow(func_node)
        if control_flow['has_conditionals']:
            summary_parts.append("Contains conditional logic")
        if control_flow['has_loops']:
            summary_parts.append("Contains loops")
        if control_flow['has_try_except']:
            summary_parts.append("Handles exceptions")
        
        calls = self._extract_function_calls(func_node)
        if calls:
            call_names = [call['name'] for call in calls if call['name']]
            if call_names:
                summary_parts.append(f"Calls: {', '.join(call_names[:3])}" + 
                                     ("..." if len(call_names) > 3 else ""))
        
        summary = " | ".join(summary_parts)
        return summary 

class ControlFlowAnalyzer(ast.NodeVisitor):
    """Analyzes control flow in Python code to extract execution paths."""
    
    def __init__(self):
        """Initialize the control flow analyzer."""
        self.paths = []
        self.current_path = []
        self.branch_stack = []
    
    def generic_visit(self, node):
        """Visit a node in the AST.
        
        Args:
            node: The AST node to visit
        """
        # Store the node in the current path
        self.current_path.append(node)
        
        # Continue traversing
        super().generic_visit(node)
    
    def visit_If(self, node):
        """Visit an if statement in the AST.
        
        Args:
            node: The AST node to visit
        """
        # Record the branch point
        self.branch_stack.append(len(self.current_path))
        
        # Visit the test condition
        self.visit(node.test)
        
        # Visit the then branch
        for stmt in node.body:
            self.visit(stmt)
        
        # Save the path up to this point
        self.paths.append(self.current_path[:])
        
        # Restore the path to the branch point
        self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit the else branch if it exists
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            
            # Save the path
            self.paths.append(self.current_path[:])
        
        # Pop the branch point
        self.branch_stack.pop()
    
    def visit_For(self, node):
        """Visit a for loop in the AST.
        
        Args:
            node: The AST node to visit
        """
        # Record the branch point
        self.branch_stack.append(len(self.current_path))
        
        # Visit the iteration setup
        self.visit(node.target)
        self.visit(node.iter)
        
        # Visit the loop body
        for stmt in node.body:
            self.visit(stmt)
        
        # Save the path for the loop execution
        self.paths.append(self.current_path[:])
        
        # Restore the path to the branch point
        self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit the else branch if it exists
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            
            # Save the path
            self.paths.append(self.current_path[:])
        
        # Pop the branch point
        self.branch_stack.pop()
    
    def visit_While(self, node):
        """Visit a while loop in the AST.
        
        Args:
            node: The AST node to visit
        """
        # Similar to visit_For
        self.branch_stack.append(len(self.current_path))
        
        # Visit the condition
        self.visit(node.test)
        
        # Visit the loop body
        for stmt in node.body:
            self.visit(stmt)
        
        # Save the path for the loop execution
        self.paths.append(self.current_path[:])
        
        # Restore the path to the branch point
        self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit the else branch if it exists
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            
            # Save the path
            self.paths.append(self.current_path[:])
        
        # Pop the branch point
        self.branch_stack.pop()
    
    def visit_Try(self, node):
        """Visit a try-except block in the AST.
        
        Args:
            node: The AST node to visit
        """
        # Record the branch point
        self.branch_stack.append(len(self.current_path))
        
        # Visit the try body
        for stmt in node.body:
            self.visit(stmt)
        
        # Save the path for the try block
        self.paths.append(self.current_path[:])
        
        # Restore the path to the branch point
        self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit each except handler
        for handler in node.handlers:
            if handler.type:
                self.visit(handler.type)
            if handler.name:
                self.current_path.append(ast.Name(id=handler.name, ctx=ast.Store()))
            
            for stmt in handler.body:
                self.visit(stmt)
            
            # Save the path for this handler
            self.paths.append(self.current_path[:])
            
            # Restore the path to the branch point
            self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit the else branch if it exists
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            
            # Save the path
            self.paths.append(self.current_path[:])
            
            # Restore the path to the branch point
            self.current_path = self.current_path[:self.branch_stack[-1]]
        
        # Visit the finally block if it exists
        if node.finalbody:
            for stmt in node.finalbody:
                self.visit(stmt)
            
            # No need to save a separate path for finally as it's always executed
        
        # Pop the branch point
        self.branch_stack.pop()
    
    def analyze(self, node):
        """Analyze the control flow of the given AST node.
        
        Args:
            node: The AST node to analyze
            
        Returns:
            List of execution paths
        """
        self.paths = []
        self.current_path = []
        self.branch_stack = []
        
        self.visit(node)
        
        # Add the final path if not already added
        if self.current_path and self.current_path not in self.paths:
            self.paths.append(self.current_path)
        
        return self.paths 

class DocstringAnalyzer:
    """Analyzes docstrings to extract expected behavior and documentation information."""
    
    def __init__(self):
        """Initialize the docstring analyzer."""
        pass
    
    def extract_expectations_from_docstring(self, docstring: str) -> Dict[str, Any]:
        """Extract behavioral expectations from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Dictionary containing extracted expectation information
        """
        if not docstring:
            return {
                'description': '',
                'parameters': {},
                'returns': None,
                'raises': [],
                'examples': [],
                'summary': ''
            }
        
        # Clean up the docstring
        docstring = docstring.strip()
        
        # Extract different sections
        expectations = {
            'description': self._extract_description(docstring),
            'parameters': self._extract_parameters(docstring),
            'returns': self._extract_return_value(docstring),
            'raises': self._extract_exceptions(docstring),
            'examples': self._extract_examples(docstring),
            'summary': ''  # Will be generated at the end
        }
        
        # Generate a summary from the description
        expectations['summary'] = self._generate_summary(expectations['description'])
        
        return expectations
    
    def _extract_description(self, docstring: str) -> str:
        """Extract the description part of a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            The description text
        """
        # Get the text up to the first section marker
        section_markers = ['Args:', 'Parameters:', 'Returns:', 'Raises:', 'Yields:', 'Examples:', 'Note:', 'Warning:']
        
        description_end = len(docstring)
        for marker in section_markers:
            pos = docstring.find('\n' + marker)
            if pos != -1 and pos < description_end:
                description_end = pos
        
        description = docstring[:description_end].strip()
        
        # Remove any triple quotes at the start/end
        description = re.sub(r'^[\'"].*[\'"]', '', description).strip()
        
        return description
    
    def _extract_parameters(self, docstring: str) -> Dict[str, Dict[str, Any]]:
        """Extract parameter information from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Dictionary mapping parameter names to their descriptions and types
        """
        parameters = {}
        
        # Match both 'Args:' and 'Parameters:' sections
        args_match = re.search(r'(?:Args|Parameters):(.*?)(?:Returns:|Raises:|Yields:|Examples:|Note:|Warning:|$)', 
                               docstring, re.DOTALL)
        
        if not args_match:
            return parameters
        
        args_section = args_match.group(1).strip()
        
        # Extract parameter descriptions
        # This pattern matches parameter definitions in Google style docstrings
        param_pattern = r'(\w+)(?:\s*\(([^)]+)\))?\s*:(.*?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)'
        
        for match in re.finditer(param_pattern, args_section, re.DOTALL):
            param_name = match.group(1).strip()
            param_type = match.group(2).strip() if match.group(2) else None
            param_desc = match.group(3).strip()
            
            parameters[param_name] = {
                'description': param_desc,
                'type': param_type,
                'required': 'optional' not in param_desc.lower() and 'default' not in param_desc.lower()
            }
        
        return parameters

    def _extract_return_value(self, docstring: str) -> Optional[Dict[str, Any]]:
        """Extract return value information from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Dictionary with return value information or None if no return info
        """
        returns_match = re.search(r'Returns:(.*?)(?:Raises:|Yields:|Examples:|Note:|Warning:|$)', 
                                 docstring, re.DOTALL)
        
        if not returns_match:
            return None
        
        returns_section = returns_match.group(1).strip()
        
        # Check if there's a type hint
        type_hint = None
        return_desc = returns_section
        
        # Try to extract type hint if present (e.g., "Dict[str, Any]: The result...")
        type_match = re.match(r'([^:]+):(.*)', returns_section)
        if type_match:
            type_hint = type_match.group(1).strip()
            return_desc = type_match.group(2).strip()
        
        return {
            'description': return_desc,
            'type': type_hint
        }
    
    def _extract_exceptions(self, docstring: str) -> List[Dict[str, str]]:
        """Extract exception information from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            List of exception dictionaries with name and description
        """
        exceptions = []
        
        raises_match = re.search(r'Raises:(.*?)(?:Returns:|Yields:|Examples:|Note:|Warning:|$)', 
                                docstring, re.DOTALL)
        
        if not raises_match:
            return exceptions
        
        raises_section = raises_match.group(1).strip()
        
        # Extract individual exceptions
        # This pattern matches exception definitions in Google style docstrings
        exception_pattern = r'(\w+(?:\.\w+)*)(?:\s*\([^)]+\))?\s*:(.*?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)'
        
        for match in re.finditer(exception_pattern, raises_section, re.DOTALL):
            exc_name = match.group(1).strip()
            exc_desc = match.group(2).strip()
            
            exceptions.append({
                'name': exc_name,
                'description': exc_desc
            })
        
        return exceptions
    
    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            List of code example strings
        """
        examples = []
        
        # Match Examples section
        examples_match = re.search(r'Examples:(.*?)(?:Args:|Parameters:|Returns:|Raises:|Yields:|Note:|Warning:|$)', 
                                 docstring, re.DOTALL)
        
        if not examples_match:
            return examples
        
        examples_section = examples_match.group(1).strip()
        
        # Extract code blocks (marked with triple backticks)
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, examples_section, re.DOTALL)
        examples.extend(code_blocks)
        
        # Extract doctest examples (lines starting with >>>)
        if not code_blocks:
            current_example = []
            in_example = False
            
            for line in examples_section.split('\n'):
                line = line.strip()
                if line.startswith('>>>'):
                    if current_example and not in_example:
                        examples.append('\n'.join(current_example))
                        current_example = []
                    in_example = True
                    example_line = line.replace('>>>', '', 1).strip()
                    current_example.append(example_line)
                elif line.startswith('...') and in_example:
                    example_line = line.replace('...', '', 1).strip()
                    current_example.append(example_line)
                elif in_example:
                    # This might be an expected output, which we'll ignore for now
                    if not line:  # Empty line marks the end of an example
                        examples.append('\n'.join(current_example))
                        current_example = []
                        in_example = False
            
            # Don't forget the last example
            if current_example:
                examples.append('\n'.join(current_example))
        
        return examples
    
    def _generate_summary(self, description: str) -> str:
        """Generate a summary from the description.
        
        Args:
            description: The full description text
            
        Returns:
            A concise summary
        """
        # Use the first sentence as a summary
        if not description:
            return ""
            
        # Split by sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', description)
        
        if not sentences:
            return description
            
        # Use the first sentence, limited to 100 characters
        summary = sentences[0].strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."
            
        return summary

class BidirectionalValidator:
    """Validates bidirectional alignment between code implementation and documentation."""
    
    def __init__(self, threshold: float = 0.6):
        """Initialize the bidirectional validator.
        
        Args:
            threshold: Threshold for semantic similarity (default: 0.6)
        """
        self.code_analyzer = CodeAnalyzer()
        self.docstring_analyzer = DocstringAnalyzer()
        self.threshold = threshold
        
        # Try to initialize embedding model for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            if EMBEDDINGS_AVAILABLE:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Using sentence-transformers for semantic similarity")
            else:
                self.embedder = None
                logger.warning("sentence-transformers not available, using basic similarity")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            self.embedder = None
    
    def validate_function(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Validate bidirectional alignment between function implementation and docstring.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            Validation results as a dictionary
        """
        func_name = func_node.name
        docstring = ast.get_docstring(func_node)
        
        # Extract behavior from code implementation
        actual_behavior = self.code_analyzer.extract_behavior_from_function(func_node)
        
        # Extract expectations from documentation
        expected_behavior = self.docstring_analyzer.extract_expectations_from_docstring(docstring)
        
        # Validate bidirectional alignment
        validation_results = {
            'name': func_name,
            'has_docstring': docstring is not None,
            'code_to_doc': self._validate_code_to_doc(actual_behavior, expected_behavior),
            'doc_to_code': self._validate_doc_to_code(expected_behavior, actual_behavior),
            'semantic_similarity': self._compute_semantic_similarity(
                actual_behavior['summary'], 
                expected_behavior['summary']
            )
        }
        
        # Overall validation result
        validation_results['is_aligned'] = (
            validation_results['code_to_doc']['overall_score'] >= self.threshold and
            validation_results['doc_to_code']['overall_score'] >= self.threshold and
            validation_results['semantic_similarity'] >= self.threshold
        )
        
        return validation_results
    
    def _validate_code_to_doc(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code implementation against documentation expectations (Code → Doc).
        
        Args:
            actual: Actual behavior extracted from code
            expected: Expected behavior extracted from docstring
            
        Returns:
            Validation results for Code → Doc direction
        """
        # Initialize results structure
        results = {
            'parameters': {
                'missing': [],
                'type_mismatches': [],
                'scores': {}
            },
            'returns': {
                'documented': expected['returns'] is not None,
                'score': 0.0
            },
            'exceptions': {
                'undocumented': [],
                'scores': {}
            },
            'overall_score': 0.0
        }
        
        # Check parameters
        actual_params = {param['name']: param for param in actual['parameters']}
        for param_name, param_info in actual_params.items():
            if param_name not in expected['parameters']:
                results['parameters']['missing'].append(param_name)
            else:
                # Check type hint consistency if available
                if param_info['type_hint'] and expected['parameters'][param_name]['type']:
                    if not self._are_types_compatible(param_info['type_hint'], expected['parameters'][param_name]['type']):
                        results['parameters']['type_mismatches'].append({
                            'name': param_name,
                            'code_type': param_info['type_hint'],
                            'doc_type': expected['parameters'][param_name]['type']
                        })
                
                # Compute semantic similarity between usage and description
                param_usage = f"Parameter '{param_name}'"
                param_desc = expected['parameters'][param_name]['description']
                
                similarity = self._compute_semantic_similarity(param_usage, param_desc)
                results['parameters']['scores'][param_name] = similarity
        
        # Check return values
        if actual['return_values']['has_return'] and expected['returns']:
            # Check type hint consistency if available
            actual_type = actual['return_values']['type_hint']
            expected_type = expected['returns']['type']
            
            if actual_type and expected_type and not self._are_types_compatible(actual_type, expected_type):
                results['returns']['type_mismatch'] = {
                    'code_type': actual_type,
                    'doc_type': expected_type
                }
            
            # Compute semantic similarity for return description
            if actual['return_values']['return_paths']:
                actual_return = ' '.join(actual['return_values']['return_paths'])
                expected_return = expected['returns']['description']
                
                similarity = self._compute_semantic_similarity(actual_return, expected_return)
                results['returns']['score'] = similarity
        
        # Check exceptions
        for exc in actual['raises']:
            if exc['name']:
                documented = False
                for doc_exc in expected['raises']:
                    if exc['name'] == doc_exc['name']:
                        documented = True
                        
                        # Compute semantic similarity
                        exc_context = f"Exception {exc['name']}" + (f" when {exc['condition']}" if exc['condition'] else "")
                        exc_desc = doc_exc['description']
                        
                        similarity = self._compute_semantic_similarity(exc_context, exc_desc)
                        results['exceptions']['scores'][exc['name']] = similarity
                        break
                
                if not documented:
                    results['exceptions']['undocumented'].append(exc['name'])
        
        # Calculate overall score
        scores = list(results['parameters']['scores'].values())
        if results['returns']['score'] > 0:
            scores.append(results['returns']['score'])
        scores.extend(list(results['exceptions']['scores'].values()))
        
        results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return results
    
    def _validate_doc_to_code(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documentation expectations against code implementation (Doc → Code).
        
        Args:
            expected: Expected behavior extracted from docstring
            actual: Actual behavior extracted from code
            
        Returns:
            Validation results for Doc → Code direction
        """
        # Initialize results structure
        results = {
            'parameters': {
                'extra': [],
                'unused': [],
                'scores': {}
            },
            'returns': {
                'missing': False,
                'score': 0.0
            },
            'exceptions': {
                'unraised': [],
                'scores': {}
            },
            'overall_score': 0.0
        }
        
        # Check parameters
        actual_params = {param['name']: param for param in actual['parameters']}
        for param_name, param_info in expected['parameters'].items():
            if param_name not in actual_params:
                results['parameters']['extra'].append(param_name)
            else:
                # Check if parameter is unused in the code
                if actual_params[param_name]['usage_count'] == 0:
                    results['parameters']['unused'].append(param_name)
                
                # Compute semantic similarity between description and usage
                param_desc = param_info['description']
                param_usage = f"Parameter '{param_name}'"
                
                similarity = self._compute_semantic_similarity(param_desc, param_usage)
                results['parameters']['scores'][param_name] = similarity
        
        # Check return values
        if expected['returns'] and not actual['return_values']['has_return']:
            results['returns']['missing'] = True
        elif expected['returns'] and actual['return_values']['has_return']:
            # Compute semantic similarity for return description
            expected_return = expected['returns']['description']
            actual_return = ' '.join(actual['return_values']['return_paths'])
            
            similarity = self._compute_semantic_similarity(expected_return, actual_return)
            results['returns']['score'] = similarity
        
        # Check exceptions
        actual_exc_names = [exc['name'] for exc in actual['raises'] if exc['name']]
        for exc in expected['raises']:
            if exc['name'] not in actual_exc_names:
                results['exceptions']['unraised'].append(exc['name'])
            else:
                # Compute semantic similarity
                for actual_exc in actual['raises']:
                    if actual_exc['name'] == exc['name']:
                        exc_desc = exc['description']
                        exc_context = f"Exception {actual_exc['name']}" + (f" when {actual_exc['condition']}" if actual_exc.get('condition') else "")
                        
                        similarity = self._compute_semantic_similarity(exc_desc, exc_context)
                        results['exceptions']['scores'][exc['name']] = similarity
                        break
        
        # Calculate overall score
        scores = list(results['parameters']['scores'].values())
        if results['returns']['score'] > 0:
            scores.append(results['returns']['score'])
        scores.extend(list(results['exceptions']['scores'].values()))
        
        results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return results
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        if self.embedder is not None:
            # Use sentence transformers for better similarity
            try:
                emb1 = self.embedder.encode(text1)
                emb2 = self.embedder.encode(text2)
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)
                return similarity
            except Exception as e:
                logger.warning(f"Error computing embeddings: {e}")
                # Fall back to basic similarity
                
        # Basic fallback: Jaccard similarity on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity score
        """
        import numpy as np
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(v1, v2) / (norm1 * norm2)
    
    @staticmethod
    def _are_types_compatible(type1: str, type2: str) -> bool:
        """Check if two type hints are compatible.
        
        Args:
            type1: First type hint as string
            type2: Second type hint as string
            
        Returns:
            True if the types are compatible, False otherwise
        """
        # Clean and normalize the type strings
        type1 = type1.strip().replace(' ', '')
        type2 = type2.strip().replace(' ', '')
        
        # Simple exact match
        if type1 == type2:
            return True
        
        # Common type aliases
        aliases = {
            'str': ['string', 'Text'],
            'int': ['integer', 'Integer'],
            'float': ['double', 'Float', 'Double'],
            'bool': ['boolean', 'Boolean'],
            'List': ['list'],
            'Dict': ['dict', 'dictionary', 'Dictionary', 'map', 'Map'],
            'Tuple': ['tuple'],
            'Set': ['set'],
            'Optional': ['Union[None,', 'Union[...,None]', 'Union[None,...]'],
            'Any': ['object', 'Object', 'any']
        }
        
        # Check for aliases
        for base_type, alias_list in aliases.items():
            if base_type in type1:
                for alias in alias_list:
                    if alias in type2:
                        return True
        
        # Check for Optional vs. None default
        if 'Optional' in type1 and 'None' in type2:
            return True
        if 'Optional' in type2 and 'None' in type1:
            return True
        
        # More advanced compatibility checking would require parsing the types
        # and understanding subtype relationships, which is beyond the scope of this script
        
        return False

def validate_file(file_path: str, threshold: float = 0.6, verbose: bool = False) -> Dict[str, Any]:
    """Validate bidirectional alignment for all functions in a file.
    
    Args:
        file_path: Path to the Python file
        threshold: Threshold for semantic similarity
        verbose: Whether to print detailed results
        
    Returns:
        Validation results for all functions in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        module = ast.parse(source)
        validator = BidirectionalValidator(threshold=threshold)
        
        results = {
            'file_path': file_path,
            'function_results': [],
            'summary': {
                'total_functions': 0,
                'with_docstrings': 0,
                'aligned': 0,
                'code_to_doc_issues': 0,
                'doc_to_code_issues': 0,
                'semantic_issues': 0
            }
        }
        
        # Get all function definitions
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip private methods (starting with _)
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                
                validation = validator.validate_function(node)
                results['function_results'].append(validation)
                
                # Update summary statistics
                results['summary']['total_functions'] += 1
                
                if validation['has_docstring']:
                    results['summary']['with_docstrings'] += 1
                
                if validation.get('is_aligned', False):
                    results['summary']['aligned'] += 1
                else:
                    if validation['code_to_doc']['overall_score'] < threshold:
                        results['summary']['code_to_doc_issues'] += 1
                    
                    if validation['doc_to_code']['overall_score'] < threshold:
                        results['summary']['doc_to_code_issues'] += 1
                    
                    if validation['semantic_similarity'] < threshold:
                        results['summary']['semantic_issues'] += 1
        
        # Calculate percentage metrics
        total = results['summary']['total_functions']
        if total > 0:
            results['summary']['docstring_coverage'] = results['summary']['with_docstrings'] / total
            results['summary']['alignment_score'] = results['summary']['aligned'] / total
        else:
            results['summary']['docstring_coverage'] = 0
            results['summary']['alignment_score'] = 0
        
        if verbose:
            _print_validation_results(results)
            
        return results
    
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return {
            'file_path': file_path,
            'error': str(e),
            'function_results': [],
            'summary': {
                'total_functions': 0,
                'with_docstrings': 0,
                'aligned': 0,
                'docstring_coverage': 0,
                'alignment_score': 0
            }
        }


def validate_directory(dir_path: str, threshold: float = 0.6, verbose: bool = False) -> Dict[str, Any]:
    """Validate bidirectional alignment for all Python files in a directory (recursively).
    
    Args:
        dir_path: Path to the directory
        threshold: Threshold for semantic similarity
        verbose: Whether to print detailed results
        
    Returns:
        Validation results for all files in the directory
    """
    results = {
        'directory': dir_path,
        'file_results': [],
        'summary': {
            'total_files': 0,
            'total_functions': 0,
            'with_docstrings': 0,
            'aligned': 0,
            'code_to_doc_issues': 0,
            'doc_to_code_issues': 0,
            'semantic_issues': 0
        }
    }
    
    # Walk through all Python files in the directory
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_result = validate_file(file_path, threshold, verbose=False)
                
                results['file_results'].append(file_result)
                results['summary']['total_files'] += 1
                
                # Update summary statistics
                for key in ['total_functions', 'with_docstrings', 'aligned', 
                            'code_to_doc_issues', 'doc_to_code_issues', 'semantic_issues']:
                    results['summary'][key] += file_result['summary'].get(key, 0)
    
    # Calculate percentage metrics
    total_funcs = results['summary']['total_functions']
    if total_funcs > 0:
        results['summary']['docstring_coverage'] = results['summary']['with_docstrings'] / total_funcs
        results['summary']['alignment_score'] = results['summary']['aligned'] / total_funcs
    else:
        results['summary']['docstring_coverage'] = 0
        results['summary']['alignment_score'] = 0
    
    if verbose:
        _print_directory_validation_results(results)
    
    return results


def _print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results for a file.
    
    Args:
        results: Validation results for a file
    """
    print(f"\n{'=' * 80}")
    print(f"FILE: {results['file_path']}")
    print(f"{'=' * 80}")
    
    summary = results['summary']
    print(f"Total functions: {summary['total_functions']}")
    print(f"Functions with docstrings: {summary['with_docstrings']} ({summary['docstring_coverage']:.1%})")
    print(f"Aligned functions: {summary['aligned']} ({summary['alignment_score']:.1%})")
    print(f"Functions with code-to-doc issues: {summary['code_to_doc_issues']}")
    print(f"Functions with doc-to-code issues: {summary['doc_to_code_issues']}")
    print(f"Functions with semantic similarity issues: {summary['semantic_issues']}")
    print(f"{'-' * 80}")
    
    for func_result in results['function_results']:
        print(f"\nFunction: {func_result['name']}")
        print(f"  Has docstring: {func_result['has_docstring']}")
        
        if func_result['has_docstring']:
            print(f"  Semantic similarity: {func_result['semantic_similarity']:.2f}")
            print(f"  Code-to-Doc score: {func_result['code_to_doc']['overall_score']:.2f}")
            print(f"  Doc-to-Code score: {func_result['doc_to_code']['overall_score']:.2f}")
            print(f"  Is aligned: {func_result.get('is_aligned', False)}")
            
            # Print code to doc details if issues exist
            if func_result['code_to_doc']['overall_score'] < 0.6:
                print(f"  Code-to-Doc issues:")
                
                # Parameter issues
                if func_result['code_to_doc']['parameters']['missing']:
                    print(f"    - Missing parameters in docstring: {', '.join(func_result['code_to_doc']['parameters']['missing'])}")
                
                if func_result['code_to_doc']['parameters']['type_mismatches']:
                    for mismatch in func_result['code_to_doc']['parameters']['type_mismatches']:
                        print(f"    - Type mismatch for parameter '{mismatch['name']}': {mismatch['code_type']} vs {mismatch['doc_type']}")
                
                # Return value issues
                if func_result['code_to_doc']['returns'].get('type_mismatch'):
                    mismatch = func_result['code_to_doc']['returns']['type_mismatch']
                    print(f"    - Return type mismatch: {mismatch['code_type']} vs {mismatch['doc_type']}")
                
                # Exception issues
                if func_result['code_to_doc']['exceptions']['undocumented']:
                    print(f"    - Undocumented exceptions: {', '.join(func_result['code_to_doc']['exceptions']['undocumented'])}")
            
            # Print doc to code details if issues exist
            if func_result['doc_to_code']['overall_score'] < 0.6:
                print(f"  Doc-to-Code issues:")
                
                # Parameter issues
                if func_result['doc_to_code']['parameters']['extra']:
                    print(f"    - Extra parameters in docstring: {', '.join(func_result['doc_to_code']['parameters']['extra'])}")
                
                if func_result['doc_to_code']['parameters']['unused']:
                    print(f"    - Unused parameters in code: {', '.join(func_result['doc_to_code']['parameters']['unused'])}")
                
                # Return value issues
                if func_result['doc_to_code']['returns']['missing']:
                    print(f"    - Return value documented but not returned in code")
                
                # Exception issues
                if func_result['doc_to_code']['exceptions']['unraised']:
                    print(f"    - Documented exceptions not raised in code: {', '.join(func_result['doc_to_code']['exceptions']['unraised'])}")


def _print_directory_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results for a directory.
    
    Args:
        results: Validation results for a directory
    """
    print(f"\n{'=' * 80}")
    print(f"DIRECTORY: {results['directory']}")
    print(f"{'=' * 80}")
    
    summary = results['summary']
    print(f"Total files: {summary['total_files']}")
    print(f"Total functions: {summary['total_functions']}")
    print(f"Functions with docstrings: {summary['with_docstrings']} ({summary['docstring_coverage']:.1%})")
    print(f"Aligned functions: {summary['aligned']} ({summary['alignment_score']:.1%})")
    print(f"Functions with code-to-doc issues: {summary['code_to_doc_issues']}")
    print(f"Functions with doc-to-code issues: {summary['doc_to_code_issues']}")
    print(f"Functions with semantic similarity issues: {summary['semantic_issues']}")
    
    # Display top files with issues
    file_results = sorted(results['file_results'], 
                          key=lambda x: x['summary'].get('alignment_score', 0))
    
    print(f"\n{'-' * 80}")
    print("Top files with alignment issues:")
    for file_result in file_results[:5]:
        file_path = file_result['file_path']
        alignment = file_result['summary'].get('alignment_score', 0)
        total = file_result['summary'].get('total_functions', 0)
        aligned = file_result['summary'].get('aligned', 0)
        
        if total > 0:
            print(f"  {file_path}: {aligned}/{total} aligned ({alignment:.1%})")


def analyze_file(file_path: str, threshold: float = 0.6, verbose: bool = False, output_format: str = "text"):
    """Analyze bidirectional alignment for a file and output results.
    
    Args:
        file_path: Path to the Python file
        threshold: Threshold for semantic similarity
        verbose: Whether to print detailed results
        output_format: Output format (text or json)
    
    Returns:
        Results dictionary if output_format is json, None otherwise
    """
    results = validate_file(file_path, threshold, verbose=(verbose and output_format == "text"))
    
    if output_format == "json":
        return results
    elif output_format == "text" and not verbose:
        # Print summary only
        summary = results['summary']
        print(f"File: {file_path}")
        print(f"Functions: {summary['total_functions']}, With docstrings: {summary['with_docstrings']} ({summary['docstring_coverage']:.1%})")
        print(f"Aligned: {summary['aligned']} ({summary['alignment_score']:.1%})")
    
    return None


def analyze_directory(dir_path: str, threshold: float = 0.6, verbose: bool = False, output_format: str = "text"):
    """Analyze bidirectional alignment for a directory and output results.
    
    Args:
        dir_path: Path to the directory
        threshold: Threshold for semantic similarity
        verbose: Whether to print detailed results
        output_format: Output format (text or json)
    
    Returns:
        Results dictionary if output_format is json, None otherwise
    """
    results = validate_directory(dir_path, threshold, verbose=(verbose and output_format == "text"))
    
    if output_format == "json":
        return results
    elif output_format == "text" and not verbose:
        # Print summary only
        summary = results['summary']
        print(f"Directory: {dir_path}")
        print(f"Files: {summary['total_files']}, Functions: {summary['total_functions']}")
        print(f"Functions with docstrings: {summary['with_docstrings']} ({summary['docstring_coverage']:.1%})")
        print(f"Aligned functions: {summary['aligned']} ({summary['alignment_score']:.1%})")
    
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate bidirectional alignment between code and docstrings")
    parser.add_argument("path", help="Path to a Python file or directory")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for semantic similarity (default: 0.6)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format (default: text)")
    
    args = parser.parse_args()
    
    # Check if embeddings are available
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("sentence-transformers package not found. Using basic similarity metrics.")
        logger.warning("For better results, install with: pip install sentence-transformers")
    
    # Check if path exists
    path = args.path
    if not os.path.exists(path):
        logger.error(f"Path does not exist: {path}")
        return 1
    
    # Process file or directory
    if os.path.isfile(path):
        if not path.endswith('.py'):
            logger.error(f"Not a Python file: {path}")
            return 1
        
        if args.output == "json":
            results = analyze_file(path, args.threshold, args.verbose, args.output)
            print(json.dumps(results, indent=2))
        else:
            analyze_file(path, args.threshold, args.verbose, args.output)
    
    elif os.path.isdir(path):
        if args.output == "json":
            results = analyze_directory(path, args.threshold, args.verbose, args.output)
            print(json.dumps(results, indent=2))
        else:
            analyze_directory(path, args.threshold, args.verbose, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())