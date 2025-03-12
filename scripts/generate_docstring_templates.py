#!/usr/bin/env python3
"""
Docstring Template Generator for the WITHIN ML Prediction System.

This script generates Google-style docstring templates for undocumented or
partially documented classes and methods in Python files. It helps accelerate
the documentation improvement process by creating skeleton docstrings that
follow the project's documentation standards.

Usage:
    python generate_docstring_templates.py [file_or_directory] [options]

Examples:
    # Generate templates for a single file
    python generate_docstring_templates.py app/models/ml/prediction/ad_score_predictor.py
    
    # Generate templates for a specific class in a file
    python generate_docstring_templates.py app/models/ml/prediction/ad_score_predictor.py --class AdScorePredictor
    
    # Generate templates for all Python files in a directory
    python generate_docstring_templates.py app/models/ml/prediction --recursive
"""

import os
import re
import ast
import inspect
import argparse
import logging
import sys
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
import importlib.util
import textwrap
import builtins

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes code to extract contextual information for better docstring generation."""
    
    # Common patterns in function names and their descriptions
    FUNCTION_PATTERNS = {
        r'^get_': 'Retrieve',
        r'^fetch_': 'Fetch',
        r'^load_': 'Load',
        r'^calculate_': 'Calculate',
        r'^compute_': 'Compute',
        r'^process_': 'Process',
        r'^transform_': 'Transform',
        r'^convert_': 'Convert',
        r'^normalize_': 'Normalize',
        r'^validate_': 'Validate',
        r'^check_': 'Check',
        r'^is_': 'Determine if',
        r'^has_': 'Check if',
        r'^create_': 'Create',
        r'^generate_': 'Generate',
        r'^build_': 'Build',
        r'^save_': 'Save',
        r'^store_': 'Store',
        r'^update_': 'Update',
        r'^delete_': 'Delete',
        r'^remove_': 'Remove',
        r'^handle_': 'Handle',
        r'^format_': 'Format',
        r'^parse_': 'Parse',
        r'^extract_': 'Extract',
        r'^filter_': 'Filter',
        r'^sort_': 'Sort',
        r'^search_': 'Search',
        r'^find_': 'Find',
        r'^predict_': 'Predict',
        r'^train_': 'Train',
        r'^evaluate_': 'Evaluate',
        r'^preprocess_': 'Preprocess',
        r'^postprocess_': 'Postprocess',
        r'^plot_': 'Plot',
        r'^visualize_': 'Visualize',
        r'^render_': 'Render',
        r'^log_': 'Log',
        r'^init_': 'Initialize',
        r'^setup_': 'Set up',
        r'^configure_': 'Configure',
        r'^clean_': 'Clean'
    }
    
    # Common class name patterns and their descriptions
    CLASS_PATTERNS = {
        r'.*Model$': 'a machine learning model',
        r'.*Estimator$': 'an estimator',
        r'.*Predictor$': 'a predictor',
        r'.*Classifier$': 'a classifier',
        r'.*Regressor$': 'a regressor',
        r'.*Transformer$': 'a data transformer',
        r'.*Encoder$': 'an encoder',
        r'.*Decoder$': 'a decoder',
        r'.*Processor$': 'a processor',
        r'.*Handler$': 'a handler',
        r'.*Manager$': 'a manager',
        r'.*Factory$': 'a factory',
        r'.*Builder$': 'a builder',
        r'.*Controller$': 'a controller',
        r'.*Service$': 'a service',
        r'.*Repository$': 'a repository',
        r'.*DataLoader$': 'a data loader',
        r'.*DataSet$': 'a dataset',
        r'.*Storage$': 'a storage',
        r'.*Cache$': 'a cache',
        r'.*Iterator$': 'an iterator',
        r'.*Generator$': 'a generator',
        r'.*Validator$': 'a validator',
        r'.*Filter$': 'a filter',
        r'.*Parser$': 'a parser',
        r'.*Formatter$': 'a formatter',
        r'.*Renderer$': 'a renderer',
        r'.*Visualizer$': 'a visualizer',
        r'.*Exception$': 'an exception',
        r'.*Error$': 'an error',
        r'.*Mixin$': 'a mixin',
        r'.*Interface$': 'an interface',
        r'.*Strategy$': 'a strategy',
        r'.*Observer$': 'an observer',
        r'.*Listener$': 'a listener',
        r'.*Detector$': 'a detector',
        r'.*Trainer$': 'a trainer',
        r'.*Evaluator$': 'an evaluator',
        r'.*Wrapper$': 'a wrapper',
        r'.*Adapter$': 'an adapter',
        r'.*Proxy$': 'a proxy',
        r'.*Visitor$': 'a visitor',
        r'.*Serializer$': 'a serializer',
        r'.*Deserializer$': 'a deserializer',
        r'.*Config$': 'a configuration',
        r'.*Settings$': 'settings',
        r'.*Utils$': 'utility functions',
        r'.*Helper$': 'helper functions'
    }
    
    @staticmethod
    def analyze_function_name(func_name: str) -> str:
        """Generate a description based on function name pattern.
        
        Args:
            func_name: Name of the function
            
        Returns:
            str: Description of the function based on its name pattern
        """
        # Special case for __init__
        if func_name == '__init__':
            return "Initialize the instance"
            
        # Special case for __call__
        if func_name == '__call__':
            return "Call the instance as a function"
            
        # Check for patterns
        for pattern, description in CodeAnalyzer.FUNCTION_PATTERNS.items():
            if re.match(pattern, func_name):
                # Extract the object of the action from the function name
                action_object = func_name.replace(pattern[1:-1], '')
                # Convert snake_case to words
                action_object = ' '.join(action_object.split('_'))
                return f"{description} {action_object}"
        
        # Default case
        func_words = ' '.join(func_name.split('_'))
        return f"Perform operations related to {func_words}"
    
    @staticmethod
    def analyze_class_name(class_name: str) -> str:
        """Generate a description based on class name pattern.
        
        Args:
            class_name: Name of the class
            
        Returns:
            str: Description of the class based on its name pattern
        """
        for pattern, description in CodeAnalyzer.CLASS_PATTERNS.items():
            if re.match(pattern, class_name):
                return f"Class representing {description}"
        
        # Default case: convert CamelCase to words
        words = re.findall(r'[A-Z][a-z]*', class_name)
        if words:
            class_desc = ' '.join(words).lower()
            return f"Class representing a {class_desc}"
        return f"Class representing {class_name}"
    
    @staticmethod
    def analyze_function_body(node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze function body to extract contextual information.
        
        Args:
            node: AST node for the function
            
        Returns:
            Dict with extracted information
        """
        info = {
            'raises': set(),
            'uses_yield': False,
            'calls': set(),
            'property_access': set(),
            'has_return': False,
            'accesses_self': False,
            'modifies_self': False,
            'modifies_parameters': set(),
            'uses_libraries': set(),
        }
        
        # Visitor to analyze the function body
        class FunctionAnalyzer(ast.NodeVisitor):
            def visit_Raise(self, node):
                # Extract exception types
                if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    info['raises'].add(node.exc.func.id)
                elif isinstance(node.exc, ast.Name):
                    info['raises'].add(node.exc.id)
                self.generic_visit(node)
                
            def visit_Yield(self, node):
                info['uses_yield'] = True
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # Extract function calls
                if isinstance(node.func, ast.Name):
                    info['calls'].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == 'self':
                            info['calls'].add(f"self.{node.func.attr}")
                        else:
                            # Track library usage
                            info['uses_libraries'].add(node.func.value.id)
                self.generic_visit(node)
                
            def visit_Attribute(self, node):
                # Track attribute access
                if isinstance(node.value, ast.Name):
                    if node.value.id == 'self':
                        info['accesses_self'] = True
                        info['property_access'].add(f"self.{node.attr}")
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                # Track attribute assignments
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name):
                            if target.value.id == 'self':
                                info['modifies_self'] = True
                                info['property_access'].add(f"self.{target.attr}")
                            else:
                                # Track parameter modification
                                info['modifies_parameters'].add(target.value.id)
                self.generic_visit(node)
                
            def visit_Return(self, node):
                info['has_return'] = True
                self.generic_visit(node)
        
        analyzer = FunctionAnalyzer()
        analyzer.visit(node)
        
        return info
    
    @staticmethod
    def infer_return_description(func_info: Dict[str, Any], func_name: str, return_type: Optional[str]) -> str:
        """Infer a description for the return value based on function analysis.
        
        Args:
            func_info: Information extracted from function analysis
            func_name: Name of the function
            return_type: Return type hint if available
            
        Returns:
            str: Description of the return value
        """
        if not func_info['has_return'] and func_info['uses_yield']:
            if return_type and 'Iterator' in return_type:
                return f"Iterator of items"
            elif return_type and 'Generator' in return_type:
                return f"Generator of items"
            return "Yields a sequence of results"
            
        if not func_info['has_return']:
            return "None"
            
        if func_name.startswith(('is_', 'has_', 'check_')):
            return "True if condition is met, False otherwise"
            
        if func_name.startswith(('get_', 'fetch_', 'retrieve_')):
            obj = func_name.split('_', 1)[1]
            obj = ' '.join(obj.split('_'))
            return f"The requested {obj}"
            
        if func_name.startswith(('calculate_', 'compute_')):
            obj = func_name.split('_', 1)[1]
            obj = ' '.join(obj.split('_'))
            return f"The calculated {obj}"
            
        if return_type:
            if return_type == 'bool':
                return "True if successful, False otherwise"
            elif return_type == 'int':
                return "Numeric result of the operation"
            elif return_type == 'float':
                return "Floating point result of the operation"
            elif return_type == 'str':
                return "String result of the operation"
            elif return_type == 'List' or return_type.startswith('List['):
                return "List of results"
            elif return_type == 'Dict' or return_type.startswith('Dict['):
                return "Dictionary containing the results"
            elif return_type == 'Tuple' or return_type.startswith('Tuple['):
                return "Tuple containing the results"
            elif return_type == 'Optional':
                return "Result of the operation, or None if not applicable"
            
        return "Result of the operation"
    
    @staticmethod
    def infer_parameter_description(param_name: str, param_type: Optional[str]) -> str:
        """Infer a description for a parameter based on its name and type.
        
        Args:
            param_name: Name of the parameter
            param_type: Type hint of the parameter if available
            
        Returns:
            str: Description of the parameter
        """
        # Common parameter names and their descriptions
        param_descriptions = {
            'data': 'The input data to process',
            'input': 'The input to process',
            'output': 'The output destination',
            'path': 'Path to the file or directory',
            'file_path': 'Path to the file',
            'dir_path': 'Path to the directory',
            'name': 'Name of the item',
            'key': 'Key used for identification or lookup',
            'value': 'Value to be processed or stored',
            'id': 'Identifier for the item',
            'index': 'Index position in a sequence',
            'config': 'Configuration settings',
            'options': 'Optional settings to customize behavior',
            'params': 'Parameters for the operation',
            'args': 'Additional positional arguments',
            'kwargs': 'Additional keyword arguments',
            'timeout': 'Maximum time to wait in seconds',
            'callback': 'Function to call when complete',
            'verbose': 'Whether to produce verbose output',
            'force': 'Whether to force the operation',
            'recursive': 'Whether to operate recursively',
            'overwrite': 'Whether to overwrite existing items',
            'validate': 'Whether to validate the result',
            'cache': 'Whether to use caching',
            'default': 'Default value to use if none provided',
            'threshold': 'Threshold value for the operation',
            'limit': 'Maximum number of items to process',
            'offset': 'Starting position for processing',
            'batch_size': 'Number of items to process in a batch',
            'debug': 'Whether to enable debug mode',
            'model': 'The model to use for processing',
            'train_data': 'Data used for training',
            'test_data': 'Data used for testing',
            'val_data': 'Data used for validation',
            'x': 'Input features',
            'y': 'Target values',
            'learning_rate': 'Learning rate for model training',
            'epochs': 'Number of training epochs',
            'weights': 'Model weights',
            'seed': 'Random seed for reproducibility',
            'device': 'Device to use for computation',
            'dtype': 'Data type to use',
            'filename': 'Name of the file',
            'target': 'Target for the operation',
            'source': 'Source of the data',
            'mode': 'Mode of operation',
            'format': 'Format of the data',
            'size': 'Size parameter',
            'length': 'Length parameter',
            'width': 'Width parameter',
            'height': 'Height parameter',
            'depth': 'Depth parameter',
            'color': 'Color parameter',
            'style': 'Style parameter',
            'start': 'Start position or value',
            'end': 'End position or value',
            'step': 'Step size',
            'precision': 'Precision of the calculation',
            'tolerance': 'Tolerance for the calculation',
            'epsilon': 'Small value to prevent division by zero or other numerical issues',
        }
        
        # Check common parameter descriptions
        if param_name in param_descriptions:
            return param_descriptions[param_name]
            
        # Check for common parameter name patterns
        if param_name.endswith('_path'):
            base = param_name[:-5]
            base_desc = ' '.join(base.split('_'))
            return f"Path to the {base_desc}"
            
        if param_name.endswith('_file'):
            base = param_name[:-5]
            base_desc = ' '.join(base.split('_'))
            return f"File containing {base_desc}"
            
        if param_name.endswith('_dir'):
            base = param_name[:-4]
            base_desc = ' '.join(base.split('_'))
            return f"Directory containing {base_desc}"
            
        if param_name.endswith('_data'):
            base = param_name[:-5]
            base_desc = ' '.join(base.split('_'))
            return f"Data related to {base_desc}"
            
        if param_name.endswith('_list'):
            base = param_name[:-5]
            base_desc = ' '.join(base.split('_'))
            return f"List of {base_desc}"
            
        if param_name.endswith('_dict'):
            base = param_name[:-5]
            base_desc = ' '.join(base.split('_'))
            return f"Dictionary of {base_desc}"
            
        if param_name.endswith('_set'):
            base = param_name[:-4]
            base_desc = ' '.join(base.split('_'))
            return f"Set of {base_desc}"
            
        if param_name.startswith('num_'):
            base = param_name[4:]
            base_desc = ' '.join(base.split('_'))
            return f"Number of {base_desc}"
            
        if param_name.startswith('max_'):
            base = param_name[4:]
            base_desc = ' '.join(base.split('_'))
            return f"Maximum {base_desc}"
            
        if param_name.startswith('min_'):
            base = param_name[4:]
            base_desc = ' '.join(base.split('_'))
            return f"Minimum {base_desc}"
            
        if param_name.startswith('use_'):
            base = param_name[4:]
            base_desc = ' '.join(base.split('_'))
            return f"Whether to use {base_desc}"
            
        if param_name.startswith('is_'):
            base = param_name[3:]
            base_desc = ' '.join(base.split('_'))
            return f"Whether {base_desc} is applicable"
            
        if param_name.startswith('has_'):
            base = param_name[4:]
            base_desc = ' '.join(base.split('_'))
            return f"Whether {base_desc} exists or is available"
            
        # Use type information for basic description
        if param_type:
            if param_type == 'bool':
                return f"Flag indicating whether {' '.join(param_name.split('_'))}"
            elif param_type == 'int':
                return f"Integer value for {' '.join(param_name.split('_'))}"
            elif param_type == 'float':
                return f"Floating point value for {' '.join(param_name.split('_'))}"
            elif param_type == 'str':
                return f"String representing {' '.join(param_name.split('_'))}"
            elif param_type == 'List' or param_type.startswith('List['):
                return f"List of {' '.join(param_name.split('_'))}"
            elif param_type == 'Dict' or param_type.startswith('Dict['):
                return f"Dictionary of {' '.join(param_name.split('_'))}"
            elif param_type == 'Tuple' or param_type.startswith('Tuple['):
                return f"Tuple containing {' '.join(param_name.split('_'))}"
            elif param_type == 'Optional':
                return f"Optional {' '.join(param_name.split('_'))}"
            elif param_type == 'Callable' or param_type.startswith('Callable['):
                return f"Function to handle {' '.join(param_name.split('_'))}"
            
        # Default description based on parameter name
        words = ' '.join(param_name.split('_'))
        return f"The {words} parameter"

    @staticmethod
    def infer_exception_description(exception_name: str) -> str:
        """Infer a description for an exception.
        
        Args:
            exception_name: Name of the exception
            
        Returns:
            str: Description of when the exception is raised
        """
        exception_descriptions = {
            'ValueError': 'If the input value is invalid',
            'TypeError': 'If the input type is invalid',
            'KeyError': 'If the required key is not found',
            'IndexError': 'If the index is out of range',
            'FileNotFoundError': 'If the specified file is not found',
            'IOError': 'If an I/O error occurs',
            'PermissionError': 'If permission is denied for the operation',
            'NotImplementedError': 'If the method is not implemented',
            'AttributeError': 'If the required attribute is not found',
            'ImportError': 'If the required module cannot be imported',
            'ModuleNotFoundError': 'If the required module cannot be found',
            'RuntimeError': 'If a runtime error occurs',
            'Exception': 'If an error occurs during execution',
            'AssertionError': 'If an assertion fails',
            'TimeoutError': 'If the operation times out',
            'ConnectionError': 'If a connection error occurs',
            'OSError': 'If an operating system error occurs',
            'ZeroDivisionError': 'If division by zero is attempted',
            'OverflowError': 'If a calculation exceeds the maximum representable value',
            'MemoryError': 'If there is insufficient memory for the operation',
            'RecursionError': 'If the maximum recursion depth is exceeded',
            'StopIteration': 'If the iterator has no more items',
            'SyntaxError': 'If there is a syntax error in dynamic code',
            'UnicodeError': 'If there is an encoding or decoding error',
            'UnicodeDecodeError': 'If there is a decoding error',
            'UnicodeEncodeError': 'If there is an encoding error',
            'UnicodeTranslateError': 'If there is a translation error',
            'Warning': 'In certain warning conditions',
            'DeprecationWarning': 'If a deprecated feature is used',
            'PendingDeprecationWarning': 'If a feature that will be deprecated in the future is used',
            'UserWarning': 'In user-defined warning conditions',
            'SyntaxWarning': 'In syntax warning conditions',
            'RuntimeWarning': 'In runtime warning conditions',
            'FutureWarning': 'If a feature will change in a future version',
            'ImportWarning': 'In import warning conditions',
            'ResourceWarning': 'In resource warning conditions',
        }
        
        if exception_name in exception_descriptions:
            return exception_descriptions[exception_name]
            
        # Check if it's a custom exception by looking for 'Error' or 'Exception' in the name
        if 'Error' in exception_name:
            base = exception_name.replace('Error', '')
            words = re.findall(r'[A-Z][a-z]*', base)
            if words:
                error_desc = ' '.join(words).lower()
                return f"If a {error_desc} error occurs"
                
        if 'Exception' in exception_name:
            base = exception_name.replace('Exception', '')
            words = re.findall(r'[A-Z][a-z]*', base)
            if words:
                error_desc = ' '.join(words).lower()
                return f"If a {error_desc} exception occurs"
                
        return f"If a {exception_name} error condition occurs"

class DocstringTemplateVisitor(ast.NodeVisitor):
    """AST visitor that extracts information needed for docstring templates."""

    def __init__(self, source_code: str, target_class: Optional[str] = None):
        """Initialize the visitor with empty data structures.
        
        Args:
            source_code: Source code being analyzed
            target_class: Optional class name to focus on
        """
        self.source_code = source_code.splitlines()
        self.target_class = target_class
        self.current_class = None
        self.templates = []
        self.class_info = {}
        self.method_info = {}
        
    def visit_Module(self, node: ast.Module) -> None:
        """Extract module information."""
        # Check if module has a docstring
        has_docstring = (
            len(node.body) > 0 and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)
        )
        
        if not has_docstring:
            # Generate module docstring template
            self.templates.append({
                'type': 'module',
                'name': '',
                'lineno': 1,
                'params': [],
                'returns': None,
                'indent': 0,
                'existing_docstring': False,
                'docstring': self._generate_module_template()
            })
            
        self.generic_visit(node)
            
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class information."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Skip if we're targeting a specific class and this isn't it
        if self.target_class and self.target_class != node.name:
            self.generic_visit(node)
            self.current_class = old_class
            return
        
        # Check if class has a docstring
        has_docstring = (
            len(node.body) > 0 and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)
        )
        
        # Collect base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}")
        
        # Store class info
        self.class_info[node.name] = {
            'bases': bases,
            'has_docstring': has_docstring,
            'lineno': node.lineno,
            'methods': []
        }
        
        if not has_docstring:
            # Generate class docstring template
            indent = self._get_indent(node.lineno)
            self.templates.append({
                'type': 'class',
                'name': node.name,
                'lineno': node.lineno,
                'params': [],
                'returns': None,
                'indent': indent,
                'existing_docstring': False,
                'docstring': self._generate_class_template(node.name, bases)
            })
            
        self.generic_visit(node)
        self.current_class = old_class
            
    def _extract_type_annotation(self, annotation) -> Optional[str]:
        """Extract type annotation from AST nodes.
        
        Args:
            annotation: AST node representing a type annotation
            
        Returns:
            str: String representation of the type annotation, or None if not available
        """
        if annotation is None:
            return None
            
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            # Handle module.Type annotations
            if isinstance(annotation.value, ast.Name):
                return f"{annotation.value.id}.{annotation.attr}"
            elif isinstance(annotation.value, ast.Attribute):
                base = self._extract_type_annotation(annotation.value)
                return f"{base}.{annotation.attr}" if base else None
            return None
        elif isinstance(annotation, ast.Subscript):
            # Handle container types like List[str], Dict[str, int], etc.
            if isinstance(annotation.value, ast.Name):
                container = annotation.value.id
            elif isinstance(annotation.value, ast.Attribute):
                container_base = self._extract_type_annotation(annotation.value)
                container = container_base if container_base else "Unknown"
            else:
                return None
                
            # Handle simple index like List[str]
            if isinstance(annotation.slice, ast.Index):
                if hasattr(annotation.slice, 'value'):
                    # Python 3.8 and earlier
                    slice_value = annotation.slice.value
                else:
                    # Python 3.9+
                    slice_value = annotation.slice
                    
                if isinstance(slice_value, ast.Name):
                    return f"{container}[{slice_value.id}]"
                elif isinstance(slice_value, ast.Tuple):
                    # Handle complex types like Dict[str, int]
                    slice_items = []
                    for elt in slice_value.elts:
                        if isinstance(elt, ast.Name):
                            slice_items.append(elt.id)
                        else:
                            # For complex nested types, simplify to avoid parsing errors
                            slice_items.append("...")
                    return f"{container}[{', '.join(slice_items)}]"
                
            # Fall back to a simplified representation for complex generic types
            return f"{container}[...]"
        elif isinstance(annotation, ast.Constant) and annotation.value is None:
            return "None"
        elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            # Handle Union types (X | Y in Python 3.10+)
            left = self._extract_type_annotation(annotation.left)
            right = self._extract_type_annotation(annotation.right)
            if left and right:
                return f"Union[{left}, {right}]"
            return None
        elif isinstance(annotation, ast.Call) and isinstance(annotation.func, ast.Name) and annotation.func.id == 'Optional':
            # Handle Optional[X]
            if annotation.args:
                inner_type = self._extract_type_annotation(annotation.args[0])
                return f"Optional[{inner_type}]" if inner_type else "Optional"
            return "Optional"
        
        # For other complex types or unrecognized patterns
        return "Any"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function/method information."""
        # Skip if we're targeting a specific class and not in it
        if self.target_class and self.current_class != self.target_class:
            self.generic_visit(node)
            return
        
        # Skip private methods (starting with _) unless they're __init__
        if self.current_class and node.name.startswith('_') and node.name != '__init__':
            self.generic_visit(node)
            return
            
        # Skip private functions (starting with _)
        if not self.current_class and node.name.startswith('_'):
            self.generic_visit(node)
            return
            
        # Check if function/method has a docstring
        has_docstring = (
            len(node.body) > 0 and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)
        )
        
        # Collect parameter info
        params = []
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
                
            param = {'name': arg.arg, 'type_hint': None, 'default': None}
            
            # Get type hint if available using the improved type extraction
            if arg.annotation:
                param['type_hint'] = self._extract_type_annotation(arg.annotation)
                        
            params.append(param)
        
        # Get return type hint if available using the improved type extraction
        returns = None
        if node.returns:
            returns = self._extract_type_annotation(node.returns)
        
        # Add method to class info if in a class
        if self.current_class:
            if self.current_class in self.class_info:
                self.class_info[self.current_class]['methods'].append(node.name)
        
        # Store method info
        method_key = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.method_info[method_key] = {
            'params': params,
            'returns': returns,
            'has_docstring': has_docstring,
            'lineno': node.lineno
        }
        
        if not has_docstring:
            # Generate method/function docstring template
            indent = self._get_indent(node.lineno) + 4  # Add 4 spaces for within function
            func_type = 'method' if self.current_class else 'function'
            
            self.templates.append({
                'type': func_type,
                'name': method_key,
                'lineno': node.lineno,
                'params': params,
                'returns': returns,
                'indent': indent,
                'existing_docstring': False,
                'docstring': self._generate_function_template(node.name, params, returns, func_type, node)
            })
            
        self.generic_visit(node)
    
    def _get_indent(self, lineno: int) -> int:
        """Get the indentation level of the given line number.
        
        Args:
            lineno: The line number to check
            
        Returns:
            int: Number of spaces used for indentation
        """
        if lineno <= 0 or lineno > len(self.source_code):
            return 0
            
        line = self.source_code[lineno - 1]
        return len(line) - len(line.lstrip())
    
    def _generate_module_template(self) -> str:
        """Generate a template for module docstrings.
        
        Returns:
            str: Docstring template
        """
        template = '''"""
[MODULE DESCRIPTION]

This module provides [BRIEF DESCRIPTION OF MODULE FUNCTIONALITY].

Attributes:
    [ATTRIBUTE_NAME]: [ATTRIBUTE_DESCRIPTION]

Examples:
    [EXAMPLE_USAGE]
    
    >>> from [MODULE_PATH] import [CLASS_OR_FUNCTION]
    >>> [EXAMPLE_CODE]
"""'''
        return template
        
    def _generate_class_template(self, class_name: str, bases: List[str]) -> str:
        """Generate a template for class docstrings.
        
        Args:
            class_name: Name of the class
            bases: List of base classes
            
        Returns:
            str: Docstring template
        """
        inheritance = f" inheriting from {', '.join(bases)}" if bases else ""
        
        # Generate class description based on name
        class_description = CodeAnalyzer.analyze_class_name(class_name)
        
        template = f'''"""
{class_description}{inheritance}.

[ADD MORE CLASS DETAILS HERE]

Attributes:
    [ATTRIBUTE_NAME]: [ATTRIBUTE_DESCRIPTION]

Examples:
    [EXAMPLE_USAGE]
    
    >>> [INSTANCE] = {class_name}([PARAMS])
    >>> [EXAMPLE_CODE]
"""'''
        return template
    
    def _generate_function_template(
        self, 
        func_name: str, 
        params: List[Dict[str, Any]], 
        returns: Optional[str],
        func_type: str,
        node: Optional[ast.FunctionDef] = None
    ) -> str:
        """Generate a template for function/method docstrings.
        
        Args:
            func_name: Name of the function/method
            params: List of parameters
            returns: Return type hint
            func_type: Either 'function' or 'method'
            node: AST node for the function if available
            
        Returns:
            str: Docstring template
        """
        # Analyze function name and body for contextual information
        func_description = CodeAnalyzer.analyze_function_name(func_name)
        
        # Extract additional context from function body if available
        func_info = {'has_return': False, 'uses_yield': False, 'raises': set()}
        if node:
            func_info = CodeAnalyzer.analyze_function_body(node)
        
        template = f'"""\n{func_description}.\n'
        
        # Add Args section if there are parameters
        if params:
            template += "\nArgs:\n"
            for param in params:
                type_hint = f" ({param['type_hint']})" if param['type_hint'] else ""
                param_desc = CodeAnalyzer.infer_parameter_description(param['name'], param['type_hint'])
                template += f"    {param['name']}{type_hint}: {param_desc}\n"
        
        # Add Returns section
        if func_info['uses_yield']:
            template += "\nYields:\n"
            if returns and returns.startswith(('Iterator', 'Generator')):
                inner_type = re.search(r'\[(.*)\]', returns)
                if inner_type:
                    template += f"    {inner_type.group(1)}: Items yielded by the generator\n"
                else:
                    template += "    Items yielded by the generator\n"
            else:
                template += "    Items yielded one at a time\n"
        else:
            template += "\nReturns:\n"
            if returns:
                return_desc = CodeAnalyzer.infer_return_description(func_info, func_name, returns)
                template += f"    {returns}: {return_desc}\n"
            else:
                return_desc = CodeAnalyzer.infer_return_description(func_info, func_name, None)
                template += f"    {return_desc}\n"
        
        # Add Raises section if we detected exceptions
        if func_info['raises']:
            template += "\nRaises:\n"
            for exception in sorted(func_info['raises']):
                exception_desc = CodeAnalyzer.infer_exception_description(exception)
                template += f"    {exception}: {exception_desc}\n"
        else:
            # Add a generic Raises section if we couldn't detect exceptions
            template += "\nRaises:\n    [EXCEPTION_TYPE]: [DESCRIPTION]\n"
        
        # Add Examples section
        template += "\nExamples:\n    [EXAMPLE_CODE]\n"
        
        template += '"""'
        return template


def extract_templates(file_path: str, target_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract docstring templates from a Python file.
    
    Args:
        file_path: Path to the Python file
        target_class: Optional class name to focus on
        
    Returns:
        List of docstring templates
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            visitor = DocstringTemplateVisitor(content, target_class)
            visitor.visit(tree)
            return visitor.templates
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []


def format_template(template: Dict[str, Any]) -> str:
    """Format a docstring template with proper indentation.
    
    Args:
        template: Template information
        
    Returns:
        str: Formatted docstring
    """
    docstring = template['docstring']
    indent = ' ' * template['indent']
    
    # Split the docstring into lines and indent each line
    lines = docstring.splitlines()
    indented_lines = [indent + line for line in lines]
    
    # Return the indented docstring
    return '\n'.join(indented_lines)


def insert_templates_into_file(file_path: str, templates: List[Dict[str, Any]]) -> str:
    """Insert docstring templates into a file.
    
    Args:
        file_path: Path to the Python file
        templates: List of docstring templates
        
    Returns:
        str: Modified file content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Sort templates by line number in descending order to avoid shifting line numbers
    templates.sort(key=lambda t: t['lineno'], reverse=True)
    
    for template in templates:
        lineno = template['lineno']
        if template['type'] == 'module':
            # Insert module docstring at the top of the file
            formatted_template = format_template(template) + '\n\n'
            lines.insert(0, formatted_template)
        else:
            # For classes and functions, add an appropriate number of newlines
            formatted_template = format_template(template) + '\n'
            
            # Determine where to insert the docstring
            if lineno - 1 < len(lines):
                # Find the end of the class/function definition line
                definition_line = lines[lineno - 1]
                if ':' in definition_line:
                    indent = ' ' * template['indent']
                    lines.insert(lineno, f"{indent}{formatted_template}")
    
    return ''.join(lines)


class DocstringValidator:
    """Validates and scores docstring quality."""
    
    @staticmethod
    def validate_docstring(docstring: str, params: List[Dict[str, Any]], returns: Optional[str], raises: Set[str]) -> Dict[str, Any]:
        """Validate a docstring for completeness and quality.
        
        Args:
            docstring: The docstring to validate
            params: List of function parameters
            returns: Return type of the function
            raises: Set of exceptions raised by the function
            
        Returns:
            Dict containing validation results
        """
        result = {
            'score': 0,
            'max_score': 100,
            'issues': [],
            'warnings': [],
            'suggestions': [],
        }
        
        if not docstring:
            result['issues'].append("Missing docstring")
            return result
            
        # Check if docstring is just a placeholder
        if '[DESCRIPTION]' in docstring or '[ADD MORE' in docstring:
            result['issues'].append("Docstring contains placeholder text")
        
        # Parse sections from the docstring
        sections = DocstringValidator._parse_sections(docstring)
        
        # Check for description
        if not sections.get('description', '').strip():
            result['issues'].append("Missing or empty description")
        else:
            desc_lines = sections.get('description', '').strip().split('\n')
            if len(desc_lines) < 2:
                result['warnings'].append("Description is very short (only one line)")
            
            # Check for common placeholder patterns
            for line in desc_lines:
                if re.search(r'\[.*\]', line):
                    result['issues'].append(f"Description contains placeholder: {line}")
        
        # Check Args section
        if params and 'args' not in sections:
            result['issues'].append("Missing Args section with parameters present")
        elif params:
            documented_params = set()
            for line in sections['args'].split('\n'):
                line = line.strip()
                if line and not line.startswith('Args:'):
                    # Extract parameter name from the line
                    param_match = re.match(r'\s*(\w+)(\s*\([^)]+\))?\s*:', line)
                    if param_match:
                        documented_params.add(param_match.group(1))
            
            # Check for missing params
            param_names = {p['name'] for p in params}
            missing_params = param_names - documented_params
            if missing_params:
                result['issues'].append(f"Missing documentation for parameters: {', '.join(missing_params)}")
                
            # Check for extra params
            extra_params = documented_params - param_names
            if extra_params:
                result['warnings'].append(f"Documentation for non-existent parameters: {', '.join(extra_params)}")
        
        # Check Returns section
        if 'returns' not in sections and not ('yields' in sections and returns and ('Iterator' in returns or 'Generator' in returns)):
            if returns and returns != 'None':
                result['issues'].append("Missing Returns section with non-None return type")
            else:
                result['warnings'].append("Missing Returns section")
        
        # Check Yields section for generators
        if returns and ('Iterator' in returns or 'Generator' in returns) and 'yields' not in sections:
            result['issues'].append("Missing Yields section for generator function")
        
        # Check Raises section
        if raises and 'raises' not in sections:
            result['issues'].append("Missing Raises section with exceptions present")
        elif raises:
            documented_exceptions = set()
            for line in sections['raises'].split('\n'):
                line = line.strip()
                if line and not line.startswith('Raises:'):
                    # Extract exception name from the line
                    exc_match = re.match(r'\s*(\w+)\s*:', line)
                    if exc_match:
                        documented_exceptions.add(exc_match.group(1))
            
            # Check for missing exceptions
            missing_exceptions = raises - documented_exceptions
            if missing_exceptions:
                result['issues'].append(f"Missing documentation for exceptions: {', '.join(missing_exceptions)}")
        
        # Check Examples section
        if 'examples' not in sections:
            result['warnings'].append("Missing Examples section")
        elif '[EXAMPLE' in sections['examples']:
            result['issues'].append("Examples section contains placeholder text")
            
        # Calculate score
        if result['issues']:
            # Deduct 10 points per issue
            result['score'] = max(0, 100 - (len(result['issues']) * 10))
        else:
            # Start with 100 and deduct 5 points per warning
            result['score'] = max(0, 100 - (len(result['warnings']) * 5))
            
        return result
    
    @staticmethod
    def _parse_sections(docstring: str) -> Dict[str, str]:
        """Parse a docstring into its component sections.
        
        Args:
            docstring: The docstring to parse
            
        Returns:
            Dict mapping section names to their content
        """
        # Remove triple quotes and leading/trailing whitespace
        docstring = docstring.strip().strip('"""').strip("'''").strip()
        
        sections = {}
        
        # Extract the description (everything before the first section)
        match = re.search(r'^(.*?)(?:(?:\r?\n){2,}[A-Z][a-z]+:|\Z)', docstring, re.DOTALL)
        if match:
            sections['description'] = match.group(1).strip()
        
        # Extract standard sections
        section_pattern = re.compile(r'(?:^|\r?\n{2,})(Args|Returns|Yields|Raises|Examples|Attributes|Note|Warning|Warnings|Todo|See Also|References):(.*?)(?=(?:\r?\n{2,}[A-Z][a-z]+:|$))', re.DOTALL)
        for match in section_pattern.finditer(docstring):
            section_name = match.group(1).lower()
            section_content = match.group(2).strip()
            sections[section_name] = section_content
            
        return sections
    
    @staticmethod
    def suggest_improvements(validation_result: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving a docstring based on validation results.
        
        Args:
            validation_result: Result from validate_docstring
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Add suggestions based on issues and warnings
        for issue in validation_result['issues']:
            if "Missing documentation for parameters" in issue:
                suggestions.append("Add documentation for all function parameters")
            elif "Missing Args section" in issue:
                suggestions.append("Add an Args section documenting all parameters")
            elif "Missing Returns section" in issue:
                suggestions.append("Add a Returns section documenting the return value")
            elif "Missing Raises section" in issue:
                suggestions.append("Add a Raises section documenting all exceptions")
            elif "Missing or empty description" in issue:
                suggestions.append("Add a clear, descriptive summary of the function/class purpose")
            elif "placeholder" in issue.lower():
                suggestions.append("Replace placeholder text with actual documentation")
        
        for warning in validation_result['warnings']:
            if "Description is very short" in warning:
                suggestions.append("Expand the description to provide more context")
            elif "Missing Examples section" in warning:
                suggestions.append("Add an Examples section with usage examples")
                
        # Add general suggestions for improvement
        if validation_result['score'] < 50:
            suggestions.append("Consider completely rewriting the docstring for clarity and completeness")
        elif validation_result['score'] < 80:
            suggestions.append("Review the docstring to ensure it clearly explains the purpose and usage")
            
        return suggestions


def process_file(file_path: str, target_class: Optional[str] = None, apply: bool = False, validate: bool = False) -> None:
    """Process a single Python file for docstring templates.
    
    Args:
        file_path: Path to the Python file
        target_class: Optional class name to focus on
        apply: Whether to apply the changes to the file
        validate: Whether to validate existing docstrings
    """
    templates = extract_templates(file_path, target_class)
    
    if not templates and not validate:
        logger.info(f"No missing docstrings found in {file_path}")
        return
    
    if templates:
        logger.info(f"Found {len(templates)} missing docstrings in {file_path}")
    
    if validate:
        # Parse the file to get existing docstrings
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            visitor = DocstringTemplateVisitor(content, target_class)
            visitor.visit(tree)
            
            # Validate existing docstrings
            for method_key, method_info in visitor.method_info.items():
                if method_info['has_docstring']:
                    # Extract docstring
                    if '.' in method_key:
                        class_name, method_name = method_key.split('.')
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                for child in node.body:
                                    if isinstance(child, ast.FunctionDef) and child.name == method_name:
                                        docstring = ast.get_docstring(child)
                                        if docstring:
                                            # Get function body info for validation
                                            func_info = CodeAnalyzer.analyze_function_body(child)
                                            
                                            # Validate the docstring
                                            validation_result = DocstringValidator.validate_docstring(
                                                docstring,
                                                method_info['params'],
                                                method_info['returns'],
                                                func_info['raises']
                                            )
                                            
                                            # Log validation results
                                            logger.info(f"Docstring validation for {method_key}:")
                                            logger.info(f"  Score: {validation_result['score']}/100")
                                            
                                            if validation_result['issues']:
                                                logger.info(f"  Issues:")
                                                for issue in validation_result['issues']:
                                                    logger.info(f"    - {issue}")
                                                    
                                            if validation_result['warnings']:
                                                logger.info(f"  Warnings:")
                                                for warning in validation_result['warnings']:
                                                    logger.info(f"    - {warning}")
                                                    
                                            # Get improvement suggestions
                                            suggestions = DocstringValidator.suggest_improvements(validation_result)
                                            if suggestions:
                                                logger.info(f"  Suggestions:")
                                                for suggestion in suggestions:
                                                    logger.info(f"    - {suggestion}")
                                        break
                    else:
                        # For module-level functions
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == method_key:
                                docstring = ast.get_docstring(node)
                                if docstring:
                                    # Get function body info for validation
                                    func_info = CodeAnalyzer.analyze_function_body(node)
                                    
                                    # Validate the docstring
                                    validation_result = DocstringValidator.validate_docstring(
                                        docstring,
                                        method_info['params'],
                                        method_info['returns'],
                                        func_info['raises']
                                    )
                                    
                                    # Log validation results
                                    logger.info(f"Docstring validation for {method_key}:")
                                    logger.info(f"  Score: {validation_result['score']}/100")
                                    
                                    if validation_result['issues']:
                                        logger.info(f"  Issues:")
                                        for issue in validation_result['issues']:
                                            logger.info(f"    - {issue}")
                                            
                                    if validation_result['warnings']:
                                        logger.info(f"  Warnings:")
                                        for warning in validation_result['warnings']:
                                            logger.info(f"    - {warning}")
                                            
                                    # Get improvement suggestions
                                    suggestions = DocstringValidator.suggest_improvements(validation_result)
                                    if suggestions:
                                        logger.info(f"  Suggestions:")
                                        for suggestion in suggestions:
                                            logger.info(f"    - {suggestion}")
                                    break
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
    
    if apply and templates:
        modified_content = insert_templates_into_file(file_path, templates)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        logger.info(f"Applied {len(templates)} docstring templates to {file_path}")
    elif templates:
        # Just print the templates
        for i, template in enumerate(templates):
            type_name = f"{template['type']} {template['name']}" if template['name'] else f"{template['type']}"
            logger.info(f"Template {i+1} for {type_name} (line {template['lineno']}):")
            print(f"\n{format_template(template)}\n")


def process_directory(directory: str, recursive: bool = False, target_class: Optional[str] = None, apply: bool = False, validate: bool = False) -> None:
    """Process all Python files in a directory for docstring templates.
    
    Args:
        directory: Path to the directory
        recursive: Whether to recursively process subdirectories
        target_class: Optional class name to focus on
        apply: Whether to apply the changes to the files
        validate: Whether to validate existing docstrings
    """
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    process_file(file_path, target_class, apply, validate)
    else:
        for file in os.listdir(directory):
            if file.endswith('.py'):
                file_path = os.path.join(directory, file)
                process_file(file_path, target_class, apply, validate)


def main() -> None:
    """Run the docstring template generator."""
    parser = argparse.ArgumentParser(description="Generate docstring templates for Python files")
    parser.add_argument(
        "path",
        help="Path to a Python file or directory"
    )
    parser.add_argument(
        "--class", 
        dest="target_class",
        help="Only generate templates for this class"
    )
    parser.add_argument(
        "--recursive", 
        "-r", 
        action="store_true",
        help="Recursively process directories"
    )
    parser.add_argument(
        "--apply", 
        "-a", 
        action="store_true",
        help="Apply the templates to the files"
    )
    parser.add_argument(
        "--validate", 
        "-v", 
        action="store_true",
        help="Validate existing docstrings"
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
        process_file(path, args.target_class, args.apply, args.validate)
    elif os.path.isdir(path):
        process_directory(path, args.recursive, args.target_class, args.apply, args.validate)
    
    if not args.apply:
        logger.info("Templates generated. Use --apply to apply the templates to the files.")


if __name__ == "__main__":
    main() 