"""Codebase analyzer with comprehensive type safety analysis and suggested fixes.

This script analyzes a codebase to generate detailed metrics including:
- Code complexity and size metrics
- Function and class identification
- Complexity hotspot detection
- Type safety analysis with suggested fixes
- Field usage consistency checking
- Type hint validation and cross-file analysis
- Auto-generated type annotation suggestions
"""
import os
import json
import tiktoken
from typing import Dict, List, Any, Optional, Tuple, Set, DefaultDict, Counter
import ast
import logging
import time
import re
from datetime import datetime
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CodebaseDigest')

class CodebaseDigest:
    """Codebase analyzer with complexity, function analysis, and type safety checking.
    
    Analyzes a codebase to extract metrics including complexity scores,
    function details, and identify potential hotspots requiring attention.
    Provides insights into code organization, complexity, maintenance needs,
    and type safety issues through cross-reference field analysis.
    
    Attributes:
        input_directory_path: Root directory of the codebase to analyze
        output_file_path: Path where the JSON output will be written
        ignore_patterns: List of patterns to exclude from analysis
        TEXT_EXTENSIONS: Set of file extensions to analyze
        token_encoding: tiktoken encoding for token counting
        file_information: Dictionary storing analysis results per file
        complexity_hotspots: List of files with high complexity metrics
        field_usage: Dictionary tracking field name usage and types
        type_hints: Dictionary tracking type hint usage
        type_inconsistencies: List of inconsistent type usages
        fix_suggestions: Dictionary of suggested fixes for type issues
    """
    
    def __init__(self, 
                 input_directory_path: str, 
                 output_file_path: str, 
                 ignore_patterns: Optional[List[str]] = None,
                 enable_type_analysis: bool = True,
                 suggest_fixes: bool = True):
        """Initialize the CodebaseDigest analyzer.
        
        Args:
            input_directory_path: Path to the directory containing the codebase to analyze
            output_file_path: Path where the output JSON file will be written
            ignore_patterns: Optional list of patterns to exclude from analysis (uses default if None)
            enable_type_analysis: Whether to enable the type safety analysis (can be resource intensive)
            suggest_fixes: Whether to generate suggested fixes for type inconsistencies
        """
        self.input_directory_path = input_directory_path
        self.output_file_path = output_file_path
        self.ignore_patterns = ignore_patterns or [
            "__pycache__", ".git", "*.pyc", ".DS_Store",
            "venv", "site-packages", ".pytest_cache", "htmlcov"
        ]
        self.TEXT_EXTENSIONS = {'.py', '.js', '.ts', '.txt', '.md', '.json'}
        self.token_encoding = tiktoken.get_encoding("cl100k_base")
        self.file_information = {}
        self.complexity_hotspots = []
        
        # Type safety analysis data structures
        self.enable_type_analysis = enable_type_analysis
        self.suggest_fixes = suggest_fixes
        self.field_usage: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(lambda: {"files": set(), "types": set()})
        self.type_hints: Dict[str, Dict[str, str]] = {}
        self.type_inconsistencies: List[Dict[str, Any]] = []
        self.fix_suggestions: Dict[str, Dict[str, Any]] = {}
        
        # Track all defined classes for better type inference
        self.defined_classes: Set[str] = set()
        
    def process_codebase(self) -> Dict[str, Any]:
        """Process codebase and generate metrics with type safety analysis.
        
        Walks through the directory tree, processes each file, identifies
        complexity hotspots, analyzes type safety, and generates a final output report.
        
        Returns:
            Dictionary containing the complete analysis results
        """
        logger.info("Starting codebase analysis...")
        start_time = time.time()
        
        # First pass: gather all defined classes for better type inference
        logger.info("First pass: gathering class definitions...")
        for root, _, files in os.walk(self.input_directory_path):
            if self._should_ignore_path(root):
                continue
                
            for file in files:
                if self._should_ignore_path(file) or not file.endswith('.py'):
                    continue
                    
                full_path = os.path.join(root, file)
                self._extract_class_definitions(full_path)
        
        logger.info(f"Found {len(self.defined_classes)} class definitions")
        
        # Second pass: gather file metrics and field usages
        logger.info("Second pass: analyzing files and extracting field usages...")
        for root, _, files in os.walk(self.input_directory_path):
            if self._should_ignore_path(root):
                continue
                
            for file in files:
                if self._should_ignore_path(file):
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.input_directory_path)
                self._process_file(full_path, rel_path)
        
        # Third pass: analyze type consistency (if enabled)
        if self.enable_type_analysis:
            logger.info("Third pass: analyzing type consistency...")
            self._analyze_type_consistency()
            
            # Generate fix suggestions if enabled
            if self.suggest_fixes:
                logger.info("Generating fix suggestions...")
                self._generate_fix_suggestions()
            
        # Generate report
        self._identify_hotspots()
        output = self._generate_output()
        
        duration = time.time() - start_time
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        return output

    def _extract_class_definitions(self, file_path: str) -> None:
        """Extract class definitions from a Python file.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        self.defined_classes.add(node.name)
            except Exception as e:
                logger.warning(f"AST parsing failed for {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error extracting class definitions from {file_path}: {e}")
        
    def _should_ignore_path(self, path: str) -> bool:
        """Determine if path should be excluded from analysis.
        
        Args:
            path: File or directory path to check
        
        Returns:
            True if the path should be ignored, False otherwise
        """
        parts = path.split(os.sep)
        if any(p in self.ignore_patterns for p in parts):
            return True
            
        if path.startswith('.'):
            return True
            
        ext = os.path.splitext(path)[1]
        if ext and ext.lower() not in self.TEXT_EXTENSIONS:
            return True
            
        return False
        
    def _process_file(self, full_path: str, rel_path: str) -> None:
        """Extract file metrics including complexity, tokens, and structure.
        
        Processes a single file to extract metrics such as size, line count,
        tokens, and for Python files, detailed code structure information.
        Also collects field usage data for type safety analysis.
        
        Args:
            full_path: Absolute path to the file
            rel_path: Relative path from the input directory
        """
        try:
            ext = os.path.splitext(full_path)[1]
            if ext.lower() in self.TEXT_EXTENSIONS:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                try:
                    token_count = len(self.token_encoding.encode(content, disallowed_special=()))
                except Exception as e:
                    logger.warning(f"Token encoding failed for {rel_path}: {e}")
                    token_count = 0
                    
                metrics = {
                    'path': rel_path,
                    'size': len(content),
                    'lines': content.count('\n') + 1,
                    'tokens': token_count,
                }
                
                if ext == '.py':
                    python_metrics = self._get_python_metrics(content, rel_path)
                    metrics.update(python_metrics)
                    
                    # Add complexity metrics
                    complexity_metrics = self._get_complexity_metrics(content)
                    metrics.update(complexity_metrics)
                    
                    # Extract field usages for type safety analysis if enabled
                    if self.enable_type_analysis:
                        self._extract_field_usages(content, rel_path)
                        self._extract_type_hints(content, rel_path)
                    
                self.file_information[rel_path] = metrics
                    
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            
    def _get_python_metrics(self, content: str, file_path: str) -> Dict[str, Any]:
        """Get detailed Python metrics including function and class information.
        
        Parses Python code using AST to extract detailed information about
        classes, methods, functions, imports, and their structures.
        
        Args:
            content: String content of the Python file
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing metrics about classes, functions and imports
        """
        try:
            tree = ast.parse(content)
            
            # Get class and function details
            classes = []
            functions = []
            
            class NodeVisitor(ast.NodeVisitor):
                """Visitor for extracting class and function information from an AST.
                
                Traverses Python AST nodes to extract detailed information about
                class definitions, methods, functions and their parameters.
                Also collects type annotations when present.
                """
                
                def __init__(self):
                    """Initialize the NodeVisitor with empty tracking collections."""
                    self.current_class = None
                    self.classes = []
                    self.functions = []
                
                def visit_ClassDef(self, node: ast.ClassDef):
                    """Process class definitions in the AST.
                    
                    Extracts information about the class and its methods.
        
        Args:
                        node: ClassDef node from the AST
                    """
                    prev_class = self.current_class
                    self.current_class = node
                    
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'line': item.lineno,
                                'args': len(item.args.args)
                            }
                            
                            # Extract return type annotation if present
                            if item.returns:
                                if isinstance(item.returns, ast.Name):
                                    method_info['return_type'] = item.returns.id
                                elif isinstance(item.returns, ast.Attribute):
                                    method_info['return_type'] = f"{self._get_attribute_full_name(item.returns)}"
                                
                            # Extract parameter type annotations
                            arg_types = {}
                            for arg in item.args.args:
                                if arg.annotation:
                                    if isinstance(arg.annotation, ast.Name):
                                        arg_types[arg.arg] = arg.annotation.id
                                    elif isinstance(arg.annotation, ast.Attribute):
                                        arg_types[arg.arg] = f"{self._get_attribute_full_name(arg.annotation)}"
                            
                            if arg_types:
                                method_info['arg_types'] = arg_types
                                
                            methods.append(method_info)
                    
                    class_info = {
                        'name': node.name,
                        'methods': methods,
                        'line': node.lineno
                    }
                    
                    # Extract base classes
                    if node.bases:
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                bases.append(self._get_attribute_full_name(base))
                        if bases:
                            class_info['bases'] = bases
                    
                    self.classes.append(class_info)
                    
                    # Visit class body
                    self.generic_visit(node)
                    self.current_class = prev_class
                
                def visit_FunctionDef(self, node: ast.FunctionDef):
                    """Process function definitions in the AST.
                    
                    Extracts information about functions, including name, arguments,
                    and decorators. Only adds standalone functions, not methods.
                    
                    Args:
                        node: FunctionDef node from the AST
                    """
                    # Only add if not a method
                    if not self.current_class:
                        function_info = {
                            'name': node.name,
                            'line': node.lineno,
                            'args': len(node.args.args),
                            'decorators': [d.id for d in node.decorator_list 
                                         if isinstance(d, ast.Name)]
                        }
                        
                        # Extract return type annotation if present
                        if node.returns:
                            if isinstance(node.returns, ast.Name):
                                function_info['return_type'] = node.returns.id
                            elif isinstance(node.returns, ast.Attribute):
                                function_info['return_type'] = f"{self._get_attribute_full_name(node.returns)}"
                            
                        # Extract parameter type annotations
                        arg_types = {}
                        for arg in node.args.args:
                            if arg.annotation:
                                if isinstance(arg.annotation, ast.Name):
                                    arg_types[arg.arg] = arg.annotation.id
                                elif isinstance(arg.annotation, ast.Attribute):
                                    arg_types[arg.arg] = f"{self._get_attribute_full_name(arg.annotation)}"
                        
                        if arg_types:
                            function_info['arg_types'] = arg_types
                            
                        self.functions.append(function_info)
                    self.generic_visit(node)
                    
                def _get_attribute_full_name(self, node):
                    """Get the full name of an attribute (e.g., 'module.Class')."""
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        return f"{self._get_attribute_full_name(node.value)}.{node.attr}"
                    return str(node)
            
            visitor = NodeVisitor()
            visitor.visit(tree)
            
            imports = len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            
            return {
                'classes': visitor.classes,
                'functions': visitor.functions,
                'imports': imports,
                'class_count': len(visitor.classes),
                'function_count': len(visitor.functions)
            }
        except Exception as e:
            logger.error(f"Error in _get_python_metrics for {file_path}: {e}")
            return {'classes': [], 'functions': [], 'imports': 0, 'class_count': 0, 'function_count': 0}
            
    def _get_complexity_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate code complexity metrics.
        
        Analyzes Python code to calculate cyclomatic complexity based on
        control flow statements, logical operators, and comprehensions.
        
        Args:
            content: String content of the Python file
            
        Returns:
            Dictionary containing complexity score and average function length
        """
        try:
            tree = ast.parse(content)
            complexity = 0
            
            for node in ast.walk(tree):
                # Count control flow statements
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                # Count logical operators
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                # Count comprehensions
                elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
                    complexity += 1
            
            return {
                'complexity_score': complexity,
                'avg_function_length': self._calculate_avg_function_length(tree)
            }
        except Exception as e:
            logger.error(f"Error calculating complexity metrics: {e}")
            return {'complexity_score': 0, 'avg_function_length': 0}
            
    def _calculate_avg_function_length(self, tree: ast.AST) -> float:
        """Calculate average function length in lines.
        
        Args:
            tree: AST of the Python file
            
        Returns:
            Average number of lines per function, or 0 if no functions
        """
        function_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno'):
                    length = node.end_lineno - node.lineno
                    function_lengths.append(length)
        
        return sum(function_lengths) / len(function_lengths) if function_lengths else 0
            
    def _extract_field_usages(self, content: str, file_path: str) -> None:
        """Extract field name usages and their apparent types.
        
        Scans Python code to find all attribute accesses and their context
        to determine field usage patterns and infer types.
        
        Args:
            content: String content of the Python file
            file_path: Path to the file being analyzed
        """
        try:
            tree = ast.parse(content)
            
            class FieldVisitor(ast.NodeVisitor):
                """Visitor for extracting field usage information from an AST."""
                
                def __init__(self, parent, file_path):
                    """Initialize the field visitor.
        
        Args:
                        parent: The parent CodebaseDigest instance
                        file_path: The file being analyzed
                    """
                    self.parent = parent
                    self.file_path = file_path
                    self.current_class = None
                    self.current_function = None
                
                def visit_ClassDef(self, node):
                    """Visit class definition nodes.
                    
                    Args:
                        node: The ClassDef node
                    """
                    prev_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = prev_class
                
                def visit_FunctionDef(self, node):
                    """Visit function definition nodes.
        
        Args:
                        node: The FunctionDef node
                    """
                    prev_func = self.current_function
                    self.current_function = node.name
                    self.generic_visit(node)
                    self.current_function = prev_func
                
                def visit_Attribute(self, node):
                    """Visit attribute access nodes to find field usages.
                    
                    Args:
                        node: The Attribute node
                    """
                    # Only process self.attribute pattern
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        # Record the field usage
                        field_name = node.attr
                        
                        # Track in which file this field is used
                        self.parent.field_usage[field_name]["files"].add(self.file_path)
                        
                        # Try to infer type from context
                        parent_ctx = self._get_parent_context(node)
                        if parent_ctx:
                            self.parent.field_usage[field_name]["types"].add(parent_ctx)
                    
                    self.generic_visit(node)
                
                def _get_parent_context(self, node):
                    """Try to infer type from context of the attribute usage.
        
        Args:
                        node: The AST node to analyze
            
        Returns:
                        Inferred type name or None if not determinable
                    """
                    # Find the parent assignment or similar context
                    parent = self._find_parent_node(node)
                    
                    if parent:
                        # For assignments like self.field = value
                        if isinstance(parent, ast.Assign) and node in parent.targets:
                            return self._get_value_type(parent.value)
                        # For assignments where the field is in the value
                        elif isinstance(parent, ast.Assign) and isinstance(parent.value, ast.Name) and parent.value is node:
                            for target in parent.targets:
                                if isinstance(target, ast.Name):
                                    # Look for type hints for this variable
                                    for file, hints in self.parent.type_hints.items():
                                        var_key = f"{self.current_class}.{target.id}" if self.current_class else target.id
                                        if var_key in hints:
                                            return hints[var_key]
                    
                    return None
                
                def _get_value_type(self, value_node):
                    """Get the type of a value expression.
                    
                    Args:
                        value_node: The AST node representing a value
                        
                    Returns:
                        Inferred type name as a string
                    """
                    if isinstance(value_node, ast.Constant):
                        return type(value_node.value).__name__
                    elif isinstance(value_node, ast.List):
                        return 'list'
                    elif isinstance(value_node, ast.Dict):
                        return 'dict'
                    elif isinstance(value_node, ast.Set):
                        return 'set'
                    elif isinstance(value_node, ast.Tuple):
                        return 'tuple'
                    elif isinstance(value_node, ast.Call):
                        if isinstance(value_node.func, ast.Name):
                            # Check if it's a known class instantiation
                            called_name = value_node.func.id
                            if called_name in self.parent.defined_classes:
                                return called_name
                            return called_name
                        elif isinstance(value_node.func, ast.Attribute):
                            # For method calls like module.Class()
                            return value_node.func.attr
                    elif isinstance(value_node, ast.Name):
                        # For variable assignments
                        if value_node.id in ('None', 'True', 'False'):
                            return 'bool' if value_node.id in ('True', 'False') else 'NoneType'
                        # Look for type hints for this variable
                        for file, hints in self.parent.type_hints.items():
                            var_key = f"{self.current_class}.{value_node.id}" if self.current_class else value_node.id
                            if var_key in hints:
                                return hints[var_key]
                        return value_node.id  # Use the variable name as a placeholder
                    elif isinstance(value_node, ast.BinOp):
                        # Try to infer type from binary operation
                        if isinstance(value_node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                            if isinstance(value_node.left, ast.Constant) and isinstance(value_node.left.value, (int, float)):
                                return type(value_node.left.value).__name__
                            elif isinstance(value_node.right, ast.Constant) and isinstance(value_node.right.value, (int, float)):
                                return type(value_node.right.value).__name__
                    elif isinstance(value_node, ast.UnaryOp):
                        return self._get_value_type(value_node.operand)
                    elif isinstance(value_node, ast.Attribute):
                        # For attribute access like obj.attr
                        return value_node.attr
                    
                    return None
                
                def _find_parent_node(self, node, _seen=None):
                    """Find the parent node that contains this node.
        
        Args:
                        node: The node to find the parent for
                        _seen: Set of already visited nodes to avoid cycles
            
        Returns:
                        The parent node or None if not found
                    """
                    if _seen is None:
                        _seen = set()
                    if id(node) in _seen:
                        return None
                    _seen.add(id(node))
                    
                    for parent in ast.walk(ast.parse(content)):
                        for child in ast.iter_child_nodes(parent):
                            if child is node or self._contains_node(child, node, _seen):
                                return parent
                    return None
                
                def _contains_node(self, parent, node, _seen):
                    """Check if parent contains the node.
                    
                    Args:
                        parent: Potential parent node
                        node: Node to look for
                        _seen: Set of already visited nodes
                        
                    Returns:
                        True if parent contains node, False otherwise
                    """
                    if parent is node:
                    return True
                    if id(parent) in _seen:
            return False
                    _seen.add(id(parent))
                    
                    return any(self._contains_node(child, node, _seen) 
                              for child in ast.iter_child_nodes(parent))
            
            visitor = FieldVisitor(self, file_path)
            visitor.visit(tree)
            
        except Exception as e:
            logger.error(f"Error extracting field usages from {file_path}: {e}")
    
    def _extract_type_hints(self, content: str, file_path: str) -> None:
        """Extract explicit type hints from function definitions and variable annotations.
        
        Scans Python code to find explicit type annotations and records them
        for comparison with inferred types.
        
        Args:
            content: String content of the Python file
            file_path: Path to the file being analyzed
        """
        try:
            tree = ast.parse(content)
            file_key = os.path.basename(file_path)
            self.type_hints[file_key] = {}
            
            class TypeHintVisitor(ast.NodeVisitor):
                """Visitor for extracting type hint information from an AST."""
                
                def __init__(self, parent, file_key):
                    """Initialize the type hint visitor.
                    
                    Args:
                        parent: The parent CodebaseDigest instance
                        file_key: The file being analyzed
                    """
                    self.parent = parent
                    self.file_key = file_key
                    self.current_class = None
                    self.current_function = None
                
                def visit_ClassDef(self, node):
                    """Visit class definition nodes.
                    
                    Args:
                        node: The ClassDef node
                    """
                    prev_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = prev_class
                
                def visit_FunctionDef(self, node):
                    """Visit function definition nodes to extract parameter and return type hints.
                    
                    Args:
                        node: The FunctionDef node
                    """
                    prev_func = self.current_function
                    self.current_function = node.name
                    
                    # Extract parameter type hints
                    for arg in node.args.args:
                        if arg.annotation:
                            arg_name = arg.arg
                            if arg_name == 'self':
                                continue
                                
                            # Get type hint as string
                            type_hint = self._get_annotation_name(arg.annotation)
                            if type_hint:
                                # Qualify with class if this is a method
                                key = f"{self.current_class}.{arg_name}" if self.current_class else arg_name
                                context_key = f"{self.current_class}.{self.current_function}:{arg_name}" if self.current_class else f"{self.current_function}:{arg_name}"
                                self.parent.type_hints[self.file_key][context_key] = type_hint
                    
                    # Extract return type hint
                    if node.returns:
                        return_type = self._get_annotation_name(node.returns)
                        if return_type:
                            func_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                            self.parent.type_hints[self.file_key][f"{func_name}:return"] = return_type
                    
                    self.generic_visit(node)
                    self.current_function = prev_func
                
                def visit_AnnAssign(self, node):
                    """Visit variable annotations to extract type hints.
                    
                    Args:
                        node: The AnnAssign node
                    """
                    # Extract variable annotations
                    if isinstance(node.target, ast.Name):
                        var_name = node.target.id
                        type_hint = self._get_annotation_name(node.annotation)
                        if type_hint:
                            key = f"{self.current_class}.{var_name}" if self.current_class else var_name
                            self.parent.type_hints[self.file_key][key] = type_hint
                    elif isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == 'self':
                        # Handle self.attr: type annotations
                        attr_name = node.target.attr
                        type_hint = self._get_annotation_name(node.annotation)
                        if type_hint:
                            key = f"{self.current_class}.{attr_name}"
                            self.parent.type_hints[self.file_key][key] = type_hint
                    
                    self.generic_visit(node)
                def _get_annotation_name(self, annotation):
                    """Extract the type name from an annotation node.
                    
                    Args:
                        annotation: The annotation AST node
        
        Returns:
                        String representation of the type or None if not extractable
                    """
                    if isinstance(annotation, ast.Name):
                        return annotation.id
                    elif isinstance(annotation, ast.Subscript):
                        if isinstance(annotation.value, ast.Name):
                            # Handle generic types like List[str]
                            return f"{annotation.value.id}[...]"
                    elif isinstance(annotation, ast.Attribute):
                        return f"{self._get_attribute_path(annotation)}"
                    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
                        # Handle Union types (Type1 | Type2)
                        return "Union[...]"
                    return None
                
                def _get_attribute_path(self, node):
                    """Get the full path of an attribute (e.g., typing.List).
        
        Args:
                        node: The attribute AST node
            
        Returns:
                        String representation of the attribute path
                    """
                    if isinstance(node.value, ast.Name):
                        return f"{node.value.id}.{node.attr}"
                    elif isinstance(node.value, ast.Attribute):
                        return f"{self._get_attribute_path(node.value)}.{node.attr}"
                    return node.attr
            
            visitor = TypeHintVisitor(self, file_key)
            visitor.visit(tree)
            
        except Exception as e:
            logger.error(f"Error extracting type hints from {file_path}: {e}")
    
    def _analyze_type_consistency(self) -> None:
        """Analyze field usage for type consistency across the codebase.
        
        Compares field usage patterns and inferred types to identify
        potential type inconsistencies or safety issues.
        """
        logger.info("Analyzing type consistency across codebase...")
        
        # Check for fields with multiple inferred types
        for field_name, info in self.field_usage.items():
            if len(info["types"]) > 1:
                # Multiple types assigned to the same field
                self.type_inconsistencies.append({
                    "field": field_name,
                    "files": list(info["files"]),
                    "types": list(info["types"]),
                    "issue": "multiple_types",
                    "severity": "high" if "NoneType" not in info["types"] else "medium"
                })
            
            # Check against explicit type hints
            for file_key, hints in self.type_hints.items():
                for hint_name, hint_type in hints.items():
                    # Check if this hint is related to the current field
                    if field_name in hint_name and not hint_name.endswith(":return"):
                        # Simple check for type compatibility
                        type_compatible = False
                        if not info["types"]:
                            continue
                            
                        for used_type in info["types"]:
                            # Consider the type compatible if:
                            if (
                                hint_type == used_type or  # Exact match
                                hint_type == "Any" or      # Any matches anything
                                hint_type.startswith("Optional") and "NoneType" in info["types"] or  # Optional allows None
                                hint_type.startswith("Union") or  # Union could include the type
                                (hint_type.startswith("List") and used_type == "list") or  # Generic containers
                                (hint_type.startswith("Dict") and used_type == "dict") or
                                (hint_type.startswith("Set") and used_type == "set") or
                                (hint_type.startswith("Tuple") and used_type == "tuple")
                            ):
                                type_compatible = True
                                break
                        
                        if not type_compatible:
                            self.type_inconsistencies.append({
                                "field": field_name,
                                "hint_location": file_key,
                                "declared_type": hint_type,
                                "used_types": list(info["types"]),
                                "issue": "type_mismatch",
                                "severity": "high" if "NoneType" not in info["types"] else "medium"
                            })
    
    def _generate_fix_suggestions(self) -> None:
        """Generate suggested fixes for type inconsistencies.
        
        Analyzes the detected type inconsistencies and provides specific
        suggestions for fixing them.
        """
        # Process each inconsistency and generate a fix suggestion
        for issue in self.type_inconsistencies:
            field = issue["field"]
            if field not in self.fix_suggestions:
                self.fix_suggestions[field] = {
                    "field": field,
                    "issues": []
                }
            
            if issue["issue"] == "multiple_types":
                # For multiple types, suggest using Optional or Union
                types = issue["types"]
                has_none = "NoneType" in types
                non_none_types = [t for t in types if t != "NoneType"]
                
                if has_none and len(non_none_types) == 1:
                    # Suggest Optional[Type] for a single type + None
                    suggestion = f"Optional[{non_none_types[0]}]"
                elif len(types) > 1:
                    # Suggest Union for multiple types
                    suggestion = f"Union[{', '.join(sorted(types))}]"
                else:
                    suggestion = next(iter(types))
                
                self.fix_suggestions[field]["issues"].append({
                    "issue_type": "multiple_types",
                    "suggestion": suggestion,
                    "files": issue["files"],
                    "severity": issue.get("severity", "medium")
                })
                
            elif issue["issue"] == "type_mismatch":
                # For type mismatches, suggest reconciling the declarations
                declared_type = issue["declared_type"]
                used_types = issue["used_types"]
                has_none = "NoneType" in used_types
                non_none_types = [t for t in used_types if t != "NoneType"]
                
                if has_none and not declared_type.startswith("Optional") and not declared_type.startswith("Union"):
                    # Type hint should be Optional
                    suggestion = f"Optional[{declared_type}]"
                elif len(used_types) > 1 and not declared_type.startswith("Union"):
                    # Type hint should be Union
                    suggestion = f"Union[{', '.join(sorted(used_types))}]"
                else:
                    # Suggest updating the code to match the declaration
                    suggestion = f"Change code to ensure field '{field}' is always of type {declared_type}"
                
                self.fix_suggestions[field]["issues"].append({
                    "issue_type": "type_mismatch",
                    "declared_type": declared_type,
                    "used_types": used_types,
                    "hint_location": issue["hint_location"],
                    "suggestion": suggestion,
                    "severity": issue.get("severity", "medium")
                })
            
        # Sort suggestions by severity
        for field, suggestion in self.fix_suggestions.items():
            suggestion["issues"] = sorted(suggestion["issues"], 
                                         key=lambda x: 0 if x["severity"] == "high" else 1)
    
    def _identify_hotspots(self) -> None:
        """Identify complexity hotspots in the codebase.
        
        Analyzes the collected metrics to identify files with high complexity
        or excessively long functions that may need attention. Hotspots are
        potential candidates for refactoring or code quality improvements.
        """
        # Consider files with high complexity or long functions
        hotspots = []
        for path, metrics in self.file_information.items():
            if metrics.get('complexity_score', 0) > 10 or metrics.get('avg_function_length', 0) > 20:
                hotspots.append({
                    'path': path,
                    'complexity': metrics.get('complexity_score', 0),
                    'avg_function_length': metrics.get('avg_function_length', 0),
                    'functions': len(metrics.get('functions', [])),
                    'classes': len(metrics.get('classes', []))
                })
        
        self.complexity_hotspots = sorted(hotspots, key=lambda x: x['complexity'], reverse=True)[:5]
            
    def _generate_output(self) -> Dict[str, Any]:
        """Generate analysis output with complexity metrics and type safety information.
        
        Creates a JSON file containing the full analysis results, including
        summary statistics, detailed per-file metrics, and type safety analysis.
            
        Returns:
            Dictionary containing the complete analysis results
        """
        summary = {
            'total_files': len(self.file_information),
            'total_lines': sum(f['lines'] for f in self.file_information.values()),
            'total_tokens': sum(f['tokens'] for f in self.file_information.values()),
            'python_files': len([f for f in self.file_information if f.endswith('.py')]),
            'total_functions': sum(f.get('function_count', 0) for f in self.file_information.values()),
            'total_classes': sum(f.get('class_count', 0) for f in self.file_information.values()),
            'complexity_hotspots': self.complexity_hotspots,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add type safety information if enabled
        if self.enable_type_analysis:
            # Count high severity issues
            high_severity_issues = sum(1 for issue in self.type_inconsistencies 
                                     if issue.get("severity", "medium") == "high")
            
            summary['type_safety'] = {
                'field_count': len(self.field_usage),
                'type_hints_count': sum(len(hints) for hints in self.type_hints.values()),
                'type_inconsistencies_count': len(self.type_inconsistencies),
                'high_severity_issues': high_severity_issues
            }
        
        output = {
            'summary': summary,
            'files': [
                {
                    'path': path,
                    **metrics
                }
                for path, metrics in sorted(
                    self.file_information.items(),
                    key=lambda x: x[1].get('complexity_score', 0),
                    reverse=True
                )
            ]
        }
        
        # Add type safety details if enabled
        if self.enable_type_analysis:
            # Only include fields with types for clarity
            output['type_safety'] = {
                'field_usage': {field: {
                    'files': list(info['files']),
                    'types': list(info['types']) 
                } for field, info in self.field_usage.items() if info['types']},
                'type_inconsistencies': self.type_inconsistencies
            }
            
            # Add fix suggestions if enabled
            if self.suggest_fixes:
                output['type_safety']['fix_suggestions'] = list(self.fix_suggestions.values())
        
        with open(self.output_file_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Analysis written to {self.output_file_path}")
        if self.complexity_hotspots:
            logger.info("Complexity hotspots found:")
            for hotspot in self.complexity_hotspots:
                logger.info(f"  {hotspot['path']}: complexity={hotspot['complexity']}, "
                          f"avg_function_length={hotspot['avg_function_length']:.1f}")
        
        if self.enable_type_analysis and self.type_inconsistencies:
            logger.info(f"Found {len(self.type_inconsistencies)} potential type inconsistencies")
            if high_severity_issues:
                logger.warning(f"  {high_severity_issues} high severity type issues need attention")
        
        return output

    def generate_typing_report(self, output_path: str) -> None:
        """Generate a detailed report on typing issues in the codebase.
        
        Creates a markdown report with detailed information about typing issues,
        suggested fixes, and recommendations for improving type safety.
        
        Args:
            output_path: Path where the markdown report will be written
        """
        if not self.enable_type_analysis:
            logger.warning("Cannot generate typing report - type analysis not enabled")
            return
            
        # Count issue types for summary
        issue_counts = Counter(issue["issue"] for issue in self.type_inconsistencies)
        high_severity = sum(1 for issue in self.type_inconsistencies 
                          if issue.get("severity", "medium") == "high")
        
        # Generate markdown report
        with open(output_path, 'w') as f:
            f.write("# Type Safety Analysis Report\n\n")
            
            # Overall summary
            f.write("## Summary\n\n")
            f.write(f"- **Total fields analyzed**: {len(self.field_usage)}\n")
            f.write(f"- **Type hints found**: {sum(len(hints) for hints in self.type_hints.values())}\n")
            f.write(f"- **Type inconsistencies detected**: {len(self.type_inconsistencies)}\n")
            f.write(f"- **High severity issues**: {high_severity}\n\n")
            
            f.write("### Issue Types\n\n")
            for issue_type, count in issue_counts.items():
                f.write(f"- **{issue_type}**: {count}\n")
            f.write("\n")
            
            # High severity issues
            if high_severity:
                f.write("## High Severity Issues\n\n")
                f.write("These issues should be addressed immediately to prevent potential runtime errors.\n\n")
                
                for issue in self.type_inconsistencies:
                    if issue.get("severity", "medium") == "high":
                        f.write(f"### Field: `{issue['field']}`\n\n")
                        
                        if issue["issue"] == "multiple_types":
                            f.write("**Issue**: Field used with multiple incompatible types\n\n")
                            f.write(f"**Types**: {', '.join(issue['types'])}\n")
                            f.write(f"**Files**: {', '.join(issue['files'])}\n\n")
                            
                            if issue["field"] in self.fix_suggestions:
                                for fix in self.fix_suggestions[issue["field"]]["issues"]:
                                    if fix["issue_type"] == "multiple_types":
                                        f.write(f"**Suggested Fix**: Use type annotation `{fix['suggestion']}`\n\n")
                                        
                        elif issue["issue"] == "type_mismatch":
                            f.write("**Issue**: Type hint doesn't match actual usage\n\n")
                            f.write(f"**Declared Type**: `{issue['declared_type']}`\n")
                            f.write(f"**Actual Types**: {', '.join(issue['used_types'])}\n")
                            f.write(f"**Declaration Location**: {issue['hint_location']}\n\n")
                            
                            if issue["field"] in self.fix_suggestions:
                                for fix in self.fix_suggestions[issue["field"]]["issues"]:
                                    if fix["issue_type"] == "type_mismatch":
                                        f.write(f"**Suggested Fix**: {fix['suggestion']}\n\n")
            
            # All issues by field
            f.write("## All Issues by Field\n\n")
            for field, suggestion in self.fix_suggestions.items():
                f.write(f"### Field: `{field}`\n\n")
                
                # Group issues by type
                multiple_types_issues = [i for i in suggestion["issues"] if i["issue_type"] == "multiple_types"]
                type_mismatch_issues = [i for i in suggestion["issues"] if i["issue_type"] == "type_mismatch"]
                
                if multiple_types_issues:
                    f.write("#### Multiple Types Issue\n\n")
                    for issue in multiple_types_issues:
                        f.write(f"**Severity**: {issue['severity']}\n")
                        f.write(f"**Files**: {', '.join(issue['files'])}\n")
                        f.write(f"**Suggested Fix**: Use type annotation `{issue['suggestion']}`\n\n")
                
                if type_mismatch_issues:
                    f.write("#### Type Mismatch Issues\n\n")
                    for issue in type_mismatch_issues:
                        f.write(f"**Severity**: {issue['severity']}\n")
                        f.write(f"**Declared Type**: `{issue['declared_type']}`\n")
                        f.write(f"**Actual Types**: {', '.join(issue['used_types'])}\n")
                        f.write(f"**Declaration Location**: {issue['hint_location']}\n")
                        f.write(f"**Suggested Fix**: {issue['suggestion']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Add type hints to all functions and methods**\n")
            f.write("2. **Use Optional[T] for fields that can be None**\n")
            f.write("3. **Use Union[T1, T2, ...] for fields that can have multiple types**\n")
            f.write("4. **Consider using a static type checker like mypy or Pyright**\n")
            f.write("5. **Add consistent field initializations in __init__ methods**\n\n")
            
            f.write("## Most Common Fields with Issues\n\n")
            
            # Find fields used in the most files
            common_fields = sorted(
                [(field, len(info["files"])) for field, info in self.field_usage.items() 
                 if field in self.fix_suggestions],
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            if common_fields:
                f.write("| Field | File Count | Types | Suggestion |\n")
                f.write("|-------|------------|-------|------------|\n")
                
                for field, count in common_fields:
                    types = ", ".join(list(self.field_usage[field]["types"]))
                    suggestion = next((fix["suggestion"] for fix in self.fix_suggestions[field]["issues"]), "")
                    f.write(f"| `{field}` | {count} | {types} | {suggestion} |\n")
            
        logger.info(f"Type safety report written to {output_path}")


if __name__ == "__main__":
    analyzer = CodebaseDigest(
        input_directory_path="/Users/alecposner/WITHIN",
        output_file_path="/Users/alecposner/WITHIN/code_analysis.json",
        enable_type_analysis=True,
        suggest_fixes=True
    )
    result = analyzer.process_codebase()
    analyzer.generate_typing_report("/Users/alecposner/WITHIN/typing_report.md")