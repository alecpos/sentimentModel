#!/usr/bin/env python3
"""
Mac-Compatible NLP-Enhanced Docstring Generator for WITHIN ML Prediction System.

This script provides docstring generation capabilities with state-of-the-art
NLP techniques for macOS systems. It avoids TensorFlow dependencies to work
around Metal plugin compatibility issues.

Usage:
    python mac_docstring_generator.py [file_or_directory] [options]

Examples:
    # Generate enhanced docstrings for a file
    python mac_docstring_generator.py app/models/ml/prediction/ad_score_predictor.py
    
    # Generate docstrings for all Python files in a directory
    python mac_docstring_generator.py app/models/ml/prediction --recursive
    
    # Generate docstrings in NumPy style
    python mac_docstring_generator.py app/core/validation.py --style numpy
"""

import os
import sys
import ast
import re
import json
import argparse
import logging
import importlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import PyTorch and numpy
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch libraries not available. For full functionality, install them with:\n"
        "pip install torch numpy"
    )

# Attempt to import sentence-transformers with PyTorch backend only
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
try:
    # This will prevent TensorFlow from being loaded
    sys.modules['tensorflow'] = None
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
    logger.info("Successfully loaded sentence-transformers with PyTorch backend")
except ImportError as e:
    NLP_AVAILABLE = False
    logger.warning(
        f"Failed to load sentence-transformers: {str(e)}\n"
        "For full functionality, install it with:\n"
        "pip install -U 'sentence-transformers[torch]'"
    )

@dataclass
class DocstringCandidate:
    """Represents a generated docstring candidate with quality metrics.
    
    This class stores a generated docstring along with its quality metrics
    to allow for comparison and selection of the best candidate.
    
    Attributes:
        text: The generated docstring text
        correctness_score: Score for factual accuracy (0-1)
        clarity_score: Score for clarity and readability (0-1)
        conciseness_score: Score for being concise (0-1)
        completeness_score: Score for covering all aspects (0-1)
        quality_score: Overall weighted quality score (0-1)
        style: Docstring style (google, numpy, sphinx)
    """
    
    text: str
    correctness_score: float = 0.0
    clarity_score: float = 0.0
    conciseness_score: float = 0.0
    completeness_score: float = 0.0
    quality_score: float = 0.0
    style: str = "google"
    
    def __post_init__(self):
        """Calculate the overall quality score if not already set."""
        if self.quality_score == 0.0:
            # Default weights for different aspects
            weights = {
                "correctness": 0.4,  # Highest weight for correctness
                "completeness": 0.3,
                "clarity": 0.2,
                "conciseness": 0.1
            }
            
            self.quality_score = (
                weights["correctness"] * self.correctness_score +
                weights["completeness"] * self.completeness_score +
                weights["clarity"] * self.clarity_score +
                weights["conciseness"] * self.conciseness_score
            )


class CodeAnalyzer:
    """Analyzes Python code to extract information for docstring generation.
    
    This class parses and analyzes Python code to extract relevant information
    needed for generating meaningful docstrings, including function signatures,
    variable names, control flow, and dependencies.
    """
    
    def __init__(self, file_path: Optional[str] = None, code: Optional[str] = None):
        """Initialize the code analyzer.
        
        Args:
            file_path: Path to the Python file to analyze
            code: String containing Python code to analyze
        """
        self.file_path = file_path
        self.code = code
        self.ast_tree = None
        self.imports = []
        self.classes = {}
        self.functions = {}
        self.module_docstring = None
        
        if file_path and not code:
            self._load_code_from_file()
            
        if self.code:
            self._parse_code()
    
    def _load_code_from_file(self):
        """Load code from the specified file path."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.code = f.read()
        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {str(e)}")
            raise
    
    def _parse_code(self):
        """Parse the code into an AST and extract relevant information."""
        try:
            self.ast_tree = ast.parse(self.code)
            self._extract_module_info()
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            raise
    
    def _extract_module_info(self):
        """Extract module-level information from the AST."""
        for node in ast.walk(self.ast_tree):
            # Extract module docstring
            if isinstance(node, ast.Module) and ast.get_docstring(node):
                self.module_docstring = ast.get_docstring(node)
            
            # Extract imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._process_import(node)
            
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                self._process_class(node)
            
            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node)
    
    def _process_import(self, node):
        """Process an import statement."""
        if isinstance(node, ast.Import):
            for name in node.names:
                self.imports.append({
                    'module': name.name,
                    'alias': name.asname
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for name in node.names:
                self.imports.append({
                    'module': f"{module}.{name.name}" if module else name.name,
                    'alias': name.asname,
                    'from_import': True
                })
    
    def _process_class(self, node):
        """Process a class definition."""
        docstring = ast.get_docstring(node)
        methods = {}
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item)
                methods[item.name] = method_info
        
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_full_name(base)}")
        
        self.classes[node.name] = {
            'docstring': docstring,
            'methods': methods,
            'bases': bases,
            'decorators': self._extract_decorators(node),
            'line': node.lineno
        }
    
    def _process_function(self, node):
        """Process a function definition."""
        # Skip processing if this is a method (will be handled by _process_class)
        if not self._is_class_method(node):
            function_info = self._extract_function_info(node)
            self.functions[node.name] = function_info
    
    def _extract_function_info(self, node):
        """Extract information from a function node."""
        docstring = ast.get_docstring(node)
        
        # Get args and annotations
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation)
            }
            args.append(arg_info)
        
        # Get return annotation
        returns = self._get_annotation(node.returns)
        
        # Check if this is a property
        is_property = any(
            d.id == 'property' for d in node.decorator_list 
            if isinstance(d, ast.Name)
        )
        
        return {
            'docstring': docstring,
            'args': args,
            'returns': returns,
            'decorators': self._extract_decorators(node),
            'is_property': is_property,
            'line': node.lineno,
            'has_docstring': docstring is not None
        }
    
    def _get_annotation(self, annotation):
        """Convert an annotation node to a string representation."""
        if annotation is None:
            return None
        
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return self._get_attribute_full_name(annotation)
        elif isinstance(annotation, ast.Subscript):
            return self._get_subscript_annotation(annotation)
        elif isinstance(annotation, ast.Constant):
            return repr(annotation.value)
        
        # Handle other annotation types
        return ast.unparse(annotation)
    
    def _get_attribute_full_name(self, node):
        """Get the full name of an attribute (e.g., module.submodule.name)."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_full_name(node.value)}.{node.attr}"
        return str(node)
    
    def _get_subscript_annotation(self, node):
        """Handle subscript annotations like List[str], Dict[str, int], etc."""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python versions
            if isinstance(node.value, ast.Name):
                container = node.value.id
                if hasattr(node, "slice") and isinstance(node.slice, ast.Index):
                    slice_value = node.slice.value
                    if isinstance(slice_value, ast.Name):
                        param = slice_value.id
                    else:
                        param = str(slice_value)
                    return f"{container}[{param}]"
            return "complex_type"
    
    def _extract_decorators(self, node):
        """Extract decorator information from a node."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_attribute_full_name(decorator))
        return decorators
    
    def _is_class_method(self, node):
        """Check if a function node is a method inside a class."""
        for parent in ast.walk(self.ast_tree):
            if isinstance(parent, ast.ClassDef):
                for child in parent.body:
                    if child == node:
                        return True
        return False
    
    def get_context_for_element(self, element_name: str, element_type: str = 'function') -> Dict[str, Any]:
        """Get the code context for a specific element (function or class).
        
        Args:
            element_name: Name of the function or class
            element_type: Type of element ('function' or 'class')
            
        Returns:
            Dictionary with context information
        """
        element_info = None
        if element_type == 'function':
            element_info = self.functions.get(element_name)
        elif element_type == 'class':
            element_info = self.classes.get(element_name)
        
        if not element_info:
            return {}
        
        # Get the line number of the element
        line_no = element_info.get('line', 0)
        
        # Extract the code lines for the element
        lines = self.code.split('\n')
        
        # Try to determine the end of the element
        # (This is a simplification, a more accurate approach would use the AST)
        element_lines = []
        in_element = False
        indent_level = None
        
        for i, line in enumerate(lines):
            if i + 1 == line_no:
                in_element = True
                indent_match = re.match(r'^(\s*)', line)
                indent_level = len(indent_match.group(1)) if indent_match else 0
                element_lines.append(line)
            elif in_element:
                if line.strip() == '' or line.strip().startswith('#'):
                    element_lines.append(line)
                    continue
                    
                indent_match = re.match(r'^(\s*)', line)
                current_indent = len(indent_match.group(1)) if indent_match else 0
                
                if current_indent <= indent_level and line.strip():
                    break
                    
                element_lines.append(line)
        
        return {
            'name': element_name,
            'type': element_type,
            'code': '\n'.join(element_lines),
            'info': element_info,
            'imports': self.imports
        } 

class NLPEnhancedCodeAnalyzer:
    """Enhances code analysis with NLP capabilities for better docstring generation.
    
    This class extends the code analysis capabilities with natural language processing
    techniques to improve docstring generation, leveraging PyTorch exclusively to
    avoid TensorFlow Metal plugin compatibility issues on macOS.
    """
    
    def __init__(self, code_analyzer: CodeAnalyzer):
        """Initialize the NLP enhanced code analyzer.
        
        Args:
            code_analyzer: An instance of CodeAnalyzer containing parsed code
        """
        self.code_analyzer = code_analyzer
        self.model = None
        self.embeddings_cache = {}
        
        if TORCH_AVAILABLE and NLP_AVAILABLE:
            try:
                self._initialize_model()
            except Exception as e:
                logger.warning(
                    f"Error initializing NLP model: {str(e)}. "
                    "Will continue with basic docstring generation."
                )
    
    def _initialize_model(self):
        """Initialize the sentence transformer model for code understanding."""
        try:
            # Use a smaller model that's optimized for code
            model_name = "all-MiniLM-L6-v2"  # Smaller, faster model for code embedding
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded {model_name} model with PyTorch backend")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence-transformer model: {str(e)}")
            self.model = None
    
    def analyze_code_context(self, element_name: str, element_type: str = 'function') -> Dict[str, Any]:
        """Analyze the code context of an element using NLP techniques.
        
        Args:
            element_name: Name of the function or class
            element_type: Type of element ('function' or 'class')
            
        Returns:
            Enhanced context with NLP analysis results
        """
        # Get basic context from code analyzer
        context = self.code_analyzer.get_context_for_element(element_name, element_type)
        
        if not context:
            return {}
        
        # Extract additional context information using NLP
        enhanced_context = context.copy()
        
        # Extract variable names
        enhanced_context['variable_names'] = self._extract_variable_names(context['code'])
        
        # Extract code semantics if NLP model is available
        if self.model is not None and TORCH_AVAILABLE:
            enhanced_context['embedding'] = self._generate_embedding(context['code'])
            enhanced_context['keywords'] = self._extract_keywords(context['code'])
        
        return enhanced_context
    
    def _extract_variable_names(self, code: str) -> List[str]:
        """Extract variable names from code using regex.
        
        Args:
            code: The code to analyze
            
        Returns:
            List of variable names
        """
        variable_pattern = r'(\w+)\s*='
        variables = re.findall(variable_pattern, code)
        
        # Filter out keywords and common names
        keywords = set([
            'True', 'False', 'None', 'self', 'cls', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'finally', 'with', 'as',
            'def', 'class', 'return', 'yield'
        ])
        
        return [var for var in variables if var not in keywords]
    
    def _generate_embedding(self, code: str) -> Optional[Union[List[float], np.ndarray]]:
        """Generate embedding for code using the NLP model.
        
        Args:
            code: The code to generate embedding for
            
        Returns:
            Numpy array or list containing the code embedding
        """
        if not self.model or code in self.embeddings_cache:
            return self.embeddings_cache.get(code)
        
        try:
            # Use PyTorch-only sentence-transformers
            with torch.no_grad():
                embedding = self.model.encode(code)
                
                # Convert to standard Python list for better serialization
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy().tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                self.embeddings_cache[code] = embedding
                return embedding
        except Exception as e:
            logger.warning(f"Error generating embedding: {str(e)}")
            return None
    
    def _extract_keywords(self, code: str) -> List[str]:
        """Extract important keywords from code using frequency analysis.
        
        Args:
            code: The code to analyze
            
        Returns:
            List of important keywords
        """
        # Simple word frequency analysis
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        
        # Filter out common keywords and short words
        stop_words = {
            'def', 'class', 'self', 'return', 'if', 'else', 'elif', 'for', 
            'while', 'try', 'except', 'with', 'as', 'import', 'from', 'in',
            'and', 'or', 'not', 'is', 'None', 'True', 'False', 'pass',
            'raise', 'continue', 'break', 'assert', 'print', 'len'
        }
        
        filtered_words = [
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(
            word_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top keywords (up to 10)
        return [word for word, count in sorted_words[:10]]
    
    def get_similar_functions(self, function_name: str, max_count: int = 3) -> List[str]:
        """Find semantically similar functions in the codebase.
        
        Args:
            function_name: The name of the target function
            max_count: Maximum number of similar functions to return
            
        Returns:
            List of similar function names
        """
        if not self.model or not TORCH_AVAILABLE:
            return []
        
        target_context = self.code_analyzer.get_context_for_element(function_name, 'function')
        if not target_context:
            return []
        
        target_code = target_context.get('code', '')
        target_embedding = self._generate_embedding(target_code)
        
        if target_embedding is None:
            return []
        
        # Compare with other functions
        similarities = []
        
        for name, _ in self.code_analyzer.functions.items():
            if name == function_name:
                continue
                
            context = self.code_analyzer.get_context_for_element(name, 'function')
            code = context.get('code', '')
            embedding = self._generate_embedding(code)
            
            if embedding is not None:
                # Convert to numpy arrays for cosine similarity
                a = np.array(target_embedding)
                b = np.array(embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                similarities.append((name, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top similar functions
        return [name for name, _ in similarities[:max_count]] 

class DocstringGenerator:
    """Generates NLP-enhanced docstrings with multiple candidates and validation.
    
    This class provides functionality for generating high-quality docstrings
    using NLP techniques, with PyTorch-specific implementation for macOS
    compatibility. It supports multiple docstring styles and quality metrics.
    """
    
    def __init__(self, style: str = "google"):
        """Initialize the docstring generator.
        
        Args:
            style: Docstring style ('google', 'numpy', or 'sphinx')
        """
        self.style = style
        self.nlp_analyzer = None
        
        # Template patterns for different docstring styles
        self.style_templates = {
            "google": {
                "function": self._google_function_template,
                "class": self._google_class_template,
                "method": self._google_method_template,
                "property": self._google_property_template
            },
            "numpy": {
                "function": self._numpy_function_template,
                "class": self._numpy_class_template,
                "method": self._numpy_method_template,
                "property": self._numpy_property_template
            },
            "sphinx": {
                "function": self._sphinx_function_template,
                "class": self._sphinx_class_template,
                "method": self._sphinx_method_template,
                "property": self._sphinx_property_template
            }
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file to identify docstring needs.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with analysis results
        """
        # Create a code analyzer for the file
        code_analyzer = CodeAnalyzer(file_path=file_path)
        
        # Create NLP analyzer if possible
        if TORCH_AVAILABLE and NLP_AVAILABLE:
            self.nlp_analyzer = NLPEnhancedCodeAnalyzer(code_analyzer)
        
        # Get all functions and classes that need docstrings
        missing_docstrings = {
            "functions": [],
            "classes": []
        }
        
        # Check functions
        for name, info in code_analyzer.functions.items():
            if not info.get('has_docstring'):
                missing_docstrings["functions"].append(name)
        
        # Check classes
        for name, info in code_analyzer.classes.items():
            if not info.get('docstring'):
                missing_docstrings["classes"].append(name)
            
            # Check methods
            for method_name, method_info in info.get('methods', {}).items():
                if not method_info.get('has_docstring'):
                    missing_docstrings.setdefault("methods", []).append(
                        f"{name}.{method_name}"
                    )
        
        return {
            "missing_docstrings": missing_docstrings,
            "file_path": file_path,
            "code_analyzer": code_analyzer
        }
    
    def generate_optimal_docstring(
        self, 
        code_analyzer: CodeAnalyzer,
        element_name: str, 
        element_type: str = 'function',
        num_candidates: int = 3
    ) -> List[DocstringCandidate]:
        """Generate optimal docstring for a code element with multiple candidates.
        
        Args:
            code_analyzer: The code analyzer containing the parsed code
            element_name: Name of the function or class
            element_type: Type of element ('function' or 'class')
            num_candidates: Number of docstring candidates to generate
            
        Returns:
            List of docstring candidates sorted by quality
        """
        # Create NLP analyzer if it doesn't exist
        if not self.nlp_analyzer and TORCH_AVAILABLE and NLP_AVAILABLE:
            self.nlp_analyzer = NLPEnhancedCodeAnalyzer(code_analyzer)
        
        # Get context (basic or NLP-enhanced)
        if self.nlp_analyzer:
            context = self.nlp_analyzer.analyze_code_context(element_name, element_type)
        else:
            context = code_analyzer.get_context_for_element(element_name, element_type)
        
        if not context:
            logger.warning(f"Could not get context for {element_type} '{element_name}'")
            return []
        
        # Generate template docstring
        template_func = self._get_template_func(element_type)
        if not template_func:
            logger.warning(f"No template available for {element_type}")
            return []
        
        # Generate different variations using templates
        candidates = []
        
        # Create base candidate using the template
        base_docstring = template_func(context)
        base_candidate = DocstringCandidate(
            text=base_docstring,
            style=self.style
        )
        
        # Evaluate base candidate
        self._evaluate_candidate_quality(base_candidate, context)
        candidates.append(base_candidate)
        
        # Generate additional candidates with variations if NLP is available
        if self.nlp_analyzer and num_candidates > 1:
            for i in range(num_candidates - 1):
                # Create variation with different level of detail
                detail_level = "high" if i % 2 == 0 else "medium"
                variation = self._create_docstring_variation(
                    base_docstring, 
                    detail_level=detail_level
                )
                
                candidate = DocstringCandidate(
                    text=variation,
                    style=self.style
                )
                
                # Evaluate candidate
                self._evaluate_candidate_quality(candidate, context)
                candidates.append(candidate)
        
        # Sort candidates by quality score (descending)
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        return candidates
    
    def _evaluate_candidate_quality(
        self, 
        candidate: DocstringCandidate, 
        context: Dict[str, Any]
    ) -> None:
        """Evaluate the quality of a docstring candidate.
        
        Args:
            candidate: The docstring candidate to evaluate
            context: The code context information
        """
        # Extract relevant information from context
        element_info = context.get('info', {})
        code = context.get('code', '')
        
        # 1. Correctness - Check if the docstring matches the function signature
        correctness = self._evaluate_correctness(candidate.text, element_info)
        candidate.correctness_score = correctness
        
        # 2. Completeness - Check if all parameters and return values are documented
        completeness = self._evaluate_completeness(candidate.text, element_info)
        candidate.completeness_score = completeness
        
        # 3. Clarity - Calculate based on readability metrics
        clarity = self._evaluate_clarity(candidate.text)
        candidate.clarity_score = clarity
        
        # 4. Conciseness - Evaluate based on length relative to code complexity
        conciseness = self._evaluate_conciseness(candidate.text, code)
        candidate.conciseness_score = conciseness
    
    def _evaluate_correctness(self, docstring: str, element_info: Dict[str, Any]) -> float:
        """Evaluate if docstring correctly represents the element signature.
        
        Args:
            docstring: The docstring to evaluate
            element_info: Information about the code element
            
        Returns:
            Correctness score between 0 and 1
        """
        score = 1.0
        deductions = 0
        
        # Check if all parameters are correctly documented
        if element_info.get('args'):
            param_names = {arg['name'] for arg in element_info['args'] if arg['name'] != 'self'}
            
            # Extract parameter names from docstring
            documented_params = set()
            
            if self.style == "google":
                # Google style: Args:\n    param_name:
                param_pattern = r'Args:\s*(?:\n\s+(\w+):|$)'
            elif self.style == "numpy":
                # NumPy style: Parameters\n----------\nparam_name :
                param_pattern = r'Parameters\s*\n-+\s*\n(\w+)\s*:'
            else:  # Sphinx
                # Sphinx style: :param param_name:
                param_pattern = r':param\s+(\w+):'
            
            # Extract parameter names from docstring
            found_params = re.findall(param_pattern, docstring)
            documented_params.update(found_params)
            
            # Check for mismatches
            for param in param_names:
                if param not in documented_params:
                    deductions += 0.2  # Deduct 20% for each missing parameter
            
            for param in documented_params:
                if param not in param_names:
                    deductions += 0.1  # Deduct 10% for each extra parameter
        
        # Check return annotation
        if element_info.get('returns'):
            has_returns_doc = False
            
            if self.style == "google":
                # Google style: Returns:\n
                has_returns_doc = bool(re.search(r'Returns:\s*\n', docstring))
            elif self.style == "numpy":
                # NumPy style: Returns\n-------\n
                has_returns_doc = bool(re.search(r'Returns\s*\n-+\s*\n', docstring))
            else:  # Sphinx
                # Sphinx style: :returns:
                has_returns_doc = bool(re.search(r':returns?:', docstring))
            
            if not has_returns_doc:
                deductions += 0.2  # Deduct 20% for missing return documentation
        
        # Apply deductions with a minimum score of 0.1
        score = max(0.1, score - deductions)
        return score
    
    def _evaluate_completeness(self, docstring: str, element_info: Dict[str, Any]) -> float:
        """Evaluate if docstring completely documents all aspects of the element.
        
        Args:
            docstring: The docstring to evaluate
            element_info: Information about the code element
            
        Returns:
            Completeness score between 0 and 1
        """
        # Define required sections based on docstring style
        required_sections = []
        
        if self.style == "google":
            if element_info.get('args'):
                required_sections.append(r'Args:')
            if element_info.get('returns'):
                required_sections.append(r'Returns:')
        elif self.style == "numpy":
            if element_info.get('args'):
                required_sections.append(r'Parameters\s*\n-+')
            if element_info.get('returns'):
                required_sections.append(r'Returns\s*\n-+')
        else:  # Sphinx
            if element_info.get('args'):
                required_sections.append(r':param\s+\w+:')
            if element_info.get('returns'):
                required_sections.append(r':returns?:')
        
        # Always require a description
        if len(docstring.split('\n')[0].strip()) < 10:
            # First line is too short to be a good description
            return 0.5
        
        # Calculate completeness based on presence of required sections
        if not required_sections:
            return 1.0  # No specific requirements, just description
        
        # Count how many required sections are present
        sections_present = sum(
            1 for pattern in required_sections
            if re.search(pattern, docstring)
        )
        
        return sections_present / len(required_sections)
    
    def _evaluate_clarity(self, docstring: str) -> float:
        """Evaluate docstring clarity based on readability metrics.
        
        Args:
            docstring: The docstring to evaluate
            
        Returns:
            Clarity score between 0 and 1
        """
        # Simple clarity metrics
        lines = docstring.split('\n')
        
        # Empty or very short docstring
        if not docstring or len(docstring) < 5:
            return 0.0
        
        # Calculate basic readability score
        # 1. Penalize very long lines
        long_lines = sum(1 for line in lines if len(line.strip()) > 80)
        long_line_penalty = min(0.5, long_lines / len(lines))
        
        # 2. Reward appropriate length (not too short, not too long)
        length_score = min(1.0, len(docstring) / 250)
        if len(docstring) > 1000:
            length_score = max(0.2, 1.0 - (len(docstring) - 1000) / 2000)
        
        # Combine metrics
        score = 0.7 - long_line_penalty + 0.3 * length_score
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _evaluate_conciseness(self, docstring: str, code: str) -> float:
        """Evaluate docstring conciseness relative to code complexity.
        
        Args:
            docstring: The docstring to evaluate
            code: The code being documented
            
        Returns:
            Conciseness score between 0 and 1
        """
        code_lines = len(code.split('\n'))
        docstring_lines = len(docstring.split('\n'))
        
        # Calculate a reasonable ratio
        # For short functions, docstring can be longer than code
        # For long functions, docstring should be proportionally shorter
        if code_lines <= 10:
            # For very short code, docstring can be longer
            ideal_ratio = 1.0
        elif code_lines <= 30:
            # For medium code, docstring should be about 30-50% of code size
            ideal_ratio = 0.4
        else:
            # For long code, docstring should be about 20-30% of code size
            ideal_ratio = 0.25
        
        actual_ratio = docstring_lines / max(1, code_lines)
        
        # Score based on how close the ratio is to ideal ratio
        if actual_ratio <= ideal_ratio:
            # Under the ideal ratio is good
            score = min(1.0, actual_ratio / ideal_ratio)
        else:
            # Over the ideal ratio penalizes conciseness
            score = max(0.1, 1.0 - (actual_ratio - ideal_ratio))
        
        return score
    
    def _create_docstring_variation(
        self, 
        base_docstring: str, 
        detail_level: str = "medium"
    ) -> str:
        """Create a variation of a docstring with different detail levels.
        
        Args:
            base_docstring: The base docstring to create a variation from
            detail_level: The level of detail ('low', 'medium', 'high')
            
        Returns:
            A variation of the base docstring
        """
        # Split docstring into sections
        lines = base_docstring.split('\n')
        
        # Create different versions based on detail level
        if detail_level == "low":
            # Simplified version with only core information
            # Keep only the first paragraph and parameters
            first_para_end = 0
            for i, line in enumerate(lines):
                if i > 0 and line.strip() == '':
                    first_para_end = i
                    break
            
            if first_para_end > 0:
                brief_docstring = '\n'.join(lines[:first_para_end])
                
                # Find and add the Args/Parameters section
                for i, line in enumerate(lines):
                    if any(section in line for section in ['Args:', 'Parameters', ':param']):
                        return brief_docstring + '\n\n' + '\n'.join(lines[i:])
                
                return brief_docstring
                
        elif detail_level == "high":
            # Enhanced version with more details
            # Add additional sections or enhance existing ones
            
            # Find places to add more context
            for i, line in enumerate(lines):
                if line.strip() and i > 0 and not lines[i-1].strip():
                    # Found a section start
                    if any(section in line for section in ['Args:', 'Parameters', 'Returns:']):
                        # Add explanation to parameters or return values
                        for j in range(i+1, len(lines)):
                            if ':' in lines[j] and not lines[j].startswith(' ' * 8):
                                param_name = lines[j].split(':', 1)[0].strip()
                                if len(lines[j].split(':', 1)[1].strip()) < 30:
                                    # Parameter has a short description, enhance it
                                    lines[j] += " Ensure this value is properly validated."
                    elif line.strip() == 'Examples:':
                        # Add a more detailed example
                        lines.insert(i+1, "    # Detailed usage example")
                        lines.insert(i+2, "    result = function_name(param1='value', param2=42)")
                        lines.insert(i+3, "    # Process the result")
                        lines.insert(i+4, "    processed = some_processing(result)")
                        break
            
            return '\n'.join(lines)
        
        # Medium (default) - just make minor enhancements
        # Add a note about validation or exception handling
        for i, line in enumerate(lines):
            if line.strip() == 'Args:' or line.strip() == 'Parameters':
                # Add a note about validation to the first parameter
                for j in range(i+1, min(i+4, len(lines))):
                    if ':' in lines[j]:
                        lines[j] += " Must be properly validated."
                        break
        
        return '\n'.join(lines)
    
    def _get_template_func(self, element_type: str) -> Optional[Callable]:
        """Get the appropriate template function for the element type and style.
        
        Args:
            element_type: Type of element ('function', 'class', 'method', 'property')
            
        Returns:
            Template function for the specified element type and style
        """
        # Map 'property' to the correct type
        if element_type == 'property':
            template_key = 'property'
        # Map class methods to 'method'
        elif '.' in element_type:
            template_key = 'method'
        else:
            template_key = element_type
        
        # Get the style templates
        style_templates = self.style_templates.get(self.style, {})
        return style_templates.get(template_key)

    def _google_function_template(self, context: Dict[str, Any]) -> str:
        """Generate a Google-style docstring for a function.
        
        Args:
            context: The function context information
            
        Returns:
            Google-style docstring for the function
        """
        # Extract function info
        info = context.get('info', {})
        args = info.get('args', [])
        returns = info.get('returns')
        name = context.get('name', 'function')
        
        # Build the docstring
        lines = []
        
        # Short description based on function name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the function's purpose and behavior.")
            lines.append("")
        
        # Add args section if there are parameters
        if args:
            lines.append("Args:")
            for arg in args:
                if arg['name'] != 'self' and arg['name'] != 'cls':
                    arg_type = f" ({arg['annotation']})" if arg['annotation'] else ""
                    lines.append(f"    {arg['name']}{arg_type}: Description of {arg['name']}.")
            lines.append("")
        
        # Add returns section if there's a return annotation
        if returns:
            lines.append("Returns:")
            lines.append(f"    {returns}: Description of return value.")
        
        # Add simple example
        lines.append("Examples:")
        lines.append("    >>> result = " + name + "(" + ", ".join(arg['name'] for arg in args if arg['name'] not in ['self', 'cls']) + ")")
        lines.append("    >>> print(result)")
        
        return "\n".join(lines)
    
    def _google_class_template(self, context: Dict[str, Any]) -> str:
        """Generate a Google-style docstring for a class.
        
        Args:
            context: The class context information
            
        Returns:
            Google-style docstring for the class
        """
        # Extract class info
        info = context.get('info', {})
        name = context.get('name', 'class')
        bases = info.get('bases', [])
        
        # Build the docstring
        lines = []
        
        # Short description based on class name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        
        if bases:
            base_text = ', '.join(bases)
            lines.append(f"{readable_name.capitalize()} that inherits from {base_text}.")
        else:
            lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the class's purpose and behavior.")
            lines.append("")
        
        # Add Attributes section (placeholder)
        lines.append("Attributes:")
        lines.append("    Placeholder for class attributes.")
        
        return "\n".join(lines)
    
    def _google_method_template(self, context: Dict[str, Any]) -> str:
        """Generate a Google-style docstring for a method.
        
        Args:
            context: The method context information
            
        Returns:
            Google-style docstring for the method
        """
        # For methods, use the function template with self removed from args display
        full_name = context.get('name', '')
        method_name = full_name.split('.')[-1] if '.' in full_name else full_name
        context['name'] = method_name
        
        return self._google_function_template(context)
    
    def _google_property_template(self, context: Dict[str, Any]) -> str:
        """Generate a Google-style docstring for a property.
        
        Args:
            context: The property context information
            
        Returns:
            Google-style docstring for the property
        """
        # Extract property info
        info = context.get('info', {})
        returns = info.get('returns')
        name = context.get('name', '').split('.')[-1] if '.' in context.get('name', '') else context.get('name', 'property')
        
        # Build the docstring
        lines = []
        
        # Short description based on property name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"The {readable_name} property.")
        lines.append("")
        
        # Add return type information if available
        if returns:
            lines.append(f"Returns:")
            lines.append(f"    {returns}: Description of the property value.")
        
        return "\n".join(lines)
    
    def _numpy_function_template(self, context: Dict[str, Any]) -> str:
        """Generate a NumPy-style docstring for a function.
        
        Args:
            context: The function context information
            
        Returns:
            NumPy-style docstring for the function
        """
        # Extract function info
        info = context.get('info', {})
        args = info.get('args', [])
        returns = info.get('returns')
        name = context.get('name', 'function')
        
        # Build the docstring
        lines = []
        
        # Short description based on function name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the function's purpose and behavior.")
            lines.append("")
        
        # Add parameters section if there are arguments
        if args:
            filtered_args = [arg for arg in args if arg['name'] not in ['self', 'cls']]
            
            if filtered_args:
                lines.append("Parameters")
                lines.append("----------")
                
                for arg in filtered_args:
                    arg_type = f" : {arg['annotation']}" if arg['annotation'] else ""
                    lines.append(f"{arg['name']}{arg_type}")
                    lines.append(f"    Description of {arg['name']}.")
                
                lines.append("")
        
        # Add returns section if there's a return annotation
        if returns:
            lines.append("Returns")
            lines.append("-------")
            lines.append(f"{returns}")
            lines.append(f"    Description of return value.")
            lines.append("")
            
        # Add simple example
        lines.append("Examples")
        lines.append("--------")
        lines.append(">>> result = " + name + "(" + ", ".join(arg['name'] for arg in args if arg['name'] not in ['self', 'cls']) + ")")
        lines.append(">>> print(result)")
        
        return "\n".join(lines)
    
    def _numpy_class_template(self, context: Dict[str, Any]) -> str:
        """Generate a NumPy-style docstring for a class.
        
        Args:
            context: The class context information
            
        Returns:
            NumPy-style docstring for the class
        """
        # Extract class info
        info = context.get('info', {})
        name = context.get('name', 'class')
        bases = info.get('bases', [])
        
        # Build the docstring
        lines = []
        
        # Short description based on class name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        
        if bases:
            base_text = ', '.join(bases)
            lines.append(f"{readable_name.capitalize()} that inherits from {base_text}.")
        else:
            lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the class's purpose and behavior.")
            lines.append("")
        
        # Add Attributes section (placeholder)
        lines.append("Attributes")
        lines.append("----------")
        lines.append("attr : type")
        lines.append("    Placeholder for class attributes.")
        
        return "\n".join(lines)
    
    def _numpy_method_template(self, context: Dict[str, Any]) -> str:
        """Generate a NumPy-style docstring for a method.
        
        Args:
            context: The method context information
            
        Returns:
            NumPy-style docstring for the method
        """
        # For methods, use the function template with self removed from args display
        full_name = context.get('name', '')
        method_name = full_name.split('.')[-1] if '.' in full_name else full_name
        context['name'] = method_name
        
        return self._numpy_function_template(context)
    
    def _numpy_property_template(self, context: Dict[str, Any]) -> str:
        """Generate a NumPy-style docstring for a property.
        
        Args:
            context: The property context information
            
        Returns:
            NumPy-style docstring for the property
        """
        # Extract property info
        info = context.get('info', {})
        returns = info.get('returns')
        name = context.get('name', '').split('.')[-1] if '.' in context.get('name', '') else context.get('name', 'property')
        
        # Build the docstring
        lines = []
        
        # Short description based on property name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"The {readable_name} property.")
        lines.append("")
        
        # Add return type information if available
        if returns:
            lines.append("Returns")
            lines.append("-------")
            lines.append(f"{returns}")
            lines.append(f"    Description of the property value.")
        
        return "\n".join(lines)
    
    def _sphinx_function_template(self, context: Dict[str, Any]) -> str:
        """Generate a Sphinx-style docstring for a function.
        
        Args:
            context: The function context information
            
        Returns:
            Sphinx-style docstring for the function
        """
        # Extract function info
        info = context.get('info', {})
        args = info.get('args', [])
        returns = info.get('returns')
        name = context.get('name', 'function')
        
        # Build the docstring
        lines = []
        
        # Short description based on function name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the function's purpose and behavior.")
            lines.append("")
        
        # Add parameter documentation
        for arg in args:
            if arg['name'] not in ['self', 'cls']:
                arg_type = f" ({arg['annotation']})" if arg['annotation'] else ""
                lines.append(f":param {arg['name']}: Description of {arg['name']}.")
                if arg['annotation']:
                    lines.append(f":type {arg['name']}: {arg['annotation']}")
        
        # Add return documentation
        if returns:
            lines.append(f":returns: Description of return value.")
            lines.append(f":rtype: {returns}")
            
        return "\n".join(lines)
    
    def _sphinx_class_template(self, context: Dict[str, Any]) -> str:
        """Generate a Sphinx-style docstring for a class.
        
        Args:
            context: The class context information
            
        Returns:
            Sphinx-style docstring for the class
        """
        # Extract class info
        info = context.get('info', {})
        name = context.get('name', 'class')
        bases = info.get('bases', [])
        
        # Build the docstring
        lines = []
        
        # Short description based on class name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        
        if bases:
            base_text = ', '.join(bases)
            lines.append(f"{readable_name.capitalize()} that inherits from {base_text}.")
        else:
            lines.append(f"{readable_name.capitalize()}.")
        lines.append("")
        
        # Add more detailed description (placeholder)
        if context.get('code', ''):
            lines.append("Detailed description of the class's purpose and behavior.")
            lines.append("")
        
        # Add attribute documentation (placeholder)
        lines.append(":ivar attr: Placeholder for class attributes.")
        lines.append(":vartype attr: type")
        
        return "\n".join(lines)
    
    def _sphinx_method_template(self, context: Dict[str, Any]) -> str:
        """Generate a Sphinx-style docstring for a method.
        
        Args:
            context: The method context information
            
        Returns:
            Sphinx-style docstring for the method
        """
        # For methods, use the function template with self removed from args display
        full_name = context.get('name', '')
        method_name = full_name.split('.')[-1] if '.' in full_name else full_name
        context['name'] = method_name
        
        return self._sphinx_function_template(context)
    
    def _sphinx_property_template(self, context: Dict[str, Any]) -> str:
        """Generate a Sphinx-style docstring for a property.
        
        Args:
            context: The property context information
            
        Returns:
            Sphinx-style docstring for the property
        """
        # Extract property info
        info = context.get('info', {})
        returns = info.get('returns')
        name = context.get('name', '').split('.')[-1] if '.' in context.get('name', '') else context.get('name', 'property')
        
        # Build the docstring
        lines = []
        
        # Short description based on property name
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
        readable_name = ' '.join(words).lower()
        lines.append(f"The {readable_name} property.")
        lines.append("")
        
        # Add return type information if available
        if returns:
            lines.append(f":returns: Description of the property value.")
            lines.append(f":rtype: {returns}")
        
        return "\n".join(lines)


class DocstringApplier:
    """Applies generated docstrings to Python code files.
    
    This class handles the application of generated docstrings to Python code,
    either by modifying files directly or outputting the changes to be applied
    manually.
    """
    
    def __init__(self, code_analyzer: CodeAnalyzer):
        """Initialize the docstring applier.
        
        Args:
            code_analyzer: The code analyzer containing the parsed code
        """
        self.code_analyzer = code_analyzer
        self.file_path = code_analyzer.file_path
        self.code = code_analyzer.code
    
    def apply_docstring(
        self, 
        element_name: str, 
        docstring: str, 
        element_type: str = 'function'
    ) -> Tuple[bool, str]:
        """Apply a docstring to a specific code element.
        
        Args:
            element_name: Name of the function or class
            docstring: The docstring to apply
            element_type: Type of element ('function' or 'class')
            
        Returns:
            Tuple of (success, modified_code or error_message)
        """
        element_info = None
        
        # Get element info based on type
        if element_type == 'function':
            element_info = self.code_analyzer.functions.get(element_name)
        elif element_type == 'class':
            element_info = self.code_analyzer.classes.get(element_name)
        elif '.' in element_name:  # Method
            class_name, method_name = element_name.split('.', 1)
            class_info = self.code_analyzer.classes.get(class_name)
            if class_info and 'methods' in class_info:
                element_info = class_info['methods'].get(method_name)
                
        if not element_info:
            return False, f"Could not find {element_type} '{element_name}'"
        
        # Get line number of the element
        line_no = element_info.get('line', 0)
        if line_no <= 0:
            return False, f"Invalid line number for {element_type} '{element_name}'"
        
        # Get the code lines
        lines = self.code.split('\n')
        
        # Find the position to insert the docstring
        # This depends on the exact structure of the code
        try:
            # Check if there is already a docstring
            has_docstring = element_info.get('has_docstring', False)
            
            # Split docstring into lines and ensure proper indentation
            docstring_lines = docstring.split('\n')
            
            # Get indentation level of the element definition
            def_line = lines[line_no - 1]
            indent_match = re.match(r'^(\s*)', def_line)
            indent = indent_match.group(1) if indent_match else ''
            
            # Add additional indentation for the docstring content
            indented_docstring = f'{indent}"""\n'
            for ds_line in docstring_lines:
                if ds_line:
                    indented_docstring += f"{indent}{ds_line}\n"
                else:
                    indented_docstring += "\n"
            indented_docstring += f'{indent}"""'
            
            # Different handling based on whether there's an existing docstring
            modified_lines = lines.copy()
            
            if has_docstring:
                # Replace existing docstring - this is more complex and would require
                # finding the start and end of the existing docstring
                return False, "Replacing existing docstrings not yet implemented"
            else:
                # Insert new docstring after the definition line
                # Find the position where the function or class body starts
                def_line = modified_lines[line_no - 1]
                
                # Check if there's a colon at the end of the definition line
                if def_line.rstrip().endswith(':'):
                    # Insert after the definition line
                    modified_lines.insert(line_no, indented_docstring)
                else:
                    # Multi-line definition, find where it ends with a colon
                    end_line = line_no - 1
                    for i in range(line_no, min(line_no + 10, len(modified_lines))):
                        if modified_lines[i].rstrip().endswith(':'):
                            end_line = i
                            break
                    
                    # Insert after the end of the definition
                    modified_lines.insert(end_line + 1, indented_docstring)
            
            # Join the modified lines back into a single string
            modified_code = '\n'.join(modified_lines)
            
            return True, modified_code
            
        except Exception as e:
            return False, f"Error applying docstring: {str(e)}"
    
    def write_modified_code(self, modified_code: str) -> bool:
        """Write the modified code back to the original file.
        
        Args:
            modified_code: The modified code to write
            
        Returns:
            Whether the write operation was successful
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {self.file_path}: {str(e)}")
            return False


def parse_arguments():
    """Parse and validate command line arguments.
    
    Returns:
        The parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Generate and apply NLP-enhanced docstrings for Python files"
    )
    
    parser.add_argument(
        "target",
        help="Python file or directory to process"
    )
    
    parser.add_argument(
        "--style", "-s",
        choices=["google", "numpy", "sphinx"],
        default="google",
        help="Docstring style to use (default: google)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--apply", "-a",
        action="store_true",
        help="Apply docstrings to files (otherwise just output them)"
    )
    
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate existing docstrings for quality"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (default: stdout)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate the target path
    target_path = Path(args.target)
    if not target_path.exists():
        parser.error(f"Target path does not exist: {args.target}")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    return args


def process_file(file_path: str, args) -> Dict[str, Any]:
    """Process a single Python file to generate or validate docstrings.
    
    Args:
        file_path: Path to the Python file
        args: Command line arguments
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "file_path": file_path,
        "success": False,
        "generated": 0,
        "applied": 0,
        "validated": 0,
        "errors": []
    }
    
    try:
        # Initialize generator with specified style
        generator = DocstringGenerator(style=args.style)
        
        # Analyze the file
        analysis = generator.analyze_file(file_path)
        code_analyzer = analysis.get("code_analyzer")
        missing_docstrings = analysis.get("missing_docstrings", {})
        
        if args.validate:
            # TODO: Implement validation of existing docstrings
            results["validated"] = 0
        else:
            # Generate docstrings for missing elements
            docstring_applier = DocstringApplier(code_analyzer)
            modified_code = None
            
            # Process functions
            for function_name in missing_docstrings.get("functions", []):
                logger.info(f"Generating docstring for function '{function_name}'")
                candidates = generator.generate_optimal_docstring(
                    code_analyzer, function_name, "function"
                )
                
                if candidates:
                    best_candidate = candidates[0]
                    results["generated"] += 1
                    
                    if args.apply:
                        success, result = docstring_applier.apply_docstring(
                            function_name, best_candidate.text, "function"
                        )
                        
                        if success:
                            modified_code = result
                            results["applied"] += 1
                        else:
                            results["errors"].append(
                                f"Error applying docstring to function '{function_name}': {result}"
                            )
                    else:
                        logger.info(f"Generated docstring for '{function_name}':\n{best_candidate.text}")
            
            # Process classes
            for class_name in missing_docstrings.get("classes", []):
                logger.info(f"Generating docstring for class '{class_name}'")
                candidates = generator.generate_optimal_docstring(
                    code_analyzer, class_name, "class"
                )
                
                if candidates:
                    best_candidate = candidates[0]
                    results["generated"] += 1
                    
                    if args.apply:
                        success, result = docstring_applier.apply_docstring(
                            class_name, best_candidate.text, "class"
                        )
                        
                        if success:
                            modified_code = result
                            results["applied"] += 1
                        else:
                            results["errors"].append(
                                f"Error applying docstring to class '{class_name}': {result}"
                            )
                    else:
                        logger.info(f"Generated docstring for '{class_name}':\n{best_candidate.text}")
            
            # Process methods
            for method_name in missing_docstrings.get("methods", []):
                logger.info(f"Generating docstring for method '{method_name}'")
                candidates = generator.generate_optimal_docstring(
                    code_analyzer, method_name, "method"
                )
                
                if candidates:
                    best_candidate = candidates[0]
                    results["generated"] += 1
                    
                    if args.apply:
                        success, result = docstring_applier.apply_docstring(
                            method_name, best_candidate.text, "method"
                        )
                        
                        if success:
                            modified_code = result
                            results["applied"] += 1
                        else:
                            results["errors"].append(
                                f"Error applying docstring to method '{method_name}': {result}"
                            )
                    else:
                        logger.info(f"Generated docstring for '{method_name}':\n{best_candidate.text}")
            
            # Write modified code back to file if needed
            if args.apply and modified_code:
                if docstring_applier.write_modified_code(modified_code):
                    logger.info(f"Applied docstrings to {file_path}")
                else:
                    results["errors"].append(f"Error writing modified code to {file_path}")
        
        if not results["errors"]:
            results["success"] = True
        
    except Exception as e:
        results["errors"].append(f"Error processing file {file_path}: {str(e)}")
        logger.error(f"Error processing file {file_path}", exc_info=True)
    
    return results


def process_directory(dir_path: str, args) -> List[Dict[str, Any]]:
    """Process all Python files in a directory.
    
    Args:
        dir_path: Path to the directory
        args: Command line arguments
        
    Returns:
        List of processing results for all files
    """
    results = []
    
    # Get all Python files in the directory
    path = Path(dir_path)
    
    if args.recursive:
        # Process all Python files in directory and subdirectories
        python_files = list(path.glob("**/*.py"))
    else:
        # Process only Python files in this directory
        python_files = list(path.glob("*.py"))
    
    logger.info(f"Found {len(python_files)} Python files to process")
    
    for py_file in python_files:
        file_result = process_file(str(py_file), args)
        results.append(file_result)
    
    return results


def output_results(results: List[Dict[str, Any]], args):
    """Output processing results.
    
    Args:
        results: List of processing results
        args: Command line arguments
    """
    # Calculate summary statistics
    total_files = len(results)
    successful_files = sum(1 for r in results if r["success"])
    total_generated = sum(r["generated"] for r in results)
    total_applied = sum(r["applied"] for r in results)
    total_validated = sum(r["validated"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)
    
    summary = {
        "total_files": total_files,
        "successful_files": successful_files,
        "total_generated": total_generated,
        "total_applied": total_applied,
        "total_validated": total_validated,
        "total_errors": total_errors,
        "results": results
    }
    
    # Format as JSON
    json_output = json.dumps(summary, indent=2)
    
    # Output to file or stdout
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(json_output)
            logger.info(f"Results written to {args.output}")
        except Exception as e:
            logger.error(f"Error writing results to {args.output}: {str(e)}")
            print(json_output)
    else:
        print(json_output)


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Starting docstring generation with style: {args.style}")
    
    # Check if target is a file or directory
    target_path = Path(args.target)
    
    if target_path.is_file():
        logger.info(f"Processing single file: {args.target}")
        results = [process_file(args.target, args)]
    else:
        logger.info(f"Processing directory: {args.target}")
        results = process_directory(args.target, args)
    
    # Output results
    output_results(results, args)
    
    logger.info("Docstring generation completed")


if __name__ == "__main__":
    main() 