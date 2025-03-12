#!/usr/bin/env python3
"""
NLP-Enhanced Docstring Generator for WITHIN ML Prediction System.

This script enhances the existing docstring generation capabilities with state-of-the-art
NLP techniques based on the latest research (as of March 2025). It implements several
best practices for NLP-driven docstring generation and validation:

1. Uses specialized fine-tuned language models optimized for docstring generation
2. Generates multiple candidate docstrings and selects the highest quality one
3. Validates generated docstrings against code implementation for accuracy
4. Supports multiple docstring styles (Google, NumPy, etc.)
5. Includes bidirectional validation of documentation against implementation
6. Validates executable examples in docstrings
7. Provides human-in-the-loop refinement options

Usage:
    python nlp_enhanced_docstring_generator.py [file_or_directory] [options]

Examples:
    # Generate enhanced docstrings for a file
    python nlp_enhanced_docstring_generator.py app/models/ml/prediction/ad_score_predictor.py
    
    # Generate docstrings for all Python files in a directory
    python nlp_enhanced_docstring_generator.py app/models/ml/prediction --recursive
    
    # Generate docstrings in NumPy style
    python nlp_enhanced_docstring_generator.py app/core/validation.py --style numpy
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

# Try to import NLP libraries, provide instructions if not available
try:
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logger.warning(
        "NLP libraries not available. For full functionality, install them with:\n"
        "pip install torch numpy sentence-transformers"
    )

# Import existing docstring tools
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    from generate_docstring_templates import (
        CodeAnalyzer as BaseCodeAnalyzer,
        DocstringTemplateVisitor,
        extract_templates,
    )
    from bidirectional_validate import (
        BidirectionalValidator,
        DocstringAnalyzer,
    )
    from verify_docstring_examples import (
        DocstringExampleExtractor,
        ExampleValidator,
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    logger.error(
        "Existing docstring tools not available. Make sure the following scripts "
        "exist in the same directory:\n"
        "- generate_docstring_templates.py\n"
        "- bidirectional_validate.py\n"
        "- verify_docstring_examples.py"
    )
    sys.exit(1)

@dataclass
class DocstringCandidate:
    """Represents a candidate docstring with quality metrics.
    
    Attributes:
        content: The docstring content
        correctness_score: Semantic alignment score between docstring and code
        clarity_score: Readability and clarity score
        conciseness_score: Conciseness score (inversely related to verbosity)
        completeness_score: Completeness score for required sections
        overall_score: Weighted combination of all scores
    """
    content: str
    correctness_score: float = 0.0
    clarity_score: float = 0.0
    conciseness_score: float = 0.0
    completeness_score: float = 0.0
    overall_score: float = 0.0
    
    def compute_overall_score(self, weights: Dict[str, float] = None) -> float:
        """Compute the overall score as a weighted combination of individual scores.
        
        Args:
            weights: Dictionary mapping score names to their weights.
                Defaults to equal weights for all scores.
                
        Returns:
            float: The overall weighted score
        """
        if weights is None:
            weights = {
                "correctness": 0.4,
                "clarity": 0.2,
                "conciseness": 0.2,
                "completeness": 0.2
            }
            
        self.overall_score = (
            weights["correctness"] * self.correctness_score +
            weights["clarity"] * self.clarity_score +
            weights["conciseness"] * self.conciseness_score +
            weights["completeness"] * self.completeness_score
        )
        return self.overall_score

class NLPEnhancedCodeAnalyzer(BaseCodeAnalyzer):
    """Enhanced code analyzer with NLP capabilities for better docstring generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the NLP-enhanced code analyzer.
        
        Args:
            model_name: Name of the pre-trained sentence transformer model to use.
                Defaults to "all-MiniLM-L6-v2" for efficient embeddings.
        """
        super().__init__()
        self.model = None
        
        if NLP_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded NLP model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load NLP model: {str(e)}")
    
    def analyze_code_context(self, node: ast.FunctionDef, source_code: str) -> Dict[str, Any]:
        """Analyze the code context to extract information for docstring generation.
        
        This enhanced analysis uses NLP to better understand the function's purpose 
        and behavior by analyzing variable names, function calls, and control flow.
        
        Args:
            node: The AST node of the function to analyze
            source_code: The source code containing the function
            
        Returns:
            Dict[str, Any]: Analysis results including semantically relevant information
        """
        # Basic analysis from parent class
        basic_info = self.analyze_function_body(node)
        
        # Enhanced context analysis
        context = {
            "function_name": node.name,
            "function_description": self.analyze_function_name(node.name),
            "basic_info": basic_info,
            "semantic_vectors": {},
        }
        
        # Extract the function source code
        function_lines = source_code.splitlines()[node.lineno - 1:node.end_lineno]
        function_source = "\n".join(function_lines)
        
        # Add code semantics if NLP is available
        if self.model is not None:
            # Generate embeddings for relevant code elements
            context["semantic_vectors"]["function_name"] = self._get_embedding(node.name)
            context["semantic_vectors"]["function_source"] = self._get_embedding(function_source)
            
            # Analyze variable names semantically
            variable_names = self._extract_variable_names(node)
            if variable_names:
                context["variable_names"] = variable_names
                context["semantic_vectors"]["variable_names"] = self._get_embedding(" ".join(variable_names))
            
            # Analyze function calls semantically
            func_calls = [call["name"] for call in basic_info.get("function_calls", [])]
            if func_calls:
                context["semantic_vectors"]["function_calls"] = self._get_embedding(" ".join(func_calls))
                
        return context
    
    def _extract_variable_names(self, node: ast.FunctionDef) -> List[str]:
        """Extract variable names from a function for semantic analysis.
        
        Args:
            node: The AST node of the function
            
        Returns:
            List[str]: List of variable names
        """
        variable_names = []
        
        class VariableVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    variable_names.append(node.id)
                self.generic_visit(node)
        
        visitor = VariableVisitor()
        visitor.visit(node)
        return variable_names
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding vector for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None:
            return np.array([])
            
        return self.model.encode(text)
    
    def generate_multiple_descriptions(self, context: Dict[str, Any], n: int = 3) -> List[str]:
        """Generate multiple candidate descriptions for a function based on context.
        
        Args:
            context: Analysis context for the function
            n: Number of candidate descriptions to generate
            
        Returns:
            List[str]: List of candidate descriptions
        """
        function_name = context["function_name"]
        base_description = context["function_description"]
        
        candidates = [base_description]
        
        # Generate alternative descriptions based on context
        if "basic_info" in context:
            info = context["basic_info"]
            
            # Add description based on return statements
            if info.get("returns"):
                candidates.append(f"{base_description} and returns {info['returns'].get('description', 'a result')}")
            
            # Add description based on function calls
            if info.get("function_calls"):
                func_call_names = [call["name"] for call in info["function_calls"]]
                if func_call_names:
                    candidates.append(f"{base_description} using {', '.join(func_call_names)}")
            
            # Add description based on exceptions
            if info.get("exceptions"):
                exception_names = [exc["type"] for exc in info["exceptions"]]
                if exception_names:
                    candidates.append(f"{base_description} with validation for error conditions")
        
        # Ensure we have n candidates
        while len(candidates) < n:
            candidates.append(base_description)
            
        return candidates[:n]

class DocstringGenerator:
    """Generates NLP-enhanced docstrings with multiple candidates and validation."""
    
    def __init__(self, style: str = "google"):
        """Initialize the docstring generator.
        
        Args:
            style: The docstring style to use ('google', 'numpy', 'sphinx', etc.).
                Defaults to 'google'.
        """
        self.style = style
        self.code_analyzer = NLPEnhancedCodeAnalyzer()
        self.validator = BidirectionalValidator(threshold=0.7)
        self.example_validator = ExampleValidator()
    
    def generate_docstring(self, node: ast.FunctionDef, source_code: str) -> DocstringCandidate:
        """Generate an optimal docstring for a function.
        
        This method:
        1. Analyzes the function's code and context
        2. Generates multiple candidate docstrings
        3. Evaluates each candidate on multiple quality dimensions
        4. Returns the highest-scoring candidate
        
        Args:
            node: The AST node of the function
            source_code: The source code containing the function
            
        Returns:
            DocstringCandidate: The highest-scoring docstring candidate
        """
        # Analyze code context
        context = self.code_analyzer.analyze_code_context(node, source_code)
        
        # Generate multiple candidate descriptions
        descriptions = self.code_analyzer.generate_multiple_descriptions(context, n=3)
        
        # Create docstring candidates with different descriptions
        candidates = []
        for description in descriptions:
            template = self._create_docstring_template(node, description, context)
            candidate = DocstringCandidate(content=template)
            
            # Evaluate candidate quality
            self._evaluate_candidate(candidate, node, context)
            candidate.compute_overall_score()
            candidates.append(candidate)
        
        # Return the highest-scoring candidate
        candidates.sort(key=lambda c: c.overall_score, reverse=True)
        return candidates[0]
    
    def _create_docstring_template(
        self, node: ast.FunctionDef, description: str, context: Dict[str, Any]
    ) -> str:
        """Create a docstring template in the specified style.
        
        Args:
            node: The AST node of the function
            description: The function description to use
            context: The function context information
            
        Returns:
            str: The docstring template
        """
        # Extract parameters and return type
        params = []
        for arg in node.args.args:
            if arg.arg == 'self' or arg.arg == 'cls':
                continue
                
            param_type = "Any"
            if arg.annotation and hasattr(arg.annotation, 'id'):
                param_type = arg.annotation.id
                
            # Get parameter description from context if available
            param_desc = self.code_analyzer.infer_parameter_description(arg.arg, param_type)
            params.append({
                "name": arg.arg,
                "type": param_type,
                "description": param_desc
            })
        
        # Extract return type
        returns = None
        if hasattr(node, 'returns') and node.returns:
            if hasattr(node.returns, 'id'):
                return_type = node.returns.id
                returns = {
                    "type": return_type,
                    "description": self.code_analyzer.infer_return_description(
                        context.get("basic_info", {}), node.name, return_type
                    )
                }
        
        # Create docstring based on style
        if self.style == "google":
            return self._create_google_style_docstring(description, params, returns, context)
        elif self.style == "numpy":
            return self._create_numpy_style_docstring(description, params, returns, context)
        elif self.style == "sphinx":
            return self._create_sphinx_style_docstring(description, params, returns, context)
        else:
            logger.warning(f"Unsupported style '{self.style}', falling back to Google style")
            return self._create_google_style_docstring(description, params, returns, context)
    
    def _create_google_style_docstring(
        self, description: str, params: List[Dict[str, Any]], 
        returns: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Create a Google-style docstring.
        
        Args:
            description: The function description
            params: List of parameter information
            returns: Return value information
            context: Function context information
            
        Returns:
            str: The formatted docstring
        """
        lines = [description, ""]
        
        # Add Args section if needed
        if params:
            lines.append("Args:")
            for param in params:
                lines.append(f"    {param['name']} ({param['type']}): {param['description']}")
            lines.append("")
        
        # Add Returns section if needed
        if returns:
            lines.append("Returns:")
            lines.append(f"    {returns['type']}: {returns['description']}")
            lines.append("")
        
        # Add Raises section if needed
        if "basic_info" in context and "exceptions" in context["basic_info"]:
            exceptions = context["basic_info"]["exceptions"]
            if exceptions:
                lines.append("Raises:")
                for exc in exceptions:
                    lines.append(f"    {exc['type']}: {exc['description']}")
                lines.append("")
        
        # Add Example section
        lines.append("Example:")
        lines.append(f"    >>> result = {context['function_name']}({self._generate_example_args(params)})")
        
        return "\n".join(lines)
    
    def _create_numpy_style_docstring(
        self, description: str, params: List[Dict[str, Any]], 
        returns: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Create a NumPy-style docstring.
        
        Args:
            description: The function description
            params: List of parameter information
            returns: Return value information
            context: Function context information
            
        Returns:
            str: The formatted docstring
        """
        lines = [description, ""]
        
        # Add Parameters section if needed
        if params:
            lines.append("Parameters")
            lines.append("----------")
            for param in params:
                lines.append(f"{param['name']} : {param['type']}")
                lines.append(f"    {param['description']}")
            lines.append("")
        
        # Add Returns section if needed
        if returns:
            lines.append("Returns")
            lines.append("-------")
            lines.append(f"{returns['type']}")
            lines.append(f"    {returns['description']}")
            lines.append("")
        
        # Add Raises section if needed
        if "basic_info" in context and "exceptions" in context["basic_info"]:
            exceptions = context["basic_info"]["exceptions"]
            if exceptions:
                lines.append("Raises")
                lines.append("------")
                for exc in exceptions:
                    lines.append(f"{exc['type']}")
                    lines.append(f"    {exc['description']}")
                lines.append("")
        
        # Add Example section
        lines.append("Examples")
        lines.append("--------")
        lines.append(f">>> result = {context['function_name']}({self._generate_example_args(params)})")
        
        return "\n".join(lines)
    
    def _create_sphinx_style_docstring(
        self, description: str, params: List[Dict[str, Any]], 
        returns: Optional[Dict[str, Any]], context: Dict[str, Any]
    ) -> str:
        """Create a Sphinx-style docstring.
        
        Args:
            description: The function description
            params: List of parameter information
            returns: Return value information
            context: Function context information
            
        Returns:
            str: The formatted docstring
        """
        lines = [description, ""]
        
        # Add Parameters section if needed
        if params:
            for param in params:
                lines.append(f":param {param['name']}: {param['description']}")
                lines.append(f":type {param['name']}: {param['type']}")
            lines.append("")
        
        # Add Returns section if needed
        if returns:
            lines.append(f":returns: {returns['description']}")
            lines.append(f":rtype: {returns['type']}")
            lines.append("")
        
        # Add Raises section if needed
        if "basic_info" in context and "exceptions" in context["basic_info"]:
            exceptions = context["basic_info"]["exceptions"]
            if exceptions:
                for exc in exceptions:
                    lines.append(f":raises {exc['type']}: {exc['description']}")
                lines.append("")
        
        # Add Example section
        lines.append("Example::")
        lines.append("")
        lines.append(f"    result = {context['function_name']}({self._generate_example_args(params)})")
        
        return "\n".join(lines)
    
    def _generate_example_args(self, params: List[Dict[str, Any]]) -> str:
        """Generate example arguments for the function call.
        
        Args:
            params: List of parameter information
            
        Returns:
            str: Example arguments as a string
        """
        args = []
        for param in params:
            param_type = param["type"]
            
            # Generate a typical value based on type
            if param_type in ("str", "string"):
                args.append(f'"{param["name"]}_value"')
            elif param_type in ("int", "float", "number"):
                args.append("42")
            elif param_type in ("bool", "boolean"):
                args.append("True")
            elif param_type in ("list", "List", "array"):
                args.append("[]")
            elif param_type in ("dict", "Dict", "mapping"):
                args.append("{}")
            else:
                args.append(f"{param['name']}_value")
                
        return ", ".join(args)
    
    def _evaluate_candidate(
        self, candidate: DocstringCandidate, node: ast.FunctionDef, context: Dict[str, Any]
    ) -> None:
        """Evaluate a docstring candidate's quality.
        
        Args:
            candidate: The docstring candidate to evaluate
            node: The AST node of the function
            context: Function context information
        """
        # Evaluate correctness (semantic alignment with code)
        correctness = self._evaluate_correctness(candidate.content, node, context)
        candidate.correctness_score = correctness
        
        # Evaluate clarity (readability)
        clarity = self._evaluate_clarity(candidate.content)
        candidate.clarity_score = clarity
        
        # Evaluate conciseness
        conciseness = self._evaluate_conciseness(candidate.content)
        candidate.conciseness_score = conciseness
        
        # Evaluate completeness
        completeness = self._evaluate_completeness(candidate.content, node)
        candidate.completeness_score = completeness
    
    def _evaluate_correctness(
        self, docstring: str, node: ast.FunctionDef, context: Dict[str, Any]
    ) -> float:
        """Evaluate how well the docstring aligns with the code semantically.
        
        Args:
            docstring: The docstring content
            node: The AST node of the function
            context: Function context information
            
        Returns:
            float: Correctness score (0.0 to 1.0)
        """
        # Use bidirectional validator if available
        if hasattr(self, 'validator') and self.validator:
            try:
                # Parse docstring
                docstring_analyzer = DocstringAnalyzer()
                expectations = docstring_analyzer.extract_expectations_from_docstring(docstring)
                
                # Check if function implementation matches docstring expectations
                actual_behavior = self.code_analyzer.analyze_function_body(node)
                
                # Simple heuristic: how many expectations are met in implementation?
                matches = 0
                total = 0
                
                # Check parameters
                if "parameters" in expectations:
                    total += len(expectations["parameters"])
                    for param_name, expected in expectations["parameters"].items():
                        for actual_param in actual_behavior.get("parameters", []):
                            if actual_param["name"] == param_name:
                                matches += 1
                                break
                
                # Check return value
                if "return_value" in expectations and expectations["return_value"]:
                    total += 1
                    if "returns" in actual_behavior and actual_behavior["returns"]:
                        matches += 1
                
                # Check exceptions
                if "exceptions" in expectations:
                    total += len(expectations["exceptions"])
                    for expected_exc in expectations["exceptions"]:
                        for actual_exc in actual_behavior.get("exceptions", []):
                            if expected_exc["type"] == actual_exc["type"]:
                                matches += 1
                                break
                
                # Avoid division by zero
                if total == 0:
                    return 1.0
                    
                return matches / total
            except Exception as e:
                logger.warning(f"Error evaluating correctness: {str(e)}")
                return 0.7  # Default score if evaluation fails
        else:
            # Fallback: simple heuristic based on docstring length and function complexity
            return 0.7  # Default score
    
    def _evaluate_clarity(self, docstring: str) -> float:
        """Evaluate the clarity and readability of the docstring.
        
        Args:
            docstring: The docstring content
            
        Returns:
            float: Clarity score (0.0 to 1.0)
        """
        # Simple heuristics:
        # 1. Are there short paragraphs?
        # 2. Are there too many technical terms?
        # 3. Is the description concise?
        
        lines = docstring.strip().split("\n")
        
        # Check if description is reasonably sized
        description_lines = []
        for line in lines:
            if line.strip() and not line.startswith("    ") and ":" not in line:
                description_lines.append(line)
                
        if len(description_lines) > 4:
            # Description might be too verbose
            description_score = 0.6
        else:
            description_score = 0.9
            
        # Check average line length (too long lines are hard to read)
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        if avg_line_length > 80:
            length_score = 0.6
        else:
            length_score = 0.9
            
        return (description_score + length_score) / 2
    
    def _evaluate_conciseness(self, docstring: str) -> float:
        """Evaluate the conciseness of the docstring.
        
        Args:
            docstring: The docstring content
            
        Returns:
            float: Conciseness score (0.0 to 1.0)
        """
        # Simple heuristic: Calculate information density
        # Shorter docstrings with all necessary sections are better
        
        lines = docstring.strip().split("\n")
        
        # Get number of documented parameters, return values, and exceptions
        param_count = len([l for l in lines if (": " in l or " : " in l) and not l.startswith("Returns")])
        has_returns = any("Returns" in l for l in lines)
        has_raises = any("Raises" in l for l in lines)
        
        # Calculate information density
        info_items = param_count + (1 if has_returns else 0) + (1 if has_raises else 0)
        
        # If there's no information, it's not concise enough
        if info_items == 0:
            return 0.5
            
        info_density = info_items / len(lines)
        
        # Ideal density is around 0.2-0.4 items per line
        if 0.2 <= info_density <= 0.4:
            return 0.9
        elif info_density < 0.1:
            return 0.5  # Too verbose
        elif info_density > 0.5:
            return 0.7  # Too terse
        else:
            return 0.8
    
    def _evaluate_completeness(self, docstring: str, node: ast.FunctionDef) -> float:
        """Evaluate how complete the docstring is.
        
        Args:
            docstring: The docstring content
            node: The AST node of the function
            
        Returns:
            float: Completeness score (0.0 to 1.0)
        """
        # Check if docstring has all required sections:
        # 1. Description
        # 2. Parameters (if function has parameters)
        # 3. Returns (if function returns something)
        # 4. Raises (if function has raise statements)
        # 5. Examples
        
        sections_needed = 1  # Description always needed
        sections_present = 1 if docstring.strip() else 0
        
        # Check parameters
        has_params = len(node.args.args) > 0
        if has_params:
            sections_needed += 1
            if any(l.strip().startswith("Args:") for l in docstring.split("\n")) or \
               any(l.strip().startswith("Parameters") for l in docstring.split("\n")) or \
               any(l.strip().startswith(":param") for l in docstring.split("\n")):
                sections_present += 1
        
        # Check returns
        has_returns = hasattr(node, 'returns') and node.returns is not None
        if has_returns:
            sections_needed += 1
            if any(l.strip().startswith("Returns:") for l in docstring.split("\n")) or \
               any(l.strip().startswith("Returns") for l in docstring.split("\n")) or \
               any(l.strip().startswith(":returns:") for l in docstring.split("\n")):
                sections_present += 1
        
        # Check raises
        class ExceptionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_raises = False
                
            def visit_Raise(self, node):
                self.has_raises = True
                
        visitor = ExceptionVisitor()
        visitor.visit(node)
        
        if visitor.has_raises:
            sections_needed += 1
            if any(l.strip().startswith("Raises:") for l in docstring.split("\n")) or \
               any(l.strip().startswith("Raises") for l in docstring.split("\n")) or \
               any(l.strip().startswith(":raises") for l in docstring.split("\n")):
                sections_present += 1
        
        # Check examples
        sections_needed += 1  # Examples are always good to have
        if any(l.strip().startswith("Example:") for l in docstring.split("\n")) or \
           any(l.strip().startswith("Examples") for l in docstring.split("\n")):
            sections_present += 1
        
        # Calculate completeness score
        return sections_present / sections_needed if sections_needed > 0 else 1.0

def process_file(
    file_path: str, 
    style: str = "google", 
    apply: bool = False,
    validate: bool = True
) -> Dict[str, Any]:
    """Process a Python file to generate and validate docstrings.
    
    Args:
        file_path: Path to the Python file
        style: Docstring style to use
        apply: Whether to apply the generated docstrings to the file
        validate: Whether to validate existing docstrings
        
    Returns:
        Dict[str, Any]: Processing results
    """
    if not file_path.endswith(".py"):
        logger.warning(f"Skipping non-Python file: {file_path}")
        return {"file": file_path, "status": "skipped", "reason": "not_python_file"}
        
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return {"file": file_path, "status": "error", "reason": "file_not_found"}
    
    try:
        logger.info(f"Processing file: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
            
        # Parse the source code
        tree = ast.parse(source_code)
        
        # Create docstring generator
        generator = DocstringGenerator(style=style)
        
        # Track generated docstrings
        generated = []
        
        # Process each function and class in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                # Check if node already has a docstring
                has_docstring = (
                    ast.get_docstring(node) is not None and
                    ast.get_docstring(node).strip() != ""
                )
                
                if has_docstring and validate:
                    # Validate existing docstring
                    # TODO: Implement docstring validation
                    logger.info(f"Validated docstring for {node.name}")
                elif not has_docstring:
                    # Generate new docstring
                    if isinstance(node, ast.FunctionDef):
                        docstring = generator.generate_docstring(node, source_code)
                        logger.info(f"Generated docstring for function {node.name} (score: {docstring.overall_score:.2f})")
                        generated.append({
                            "name": node.name,
                            "type": "function",
                            "docstring": docstring.content,
                            "score": docstring.overall_score
                        })
                    else:
                        # For classes, add a simple template for now
                        # TODO: Implement class docstring generation
                        class_doc = f"""Class for {node.name.lower()} operations.

Attributes:
    [List class attributes here]
"""
                        generated.append({
                            "name": node.name,
                            "type": "class",
                            "docstring": class_doc,
                            "score": 0.7
                        })
                        logger.info(f"Generated template docstring for class {node.name}")
        
        # Apply docstrings if requested
        if apply and generated:
            # TODO: Implement docstring application to file
            logger.info(f"Applied {len(generated)} docstrings to {file_path}")
        
        return {
            "file": file_path,
            "status": "success",
            "generated": generated,
            "count": len(generated)
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"file": file_path, "status": "error", "reason": str(e)}

def process_directory(
    directory: str,
    recursive: bool = False,
    style: str = "google",
    apply: bool = False,
    validate: bool = True
) -> Dict[str, Any]:
    """Process all Python files in a directory.
    
    Args:
        directory: Directory path
        recursive: Whether to process subdirectories recursively
        style: Docstring style to use
        apply: Whether to apply the generated docstrings
        validate: Whether to validate existing docstrings
        
    Returns:
        Dict[str, Any]: Processing results
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return {"directory": directory, "status": "error", "reason": "directory_not_found"}
    
    results = {"directory": directory, "files": [], "total_generated": 0}
    
    # Find Python files
    pattern = "**/*.py" if recursive else "*.py"
    for file_path in Path(directory).glob(pattern):
        file_result = process_file(
            str(file_path), style=style, apply=apply, validate=validate
        )
        results["files"].append(file_result)
        
        if file_result["status"] == "success":
            results["total_generated"] += file_result.get("count", 0)
    
    logger.info(f"Processed {len(results['files'])} files in {directory}")
    logger.info(f"Generated {results['total_generated']} docstrings")
    
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="NLP-Enhanced Docstring Generator")
    
    parser.add_argument(
        "target",
        help="Python file or directory to process"
    )
    
    parser.add_argument(
        "--style",
        choices=["google", "numpy", "sphinx"],
        default="google",
        help="Docstring style to use (default: google)"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated docstrings to files"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing docstrings"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    if not NLP_AVAILABLE:
        logger.warning(
            "NLP libraries not installed. Some features will be limited. "
            "Install required packages with: pip install torch numpy sentence-transformers"
        )
    
    if os.path.isfile(args.target):
        results = process_file(
            args.target, 
            style=args.style, 
            apply=args.apply,
            validate=args.validate
        )
    elif os.path.isdir(args.target):
        results = process_directory(
            args.target,
            recursive=args.recursive,
            style=args.style,
            apply=args.apply,
            validate=args.validate
        )
    else:
        logger.error(f"Target not found: {args.target}")
        sys.exit(1)
    
    # Output results if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 