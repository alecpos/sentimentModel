#!/usr/bin/env python3
"""
Docstring Verification Script

This script analyzes Python files to verify docstring quality and identify docstrings
that may need improvement. It implements the guidelines from the docstring verification
document to help maintain high documentation standards.

Usage:
    python verify_docstrings.py path/to/file_or_directory [options]

Options:
    --recursive, -r    Process directories recursively
    --format FORMAT    Output format (text, json, html) [default: text]
    --output FILE      Output file for results [default: stdout]
    --min-score SCORE  Minimum quality score threshold [default: 0.6]
    --protected FILE   File containing list of protected files/functions
"""

import os
import re
import ast
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class DocstringAnalysisResult:
    """Results from analyzing a docstring.
    
    Attributes:
        file_path: Path to the file containing the docstring
        element_name: Name of the function or class
        element_type: Type of element ('function', 'class', 'method')
        has_docstring: Whether the element has a docstring
        docstring: The actual docstring text if present
        quality_score: Overall quality score (0-1)
        issues: List of identified issues
        improvement_suggestions: Suggestions for improvement
    """
    
    file_path: str
    element_name: str
    element_type: str
    has_docstring: bool
    docstring: Optional[str] = None
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


class DocstringVerifier:
    """Verifies docstring quality based on project guidelines.
    
    This class implements the guidelines from the docstring verification document
    to analyze docstrings and identify potential issues.
    """
    
    def __init__(self, min_quality_score: float = 0.6, protected_elements: Set[str] = None):
        """Initialize the docstring verifier.
        
        Args:
            min_quality_score: Minimum acceptable quality score (0-1)
            protected_elements: Set of functions/classes that should not be modified
        """
        self.min_quality_score = min_quality_score
        self.protected_elements = protected_elements or set()
        
        # Common generic terms that indicate low-quality docstrings
        self.generic_terms = {
            "that inherits from",
            "Description of the",
            "Placeholder for",
            "Description of "
        }
        
    def analyze_file(self, file_path: str) -> List[DocstringAnalysisResult]:
        """Analyze docstrings in a Python file.
        
        Args:
            file_path: Path to the Python file
        
        Returns:
            List of docstring analysis results
        """
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            tree = ast.parse(file_content)
            
            # Analyze module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                result = self._analyze_docstring(
                    file_path=file_path,
                    element_name=os.path.basename(file_path),
                    element_type="module",
                    docstring=module_docstring
                )
                results.append(result)
            
            # Visit all nodes in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Analyze class docstring
                    class_docstring = ast.get_docstring(node)
                    full_class_name = node.name
                    
                    result = self._analyze_docstring(
                        file_path=file_path,
                        element_name=full_class_name,
                        element_type="class",
                        docstring=class_docstring
                    )
                    results.append(result)
                    
                    # Analyze method docstrings
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_docstring = ast.get_docstring(item)
                            full_method_name = f"{full_class_name}.{item.name}"
                            
                            result = self._analyze_docstring(
                                file_path=file_path,
                                element_name=full_method_name,
                                element_type="method",
                                docstring=method_docstring
                            )
                            results.append(result)
                            
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node):
                    # Analyze function docstring
                    function_docstring = ast.get_docstring(node)
                    full_function_name = node.name
                    
                    result = self._analyze_docstring(
                        file_path=file_path,
                        element_name=full_function_name,
                        element_type="function",
                        docstring=function_docstring
                    )
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            
        return results
    
    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function definition is a method in a class.
        
        Args:
            node: AST node for a function definition
            
        Returns:
            True if the function is a method, False otherwise
        """
        # Check if the parent is a class definition
        for parent in ast.walk(ast.parse(ast.unparse(node.parent)) if hasattr(node, 'parent') else ast.Module(body=[])):
            if isinstance(parent, ast.ClassDef):
                return True
        return False
    
    def _analyze_docstring(
        self, 
        file_path: str, 
        element_name: str, 
        element_type: str, 
        docstring: Optional[str]
    ) -> DocstringAnalysisResult:
        """Analyze a docstring for quality.
        
        Args:
            file_path: Path to the file
            element_name: Name of the function/class
            element_type: Type of element ('function', 'class', 'method')
            docstring: The docstring to analyze
            
        Returns:
            DocstringAnalysisResult with analysis results
        """
        # Initialize result
        result = DocstringAnalysisResult(
            file_path=file_path,
            element_name=element_name,
            element_type=element_type,
            has_docstring=docstring is not None,
            docstring=docstring
        )
        
        # Skip further analysis if no docstring
        if not docstring:
            result.issues.append("Missing docstring")
            result.improvement_suggestions.append("Add a descriptive docstring")
            return result
        
        # Check if this element is protected
        element_id = f"{file_path}:{element_name}"
        if element_id in self.protected_elements:
            result.quality_score = 1.0  # Assume protected docstrings are high quality
            return result
        
        # Analyze docstring quality
        quality_scores = {}
        
        # 1. Check for generic/template content
        generic_score = self._evaluate_generic_content(docstring)
        quality_scores["generic"] = generic_score
        
        if generic_score < 0.5:
            result.issues.append("Contains generic template content")
            result.improvement_suggestions.append(
                "Replace generic descriptions with specific details about the function/class purpose"
            )
        
        # 2. Check for completeness
        completeness_score = self._evaluate_completeness(docstring, element_type)
        quality_scores["completeness"] = completeness_score
        
        if completeness_score < 0.7:
            result.issues.append("Incomplete documentation")
            
            # Check for specific missing sections
            if "Args:" not in docstring and "Parameters" not in docstring and ":param" not in docstring:
                if element_type in ["function", "method"]:
                    result.improvement_suggestions.append("Add parameter documentation")
            
            if "Returns:" not in docstring and "Return:" not in docstring and ":return:" not in docstring:
                if element_type in ["function", "method"]:
                    result.improvement_suggestions.append("Add return value documentation")
        
        # 3. Check for specificity
        specificity_score = self._evaluate_specificity(docstring)
        quality_scores["specificity"] = specificity_score
        
        if specificity_score < 0.6:
            result.issues.append("Low specificity")
            result.improvement_suggestions.append(
                "Add domain-specific details relevant to the function/class purpose"
            )
        
        # 4. Calculate overall score
        # Weighted average of individual scores
        weights = {
            "generic": 0.4,       # Heavy weight on avoiding generic content
            "completeness": 0.4,  # Equal weight on completeness
            "specificity": 0.2    # Less weight on specificity (harder to measure)
        }
        
        result.quality_score = sum(score * weights[metric] for metric, score in quality_scores.items())
        
        # Add general improvement suggestion if score is low
        if result.quality_score < self.min_quality_score:
            if not result.improvement_suggestions:
                result.improvement_suggestions.append(
                    "Improve docstring with more specific details and complete documentation"
                )
        
        return result
    
    def _evaluate_generic_content(self, docstring: str) -> float:
        """Evaluate whether docstring contains generic template content.
        
        Args:
            docstring: The docstring to evaluate
            
        Returns:
            Score between 0 (very generic) and 1 (not generic)
        """
        # Count occurrences of generic terms
        generic_count = sum(1 for term in self.generic_terms if term in docstring)
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r"Description of \w+",
            r"Placeholder for",
            r"that inherits from \w+",
            r"Detailed description of the .* purpose and behavior"
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, docstring):
                generic_count += 1
        
        # Calculate score (inversely proportional to generic content)
        if generic_count == 0:
            return 1.0
        else:
            return max(0.0, 1.0 - (generic_count * 0.25))
    
    def _evaluate_completeness(self, docstring: str, element_type: str) -> float:
        """Evaluate the completeness of a docstring.
        
        Args:
            docstring: The docstring to evaluate
            element_type: Type of element ('function', 'class', 'method')
            
        Returns:
            Score between 0 (incomplete) and 1 (complete)
        """
        # Expected sections for different element types
        expected_sections = {
            "function": ["Args", "Returns", "Raises"],
            "method": ["Args", "Returns", "Raises"],
            "class": ["Attributes", "Methods"],
            "module": ["Usage", "Examples"]
        }
        
        # Check how many expected sections are present
        sections_present = 0
        sections_expected = expected_sections.get(element_type, [])
        
        google_sections = {
            "Args:": False,
            "Returns:": False,
            "Raises:": False,
            "Attributes:": False,
            "Methods:": False,
            "Examples:": False,
            "Note:": False
        }
        
        numpy_sections = {
            "Parameters\n": False,
            "Returns\n": False,
            "Raises\n": False,
            "Attributes\n": False,
            "Methods\n": False,
            "Examples\n": False,
            "Notes\n": False
        }
        
        sphinx_sections = {
            ":param": False,
            ":return:": False,
            ":raises:": False,
            ":ivar:": False,
            ":meth:": False,
            ":Example:": False,
            ":note:": False
        }
        
        # Detect style - simple heuristic
        style = "unknown"
        if any(section in docstring for section in google_sections):
            style = "google"
            sections_dict = google_sections
        elif any(section in docstring for section in numpy_sections):
            style = "numpy"
            sections_dict = numpy_sections
        elif any(section in docstring for section in sphinx_sections):
            style = "sphinx"
            sections_dict = sphinx_sections
        else:
            style = "unknown"
            sections_dict = {}
            
        # Count present sections based on style
        if style != "unknown":
            for section in sections_dict:
                if section in docstring:
                    sections_dict[section] = True
                    
            # Map section names to expected sections
            style_mapping = {
                "google": {
                    "Args": "Args:", 
                    "Returns": "Returns:", 
                    "Raises": "Raises:",
                    "Attributes": "Attributes:",
                    "Methods": "Methods:",
                    "Examples": "Examples:"
                },
                "numpy": {
                    "Args": "Parameters\n", 
                    "Returns": "Returns\n", 
                    "Raises": "Raises\n",
                    "Attributes": "Attributes\n",
                    "Methods": "Methods\n",
                    "Examples": "Examples\n"
                },
                "sphinx": {
                    "Args": ":param", 
                    "Returns": ":return:", 
                    "Raises": ":raises:",
                    "Attributes": ":ivar:",
                    "Methods": ":meth:",
                    "Examples": ":Example:"
                }
            }
            
            mapping = style_mapping.get(style, {})
            for section in sections_expected:
                style_section = mapping.get(section, "")
                if style_section and sections_dict.get(style_section, False):
                    sections_present += 1
        
        # Calculate completeness score
        if not sections_expected:
            # If no sections are expected, just check for non-empty docstring
            return 1.0 if docstring.strip() else 0.0
        else:
            return sections_present / len(sections_expected)
    
    def _evaluate_specificity(self, docstring: str) -> float:
        """Evaluate the specificity of a docstring.
        
        Args:
            docstring: The docstring to evaluate
            
        Returns:
            Score between 0 (not specific) and 1 (very specific)
        """
        # Simple heuristics for specificity
        lines = docstring.strip().split('\n')
        first_line = lines[0] if lines else ""
        
        # 1. Length of first line (summary)
        first_line_score = min(1.0, len(first_line) / 50)
        
        # 2. Overall length relative to expected minimum
        min_expected_length = 50
        length_score = min(1.0, len(docstring) / min_expected_length)
        
        # 3. Presence of specific details (numbers, technical terms)
        detail_indicators = [
            r'\d+',  # Numbers
            r'[A-Z][a-z]+',  # Capitalized terms (potential technical terms)
            r'[a-z]+_[a-z]+',  # Snake_case (potential variable names)
            r'".*?"',  # Quoted text
            r'`.*?`'   # Code-formatted text
        ]
        
        detail_count = 0
        for pattern in detail_indicators:
            detail_count += len(re.findall(pattern, docstring))
            
        detail_score = min(1.0, detail_count / 5)  # Cap at 1.0
        
        # Combine scores with weights
        weighted_score = (0.3 * first_line_score) + (0.3 * length_score) + (0.4 * detail_score)
        
        return weighted_score


def find_python_files(path: str, recursive: bool = False) -> List[str]:
    """Find Python files in the specified path.
    
    Args:
        path: Directory or file path
        recursive: Whether to search recursively
        
    Returns:
        List of Python file paths
    """
    path = Path(path)
    
    if path.is_file():
        if path.suffix == '.py':
            return [str(path)]
        else:
            return []
    
    if recursive:
        return [str(p) for p in path.glob('**/*.py') if p.is_file()]
    else:
        return [str(p) for p in path.glob('*.py') if p.is_file()]


def load_protected_elements(file_path: Optional[str]) -> Set[str]:
    """Load the set of protected elements from a file.
    
    Args:
        file_path: Path to the file containing protected elements
        
    Returns:
        Set of protected element identifiers
    """
    protected_elements = set()
    
    if not file_path:
        return protected_elements
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    protected_elements.add(line)
                    
        logger.info(f"Loaded {len(protected_elements)} protected elements")
    except Exception as e:
        logger.error(f"Error loading protected elements from {file_path}: {str(e)}")
    
    return protected_elements


def format_results(results: List[DocstringAnalysisResult], format_type: str) -> str:
    """Format analysis results based on specified format.
    
    Args:
        results: List of docstring analysis results
        format_type: Format type (text, json, html)
        
    Returns:
        Formatted results as string
    """
    if format_type == "json":
        # Convert to JSON
        results_dict = [result.__dict__ for result in results]
        return json.dumps(results_dict, indent=2)
    
    elif format_type == "html":
        # Create an HTML report
        html = ["<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<title>Docstring Verification Report</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 20px; }",
                "h1 { color: #333; }",
                "table { border-collapse: collapse; width: 100%; }",
                "th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }",
                "tr:nth-child(even) { background-color: #f2f2f2; }",
                "th { background-color: #4CAF50; color: white; }",
                ".high { background-color: #dff0d8; }",
                ".medium { background-color: #fcf8e3; }",
                ".low { background-color: #f2dede; }",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Docstring Verification Report</h1>"]
        
        # Summary statistics
        total_elements = len(results)
        missing_docstrings = sum(1 for r in results if not r.has_docstring)
        low_quality = sum(1 for r in results if r.has_docstring and r.quality_score < 0.6)
        medium_quality = sum(1 for r in results if r.has_docstring and 0.6 <= r.quality_score < 0.8)
        high_quality = sum(1 for r in results if r.has_docstring and r.quality_score >= 0.8)
        
        html.append("<h2>Summary</h2>")
        html.append("<ul>")
        html.append(f"<li>Total elements: {total_elements}</li>")
        html.append(f"<li>Missing docstrings: {missing_docstrings} ({missing_docstrings/total_elements*100:.1f}%)</li>")
        html.append(f"<li>Low quality docstrings: {low_quality} ({low_quality/total_elements*100:.1f}%)</li>")
        html.append(f"<li>Medium quality docstrings: {medium_quality} ({medium_quality/total_elements*100:.1f}%)</li>")
        html.append(f"<li>High quality docstrings: {high_quality} ({high_quality/total_elements*100:.1f}%)</li>")
        html.append("</ul>")
        
        # Detailed table
        html.append("<h2>Detailed Analysis</h2>")
        html.append("<table>")
        html.append("<tr><th>File</th><th>Element</th><th>Type</th><th>Quality</th><th>Issues</th><th>Suggestions</th></tr>")
        
        # Sort by quality score (ascending) to highlight problematic docstrings first
        sorted_results = sorted(results, key=lambda r: r.quality_score if r.has_docstring else -1)
        
        for result in sorted_results:
            if result.has_docstring:
                if result.quality_score < 0.6:
                    quality_class = "low"
                elif result.quality_score < 0.8:
                    quality_class = "medium"
                else:
                    quality_class = "high"
            else:
                quality_class = "low"
                
            html.append(f"<tr class='{quality_class}'>")
            html.append(f"<td>{result.file_path}</td>")
            html.append(f"<td>{result.element_name}</td>")
            html.append(f"<td>{result.element_type}</td>")
            
            if result.has_docstring:
                html.append(f"<td>{result.quality_score:.2f}</td>")
            else:
                html.append("<td>Missing</td>")
                
            html.append(f"<td>{', '.join(result.issues) if result.issues else 'None'}</td>")
            html.append(f"<td>{', '.join(result.improvement_suggestions) if result.improvement_suggestions else 'None'}</td>")
            html.append("</tr>")
            
        html.append("</table>")
        html.append("</body></html>")
        
        return "\n".join(html)
    
    else:  # text format (default)
        lines = ["Docstring Verification Report", "=" * 30, ""]
        
        # Group by file
        results_by_file = {}
        for result in results:
            if result.file_path not in results_by_file:
                results_by_file[result.file_path] = []
            results_by_file[result.file_path].append(result)
        
        # Print summary per file
        for file_path, file_results in results_by_file.items():
            lines.append(f"File: {file_path}")
            lines.append("-" * len(f"File: {file_path}"))
            
            missing = sum(1 for r in file_results if not r.has_docstring)
            low_quality = sum(1 for r in file_results if r.has_docstring and r.quality_score < 0.6)
            
            lines.append(f"Elements: {len(file_results)}")
            lines.append(f"Missing docstrings: {missing}")
            lines.append(f"Low quality docstrings: {low_quality}")
            lines.append("")
            
            # List elements with issues
            problem_results = [r for r in file_results if not r.has_docstring or r.quality_score < 0.6]
            
            if problem_results:
                lines.append("Elements needing attention:")
                for result in problem_results:
                    if not result.has_docstring:
                        lines.append(f"  {result.element_name} ({result.element_type}): Missing docstring")
                    else:
                        lines.append(f"  {result.element_name} ({result.element_type}): Quality score {result.quality_score:.2f}")
                        if result.issues:
                            lines.append(f"    Issues: {', '.join(result.issues)}")
                        if result.improvement_suggestions:
                            lines.append(f"    Suggestions: {', '.join(result.improvement_suggestions)}")
                lines.append("")
        
        # Overall statistics
        total_elements = len(results)
        total_missing = sum(1 for r in results if not r.has_docstring)
        total_low = sum(1 for r in results if r.has_docstring and r.quality_score < 0.6)
        total_ok = sum(1 for r in results if r.has_docstring and r.quality_score >= 0.6)
        
        lines.append("Overall Statistics")
        lines.append("=" * 20)
        lines.append(f"Total elements: {total_elements}")
        lines.append(f"Missing docstrings: {total_missing} ({total_missing/total_elements*100:.1f}%)")
        lines.append(f"Low quality docstrings: {total_low} ({total_low/total_elements*100:.1f}%)")
        lines.append(f"Acceptable docstrings: {total_ok} ({total_ok/total_elements*100:.1f}%)")
        
        return "\n".join(lines)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Verify docstring quality in Python files")
    
    parser.add_argument(
        "target",
        help="Python file or directory to process"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for results (default: stdout)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum quality score threshold"
    )
    
    parser.add_argument(
        "--protected",
        help="File containing list of protected files/functions"
    )
    
    args = parser.parse_args()
    
    # Validate target path
    target_path = Path(args.target)
    if not target_path.exists():
        parser.error(f"Target path does not exist: {args.target}")
    
    # Load protected elements
    protected_elements = load_protected_elements(args.protected)
    
    # Find Python files
    py_files = find_python_files(args.target, args.recursive)
    logger.info(f"Found {len(py_files)} Python files to process")
    
    # Initialize verifier
    verifier = DocstringVerifier(min_quality_score=args.min_score, protected_elements=protected_elements)
    
    # Process files
    all_results = []
    for py_file in py_files:
        logger.info(f"Analyzing {py_file}")
        results = verifier.analyze_file(py_file)
        all_results.extend(results)
    
    # Format results
    output = format_results(all_results, args.format)
    
    # Write results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main() 