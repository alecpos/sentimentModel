#!/usr/bin/env python3
"""
Docstring Validation Visualization Tool for the WITHIN ML Prediction System.

This script provides visual explanations of the NLP-driven docstring validation
process. It highlights the connections between code and generated docstrings,
showing which parts of the code influenced specific parts of the documentation.
This helps developers understand and validate the generated docstrings.

Features:
- Visualizes bidirectional alignment between code and docstrings
- Highlights source code elements used to generate each docstring section
- Provides interactive HTML reports for reviewing docstring quality
- Supports comparison of multiple generated docstring candidates

Usage:
    python visualize_docstring_validation.py [file] [options]

Examples:
    python visualize_docstring_validation.py app/models/ml/prediction/ad_score_predictor.py
    python visualize_docstring_validation.py app/core/validation.py --output validation_report.html
"""

import os
import sys
import ast
import json
import argparse
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import re
import tempfile
import subprocess
import webbrowser
import html
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import visualization libraries, provide instructions if not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning(
        "Visualization libraries not available. For full functionality, install them with:\n"
        "pip install matplotlib numpy"
    )

# Import existing docstring tools
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    from nlp_enhanced_docstring_generator import (
        DocstringGenerator,
        DocstringCandidate,
        NLPEnhancedCodeAnalyzer
    )
    from bidirectional_validate import (
        BidirectionalValidator,
        DocstringAnalyzer,
    )
    NLP_TOOLS_AVAILABLE = True
except ImportError:
    NLP_TOOLS_AVAILABLE = False
    logger.warning(
        "NLP docstring tools not available. Make sure the following scripts exist:\n"
        "- nlp_enhanced_docstring_generator.py\n"
        "- bidirectional_validate.py"
    )

class DocstringAlignmentVisualizer:
    """Visualizes the alignment between code and generated docstrings."""
    
    def __init__(self):
        """Initialize the docstring alignment visualizer."""
        self.code_analyzer = NLPEnhancedCodeAnalyzer() if NLP_TOOLS_AVAILABLE else None
        self.validator = BidirectionalValidator(threshold=0.7) if NLP_TOOLS_AVAILABLE else None
    
    def generate_html_report(
        self, file_path: str, candidates: List[DocstringCandidate] = None
    ) -> str:
        """Generate an HTML report for visualizing docstring alignment.
        
        Args:
            file_path: Path to the Python file to analyze
            candidates: Optional list of docstring candidates to compare
            
        Returns:
            str: HTML report content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            
            # Parse the source code
            tree = ast.parse(source_code)
            
            # Get functions and classes
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)
            
            # Generate the report
            html_parts = [self._generate_html_header()]
            
            # Add file information
            file_name = os.path.basename(file_path)
            html_parts.append(f"<h1>Docstring Validation Report</h1>")
            html_parts.append(f"<h2>File: {html.escape(file_name)}</h2>")
            html_parts.append(f"<p class='report-info'>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Process functions
            if functions:
                html_parts.append("<h2>Functions</h2>")
                
                for func in functions:
                    html_parts.append(self._visualize_function(func, source_code, candidates))
            
            # Process classes
            if classes:
                html_parts.append("<h2>Classes</h2>")
                
                for cls in classes:
                    html_parts.append(self._visualize_class(cls, source_code))
            
            html_parts.append(self._generate_html_footer())
            
            return "\n".join(html_parts)
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def _visualize_function(
        self, node: ast.FunctionDef, source_code: str, candidates: List[DocstringCandidate] = None
    ) -> str:
        """Generate HTML for visualizing a function and its docstring.
        
        Args:
            node: The AST node of the function
            source_code: The full source code
            candidates: Optional list of docstring candidates to compare
            
        Returns:
            str: HTML content for the function visualization
        """
        # Get the function source code
        func_lines = source_code.splitlines()[node.lineno - 1:node.end_lineno]
        func_source = "\n".join(func_lines)
        
        # Get existing docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None and docstring.strip() != ""
        
        # Analyze function context
        if self.code_analyzer and NLP_TOOLS_AVAILABLE:
            context = self.code_analyzer.analyze_code_context(node, source_code)
        else:
            context = {"function_name": node.name}
        
        html_parts = [
            f"<div class='function-section'>",
            f"<h3 class='function-name'>{html.escape(node.name)}</h3>"
        ]
        
        # Function metadata
        params = []
        for arg in node.args.args:
            if arg.arg == 'self' or arg.arg == 'cls':
                continue
                
            param_type = "Any"
            if arg.annotation and hasattr(arg.annotation, 'id'):
                param_type = arg.annotation.id
                
            params.append({
                "name": arg.arg,
                "type": param_type
            })
        
        returns = None
        if hasattr(node, 'returns') and node.returns:
            if hasattr(node.returns, 'id'):
                returns = {"type": node.returns.id}
        
        # Function metadata panel
        html_parts.append("<div class='metadata-panel'>")
        html_parts.append("<h4>Function Metadata</h4>")
        
        # Parameters table
        if params:
            html_parts.append("<h5>Parameters</h5>")
            html_parts.append("<table class='params-table'>")
            html_parts.append("<tr><th>Name</th><th>Type</th></tr>")
            for param in params:
                html_parts.append(f"<tr><td>{html.escape(param['name'])}</td><td>{html.escape(param['type'])}</td></tr>")
            html_parts.append("</table>")
        
        # Return type
        if returns:
            html_parts.append("<h5>Returns</h5>")
            html_parts.append(f"<p>{html.escape(returns['type'])}</p>")
        
        html_parts.append("</div>")  # End metadata panel
        
        # Source code and docstring panel
        html_parts.append("<div class='side-by-side'>")
        
        # Source code panel
        html_parts.append("<div class='code-panel'>")
        html_parts.append("<h4>Source Code</h4>")
        html_parts.append(f"<pre class='code'>{html.escape(func_source)}</pre>")
        html_parts.append("</div>")
        
        # Docstring panel
        html_parts.append("<div class='docstring-panel'>")
        
        if has_docstring:
            html_parts.append("<h4>Current Docstring</h4>")
            html_parts.append(f"<pre class='docstring'>{html.escape(docstring)}</pre>")
            
            # Add validation info if available
            if self.validator and NLP_TOOLS_AVAILABLE:
                try:
                    validation_results = self.validator.validate_function(node)
                    html_parts.append("<h5>Validation Results</h5>")
                    html_parts.append(self._format_validation_results(validation_results))
                except Exception as e:
                    logger.warning(f"Error validating function {node.name}: {str(e)}")
        else:
            html_parts.append("<h4>No Current Docstring</h4>")
        
        # Show generated candidates if available
        if candidates:
            html_parts.append("<h4>Generated Docstring Candidates</h4>")
            
            for i, candidate in enumerate(candidates):
                html_parts.append(f"<h5>Candidate {i+1} (Score: {candidate.overall_score:.2f})</h5>")
                html_parts.append(f"<pre class='docstring candidate'>{html.escape(candidate.content)}</pre>")
                
                # Add detailed scores
                html_parts.append("<div class='score-details'>")
                html_parts.append("<table class='score-table'>")
                html_parts.append("<tr><th>Dimension</th><th>Score</th></tr>")
                html_parts.append(f"<tr><td>Correctness</td><td>{candidate.correctness_score:.2f}</td></tr>")
                html_parts.append(f"<tr><td>Clarity</td><td>{candidate.clarity_score:.2f}</td></tr>")
                html_parts.append(f"<tr><td>Conciseness</td><td>{candidate.conciseness_score:.2f}</td></tr>")
                html_parts.append(f"<tr><td>Completeness</td><td>{candidate.completeness_score:.2f}</td></tr>")
                html_parts.append("</table>")
                html_parts.append("</div>")
        
        html_parts.append("</div>")  # End docstring panel
        html_parts.append("</div>")  # End side-by-side panel
        html_parts.append("</div>")  # End function section
        
        return "\n".join(html_parts)
    
    def _visualize_class(self, node: ast.ClassDef, source_code: str) -> str:
        """Generate HTML for visualizing a class and its docstring.
        
        Args:
            node: The AST node of the class
            source_code: The full source code
            
        Returns:
            str: HTML content for the class visualization
        """
        # Get the class source code
        class_lines = source_code.splitlines()[node.lineno - 1:node.end_lineno]
        class_source = "\n".join(class_lines)
        
        # Get existing docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None and docstring.strip() != ""
        
        html_parts = [
            f"<div class='class-section'>",
            f"<h3 class='class-name'>{html.escape(node.name)}</h3>"
        ]
        
        # Class metadata
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # Class metadata panel
        html_parts.append("<div class='metadata-panel'>")
        html_parts.append("<h4>Class Metadata</h4>")
        
        if bases:
            html_parts.append("<h5>Base Classes</h5>")
            base_list = ", ".join(html.escape(base) for base in bases)
            html_parts.append(f"<p>{base_list}</p>")
        
        html_parts.append("</div>")  # End metadata panel
        
        # Class methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        if methods:
            html_parts.append("<div class='methods-panel'>")
            html_parts.append("<h4>Methods</h4>")
            method_list = ", ".join(html.escape(method) for method in methods)
            html_parts.append(f"<p>{method_list}</p>")
            html_parts.append("</div>")
        
        # Source code and docstring panel
        html_parts.append("<div class='side-by-side'>")
        
        # Source code panel (truncated for classes to avoid overwhelming)
        html_parts.append("<div class='code-panel'>")
        html_parts.append("<h4>Class Definition</h4>")
        
        # Get just the class definition (first few lines)
        def_lines = class_source.split("\n")
        if len(def_lines) > 10:
            visible_source = "\n".join(def_lines[:10]) + "\n..."
        else:
            visible_source = class_source
            
        html_parts.append(f"<pre class='code'>{html.escape(visible_source)}</pre>")
        html_parts.append("</div>")
        
        # Docstring panel
        html_parts.append("<div class='docstring-panel'>")
        
        if has_docstring:
            html_parts.append("<h4>Current Docstring</h4>")
            html_parts.append(f"<pre class='docstring'>{html.escape(docstring)}</pre>")
        else:
            html_parts.append("<h4>No Current Docstring</h4>")
        
        html_parts.append("</div>")  # End docstring panel
        html_parts.append("</div>")  # End side-by-side panel
        html_parts.append("</div>")  # End class section
        
        return "\n".join(html_parts)
    
    def _format_validation_results(self, validation_results: Dict[str, Any]) -> str:
        """Format validation results as HTML.
        
        Args:
            validation_results: Validation results from the validator
            
        Returns:
            str: HTML content for the validation results
        """
        html_parts = ["<div class='validation-results'>"]
        
        # Get core results
        code_to_doc = validation_results.get("code_to_doc", {})
        doc_to_code = validation_results.get("doc_to_code", {})
        
        # Overall alignment score
        alignment_score = validation_results.get("alignment_score", 0.0)
        color_class = "good" if alignment_score >= 0.8 else "medium" if alignment_score >= 0.6 else "poor"
        html_parts.append(f"<p class='alignment-score {color_class}'>Overall Alignment: {alignment_score:.2f}</p>")
        
        # Code to Doc issues
        if "issues" in code_to_doc and code_to_doc["issues"]:
            html_parts.append("<h5>Code → Doc Issues</h5>")
            html_parts.append("<ul class='issues-list'>")
            for issue in code_to_doc["issues"]:
                html_parts.append(f"<li>{html.escape(issue)}</li>")
            html_parts.append("</ul>")
        
        # Doc to Code issues
        if "issues" in doc_to_code and doc_to_code["issues"]:
            html_parts.append("<h5>Doc → Code Issues</h5>")
            html_parts.append("<ul class='issues-list'>")
            for issue in doc_to_code["issues"]:
                html_parts.append(f"<li>{html.escape(issue)}</li>")
            html_parts.append("</ul>")
        
        html_parts.append("</div>")
        return "\n".join(html_parts)
    
    def _generate_html_header(self) -> str:
        """Generate the HTML header with CSS styles.
        
        Returns:
            str: HTML header content
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Docstring Validation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        h1, h2, h3, h4, h5 {
            color: #2c3e50;
            margin-top: 0.7em;
            margin-bottom: 0.5em;
        }
        
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        
        h2 {
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 0.2em;
        }
        
        .report-info {
            color: #7f8c8d;
            font-style: italic;
        }
        
        .function-section, .class-section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            padding: 20px;
        }
        
        .function-name, .class-name {
            color: #2980b9;
        }
        
        .metadata-panel, .methods-panel {
            background-color: #f1f8fe;
            border-radius: 5px;
            padding: 10px 15px;
            margin-bottom: 15px;
        }
        
        .side-by-side {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }
        
        .code-panel, .docstring-panel {
            flex: 1;
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px 15px;
        }
        
        pre.code {
            background-color: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        
        pre.docstring {
            background-color: #f0f7ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 0 5px 5px 0;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        
        pre.docstring.candidate {
            border-left: 4px solid #27ae60;
            background-color: #f0fff5;
        }
        
        .validation-results {
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        
        .alignment-score {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 3px;
        }
        
        .alignment-score.good {
            background-color: #e6ffed;
            color: #22863a;
        }
        
        .alignment-score.medium {
            background-color: #fff5b1;
            color: #735c0f;
        }
        
        .alignment-score.poor {
            background-color: #ffeef0;
            color: #cb2431;
        }
        
        .issues-list {
            margin: 5px 0;
            padding-left: 20px;
        }
        
        .issues-list li {
            margin: 5px 0;
        }
        
        table.params-table, table.score-table {
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }
        
        table.params-table th, table.params-table td,
        table.score-table th, table.score-table td {
            border: 1px solid #dfe2e5;
            padding: 8px 12px;
            text-align: left;
        }
        
        table.params-table th, table.score-table th {
            background-color: #f1f8fe;
        }
        
        .score-details {
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .side-by-side {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
"""
    
    def _generate_html_footer(self) -> str:
        """Generate the HTML footer with JavaScript.
        
        Returns:
            str: HTML footer content
        """
        return """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add interactivity here if needed
});
</script>
</body>
</html>
"""

def process_file(file_path: str, output_path: Optional[str] = None) -> str:
    """Process a Python file to visualize docstring validation.
    
    Args:
        file_path: Path to the Python file
        output_path: Optional path to save the HTML report
    
    Returns:
        str: Path to the generated HTML report
    """
    if not file_path.endswith(".py"):
        logger.warning(f"Not a Python file: {file_path}")
        return None
        
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Generate docstring candidates if NLP tools are available
        candidates = None
        if NLP_TOOLS_AVAILABLE:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                    
                tree = ast.parse(source_code)
                
                # Get the first function without a docstring for demonstration
                target_node = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        docstring = ast.get_docstring(node)
                        if docstring is None or not docstring.strip():
                            target_node = node
                            break
                
                if target_node:
                    # Generate docstring candidates
                    generator = DocstringGenerator()
                    candidate1 = generator.generate_docstring(target_node, source_code)
                    
                    # Create a second candidate with slightly different settings
                    generator2 = DocstringGenerator(style="numpy")
                    candidate2 = generator2.generate_docstring(target_node, source_code)
                    
                    candidates = [candidate1, candidate2]
                    logger.info(f"Generated {len(candidates)} docstring candidates")
            except Exception as e:
                logger.warning(f"Error generating docstring candidates: {str(e)}")
        
        # Generate visualization
        visualizer = DocstringAlignmentVisualizer()
        html_content = visualizer.generate_html_report(file_path, candidates)
        
        # Save the report
        if output_path:
            report_path = output_path
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".html", prefix="docstring_report_"
            ) as tmp:
                report_path = tmp.name
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Report saved to: {report_path}")
        
        return report_path
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Visualize docstring validation for Python files"
    )
    
    parser.add_argument(
        "file",
        help="Python file to analyze"
    )
    
    parser.add_argument(
        "--output",
        help="Output path for the HTML report"
    )
    
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the report in the default browser"
    )
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        logger.warning(
            "Visualization libraries not installed. Some features may be limited. "
            "Install required packages with: pip install matplotlib numpy"
        )
    
    if not NLP_TOOLS_AVAILABLE:
        logger.warning(
            "NLP docstring tools not available. Some features will be limited. "
            "Make sure required scripts are in the same directory."
        )
    
    report_path = process_file(args.file, args.output)
    
    if report_path and args.open:
        # Open the report in the default browser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")

if __name__ == "__main__":
    main() 